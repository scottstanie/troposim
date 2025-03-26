"""Module for simulating isotropic tropospheric turbulence and
estimating turbulence properties.

Author: Scott Staniewicz
Adapted from MintPy fractal.py module,
  https://github.com/insarlab/MintPy/blob/4c2b6f8f80c86f245f8635bbb0ce2e46b29cb576/mintpy/simulation/fractal.py
  (Author: Zhang Yunjun, 2019)
which was a Python translation in MintPy of the matlab scripts written by
Ramon Hanssen, May 2000, available in the following website:
    http://doris.tudelft.nl/software/insarfractal.tar.gz
"""

import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

import numpy as np
from numpy.polynomial.polynomial import Polynomial, polyval
from numpy.random import SeedSequence, default_rng
from scipy import ndimage
from scipy.fft import fft2, fftfreq, fftshift, ifft2
from tqdm.auto import tqdm

from . import utils

MAX_WORKERS = 8
RNG = default_rng()
# freq0 defined here so that Psd estimation can auto-pick if it's outside
# the range of the PSD
DEFAULT_FREQ0 = 1e-4


def simulate(
    shape=(300, 300),
    beta=8 / 3,
    resolution=60.0,
    p0=10.0,
    freq0=DEFAULT_FREQ0,
    max_amp=None,
    seed=None,
    verbose=False,
):
    """Simulate 2D isotropic turbulence with a power-law power spectral density (PSD)

    Can use either a single slope for beta, or a numpy Polynomial which is fit
    to log10(power) vs log10(frequency)

    For simple turbulence (one PSD slope), try out:
    beta = 3.0 for larger scale turbulence (long spatial correlations, large blobs)
    beta = 2.0 for middle scale turbulence
    beta = 1.3 for small scale turbulence

    Parameters
    ----------
    shape : tuple[float]
        (rows, cols), or (num_images, rows, cols) of output
        If passing 3D shape, will use same `beta` for all layers of output
        (Default value = (300, 300))
    beta : float, ndarray[float], ndarray[Polynomial], or array of polynomial coeffs,
        For scalar: power law exponent for the slope of radially averaged 2D spectrum
        Polynomial: result from fit of radially averaged log PSD vs log frequency
            (or, see result from `get_psd`)
        array of poly: one for each layer of output. Must match 3D shape.
        (Default value = 8/3)
    resolution : float
        spatial resolution of output pixels, in meters (Default value = 60.0)
    p0 : float
        Power level of PSD at the refernce frequency `freq0`
        Units are m^2 / (1/m^2) (Default value = 10.0)
    freq0 (float), reference spatial frequency where `p0` defined), in cycle / m
        (default 1e-4, or 1 cycle/10 km)
    max_amp : float or ndarray[float], optional
        maximum amplitude (in meters) of the simulated turbulence.
        Alternative to passing `p0` and `freq0`
        If passing an array, must match shape[0] of the stack
    seed : int
        number to seed random numbers, for reproducible turbulence
        (Default value = None)
    verbose : bool
        print extra debug info (Default value = False)

    Returns
    -------
    ndarray
        simulated turbulence, either 2D or 3D, depending on shape
        Units are meters

    Examples
    --------
    >>> from troposim import turbulence
    >>> out = turbulence.simulate(shape=(200, 200), beta=8/3)
    >>> # Stack of 4 images of 200x200 pixels, noise increasing in spatial scale
    >>> out = turbulence.simulate(shape=(4, 200, 200), beta=[2.0, 2.2, 2.7, 3.0])
    """
    if np.all(np.array(p0) == 0):
        return np.zeros(shape)

    try:
        num_images, length, width = shape
    except ValueError:
        num_images = 1
        length, width = shape

    # Make `beta` into an array of Polynomials, with length = num_images
    beta = _standardize_beta(beta, num_images, verbose=verbose)
    if num_images > 1:
        #  3D output, so make sure `p0` is an array of the correct length
        if np.atleast_1d(p0).size == 1:
            p0 = np.repeat(p0, num_images)
        out_list = [None for _ in range(num_images)]  # initialize list of outputs
        # https://numpy.org/devdocs/reference/random/parallel.html
        ss = SeedSequence(seed)
        child_seeds = ss.spawn(num_images)

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_idx_map = {
                executor.submit(
                    simulate,
                    shape=(length, width),
                    beta=beta[idx],
                    p0=p0[idx],
                    freq0=freq0,
                    max_amp=max_amp,
                    resolution=resolution,
                    seed=child_seeds[idx],  # todo: how to make deterministic parallel?
                    verbose=verbose,
                ): idx
                for idx in range(num_images)
            }
            # Only show tqdm for longer simulations
            skip_bar = (length * width * num_images) < 10e6
            for fut in tqdm(
                as_completed(future_to_idx_map), total=num_images, disable=skip_bar
            ):
                idx = future_to_idx_map[fut]
                out_list[idx] = fut.result()
        return np.stack(out_list)

    # Start with the 2D PSD of white noise: flat amplitude, random phase
    rng = np.random.default_rng(seed) if seed is not None else RNG
    h = rng.uniform(size=shape)
    H = np.exp(1j * 2 * np.pi * h)

    # spatial frequencies for polynomial evaluation (units: cycles / m)
    fx = _get_freqs(width, resolution)
    fy = _get_freqs(length, resolution)
    # Broadcast 1D vectors to square: (N,1) + (1, N) = (N, N)
    f = np.sqrt(fy[:, np.newaxis] ** 2 + fx[np.newaxis, :] ** 2)

    # The power beta/2 is used because power ~ amplitude**2
    # Using amplitude means we take sqrt( k**beta) = k**(beta/2)
    beta_amp = beta / 2
    # Now evaluate each using the coefficients
    # Note: the polyval is like doing P = k ** (beta), but allows cubic, etc.
    b_coeffs = beta_amp[0].coef
    # OLD VERSION: b_coeffs = np.array([b.coef for b in beta_amp])  # for 3D eval
    # places where f=0 throw warnings in log10
    with np.errstate(invalid="ignore", divide="ignore"):
        logP = polyval(np.log10(f), b_coeffs.T)

    # create the envelope to shape the power of the white noise
    P = 10**logP
    # correct dividing by zero
    P[f == 0] = 1.0

    H_shaped = H * P
    # Make output zero mean by zeroing the top left (0 freq,constant) element
    H_shaped[0, 0] = 0.0
    out = ifft2(H_shaped, workers=-1).real
    # If passed the option of a maximum amplitude, use that and return
    return _scale_output(out, H_shaped, p0, freq0, resolution, max_amp)


def _standardize_beta(beta, num_images, verbose=False):
    """Get beta into an ndarray of Polynomials with length = num_images

    Parameters
    ----------
    beta : float or list or ndarray
        Allowed options are
        1. a single scalar, used as the linear slope
        2. a list/array of scalars, equal in length to `num_images`, treated as
                the linear slope for each image
        3. a 2d array of shape (num_images, deg+1) of polynomial coefficients
        4. a list/array of Polynomials, equal in length to `num_images`
    num_images : int
        number of images to simulate
    verbose : bool
        print extra debug info (Default value = False)

    Returns
    -------
    ndarray
        array of Polynomials, with length = num_images

    """
    # If Polynomials are passed, extract the coefficients to simplify later logic
    if isinstance(beta, Polynomial):
        beta = np.array([beta])
    if isinstance(beta, list) and isinstance(beta[0], Polynomial):
        beta = np.array([b.coef for b in beta])
    if (
        isinstance(beta, np.ndarray)
        and beta.ndim > 0
        and isinstance(beta[0], Polynomial)
    ):
        beta = np.array([b.coef for b in beta])

    # Make sure we now have just an array of coefficients
    try:
        beta = np.array(beta).astype(float)
    except TypeError:
        raise ValueError(
            f"beta must be an array of Polynomials or numeric coefficients: {beta}"
        )

    # 1. A single scalar means the linear slope used for all images
    if beta.ndim == 0:
        if beta > 0:
            # make sure the power slope is negative
            beta *= -1.0
            if verbose:
                print(f"reversed sign on scalar beta slope: {beta}")
        # Convert to linear polynomial
        beta = np.array([Polynomial([0, beta])])
        # beta = beta * np.ones(3)

    # 2. Passing a list of scalars (slopes)
    # There should be 1 for each requested image, and the input
    # should be 1D
    elif beta.ndim == 1:
        # make sure the power slope is negative for all cases
        beta[beta > 0] *= -1.0
        beta = np.array([Polynomial([0, b]) for b in beta])
    # 3. Passing an array of coefficients. Each row is the 2 (or 4)
    # coefficients of a linear (cubic) polynomial
    elif beta.ndim == 2:
        if len(beta[0]) not in (2, 4):
            raise ValueError("2D beta input must be list of polynomial coeffs")
        beta = np.array([Polynomial(b) for b in beta])
    else:
        raise ValueError(f"Invalid beta parameters: {beta}")

    if len(beta) == 1:
        beta = np.repeat(beta, num_images)
    if len(beta) != num_images:
        raise ValueError(
            f"len(beta)={len(beta)} does not match num_images={num_images}"
        )

    if verbose:
        print(f"Simulation PSD beta: {beta}")
    return beta


def _scale_output(out, H_shaped, p0, freq0, resolution, max_amp):
    # Note that this will only be called for 2D cases at bottom of `simulate`
    if max_amp is not None:
        out *= max_amp / np.max(out)
    else:
        # Otherwise, calculate the power spectral density of 1st
        # realization so that we can scale
        psd = Psd.from_image(out, resolution=resolution, freq0=freq0)
        norm_factor = np.sqrt(np.array(p0) / psd.p0)
        H_shaped *= norm_factor
        out = ifft2(H_shaped).real
    return np.squeeze(out)


def _get_freqs(N, resolution, positive=False, shift=False):
    """Get the frequencies for a given image size and resolution

    Parameters
    ----------
    N : int
        size of image
    resolution : float
        spatial resolution of image in meters
    positive : bool, optional (default=False)
        return positive frequencies
    shift : bool, optional (Default value=False)
        fftshift the frequencies to the center of the image

    Returns
    -------
    freqs : ndarray
        frequencies in cycles / m
    """
    freqs = fftfreq(N, d=resolution)
    if shift:
        freqs = fftshift(freqs)
    if positive:
        freqs = freqs[freqs > 0]
    return freqs


def _resolution_from_freqs(freqs):
    """Find the resolution back from the fft frequencies"""
    df = np.diff(freqs)[0]
    fmax = np.max(freqs) + df
    return round(1.0 / 2 / fmax)


class Psd:
    """Class for representing a power spectral density.

    Attributes
    ----------
    beta : Polynomial
        Estimated slope of PSD(s).
        Several options for input:
        1. a single scalar, used as the linear slope
        3. a 2d array of shape (1, deg+1) of polynomial coefficients
        4. a numpy.polynomial.Polynomial
        Each of these options will be converted to a Polynomial
    resolution : float
        Spatial resolution of image in meters.
    shape : tuple
        Shape of the image(s) that this PSD was calculated from.
    freq0 : Optional[float]
        Reference frequency in cycles / m, used to compute `p0`.
    freq : Optional[ndarray]
        Spatial frequencies in cycle / m.
    psd1d : ndarray
        1D power spectral density at each spatial frequency in `freq`.
        Units are m^2 / (1/m^2)
    p0 : ndarray
        Estimated power at reference frequency `freq0`.
    """

    def __init__(
        self,
        beta=None,
        resolution=None,
        shape=(300, 300),
        freq0=None,
        freq=None,
        psd1d=None,
        p0=None,
    ):
        self.beta = _standardize_beta(beta, 1)[0]
        self.shape = shape
        if resolution is None and freq is None:
            raise ValueError("Must provide freqs or resolution")
        if freq is not None:
            resolution = _resolution_from_freqs(freq)
        else:
            freq = _get_freqs(min(shape[-2:]), resolution, positive=True, shift=True)

        self.resolution = resolution
        self.freq = freq
        self.freq0 = self._get_freq0(freq, freq0)

        self.p0 = p0 if p0 is not None else self._eval_freq(self.freq0, self.beta)
        if psd1d is not None:
            self.psd1d = np.array(psd1d)
        else:
            psd1d = self._eval_freq(self.freq, self.beta)
            psd1d *= self.p0 / self._eval_freq(self.freq0, self.beta)
            self.psd1d = psd1d

    @staticmethod
    def _eval_freq(freq, beta):
        """Evaluate the power spectral density at a given frequency"""
        return np.power(10, beta(np.log10(freq)))

    def simulate(self, shape=None, **kwargs):
        """Simulate a power spectral density.
        See docstring of turbulence.simulate for full information
        """
        if shape is None:
            shape = self.shape
        return simulate(
            shape=shape,
            beta=self.beta,
            p0=self.p0,
            freq0=self.freq0,
            resolution=self.resolution,
            **kwargs,
        )

    @classmethod
    def from_image(
        cls,
        image,
        resolution,
        freq0=None,
        deg=3,
        crop=True,
        N=None,
    ):
        """Get the radially averaged 1D PSD of in input 2D/3D image

        Parameters
        ----------
        image : 2D or 3D ndarray
            image or stack of images, units in meters
        resolution : float
            spatial resolution of input image in meters (Default value = 60)
        freq0 : float
            Reference spatial frequency in cycle / m.
            If not passed, default value = 1e-4, as long as it falls
            within the possible range of the image.
            If the image is too small, will divide by 2 until it fits.
        deg : int
            degree of Polynomial to fit to PSD. default = 3, cubic
        crop : bool
            crop the image into a square image with fewest non-zero pixels
            (Default value = True)
        N : int
            Crop the image to a square with max size N.
            If None, will use the largest size that keeps the image square.

        Returns
        -------
        Psd object, or PsdStack if image is 3D

        Examples
        --------
        >>> from troposim.turbulence import Psd
        >>> import numpy as np
        >>> image = np.random.rand(100, 100)
        >>> psd = Psd.from_image(image, resolution=180)
        """
        if image.ndim > 2:
            return PsdStack.from_images(
                image,
                resolution=resolution,
                freq0=freq0,
                deg=deg,
                crop=crop,
                N=N,
            )

        psd2d = cls._get_psd2d(image, shift=True, crop=crop, N=N, resolution=resolution)

        # calculate the radially average spectrum
        # freq, psd1d = radial_average_spectrum(psd2d, resolution)
        freq, psd1d = cls._average_radially(psd2d, resolution=resolution)
        freq0 = cls._get_freq0(freq, freq0)

        # calculate slopes from spectrum
        beta_hat = cls.fit_psd1d(freq, psd1d, deg=deg)
        p0 = cls._eval_freq(freq0, beta_hat)
        beta_hat = np.array([beta_hat])
        return Psd(
            beta=beta_hat,
            resolution=resolution,
            shape=image.shape,
            psd1d=psd1d,
            freq0=freq0,
            p0=p0,
        )

    @staticmethod
    def _get_freq0(freq, freq0):
        """Check that freq0 is in the range of the image"""
        # If it's specified and incorrect, raise an error
        if freq0 is not None:
            if freq0 < np.min(freq) or freq0 > np.max(freq):
                raise ValueError(
                    f"freq0={freq0} is out of range {np.min(freq):.2E}-{np.max(freq):.2E}."
                )
            else:
                return freq0
        # Otherwise, pick one for the user
        # print("Finding suitable reference frequency")
        if freq0 is None:
            freq0 = DEFAULT_FREQ0
        # For smaller images, 1e-4 might be too small, so try increasing
        pos_freqs = freq[freq > 0]
        while freq0 < np.min(pos_freqs):
            freq0 *= 2
        # and double check that we didn't overshoot somehow
        freq0 = np.clip(freq0, np.min(pos_freqs), np.max(pos_freqs))
        return freq0

    def save(self, filename="psd_params.npz", save_dir=None):
        """Save the PSD parameters to a file.

        Parameters
        ----------
        filename : str, optional
            Filename to save the PSD parameters
            If not passed, will save to `psd_params.npz`
        save_dir : str, optional
            Directory to save the PSD parameters
        """
        if filename is None:
            filename = "psd_params.npz"
        if save_dir is None:
            save_dir = Path()
        save_name = save_dir / Path(filename)

        np.savez(
            save_name,
            beta=[b.coef],
            psd1d=self.psd1d,
            resolution=self.resolution,
            shape=self.shape,
            freq0=self.freq0,
        )
        return save_name

    @classmethod
    def load(cls, filename):
        """Load a saved PSD parameter file

        Parameters
        ----------
            filename (str or Path): Name of the file to load.

        Returns
        -------
            new Psd object
        """
        # TODO: Do i also wanna allow other formats (h5py)?
        if not str(filename).endswith(".npz"):
            filename = str(filename) + ".npz"
        if not Path(filename).exists():
            raise ValueError(f"PSD parameter from {filename} does not exist")

        with np.load(filename) as data:
            beta = data["beta"]
            shape = data.get["shape"]
            resolution = data["resolution"]
            freq0 = data.get("freq0")
            psd1d = data.get("psd1d")

        return cls(
            beta=beta, resolution=resolution, shape=shape, freq0=freq0, psd1d=psd1d
        )

    def asdict(self, include_psd1d=True) -> dict:
        """Save the PSD parameters to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the PSD parameters
        """
        d = {
            # "p0": self.p0.item(),
            "beta": [self.beta.coef.tolist()],  # list of lists, so it's (1, 4) not (4,)
            "shape": self.shape,
            "resolution": self.resolution,
            "freq0": self.freq0,
            # "psd1d": self.psd1d.ravel().tolist(),
            # Freq can be found from the resolution, so don't save it
            # "freq": self.freq.tolist(),
        }
        if include_psd1d and self.psd1d is not None:
            d["psd1d"] = self.psd1d.ravel().tolist()
        return d

    @classmethod
    def from_dict(cls, psd_dict: dict):
        """Load a saved PSD parameter file

        Parameters
        ----------
            psd_dict (dict): Dictionary representation of the PSD parameters

        Returns
        -------
            new Psd object
        """
        beta = _standardize_beta(psd_dict["beta"], 1)[0]
        shape = tuple(psd_dict["shape"])
        resolution = psd_dict["resolution"]
        freq0 = psd_dict.get("freq0")
        psd1d = psd_dict.get("psd1d")
        return cls(
            beta=beta, shape=shape, resolution=resolution, freq0=freq0, psd1d=psd1d
        )

    @staticmethod
    def _get_psd2d(
        image=None,
        shift=True,
        crop=False,
        N=None,
        resolution=60.0,
    ):
        """Calculate the 2d Power spectral density of and image/stack of images

        Parameters
        ----------
        image : ndarray
            image or stack of images (Default value = None)
        shift : bool
            apply `fftshift` in the Fourier domain. Defaults to True.
        crop : bool
            Crop the image to a square of size `N`. Defaults to False.
        N : [type]
            For cropping, size of square. Defaults to None.
            If None, will find the largest sqaure with greatest number of nonzeros
            (see `utils.crop_square_data`)
        resolution : float
            spatial resolution of input image in meters
            used if to normalize.

        Returns
        -------
        psd2d: ndarray
            2D Power spectral density of input image.
        """
        # use the square part of the matrix for spectrum calculation
        if crop:
            image = utils.crop_square_data(image, N=N)
        rows, cols = image.shape[-2:]

        # calculate the normalized power spectrum (spectral density)
        fdata2d = fft2(np.nan_to_num(image, nan=0.0))
        if shift:
            fdata2d = fftshift(fdata2d, axes=(-2, -1))
        psd2d = np.abs(fdata2d) ** 2 / (rows * cols)

        # Convert to density units (m^2 / (1/m^2), or (Amplitude^2) / (1 / (sampling units)^2))
        # Here the sampling units are meters, given by the resoltution
        psd2d *= resolution**2
        # print(f"mult psd2d by {resolution**2:.1f}")
        # print(f"if in [km]: {(resolution/1000)**2:.1f}")
        # print(f"So dividing by Fs**2 leads to boost of {(1000/resolution)**2:.1f}")

        return psd2d

    @staticmethod
    def _average_radially(
        psd2d,
        resolution=60.0,
    ):
        """Calculate the radially averaged power spectrum (assumes isotropy)

        Parameters
        ----------
        psd2d : ndarray
            precomputed 2d psd
        resolution : float
            (Default value = 60.0)

        Returns
        -------
        freq_pos : ndarray
            radial frequency in cycles / m
        psd1d : ndarray
            power spectral density in (m^2) / (1 / m^2)
        """
        h, w = psd2d.shape[-2:]
        hc, wc = h // 2, w // 2
        # Only extend out to shortest dimension
        # this will miss power contributions in 'corners' r > min(hc, wc)
        num_r = min(hc, wc)

        # create an array of integer radial distances from the center
        Y, X = np.ogrid[0:h, 0:w]
        # These act as "labels" in the ndimage summation
        r = np.hypot(X - wc, Y - hc).round().astype(int)
        # r *= resolution # for the actual radial distances in image

        # average all psd2d pixels with label 'r' for 0<=r<=wc
        psd1d = ndimage.mean(psd2d, r, index=np.arange(0, num_r))

        # Also return the positive frequencies with this
        N = min(h, w)
        freq_pos = _get_freqs(N, resolution, positive=True, shift=True)
        # For even-sized images, there's an offset
        # see fftfreq's case, where there are (N - 1) // 2 positive freqs
        if N % 2 == 0:
            psd1d = psd1d[1:]
        return freq_pos, psd1d

    @staticmethod
    def fit_psd1d(freq, psd, deg=3, verbose=False):
        """Fit the slope `beta` and p0 of a 1D PSD in loglog scale
        p = p0 * (freq/freq0)^(-beta)

        Python translation of pslope.m (Ramon Hanssen, 2000), with added
        higher order polynomial option

        Parameters
        ----------
        freq : 1D / 2D ndarray
            in cycle / m
        psd : 1D / 2D ndarray
            power spectral density
        freq0 : float
            reference frequency in cycle / m
        deg : int
            degree of polynomial fit to log-log plot (default = 3, cubic)
        verbose : bool
            (Default value = False)

        Returns
        -------
        p0 : float
            power at reference frequency
        beta : np.Polynomial
            polynomial representing the fitted 1D PSD
        """
        freq = freq.flatten()
        psd = psd.flatten()

        # check if there is zero frequency. If yes, remove it.
        idx = freq != 0.0
        if not np.all(idx):
            freq = freq[idx]
            psd = psd[idx]

        if np.all(psd == 0.0):
            print("Warning: All PSD values are zero")
            beta_poly = Polynomial.fit(freq, psd, deg=deg, domain=[])
            return 0.0, beta_poly

        # convert to log-log scale
        logf = np.log10(freq)
        logp = np.log10(psd)

        # Ignore once it's flat at basically 0
        nonzero_idx = (np.abs(logp) > 1e-16) & np.isfinite(logp) & ~np.isnan(logp)
        logp = logp[nonzero_idx]
        logf = logf[nonzero_idx]

        # Note: domain=[] necessary to return expected typical power series coefficients
        # Makes it equivalent to older `np.polyfit(logf, logp, deg=deg)`
        # See https://github.com/numpy/numpy/issues/9533, https://stackoverflow.com/a/52339988
        # Reminder that fitting without domain=[] requires .convert() to remove affine map
        beta_poly = Polynomial.fit(logf, logp, deg=deg, domain=[])
        if verbose:
            print(f"Polyfit fit: {beta_poly}")

        return beta_poly

    @classmethod
    def from_hdf5(
        cls,
        hdf5_file,
        dataset,
        resolution,
        freq0=None,
        deg=3,
        crop=True,
    ):
        """Find the PSD of a stack of images saved in an HDF5 file

        Parameters
        ----------
        h5_file : str
            Name of HDF5 file
        dataset : str
            dataset within the HDF5 containing the images
        resolution : float

        Returns
        -------
        Psd object
        """
        import h5py

        with h5py.File(hdf5_file, "r") as f:
            dset = f[dataset]
            if dset.ndim == 3 and len(dset) > 1:
                return PsdStack.from_images(
                    dset, resolution, freq0=freq0, deg=deg, crop=crop
                )
            else:
                return cls.from_image(dset, resolution, freq0=freq0, deg=deg, crop=crop)
            # stack = f[dataset][:]

    def plot(self, idxs=None, ax=None, **kwargs):
        from troposim import plotting

        if idxs is None:
            idxs = slice(None)
        if "color" not in kwargs:
            kwargs["color"] = "black"

        return plotting.plot_psd1d(self.freq, self.psd1d[idxs].T, ax=ax, **kwargs)

    def __repr__(self):
        # The repr version can be used to reconstruct the object
        return self._to_str(poly_as_str=False)

    def __str__(self):
        return self._to_str(poly_as_str=True, precision=2)

    def _to_str(self, poly_as_str: bool, precision=None):
        with np.printoptions(precision=precision):
            if poly_as_str:
                beta_str = str(self.beta)
            else:
                beta_str = repr(self.beta)
        s = f"Psd(beta={beta_str}, shape={self.shape}, resolution={self.resolution}"
        if self.freq0 is not None:
            s += f", freq0={self.freq0}"
        s += ")"
        return s

    def __rich_repr__(self):
        yield "beta", self.beta
        yield "shape", self.shape
        yield "resolution", self.resolution
        yield "freq0", self.freq0

    def __eq__(self, other):
        a = self.shape == other.shape
        b = np.array_equal(self.beta, other.beta)
        c = self.resolution == other.resolution
        return a and b and c

    def __add__(self, other):
        self._assert_compatible(other)
        return Psd(
            beta=(self.beta + other.beta) / 2,  # average the beta polynomials
            resolution=self.resolution,
            shape=self.shape,
            freq0=self.freq0,
            psd1d=self.psd1d + other.psd1d,
        )

    def __mul__(self, c):
        if isinstance(c, (int, float)):
            # Scalar multiplication
            # Note: this is designed to match the result of multiplying the
            # 1D PSD by a constant. The 1D PSD is a log-log curve, the beta
            # polynomial gets changed 1) only in the constant term, and 2) by
            # a factor of log10(c). The 1D PSD is then multiplied by c.
            psd1d = self.psd1d * c if self.psd1d is not None else None
            constant_factor = np.log10(c)
            beta = self.beta.copy()
            beta.coef[0] += constant_factor
            return Psd(
                beta=beta,
                resolution=self.resolution,
                shape=self.shape,
                freq0=self.freq0,
                psd1d=psd1d,
            )
        else:
            raise TypeError("Multiplication only supported by scalar")

    def __rmul__(self, c):
        return self.__mul__(c)

    # the div allows us to do (psd1 + psd2) / 2
    def __truediv__(self, c):
        if isinstance(c, (int, float)):
            # See comment in __mul__ for explanation of this
            psd1d = self.psd1d * c if self.psd1d is not None else None
            constant_factor = np.log10(c)
            beta = self.beta.copy()
            beta.coef[0] -= constant_factor
            return Psd(
                beta=beta,
                freq0=self.freq0,
                shape=self.shape,
                resolution=self.resolution,
                psd1d=psd1d,
            )
        else:
            raise TypeError("Division only supported by scalar")

    def _assert_compatible(self, other):
        if not isinstance(other, Psd):
            raise TypeError("Both objects must be `Psd` instances")
        if not self.freq0 == other.freq0:
            raise ValueError("Psd objects must have same freq0")
        if not self.shape == other.shape:
            raise ValueError("Psd objects must have same shape")

    def copy(self, deep=True):
        return copy.deepcopy(self) if deep else copy.copy(self)

    @classmethod
    def from_p0_beta(cls, p0, beta, resolution, shape, freq0):
        """Reconstruct 1D power spectral density array from p0 and beta

        Parameters
        ----------
        p0 : float
            power spectral density in m^2 at `f0` Hz
        beta
            power spectra slope in loglog scale
            if scalar passed, assuming it single slope of loglog plot
        resolution : float
            spatial resolution of the image in meters
        shape : tuple[int]
            desired shape of image.
        freq0 : float
            reference spatial frequency in cycle / m

        Returns
        -------
        Psd object
        """
        # frequency for x-axis after FFT
        N = min(shape[-2:])
        freq = _get_freqs(N, resolution, positive=True, shift=True)
        freq0 = cls._get_freq0(freq, freq0)

        # Make `beta` into a Polynomial
        beta = _standardize_beta(beta, 1)[0]

        psd1d = cls._eval_freq(freq, beta)
        psd1d *= p0 / cls._eval_freq(freq0, beta)

        beta_hat = cls.fit_psd1d(freq, psd1d, deg=beta.degree())
        return Psd(
            beta=beta_hat,
            resolution=resolution,
            shape=tuple(shape),
            freq0=freq0,
            # Note: not including psd1d here because it came from beta evaluation,
            # not from fitting.
        )


class PsdStack:
    psd_list: List[Psd]

    @classmethod
    def from_images(
        cls,
        stack,
        resolution,
        freq0=None,
        deg=3,
        crop=True,
        N=None,
    ):
        """Find the PSD estimates for a stack of images.

        Parameters
        ----------
        stack : 3D ndarray
            displacement in meters (num_images, rows, cols)
        resolution : float
            spatial resolution of input data in meters
        freq0 : float
            reference spatial frequency in cycle / m.
        deg : int
            degree of Polynomial to fit to PSD. default = 3, cubic
        crop : bool
            crop the data into a square image with fewest non-zero pixels
            (Default value = True)
        N : int
            size to crop square (defaults to size of smaller side)
        resolution : float
            (Default value = 60.0)
        freq0 : float
            (Default value = 1e-4)

        Returns
        -------
        Psd object with iterables for p0, beta, and psd1d
        """
        psd_list = []
        skip_progress_bar = len(stack) < 5
        for image in tqdm(stack, disable=skip_progress_bar):
            psd_list.append(
                Psd.from_image(
                    image,
                    resolution=resolution,
                    freq0=freq0,
                    deg=deg,
                    crop=crop,
                    N=N,
                )
            )
        return cls(psd_list)

    def simulate(
        self, num_days=None, shape=None, randomize=False, **kwargs
    ) -> np.ndarray:
        """Simulate the stack of images from the PSD

        Returns
        -------
        ndarray
            simulated stack of images
        """
        if num_days is None:
            num_days = self.shape[0]
        if shape is None:
            shape = self.shape[-2:]
        stack_shape = (num_days,) + shape
        if not randomize:
            # if len(beta)=3 does not match num_images=50
            if len(self.beta) > 1 and len(self.beta) != num_days:
                raise ValueError(
                    "Number of days must match number of beta values if randomize=False"
                )
            return simulate(
                shape=stack_shape,
                beta=self.beta,
                resolution=self.resolution,
                p0=self.p0,
                freq0=self.freq0,
                **kwargs,
            )

        # Now we need to randomize the PSDs with replacement
        psd_list = self.psd_list
        # if num_days > len(psd_list):
        #     # pad the list with copies of itself if we need more
        #     psd_list = psd_list * (num_days // len(psd_list) + 1)

        psd_list = np.random.choice(psd_list, size=(num_days,), replace=True)

        # Get a set of random indices with replacement
        beta = np.array([psd.beta for psd in psd_list]).ravel()
        p0_array = np.array([psd.p0 for psd in psd_list]).ravel()

        return simulate(
            shape=stack_shape,
            beta=beta,
            resolution=self.resolution,
            p0=p0_array,
            freq0=self.freq0,
            **kwargs,
        )

    def __init__(self, psd_list):
        if not all([psd.shape == psd_list[0].shape for psd in psd_list]):
            raise ValueError("All Psd objects must have same shape")
        if not all([psd.resolution == psd_list[0].resolution for psd in psd_list]):
            raise ValueError("All Psd objects must have same resolution")
        if not all([psd.freq0 == psd_list[0].freq0 for psd in psd_list]):
            raise ValueError("All Psd objects must have same freq0")
        self.psd_list = psd_list

    def __str__(self):
        return f"PsdStack with {len(self)} images"

    def __repr__(self) -> str:
        return (
            f"PsdStack(psd_list=[{', '.join([repr(psd) for psd in self.psd_list])}]])"
        )

    def __rich_repr__(self):
        yield "psd_list", self.psd_list[:5]
        if len(self.psd_list) > 5:
            yield f"...({len(self.psd_list) - 5} more)"

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, PsdStack):
            return False
        return all([psd1 == psd2 for psd1, psd2 in zip(self.psd_list, o.psd_list)])

    @property
    def p0(self):
        return np.array([psd.p0 for psd in self.psd_list])

    @property
    def beta(self):
        return np.array([psd.beta for psd in self.psd_list])

    @property
    def psd1d(self):
        return np.array([psd.psd1d for psd in self.psd_list])

    @property
    def freq(self):
        return self.psd_list[0].freq

    @property
    def freq0(self):
        return self.psd_list[0].freq0

    @property
    def resolution(self):
        return self.psd_list[0].resolution

    @property
    def shape(self):
        return (len(self.psd_list),) + self.psd_list[0].shape

    def __getitem__(self, idx):
        p = self.psd_list[idx]
        if isinstance(idx, int):
            return p
        return PsdStack(p)

    def __setitem__(self, idx, value):
        self.psd_list[idx] = value

    def __len__(self):
        return len(self.psd_list)

    def __add__(self, other):
        if not isinstance(other, PsdStack):
            raise TypeError("Both objects must be `PsdStack` instances")
        return PsdStack(self.psd_list + other.psd_list)

    def asdict(self, include_psd1d=True):
        return dict(psd_list=[psd.asdict(include_psd1d) for psd in self.psd_list])

    @classmethod
    def from_dict(cls, d):
        return cls([Psd.from_dict(psd) for psd in d["psd_list"]])

    def plot(self, idxs=None, ax=None, **kwargs):
        from troposim import plotting

        if idxs is None:
            idxs = slice(None)
        if "color" not in kwargs:
            kwargs["color"] = "grey" if len(self) > 1 else "black"

        return plotting.plot_psd1d(self.freq, self.psd1d[idxs].T, ax=ax, **kwargs)
