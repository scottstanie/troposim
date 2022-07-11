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
from pathlib import Path

import numpy as np
from numpy.polynomial.polynomial import Polynomial, polyval
from scipy import ndimage
from scipy.fft import fft2, fftfreq, fftshift, ifft2
from tqdm import tqdm

from . import utils

RNG = np.random.default_rng()

# TODO: auto-pick a freq0? Shouldn't have to make people know
# in advance

def simulate(
    shape=(300, 300),
    beta=2.5,
    p0=10.0,
    freq0=1e-4,
    max_amp=None,
    resolution=60.0,
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
        (Default value = 2.5)
    p0 : float
        Power level of PSD at the refernce freqency `freq0`
        Units are m^2 / (1/m^2) (Default value = 10.0)
    freq0 (float), reference spatial freqency where `p0` defined), in cycle / m
        (default 1e-4, or 1 cycle/10 km)
    max_amp : float or ndarray[float], optional
        maximum amplitude (in meters) of the simulated turbulence.
        Alternative to passing `p0` and `freq0`
        If passing an array, must match shape[0] of the stack
    resolution : float
        spatial resolution of output pixels, in meters (Default value = 60.0)
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
    >>> out = turbulence.simulate(shape=(200, 200), beta=2.5)
    >>> # Stack of 4 images of 200x200 pixels, noise increasing in spatial scale
    >>> out = turbulence.simulate(shape=(4, 200, 200), beta=[2.0, 2.2, 2.7, 3.0])
    """
    if p0 is None or np.all(np.array(p0) == 0):
        return np.zeros(shape)

    try:
        num_images, length, width = shape
    except ValueError:
        num_images = 1
        length, width = shape

    # Start with the 2D PSD of white noise: flat amplitude, random phase
    rng = np.random.default_rng(seed) if seed is not None else RNG
    h = rng.uniform(size=shape)
    H = np.exp(1j * 2 * np.pi * h)

    # spatial frequencies for polynomial evaluation (units: cycles / m)
    fx = _get_freqs(width, resolution)
    fy = _get_freqs(length, resolution)
    # Broadcast 1D vectors to square: (N,1) + (1, N) = (N, N)
    f = np.sqrt(fy[:, np.newaxis] ** 2 + fx[np.newaxis, :] ** 2)

    # Make `beta` into an array of Polynomials
    beta = _standardize_beta(beta, num_images, verbose=verbose)
    # The power beta/2 is used because power ~ amplitude**2
    # Using amplitude means we take sqrt( k**beta) = k**(beta/2)
    beta_amp = beta / 2

    # Now evaluate each using the coefficients
    # Note: the polyval is like doing P = k ** (beta), but allows cubic, etc.
    b_coeffs = np.array([b.coef for b in beta_amp])
    # places where f=0 will fail in log10
    with np.errstate(invalid="ignore", divide="ignore"):
        logP = polyval(np.log10(f), b_coeffs.T)

    # create the envelope to shape the power of the white noise
    P = 10**logP
    # correct dividing by zero, paind in case simulating multiple images
    P[..., f == 0] = 1.0

    H_shaped = H * P
    # Make output zero mean by zeroing the top left (0 freq) element
    H_shaped[..., 0, 0] = 0.0
    out = ifft2(H_shaped, workers=-1).real
    # If passed the option of a maximum amplitude, use that and return
    return _scale_output(out, H_shaped, p0, freq0, resolution, max_amp)


def _standardize_beta(beta, num_images, verbose=False):
    """Get beta into an ndarray of Polynomials with length = num_images

    Allowed options are
    1. a single scalar, used as the linear slope
    2. a list/array of scalars, equal in length to `num_images`
    3. a 2d array of shape (num_images, deg+1) of polynomial coefficients
    """
    # If Polynomials are passed, extract the coefficients to simplify later logic
    if isinstance(beta, Polynomial):
        beta = np.array([beta])
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
                print(f"reversed sign on scalar slope: {beta = }")
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
        raise ValueError(f"{len(beta) = } does not match {num_images = }")

    if verbose:
        print(f"Simulation PSD polynomial: {beta = }")
    return beta


def _scale_output(out, H_shaped, p0, freq0, resolution, max_amp):
    num_images = len(out)
    if max_amp is not None:
        if np.isscalar(max_amp):
            max_amp = np.ones(num_images) * max_amp
        max_amp = np.expand_dims(max_amp, axis=(-2, -1))
        img_maxes = np.max(out, axis=(-2, -1), keepdims=True)
        out *= max_amp / img_maxes
    else:
        # Otherwise, calculate the power spectral density of 1st
        # realization so that we can scale
        # psd = get_psd(out, resolution=resolution, freq0=freq0)[0]
        psd = Psd.from_image(out, resolution=resolution, freq0=freq0)
        norm_factor = np.sqrt(np.array(p0) / psd.p0)
        # shape will be (num_images,), or () for 1 image case
        # add the "expand_dims" for the 3D case to broadcast to (num_images, rows, cols)
        H_shaped *= np.expand_dims(norm_factor, axis=(-2, -1))
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
    p0 : ndarray
        Estimated power at reference frequency.
    beta : ndarray[np.Polynomial]
        Estimated slope of PSD(s). Polynomial of degree `deg`.
        one for each image passed (if a stack), but will
        always return an array even for 1
    freq : ndarray
        Spatial frequencies in cycle / m.
    psd1d : ndarray
        1D power spectral density at each spatial frequency in `freq`.
        Units are m^2 / (1/m^2)
    """

    def __init__(
        self, p0=None, beta=None, freq=None, psd1d=None, freq0=None, shape=(300, 300)
    ):
        self.p0 = p0
        self.beta = beta
        self.freq = freq
        self.psd1d = psd1d
        self.freq0 = freq0
        self.shape = shape

    def simulate(self, shape=None, **kwargs):
        """Simulate a power spectral density.
        See docstring of turbulence.simulate for full information
        """
        if shape is None:
            shape = self.shape
        return simulate(
            shape=shape, beta=self.beta, p0=self.p0, freq0=self.freq0, **kwargs
        )

    @classmethod
    def from_image(
        cls,
        image,
        resolution=60.0,
        freq0=1e-4,
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
            Reference spatial frequency in cycle / m. (Default value = 1e-4)
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
        Psd object

        Examples
        --------
        >>> from troposim.turbulence import Psd
        >>> import numpy as np
        >>> image = np.random.rand(100, 100)
        >>> psd = Psd.from_image(image, resolution=180)
        """
        if image.ndim > 2:
            return cls._get_psd_stack(
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
        freq, psd1d = cls.average_radial(psd2d, resolution=resolution)

        # calculate slopes from spectrum
        p0_hat, beta_hat = cls.fit_psd1d(freq, psd1d, freq0=freq0, deg=deg)
        beta_hat = np.array([beta_hat])
        return Psd(
            p0_hat, beta_hat, freq=freq, psd1d=psd1d, freq0=freq0, shape=image.shape
        )

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

        beta_coeffs = [b.coef for b in self.beta]
        np.savez(
            save_name,
            p0=self.p0,
            beta=beta_coeffs,
            freq=self.freq,
            psd1d=self.psd1d,
            shape=self.shape,
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
            p0 = data["p0"]
            beta = data["beta"]
            freq = data["freq"]
            psd1d = data["psd1d"]
            shape = data.get("shape")
        # convert beta to array of polynomials
        beta = np.array([Polynomial(b) for b in beta])
        try:
            p0 = p0.item()  # if scalar, convert to float
        except ValueError:
            pass

        return cls(p0, beta, freq, psd1d, shape=shape)

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
        psd2d *= resolution**2
        # print(f"mult psd2d by {resolution**2:.1f}")
        # print(f"if in [km]: {(resolution/1000)**2:.1f}")
        # print(f"So dividing by Fs**2 leads to boost of {(1000/resolution)**2:.1f}")

        return psd2d

    @classmethod
    def average_radial(
        cls,
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
    def fit_psd1d(freq, psd, freq0=1e-4, deg=3, verbose=False):
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
            reference freqency in cycle / m
        deg : int
            degree of polynomial fit to log-log plot (default = 3, cubic)
        freq0 : float
            (Default value = 1e-4)
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

        # convert to log-log scale
        logk = np.log10(freq)
        logp = np.log10(psd)

        # Ignore once it's flat at basically 0
        nonzero_idx = (np.abs(logp) > 1e-16) & np.isfinite(logp) & ~np.isnan(logp)
        logp = logp[nonzero_idx]
        logk = logk[nonzero_idx]

        # Note: domain=[] necessary to return expected typical power series coefficients
        # Makes it equivalent to older `np.polyfit(logk, logp, deg=deg)`
        # See https://github.com/numpy/numpy/issues/9533, https://stackoverflow.com/a/52339988
        # Reminder that fitting without domain=[] requires .convert() to remove affine map
        beta_poly = Polynomial.fit(logk, logp, deg=deg, domain=[])
        if verbose:
            print(f"Polyfit fit: {beta_poly}")
        # interpolate psd at reference frequency
        if freq0 < freq[0] or freq0 > freq[-1]:
            raise ValueError(
                "input frequency of interest {} is out of range ({}, {})".format(
                    freq0, freq[0], freq[-1]
                )
            )
        # # Interpolate using only two nearest points:
        # logp0 = np.interp(np.log10(freq0), logk, logp)
        # p0 = np.power(10, logp0)
        # Use the fitted polynomial for smoother p0 estimate
        p0 = np.power(10, beta_poly(np.log10(freq0)))

        if verbose:
            print(f"estimated {p0 = :.4g}")
        return p0, beta_poly

    @classmethod
    def _get_psd_stack(
        cls,
        stack,
        resolution=60.0,
        freq0=1e-4,
        deg=3,
        crop=True,
        N=None,
    ):
        """Find the PSD estimates for a stack of images
        Passed onto get_psd

        Parameters
        ----------
        stack : 3D ndarray
            displacement in meters (num_images, rows, cols)
        stack : 3D ndarray
            displacement in meters (num_images, rows, cols)
            resolution (float), spatial resolution of input data in meters
        stack : 3D ndarray
            displacement in meters (num_images, rows, cols)
            resolution (float), spatial resolution of input data in meters
            freq0 (float), reference spatial freqency in cycle / m.
        deg : int
            degree of Polynomial to fit to PSD. default = 3, cubic
        crop : bool
            crop the data into a square image with fewest non-zero pixels (Default value = True)
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
        p0_arr = []
        beta_arr = []
        psd1d_arr = []
        for image in tqdm(stack):
            psd = cls.from_image(
                image,
                resolution=resolution,
                freq0=freq0,
                deg=deg,
                crop=crop,
                N=N,
            )
            p0_arr.append(psd.p0)
            beta_arr.append(psd.beta[0])
            psd1d_arr.append(psd.psd1d)
        p0_arr = np.array(p0_arr)
        beta_arr = np.array(beta_arr)
        psd1d_arr = np.stack(psd1d_arr)
        return cls(
            p0_arr, beta_arr, psd.freq, psd1d_arr, freq0=freq0, shape=image.shape
        )

    def plot(self, idxs=0, ax=None, **kwargs):
        from troposim import plotting

        return plotting.plot_psd1d(self.freq, self.psd1d[idxs], ax=ax, **kwargs)

    def __repr__(self):
        with np.printoptions(precision=2):
            return f"Psd(p0={self.p0}, beta={self.beta}, freq0={self.freq0})"

    def __eq__(self, other):
        a = np.allclose(self.p0, other.p0)
        b = np.array_equal(self.beta, other.beta)
        c = np.allclose(self.freq, other.freq)
        d = np.allclose(self.psd1d, other.psd1d)
        return a and b and c and d

    @classmethod
    def from_p0_beta(cls, p0, beta, resolution, shape, freq0=1e-4):
        """Reconstruct 1D power spectral density array from p0 and beta

        Parameters
        ----------
        psd : Psd object
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
        freq0_idx = np.argmin(np.abs(freq - freq0))

        # logk = np.log10(freq)
        # Make `beta` into an array of Polynomials
        beta = _standardize_beta(beta, 1)

        # logp = beta(logk)
        b_coeffs = np.array([b.coef for b in beta])
        logp = polyval(np.log10(freq), b_coeffs.T)

        psd1d = 10**logp
        psd1d *= p0 / psd1d[freq0_idx]
        return cls(p0, beta, freq, psd1d, shape=shape, freq0=freq0)
