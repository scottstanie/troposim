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
from scipy.fft import fft2, fftfreq, fftshift, ifft2
from scipy import ndimage
from tqdm import tqdm

from . import utils


RNG = np.random.default_rng()


def simulate(
    shape=(300, 300),
    beta=2.5,
    p0=10.0,
    freq0=1e-4,
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
        multiplier of power spectral density
        Units are m^2 / (1/m^2) (Default value = 10.0)
    freq0 (float), reference spatial freqency where `p0` defined), in cycle / m
        (default 1e-4, or 1 cycle/10 km)
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
    P = 10 ** logP
    # correct dividing by zero, paind in case simulating multiple images
    P[..., f == 0] = 1.0

    H_shaped = H * P
    # Make output zero mean by zeroing the top left (0 freq) element
    H_shaped[..., 0, 0] = 0.0
    out = ifft2(H_shaped, workers=-1).real
    # calculate the power spectral density of 1st realization so that we can scale
    p1 = get_psd(out, resolution=resolution, freq0=freq0)[0]

    # scale the spectrum to match the input power spectral density.
    norm_factor = np.sqrt(np.array(p0) / p1)
    # shape will be (num_images,), or () for 1 image case
    # add the "expand_dims" for the 3D case to broadcast to (num_images, rows, cols)
    H_shaped *= np.expand_dims(norm_factor, axis=(-2, -1))
    out = ifft2(H_shaped).real
    return np.squeeze(out)


def get_psd(
    image,
    resolution=60.0,
    freq0=1e-4,
    deg=3,
    crop=True,
    N=None,
    outname=None,
):
    """Get the radially averaged 1D PSD of input 2D matrix

    Parameters
    ----------
    image : 2D ndarray
        displacement in meters
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
    outname : str, optional
        Save the PSD parameters to a file.
        If this name exists, the parameters will be loaded from the file.

    Returns
    -------
    p0_hat : ndarray
        Estimated power at reference frequency.
    beta_hat : ndarray[np.Polynomial]
        Estimated slope of PSD(s). Polynomial of degree `deg`.
        Returns one for each image passed (if a stack), but will
        always return an array even for 1
    freq : ndarray
        Spatial frequencies in cycle / m.
    psd1d : ndarray
        1D power spectral density at each spatial frequency in `freq`.
        Units are m^2 / (1/m^2)

    """
    if outname and not str(outname).endswith(".npz"):
        outname = str(outname) + ".npz"
    if outname and Path(outname).exists():
        print(f"Loading PSD parameters from {outname}")
        p0_hat, beta_hat, freq, psd1d = load_psd(outname)
        return p0_hat, beta_hat, freq, psd1d

    if image.ndim > 2:
        return _get_psd_stack(
            image,
            resolution=resolution,
            freq0=freq0,
            deg=deg,
            crop=crop,
            N=N,
            outname=outname,
        )

    psd2d = get_psd2d(image, shift=True, crop=crop, N=N, resolution=resolution)

    # calculate the radially average spectrum
    # freq, psd1d = radial_average_spectrum(psd2d, resolution)
    freq, psd1d = average_psd_radial(psd2d, resolution=resolution)

    # calculate slopes from spectrum
    p0_hat, beta_hat = fit_psd1d(freq, psd1d, freq0=freq0, deg=deg)
    beta_hat = np.array([beta_hat])
    if outname is not None:
        save_psd(p0_hat, beta_hat, freq, psd1d, outname=outname)
    return p0_hat, beta_hat, freq, psd1d


def save_psd(p0_hat, beta_hat, freq, psd1d, outname=None, save_dir=None):
    """Save the PSD parameters to a file.

    Parameters
    ----------
    p0_hat : ndarray
        Estimated power at reference frequency.
    beta_hat : np.Polynomial
        Estimated slope of PSD. Polynomial of degree `deg`.
    freq : ndarray
        Spatial frequencies in cycle / m.
    psd1d : ndarray
        1D power spectral density at each spatial frequency in `freq`.
        Units are m^2 / (1/m^2)
    outname : str, optional
        Filename to save the PSD parameters
        If not passed, will save to `psd_params.npz`
    save_dir : str, optional
        Directory to save the PSD parameters
    """
    if outname is None:
        outname = "psd_params.npz"
    if save_dir is None:
        save_dir = Path()
    save_name = save_dir / Path(outname)

    beta_coeffs = [b.coef for b in beta_hat]
    np.savez(
        save_name,
        p0=p0_hat,
        beta=beta_coeffs,
        freq=freq,
        psd1d=psd1d,
    )
    return save_name


def load_psd(filename):
    """Load a saved PSD parameter file

    Args:
        filename (str or Path): Name of the file to load.

    Returns:
        p0_hat (ndarray): Estimated power at reference frequency.
        beta_hat (np.Polynomial): Estimated slope of PSD. Polynomial of degree `deg`.
        freq (ndarray): Spatial frequencies in cycle / m.
        psd1d (ndarray): 1D power spectral density at each spatial frequency in `freq`.
    """
    with np.load(filename) as data:
        p0_hat = data["p0"]
        beta_hat = data["beta"]
        freq = data["freq"]
        psd1d = data["psd1d"]
    # convert beta_hat to array of polynomials
    beta_hat = np.array([Polynomial(b) for b in beta_hat])
    try:
        p0_hat = p0_hat.item()
    except ValueError:
        pass
    return p0_hat, beta_hat, freq, psd1d


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


def get_psd2d(
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
    psd2d *= resolution ** 2
    # print(f"mult psd2d by {resolution**2:.1f}")
    # print(f"if in [km]: {(resolution/1000)**2:.1f}")
    # print(f"So dividing by Fs**2 leads to boost of {(1000/resolution)**2:.1f}")

    return psd2d


def average_psd_radial(
    psd2d=None,
    image=None,
    resolution=60.0,
):
    """Calculate the radially averaged power spectrum (assumes isotropy)

    Parameters
    ----------
    image : ndarray
        (ndarray), (optional) if psd2d=None, pass original image to transform
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
    if psd2d is None:
        if image is None:
            raise ValueError("need either psd, or pre-transformed image")
        psd2d = get_psd2d(image, resolution=resolution)
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


def _get_psd_stack(
    stack,
    resolution=60.0,
    freq0=1e-4,
    deg=3,
    crop=True,
    N=None,
    outname=None,
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
    outname : str
        Name of output file. If None, no output file is written.

    Returns
    -------
    Same as get_psd, but each item is an iterable of (p0_hat, beta_hat, freq, psd1d)
    """
    p0_hat_arr = []
    beta_hat_arr = []
    psd1d_arr = []
    freq = None
    for image in tqdm(stack):
        p0_hat, beta_hat, freq, psd1d = get_psd(
            image,
            resolution=resolution,
            freq0=freq0,
            deg=deg,
            crop=crop,
            N=N,
        )
        p0_hat_arr.append(p0_hat)
        beta_hat_arr.append(beta_hat[0])
        psd1d_arr.append(psd1d)
    p0_hat_arr = np.array(p0_hat_arr)
    beta_hat_arr = np.array(beta_hat_arr)
    psd1d_arr = np.stack(psd1d_arr)
    if outname is not None:
        save_psd(p0_hat_arr, beta_hat_arr, freq, psd1d_arr, outname=outname)
    return p0_hat_arr, beta_hat_arr, freq, psd1d_arr


def get_psd1d_from_p0_beta(p0, beta, resolution, freq0, N):
    """Reconstruct 1D power spectral density array from p0 and beta

    Parameters
    ----------
    p0 : float
        power spectral density in m^2 at `f0` Hz
    beta : Polynomial
        power spectra slope in loglog scale
        if scalare passed, assuming it single slope of loglog plot
    resolution : float
        spatial resolution of the image in meters
    freq0 : float
        reference spatial frequency in cycle / m
    N : int
        number of points in the 1D PSD

    Returns
    -------
    freq : 1D ndarray
        spatial frequency in cycle / m
    psd1d : ndarray
        1D power spectral density
    """
    # frequency for x-axis after FFT
    freq = _get_freqs(N, resolution, positive=True, shift=True)

    logk = np.log10(freq)
    logf0 = np.log10(freq0)

    if np.isscalar(beta):
        # slop value passed: only linear fit
        logp = -beta * (logk - logf0) + np.log10(p0)
    elif isinstance(beta, Polynomial):
        logp = beta(logk)
    else:
        beta_poly = Polynomial(beta)
        logp = beta_poly(logk)

    p = 10 ** logp
    freq = freq.flatten()
    return freq, p
