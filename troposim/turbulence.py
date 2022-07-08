# Author: Scott Staniewicz
# Adapted from MintPy fractal.py module,
#   https://github.com/insarlab/MintPy/blob/4c2b6f8f80c86f245f8635bbb0ce2e46b29cb576/mintpy/simulation/fractal.py
#   (Author: Zhang Yunjun, 2019)
# which was a Python translation in MintPy of the matlab scripts written by
# Ramon Hanssen, May 2000, available in the following website:
#     http://doris.tudelft.nl/software/insarfractal.tar.gz
import numpy as np
from numpy.fft import fft2, fftfreq, fftshift, ifft2
from numpy.polynomial.polynomial import Polynomial, polyval
from scipy import ndimage
from tqdm import tqdm

from . import utils

RNG = np.random.default_rng()


def simulate(
    shape=(300, 300),
    beta=2.5,
    p0=1e-2,
    freq0=1e-4,
    resolution=60.0,
    density=False,
    seed=None,
    verbose=False,
):
    """Simulate zero-mean, 2D isotropic turbulence with a power-law power spectral density (PSD)

    Can use either a single slope for beta, or a numpy Polynomial which is fit
    to log10(power) vs log10(frequency)

    For simple turbulence (one PSD slope), try out:
    beta = 3.0 for larger scale turbulence (long spatial correlations, large blobs)
    beta = 2.0 for middle scale turbulence
    beta = 1.3 for small scale turbulence

    Args:
        shape: Tuple[float]: (rows, cols), or (num_images, rows, cols) of output
            If passing 3D shape, will use same `beta` for all layers of output
        beta: float, Polynomial, polynomial coefficients, or array of Polynomials
            For scalar: power law exponent for the slope of the radially averaged 2D spectrum.
            Polynomial: result from fit of radially averaged PSD vs frequency, in log-log scale
                (or, see result from `get_psd`)
            array of Polynomials: one for each layer of output. Must match 3D shape.
        p0 (float, ndarray[float]) default=1e-2: multiplier of power spectral density
            in m^2 (if `density=False`. otherwise, unit is m^2 / (1/m^2))
        freq0 (float), reference spatial freqency (where p0 defined), in cycle / m
            (default 1e-4, or 1 cycle/10 km)
        resolution (float): spatial resolution of output, in meters
        density (bool, optional): Indicates the `p0` argument is for a density
            If True, output units will be (Amplitude^2) / (1 / (sampling units)^2)
        seed (int): number to seed random numbers, for reproducible turbulence
        verbose (bool): print extra debug info

    Returns:
        fsurf (ndarray) equal size to `shape`, units = meters

    Example:
        step = 120 # meters
        shape = (500, 500)
        tropo = simulate(shape=shape, resolution=step)

        # To get noise which matches the power from real image:
        p0 = get_psd(image, resolution=step)[0]
        tropo = simulate(shape=shape, resolution=step, p0=p0)

    Notes:
        1. The beta spectral index values are for the 2D radial average for a surface,
        P(k) for k = sqrt(kx**2 + ky**2).
        The 1D version obtained by a line slice through the surface will have a PSD slope
        of beta-1. For example, to get the same roughness as a line with spectral index
        beta = 1.5, you should pass the value beta=2.5
        2. We use the 2D beta so the output of the simulator matches the result from
        get_psd (which radially averages data)

    Originally based on the [5/3, 8/3, 2/3] power law from Hanssen (2001):
    E.g. equation (4.7.28) from Hanssen (2001):
    P_phi(f) =  P_I(f/f0)   ^ -5/3    for 1.5  <= f0/f <= 50   km (regime[1]-regime[0])
                P_0(f/f0)   ^ -8/3    for 0.25 <= f0/f <= 1.5  km (regime[0])
                P_III(f/f0) ^ -2/3    for 0.02 <= f0/f <= 0.25 km (regime[2]-regime[1])
    # (TODO: these inequalities dont make sense with units? might just be 1/f)


    Reference:
        Hanssen, R. F. (2001). Radar interferometry: data interpretation and
        error analysis (Vol. 2). Springer Science & Business Media.
        Code orignially based on the fracsurfatmo.m written by Ramon Hanssen, 2000.
    """
    if p0 is None or np.all(np.array(p0) == 0):
        return np.zeros(shape)

    try:
        num_images, length, width = shape
    except ValueError:
        num_images = 1
        length, width = shape

    # Get the beta passed into an array of Polynomial
    if np.isscalar(beta):
        if beta > 0:
            beta *= -1.0
            if verbose:
                print(f"reversed sign on scalar slope: {beta = }")
        # Convert to linear polynomial
        beta = np.array([Polynomial([0, beta])])
        # beta = beta * np.ones(3)
    elif len(beta) == num_images:
        # beta is a list of Polynomials. Make into array for broadcasting
        if isinstance(beta[0], Polynomial):
            beta = np.array(beta)
        else:
            beta = np.array([Polynomial(b) for b in beta])
    elif not isinstance(beta, Polynomial):
        beta = np.array([Polynomial(beta)])
    if verbose:
        print(f"Simulation PSD polynomial: {beta = }")

    # simulate white noise signal
    rng = np.random.default_rng(seed) if seed is not None else RNG
    h = rng.uniform(size=shape)
    H = fft2(h)

    # The power beta/2 is used because the power spectral
    # density is proportional to the amplitude squared
    # Here we work with the amplitude, instead of the power
    # so we should take sqrt( k.^beta) = k.^(beta/2)  RH
    beta_amp = beta / 2

    # lambda is radial distance vector (or wavelength. units: meters)
    lmbda_x = np.concatenate((np.arange(width / 2 + 1), np.arange(-width / 2 + 1, 0)))
    lmbda_y = np.concatenate((np.arange(length / 2 + 1), np.arange(-length / 2 + 1, 0)))
    # Broadcast 1D vectors to square: (N,1) + (1, N) = (N, N)
    lmbda = (
        np.sqrt(lmbda_y[:, np.newaxis] ** 2 + lmbda_x[np.newaxis, :] ** 2) * resolution
    )

    # Convert to spatial frequencies for polynomial evaluation
    with np.errstate(divide="ignore"):
        k = 1 / lmbda

    # logP = beta_amp(np.log10(k))
    # create the envelope to shape the power of the white noise
    if num_images == 1:
        # Make into same form as the multiple-beta case
        beta_amp = np.array([beta_amp]).ravel()
    if len(beta_amp) > num_images:
        beta_amp = beta_amp[:num_images]
    elif len(beta_amp) < num_images:
        beta_amp = np.repeat(beta_amp, num_images)[:num_images]
    
    # Now evaluate each using the coefficients
    bcoeffs = np.array([b.coef for b in beta_amp])
    logP = polyval(np.log10(k), bcoeffs.T)

    P = 10 ** logP
    # correct dividing by zero
    P[..., lmbda == 0] = 1.0
    # Pad with front axis if simulating multiple images
    H_shaped = H / P

    # Make output zero mean by zeroing the top left (0 freq) element
    H_shaped[..., 0, 0] = 0.0
    fsurf = ifft2(H_shaped).real
    # calculate the power spectral density of 1st realization so that we can scale
    p1 = get_psd(fsurf, resolution=resolution, freq0=freq0, density=density)[0]

    # scale the spectrum to match the input power spectral density.
    norm_factor = np.sqrt(np.array(p0) / p1)
    # shape will be (num_images,), or () for 1 image case
    # add the "expand_dims" for the 3D case to broadcast to (num_images, rows, cols)
    H_shaped *= np.expand_dims(norm_factor, axis=(-2, -1))
    fsurf = ifft2(H_shaped).real
    return np.squeeze(fsurf)


def get_psd(
    image=None,
    resolution=60.0,
    freq0=1e-4,
    deg=3,
    crop=True,
    N=None,
    density=False,
):
    """Get the radially averaged 1D PSD of input 2D matrix
    Table 4.5 in Hanssen, 2001 (Page 143) has further explaination of outputs.

    Args:
        image (2D ndarray) :  displacement in m.
        resolution (float), spatial resolution of input image in meters
        freq0 (float), reference spatial freqency in cycle / m.
        deg (int): degree of Polynomial to fit to PSD. default = 3, cubic
        crop (bool): crop the image into a square image with fewest non-zero pixels
        N (int): size to crop square (defaults to size of smaller side)
    Returns:
        p0_hat (float) estimate of power spectral density at ref frequency [in m^2]
        beta_hat : numpy.polynomial.Polynomial, the best bit to power profile in loglog scale
        freq (1D ndarray): frequency [cycle/m]
        psd1d (1D ndarray): power spectral density [m^2]

    Notes:
        Variation of checkfr.m (Ramon Hanssen, 2000)

    """
    if image.ndim > 2:
        return get_psd_stack(
            image,
            resolution=resolution,
            freq0=freq0,
            deg=deg,
            crop=crop,
            N=N,
            density=density,
        )

    psd2d = get_psd2d(
        image, shift=True, crop=crop, N=N, resolution=resolution, density=density
    )

    # calculate the radially average spectrum
    # freq, psd1d = radial_average_spectrum(psd2d, resolution)
    freq, psd1d = average_psd_radial(psd2d, resolution=resolution, density=density)

    # calculate slopes from spectrum
    p0_hat, beta_hat = power_slope(freq, psd1d, freq0=freq0, deg=deg)
    return p0_hat, beta_hat, freq, psd1d


def get_psd2d(
    image=None,
    shift=True,
    crop=False,
    N=None,
    density=False,
    resolution=None,
):
    """Calculate the 2d Power spectral density of and image/stack of images

    Args:
        image (ndarray): image or stack of images
        shift (bool, optional): apply `fftshift` in the Fourier domain. Defaults to True.
        crop (bool, optional): Crop the image to a square of size `N`. Defaults to False.
        N ([type], optional): For cropping, size of square. Defaults to None.
            If None, will find the largest sqaure with greatest number of nonzeros
            (see `utils.crop_square_data`)
        density (bool, optional): Normalize the PSD to be a density. Defaults to True.
            If True, output units will be (Amplitude^2) / (1 / (sampling units)^2)
            E.g. for an image with units centimeters, with pixel spacing of 1 meters,
            the PSD will have units (cm^2) / (1/m)^2.
            Otherwise, will only normalize by (1/(rows*cols))
        resolution (float, optional): spatial resolution of input image in meters
            used if `density=False,` to normalize.

    Returns:
        2d ndarray: Power spectral density of image
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

    if density:

        psd2d *= resolution ** 2
        # print(f"mult psd2d by {resolution**2:.1f}")
        # print(f"if in [km]: {(resolution/1000)**2:.1f}")
        # print(f"So dividing by Fs**2 leads to boost of {(1000/resolution)**2:.1f}")
    # else:
    # psd2d /= (rows * cols)

    return psd2d


def get_psd_stack(
    stack,
    resolution=60.0,
    freq0=1e-4,
    deg=3,
    crop=True,
    N=None,
    density=False,
):
    """Find the PSD estimates for a stack of images
    Passed onto get_psd

    Args:
        stack (3D ndarray): displacement in meters (num_iamges, rows, cols)
        resolution (float), spatial resolution of input data in meters
        freq0 (float), reference spatial freqency in cycle / m.
        deg (int): degree of Polynomial to fit to PSD. default = 3, cubic
        crop (bool): crop the data into a square image with fewest non-zero pixels
        N (int): size to crop square (defaults to size of smaller side)

    Returns:
        p0_hat_arr (ndarray[float]) estimates of power spectral density at ref frequency [in m^2]
        beta_hat_list (List[Polynomial]): list of estimates of slope of loglog power profile
        freq (1D ndarray): frequency [cycle/m]
        psd1d_arr (ndarray)): list of power spectral density [m^2] size=(num_images, len(freq))
    """
    p0_hat_arr = []
    beta_hat_list = []
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
            density=density,
        )
        p0_hat_arr.append(p0_hat)
        beta_hat_list.append(beta_hat)
        psd1d_arr.append(psd1d)
    return np.array(p0_hat_arr), beta_hat_list, freq, np.stack(psd1d_arr)


def average_psd_radial(
    psd2d=None,
    image=None,
    resolution=50.0,
    density=False,
):
    """Calculate the radially averaged power spectrum (assumes isotropy)
    Args:
        psd2d (ndarray) of size (N, N) for 2D power spectral density.
            non-square images sized (r, c) will use N = min(r, c)
        image: (ndarray), (optional) if psd2d=None, pass original image to transform
        resolution (float), spatial resolution of input data in meters
    Returns:
        freq: 1D ndarray, size (N - 1)//2, spatial frequency in radial direction
        psd1d: ndarray of size (N - 1)//2 for power spectral density

    Starting point for code:
        https://medium.com/tangibit-studios/2d-spectrum-characterization-e288f255cc59
    """
    if psd2d is None:
        if image is None:
            raise ValueError("need either psd, or pre-transformed image")
        psd2d = get_psd2d(image, resolution=resolution, density=density)
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
    freq = fftshift(fftfreq(N, d=resolution))
    freq_pos = freq[freq > 0]
    # For even-sized images, there's an offset
    # see fftfreq's case, where there are (N - 1) // 2 positive freqs
    if N % 2 == 0:
        psd1d = psd1d[1:]
    return freq_pos, psd1d
    # return freq_pos, psd1d, r


def power_slope(freq, psd, freq0=1e-4, deg=3, verbose=False):
    """Derive the slope beta and p0 of an exponential function in loglog scale
    p = p0 * (freq/freq0)^(-beta)

    Python translation of pslope.m (Ramon Hanssen, 2000), with added
    higher polynomials

    Args:
        freq (1D / 2D ndarray): in cycle / m
        psd (1D / 2D ndarray): power spectral density
        freq0 (float) reference freqency in cycle / m
        deg (int): degree of polynomial fit to log-log plot (default = 3, cubic)
        verbose (bool): print debug info
    Returns:
    p0 (float), power spectral density at reference frequency
        (same unit as the input psd)
    beta (float): slope of power profile in loglog scale
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


def get_psd_blocks(
    data, block_size=None, block_step=None, resolution=60.0, freq0=1e-4, deg=3
):
    """For one image, get radially averaged PSD from multiple blocks within image
    Crops into subsets, calls `get_psd` on each

    Args:
        data (2D ndarray) :  displacement in m.
        resolution (float), spatial resolution of input data in meters
        block_size (float): size of block side, in meters
        block_step (float): amount to shift the block window, in m
        freq0 (float), reference spatial freqency in cycle / m.
        deg (int): degree of Polynomial to fit to PSD. default = 3, cubic
    Returns:
        p0_hat_arr (List[float]) estimates of power spectral density at ref frequency [in m^2]
        beta_hat_list (List[float]): list of estimates of slope of power profile in loglog scale
        freq (1D ndarray): frequency [cycle/m]
        psd1d_arr (List[ndarray])): list of power spectral density [m^2]
    """
    nrow, ncol = data.shape
    if block_size is None:
        block_pix = min(nrow, ncol) // 2
    else:
        block_pix = int(block_size / resolution)
    if block_step is None:
        step = min(nrow, ncol) // 4
    else:
        step = int(block_step / resolution)

    p0_hat_arr, beta_hat_list, psd1d_arr = [], [], []
    freq = None
    row, col = 0, 0
    while row + block_pix - 1 < data.shape[0]:
        while col + block_pix - 1 < data.shape[1]:
            # print(row, col, block_size, block_pix, block_step, step)
            block = data[row : row + block_pix, col : col + block_pix]
            if block.shape != (block_pix, block_pix):
                continue
            p0_hat, beta_hat, freq, psd1d = get_psd(
                block,
                resolution=resolution,
                freq0=freq0,
                crop=False,
                deg=deg,
            )
            p0_hat_arr.append(p0_hat)
            beta_hat_list.append(beta_hat)
            psd1d_arr.append(psd1d)
            col += step
        col = 0
        row += step
    return np.array(p0_hat_arr), np.array(beta_hat_list), freq, np.array(psd1d_arr)


def recon_power_spectral_density(p0, beta, resolution, freq0, N):
    """Reconstruct 1D power spectral density from input p0 and beta

    Args:
        p0 (float): power spectral density in m^2 at `f0` Hz
        beta (Polynomial, float): power spectra slope in loglog scale
            if scalare passed, assuming it single slope of loglog plot
        resolution (float): spatial resolution of the image in meters
        freq0 (float): reference spatial frequency in cycle / m
        N (int): min size of the image
    Returns:
        f: 1D np.ndarray, spatial frequency sequence in cycle / m
        p: 1D np.ndarray, line/polynomial power spectral density fit
    Examples:
        freq, psd = recon_power_spectral_density(p0, beta, resolution, freq0)
    """
    # frequency for x-axis after FFT
    k = fftfreq(N, d=resolution)
    k = k[k > 0]

    logk = np.log10(k)
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
    k = k.flatten()
    return k, p


def fractal_dimension(beta):
    """Convert the beta slope to D, fractal dimension
    This D2 is from Hanssen Section 4.7"""
    return (7.0 - beta + 1.0) / 2.0


def debug(surf):
    return (
        f"min:{np.min(surf):.2g}, max:{np.max(surf):.2g}, "
        f"ptp:{np.ptp(surf):.2g}, mean:{np.mean(surf):.2g}"
    )


# TODO: 0-180 and 180-360 are the same
def average_psd_azimuth(
    psd2d=None, image=None, resolution=60.0, dTheta=30, r_min=0, r_max=np.inf
):
    """Get 1D power spectrum averaged by angular bin (azimuthal average)

    Args:
        psd2d (ndarray): 2D power spectral density, size (N, N)
            non-square images sized (r, c) will use N = min(r, c)
        image (ndarray): (optional) if psd2d=None, pass original image to transform
        dTheta (float): spacing between angle bins, in degrees. default=30.
        r_min (float): optional, to limit the radius where the averaging takes place,
            pass a minimum radius (in pixels)
        r_max (float): optional, maximum radius (in pixels) where the averaging takes place
            r_min / r_max can be used to look in an annulus
    Returns:
        angles: 1D ndarray, size 360 / dTheta, of angles used to average
        psd1d: azimuthal average of each angle bin.

    Reference:
        https://medium.com/tangibit-studios/2d-spectrum-characterization-e288f255cc59
    """
    if psd2d is None:
        if image is None:
            raise ValueError("need either psd, or pre-transformed image")
        psd2d = get_psd2d(image, resolution=resolution)
    theta = _get_theta_sectors(psd2d, dTheta, r_min=r_min, r_max=r_max)

    # use all psd2d pixels with label 'theta' (0 to 180) between (r_min, r_max)
    # psd is symmetric for real data
    angles = np.arange(0, 180, int(round(dTheta)))
    psd1d = ndimage.sum(psd2d, theta, index=angles)

    # normalize each sector to the total sector power
    psd1d /= np.sum(psd1d)

    return angles, psd1d


def _get_theta_sectors(psd2d, dTheta, r_min=0, r_max=np.inf):
    """Helper to make a mask of azimuthal angles of image"""
    h, w = psd2d.shape
    hc, wc = h // 2, w // 2

    # note that displaying PSD as image inverts Y axis
    # create an array of integer angular slices of dTheta
    Y, X = np.ogrid[0:h, 0:w]
    theta = np.rad2deg(np.arctan2(-(Y - hc), (X - wc)))
    theta = np.mod(theta + dTheta / 2 + 360, 360)
    theta = dTheta * (theta // dTheta)
    theta = theta.round().astype(np.int)

    # mask below r_min and above r_max by setting to -999
    R = np.hypot((Y - hc), (X - wc))
    mask = np.logical_and(R >= r_min, R < r_max)
    theta = np.where(mask, theta, np.nan)
    return theta
