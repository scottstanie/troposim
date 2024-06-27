"""Module for simple synthetic deformations, not following any real physical model"""

import numpy as np


def gaussian(
    shape: tuple[int, int],
    sigma: float | tuple[float, float],
    row: int | None = None,
    col: int | None = None,
    normalize: bool = False,
    amp: float | None = None,
    noise_sigma: float = 0.0,
) -> np.ndarray:
    """Create a 2D Gaussian of given shape and width.

    Parameters
    ----------
    shape : tuple[int, int]
        (rows, cols) of the output array
    sigma : float or tuple[float, float]
        Standard deviation of the Gaussian.
        If one float provided, makes an isotropic Gaussian.
        Otherwise, uses [sigma_row, sigma_col] to make elongated Gaussian.
    row : int, optional
        Center row of the Gaussian. Defaults to the middle of the array.
    col : int, optional
        Center column of the Gaussian. Defaults to the middle of the array.
    normalize : bool, optional
        Normalize the amplitude peak to 1. Defaults to False.
    amp : float, optional
        Peak height of Gaussian. If None, peak will be 1/(2*pi*sigma^2).
    noise_sigma : float, optional
        Std. dev of random Gaussian noise added to image. Defaults to 0.0.

    Returns
    -------
    np.ndarray
        2D array containing the Gaussian
    """
    rows, cols = shape
    if not isinstance(sigma, tuple):
        sigma = (sigma, sigma)
    sy, sx = sigma

    # Set default center if not provided
    if row is None:
        row = rows // 2
    if col is None:
        col = cols // 2

    # Create coordinate grids
    y, x = np.ogrid[:rows, :cols]

    # Calculate the 2D Gaussian
    g = np.exp(
        -((x - col) ** 2.0 / (2.0 * sx**2.0) + (y - row) ** 2.0 / (2.0 * sy**2.0))
    )
    normed = _normalize_gaussian(g, normalize=normalize, amp=amp)
    if noise_sigma > 0:
        normed += noise_sigma * np.random.standard_normal(shape)
    return normed


def ramp(
    shape: tuple[int, int], amplitude: float = 1, rotate_degrees: float = 0
) -> np.ndarray:
    """Create a synthetic ramp with optional rotation.

    Parameters
    ----------
    shape : tuple[int, int]
        (rows, cols) of the output array.
    amplitude : float
        The maximum amplitude the ramp reaches.
        Default = 1.
    rotate_degrees : float, optional
        Rotation of the ramp in degrees. 0 degrees is a left-to-right ramp.
        Positive values rotate counterclockwise. Defaults to 0.

    Returns
    -------
    np.ndarray
        2D array containing the synthetic ramp
    """
    rows, cols = shape

    # Create coordinate grids
    y, x = np.ogrid[:rows, :cols]

    # Normalize coordinates to [-1, 1] range
    x = (x - cols / 2) / (cols / 2)
    y = (y - rows / 2) / (rows / 2)

    # Convert rotation to radians
    theta = np.radians(rotate_degrees)

    # Apply rotation
    x_rot = x * np.cos(theta) + y * np.sin(theta)

    # Create ramp
    ramp = (x_rot + 1) / 2  # Normalize to [0, 1] range

    # Scale by amplitude
    ramp *= amplitude

    return ramp


def _rotation_matrix(theta):
    """CCW rotation matrix by `theta` degrees"""
    theta_rad = np.deg2rad(theta)
    return np.array(
        [
            [np.cos(theta_rad), np.sin(theta_rad)],
            [-np.sin(theta_rad), np.cos(theta_rad)],
        ]
    )


def _normalize_gaussian(out, normalize=False, amp=None):
    """Normalize either to 1 max, or to `amp` max"""
    if normalize or amp is not None:
        out /= out.max()
    if amp is not None:
        out *= amp
    return out


def _calc_ab(sigma, ecc):
    """Calculate semi-major/semi-minor axis length from `sigma` and `ecc`entricity"""
    a = np.sqrt(sigma**2 / (1 - ecc))
    b = a * (1 - ecc)
    return a, b


def _xy_grid(shape, xmin=None, xmax=None, ymin=None, ymax=None):
    """Make an xy grid centered at 0,0 in the middle"""
    if xmin is None or xmax is None:
        xmin, xmax = (1, shape[1])
    if ymin is None or ymax is None:
        ymin, ymax = (1, shape[0])
    xx = np.linspace(xmin, xmax, shape[1])
    yy = np.linspace(ymin, ymax, shape[0])
    return np.meshgrid(xx, yy)


def gaussian_ellipse(
    shape: tuple[int, int],
    a: float | None = None,
    b: float | None = None,
    sigma: float | None = None,
    ecc: float | None = None,
    row: int | None = None,
    col: int | None = None,
    theta: float = 0,
    normalize: bool = False,
    amp: float | None = None,
    noise_sigma: float = 0,
) -> np.ndarray:
    """Make an ellipse using multivariate gaussian.

    Parameters
    ----------
    shape : tuple[int, int]
        size of grid
    a : float | None
        semi major axis length
    b : float | None
        semi minor axis length
    sigma : float | None
        std dev of gaussian, if it were circular
    ecc : float | None
        from 0 to 1, alternative to (a, b) specification is (sigma, ecc)
        ecc = 1 - (b/a), and area = pi*sigma**2 = pi*a*b
    row : int | None
        row of center
    col : int | None
        col of center
    theta : float
        degrees of rotation (CCW)
    normalize : bool
        if true, set max value to 1
    amp : float | None
        value of peak of gaussian
    noise_sigma : float
        optional, adds gaussian noise to blob

    Returns
    -------
    np.ndarray
        2D array containing the gaussian ellipse
    """
    from scipy.stats import multivariate_normal

    if row is None or col is None:
        nrows, ncols = shape
        row, col = nrows // 2, ncols // 2

    if sigma is not None and ecc is not None:
        a, b = _calc_ab(sigma, ecc)
    if a is None or b is None:
        raise ValueError("Need a,b or sigma,ecc")

    R = _rotation_matrix(theta)
    # To rotate, we do R @ P @ R.T to rotate eigenvalues of P = S L S^-1
    cov = np.dot(R, np.array([[b**2, 0], [0, a**2]]))
    cov = np.dot(cov, R.T)
    var = multivariate_normal(mean=[col, row], cov=cov)

    xx, yy = _xy_grid(shape)
    xy = np.vstack((xx.flatten(), yy.flatten())).T
    out = var.pdf(xy).reshape(shape)
    normed = _normalize_gaussian(out, normalize=normalize, amp=amp)
    if noise_sigma > 0:
        normed += noise_sigma * np.random.standard_normal(shape)
    return normed


def valley(shape, rotate_degrees=0):
    """Make a valley in image center (curvature only in 1 direction)"""
    from scipy.ndimage import rotate

    rows, cols = shape
    out = np.dot(np.ones((rows, 1)), (np.linspace(-1, 1, cols) ** 2).reshape((1, cols)))
    if rotate_degrees > 0:
        rotate(out, rotate_degrees, mode="edge", output=out)
    return out


def bowl(shape):
    """Simple quadratic bowl"""
    xx, yy = _xy_grid(shape)
    z = xx**2 + yy**2
    return z / np.max(z)


def quadratic(shape, coeffs):
    """2D quadratic function"""
    nrows, ncols = shape
    row_block, col_block = np.mgrid[0:nrows, 0:ncols]
    yy, xx = row_block.flatten(), col_block.flatten()
    idx_matrix = np.c_[np.ones(xx.shape), xx, yy, xx * yy, xx**2, yy**2]
    return np.dot(idx_matrix, coeffs).reshape(shape)


def stack(N=501, max_amp=3, plot=False, cmap="jet"):
    """Simpler composite of 3 areas of blob of different sizes

    Parameters
    ----------
    N :
        (Default value = 501)
    max_amp :
        (Default value = 3)
    plot :
        (Default value = False)
    cmap :
        (Default value = "jet")

    Returns
    -------


    """
    shape = (N, N)
    b1 = gaussian(shape, 60, N // 3, 2 * N // 3)

    b2 = gaussian(shape, 20, N // 3, N // 3)

    # Group of 4 close ones
    b4 = gaussian(shape, 29, 345, 345)
    b4 += gaussian(shape, 29, 73 + 345, 345)
    b4 += gaussian(shape, 29, 345, 73 + 345)
    b4 += gaussian(shape, 29, 73 + 345, 73 + 345)
    out = b1 - 0.65 * b2 + 0.84 * b4
    out *= max_amp / np.max(out)

    if plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(out, cmap=cmap)
        plt.colorbar()
    return out
