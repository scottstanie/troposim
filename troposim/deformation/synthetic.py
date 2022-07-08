"""Module for simple synthetic deformations, not following any real physical model"""
import numpy as np
import scipy.ndimage as ndi


def gaussian(
    shape,
    sigma,
    row=None,
    col=None,
    normalize=False,
    amp=None,
    noise_sigma=0.0,
):
    """Create a gaussian bowl of given shape and width

    Parameters
    ----------
    shape : tuple[int]
        (rows, cols)
    sigma : float
        std dev of gaussian
    row : int
        center of blob. Defaults to None.
    col : int
        center col of blob. Defaults to None.
    normalize : bool
        Normalize the amplitude peak to 1.
        Defaults to False.
    amp : float
        peak height of gaussian. Defaults to None.
    noise_sigma : float
        Std. dev of random gaussian noise added to image. (Default value = 0.0)

    Returns
    -------
    ndarray
    """
    d = delta(shape, row, col)
    out = ndi.gaussian_filter(d, sigma, mode="constant") * sigma ** 2
    normed = _normalize_gaussian(out, normalize=normalize, amp=amp)
    if noise_sigma > 0:
        normed += noise_sigma * np.random.standard_normal(shape)
    return normed


def delta(shape, row=None, col=None):
    """Create a spike in the middle of an image

    Parameters
    ----------
    shape : tuple[int]
       size of image to maek 
    row :
        (Default value = None)
    col :
        (Default value = None)

    Returns
    -------
    ndarray
    """
    delta = np.zeros(shape)
    rows, cols = shape
    if col is None:
        col = cols // 2
    if row is None:
        row = rows // 2
    delta[row, col] = 1
    return delta


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
    a = np.sqrt(sigma ** 2 / (1 - ecc))
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
    shape,
    a=None,
    b=None,
    sigma=None,
    ecc=None,
    row=None,
    col=None,
    theta=0,
    normalize=False,
    amp=None,
    noise_sigma=0,
):
    """Make an ellipse using multivariate gaussian

    Parameters
    ----------
    shape : tuple[int
        size of grid
    a :
        semi major axis length (Default value = None)
    b :
        semi minor axis length (Default value = None)
    sigma :
        std dev of gaussian, if it were circular (Default value = None)
    ecc :
        from 0 to 1, alternative to (a, b) specification is (sigma, ecc)
        ecc = 1 - (b/a), and area = pi*sigma**2 = pi*a*b (Default value = None)
    row :
        row of center (Default value = None)
    col :
        col of center (Default value = None)
    theta :
        degrees of rotation (CCW) (Default value = 0)
    normalize : bool
        if true, set max value to 1 (Default value = False)
    amp : float
        value of peak of gaussian (Default value = None)
    noise_sigma : float
        optional, adds gaussian noise to blob (Default value = 0)
    
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
    cov = np.dot(R, np.array([[b ** 2, 0], [0, a ** 2]]))
    cov = np.dot(cov, R.T)
    var = multivariate_normal(mean=[col, row], cov=cov)

    xx, yy = _xy_grid(shape)
    xy = np.vstack((xx.flatten(), yy.flatten())).T
    out = var.pdf(xy).reshape(shape)
    normed = _normalize_gaussian(out, normalize=normalize, amp=amp)
    if noise_sigma > 0:
        normed += noise_sigma * np.random.standard_normal(shape)
    return normed


def valley(shape, rotate=0):
    """Make a valley in image center (curvature only in 1 direction) """
    from skimage import transform

    rows, cols = shape
    out = np.dot(np.ones((rows, 1)), (np.linspace(-1, 1, cols) ** 2).reshape((1, cols)))
    if rotate > 0:
        out = transform.rotate(out, None, mode="edge")
    return out


def bowl(shape):
    """Simple quadratic bowl"""
    xx, yy = _xy_grid(shape)
    z = xx ** 2 + yy ** 2
    return z / np.max(z)


def quadratic(shape, coeffs):
    """2D quadratic function"""
    nrows, ncols = shape
    row_block, col_block = np.mgrid[0:nrows, 0:ncols]
    yy, xx = row_block.flatten(), col_block.flatten()
    idx_matrix = np.c_[np.ones(xx.shape), xx, yy, xx * yy, xx ** 2, yy ** 2]
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
