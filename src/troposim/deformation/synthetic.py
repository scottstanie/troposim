"""Module for simple synthetic deformations, not following any real physical model"""

import random
from collections.abc import Sequence

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


def valley(shape, rotate=0):
    """Make a valley in image center (curvature only in 1 direction)"""
    from skimage import transform

    rows, cols = shape
    out = np.dot(np.ones((rows, 1)), (np.linspace(-1, 1, cols) ** 2).reshape((1, cols)))
    if rotate > 0:
        out = transform.rotate(out, None, mode="edge")
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


def check_overlap(
    new_blob: tuple[int, int, float],
    current_blobs: Sequence[tuple[int, int, float]],
    min_distance_factor: float = 3.0,
) -> bool:
    """
    Check if a new Gaussian blob would overlap with existing ones.

    Parameters
    ----------
    new_blob : tuple[int, int]
        Potential new blob (row, col, sigma).
    current_blobs : Sequence[tuple[int, int]]
        Sequence of existing blob centers (row, col, sigma).
    min_distance_factor : float, default=3.0
        Factor multiplied by the sum of sigmas to determine minimum distance.

    Returns
    -------
    bool
        True if the new blob would NOT overlap with existing ones.
    """
    new_row, new_col, new_sigma = new_blob
    for row, col, sigma in current_blobs:
        # Calculate distance between centers
        distance = np.sqrt((row - new_row) ** 2 + (col - new_col) ** 2)
        # Minimum distance to avoid overlap (using 3-sigma rule)
        min_distance = min_distance_factor * (sigma + new_sigma)
        if distance < min_distance:
            return False
    return True


def multiple_gaussians(
    shape: tuple[int, int],
    num_blobs: int,
    sigma_range: tuple[float, float],
    amp_range: tuple[float, float] | None = None,
    avoid_overlap: bool = True,
    min_distance_factor: float = 3.0,
    border_buffer: int | None = None,
    normalize: bool = True,
    noise_sigma: float = 0.0,
    max_attempts: int = 1000,
    seed: int | None = None,
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    """
    Create an image with multiple Gaussian blobs.

    Parameters
    ----------
    shape : tuple[int, int]
        Image dimensions (rows, cols).
    num_blobs : int
        Number of Gaussian blobs to create.
    sigma_range : tuple[float, float]
        Range (min, max) for random sigma values.
    amp_range : tuple[float, float], optional
        Range (min, max) for random amplitude values.
        If None, all blobs have amplitude 1.0.
    avoid_overlap : bool, default=True
        If True, ensures blobs don't overlap significantly.
    min_distance_factor : float, default=3.0
        Minimum distance between blobs as a factor of their combined sigma.
        Only used when avoid_overlap is True.
    border_buffer : int, optional
        Buffer space from image edges where blobs cannot be centered.
        If None, uses half of `max(sigma_range)`.
        Default is None.
    normalize : bool, default=True
        Normalize each Gaussian blob's peak to its amplitude.
    noise_sigma : float, default=0.0
        Standard deviation of random Gaussian noise added to the final image.
    max_attempts : int, default=1000
        Maximum attempts to place a non-overlapping blob before giving up.
    seed: int, optional
        Seed for random generation.
        Default is None (no seeding)

    Returns
    -------
    image: np.ndarray
        2D array containing multiple Gaussian blobs.
    blob_locations: list[tuple[int, int, float]]
        The list of (row, col, sigma) blobs placed in the image

    Notes
    -----
    If avoid_overlap is True and the function cannot place all blobs without
    overlap after max_attempts, it will return with fewer blobs than requested.
    """
    image = np.zeros(shape, dtype=float)
    blob_locations: list[tuple[int, int, float]] = []

    rows, cols = shape
    min_sigma, max_sigma = sigma_range

    # Set default amplitude range if not provided
    if amp_range is None:
        amp_range = (1.0, 1.0)

    if border_buffer is None:
        border_buffer = int(max_sigma // 2)
    # Adjusted placement boundaries accounting for border buffer
    min_row, max_row = border_buffer, rows - border_buffer - 1
    min_col, max_col = border_buffer, cols - border_buffer - 1

    blobs_placed = 0

    if seed:
        random.seed(seed)
    for _ in range(num_blobs):
        # Randomly select parameters for this blob
        sigma = random.uniform(min_sigma, max_sigma)
        amp = random.uniform(amp_range[0], amp_range[1])

        if avoid_overlap and blobs_placed > 0:
            # Try to find a non-overlapping position
            for attempt in range(max_attempts):
                row = random.randint(min_row, max_row)
                col = random.randint(min_col, max_col)

                if check_overlap(
                    (row, col, sigma), blob_locations, min_distance_factor
                ):
                    break

                if attempt == max_attempts - 1:
                    # Couldn't place the blob without overlap after max attempts
                    print(
                        f"Warning: Could only place {blobs_placed} blobs without overlap."
                    )
                    break
            else:
                # This executes if the for loop completes normally (not via break)
                continue
        else:
            # No overlap checking, just place randomly
            row = random.randint(min_row, max_row)
            col = random.randint(min_col, max_col)

        # Add the blob to the image
        blob = gaussian(
            shape,
            sigma,
            row=row,
            col=col,
            normalize=normalize,
            amp=amp,
            noise_sigma=0.0,  # We'll add noise to the final image
        )

        image += blob
        blob_locations.append((row, col, sigma))
        blobs_placed += 1

    # Add noise to the final image if requested
    if noise_sigma > 0:
        image += noise_sigma * np.random.standard_normal(shape)

    return image, blob_locations
