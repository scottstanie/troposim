import numpy as np

DATE_FMT = "%Y%m%d"


def crop_square_data(data, N=None):
    """Grab the largest square portion of size N from input 2D input

    Searches for square with greatest number of non-zero values.

    Inputs:
        data (2D ndarray): input image
        N (int): optional, size of square input.
            If not provided, uses N = min(data.shape)

    Uses the integral image (summed area table) of data to quickly
    calculate number of zeros in a square subset:
    https://en.wikipedia.org/wiki/Summed-area_table

    Returns:
       square subset of data, size = (N, N)
    """
    row_slice, col_slice = find_largest_square(data, N=N)
    return data[row_slice, col_slice]


def find_largest_square(data, N=None):
    if N is None:
        N = min(data.shape)
    if N > min(data.shape):
        raise ValueError(
            f"N={N} must be less than the smallest dimension: {min(data.shape)}"
        )

    nonzeros = np.logical_and(data != 0, ~np.isnan(data))
    # Use integral image for quick rectangle area summing of # of True
    S = integral_image(nonzeros)
    row = col = 0
    max_nonzeros = 0
    max_row, max_col = 0, 0
    # Move square subset around to test
    # NOTE: you only ever need to move in one direction
    # (since N == one of the dimensions)
    # so this runs quickly
    while row + N - 1 < data.shape[0]:
        while col + N - 1 < data.shape[1]:
            # See https://en.wikipedia.org/wiki/Summed-area_table
            A = (row, col)
            B = (row + N - 1, col)
            C = (row, col + N - 1)
            D = (row + N - 1, col + N - 1)
            num_nonzeros = S[A] + S[D] - S[B] - S[C]

            if num_nonzeros > max_nonzeros:
                max_nonzeros = num_nonzeros
                max_row, max_col = row, col
            col += 1
        col = 0
        row += 1
    return slice(max_row, max_row + N), slice(max_col, max_col + N)


def integral_image(image):
    """Integral image / summed area table.

    The integral image contains the sum of all elements above and to the
    left of it, i.e.:

    Source: skimage.transform
    ----------
    image : ndarray
        Input image.
    Returns
    -------
    S : ndarray
        Integral image/summed area table of same shape as input image.
    References
    ----------
    .. [1] F.C. Crow, "Summed-area tables for texture mapping,"
           ACM SIGGRAPH Computer Graphics, vol. 18, 1984, pp. 207-212.
    """
    S = image.copy()
    for i in range(image.ndim):
        S = S.cumsum(axis=i)
    return S


def piecewise_linear(x, x0, y0, k1, k2):
    y = np.piecewise(
        x,
        [x < x0, x >= x0],
        [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0],
    )
    return y


def extrema(x):
    return np.min(x), np.max(x)


def find_valid(
    sar_date_list,
    igram_date_list,
    min_date=None,
    max_date=None,
    max_temporal_baseline=900,
    verbose=True,
):
    """Cut down the full list of interferograms and geo dates

    - Cuts date ranges (e.g. piecewise linear solution) with  `min_date` and `max_date`
    - Can ignore whole dates by reading `ignore_geo_file`
    - Prunes long time baseline interferograms with `max_temporal_baseline`
    """
    if verbose:
        ig1 = len(igram_date_list)  # For logging purposes, what do we start with

    valid_geos = sar_date_list
    valid_igrams = igram_date_list

    # Remove geos and igrams outside of min/max range
    if min_date is not None:
        if verbose:
            print(f"Keeping data after min_date: {min_date}")
        valid_geos = [g for g in valid_geos if g > min_date]
        valid_igrams = [
            ig for ig in valid_igrams if (ig[0] > min_date and ig[1] > min_date)
        ]

    if max_date is not None:
        if verbose:
            print(f"Keeping data only before max_date: {max_date}")
        valid_geos = [g for g in valid_geos if g < max_date]
        valid_igrams = [
            ig for ig in valid_igrams if (ig[0] < max_date and ig[1] < max_date)
        ]

    if verbose:
        # This is just for logging purposes:
        too_long_igrams = [
            ig for ig in valid_igrams if temporal_baseline(ig) > max_temporal_baseline
        ]
        print(
            f"Ignoring {len(too_long_igrams)} igrams with longer baseline "
            f"than {max_temporal_baseline} days"
        )

    # ## Remove long time baseline igrams ###
    valid_igrams = [
        ig for ig in valid_igrams if temporal_baseline(ig) <= max_temporal_baseline
    ]

    if verbose:
        print(f"Ignoring {ig1 - len(valid_igrams)} igrams total")

    # Now go back to original full list and see what the indices were
    # used for subselecting from the unw_stack by a pixel
    valid_idxs = np.searchsorted(
        intlist_to_string(igram_date_list),  # Numpy searchsorted can't use datetimes
        intlist_to_string(valid_igrams),
    )
    return valid_geos, valid_igrams, valid_idxs


def temporal_baseline(igram):
    return (igram[1] - igram[0]).days


def intlist_to_string(ifg_date_list, ext=".int"):
    """Convert date pairs to list of string filenames"""
    return [
        "{}_{}{ext}".format(a.strftime(DATE_FMT), b.strftime(DATE_FMT), ext=ext)
        for a, b in ifg_date_list
    ]


# def interpolate_nans(image, order=1):
#     """Interpolate the nan values in an image"""
#     from scipy import interpolate
#     nan_idxs = np.isnan(image)
#     nan_rows, nan_cols = nan_idxs.nonzero()
#     good_rows, good_cols = (~nan_idxs).nonzero()
#     print(np.sum(np.diff(good_cols) < 0))

#     # Use row/columns number as the x/y grid
#     ny, nx = image.shape
#     x = np.arange(nx)
#     y = np.arange(ny)
#     # Make the iinterpolator only use non-nans
#     # interpolater = interpolate.RectBivariateSpline(
#     # x[good_cols], y[good_rows], image[~nan_idxs], kx=order, ky=order
#     # )
#     interpolater = interpolate.interp2d(
#         x[good_cols],
#         y[good_rows],
#         image[~nan_idxs],
#     )

#     # Now fill just in indices where there were nans
#     output = image.copy()
#     output[nan_idxs] = interpolater(nan_rows, nan_cols)
#     return output


def anisotropy_var(azi_psd1d):
    return np.var(azi_psd1d)


def anisotropy_A(azi_psd1d):
    return 2 * np.ptp(azi_psd1d)


def rms_noise_vs_dist(image, resolution, crop=True):
    """Get the rms noise vs distance for a given image

    Args:
        image (2D or 3D ndarray): Noise image to get rms noise vs distance for
        resolution (float): Resolution of the image (in meters)

    Returns:
        bins, rms_noise_vs_dist (ndarray, ndarray): The 1D array of bins,
            and the array of rms noise at each distance bin
            If image is 3D, the rms noise is return for each image.
            `bins` will always be 1D, even if passing multiple images.
    """
    from scipy import ndimage

    if image.ndim > 2:
        out = [rms_noise_vs_dist(s, resolution) for s in image]
        bins_arr, rms_noise_arr = zip(*out)
        return bins_arr[0], np.array(rms_noise_arr)

    if crop:
        image = crop_square_data(image)

    h, w = image.shape[-2:]
    hc, wc = h // 2, w // 2
    image_centered = image - image[hc, wc]
    # Only extend out to shortest dimension
    # this will miss power contributions in 'corners' r > min(hc, wc)
    num_r = min(hc, wc)

    # create an array of integer radial distances from the center
    Y, X = np.ogrid[0:h, 0:w]
    # These act as "labels" in the ndimage summation
    r = np.hypot(X - wc, Y - hc).round().astype(int)
    # Ignore nans:
    nan_idxs = np.isnan(image_centered)
    r[nan_idxs] = -9999

    # average all image pixels with label 'r' for 0<=r<=wc
    noise_squared = ndimage.mean(image_centered**2, r, index=np.arange(0, num_r))
    rms_noise = np.sqrt(noise_squared)

    bins = r[hc, wc:] * resolution  # for the actual radial distances in image
    assert len(bins) == len(rms_noise)
    return bins, rms_noise
