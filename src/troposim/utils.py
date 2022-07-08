import numpy as np

DATE_FMT = "%Y%m%d"


def crop_square_data(data, N=None):
    """Grab the largest square portion of size N from input 2D input

    Searches for square with greatest number of non-zero values.

    Parameters
    ----------
    data : 2D ndarray
        input image

    N : int optional
        Size of square input.
        If not provided, uses N = min(data.shape)
        Uses the integral image (summed area table) of data to quickly
        calculate number of zeros in a square subset:
        https://en.wikipedia.org/wiki/Summed-area_table (Default value = None)

    Returns
    -------
    ndarray
        square subset of data, size = (N, N)

    """
    row_slice, col_slice = find_largest_square(data, N=N)
    return data[row_slice, col_slice]


def find_largest_square(data, N=None):
    """Get the row and column slices of the largest square in a 2D array

    Parameters
    ----------
    data : 2D ndarray

    N : int
        size of square to search for (Default value = min(data.shape))

    Returns
    -------
    tuple[slice, slice]

    """
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
    r"""Integral image / summed area table.

    The integral image contains the sum of all elements above and to the
    left of it, i.e.:

    .. math::

       S[m, n] = \sum_{i \leq m} \sum_{j \leq n} X[i, j]

    Parameters
    ----------
    image : ndarray
        Input image.

    Returns
    -------
    S : ndarray
        Summed area table, same size as `image`

    Source
    ------
    skimage.transform module

    References
    ----------
    .. [1] F.C. Crow, "Summed-area tables for texture mapping,"
           ACM SIGGRAPH Computer Graphics, vol. 18, 1984, pp. 207-212.
    """
    S = image.copy()
    for i in range(image.ndim):
        S = S.cumsum(axis=i)
    return S


def extrema(x):
    """Get the (min, max) of an array"""
    return np.min(x), np.max(x)


def find_valid(
    sar_date_list,
    ifg_date_list,
    min_date=None,
    max_date=None,
    max_temporal_baseline=900,
    verbose=True,
):
    """Cut down the full list of interferograms and geo dates

    - Cuts date ranges (e.g. piecewise linear solution) with  `min_date` and `max_date`
    - Can ignore whole dates by reading `ignore_geo_file`
    - Prunes long time baseline interferograms with `max_temporal_baseline`

    Parameters
    ----------
    sar_date_list : iterable of datetime

    ifg_date_list : iterable of (datetime, datetime)

    min_date : datetime
         (Default value = None)
    max_date : datetime
         (Default value = None)
    max_temporal_baseline : int
         (Default value = 900)
    verbose : bool
         (Default value = True)

    Returns
    -------
    valid

    """
    if verbose:
        ig1 = len(ifg_date_list)  # For logging purposes, what do we start with

    valid_slcs = sar_date_list
    valid_ifgs = ifg_date_list

    # Remove slcs and ifgs outside of min/max range
    if min_date is not None:
        if verbose:
            print(f"Keeping data after min_date: {min_date}")
        valid_slcs = [g for g in valid_slcs if g > min_date]
        valid_ifgs = [
            ig for ig in valid_ifgs if (ig[0] > min_date and ig[1] > min_date)
        ]

    if max_date is not None:
        if verbose:
            print(f"Keeping data only before max_date: {max_date}")
        valid_slcs = [g for g in valid_slcs if g < max_date]
        valid_ifgs = [
            ig for ig in valid_ifgs if (ig[0] < max_date and ig[1] < max_date)
        ]

    if verbose:
        # This is just for logging purposes:
        too_long_ifgs = [
            ig for ig in valid_ifgs if temporal_baseline(ig) > max_temporal_baseline
        ]
        print(
            f"Ignoring {len(too_long_ifgs)} ifgs with longer baseline "
            f"than {max_temporal_baseline} days"
        )

    # ## Remove long time baseline ifgs ###
    valid_ifgs = [
        ig for ig in valid_ifgs if temporal_baseline(ig) <= max_temporal_baseline
    ]

    if verbose:
        print(f"Ignoring {ig1 - len(valid_ifgs)} ifgs total")

    # Now go back to original full list and see what the indices were
    # used for subselecting from the unw_stack by a pixel
    valid_idxs = np.searchsorted(
        intlist_to_string(ifg_date_list),  # Numpy searchsorted can't use datetimes
        intlist_to_string(valid_ifgs),
    )
    return valid_slcs, valid_ifgs, valid_idxs


def temporal_baseline(ifg):
    """Get the temporal baseline (int days) of an interferogram

    Parameters
    ----------
    ifg : tuple[datetime, datetime]

    Returns
    -------
    T : int

    """
    return (ifg[1] - ifg[0]).days


def intlist_to_string(ifg_date_list, ext=".int"):
    """Convert date pairs to list of string filenames

    Parameters
    ----------
    ifg_date_list : iteratble of (datetime, datetime)

    ext : str
        Filename extension to strip (Default value = ".int")

    Returns
    -------

    """
    return [
        "{}_{}{ext}".format(a.strftime(DATE_FMT), b.strftime(DATE_FMT), ext=ext)
        for a, b in ifg_date_list
    ]


def anisotropy_var(azi_psd1d):
    return np.var(azi_psd1d)


def anisotropy_A(azi_psd1d):
    return 2 * np.ptp(azi_psd1d)


def rms_noise_vs_dist(image, resolution, crop=True):
    """Get the rms noise vs distance for a given image

    Parameters
    ----------
    image : 2D or 3D ndarray
        Noise image to get rms noise vs distance for
    resolution : float
        Resolution of the image (in meters)
    crop :
         (Default value = True)

    Returns
    -------
    bins : ndarray
        The 1D array of bins.
        `bins` will always be 1D, even if passing multiple images.
    rms_noise_vs_dist : ndarray
        The array of rms noise at each distance bin
        If image is 3D, the rms noise is return for each image.
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


def take_looks(arr, row_looks, col_looks, separate_complex=False, **kwargs):
    """Downsample a numpy matrix by summing blocks of (row_looks, col_looks)

    Cuts off values if the size isn't divisible by num looks

    Parameters
    ----------
    arr : ndarray
        2D array of an image
    row_looks : int
        the reduction rate in row direction
    col_looks : int
        the reduction rate in col direction
    separate_complex : bool
        take looks on magnitude and phase separately
        Better to preserve the look of the magnitude (Default value = False)


    Returns
    -------
        ndarray, size = ceil(rows / row_looks, cols / col_looks)

    Notes
    -----
    For complex data, looks on the magnitude are done separately
    from looks on the phase

    """
    if row_looks == 1 and col_looks == 1:
        return arr
    if arr.ndim == 3:
        return np.stack(
            [
                take_looks(
                    a, row_looks, col_looks, separate_complex=separate_complex, **kwargs
                )
                for a in arr
            ]
        )
    if np.iscomplexobj(arr) and separate_complex:
        mag_looked = take_looks(np.abs(arr), row_looks, col_looks)
        phase_looked = take_looks(np.angle(arr), row_looks, col_looks)
        return mag_looked * np.exp(1j * phase_looked)

    rows, cols = arr.shape
    new_rows = rows // row_looks
    new_cols = cols // col_looks

    row_cutoff = rows % row_looks
    col_cutoff = cols % col_looks

    if row_cutoff != 0:
        arr = arr[:-row_cutoff, :]
    if col_cutoff != 0:
        arr = arr[:, :-col_cutoff]
    # For taking the mean, treat integers as floats
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype("float")

    return np.mean(
        np.reshape(arr, (new_rows, row_looks, new_cols, col_looks)), axis=(3, 1)
    )


def block_iterator(arr_shape, block_shape, overlaps=(0, 0), start_offsets=(0, 0)):
    """Iterator to get indexes for accessing blocks of a raster

    Parameters
    ----------
    arr_shape :
        (num_rows, num_cols), full size of array to access
    block_shape :
        (height, width), size of accessing blocks
    overlaps : tuple[int, int]
        (row_overlap, col_overlap), number of pixels to re-include
        after sliding the block (default (0, 0))
        
    start_offsets :
         (Default value = (0)

    Yields
    -------
    Iterator of ((row_start, row_stop), (col_start, col_stop))

    Examples
    --------
    >>> list(block_iterator((180, 250), (100, 100)))
    [((0, 100), (0, 100)), ((0, 100), (100, 200)), ((0, 100), (200, 250)), \
((100, 180), (0, 100)), ((100, 180), (100, 200)), ((100, 180), (200, 250))]
    >>> list(block_iterator((180, 250), (100, 100), overlaps=(10, 10)))
    [((0, 100), (0, 100)), ((0, 100), (90, 190)), ((0, 100), (180, 250)), \
((90, 180), (0, 100)), ((90, 180), (90, 190)), ((90, 180), (180, 250))]

    """
    rows, cols = arr_shape
    row_off, col_off = start_offsets
    row_overlap, col_overlap = overlaps
    height, width = block_shape

    if height is None:
        height = rows
    if width is None:
        width = cols

    # Check we're not moving backwards with the overlap:
    if row_overlap >= height:
        raise ValueError(f"{row_overlap = } must be less than {height = }")
    if col_overlap >= width:
        raise ValueError(f"{col_overlap = } must be less than {width = }")
    while row_off < rows:
        while col_off < cols:
            row_end = min(row_off + height, rows)  # Dont yield something OOB
            col_end = min(col_off + width, cols)
            yield ((row_off, row_end), (col_off, col_end))

            col_off += width
            if col_off < cols:  # dont bring back if already at edge
                col_off -= col_overlap

        row_off += height
        if row_off < rows:
            row_off -= row_overlap
        col_off = 0
