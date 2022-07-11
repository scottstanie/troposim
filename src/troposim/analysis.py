import os
import shutil

import numpy as np

from troposim import deramp, turbulence


def create_deramped_zarr(
    deramped_file,
    avg_slc_file,
    mask=None,
    ds="igrams",
    units="rad",
    scaling=1.0,
    overwrite=False,
):
    """Create a deramped zarr file from a list of average SLCs

    Parameters
    ----------
    deramped_file : str
        output file name to save deramped data to
    avg_slc_file : str
        path to the average SLC file created by trodi package
    mask : ndarray
        (Default value = None)
    ds : str
        (Default value = "igrams")
    units : str
        (Default value = "rad")
    scaling : float
        (Default value = 1.0)
    overwrite : bool
        (Default value = False)
    """
    try:
        import xarray as xr
        from numcodecs import Blosc
    except ImportError:
        print("xarray and numcodecs must be installed to use this function")
        raise

    if os.path.exists(deramped_file):
        print(f"{deramped_file} already exists")
        if not overwrite:
            return deramped_file
        else:
            print(f"Removing {deramped_file}")
            shutil.rmtree(deramped_file)

    avg_slcs = xr.open_dataset(avg_slc_file)
    outstack = deramp.remove_ramp(
        avg_slcs[ds].data, copy=True, deramp_order=2, mask=mask
    )
    if mask is not None:
        if mask.ndim > 2:
            outstack[mask] = np.nan
        else:
            outstack[:, mask] = np.nan

    avg_slcs[ds].data = outstack * scaling

    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    avg_slcs[ds].attrs["units"] = units
    print(f"saving to {deramped_file}")
    avg_slcs.to_zarr(deramped_file, encoding={ds: {"compressor": compressor}})


# Get all 1d PSD curves from the average ifgs in a region
def get_all_1d_psd_curves(
    avg_ifgs,
    resolution=180,
    min_date=None,
    max_date=None,
    freq0=1e-4,
    deg=3,
    save_dir=".",
    load=True,
):
    """Get all 1D radially averaged power spectrum curves averaged for a region

    Parameters
    ----------
    avg_ifgs : xr.DataArray
        average interferograms
    resolution : float
        spatial resolution of input data in meters (Default value = 180)
    min_date : datetime
        minimum date to include in the analysis (Default value = None)
    max_date : datetime
        maximum date to include in the analysis (Default value = None)
    save_dir : str
        directory to save the output to in .npz file (Default value = ".")
    freq0 :
        (Default value = 1e-4)
    deg :
        (Default value = 3)
    load :
        (Default value = True)
    """
    stack = avg_ifgs.sel(date=slice(min_date, max_date))
    # stack_data = stack.as_numpy()
    # stack_data.load()
    if load:
        stack.load()
    p0_arr, beta_arr, freq_arr, psd1d_arr = turbulence.get_psd(
        image=stack.data,
        resolution=resolution,
        freq0=freq0,
        deg=deg,
        crop=True,
    )
    beta_arr_coeffs = np.array([b.coef for b in beta_arr])
    beta_arr_mean = np.polynomial.Polynomial(np.mean(beta_arr_coeffs, axis=0))

    if save_dir:
        fname = f"psd1d_maxdate_{max_date}.npz" if max_date else "psd1d.npz"
        save_name = os.path.join(save_dir, fname)
        print(f"Saving to {save_name}")
        np.savez(
            save_name,
            p0_arr=p0_arr,
            beta_arr=beta_arr,
            beta_mean=beta_arr_mean,
            freq_arr=freq_arr,
            psd1d_arr=psd1d_arr,
        )
    return p0_arr, beta_arr, freq_arr, psd1d_arr



def get_psd_blocks(
    data, block_size=None, block_step=None, resolution=60.0, freq0=1e-4, deg=3
):
    """For one image, get radially averaged PSD from multiple blocks within image
    Crops into subsets, calls `get_psd` on each

    Parameters
    ----------
    data : 2D ndarray
        displacement in m.
    data : 2D ndarray
        displacement in m.
        resolution (float), spatial resolution of input data in meters
    block_size : float
        size of block side, in meters (Default value = None)
    block_step : float
        amount to shift the block window, in m (Default value = None)
    block_step : float
        amount to shift the block window, in m
        freq0 (float), reference spatial freqency in cycle / m. (Default value = None)
    resolution :
        (Default value = 60.0)
    freq0 :
        (Default value = 1e-4)
    deg :
        (Default value = 3)

    Returns
    -------


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
            p0_hat, beta_hat, freq, psd1d = turbulence.get_psd(
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


def fractal_dimension(beta):
    """Convert the beta slope to D, fractal dimension (D2 is from Hanssen Section 4.7"""
    return (7.0 - beta + 1.0) / 2.0


def debug(surf):
    return (
        f"min:{np.min(surf):.2g}, max:{np.max(surf):.2g}, "
        f"ptp:{np.ptp(surf):.2g}, mean:{np.mean(surf):.2g}"
    )


def average_psd_azimuth(
    psd2d=None, image=None, resolution=60.0, dTheta=30, r_min=0, r_max=np.inf
):
    """Get 1D power spectrum averaged by angular bin (azimuthal average)

    Parameters
    ----------
    psd2d : ndarray
        2D power spectral density, size (N, N)
        non-square images sized (r, c) will use N = min(r, c) (Default value = None)
    image : ndarray, optional
        if psd2d=None, pass original image to transform
        (Default value = None)
    dTheta : float
        spacing between angle bins, in degrees. default=30.
    r_min : float
        optional, to limit the radius where the averaging takes place,
        pass a minimum radius (in pixels) (Default value = 0)
    r_max : float
        optional, maximum radius (in pixels) where the averaging takes place
        (Default value = np.inf)
    resolution :
        (Default value = 60.0)

    Returns
    -------
    angles : ndarray
        angles in degrees from 0 to 180
    psd1d : ndarray
        1D power spectral density, size same as angles
    """
    from scipy import ndimage
    if psd2d is None:
        if image is None:
            raise ValueError("need either psd, or pre-transformed image")
        psd2d = turbulence.get_psd2d(image, resolution=resolution)
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
