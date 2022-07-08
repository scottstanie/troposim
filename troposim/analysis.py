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
    density=True,
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
    density :
        (Default value = True)
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
        density=density,
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
