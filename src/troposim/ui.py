import concurrent.futures
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import rasterio
import rasterio as rio
from pydantic import BaseModel, Field

from ._types import Bbox, PathOrStr

SENTINEL_WAVELENGTH = 0.055465763  # meters
METERS_TO_PHASE = 4 * 3.14159 / SENTINEL_WAVELENGTH
SEASONS = []
HDF5_KWARGS = {"chunks": True, "compression": "lzf"}


class SimulationInputs(BaseModel):
    output_dir: Path = Path()
    start_date: datetime
    dt: int = Field(..., ge=1, le=365, description="Time step [days]")
    num_dates: int = Field(..., ge=2, le=100)
    res_y: float = Field(..., ge=1, le=1000, description="Y resolution [meters]")
    res_x: float = Field(..., ge=1, le=1000, description="X resolution [meters]")
    bounding_box: Bbox = Field(
        ..., description="(left, bottom, right, top) in EPSG:4326"
    )
    include_turbulence: bool = True
    include_deformation: bool = True
    include_ramps: bool = True
    # include_stratified: bool = True
    max_defo_amplitude: float = 5


def create_simulation_data(inps: SimulationInputs):
    from . import global_coherence
    from . import covariance

    # from .deformation import synthetic
    # Create the times vector
    time = [inps.start_date + timedelta(days=inps.dt) for i in range(inps.num_dates)]
    x_arr = np.array([(t - time[0]).days for t in time])

    outdir = inps.output_dir
    # The global coherence is at 90 meters:
    upsample_y = int(round(90 / inps.res_y))
    upsample_x = int(round(90 / inps.res_x))
    upsample = (upsample_y, upsample_x)
    print(f"Upsampling by {upsample}")
    # rasters, profiles = get_global_coherence(
    #     inps.bounding_box, outdir=outdir, upsample=(upsample_y, upsample_x)
    # )
    print("Getting Rhos, Tau rasters")
    (rhos, taus, seasonal_A, seasonal_B, seasonal_mask, profile) = (
        global_coherence.get_coherence_model_coeffs(
            bounds=inps.bounding_box,
            upsample=upsample,
            output_dir=outdir,
        )
    )

    print("Getting amplitudes")
    amplitudes, amp_prof = global_coherence.fetch_amplitudes(
        bounds=inps.bounding_box,
        output_dir=outdir,
        upsample=upsample,
    )
    assert rhos.ndim == 2
    assert (
        rhos.shape
        == taus.shape
        == seasonal_A.shape
        == seasonal_B.shape
        == seasonal_mask.shape
    )
    shape2d = rhos.shape[0], rhos.shape[1]
    shape3d = (inps.num_dates, shape2d[0], shape2d[1])
    print(f"{shape3d = }")
    propagation_phase = np.zeros(shape3d, dtype="float32")
    files = {}
    # signal = np.zeros_like(shape2d, dtype="float32") # deformation only
    if inps.include_turbulence:
        print("Generating turbulence")
        files["turbulence"] = outdir / "turbulence.h5"
        turb_stack = create_turbulence(
            shape2d=shape2d, num_days=inps.num_dates, out_hdf5=files["turbulence"]
        )
        propagation_phase += turb_stack
        # signal += turb_stack
        # Save:
        # prof = profile.copy()
    if inps.include_deformation:
        print("Generating deformation")
        files["deformation"] = outdir / "deformation.h5"
        create_defo_stack(
            shape=shape3d,
            sigma=shape2d[0] / 5,
            max_amplitude=inps.max_defo_amplitude,
            out_hdf5=files["deformation"],
        )
        # propagation_phase += defo_stack
    if inps.include_ramp:
        print("Generating ramps")
        files["phase_ramps"] = outdir / "phase_ramps.h5"
        create_ramps(
            shape2d=shape2d,
            num_days=inps.num_dates,
            out_hdf5=files["phase_ramps"],
        )

    print("Creating coherence matrices for each pixel")
    C_arrays = covariance.simulate_coh_stack(
        time=x_arr,
        gamma_inf=rhos,
        gamma0=np.ones_like(rhos),  # the global coherence raster model assumes gamma0=1
        Tau0=taus,
        seasonal_A=seasonal_A,
        seasonal_B=seasonal_B,
        seasonal_mask=seasonal_mask,
    )
    print("Simulating correlated noise")
    noisy_stack = covariance.make_noisy_samples(
        C=C_arrays, defo_stack=propagation_phase, amplitudes=amplitudes
    )
    for date, layer in zip(time, noisy_stack):
        filename = outdir / f"{date.strftime('%Y%m%d')}.slc.tif"

        with rio.open(filename, "w", **profile) as dst:
            dst.write(layer, 1)
    return noisy_stack


def create_ramps(
    shape2d: tuple[int, int], num_days: int, out_hdf5: PathOrStr, amplitude: float = 1
):
    from .deformation import synthetic

    rotations = [np.random.randint(0, 360, size=(num_days,))]
    synthetic.ramp(shape=shape2d, amplitude=amplitude, rotate_degrees=r)


def create_stratified(dem, num_days, out_hdf5: PathOrStr):
    from . import stratified

    stratified_kwargs = {"K_params": {"shape": (num_days,)}}
    print(stratified_kwargs)
    return stratified.simulate(dem, **stratified_kwargs)


def create_turbulence(
    shape2d: tuple[int, int],
    num_days: int,
    out_hdf5: PathOrStr,
    resolution: float = 30,
):
    from . import turbulence

    shape3d = (num_days, *shape2d)
    with h5py.File(out_hdf5, "w") as hf:
        dset = hf.create_dataset("data", shape=shape3d, dtype="float32", **HDF5_KWARGS)
        for idx in range(num_days):
            turb_meters = turbulence.simulate(shape=shape2d, resolution=resolution)
            dset.write_direct(turb_meters * METERS_TO_PHASE, dest_sel=idx)


def create_defo_stack(
    shape: tuple[int, int, int],
    sigma: float,
    max_amplitude: float = 1,
    out_hdf5: PathOrStr | None = None,
) -> np.ndarray | None:
    """Create the time series of deformation to add to each SAR date.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Shape of the deformation stack (num_time_steps, rows, cols).
    sigma : float
        Standard deviation of the Gaussian deformation.
    max_amplitude : float, optional
        Maximum amplitude of the final deformation. Defaults to 1.

    Returns
    -------
    np.ndarray | None
        Deformation stack with time series, shape (num_time_steps, rows, cols).
        If `out_hdf5` is passed, None is returned, as the file has been saved to disk.

    """
    from .deformation.synthetic import gaussian

    num_time_steps, rows, cols = shape
    shape2d = (rows, cols)
    # Get shape of deformation in final form (normalized to 1 max)
    final_defo = gaussian(shape=shape2d, sigma=sigma).reshape((1, *shape2d))
    final_defo *= max_amplitude / np.max(final_defo)
    # Broadcast this shape with linear evolution
    time_evolution = np.linspace(0, 1, num=num_time_steps)

    with h5py.File(out_hdf5) as hf:
        dset = hf.create_dataset("data", shape=shape, dtype="float32", **HDF5_KWARGS)
        for idx, t in enumerate(time_evolution):
            dset.write_direct(final_defo * t, dest_sel=idx)


def fetch_dem(bounds: Bbox):
    from sardem import cop_dem

    cop_dem.download_and_stitch(
        output_name="dem.tif",
        bbox=bounds,
        xrate=1,
        yrate=1,
        output_format="GTiff",
        output_type="float32",
    )


def main(infile, outfile, num_workers=4):
    """Process infile block-by-block and write to a new file

    The output is the same as the input, but with band order
    reversed.
    """

    with rasterio.open(infile) as src:
        # Create a destination dataset based on source params. The
        # destination will be tiled, and we'll process the tiles
        # concurrently.
        profile = src.profile
        profile.update(blockxsize=128, blockysize=128, tiled=True)

        with rasterio.open(outfile, "w", **src.profile) as dst:
            windows = [window for ij, window in dst.block_windows()]

            # We cannot write to the same file from multiple threads
            # without causing race conditions. To safely read/write
            # from multiple threads, we use a lock to protect the
            # DatasetReader/Writer
            read_lock = threading.Lock()
            write_lock = threading.Lock()

            def process(window):
                with read_lock:
                    src_array = src.read(window=window)

                # The computation can be performed concurrently
                # result = compute(src_array)

                with write_lock:
                    dst.write(result, window=window)

            # We map the process() function over the list of
            # windows.
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_workers
            ) as executor:
                executor.map(process, windows)


# def get_global_coherence(
#     bounds: Bbox, outdir=Path("."), upsample: tuple[int, int] = (1, 1)
# ):
#     from .global_coherence import get_rasters
#     import rasterio as rio

#     rasters = {}
#     profiles = {}
#     # TODO: need defaultdict
#     for variable in ["rho", "tau"]:
#         for season in ["fall", "winter", "spring", "summer"]:
#             outname = outdir / f"coherence_{variable}_{season}.tif"
#             if outname.exists():
#                 with rio.open(outname) as src:
#                     X = src.read(1)
#                     p = src.profile.copy()
#             else:
#                 X, p = get_rasters(
#                     bounds=bounds, season=season, upsample_factors=upsample
#                 )
#             rasters[season] = X
#             profiles[season] = p
#     return rasters, profiles
