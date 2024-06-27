import concurrent.futures
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import rasterio
import rasterio as rio
from pydantic import BaseModel, Field

from ._blocks import iter_blocks
from ._types import Bbox, PathOrStr

SENTINEL_WAVELENGTH = 0.055465763  # meters
METERS_TO_PHASE = 4 * 3.14159 / SENTINEL_WAVELENGTH
SEASONS = []
HDF5_KWARGS = {"chunks": True, "compression": "lzf"}
BLOCK_SHAPE = (256, 256)

logger = logging.getLogger(__name__)


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
    time = [
        inps.start_date + idx * timedelta(days=inps.dt) for idx in range(inps.num_dates)
    ]
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
    (amps, rhos, taus, seasonal_A, seasonal_B, seasonal_mask, profile) = (
        global_coherence.get_coherence_model_coeffs(
            bounds=inps.bounding_box,
            upsample=upsample,
            output_dir=outdir,
        )
    )

    # print("Getting amplitudes")
    # amplitudes, amp_prof = global_coherence.fetch_amplitudes(
    #     bounds=inps.bounding_box,
    #     output_dir=outdir,
    #     upsample=upsample,
    # )
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
    # propagation_phase = np.zeros(shape3d, dtype="float32")
    files = {}
    # signal = np.zeros_like(shape2d, dtype="float32") # deformation only
    if inps.include_turbulence:
        print("Generating turbulence")
        files["turbulence"] = outdir / "turbulence.h5"
        create_turbulence(
            shape2d=shape2d, num_days=inps.num_dates, out_hdf5=files["turbulence"]
        )
        # propagation_phase += turb_stack
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
    if inps.include_ramps:
        print("Generating ramps")
        files["phase_ramps"] = outdir / "phase_ramps.h5"
        create_ramps(
            shape2d=shape2d,
            num_days=inps.num_dates,
            out_hdf5=files["phase_ramps"],
        )

    print("Creating coherence matrices for each pixel")
    b_iter = list(
        iter_blocks(
            arr_shape=shape2d,
            block_shape=BLOCK_SHAPE,
        )
    )
    # with h5py.File("C_arrays.h5", "w") as hf_in, h5py.File("noisy_stack.h5", "w") as hf_out:
    #     dset = hf_in.create_dataset(
    #         "data", shape=shape3d, dtype="complex64", **HDF5_KWARGS
    #     )
    #     dset = hf_out.create_dataset(
    #         "data", shape=shape3d, dtype="complex64", **HDF5_KWARGS
    #     )
    with h5py.File("noisy_stack.h5", "w") as hf_out:
        dset_out = hf_out.create_dataset(
            "data", shape=shape3d, dtype="complex64", **HDF5_KWARGS
        )
        for rows, cols in b_iter:
            print(f"Simulating correlated noise for {rows}, {cols}")
            C_arrays = covariance.simulate_coh_stack(
                time=x_arr,
                gamma_inf=rhos[rows, cols],
                # the global coherence raster model assumes gamma0=1
                gamma0=0.95 * np.ones_like(rhos[rows, cols]),
                Tau0=taus[rows, cols],
                seasonal_A=seasonal_A[rows, cols],
                seasonal_B=seasonal_B[rows, cols],
                seasonal_mask=seasonal_mask[rows, cols],
            )
            propagation_phase = load_current_phase(files, rows, cols)

            print(C_arrays.shape, propagation_phase.shape, amps[rows, cols].shape)
            noisy_stack = covariance.make_noisy_samples(
                C=C_arrays, defo_stack=propagation_phase, amplitudes=amps[rows, cols]
            )
            # dset_out.write_direct(noisy_stack, dest_sel=np.s_[:, rows, cols])
            dset_out[:, rows, cols] = noisy_stack

            # for date, layer in zip(time, noisy_stack):
            #     filename = outdir / f"{date.strftime('%Y%m%d')}.slc.tif"

            #     with rio.open(filename, "w", **profile) as dst:
            #         dst.write(layer, 1)
    return noisy_stack


def load_current_phase(files: dict[str, Path], rows: slice, cols: slice) -> np.ndarray:
    """Load and sum the phase data from multiple HDF5 files for a row/column block.

    Parameters
    ----------
    files (Dict[str, Path])
        Dictionary of file paths for different phase components.
    rows (slice)
        Row slice to extract.
    cols (slice)
        Column slice to extract.

    Returns
    -------
    np.ndarray: 3D array representing the summed phase data for the specified block.

    """
    summed_phase = None

    for component, file_path in files.items():
        logger.debug(f"Loading {component}")
        with h5py.File(file_path, "r") as f:
            # Assume the main dataset is named 'data'. Adjust if necessary.
            dset: h5py.Dataset = f["data"]

            # Check if the dset is 3D
            if dset.ndim == 3:
                # For 3D datasets, load the full depth
                data = dset[:, rows, cols]
            elif dset.ndim == 2:
                # For 2D datasets, add a depth dimension of 1
                data = dset[rows, cols][np.newaxis, :, :]
            else:
                raise ValueError(f"Unexpected dset shape in {file_path}: {dset.shape}")

            if summed_phase is None:
                summed_phase = data
            else:
                # Ensure the shapes match before adding
                if summed_phase.shape != data.shape:
                    raise ValueError(
                        f"Shape mismatch: {summed_phase.shape} vs {data.shape}"
                    )
                summed_phase += data

    if summed_phase is None:
        raise ValueError("No valid data found in the provided files.")

    return summed_phase


def create_ramps(
    shape2d: tuple[int, int], num_days: int, out_hdf5: PathOrStr, amplitude: float = 1
):
    from .deformation import synthetic

    shape3d = (num_days, *shape2d)
    rotations = np.random.randint(0, 360, size=(num_days,))
    with h5py.File(out_hdf5, "w") as hf:
        dset = hf.create_dataset("data", shape=shape3d, dtype="float32", **HDF5_KWARGS)
        for idx, r in enumerate(rotations):
            ramp_phase = synthetic.ramp(
                shape=shape2d, amplitude=amplitude, rotate_degrees=r
            )
            dset.write_direct(ramp_phase, dest_sel=idx)


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

    with h5py.File(out_hdf5, "w") as hf:
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
