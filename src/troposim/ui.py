import logging
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import rasterio
import rasterio as rio
import rasterio.windows
from jax import random
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from ._blocks import iter_blocks
from ._types import Bbox, PathOrStr

SENTINEL_WAVELENGTH = 0.055465763  # meters
METERS_TO_PHASE = 4 * 3.14159 / SENTINEL_WAVELENGTH
SEASONS = []
HDF5_KWARGS = {"chunks": True, "compression": "gzip"}
BLOCK_SHAPE = (256, 256)

logger = logging.getLogger("troposim")


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
    max_turbulence_amplitude: float = 5
    include_deformation: bool = True
    max_defo_amplitude: float = 5
    include_ramps: bool = True
    max_ramp_amplitude: float = 1.0
    include_stratified: bool = False


def create_simulation_data(
    inps: SimulationInputs, seed: int = 0, verbose: bool = False
):
    """Create realistic SLC simulated data.

    This function generates the necessary data layers for a TropoSim simulation, including
    turbulence, deformation, and phase ramps. It loads the global coherence model
    coefficients and uses them to simulate correlated noise for each pixel in the
    simulation domain.

    Parameters
    ----------
    inps : SimulationInputs
        Input parameters for the simulation, including the bounding box, resolution,
        number of dates, and flags for including various components.
    seed : int, optional
        Seed for the random number generator, by default 0.
    verbose : bool, optional
        Whether to print additional progress information, by default False.

    Returns
    -------
    np.ndarray
        The simulated noisy stack of SLC data.

    """
    from troposim import covariance, global_coherence

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler())

    # Create the times vector
    time = [
        inps.start_date + idx * timedelta(days=inps.dt) for idx in range(inps.num_dates)
    ]
    x_arr = np.array([(t - time[0]).days for t in time])

    outdir = inps.output_dir
    layers_dir = outdir / "input_layers"
    output_slc_dir = outdir / "slcs"
    layers_dir.mkdir(exist_ok=True, parents=True)
    output_slc_dir.mkdir(exist_ok=True, parents=True)

    # The global coherence is at 90 meters:
    upsample_y = int(round(90 / inps.res_y))
    upsample_x = int(round(90 / inps.res_x))
    upsample = (upsample_y, upsample_x)
    logger.info(f"Upsampling by {upsample}")
    # rasters, profiles = get_global_coherence(
    #     inps.bounding_box, outdir=outdir, upsample=(upsample_y, upsample_x)
    # )
    logger.info("Getting Rhos, Tau rasters")
    # (amps, rhos, taus, seasonal_A, seasonal_B, seasonal_mask, profile) = (
    # (
    #     amp_file,
    #     rho_file,
    #     tau_file,
    #     seasonal_A_file,
    #     seasonal_B_file,
    #     seasonal_mask_file,
    # ) = global_coherence.get_coherence_model_coeffs(
    coherence_files = global_coherence.get_coherence_model_coeffs(
        bounds=inps.bounding_box,
        upsample=upsample,
        output_dir=layers_dir,
    )
    with rio.open(coherence_files[0]) as src:
        shape2d = src.shape
        profile = src.profile

    # assert rhos.ndim == 2
    # assert (
    #     rhos.shape
    #     == taus.shape
    #     == seasonal_A.shape
    #     == seasonal_B.shape
    #     == seasonal_mask.shape
    # )
    # shape2d = rhos.shape[0], rhos.shape[1]

    shape3d = (inps.num_dates, shape2d[0], shape2d[1])
    logger.info(f"{profile=}")
    logger.info(f"{shape3d = }")
    # propagation_phase = np.zeros(shape3d, dtype="float32")
    files = {}
    # signal = np.zeros_like(shape2d, dtype="float32") # deformation only
    if inps.include_turbulence:
        logger.info("Generating turbulence")
        files["turbulence"] = layers_dir / "turbulence.h5"
        create_turbulence(
            shape2d=shape2d, num_days=inps.num_dates, out_hdf5=files["turbulence"]
        )
        # propagation_phase += turb_stack
        # signal += turb_stack
        # Save:
        # prof = profile.copy()
    if inps.include_deformation:
        logger.info("Generating deformation")
        files["deformation"] = layers_dir / "deformation.h5"
        create_defo_stack(
            shape=shape3d,
            sigma=shape2d[0] / 5,
            max_amplitude=inps.max_defo_amplitude,
            out_hdf5=files["deformation"],
        )
        # propagation_phase += defo_stack
    if inps.include_ramps:
        logger.info("Generating ramps")
        files["phase_ramps"] = layers_dir / "phase_ramps.h5"
        create_ramps(
            shape2d=shape2d,
            num_days=inps.num_dates,
            out_hdf5=files["phase_ramps"],
        )

    # Setup output tif files
    output_slc_filenames = [
        output_slc_dir / f"{date.strftime('%Y%m%d')}.slc.tif" for date in time
    ]
    slc_profile = profile.copy() | {"dtype": "complex64", "compression": "lzw"}
    for filename in output_slc_filenames:
        with rio.open(filename, "w", **slc_profile) as dst:
            pass

    b_iter = list(
        iter_blocks(
            arr_shape=shape2d,
            block_shape=BLOCK_SHAPE,
        )
    )
    key = random.key(seed)
    # # If we want to write the SLCs to an HDF5 stack:
    # with h5py.File("noisy_stack.h5", "w") as hf_out:
    #     dset_out = hf_out.create_dataset(
    #         "data", shape=shape3d, dtype="complex64", **HDF5_KWARGS
    #     )
    logger.info("Creating coherence matrices for each pixel")
    for rows, cols in tqdm(b_iter):
        amps, rhos, taus, seasonal_A, seasonal_B, seasonal_mask = load_coherence_files(
            coherence_files, rows, cols
        )
        if verbose:
            tqdm.write(f"Simulating correlated noise for {rows}, {cols}")
        key, subkey = random.split(key)
        C_arrays = covariance.simulate_coh_stack(
            time=x_arr,
            gamma_inf=rhos,
            # the global coherence raster model assumes gamma0=1
            gamma0=0.99 * np.ones_like(rhos),
            Tau0=taus,
            seasonal_A=seasonal_A,
            seasonal_B=seasonal_B,
            seasonal_mask=seasonal_mask,
        )
        propagation_phase = load_current_phase(files, rows, cols)

        # logger.info(C_arrays.shape, propagation_phase.shape, amps[rows, cols].shape)
        # noisy_stack = covariance.make_noisy_samples(
        noisy_stack = covariance.make_noisy_samples_jax(
            subkey, C=C_arrays, defo_stack=propagation_phase, amplitudes=amps
        )
        # dset_out[:, rows, cols] = noisy_stack

        window = rasterio.windows.Window.from_slices(rows, cols)
        for filename, layer in zip(output_slc_filenames, noisy_stack):
            with rio.open(filename, "r+", **profile) as dst:
                dst.write(layer, 1, window=window)
    return noisy_stack


def load_coherence_files(
    coherence_files: list[Path], rows: slice, cols: slice
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the coherence rasters for a block of pixels.

    Parameters
    ----------
    coherence_files : list[Path]
        List of paths to the coherence rasters.
    rows : slice
        Row slice to extract.
    cols : slice
        Column slice to extract.

    """
    # amp_file, rho_file, tau_file, seasonal_A_file, seasonal_B_file, seasonal_mask_file
    assert len(coherence_files) == 6
    out_arrays = []
    for f in coherence_files:
        with rio.open(f) as src:
            data = src.read(
                1, window=rasterio.windows.Window.from_slices(rows, cols), masked=True
            )
            if data.dtype == np.uint8:
                data = data.filled(0).astype(bool)
            else:
                data = data.filled()
            out_arrays.append(data)
    return out_arrays
    # return rhos, taus, seasonal_A, seasonal_B, seasonal_mask


def load_current_phase(files: dict[str, Path], rows: slice, cols: slice) -> np.ndarray:
    """Load and sum the phase data from multiple HDF5 files for a row/column block.

    Parameters
    ----------
    files : dict[str, Path]
        Dictionary of file paths for different phase components.
    rows : slice
        Row slice to extract.
    cols : slice
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
                summed_phase += data

    if summed_phase is None:
        raise ValueError("No valid data found in the provided files.")

    return summed_phase


def create_ramps(
    shape2d: tuple[int, int],
    num_days: int,
    out_hdf5: PathOrStr,
    amplitude: float = 1,
    overwrite: bool = False,
):
    """Create a stack of ramp phase data for a given shape and number of days.

    Parameters
    ----------
    shape2d : tuple[int, int]
        The 2D shape of the ramp data.
    num_days : int
        The number of days to generate ramp data for.
    out_hdf5 : PathOrStr
        The output HDF5 file path to save the ramp data.
    amplitude : float, optional
        The maximum amplitude of the ramp, by default 1.
    overwrite : bool, optional
        Whether to overwrite the output file if it already exists, by default False.

    Returns
    -------
    None
        The ramp data is saved to the specified HDF5 file.

    """
    from .deformation import synthetic

    shape3d = (num_days, *shape2d)
    rotations = np.random.randint(0, 360, size=(num_days,))
    if Path(out_hdf5).exists() and not overwrite:
        logger.info(f"Not overwriting {out_hdf5}")
        return

    with h5py.File(out_hdf5, "w") as hf:
        dset = hf.create_dataset("data", shape=shape3d, dtype="float32", **HDF5_KWARGS)
        for idx, r in enumerate(rotations):
            ramp_phase = synthetic.ramp(
                shape=shape2d, amplitude=amplitude, rotate_degrees=r
            )
            dset.write_direct(ramp_phase, dest_sel=idx)


def create_stratified(dem, num_days, out_hdf5: PathOrStr, overwrite: bool = False):
    """Create a stack of stratified atmospheric noise for a given DEM.

    Parameters
    ----------
    dem : np.ndarray
        The digital elevation model (DEM) to use for generating the stratified phase.
    num_days : int
        The number of days to generate stratified phase data for.
    out_hdf5 : PathOrStr
        The output HDF5 file path to save the stratified phase data.
    overwrite : bool, optional
        Whether to overwrite the output file if it already exists, by default False.

    Returns
    -------
    None
        The stratified phase data is saved to the specified HDF5 file.

    """
    from . import stratified

    if Path(out_hdf5).exists() and not overwrite:
        logger.info(f"Not overwriting {out_hdf5}")
        return
    shape2d = dem.shape
    shape3d = (num_days, *shape2d)
    # stratified_kwargs = {"K_params": {"shape": (num_days,)}}
    # print(stratified_kwargs)
    with h5py.File(out_hdf5, "w") as hf:
        dset = hf.create_dataset("data", shape=shape3d, dtype="float32", **HDF5_KWARGS)
        for idx in range(num_days):
            n = stratified.simulate(dem)
            dset.write_direct(n, dest_sel=idx)


def create_turbulence(
    shape2d: tuple[int, int],
    num_days: int,
    out_hdf5: PathOrStr,
    overwrite: bool = False,
    max_amplitude: float = 1.0,
    resolution: float = 30,
):
    """Create a stack of turbulent atmospheric noise.

    Parameters
    ----------
    shape2d : tuple[int, int]
        The 2D shape (rows, cols) of the output data.
    num_days : int
        The number of days to generate turbulence data for.
    out_hdf5 : PathOrStr
        The output HDF5 file path to save the turbulence data.
    overwrite : bool, optional
        Whether to overwrite the output file if it already exists, by default False.
    max_amplitude : float, optional
        The maximum amplitude of the turbulence, by default 1.0.
    resolution : float, optional
        The resolution of the turbulence simulation, by default 30.

    Returns
    -------
    None
        The turbulence data is saved to the specified HDF5 file.

    """
    from . import turbulence

    if Path(out_hdf5).exists() and not overwrite:
        logger.info(f"Not overwriting {out_hdf5}")
        return
    shape3d = (num_days, *shape2d)
    max_amp_meters = max_amplitude / METERS_TO_PHASE
    with h5py.File(out_hdf5, "w") as hf:
        dset = hf.create_dataset("data", shape=shape3d, dtype="float32", **HDF5_KWARGS)
        for idx in range(num_days):
            turb_meters = turbulence.simulate(
                shape=shape2d, resolution=resolution, max_amp=max_amp_meters
            )
            dset.write_direct(turb_meters * METERS_TO_PHASE, dest_sel=idx)


def create_defo_stack(
    shape: tuple[int, int, int],
    sigma: float,
    max_amplitude: float = 1,
    out_hdf5: PathOrStr | None = None,
    overwrite: bool = False,
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
    out_hdf5 : PathOrStr | None, optional
        Path to output HDF5 file, by default None. If None, the deformation is not saved to disk.
    overwrite : bool, optional
        Whether to overwrite the output file if it already exists, by default False.

    Returns
    -------
    np.ndarray | None
        Deformation stack with time series, shape (num_time_steps, rows, cols).
        If `out_hdf5` is passed, None is returned, as the file has been saved to disk.

    """
    from .deformation.synthetic import gaussian

    if Path(out_hdf5).exists() and not overwrite:
        logger.info(f"Not overwriting {out_hdf5}")
        return

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


def fetch_dem(bounds: Bbox, output_dir: Path, upsample_factor: tuple[int, int]):
    """Download and stitch a DEM (Digital Elevation Model) for the given bounding box.

    Parameters
    ----------
    bounds : Bbox
        The bounding box to download the DEM for.
    output_dir : Path
        The directory to save the DEM file to.
    upsample_factor : tuple[int, int]
        The factors to upsample the DEM in the y and x dimensions.

    """
    from sardem import cop_dem

    cop_dem.download_and_stitch(
        output_name=output_dir / "dem.tif",
        bbox=bounds,
        yrate=upsample_factor[0],
        xrate=upsample_factor[1],
        output_format="GTiff",
        output_type="float32",
    )
