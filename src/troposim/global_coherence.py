import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import rasterio
from affine import Affine
from numpy.typing import ArrayLike
from rasterio import windows

from ._types import Bbox, PathOrStr

logger = logging.getLogger(__name__)

COHERENCE_GPKG_ZIP = Path(__file__).parent / "data/global_coherence_layers.gti.gpkg.zip"
COHERENCE_GDAL_PATH = f"/vsizip/{COHERENCE_GPKG_ZIP}/global_coherence_layers.gti.gpkg"
COHERENCE_GPKG_TEMPLATE = "/vsizip/" + str(
    Path(__file__).parent
    / "data/coherence_{season}_{variable}.gti.gpkg.zip/coherence_{season}_{variable}.gti.gpkg"
)


class Season(Enum):
    """Seasonal periods for the year."""

    WINTER = "winter"
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"


class Variable(Enum):
    """Variables for the global coherence raster."""

    AMP = "amp"  # note: capitalized in the dataset
    TAU = "tau"
    RHO = "rho"
    RMSE = "rmse"


def convert_to_float(X: np.ndarray, variable: Variable) -> np.ndarray:
    """Convert the byte/int DNs to floats.

    Notes
    -----
    Backscatter amplitudes (AMP.tif files):
        γ0[dB]=20∗log10(DN)−83
        γ0[power]=DN2/199526231

        No data value 0
        DN stored as unsigned 16 bit integers


    Coherence (COH06,...,COH48.tif files):
        γ=DN/100

        No data value 0
        DN stored as unsigned 8 bit integers

    rho, tau, rmse (rho.tif,tau.tif,rmse.tif files):
        ρ|τ|rmse=DN/1_000

        No data value 0
        DN stored as unsigned 16 bit integers

    References
    ----------
    http://sentinel-1-global-coherence-earthbigdata.s3-website-us-west-2.amazonaws.com/

    """
    if variable == Variable.AMP:
        data = X.astype(float) ** 2 / 199526231
    else:
        data = X.astype(float) / 1000.0
    return data.astype("float32")


def get_rasters(
    bounds: Bbox,
    variable: Variable | str,
    season: Season | str,
    shape: tuple[int, int] | None = None,
    upsample_factors: tuple[int, int] | None = None,
    convert_data: bool = True,
    interp_method: str = "linear",
    outfile: PathOrStr | None = None,
):
    """Retrieve Sentinel-1 global coherence dataset rasters.

    The Sentinel-1 global coherence dataset is documented at
    http://sentinel-1-global-coherence-earthbigdata.s3-website-us-west-2.amazonaws.com/

    Parameters
    ----------
    bounds : Bbox
        Bounding box for the region of interest.
    variable : Variable | str
        The variable to retrieve, e.g. "rho".
        Options are "AMP", "tau", "rho", "rmse".
    season : Season | str
        The season to retrieve, e.g. "winter".
        Options are "winter", "spring", "summer", "fall".
    shape : tuple[int, int] | None, optional
        The desired output shape of the raster. If provided, the raster will be
        interpolated to this shape.
    upsample_factors : tuple[int, int] | None, optional
        Alternative to `shape`: The upsampling factors for the x and y axes.
    convert_data : bool, optional
        If True, the data will be converted to the appropriate units.
        AMP rasters will be converted to power
        rho, tau, rmse are converted to floats.
        See `convert_to_float` for details.
    interp_method : str, optional
        The interpolation method to use, by default "linear".
    outfile : str | None, optional
        If provided, the raster data will be saved to this file path.

    Returns
    -------
    np.ndarray
        The raster data.
    rasterio.DatasetReader
        The raster profile.

    References
    ----------
    Kellndorfer, J., Cartus, O., Lavalle, M. et al. Global seasonal Sentinel-1
    interferometric coherence and backscatter data set. Sci Data 9, 73 (2022).
    https://doi.org/10.1038/s41597-022-01189-6

    """
    if outfile and Path(outfile).exists():
        logger.info("Reading from %s", outfile)
        with rasterio.open(outfile) as src:
            return src.read(1), src.profile

    variable = Variable(variable)
    season = Season(season)

    gdal_path = COHERENCE_GPKG_TEMPLATE.format(
        season=season.value, variable=variable.value
    )

    with rasterio.open(gdal_path) as src:
        # Get the window for the bounds
        window = windows.from_bounds(*bounds, src.transform)
        shape = shape or (window.height, window.width)

        profile = src.profile.copy()
        if upsample_factors is not None and upsample_factors != (1, 1):
            shape = tuple(np.round(np.array(shape) * upsample_factors).astype(int))
            # shape = int(
            #     shape[0] * upsample_factors[0], int(shape[1] * upsample_factors[1])
            # )

        # Read the data
        data = src.read(
            1,
            window=window,
            out_shape=shape,
            resampling=rasterio.enums.Resampling.bilinear,
        )

        transform_scale = Affine.scale(
            window.width / data.shape[-1], window.height / data.shape[-2]
        )
        # Get the profile for the windowed read
        profile.update(
            {
                "height": data.shape[-2],
                "width": data.shape[-1],
                "transform": windows.transform(window, src.transform) * transform_scale,
                "driver": "GTiff",
            }
        )

        data = convert_to_float(data, variable)
        # print(f"{season=} {variable=} {data.max()=}")
        profile.update(dtype="float32")

    # if shape is not None or (
    #     upsample_factors is not None and upsample_factors != (1, 1)
    # ):
    #     if shape is None:
    #         shape = (
    #             int(profile["height"] * upsample_factors[0]),
    #             int(profile["width"] * upsample_factors[1]),
    #         )
    #     data = _interpolate_data(data, shape, method=interp_method)

    #     # Update the profile with the new shape
    #     profile.update(height=shape[0], width=shape[1])
    #     profile["transform"] *= Affine.scale(
    #         1 / upsample_factors[1], 1 / upsample_factors[0]
    #     )

    # data = data.astype("float16")
    if outfile:
        compression = {"compress": "lzw", "nbits": "16"}
        with rasterio.open(outfile, "w", **(profile | compression)) as dst:
            dst.write(data, 1)
        logger.info(f"Raster data saved to {outfile}")

    return data, profile


def _interpolate_data(
    data: np.ndarray, shape: tuple[int, int], method="linear"
) -> np.ndarray:
    from scipy.interpolate import RegularGridInterpolator

    # Create coordinate arrays for the original data
    orig_coords = [np.linspace(0, 1, s) for s in data.shape]

    # Create coordinate arrays for the desired output shape
    new_coords = [np.linspace(0, 1, s) for s in shape]

    # Create the interpolator
    interp = RegularGridInterpolator(orig_coords, data, method=method)

    # Create a mesh grid for the new coordinates
    mesh = np.meshgrid(*new_coords, indexing="xy", sparse=True)

    # Perform the interpolation
    return interp(np.array(mesh).T)


def model_2param(t: np.ndarray, rho_inf: float, tau: float, *args) -> np.ndarray:
    """Two-parameter model for coherence decay.

    Parameters
    ----------
    t : np.ndarray
        Array of temporal baselines.
    rho_inf : float
        Asymptotic coherence value.
    tau : float
        Coherence decay time constant.

    Returns
    -------
    np.ndarray
        Array of coherence values.

    """
    return (1 - rho_inf) * np.exp(-t / tau) + rho_inf


def model_3param(t: np.ndarray, rho_0: float, rho_inf: float, tau: float) -> np.ndarray:
    """Three-parameter model for coherence decay.

    Parameters
    ----------
    t : np.ndarray
        Array of temporal baselines.
    rho_0 : float
        Initial coherence value.
    rho_inf : float
        Asymptotic coherence value.
    tau : float
        Coherence decay time constant.

    Returns
    -------
    np.ndarray
        Array of coherence values.

    """
    return (rho_0 - rho_inf) * np.exp(-t / tau) + rho_inf


def fit_model(
    T: ArrayLike, gamma: ArrayLike, num_params: int = 2, plot: bool = True, ax=None
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the model to the data."""
    from scipy.optimize import curve_fit

    if num_params == 2:
        p0 = [0.5, 12.0]  # Initial guess for [rho_inf, tau]
        model = model_2param
    elif num_params == 3:
        p0 = [0.7, 0.2, 12.0]  # Initial guess for [rho_0, rho_inf, tau]
        # ignore the mypy error
        model = model_3param  # type: ignore[assignment]
    else:
        raise ValueError("num_params must be 2 or 3")
    # popt now contains the optimal values for [rho_inf, tau] or [rho_0, rho_inf, tau]
    popt, pcov = curve_fit(model, T, gamma, p0=p0)

    if plot:
        import matplotlib.pyplot as plt

        param_names = [r"\rho_{\infty}", r"\tau"]
        if num_params == 3:
            param_names.insert(0, r"\rho_{0}")

        T_fit = np.linspace(min(T), max(T), 100)
        gamma_fit = model(T_fit, *popt)
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(T, gamma, label="Data")
        ax.plot(T_fit, gamma_fit, "r-", label="Fitted model")
        ax.set_xlabel("Temporal baseline (T)")
        ax.set_ylabel("Coherence (gamma)")
        title_parts = [f"{name} = {p:.2f}" for p, name in zip(popt, param_names)]
        title = r"$" + r",\;".join(title_parts) + r"$"
        ax.set_title(title)
        ax.legend()

    return popt, pcov


from collections.abc import Callable
from concurrent.futures import Executor, Future
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


class DummyProcessPoolExecutor(Executor):
    """Dummy ProcessPoolExecutor for to avoid forking for single_job purposes."""

    def __init__(self, max_workers: int | None = None, **kwargs):  # noqa: D107
        self._max_workers = max_workers

    def submit(  # noqa: D102
        self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        future: Future = Future()
        result = fn(*args, **kwargs)
        future.set_result(result)
        return future

    def shutdown(self, wait: bool = True, cancel_futures: bool = True):  # noqa: D102
        pass


def fetch_rho_tau_amp(
    bounds: Bbox,
    upsample: tuple[int, int] = (1, 1),
    output_dir=Path(),
    max_workers: int = 4,
):
    def fetch_single(variable, season) -> Path:
        outfile = output_dir / f"{variable}_{season}.tif"
        get_rasters(
            bounds,
            season=season,
            variable=variable,
            outfile=outfile,
            upsample_factors=upsample,
        )
        return outfile

    seasons = [s.value for s in Season]

    results = {}
    Executor = DummyProcessPoolExecutor if max_workers == 1 else ThreadPoolExecutor
    fetch_single("rho", "winter")
    with Executor(max_workers=max_workers) as executor:
        future_to_params = {
            executor.submit(fetch_single, variable, season): (variable, season)
            for season in seasons
            for variable in ["rho", "tau", "amp"]
        }

        for future in as_completed(future_to_params):
            variable, season = future_to_params[future]
            try:
                results[(variable, season)] = future.result()
            except Exception as exc:
                print(f"{variable}_{season} generated an exception: {exc}")

    rho_files = [results[("rho", season)] for season in seasons]
    tau_files = [results[("tau", season)] for season in seasons]
    amp_files = [results[("amp", season)] for season in seasons]
    return rho_files, tau_files, amp_files

    # # Pick out the array, not the profile
    # rho_stack = np.stack([r[0] for r in rhos])
    # tau_stack = np.stack([t[0] for t in taus])
    # amp_stack = np.stack([t[0] for t in amps])
    # # get one profile:
    # profile = rhos[0][1]
    # return rho_stack, tau_stack, amp_stack, profile


def get_coherence_model_coeffs(
    bounds: Bbox,
    seasonal_ptp_cutoff: float = 0.5,
    upsample: tuple[int, int] = (1, 1),
    output_dir: Path = Path(),
    max_workers: int = 4,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    # rho_stack, tau_stack, amp_stack, profile = fetch_rho_tau_amp(
    rho_files, tau_files, amp_files = fetch_rho_tau_amp(
        bounds=bounds,
        upsample=upsample,
        output_dir=output_dir,
        max_workers=max_workers,
    )

    # A, B, seasonal_mask = calculate_seasonal_coeffs(
    # rho_stack=rho_stack, seasonal_ptp_cutoff=seasonal_ptp_cutoff

    (
        amp_mean_file,
        rho_max_file,
        tau_max_file,
        seasonal_A_file,
        seasonal_B_file,
        seasonal_mask_file,
    ) = calculate_seasonal_coeffs_files(
        rho_files=rho_files,
        tau_files=tau_files,
        amp_files=amp_files,
        seasonal_ptp_cutoff=seasonal_ptp_cutoff,
    )
    return (
        amp_mean_file,
        rho_max_file,
        tau_max_file,
        seasonal_A_file,
        seasonal_B_file,
        seasonal_mask_file,
    )
    # rho_max = np.max(rho_stack, axis=0)
    # tau_max = tau_stack.max(axis=0)
    # amp_mean = np.mean(amp_stack, axis=0)
    # save_coherence_data(
    #     output_dir / "global_coherence_data.h5",
    #     amp_mean,
    #     rho_max,
    #     tau_max,
    #     A,
    #     B,
    #     seasonal_mask,
    #     profile,
    # )
    # return amp_mean, rho_max, tau_max, A, B, seasonal_mask, profile


def save_coherence_data(
    outfile, amps, rhos, taus, seasonal_A, seasonal_B, seasonal_mask, profile
):
    import h5py

    assert rhos.ndim == 2
    assert (
        rhos.shape
        == taus.shape
        == seasonal_A.shape
        == seasonal_B.shape
        == seasonal_mask.shape
    )
    with h5py.File(outfile, "w") as hf:
        hf["amps"] = amps
        hf["rhos"] = rhos
        hf["taus"] = taus
        hf["seasonal_A"] = seasonal_A
        hf["seasonal_B"] = seasonal_B
        hf["seasonal_mask"] = seasonal_mask
        # could possibly serialize... but dont need now
        # https://rasterio.groups.io/g/main/topic/serializing_and_deserializing/34163315
        # hf["profile"] = profile


def calculate_seasonal_coeffs(
    rho_stack: np.ndarray, seasonal_ptp_cutoff: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho_ptp = np.ptp(rho_stack, axis=0)
    # For pixels where there's more than `seasonal_ptp_cutoff`, we'll model the decorrelation
    # as seasonal instead of exponential decay
    seasonal_pixels = rho_ptp > seasonal_ptp_cutoff
    rho_min = rho_stack.min(axis=0)
    # A, B: Where theres big variation, we use A, B for seasonal simulations
    A, B = rho_to_AB(rho_min)
    # otherwise,  we just do an exponential decay
    return A, B, seasonal_pixels


def _log_and_run(cmd: str):
    import subprocess

    logger.info(cmd)
    # print(cmd)
    subprocess.run(cmd, shell=True, check=True)


def calculate_seasonal_coeffs_files(
    rho_files: Sequence[Path],
    tau_files: Sequence[Path],
    amp_files: Sequence[Path],
    seasonal_ptp_cutoff: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the seasonal A and B parameters from a list of rho files."""
    # Using gdal calc, we can avoid loading everything, and it will save to a new file
    common_opts = " --co TILED=YES --co COMPRESS=LZW --co BIGTIFF=YES"
    common_opts += " --co NBITS=16 --NoDataValue 0 "

    base_cmd = f"gdal_calc --quiet {common_opts} "
    # Get the mean of the amp files:
    amp_mean_out = amp_files[0].parent / "amp_mean.tif"
    cmd = f"{base_cmd} -A {' '.join(map(str, amp_files))} --outfile={amp_mean_out} --calc='numpy.mean(A,axis=0)'"
    if not amp_mean_out.exists():
        _log_and_run(cmd)

    # Like the example  https://gdal.org/programs/gdal_calc.html
    # gdal_calc -A input1.tif input2.tif input3.tif --outfile=result.tif --calc="numpy.max(A,axis=0)"
    # Get the peak-to-peak raster first:
    a, b, c, d = rho_files
    ptp_out = a.parent / "rho_ptp.tif"
    cmd = f"{base_cmd} -A {a} {b} {c} {d} --outfile={ptp_out} --calc='numpy.ptp(A,axis=0)'"
    if not ptp_out.exists():
        _log_and_run(cmd)

    # Get the minimum of rho:
    rho_min_out = a.parent / "rho_min_out.tif"
    cmd = f"{base_cmd} -A {a} {b} {c} {d} --outfile={rho_min_out} --calc='numpy.min(A,axis=0)'"
    if not rho_min_out.exists():
        _log_and_run(cmd)

    # Get the maximum of rho:
    rho_max_out = a.parent / "rho_max.tif"
    cmd = f"{base_cmd} -A {a} {b} {c} {d} --outfile={rho_max_out} --calc='numpy.max(A,axis=0)'"
    if not rho_max_out.exists():
        _log_and_run(cmd)

    # Calculate the seasonal mask:
    seasonal_mask_out = a.parent / "seasonal_mask.tif"
    cmd = f"{base_cmd} -A {ptp_out} --outfile={seasonal_mask_out} --calc='numpy.where(A > {seasonal_ptp_cutoff}, 1, 0)' --NoDataValue=255 --type=Byte"
    if not seasonal_mask_out.exists():
        _log_and_run(cmd)

    # Calculate seasonal_A.tif
    seasonal_A_out = a.parent / "seasonal_A.tif"
    cmd = f"{base_cmd} -A {rho_min_out} --outfile={seasonal_A_out} --calc='0.5 * (1 + numpy.sqrt(A))'"
    if not seasonal_A_out.exists():
        _log_and_run(cmd)

    # Calculate seasonal_B.tif
    seasonal_B_out = a.parent / "seasonal_B.tif"
    cmd = f"{base_cmd} -A {rho_min_out} --outfile={seasonal_B_out} --calc='0.5 * (1 - numpy.sqrt(A))'"
    if not seasonal_B_out.exists():
        _log_and_run(cmd)

    # Get the maximum of tau:
    tau_max_out = a.parent / "tau_max.tif"
    cmd = f"{base_cmd} -A {' '.join(map(str, tau_files))} --outfile={tau_max_out} --calc='numpy.max(A,axis=0)'"
    if not tau_max_out.exists():
        _log_and_run(cmd)

    return (
        amp_mean_out,
        rho_max_out,
        tau_max_out,
        seasonal_A_out,
        seasonal_B_out,
        seasonal_mask_out,
    )


def rho_to_AB(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert rho to A and B parameters."""
    A = 0.5 * (1 + np.sqrt(rho))
    B = 0.5 * (1 - np.sqrt(rho))
    return A, B
