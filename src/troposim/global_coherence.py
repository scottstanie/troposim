import logging
from enum import Enum
import numpy as np
from numpy.typing import ArrayLike

from ._types import Bbox, PathOrStr

logger = logging.getLogger(__name__)


class Season(Enum):
    """Seasonal periods for the year."""

    WINTER = "winter"
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"


class Variable(Enum):
    """Variables for the global coherence raster."""

    AMP = "AMP"  # capitalized in the dataset
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
        return X.astype("float32") ** 2 / 199526231
    else:
        return X.astype("float32") / 1000.0


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
    import rasterio
    from tile_mate import get_raster_from_tiles

    variable = Variable(variable)
    season = Season(season)

    X, profile = get_raster_from_tiles(
        bounds,
        tile_shortname="s1_coherence_2020",
        season=season.value,
        s1_decay_model_param=variable.value,
    )
    if convert_data:
        data = convert_to_float(X[0], variable)
        profile.update(dtype="float32")
    else:
        data = X[0]
    if shape is not None or (
        upsample_factors is not None and upsample_factors != (1, 1)
    ):
        if shape is None:
            shape = (
                int(profile["height"] * upsample_factors[0]),
                int(profile["width"] * upsample_factors[1]),
            )
        # interpolate to be the same shape
        data = _interpolate_data(data, shape, method=interp_method)
        profile.update(height=shape[0], width=shape[1])

    if outfile:
        # Update the profile with the new shape if it was changed

        # Write the data to the output file
        with rasterio.open(outfile, "w", **profile) as dst:
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
    mesh = np.meshgrid(*new_coords, indexing="xy")

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


def get_yearly_coeffs(bounds: Bbox, seasonal_ptp_cutoff: float = 0.5):
    rhos = [
        get_rasters(bounds, season=season, variable="rho", outfile=f"rho_{season}.tif")
        for season in ["fall", "winter", "spring", "summer"]
    ]
    taus = [
        get_rasters(bounds, season=season, variable="tau", outfile=f"tau_{season}.tif")
        for season in ["fall", "winter", "spring", "summer"]
    ]
    # Pick out the array, not the profile
    rho_stack = np.stack([r[0] for r in rhos])
    tau_stack = np.stack([t[0] for t in taus])

    return calculate_coeffs(rho_stack=rho_stack, tau_stack=tau_stack)


def calculate_coeffs(rho_stack, tau_stack, seasonal_ptp_cutoff: float = 0.5):
    rho_ptp = np.ptp(rho_stack, axis=0)
    # For pixels where there's more than `seasonal_ptp_cutoff`, we'll model the decorrelation
    # as seasonal instead of exponential decay
    seasonal_pixels = rho_ptp > seasonal_ptp_cutoff
    rho_min = rho_stack.min(axis=0)
    rho_max = rho_stack.max(axis=0)
    tau_max = tau_stack.max(axis=0)
    A, B = rho_to_AB(rho_min)
    # Two coefficients:
    # 1. A, B: Where theres big variation, we use A, B for seasonal simulations
    # 2. rho_inf, tau: Where it's similar, we just do an exponential decay
    coeff1 = np.where(seasonal_pixels, A, rho_max)
    coeff2 = np.where(seasonal_pixels, B, tau_max)
    return coeff1, coeff2, seasonal_pixels


def rho_to_AB(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert rho to A and B parameters."""
    A = 0.5 * (1 + np.sqrt(rho))
    B = 0.5 * (1 - np.sqrt(rho))
    return A, B
