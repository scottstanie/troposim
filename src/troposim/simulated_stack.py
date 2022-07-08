import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
import rasterio as rio

from troposim.deformation import okada

from . import decorrelation, deformation, stratified, turbulence, utils

SENTINEL_WAVELENGTH = 5.5465763  # cm

# TODO: utils.record_params_as_yaml
# TODO: downsample as data augmentation
def generate_stacks(
    demfile,
    outfile="simulated_stack.h5",
    dsets=["defo", "stratified", "turbulence", "decorrelation"],
    num_days=9,
    num_defos=120,
    defo_shape=(200, 200),
    add_day1_turbulence=False,
    turbulence_kwargs={},  # p0=1e-2,
    stratified_kwargs={},
    deformation_kwargs={},
):
    """

    Parameters
    ----------
    demfile :
        
    outfile :
        (Default value = "simulated_stack.h5")
    dsets :
        (Default value = ["defo")
    "stratified" :
        
    "turbulence" :
        
    "decorrelation"] :
        
    num_days : int
        (Default value = 9)
    num_defos : int
        (Default value = 120)
    defo_shape : tuple[int, int]
        (Default value = (200, 200))
    add_day1_turbulence :
        (Default value = False)
    turbulence_kwargs :
        (Default value = {})
    stratified_kwargs :
        (Default value = {})
    deformation_kwargs :
        (Default value = {})

    Returns
    -------
    outfile : str
        path to output file
    """
    # SRTM DEM pixel spacing, in degrees
    res_30_degrees = 0.000277777777

    with rio.open(demfile) as src:
        dem = src.read(1)
        dem = dem.astype(np.float32)
        resolution = round(src.transform[0] / res_30_degrees * 30)

    # dem = dem[:500, :500]
    shape = dem.shape
    stratified_kwargs.update({"K_params": {"shape": (num_days,)}})
    print(stratified_kwargs)
    strat = stratified.simulate(dem, **stratified_kwargs)
    print(
        f"Stratified (min, max, mean): {strat.min(), strat.max(), strat.mean(axis=(1,2))}"
    )
    # Multiply by 100 to make into cm
    turb = 100 * turbulence.simulate(
        shape=(num_days, *shape), resolution=resolution, **turbulence_kwargs
    )
    if add_day1_turbulence:
        turb_day1 = 100 * turbulence.simulate(
            shape=shape, resolution=resolution, **turbulence_kwargs
        )
        turb += turb_day1.reshape((1, *shape))
    print(
        f"Turbulence (min, max, mean): {turb.min(), turb.max(), turb.mean(axis=(1,2))}"
    )

    # Decorrelation phase from a random coherence map
    # The coherence map is generated from the turbulence function, but it's just
    # a random function to spatially vary the coherence.
    looks = 50
    coh = np.clip(0.5 + 100 * turbulence.simulate(shape=shape), 0, 1)
    coh = np.tile(coh, (num_days, 1, 1))
    cohphase = decorrelation.simulate(coh, looks) * PHASE_TO_CM_S1
    # cohphase = np.zeros((num_days, shape[0], shape[1]), dtype=np.float32)
    # for i in range(num_days):
    #     cohphase[i] = decorr.coherence2decorrelation_phase(coh, looks)
    #     cohphase[i] *= constants.PHASE_TO_CM_S1
    # with ProcessPoolExecutor(max_workers=num_days) as executor:
    #     futures = [
    #         executor.submit(
    #             decorrelation.simulate
    #             coh,
    #             looks,
    #         )
    #         for _ in range(num_days)
    #     ]
    #     for future in as_completed(futures):
    #         cohphase.append(future.result() * PHASE_TO_CM_S1)
    # cohphase = np.array(cohphase)
    print(f"Cohphase (min, max): {cohphase.min(), cohphase.max()}")

    # noise_total = strat + turb + cohphase

    defo_stack = np.zeros((num_days, shape[0], shape[1]), dtype=np.float32)

    with ProcessPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(
                deformation.okada.random_displacement,
                resolution=resolution,
                shape=defo_shape,
                **deformation_kwargs,
            )
            for _ in range(num_defos)
        ]
        for future in as_completed(futures):
            defo, _, _ = future.result()
            defo *= 100
            los = _get_random_los()
            defo = okada.project_to_los(defo, los)

            row = np.random.randint(0, shape[0] - defo_shape[0])
            col = np.random.randint(0, shape[1] - defo_shape[1])
            #  randint is (inclusive, exclusive)
            # date_idx = np.random.randint(1, num_days - 1)
            # defo can go on any date, but the network will ignore it if it
            # appears on the first or last date
            date_idx = np.random.randint(0, num_days)
            # add the deformation to the stack at all points on or after `date_idx`
            defo_stack[
                date_idx:, row : row + defo_shape[0], col : col + defo_shape[1]
            ] += defo
    print(f"Defo stack (min, max): {defo_stack.min(), defo_stack.max()}")

    attrs = {
        "defo": {
            "shape": defo_shape,
            "num_defos": num_defos,
            "kwargs": str(deformation_kwargs),
        },
        "stratified": {"kwargs": str(stratified_kwargs)},
        "turbulence": {
            "add_day1_turbulence": add_day1_turbulence,
            "kwargs": str(turbulence_kwargs),
        },
        "decorrelation": {"looks": looks},
    }
    print("Saving stacks")
    comp_dict = {"compression": 32001, "compression_opts": (0, 0, 0, 0, 5, 1, 1)}
    stacks = [defo_stack, strat, turb, cohphase]
    with h5py.File(outfile, "w") as hf:
        hf.attrs["resolution"] = resolution
        for data, dset in zip(stacks, dsets):
            hf.create_dataset(
                dset,
                shape=data.shape,
                dtype="float32",
                data=data,
                chunks=True,
                **comp_dict,
            )
            for k, v in attrs[dset].items():
                hf[dset].attrs[k] = v
    # stack = noise_total.copy()
    # stack += defo_stack
    return outfile


def _get_random_los():
    """Picks a random LOS, similar to Sentinel-1 range"""
    north = 0.1
    up = -1 * np.random.uniform(0.5, 0.85)
    east = np.sqrt(1 - up**2 - north**2)
    # sometimes ascending pointing east, sometimes descending pointing west
    east *= np.random.choice([-1, 1])
    return [east, north, up]


def data_loader(
    sim_file,
    dem_file,
    chunk_size=(48, 48),
    looks=(1, 1),
    normalize=False,
    load=True,
    no_noise=False,
):
    """Create a generator that yields chunks of data from the noise and defo files.

    Parameters
    ----------
    noise_file : str
        Name of the noise file.
    defo_file : str
        Name of the defo file.
    dem_file : str
        Name of the DEM file.
    chunk_size : tuple
        Spatial size of chuns to yield. Defaults to (48, 48).
    looks :
        (Default value = (1, 1)
    normalize : bool
        Whether to normalize the data to [-1, 1].
        Defaults to True.
    load : bool
        Whether to load the data into memory. Defaults to True.
    no_noise : bool
        Whether to skip loading the noise file. Defaults to False.
    sim_file :
        

    Yields
    -------
    X : 3D np.ndarray
        data input
    y : np.ndarray
        target. Really 2D, but padded to match the shape of X.
    """
    hf = h5py.File(sim_file, "r")
    full_shape = hf["defo"].shape
    defo_stack = hf["defo"]

    if not no_noise:
        strat_stack = hf["stratified"]
        turb_stack = hf["turbulence"]
        decor_stack = hf["decorrelation"]
    else:
        strat_stack = turb_stack = decor_stack = np.zeros_like(defo_stack)
    if load:
        defo_stack = defo_stack[()]
        strat_stack = strat_stack[()]
        turb_stack = turb_stack[()]
        decor_stack = decor_stack[()]
        if looks != (1, 1):
            defo_stack = utils.take_looks(defo_stack, *looks)
            strat_stack = utils.take_looks(strat_stack, *looks)
            turb_stack = utils.take_looks(turb_stack, *looks)
            decor_stack = utils.take_looks(decor_stack, *looks)
        print(f"{defo_stack.shape = }")

    if normalize:
        c1 = np.max(np.abs(defo_stack))
        c2 = np.max(np.abs(strat_stack))
        c3 = np.max(np.abs(turb_stack))
        c4 = np.max(np.abs(decor_stack))
        c = np.max([c1, c2, c3, c4])
    else:
        c = 1.0

    with rio.open(dem_file) as src:
        dem = src.read(1).astype(np.float32)
        if normalize:
            # Scale DEM to be between 0 and 1
            # dem = (dem - dem.min()) / (dem.max() - dem.min())
            # Scale DEM to be zero mean and unit variance
            dem = (dem - dem.mean()) / dem.std()
            # does it make a difference which normalizing?
        if looks != (1, 1):
            dem = utils.take_looks(dem, *looks)

    blk_slices = utils.block_iterator(
        full_shape[-2:], chunk_size, overlaps=(0, 0), start_offsets=(0, 0)
    )

    for (rows, cols) in blk_slices:
        # with h5py.File(sim_file) as hf:
        dem_chunk = dem[slice(*rows), slice(*cols)].astype(np.float32)
        defo = defo_stack[:, slice(*rows), slice(*cols)]

        strat = strat_stack[:, slice(*rows), slice(*cols)]
        turb = turb_stack[:, slice(*rows), slice(*cols)]
        decor = decor_stack[:, slice(*rows), slice(*cols)]
        noise = strat + turb + decor
        # with rio.open(dem_file) as src:
        #     window = rio.windows.Window.from_slices(rows, cols)
        #     dem_chunk = src.read(1, window=window)
        #     if normalize:
        #         # Turn the DEM into a kilometer scale
        #         dem_chunk = (dem_chunk / 1000).astype(np.float32)
        if defo.shape[-2:] != chunk_size or noise.shape[-2:] != chunk_size:
            continue

        X = (noise + defo) / c
        # Make the DEM the first layer of the stack
        X = np.concatenate([dem_chunk[np.newaxis], X], axis=0)

        # For the target, we don't want to include the deformation that occurs on the first
        # date (since we'll be sliding this along through time and don't want to keep outputting
        # the same deformation), and we also want to ignore the last date's deformation
        target_defo = defo[-2] - defo[0]
        y = target_defo[np.newaxis, :, :] / c
        yield X, y
    hf.close()


def load_all_data(
    sim_file_glob=None,
    dem_file=None,
    sim_file_list=None,
    chunk_size=(48, 48),
    looks=(1, 1),
    normalize=False,
    no_noise=False,
    load=True,
    min_y=0.7,  # cm, unnormalized
    max_y=5,  # cm, unnormalized
):
    """

    Parameters
    ----------
    sim_file_glob :
        (Default value = None)
    dem_file :
        (Default value = None)
    sim_file_list :
        (Default value = None)
    chunk_size :
        (Default value = (48)
    48) :
        
    looks : tuple[int, int]
        (Default value = (1, 1)
        
    normalize :
        (Default value = False)
    no_noise :
        (Default value = False)
    load : bool
        (Default value = True)
    min_y : float
        (Default value = 0.7 cm)
    max_y : float
        (Default value = 5 cm)
    
    """
    if sim_file_list is None:
        sim_file_list = glob.glob(sim_file_glob)
    X_all = None
    y_all = None
    for f in sim_file_list:
        dl = data_loader(
            f,
            dem_file,
            chunk_size=chunk_size,
            normalize=normalize,
            load=load,
            no_noise=no_noise,
            looks=looks,
        )
        rows = list(dl)
        Xs, ys = zip(*rows)
        Xs, ys = np.array(Xs), np.array(ys)
        # TODO: add a random sample of zeros?
        # zero_idxs = [i for i, y in enumerate(ys) if np.sum(np.abs(y)) == 0]
        # num_zeros_to_keep = int(pct_zeros * len(ys))
        axis = (1, 2, 3) if ys.ndim == 4 else (1, 2)
        y_maxes = np.max(np.abs(ys), axis=axis)
        keep_idxs = np.logical_and(y_maxes > min_y, y_maxes < max_y)
        # keep_idxs = [i for i, y in enumerate(ys) if np.max(np.abs(y)) > min_y]
        Xs = Xs[keep_idxs]
        ys = ys[keep_idxs]

        if X_all is None:
            X_all = Xs
            y_all = ys
        else:
            X_all = np.concatenate((X_all, Xs))
            y_all = np.concatenate((y_all, ys))

    return X_all, y_all


def plot_data(X, y, dem, vm=None, cmap="RdBu"):
    import matplotlib.pyplot as plt

    vm = vm or np.max(np.abs(X))
    # fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    fig = plt.figure(constrained_layout=True, figsize=(9, 6))
    gs0 = fig.add_gridspec(1, 2)

    gs_left = gs0[0].subgridspec(3, 3)
    gs_right = gs0[1].subgridspec(2, 1)
    # Input timesteps
    for a in range(3):
        for b in range(3):
            i = 3 * a + b
            ax = fig.add_subplot(gs_left[a, b])
            ax.imshow(X[i], cmap=cmap, vmin=-vm, vmax=vm)
            ax.set_title(f"{i}")

    # Right side: DEM, and target deformation
    # axes = axes.ravel()
    ax = fig.add_subplot(gs_right[0, 0])
    axim = ax.imshow(dem, cmap="terrain")
    fig.colorbar(axim, ax=ax)
    ax.set_title("DEM")

    ax = fig.add_subplot(gs_right[1, 0])
    axim = ax.imshow(y, cmap=cmap, vmax=vm, vmin=-vm)
    fig.colorbar(axim, ax=ax)
    ax.set_title("Target deformation")
    # axes[i].set_title(f"{y[i]:.2f} cm")


def power(x):
    return np.mean(x**2)


def load_noise(hf):
    return hf["turbulence"][()] + hf["stratified"][()] + hf["decorrelation"][()]


def snr(hf):
    return power(hf["defo"][()]) / power(load_noise(hf))


def peak_ratio(hf, axis=None):
    defo_max = np.max(np.abs(hf["defo"][()]), axis=axis)
    noise_max = np.max(np.abs(load_noise(hf)), axis=axis)
    return defo_max / noise_max


def get_stack_metrics(glob_str, func, *args, **kwargs):
    file_list = glob.glob(glob_str)
    metrics = []
    for f in file_list:
        with h5py.File(f) as hf:
            metrics.append(func(hf, *args, **kwargs))
    return file_list, metrics


def get_snrs(glob_str, func=snr):
    return get_stack_metrics(glob_str, func)


def get_peaks(glob_str, func=peak_ratio):
    return get_stack_metrics(glob_str, func)


class MeanMaxScaler:
    def __init__(self):
        pass

    def fit(self, X):
        self.mean_ = np.average(X)
        # self.scale_ = np.max(np.abs(X))
        self.scale_ = np.max(np.abs(X - self.mean_))  # check with Scott on this
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X):
        return X * self.scale_ + self.mean_


class LogMeanScaler:
    """Can work well to normalize skewed data, I tried it but no huge improvement at least
    on very noisy data.
    """

    def __init__(self):
        pass

    def fit(self, X):
        self.shift_ = (
            0 if np.min(X) >= 0 else -np.min(X)
        ) + 0.0001  # log10(0) is undefined so slight shift
        self.mean_ = np.average(np.log(X + self.shift_))
        return self

    def transform(self, X):
        return np.log(X + self.shift_) - self.mean_

    def inverse_transform(self, X):
        return np.exp(X + self.mean_) - self.shift_


class Scaler:
    """ """
    def __init__(self, scale_percentile=95):
        self.mean_, self.scale_ = None, None
        self.dem_mean_, self.dem_scale_ = None, None
        self.scale_percentile = scale_percentile

    def fit(self, X):
        # Divide into dem/timeseries
        dems, timeseries = self._unstack_dem_timeseries(X)

        if dems is not None:
            # Scale DEM to be between -1 and 1
            self.dem_mean_ = np.mean(dems)
            # self.dem_scale_ = np.max(np.abs(dems - self.dem_mean_))
            self.dem_scale_ = np.std(dems)  # For setting unit variance

        self.mean_ = np.mean(timeseries)
        # self.scale_ = np.max(np.abs(timeseries))
        self.scale_ = np.percentile(np.abs(timeseries), self.scale_percentile)

        return self

    def transform(self, X):
        dems, timeseries = self._unstack_dem_timeseries(X)

        if dems is not None:
            if self.dem_mean_ is None:
                raise ValueError("Must fit DEM before transforming")
            dems_scaled = (dems - self.dem_mean_) / self.dem_scale_
            timeseries_scaled = (timeseries - self.mean_) / self.scale_
            return self._stack_dem_timeseries(dems_scaled, timeseries_scaled)
        else:
            return (timeseries - self.mean_) / self.scale_

    def inverse_transform(self, X):
        dems, timeseries = self._unstack_dem_timeseries(X)

        if dems is not None:
            if self.dem_mean_ is None:
                raise ValueError("Must fit DEM before inverse transforming")
            dems_unscaled = dems * self.dem_scale_ + self.dem_mean_
            ts_unscaled = timeseries * self.scale_ + self.mean_
            return self._stack_dem_timeseries(dems_unscaled, ts_unscaled)
        else:
            return timeseries * self.scale_ + self.mean_

    def _unstack_dem_timeseries(self, X):
        layer0 = X[:, :1]
        other_layers = X[:, 1:]
        # The y targets will be shape (batch, 1, rows, cols)
        # So `other_layers` will be shape (batch, 0, rows, cols) after unstacking.

        if other_layers.size > 0:
            dems = layer0
            timeseries = other_layers
        else:
            dems = None
            timeseries = layer0
        return dems, timeseries

    def _stack_dem_timeseries(self, dems, timeseries):
        return np.concatenate((dems, timeseries), axis=1)

    def asdict(self):
        d = {"mean": self.mean_, "scale": self.scale_}
        if self.dem_mean_ is not None:
            d["dem_mean"] = self.dem_mean_
            d["dem_scale"] = self.dem_scale_
        return d


def plot_model_filters(model, layer_num=-1, cmap="plasma", in_idx=0):
    import matplotlib.pyplot as plt

    weights = model.get_weights()[layer_num]
    n_filters = weights.shape[-1]

    n_rows = int(np.ceil(np.sqrt(n_filters)))
    n_cols = int(np.ceil(n_filters / n_rows))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10), squeeze=False)

    for ax, i in zip(axes.flat, range(n_filters)):
        vm = np.max(np.abs(weights[..., in_idx, :]))
        filt = np.squeeze(weights[..., in_idx, i])
        filt = filt[0] if len(filt.shape) == 3 else filt
        ax.imshow(filt, cmap=cmap, vmax=vm, vmin=-vm)
        ax.axis("off")
        ax.set_title("Filter {}".format(i))


def save_predictions_as_tifs(
    arr, ds_file, idxs, outfile_template="{inp}_predicted_{d1}_{d2}.tif"
):
    import xarray as xr
    from apertools import sario

    with xr.open_dataset(ds_file) as ds:
        lats = ds["lat"]
        lons = ds["lon"]
        date_ranges = ds["date_ranges"]

    for idx, img in zip(idxs, arr):
        dates = date_ranges[idx].dt.strftime("%Y%m%d")
        d1, d2 = dates[0].item(), dates[-1].item()
        da = xr.DataArray(img, coords={"lat": lats, "lon": lons}, dims=["lat", "lon"])

        outfile = outfile_template.format(inp=ds_file.replace(".nc", ""), d1=d1, d2=d2)
        sario.save_xr_tif(outfile, da)
