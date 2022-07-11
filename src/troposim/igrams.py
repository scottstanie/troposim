from dataclasses import dataclass
from datetime import date, timedelta
from itertools import combinations
from typing import Tuple

import numpy as np

from . import turbulence, utils


@dataclass
class IgramMaker:
    """Class for creating synthetic interferograms """

    num_days: int = 10
    shape: Tuple[int] = (700, 700)
    start_date: date = date(2018, 1, 1)
    repeat_interval_days: int = 12
    resolution: int = 400
    p0_default: float = 10.0
    freq0: float = 1e-4
    sar_stack: np.ndarray = None
    to_cm: bool = True
    defo_rates: Tuple[float] = (0.0,)
    defo_sigma = 5
    smooth_defo_days: int = 2
    # num_bad_days = 5
    # bad_day_mult = 8
    ref: Tuple[int] = (5, 5)
    distribution: str = "normal"

    def make_sar_dates(self):
        """ """
        sar_date_list = []
        for idx in range(self.num_days):
            sar_date_list.append(
                self.start_date + timedelta(days=idx * self.repeat_interval_days)
            )
        return sar_date_list

    def make_sar_stack(
        self,
        beta=None,
        beta_arr=None,
        beta_savename=None,
        p0_params=None,
        p0_arr=None,
        # p0_rv="lognorm",
        p0_rv=None,
        seed=None,
    ):
        """

        Parameters
        ----------
        beta :
             (Default value = None)
        beta_arr :
             (Default value = None)
        beta_savename :
             (Default value = None)
        p0_params :
             (Default value = None)
        p0_arr :
             (Default value = None)
        p0_rv :
             (Default value = None)
        seed :
             (Default value = None)

        """

        from scipy import stats

        stack_shape = (self.num_days, *self.shape)
        # Load/create the beta PSD slopes
        if beta_arr is not None and len(beta_arr):
            beta = np.random.choice(beta_arr, size=(self.num_days,), replace=True)

        if beta is None:
            if beta_savename is not None:
                print(f"Loading beta polynomial from {beta_savename}")
                beta = np.polynomial.Polynomial(
                    np.load(beta_savename, allow_pickle=True)
                )
            else:
                beta = 8.0 / 3.0
                print(f"Using {beta = :.3f}")
        self.beta = beta

        # Load/create the power random generator
        if p0_arr is not None and len(p0_arr):
            self.p0_arr = np.random.choice(p0_arr, size=(self.num_days,), replace=True)
        else:
            if isinstance(p0_params, str):
                # get the 'lognorm' or 'expon' from the filename
                if p0_rv is None:
                    p0_rv = p0_params.replace("params_", "").replace(".npy", "")
                # Then load the params file
                print(f"Loading p0 RV data from {p0_params}")
                p0_params = np.load(p0_params)
                print(f"{p0_params = }")

            if isinstance(p0_rv, str):
                p0_frozen = getattr(stats, p0_rv)(**p0_params)
                self.p0_arr = p0_frozen.rvs(self.num_days)
            elif isinstance(p0_rv, stats.rv_continuous):
                p0_frozen = p0_rv(**p0_params)
                self.p0_arr = p0_frozen.rvs(self.num_days)
            else:
                # raise ValueError("Unknown p0_rv")
                # TODO: make a sane default, same with p0_params
                self.p0_arr = np.repeat(self.p0_default, self.num_days)

        # Create the 3D stack of turbulence
        print(f"{self.p0_arr[:5] = }")
        sar_stack = turbulence.simulate(
            stack_shape,
            beta=self.beta,
            p0=self.p0_arr,
            freq0=self.freq0,
            resolution=self.resolution,
            distribution=self.distribution,
            seed=seed,
        )
        if self.to_cm:
            sar_stack *= 100

        self.sar_stack = sar_stack
        sar_date_list = self.make_sar_dates()
        self.sar_date_list = sar_date_list
        return sar_stack, sar_date_list

    def make_defo_stack(self, defo_shape="gaussian", **kwargs):
        """Create the time series of deformation to add to each SAR date

        Parameters
        ----------
        defo_shape : str
            Name of a function from `troposim.synthetic`. Defaults to "gaussian".

        Returns
        -------
        defo_stack : np.ndarray (3D)

        """
        from .deformation import synthetic

        # out = np.zeros_like(self.sar_stack)
        # Get the final cumulative deformation
        try:
            defo_func = getattr(synthetic, defo_shape)
        except AttributeError:
            raise ValueError(f"{defo_shape} is not a valid deformation shape")

        # Get shape of deformation in final form (normalized to 1 max)
        final_defo = defo_func(shape=self.shape, **kwargs).reshape((1, *self.shape))
        final_defo /= np.max(final_defo)
        # Broadcast this shape with piecewise linear evolution
        time_evolution = self._make_time_evolution(
            self.defo_rates, smooth=self.smooth_defo_days
        )
        self.defo_stack = final_defo * time_evolution

        return self.defo_stack

    def make_igram_stack(
        self,
        save_ext=None,
        independent=True,
        max_date=None,
        max_date_idx=None,
        max_temporal_baseline=5000,
        sar_stack=None,
        **sar_stack_kwargs,
    ):
        """

        Parameters
        ----------
        save_ext :
             (Default value = None)
        independent :
             (Default value = True)
        max_date :
             (Default value = None)
        max_date_idx :
             (Default value = None)
        max_temporal_baseline :
             (Default value = 5000)
        sar_stack :
             (Default value = None)
        **sar_stack_kwargs :


        Returns
        -------
        ndarray
            3D stack of interferograms
        """
        if sar_stack is None:
            if self.sar_stack is None:
                print("Creating sar_stack")
                self.make_sar_stack(**sar_stack_kwargs)
            sar_stack = self.sar_stack

        if max_date is not None:
            max_date_idx = self.sar_date_list.index(max_date)
        if max_date_idx is None:
            max_date_idx = None

        if independent:
            igram_date_list = self._select_independent(
                self.sar_date_list[:max_date_idx]
            )
        else:
            igram_date_list = self._select_redundant(self.sar_date_list[:max_date_idx])
        # igram_fname_list = []  # list of strings

        igram_stack = []  # list of arrays
        temporal_baselines = []
        for early_date, late_date in igram_date_list:
            if (late_date - early_date).days > max_temporal_baseline:
                continue
            # Using simulated SAR dates to form igrams:
            early_idx = self.sar_date_list.index(early_date)
            late_idx = self.sar_date_list.index(late_date)
            early, late = self.sar_stack[early_idx], self.sar_stack[late_idx]

            # igram = np.abs(late) - np.abs(early) . # Why abs val??
            igram = late - early
            igram_stack.append(igram)
            temporal_baselines.append((late_date - early_date).days)
            if save_ext:
                fname = "{}_{}{}".format(
                    early_date.strftime("%Y%m%d"),
                    late_date.strftime("%Y%m%d"),
                    save_ext,
                )
                igram.tofile(fname)
                # igram_fname_list.append(fname)

        igram_stack = np.array(igram_stack)
        self.igram_date_list = igram_date_list
        self.temporal_baselines = np.array(temporal_baselines)
        return igram_stack

    def subtract_reference(self, igram_stack, ref=None):
        """Subtract a reference from the interferogram stack"""
        if ref is None:
            ref = self.ref
        # Add nones to keep 3D
        ref_points = igram_stack[:, ref[0], ref[1]][:, None, None]
        return igram_stack - ref_points

    def _make_time_evolution(self, rates, smooth):
        """Create a 1d piecewise linear deformation time evolution
        Reshapes into (1, 1, N) array to broadcast to defo stack
        """
        if np.isscalar(rates):
            rates = [rates]
        rate_idxs = np.linspace(
            0, len(rates), len(self.sar_date_list), endpoint=False
        ).astype(int)
        rates_intp = np.array(rates)[rate_idxs]
        diffs = [d.days for d in np.diff(self.sar_date_list)]

        time_evolution = [0]
        for days, rate in zip(diffs, rates_intp[1:]):
            time_evolution.append(time_evolution[-1] + days * (rate / 365.25))
        time_evolution = np.array(time_evolution)

        if smooth:
            time_evolution = np.convolve(
                time_evolution, np.ones(smooth) / smooth, mode="same"
            )
        return time_evolution.reshape((-1, 1, 1))

    def _select_independent(self, sar_date_list, mid_date=None, num_ifg=None):
        """Choose a list of `num_ifg` independent interferograms spanning `mid_date`

        Parameters
        ----------
        sar_date_list : list[datetime]

        mid_date : datetime, optional
             (Default value = None)
        num_ifg : int, optional
             (Default value = None)

        Returns
        -------
        stack_ifgs : list[(datetime, datetime)]
        """

        if mid_date is not None:
            insert_idx = np.searchsorted(sar_date_list, mid_date)
        else:
            insert_idx = len(sar_date_list) // 2
        num_ifg = num_ifg or len(sar_date_list) - insert_idx

        # Since `mid_date` will fit in the sorted array at `insert_idx`, then
        # sar_date_list[insert_idx] is the first date AFTER the event
        start_idx = np.clip(insert_idx - num_ifg, 0, None)
        end_idx = insert_idx + num_ifg
        date_subset = sar_date_list[start_idx:end_idx]

        stack_ifgs = list(zip(date_subset[:num_ifg], date_subset[num_ifg:]))
        return stack_ifgs

    # TODO: any reason not to do this like Yijue's thesis?
    def _select_redundant(self, sar_date_list):
        """Form all possible interferogram pairs from the date list"""
        return [tuple(pair) for pair in combinations(sar_date_list, 2)]


def create_igrams(num_days=50):
    igm = IgramMaker(num_days=num_days)
    _, sar_date_list = igm.make_sar_stack(seed=None)
    igram_stack, igram_date_list = igm.make_igram_stack()
    return sar_date_list, igram_date_list, igram_stack


def stack_over_time(
    sar_date_list,
    igram_date_list,
    igram_stack,
    start_idx=5,
    skip=2,
    ntotal=9,
    verbose=False,
):
    """Create a series of stacking solutions with progressively longer time window

    Can be plotted with plotting.plot_stack_over_time

    Parameters
    ----------
    sar_date_list : list[datetime]

    igram_date_list : list[tuple(datetime, datetime)]

    igram_stack : np.ndarray

    start_idx : int, optional
         (Default value = 5)
    skip : int, optional
         (Default value = 2)
    ntotal : int, optional
         (Default value = 9)
    verbose : bool, optional
         (Default value = False)

    Returns
    -------
    defos : list[np.ndarray]
    last_dates : list[datetime]
    num_ifgs : list[int]
    num_sar : list[int]
    """

    defos, last_dates, num_ifgs, num_sar = [], [], [], []

    for idx in range(ntotal):
        last_date = sar_date_list[skip * idx + start_idx]
        valid_sar, valid_igrams, valid_idxs = utils.find_valid(
            sar_date_list, igram_date_list, max_date=last_date, verbose=verbose
        )
        phase_sum = np.sum(igram_stack[valid_idxs], axis=0)  # units: cm
        timediffs = [utils.temporal_baseline(ig) for ig in valid_igrams]  # units: days
        avg_velo = phase_sum / np.sum(timediffs)  # cm / day
        # vv = P2MM * avg_velo  # MM / year
        cum_defo = avg_velo * np.max(timediffs)  # cm, cumulative over current interval

        last_dates.append(last_date)
        num_ifgs.append(len(valid_igrams))
        num_sar.append(len(valid_sar))
        defos.append(cum_defo)

    defos = np.stack(defos, axis=0)
    return defos, last_dates, np.array(num_ifgs), np.array(num_sar)
