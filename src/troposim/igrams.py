from dataclasses import dataclass
from datetime import date, timedelta
from itertools import combinations
from typing import Optional, Tuple

import numpy as np

from . import turbulence, utils
from .log import get_log


logger = get_log(__name__)


@dataclass
class IgramMaker:
    """Class for creating synthetic interferograms"""

    psd_stack: turbulence.PsdStack
    num_days: int = 10
    shape: Optional[Tuple[int, int]] = None
    randomize: bool = True
    # resolution: int = 400
    # p0_default: float = 10.0
    # freq0: float = 1e-4
    start_date: date = date(2018, 1, 1)
    repeat_interval_days: int = 12
    sar_stack: np.ndarray | None = None
    to_cm: bool = True
    defo_rates: Tuple[float] = (0.0,)
    defo_sigma = 5
    smooth_defo_days: int = 2
    # num_bad_days = 5
    # bad_day_mult = 8
    ref: Tuple[int, int] = (5, 5)

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
        seed=None,
    ):
        # Create the 3D stack of turbulence
        sar_stack = self.psd_stack.simulate(
            num_days=self.num_days,
            shape=self.shape,
            randomize=self.randomize,
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
        """Create the interferogram stack from the SAR stack.

        Parameters
        ----------
        save_ext : str
            Extension to save the igram stack to. Defaults to None.
        independent : bool
            Whether to select independent or redundant dates. Defaults to True.
        max_date : str
            The maximum date to use in the stack. Defaults to None.
        max_date_idx : int
            The maximum date index to use in the stack. Defaults to None.
        max_temporal_baseline : int
            The maximum temporal baseline (in days) to use in the stack. Defaults to 5000.
        sar_stack : np.ndarray
            The SAR stack to use. Defaults to None.
        sar_stack_kwargs : dict
            Keyword arguments to pass to `make_sar_stack` if `sar_stack` is None.

        Returns
        -------
        igram_stack : np.ndarray (3D)
            The interferogram stack.
        """
        if sar_stack is None:
            if self.sar_stack is None:
                logger.debug("Creating sar_stack")
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
        selected_dates = []
        for early_date, late_date in igram_date_list:
            if (late_date - early_date).days > max_temporal_baseline:
                continue
            # Using simulated SAR dates to form igrams:
            early_idx = self.sar_date_list.index(early_date)
            late_idx = self.sar_date_list.index(late_date)
            early, late = self.sar_stack[early_idx], self.sar_stack[late_idx]

            selected_dates.append((early_date, late_date))
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
        self.igram_date_list = selected_dates
        self.temporal_baselines = np.array(temporal_baselines)
        return igram_stack

    def subtract_reference(self, igram_stack, ref=None):
        """Subtract a reference from the interferogram stack"""
        if ref is None:
            ref = self.ref
        # Add nones to keep 3D
        ref_points = igram_stack[:, ref[0], ref[1]]
        if ref_points.ndim == 1:
            ref_points = ref_points[:, None, None]
        else:
            ref_points = ref_points.mean(axis=(1, 2), keepdims=True)
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
        list of simulated SAR dates
    igram_date_list : list[tuple(datetime, datetime)]
        list of simulated interferogram dates
    igram_stack : np.ndarray
        simulated interferogram stack
    start_idx : int, optional, Default value = 5
        Index of first date to use in the stack
    skip : int, optional, Default value = 2
        Number of dates to skip between stacks
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
