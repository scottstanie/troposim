try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not available, skipping plotting")
import numpy as np

DATE_FMT = "%Y%m%d"
CMAP = "RdBu_r"


def plot_stack(stack, titles=None, ntotal=9, cmap=CMAP, figsize=(8, 8)):
    """Plot a portion of a stack of images in a facet grid

    Parameters
    ----------
    stack : ndarray
        3D array of images
    titles : list of str
        (Default value = None)
    ntotal : int
        (Default value = 9)
    cmap :
        (Default value = CMAP)
    figsize : tuple[int, int]
        (Default value = (8)
    """
    nrow = ncol = int(np.sqrt(ntotal))
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    vmin = np.min(stack)
    vmax = np.max(stack)
    for idx, ax in enumerate(axes.ravel()):
        axim = ax.imshow(stack[idx], cmap=cmap, vmax=vmax, vmin=vmin)
        fig.colorbar(axim, ax=ax)
        if titles is not None:
            ax.set_title(titles[idx])


def plot_psd(
    data,
    freq=None,
    psd1d=None,
    freq0=1e-4,
    resolution=None,
    axes=None,
    outfig=None,
    cmap=CMAP,
    color="k",
    label=None,
    marker=None,
    in_mm=False,
    per_km=True,
    slopes=[],
    vm=None,
):
    """Plot the result of `turbulence.get_psd`

    Parameters
    ----------
    data : ndarray
        Noise image to plot the PSD of
    freq : ndarray, optional
        Frequecny array to use for plot
        if not passed, will be generated from the data
    psd1d : ndarray
        PSD to use for plot
        if not passed, will be generated from the data
    freq0 : float, default value = 1e-4
        reference frequency line to plot, showing p0
    resolution : float, optional
        Resolution of data (if not passing freq + psd1d)
    axes : matplotlib.axes.Axes
    outfig : str
        path to save the figure to
    cmap : str, default value = RdBu_r
    color : str, default value = "k"
        Line color for plot
    label :
        (Default value = None)
    marker :
        (Default value = None)
    in_mm :
        (Default value = False)
    per_km :
        (Default value = True)
    slopes :
        (Default value = [])
    vm :
        (Default value = None)

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    axes : matplotlib.axes.Axes
        Axes object
    """
    if freq is None or psd1d is None:
        from . import turbulence

        if resolution is None:
            raise ValueError("resolution must be specified if freq and psd1d are not")

        _, _, freq, psd1d = turbulence.get_psd(
            data,
            resolution=resolution,
            freq0=freq0,
        )

    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[10, 3])
    else:
        ax = axes[0]
        fig = ax.get_figure()

    if vm is None:
        vm = np.nanmax(np.abs(data * 100))
    axim = axes[0].imshow(data * 100, cmap=cmap, vmin=-vm, vmax=vm)
    axes[0].set_axis_off()
    cbar = fig.colorbar(axim, ax=axes[0])
    cbar.set_label("cm")

    plot_psd1d(
        freq,
        psd1d,
        ax=axes[1],
        freq0=freq0,
        label=label,
        color=color,
        marker=marker,
        per_km=per_km,
        in_mm=in_mm,
        slopes=slopes,
    )

    fig.tight_layout()

    if outfig:
        fig.savefig(outfig, bbox_inches="tight", transparent=True, dpi=300)
        print("save figure to", outfig)
    return fig, axes
    # plt.show()


def plot_psd1d(
    freq,
    psd1d,
    ax=None,
    freq0=None,
    label=None,
    color="k",
    linestyle="-",
    marker=None,
    per_km=True,
    in_mm=False,
    slopes=[],
    **plotkwargs,
):
    if ax is None:
        fig, ax = plt.subplots()

    # denom: go from dividing by (1/m)**2 to (1/km)**2 -> (1000 m/1km)**2
    denom_scale = 1e6 if per_km else 1
    numer_scale = 1e4  # from (meters**2 to cm**2)
    if in_mm:
        # extra (10^2) for cm->mm
        numer_scale *= 100
    freq_plot = freq * (1e3 if per_km else 1)
    ax.loglog(
        freq_plot,
        psd1d * numer_scale / denom_scale,
        label=label,
        color=color,
        linestyle=linestyle,
        marker=marker,
        **plotkwargs,
    )
    if freq0 is not None:
        # vertical line marking reference frequency
        ax.axvline(freq0 * 1e3, linestyle="--", color="k")

    # ax.set_xlabel("Wavenumber [cycle/km]")
    ax.set_xlabel(r"$k$ [cycle/km]")
    ylabel = "PSD "
    ylabel += r"$[cm^2 / (1/m^2)]$"
    if per_km:
        ylabel = ylabel.replace("/m", "/km")
    if in_mm:
        ylabel = ylabel.replace("cm", "mm")
    ax.set_ylabel(ylabel)
    # Plot the slopes, if passed, to compare to the PSD
    for slope in slopes:
        lines = freq ** slope
        lines *= numer_scale / denom_scale * (psd1d[0] / lines[0])
        ax.loglog(freq_plot, lines, color="k", linestyle="--")
    return ax


def plot_stack_over_time(
    avgs,
    sar_dates_used,
    vm=None,
    igram_stack=None,  # Just for colorbar
    cmap="seismic_wide_y",
):
    if vm is None:
        if igram_stack is not None:
            vm = np.nanpercentile(np.nanmax(np.abs(igram_stack)), 80)
        else:
            vm = 2

    ntotal = len(avgs)
    nrow = ncol = int(np.ceil(np.sqrt(ntotal)))
    fig, axes = plt.subplots(nrow, ncol)

    for vv, last_date, ax in zip(avgs, sar_dates_used, axes.ravel()):
        axim = ax.imshow(vv, vmax=vm, vmin=-vm, cmap=cmap)
        ax.set_title(last_date.strftime(DATE_FMT))
        ax.set_axis_off()
        fig.colorbar(axim, ax=ax)

    return fig, axes


def plot_psd1d_azimuth(
    angles,
    psd1d,
    ax=None,
    label=None,
    ylim=(0, 1),
    color="k",
    # linestyle="-",
    marker="o",
    **plotkwargs,
):
    if not ax:
        fig, ax = plt.subplots()

    ax.plot(
        angles,
        psd1d,
        label=label,
        color=color,
        # linestyle=linestyle,
        marker=marker,
        **plotkwargs,
    )
    from . import utils

    a1 = utils.anisotropy_var(psd1d)
    a2 = utils.anisotropy_A(psd1d)
    title = f"A = {a2:.2f}, var = {a1:.2g}"
    ax.set_title(title)

    ax.set_ylim(ylim)
    ax.set_xlabel("Angular bin [degrees]")
    ax.set_ylabel("Average Power")
