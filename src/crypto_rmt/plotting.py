"""Figure rendering for the crypto-RMT analysis.

Every function in this module is a pure *drawing* routine: it receives arrays
already computed by :mod:`crypto_rmt.rmt`, :mod:`crypto_rmt.clustering`, and
:mod:`crypto_rmt.dynamics` and turns them into a figure. No correlation,
eigenvalue, linkage, or rolling-window analysis is performed here -- that logic
lives in the analysis modules.

The figures fall into two groups:

* **Static** -- ``plot_spectrum_vs_null``, ``plot_participation``,
  ``plot_cluster_map`` describe the correlation structure of a single window.
* **Temporal** -- ``plot_collectivity`` and ``plot_regime_panel`` trace how that
  structure evolves through time and annotate known regime events, turning the
  RMT picture into a regime-detection story.

Notes
-----
The shared axis/save boilerplate (create a fresh figure when no axes are
supplied, save on request) is centralized in :func:`_prepare_axes` and
:func:`_finalize`; date-axis formatting and event annotation are centralized in
:func:`_format_date_axis` and :func:`_mark_events`, so the individual plotting
functions stay focused on what they draw.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.transforms import offset_copy

__all__ = [
    "plot_spectrum_vs_mp",
    "plot_spectrum_vs_null",
    "plot_participation",
    "plot_cluster_map",
    "plot_collectivity",
    "plot_regime_panel",
]

#: Default number of histogram bins for the shuffled-null spectrum.
_NULL_HIST_BINS: int = 60

#: Colour of the market-collectivity trace.
_COLLECTIVITY_COLOR: str = "C0"
#: Colour of the effective-rank trace.
_EFFECTIVE_RANK_COLOR: str = "C1"
#: Colour used to highlight the collectivity peak.
_PEAK_COLOR: str = "C3"
#: Shortest allowed peak-label leader line, as a fraction of the axes diagonal.
_PEAK_LABEL_MIN_ARROW: float = 0.07
#: Colour of the dashed event markers.
_EVENT_COLOR: str = "0.5"


def _prepare_axes(ax: Axes | None) -> tuple[Figure, Axes]:
    """Return a figure/axes pair, creating them when none is supplied.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        An existing axes to draw on, or ``None`` to create a new figure/axes.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure owning ``ax``.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    return fig, ax


def _finalize(fig: Figure, save: str | Path | None) -> None:
    """Save the figure when a path is given.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    save : str or pathlib.Path or None
        Destination path, or ``None`` to skip saving.
    """
    if save is not None:
        fig.savefig(save, bbox_inches="tight")


def _to_datetimes(timestamps: npt.NDArray[np.int64]) -> pd.DatetimeIndex:
    """Convert unix-second timestamps to a pandas ``DatetimeIndex``.

    Parameters
    ----------
    timestamps : numpy.ndarray
        ``int64`` unix-second timestamps (as returned by the dynamics layer).

    Returns
    -------
    pandas.DatetimeIndex
        The timestamps as datetimes, suitable for a matplotlib date axis.
    """
    return pd.to_datetime(np.asarray(timestamps, dtype=np.int64), unit="s")


def _format_date_axis(ax: Axes) -> None:
    """Apply an auto date locator and a concise date formatter to ``ax``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes whose x-axis holds datetimes.
    """
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


#: Half-width (in samples) of the window used to gauge local curve height when
#: deciding whether an event label belongs at the top or bottom of the axes.
_EVENT_LABEL_WINDOW: int = 4

#: Horizontal nudge (in points) so event labels sit just clear of their dashed
#: line rather than overlapping it.
_EVENT_LABEL_XOFFSET_PT: float = 3.0


def _event_label_placement(
    dates: pd.DatetimeIndex,
    values: npt.NDArray[np.float64],
    when: pd.Timestamp,
    *,
    window: int = _EVENT_LABEL_WINDOW,
) -> tuple[float, str]:
    """Pick a top/bottom anchor for an event label from the local curve height.

    The label is a vertical caption at ``when``; it should sit in whichever of
    the two horizontal bands (above or below the trace) has more empty room near
    that event, so it does not cross the plotted curve.

    Parameters
    ----------
    dates : pandas.DatetimeIndex
        The x positions of the plotted trace.
    values : numpy.ndarray
        The plotted trace values, aligned with ``dates``.
    when : pandas.Timestamp
        The event date the label annotates.
    window : int, optional
        Half-width, in samples, of the neighbourhood around the event used to
        measure the local curve extent. Defaults to :data:`_EVENT_LABEL_WINDOW`.

    Returns
    -------
    y : float
        The y anchor in axes-fraction coordinates (near ``1`` for the top band,
        near ``0`` for the bottom band).
    va : str
        The matching vertical alignment (``"top"`` or ``"bottom"``).
    """
    idx = int(np.searchsorted(np.asarray(dates), np.datetime64(when)))
    idx = min(max(idx, 0), values.size - 1)
    lo = max(0, idx - window)
    hi = min(values.size, idx + window + 1)
    local = values[lo:hi]
    vmin, vmax = float(values.min()), float(values.max())
    span = (vmax - vmin) or 1.0
    gap_top = (vmax - float(local.max())) / span
    gap_bottom = (float(local.min()) - vmin) / span
    if gap_top >= gap_bottom:
        return 0.98, "top"
    return 0.02, "bottom"


def _mark_events(
    ax: Axes,
    events: Mapping[str, str],
    *,
    label: bool = True,
    dates: pd.DatetimeIndex | None = None,
    values: npt.NDArray[np.float64] | None = None,
) -> None:
    """Draw dashed vertical lines at event dates, optionally labelled.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes with a datetime x-axis to annotate.
    events : mapping of str to str
        Mapping from event name to ``YYYY-MM-DD`` date (e.g.
        :data:`crypto_rmt.dynamics.EVENTS`).
    label : bool, optional
        If ``True``, write each event name vertically at the top or bottom of
        the axes. Set ``False`` to draw the lines without labels (e.g. on a
        lower panel that shares its labels with the panel above). Defaults to
        ``True``.
    dates : pandas.DatetimeIndex, optional
        The x positions of the plotted trace. When given together with
        ``values``, each label is anchored to whichever band (top or bottom)
        has more clearance near its event, so captions avoid the curve.
    values : numpy.ndarray, optional
        The plotted trace values aligned with ``dates`` (see ``dates``).

    Notes
    -----
    Labels are placed with a blended transform (``x`` in data coordinates, ``y``
    in axes coordinates). Without a trace (``dates``/``values``), labels default
    to the top of the axes; with one, their band is chosen per event from the
    local curve height via :func:`_event_label_placement`.
    """
    values_arr = np.asarray(values, dtype=float) if values is not None else None
    label_transform = offset_copy(
        ax.get_xaxis_transform(),
        fig=ax.figure,
        x=_EVENT_LABEL_XOFFSET_PT,
        y=0,
        units="points",
    )
    for name, date in events.items():
        when = pd.to_datetime(date)
        ax.axvline(when, color=_EVENT_COLOR, linestyle="--", linewidth=1.0, zorder=1)
        if not label:
            continue
        y, va = 0.98, "top"
        if values_arr is not None and values_arr.size and dates is not None:
            y, va = _event_label_placement(dates, values_arr, when)
        ax.text(
            when,
            y,
            name,
            transform=label_transform,
            rotation=90,
            va=va,
            ha="left",
            fontsize=8,
            color="0.35",
            zorder=6,
        )


def _label_box_fraction(ax: Axes, text: str, fontsize: float) -> tuple[float, float]:
    """Estimate a text label's ``(width, height)`` as a fraction of the axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes the label will be drawn on.
    text : str
        The label string (its character count sets the width estimate).
    fontsize : float
        Font size in points.

    Returns
    -------
    tuple of float
        ``(width, height)`` of the label's bounding box in axes-fraction units.

    Notes
    -----
    Uses the figure geometry rather than a renderer so it works before the
    figure is drawn. Character metrics are approximate (~0.6 em wide glyphs, a
    1.3 em line) with padding to match the annotation ``bbox``.
    """
    fig = ax.figure
    fig_w_in, fig_h_in = fig.get_size_inches()
    pos = ax.get_position()
    ax_w_pt = max(pos.width * fig_w_in * 72.0, 1.0)
    ax_h_pt = max(pos.height * fig_h_in * 72.0, 1.0)
    pad_pt = 0.6 * fontsize
    width_pt = 0.6 * fontsize * len(text) + 2 * pad_pt
    height_pt = 1.3 * fontsize + 2 * pad_pt
    return width_pt / ax_w_pt, height_pt / ax_h_pt


def _clearest_label_xy(
    ax: Axes,
    dates: pd.DatetimeIndex,
    values: npt.NDArray[np.float64],
    anchor: tuple[float, float],
    box: tuple[float, float],
) -> tuple[float, float]:
    """Find where to put a label so it avoids the data yet stays near ``anchor``.

    Scans a grid of candidate centres (in axes-fraction coordinates) and picks
    the one whose label box overlaps the fewest data points, breaking ties by
    proximity to ``anchor`` so the leader line stays short. This is the
    ``ggrepel``/``adjustText`` idea (bounding-box overlap minimisation) reduced
    to a single label placed against a single trace.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes holding the trace; supplies the data limits.
    dates : pandas.DatetimeIndex
        The x positions of the trace.
    values : numpy.ndarray
        The trace values aligned with ``dates``.
    anchor : tuple of float
        ``(x, y)`` of the labelled point in *data* coordinates; the label is
        pulled toward it.
    box : tuple of float
        Label ``(width, height)`` in axes-fraction units (see
        :func:`_label_box_fraction`).

    Returns
    -------
    tuple of float
        ``(x, y)`` for the label centre in axes-fraction coordinates.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xnum = mdates.date2num(dates)
    fx = (np.asarray(xnum, dtype=float) - x0) / (x1 - x0)
    fy = (values - y0) / (y1 - y0)
    ax_num = mdates.date2num(pd.Timestamp(anchor[0]))
    ap = ((ax_num - x0) / (x1 - x0), (anchor[1] - y0) / (y1 - y0))

    bw, bh = box
    half_w, half_h = bw / 2.0, bh / 2.0
    margin = 0.01
    lo_x, hi_x = half_w + margin, 1.0 - half_w - margin
    lo_y, hi_y = half_h + margin, 1.0 - half_h - margin
    if hi_x <= lo_x or hi_y <= lo_y:
        return ap  # Label bigger than axes; nothing sensible to do.

    cxs = np.linspace(lo_x, hi_x, 25)
    cys = np.linspace(lo_y, hi_y, 19)
    best_xy = (float(ap[0]), float(ap[1]))
    best_score = np.inf
    for cy in cys:
        in_y = np.abs(fy - cy) <= half_h
        for cx in cxs:
            dist = float(np.hypot(cx - ap[0], cy - ap[1]))
            if dist < _PEAK_LABEL_MIN_ARROW:
                continue
            overlaps = int(np.count_nonzero(in_y & (np.abs(fx - cx) <= half_w)))
            score = overlaps * 100.0 + dist
            if score < best_score:
                best_score = score
                best_xy = (float(cx), float(cy))
    return best_xy


def _mark_peak(
    ax: Axes, dates: pd.DatetimeIndex, values: npt.NDArray[np.float64]
) -> None:
    """Highlight the maximum of ``values`` with a marker and arrow annotation.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to annotate.
    dates : pandas.DatetimeIndex
        The x positions aligned with ``values``.
    values : numpy.ndarray
        The plotted series; its ``argmax`` is highlighted.

    Notes
    -----
    Uses the conventional maximum-annotation style: an unfilled marker on the
    exact peak plus a label whose arrow points back at it (matplotlib's
    ``annotate`` + ``arrowprops`` idiom), matching the arrow-annotated
    highlights used by the other plots in this module.

    The label position is chosen automatically: :func:`_clearest_label_xy`
    searches for the spot nearest the peak whose text box does not overlap the
    data trace (the ``ggrepel``/``adjustText`` bounding-box approach), so the
    callout never sits on top of the series.
    """
    if values.size == 0:
        return
    peak = int(np.argmax(values))
    peak_val = float(values[peak])
    ax.plot(
        dates[peak],
        peak_val,
        marker="o",
        markerfacecolor="none",
        markeredgecolor=_PEAK_COLOR,
        markeredgewidth=1.5,
        markersize=8,
        zorder=5,
    )
    fontsize = 9.0
    text = f"peak {peak_val:.2f}"
    box = _label_box_fraction(ax, text, fontsize)
    lx, ly = _clearest_label_xy(ax, dates, values, (dates[peak], peak_val), box)
    ax.annotate(
        text,
        xy=(dates[peak], peak_val),
        xycoords="data",
        xytext=(lx, ly),
        textcoords=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=_PEAK_COLOR,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": _PEAK_COLOR,
            "linewidth": 1.0,
            "alpha": 0.9,
        },
        arrowprops={"arrowstyle": "->", "color": _PEAK_COLOR, "lw": 1.5},
        zorder=6,
    )


def plot_spectrum_vs_null(
    eigenvalues: npt.NDArray[np.float64],
    null_eigenvalues: npt.NDArray[np.float64],
    threshold: float,
    *,
    ax: Axes | None = None,
    save: str | Path | None = None,
) -> Axes:
    """Plot the empirical eigenvalues against the shuffled-null noise band.

    Parameters
    ----------
    eigenvalues : numpy.ndarray
        The real correlation-matrix eigenvalues (e.g. from
        :func:`crypto_rmt.rmt.eigsystem`).
    null_eigenvalues : numpy.ndarray
        Pooled eigenvalues of many shuffled correlation matrices (the noise
        band, e.g. from :func:`crypto_rmt.rmt.null_spectrum`).
    threshold : float
        Upper edge of the null band (e.g. from
        :func:`crypto_rmt.rmt.null_threshold`). Eigenvalues to the right of this
        line are genuine signal.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created when omitted.
    save : str or pathlib.Path, optional
        If given, the figure is saved to this path.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.

    Notes
    -----
    The null band is drawn as a density histogram (``density=True``); the real
    eigenvalues are overlaid as vertical stem markers, and the threshold as a
    dashed vertical line.
    """
    fig, ax = _prepare_axes(ax)

    ax.hist(
        null_eigenvalues,
        bins=_NULL_HIST_BINS,
        density=True,
        color="0.7",
        edgecolor="white",
        linewidth=0.3,
        label="shuffled null (noise band)",
    )

    markerline, stemlines, baseline = ax.stem(
        eigenvalues,
        np.full_like(eigenvalues, ax.get_ylim()[1]),
        linefmt="C0-",
        markerfmt="C0o",
        basefmt=" ",
        label="empirical eigenvalues",
    )
    markerline.set_markersize(4)
    stemlines.set_linewidth(1.0)

    ax.axvline(
        threshold,
        color="C3",
        linestyle="--",
        linewidth=1.5,
        label=f"null threshold = {threshold:.3g}",
    )

    ax.set_xlabel("eigenvalue")
    ax.set_ylabel("density")
    ax.set_title("Eigenvalue spectrum vs shuffled null (right of threshold = signal)")
    ax.legend()

    _finalize(fig, save)
    return ax


def plot_spectrum_vs_mp(
    eigenvalues: npt.NDArray[np.float64],
    n: int,
    t: int,
    *,
    ax: Axes | None = None,
    save: str | Path | None = None,
) -> Axes:
    """Plot the empirical eigenvalues against the Marchenko-Pastur noise band.

    Parameters
    ----------
    eigenvalues : numpy.ndarray
        The real correlation-matrix eigenvalues (e.g. from
        :func:`crypto_rmt.rmt.eigsystem`).
    n : int
        Number of assets ``N`` (matrix dimension) used to estimate the
        correlation matrix.
    t : int
        Number of observations ``T`` per asset in that estimation window. Sets
        the Marchenko-Pastur ratio ``N / T`` and hence the width of the noise
        band.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created when omitted.
    save : str or pathlib.Path, optional
        If given, the figure is saved to this path.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.

    Notes
    -----
    With only ``N`` eigenvalues, individual eigenvalues are drawn as stems rather
    than a coarse histogram, and the theoretical MP density
    (:func:`crypto_rmt.rmt.marchenko_pastur_density`) is overlaid as the smooth
    noise curve. Eigenvalues inside ``[lambda_minus, lambda_plus]`` are the noise
    bulk; those above ``lambda_plus`` are genuine signal, and the largest (the
    market mode) is highlighted. The x-axis is logarithmic so the tight noise
    band near ``1`` and a market mode many times larger stay legible together --
    with hourly data ``T`` greatly exceeds ``N``, so the band is narrow and most
    eigenvalues fall outside it.
    """
    from crypto_rmt.rmt import marchenko_pastur_bounds, marchenko_pastur_density

    fig, ax = _prepare_axes(ax)

    eigenvalues = np.asarray(eigenvalues, dtype=float)
    lam_minus, lam_plus = marchenko_pastur_bounds(n, t)

    grid = np.linspace(lam_minus, lam_plus, 400)
    density = marchenko_pastur_density(grid, n, t)
    ax.plot(grid, density, color="0.35", linewidth=1.5, zorder=3, label="MP density")
    ax.fill_between(grid, density, color="0.85", zorder=1)

    stem_height = float(density.max()) if density.size else 1.0
    for edge in (lam_minus, lam_plus):
        ax.axvline(edge, color="0.5", linestyle=":", linewidth=1.0, zorder=2)

    market_mode = int(np.argmax(eigenvalues))
    above = eigenvalues > lam_plus
    above[market_mode] = False  # market mode highlighted separately
    bulk = ~above
    bulk[market_mode] = False

    ax.vlines(
        eigenvalues[bulk],
        0,
        stem_height,
        color="0.6",
        linewidth=1.2,
        zorder=4,
        label=r"bulk ($\lambda \leq \lambda_+$)",
    )
    if above.any():
        ax.vlines(
            eigenvalues[above],
            0,
            stem_height,
            color=_COLLECTIVITY_COLOR,
            linewidth=1.5,
            zorder=4,
            label=r"signal ($\lambda > \lambda_+$)",
        )
    ax.vlines(
        eigenvalues[market_mode],
        0,
        stem_height,
        color=_PEAK_COLOR,
        linewidth=2.0,
        zorder=5,
    )
    # Mark the market mode with an arrow + label, the standard RMT-finance
    # convention for the largest eigenvalue (Laloux-Bouchaud-Potters; Plerou).
    ax.annotate(
        f"market mode\n$\\lambda_{{\\max}}$ = {eigenvalues[market_mode]:.1f}",
        xy=(eigenvalues[market_mode], stem_height * 0.9),
        xytext=(-48, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=_PEAK_COLOR,
        arrowprops={"arrowstyle": "->", "color": _PEAK_COLOR, "lw": 1.5},
        zorder=6,
    )

    n_signal = int((eigenvalues > lam_plus).sum())
    ax.set_xscale("log")
    ax.set_xlabel("eigenvalue")
    ax.set_ylabel("MP density")
    ax.set_title(
        f"Eigenvalue spectrum vs Marchenko-Pastur "
        f"({n_signal} of {eigenvalues.size} above $\\lambda_+$)"
    )
    ax.legend(loc="lower right", bbox_to_anchor=(0.95, 0.02), fontsize=8)

    _finalize(fig, save)
    return ax


def plot_participation(
    eigenvalues: npt.NDArray[np.float64],
    participation: npt.NDArray[np.float64],
    threshold: float | None = None,
    *,
    ax: Axes | None = None,
    save: str | Path | None = None,
) -> Axes:
    """Scatter each mode's participation ratio against its eigenvalue.

    Parameters
    ----------
    eigenvalues : numpy.ndarray
        The correlation-matrix eigenvalues (x-axis, log-scaled).
    participation : numpy.ndarray
        Participation ratio per mode (effective number of assets), e.g. from
        :func:`crypto_rmt.rmt.participation_ratio`. Aligned with
        ``eigenvalues``.
    threshold : float, optional
        Marchenko-Pastur upper edge ``lambda_+``. If given, modes with
        eigenvalue above it (signal) are colored differently from the noise
        bulk below it, and the edge is drawn as a dotted vertical line.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created when omitted.
    save : str or pathlib.Path, optional
        If given, the figure is saved to this path.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.

    Notes
    -----
    The largest-eigenvalue point (the market mode) is highlighted with an arrow
    and label -- the same convention as :func:`plot_spectrum_vs_mp`, not a
    distinct marker. The x-axis uses a logarithmic scale so the localized
    small-eigenvalue modes remain legible alongside the market mode.
    """
    fig, ax = _prepare_axes(ax)

    eigenvalues = np.asarray(eigenvalues, dtype=float)
    participation = np.asarray(participation, dtype=float)
    market_mode = int(np.argmax(eigenvalues))

    if threshold is None:
        rest = np.ones(eigenvalues.size, dtype=bool)
        rest[market_mode] = False
        ax.scatter(eigenvalues[rest], participation[rest], color="0.6", label="modes")
    else:
        above = eigenvalues > threshold
        above[market_mode] = False  # market mode highlighted separately
        bulk = ~above
        bulk[market_mode] = False
        ax.scatter(
            eigenvalues[bulk],
            participation[bulk],
            color="0.6",
            label=r"bulk ($\lambda \leq \lambda_+$)",
        )
        if above.any():
            ax.scatter(
                eigenvalues[above],
                participation[above],
                color=_COLLECTIVITY_COLOR,
                label=r"signal ($\lambda > \lambda_+$)",
            )
        ax.axvline(threshold, color="0.5", linestyle=":", linewidth=1.0)

    # Mark the market mode with the same arrow + label convention as the
    # spectrum figure (Laloux-Bouchaud-Potters; Plerou), not a distinct glyph.
    ax.scatter(
        eigenvalues[market_mode],
        participation[market_mode],
        color=_PEAK_COLOR,
        zorder=5,
    )
    ax.annotate(
        f"market mode\n$\\lambda_{{\\max}}$ = {eigenvalues[market_mode]:.1f}",
        xy=(eigenvalues[market_mode], participation[market_mode]),
        xytext=(-48, -18),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=_PEAK_COLOR,
        arrowprops={"arrowstyle": "->", "color": _PEAK_COLOR, "lw": 1.5},
        zorder=6,
    )

    ax.set_xscale("log")
    ax.set_xlabel("eigenvalue")
    ax.set_ylabel("participation ratio (effective # assets)")
    ax.set_title("Mode participation vs eigenvalue")
    ax.legend()

    _finalize(fig, save)
    return ax


def plot_cluster_map(
    C: npt.NDArray[np.float64],  # noqa: N803  (correlation matrix, standard symbol)
    Z: npt.NDArray[np.float64],  # noqa: N803  (linkage matrix, standard symbol)
    tickers: list[str] | tuple[str, ...],
    *,
    cmap: str = "vlag",
    save: str | Path | None = None,
) -> sns.matrix.ClusterGrid:
    """Draw the correlation cluster map using a precomputed linkage.

    Parameters
    ----------
    C : numpy.ndarray
        Symmetric correlation matrix of shape ``(N, N)``.
    Z : numpy.ndarray
        Precomputed Mantegna linkage of shape ``(N - 1, 4)`` from
        :func:`crypto_rmt.clustering.linkage_from_correlation`. It is passed to
        both the row and column linkage so seaborn does not recompute its own
        (Euclidean) distance on the correlation values.
    tickers : list of str or tuple of str
        Asset labels, ordered to match the rows/columns of ``C``.
    cmap : str, optional
        Diverging colormap name. Defaults to ``"vlag"``.
    save : str or pathlib.Path, optional
        If given, the figure is saved via ``ClusterGrid.savefig``.

    Returns
    -------
    seaborn.matrix.ClusterGrid
        The clustered heatmap grid.

    Notes
    -----
    The color scale is fixed to ``[-1, 1]`` and centered at ``0`` so the
    diverging colormap maps correlation sign to color symmetrically.
    """
    df = pd.DataFrame(C, index=list(tickers), columns=list(tickers))
    g = sns.clustermap(
        df,
        row_linkage=Z,
        col_linkage=Z,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        cbar_kws={"label": "correlation"},
    )
    if save is not None:
        g.savefig(save, bbox_inches="tight")
    return g


def plot_collectivity(
    end_timestamps: npt.NDArray[np.int64],
    collectivity: npt.NDArray[np.float64],
    *,
    events: Mapping[str, str] | None = None,
    show_median: bool = True,
    mark_peak: bool = True,
    ax: Axes | None = None,
    save: str | Path | None = None,
) -> Axes:
    """Plot rolling market collectivity through time (the headline figure).

    Parameters
    ----------
    end_timestamps : numpy.ndarray
        ``int64`` closing timestamp of each rolling window, as returned by
        :func:`crypto_rmt.dynamics.collectivity_series`.
    collectivity : numpy.ndarray
        ``lambda_max / N`` per window (the variance share of the market mode),
        aligned with ``end_timestamps``.
    events : mapping of str to str, optional
        Regime events to annotate as dashed vertical lines, e.g.
        :data:`crypto_rmt.dynamics.EVENTS`. Omitted when ``None``.
    show_median : bool, optional
        If ``True``, draw a dotted horizontal line at the median collectivity as
        a "typical / calm" reference against which the peaks stand out. Defaults
        to ``True``.
    mark_peak : bool, optional
        If ``True``, highlight the maximum collectivity with a marker and
        annotate its value. Defaults to ``True``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created when omitted.
    save : str or pathlib.Path, optional
        If given, the figure is saved to this path.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.

    Notes
    -----
    The median reference and the peak annotation are computed from the passed
    ``collectivity`` array only; no values are assumed or hard-coded.
    """
    fig, ax = _prepare_axes(ax)

    dates = _to_datetimes(end_timestamps)
    collectivity = np.asarray(collectivity, dtype=float)

    ax.plot(dates, collectivity, color=_COLLECTIVITY_COLOR, linewidth=1.5, zorder=2)

    if show_median and collectivity.size:
        median = float(np.median(collectivity))
        ax.axhline(
            median,
            color="0.7",
            linestyle=":",
            linewidth=1.0,
            zorder=1,
            label=f"median = {median:.2f}",
        )

    if events:
        _mark_events(ax, events, label=True, dates=dates, values=collectivity)

    if mark_peak:
        _mark_peak(ax, dates, collectivity)

    ax.set_ylabel(r"collectivity  $\lambda_{\max}/N$")
    ax.set_title("Rolling market collectivity over time")
    _format_date_axis(ax)
    if show_median and collectivity.size:
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    _finalize(fig, save)
    return ax


def plot_regime_panel(
    end_timestamps: npt.NDArray[np.int64],
    collectivity: npt.NDArray[np.float64],
    effective_rank: npt.NDArray[np.float64],
    *,
    events: Mapping[str, str] | None = None,
    mark_peak: bool = True,
    save: str | Path | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Draw the two-panel regime figure: collectivity above, effective rank below.

    Parameters
    ----------
    end_timestamps : numpy.ndarray
        ``int64`` closing timestamp of each rolling window, as returned by
        :func:`crypto_rmt.dynamics.collectivity_series`.
    collectivity : numpy.ndarray
        ``lambda_max / N`` per window (top panel), aligned with
        ``end_timestamps``.
    effective_rank : numpy.ndarray
        Effective number of factors per window (bottom panel), aligned with
        ``end_timestamps`` (from
        :func:`crypto_rmt.dynamics.collectivity_series`).
    events : mapping of str to str, optional
        Regime events to annotate on both panels, e.g.
        :data:`crypto_rmt.dynamics.EVENTS`. Labels are drawn on the top panel
        only. Omitted when ``None``.
    mark_peak : bool, optional
        If ``True``, highlight the collectivity maximum on the top panel.
        Defaults to ``True``.
    save : str or pathlib.Path, optional
        If given, the figure is saved to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The two-panel figure.
    axes : tuple of matplotlib.axes.Axes
        The ``(top, bottom)`` axes for the collectivity and effective-rank
        panels respectively.

    Notes
    -----
    Collectivity and effective rank move inversely: as the market mode absorbs
    more variance (collectivity rises toward ``1``), the number of independent
    factors falls toward ``1``. The shared x-axis and common event markers make
    that anti-correlation legible at a glance.
    """
    dates = _to_datetimes(end_timestamps)
    collectivity = np.asarray(collectivity, dtype=float)
    effective_rank = np.asarray(effective_rank, dtype=float)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(11, 6), layout="constrained"
    )

    ax_top.plot(dates, collectivity, color=_COLLECTIVITY_COLOR, linewidth=1.5, zorder=2)
    if mark_peak:
        _mark_peak(ax_top, dates, collectivity)
    ax_top.set_ylabel(r"collectivity  $\lambda_{\max}/N$")

    ax_bottom.plot(
        dates, effective_rank, color=_EFFECTIVE_RANK_COLOR, linewidth=1.5, zorder=2
    )
    ax_bottom.set_ylabel("effective rank\n(factors)")

    if events:
        _mark_events(ax_top, events, label=True, dates=dates, values=collectivity)
        _mark_events(ax_bottom, events, label=False)

    _format_date_axis(ax_bottom)
    fig.suptitle("Crypto market regimes: collectivity vs. effective factor count")

    _finalize(fig, save)
    return fig, (ax_top, ax_bottom)
