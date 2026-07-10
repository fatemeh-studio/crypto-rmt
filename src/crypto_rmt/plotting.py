"""Figure rendering for the crypto-RMT analysis.

Every function in this module is a pure *drawing* routine: it receives arrays
already computed by :mod:`crypto_rmt.rmt` and :mod:`crypto_rmt.clustering` and
turns them into a figure. No correlation, eigenvalue, linkage, or other analysis
is performed here -- that logic lives in the analysis modules.

Notes
-----
The shared axis/save boilerplate (create a fresh figure when no axes are
supplied, save on request) is centralized in :func:`_prepare_axes` and
:func:`_finalize` so the individual plotting functions stay focused on what they
draw.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = [
    "plot_spectrum_vs_null",
    "plot_participation",
    "plot_cluster_map",
]

#: Default number of histogram bins for the shuffled-null spectrum.
_NULL_HIST_BINS: int = 60


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
        If given, points with eigenvalue above the threshold are colored
        differently from those below it.
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
    The largest-eigenvalue point (the market mode) is highlighted. The x-axis
    uses a logarithmic scale so the localized small-eigenvalue modes remain
    legible alongside the market mode.
    """
    fig, ax = _prepare_axes(ax)

    eigenvalues = np.asarray(eigenvalues, dtype=float)
    participation = np.asarray(participation, dtype=float)

    if threshold is None:
        ax.scatter(eigenvalues, participation, color="C0", label="modes")
    else:
        above = eigenvalues > threshold
        ax.scatter(
            eigenvalues[~above],
            participation[~above],
            color="0.6",
            label="within null band",
        )
        ax.scatter(
            eigenvalues[above],
            participation[above],
            color="C0",
            label="above null threshold",
        )
        ax.axvline(threshold, color="C3", linestyle="--", linewidth=1.0)

    market_mode = int(np.argmax(eigenvalues))
    ax.scatter(
        eigenvalues[market_mode],
        participation[market_mode],
        color="C1",
        marker="*",
        s=200,
        zorder=5,
        label="market mode",
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
