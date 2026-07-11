"""Headless smoke tests for the :mod:`crypto_rmt.plotting` figure layer.

These tests only confirm that each plotting function draws without error and
returns the expected object type on small synthetic inputs; they do not assert
on pixel content. The Agg backend is selected at import so the suite runs
without a display.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from crypto_rmt.clustering import linkage_from_correlation
from crypto_rmt.dynamics import EVENTS
from crypto_rmt.plotting import (
    plot_cluster_map,
    plot_collectivity,
    plot_participation,
    plot_regime_panel,
    plot_spectrum_vs_mp,
    plot_spectrum_vs_null,
)


def _synthetic_rolling_series(
    n: int = 150,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(end_timestamps, collectivity, effective_rank)`` over 2021-2024.

    The window spans the annotated events so the event markers fall inside the
    plot; values are arbitrary but lie in their valid ranges.
    """
    start = int(datetime(2021, 6, 1, tzinfo=timezone.utc).timestamp())
    week = 7 * 24 * 3600
    end_ts = np.array([start + i * week for i in range(n)], dtype=np.int64)
    rng = np.random.default_rng(0)
    collectivity = rng.uniform(0.6, 0.9, size=n)
    effective_rank = rng.uniform(2.0, 8.0, size=n)
    return end_ts, collectivity, effective_rank


def test_plot_spectrum_vs_null_returns_axes() -> None:
    """The spectrum plot returns a matplotlib Axes on synthetic arrays."""
    eigenvalues = np.array([4.0, 1.2, 0.6, 0.2])
    null_eigenvalues = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 0.4, 0.6])
    threshold = 1.3

    ax = plot_spectrum_vs_null(eigenvalues, null_eigenvalues, threshold)

    assert isinstance(ax, Axes)
    plt.close("all")


def test_plot_participation_returns_axes() -> None:
    """The participation plot returns a matplotlib Axes on synthetic arrays."""
    eigenvalues = np.array([4.0, 1.2, 0.6, 0.2])
    participation = np.array([12.0, 6.0, 3.0, 2.5])

    ax = plot_participation(eigenvalues, participation, threshold=1.3)

    assert isinstance(ax, Axes)
    plt.close("all")


def test_plot_cluster_map_saves_file(tmp_path: Path) -> None:
    """The cluster map returns a savable grid and writes a non-empty file."""
    C = np.array(  # noqa: N806  (correlation matrix, standard symbol)
        [
            [1.0, 0.8, 0.1, 0.0],
            [0.8, 1.0, 0.2, 0.1],
            [0.1, 0.2, 1.0, 0.7],
            [0.0, 0.1, 0.7, 1.0],
        ]
    )
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    Z = linkage_from_correlation(C)  # noqa: N806  (linkage matrix, standard symbol)

    out = tmp_path / "cluster_map.png"
    g = plot_cluster_map(C, Z, tickers, save=out)

    assert hasattr(g, "savefig")
    assert out.exists()
    assert out.stat().st_size > 0
    plt.close("all")


def test_plot_collectivity_returns_axes() -> None:
    """The collectivity plot returns a matplotlib Axes with events + peak."""
    end_ts, collectivity, _ = _synthetic_rolling_series()

    ax = plot_collectivity(end_ts, collectivity, events=EVENTS)

    assert isinstance(ax, Axes)
    plt.close("all")


def test_plot_collectivity_bare_options_off() -> None:
    """The collectivity plot also draws with median/peak/events disabled."""
    end_ts, collectivity, _ = _synthetic_rolling_series()

    ax = plot_collectivity(
        end_ts, collectivity, events=None, show_median=False, mark_peak=False
    )

    assert isinstance(ax, Axes)
    plt.close("all")


def test_plot_regime_panel_returns_figure_and_axes(tmp_path: Path) -> None:
    """The regime panel returns (Figure, (Axes, Axes)) and writes a file."""
    end_ts, collectivity, effective_rank = _synthetic_rolling_series()

    out = tmp_path / "regime_panel.png"
    fig, axes = plot_regime_panel(
        end_ts, collectivity, effective_rank, events=EVENTS, save=out
    )

    assert isinstance(fig, Figure)
    assert len(axes) == 2
    assert all(isinstance(a, Axes) for a in axes)
    assert out.exists()
    assert out.stat().st_size > 0
    plt.close("all")


def test_plot_spectrum_vs_mp_returns_axes() -> None:
    """The MP spectrum plot returns an Axes; eigenvalues span bulk + a market mode."""
    eigenvalues = np.array([11.3, 1.4, 1.1, 1.0, 0.9, 0.6, 0.3])
    ax = plot_spectrum_vs_mp(eigenvalues, n=eigenvalues.size, t=5000)
    assert isinstance(ax, Axes)
    plt.close("all")
    