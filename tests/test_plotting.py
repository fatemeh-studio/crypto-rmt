"""Headless smoke tests for the :mod:`crypto_rmt.plotting` figure layer.

These tests only confirm that each plotting function draws without error and
returns the expected object type on small synthetic inputs; they do not assert
on pixel content. The Agg backend is selected at import so the suite runs
without a display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from crypto_rmt.clustering import linkage_from_correlation
from crypto_rmt.plotting import (
    plot_cluster_map,
    plot_participation,
    plot_spectrum_vs_null,
)


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
