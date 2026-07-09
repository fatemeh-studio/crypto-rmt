"""Tests for the :mod:`crypto_rmt.clustering` hierarchical-clustering layer.

Mixes pure unit tests of the correlation distance with integration tests that
build the real correlation matrix from the shipped dataset via
:mod:`crypto_rmt.io` and :mod:`crypto_rmt.rmt`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from crypto_rmt.clustering import (
    cluster_labels,
    correlation_distance,
    linkage_from_correlation,
)
from crypto_rmt.io import TICKERS, load_prices, returns_matrix
from crypto_rmt.rmt import correlation_matrix

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_correlation_distance_two_by_two() -> None:
    """The 2x2 case matches sqrt(2 * (1 - C)) with a valid metric structure."""
    C = np.array([[1.0, 0.5], [0.5, 1.0]])  # noqa: N806
    D = correlation_distance(C)  # noqa: N806

    expected = np.sqrt(2.0 * (1.0 - 0.5))
    assert D[0, 1] == pytest.approx(expected)
    assert D[1, 0] == pytest.approx(expected)
    assert D[0, 0] == 0.0
    assert D[1, 1] == 0.0
    assert np.allclose(D, D.T)
    assert (D >= 0.0).all()


def test_correlation_distance_all_ones_is_zero() -> None:
    """A fully-correlated matrix yields the all-zero distance matrix."""
    n = 5
    C = np.ones((n, n))  # noqa: N806
    D = correlation_distance(C)  # noqa: N806
    assert np.array_equal(D, np.zeros((n, n)))


def test_linkage_from_correlation_shape() -> None:
    """Linkage on the real data has the canonical (N - 1, 4) shape."""
    prices = load_prices(TICKERS, DATA_DIR)
    returns = returns_matrix(prices, TICKERS)
    C = correlation_matrix(returns)  # noqa: N806
    Z = linkage_from_correlation(C)  # noqa: N806
    assert Z.shape == (len(TICKERS) - 1, 4)


def test_cluster_labels_two_clusters() -> None:
    """Cutting the real-data tree at k=2 yields exactly two distinct labels."""
    prices = load_prices(TICKERS, DATA_DIR)
    returns = returns_matrix(prices, TICKERS)
    C = correlation_matrix(returns)  # noqa: N806
    Z = linkage_from_correlation(C)  # noqa: N806
    labels = cluster_labels(Z, k=2)
    assert labels.shape == (len(TICKERS),)
    assert len(np.unique(labels)) == 2
