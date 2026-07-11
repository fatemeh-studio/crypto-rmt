"""Tests for the :mod:`crypto_rmt.io` data-loading and alignment layer.

Mixes pure unit tests of the timestamp intersection (no data files) with
integration tests that load the shipped dataset and check the aligned,
standardized return matrix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from crypto_rmt.io import TICKERS, align_prices, load_prices, returns_matrix

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_align_prices_intersects_timestamps() -> None:
    """Only timestamps present in every asset survive alignment."""
    prices = {
        "A": (
            np.array([100, 200, 300, 400], dtype=np.int64),
            np.array([1.0, 2.0, 3.0, 4.0]),
        ),
        "B": (
            np.array([200, 300, 400, 500], dtype=np.int64),
            np.array([9.0, 8.0, 7.0, 6.0]),
        ),
    }
    ts, matrix = align_prices(prices, ["A", "B"])
    assert ts.tolist() == [200, 300, 400]
    assert matrix.shape == (2, 3)
    assert matrix[0].tolist() == [2.0, 3.0, 4.0]
    assert matrix[1].tolist() == [9.0, 8.0, 7.0]


def test_align_prices_drops_null_hours() -> None:
    """Hours where any asset has a NaN price are dropped from the common grid."""
    stamps = np.array([100, 200, 300], dtype=np.int64)
    prices = {
        "A": (stamps, np.array([1.0, np.nan, 3.0])),
        "B": (stamps, np.array([9.0, 8.0, 7.0])),
    }
    ts, matrix = align_prices(prices, ["A", "B"])
    assert ts.tolist() == [100, 300]
    assert np.isfinite(matrix).all()


def test_load_prices_returns_timestamps_and_prices() -> None:
    """Each series loads as an (int64 timestamps, float64 prices) pair."""
    prices = load_prices(TICKERS, DATA_DIR)
    timestamps, values = prices["BTC"]
    assert timestamps.dtype == np.int64
    assert values.dtype == np.float64
    assert timestamps.shape == values.shape


def test_align_prices_puts_assets_on_one_grid() -> None:
    """Alignment yields an (N, T) finite matrix on a single shared grid."""
    prices = load_prices(TICKERS, DATA_DIR)
    timestamps, matrix = align_prices(prices, TICKERS)
    assert matrix.shape == (len(TICKERS), timestamps.size)
    assert np.isfinite(matrix).all()
    assert timestamps.size > 0


def test_returns_matrix_shape_and_standardized() -> None:
    """The return matrix is (N, T-1), NaN-free, and row-standardized."""
    prices = load_prices(TICKERS, DATA_DIR)
    matrix = returns_matrix(prices, TICKERS)
    assert matrix.shape[0] == len(TICKERS)
    assert not np.isnan(matrix).any()
    assert np.allclose(matrix.mean(axis=1), 0.0, atol=1e-9)
    assert np.allclose(matrix.std(axis=1), 1.0, atol=1e-9)


def test_returns_matrix_timestamps_align_to_columns() -> None:
    """With return_timestamps, the timestamps match the return columns."""
    prices = load_prices(TICKERS, DATA_DIR)
    matrix, timestamps = returns_matrix(prices, TICKERS, return_timestamps=True)
    assert timestamps.size == matrix.shape[1]
    assert timestamps.dtype == np.int64
