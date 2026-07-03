"""Smoke tests for the :mod:`crypto_rmt.io` data-loading layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from crypto_rmt.io import (
    TICKERS,
    align_window,
    default_window,
    load_prices,
    returns_matrix,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_load_prices_btc_last_value_and_length() -> None:
    """BTC series loads full-length with the expected final price."""
    prices = load_prices(["BTC"], DATA_DIR)
    btc = prices["BTC"]
    assert btc.shape[0] == 103809
    assert btc[-1] == pytest.approx(29378.2470, abs=1e-3)


def test_default_window_is_12001() -> None:
    """Window derived from the shortest (LDO) series is 12001."""
    prices = load_prices(TICKERS, DATA_DIR)
    assert default_window(prices) == 12001


def test_align_window_returns_trailing_points() -> None:
    """align_window keeps exactly the last W points."""
    series = np.arange(100, dtype=np.float64)
    windowed = align_window(series, 10)
    assert windowed.shape == (10,)
    assert windowed[0] == 90.0
    assert windowed[-1] == 99.0


def test_returns_matrix_shape_and_no_nan() -> None:
    """Returns matrix is (18, 12000) and free of NaNs."""
    prices = load_prices(TICKERS, DATA_DIR)
    matrix = returns_matrix(prices, TICKERS)
    assert matrix.shape == (18, 12000)
    assert not np.isnan(matrix).any()


def test_returns_matrix_rows_are_standardized() -> None:
    """Each row has zero mean and unit standard deviation."""
    prices = load_prices(TICKERS, DATA_DIR)
    matrix = returns_matrix(prices, TICKERS)
    for row in matrix:
        assert abs(np.mean(row)) < 1e-9
        assert abs(np.std(row) - 1.0) < 1e-9
