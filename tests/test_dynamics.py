"""Tests for the :mod:`crypto_rmt.dynamics` rolling-correlation metrics.

Mixes pure unit tests of the window and entropy helpers with an integration
test that runs the rolling analysis on the shipped dataset.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from crypto_rmt.dynamics import (
    EVENTS,
    collectivity_series,
    rolling_windows,
    spectral_effective_rank,
)
from crypto_rmt.io import TICKERS, load_prices, returns_matrix

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_rolling_windows_stride_and_size() -> None:
    """Windows advance by ``step`` and each spans ``window`` observations."""
    windows = rolling_windows(100, window=30, step=10)
    assert windows[0] == (0, 30)
    assert windows[1] == (10, 40)
    assert all(end - start == 30 for start, end in windows)
    assert windows[-1][1] <= 100


def test_rolling_windows_rejects_oversized_window() -> None:
    """A window larger than the series raises ``ValueError``."""
    with pytest.raises(ValueError):
        rolling_windows(10, window=20, step=1)


def test_effective_rank_extremes() -> None:
    """Effective rank is 1 for a single dominant mode and N when uniform."""
    assert spectral_effective_rank(np.array([5.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert spectral_effective_rank(np.ones(4)) == pytest.approx(4.0)


def test_events_are_parseable_dates() -> None:
    """Every annotated event date parses as ``YYYY-MM-DD``."""
    for date in EVENTS.values():
        datetime.strptime(date, "%Y-%m-%d")


def test_collectivity_series_shapes_and_ranges() -> None:
    """Rolling metrics have matching shapes and lie in their valid ranges."""
    prices = load_prices(TICKERS, DATA_DIR)
    returns, timestamps = returns_matrix(prices, TICKERS, return_timestamps=True)
    end_ts, collectivity, effective_rank = collectivity_series(
        returns, timestamps, window=720, step=336
    )

    n = len(TICKERS)
    assert end_ts.shape == collectivity.shape == effective_rank.shape
    assert (collectivity > 0.0).all() and (collectivity <= 1.0).all()
    assert (effective_rank >= 1.0 - 1e-9).all()
    assert (effective_rank <= n + 1e-9).all()
    assert (np.diff(end_ts) > 0).all()
