"""Tests for the :mod:`crypto_rmt.rmt` RMT core.

Mixes pure unit tests (no data) with one golden integration test that runs the
full pipeline on the shipped dataset and checks the reproduced eigenvalues.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from crypto_rmt.io import TICKERS, load_prices, returns_matrix
from crypto_rmt.rmt import (
    correlation_matrix,
    eigsystem,
    ipr,
    null_threshold,
    participation_ratio,
    shuffle_correlation,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_ipr_basis_vector_is_localized() -> None:
    """A single basis vector gives ipr == 1 and participation == 1."""
    evecs = np.eye(18)[:, :1]  # column e0 in R^18
    assert ipr(evecs)[0] == pytest.approx(1.0)
    assert participation_ratio(evecs)[0] == pytest.approx(1.0)


def test_ipr_uniform_vector_is_delocalized() -> None:
    """A uniform unit vector spreads over all 18 assets."""
    evecs = (np.ones(18) / np.sqrt(18)).reshape(18, 1)
    assert ipr(evecs)[0] == pytest.approx(1.0 / 18.0)
    assert participation_ratio(evecs)[0] == pytest.approx(18.0)


def test_shuffle_correlation_preserves_symmetry_and_diagonal() -> None:
    """Shuffling permutes off-diagonals while keeping symmetry and diagonal."""
    C = np.array(  # noqa: N806  (correlation matrix, standard symbol)
        [
            [1.0, 0.2, 0.3, 0.4],
            [0.2, 1.0, 0.5, 0.6],
            [0.3, 0.5, 1.0, 0.7],
            [0.4, 0.6, 0.7, 1.0],
        ]
    )
    rng = np.random.default_rng(0)
    shuffled = shuffle_correlation(C, rng)

    assert np.allclose(shuffled, shuffled.T)
    assert np.allclose(np.diag(shuffled), np.diag(C))

    iu = np.triu_indices(4, k=1)
    assert np.allclose(np.sort(shuffled[iu]), np.sort(C[iu]))


def test_integration_reproduces_ground_truth_spectrum() -> None:
    """End-to-end pipeline reproduces the notebook's eigenvalues and null band."""
    prices = load_prices(TICKERS, DATA_DIR)
    returns = returns_matrix(prices, TICKERS)
    C = correlation_matrix(returns)  # noqa: N806  (correlation matrix, standard symbol)
    evals, evecs = eigsystem(C)

    assert evals[0] == pytest.approx(6.078, abs=1e-2)
    assert evals.sum() == pytest.approx(18.0, abs=1e-6)
    assert np.isrealobj(evals)
    assert (evals > 0).all()

    assert 11.0 < participation_ratio(evecs)[0] < 13.0
    assert ipr(evecs)[0] < 0.1

    rng = np.random.default_rng(0)
    assert null_threshold(C, rng=rng) < evals[0]
