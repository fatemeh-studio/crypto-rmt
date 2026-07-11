"""Rolling correlation dynamics for the crypto-RMT analysis.

This module turns the static RMT picture into a temporal, regime-aware study.
Over a rolling window it tracks two quantities that summarize how *collective*
the market is at each point in time:

* **collectivity** -- the share of total variance carried by the market mode,
  ``lambda_max / N``. It rises toward 1 when every asset moves together (a
  crisis) and falls toward ``1 / N`` when assets decouple.
* **effective rank** -- ``exp`` of the spectral entropy of the eigenvalues, i.e.
  the effective number of independent factors. It is ``1`` when a single mode
  dominates and ``N`` when variance is spread evenly, and moves inversely to
  collectivity.

The functions here only consume the aligned return matrix and its timestamps
from :mod:`crypto_rmt.io`; correlation/eigenvalue primitives come from
:mod:`crypto_rmt.rmt`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from crypto_rmt.rmt import correlation_matrix

__all__ = [
    "EVENTS",
    "rolling_windows",
    "spectral_effective_rank",
    "collectivity_series",
]

#: Notable crypto regime events (UTC ``YYYY-MM-DD``) for annotating the
#: temporal figures.
EVENTS: dict[str, str] = {
    "LUNA / UST collapse": "2022-05-09",
    "FTX collapse": "2022-11-08",
    "USDC depeg (SVB)": "2023-03-10",
    "spot BTC ETF": "2024-01-11",
}


def rolling_windows(n_obs: int, window: int, step: int) -> list[tuple[int, int]]:
    """Return ``(start, end)`` index pairs for rolling windows.

    Parameters
    ----------
    n_obs : int
        Number of observations (columns) to roll over.
    window : int
        Window length in observations.
    step : int
        Stride between consecutive windows.

    Returns
    -------
    list of (int, int)
        Half-open ``(start, end)`` index pairs, each spanning ``window``
        observations, advancing by ``step``.

    Raises
    ------
    ValueError
        If ``window`` or ``step`` is non-positive, or ``window`` exceeds
        ``n_obs``.

    """
    if window <= 0 or step <= 0:
        raise ValueError("window and step must be positive.")
    if window > n_obs:
        raise ValueError(f"window ({window}) exceeds observations ({n_obs}).")
    return [(s, s + window) for s in range(0, n_obs - window + 1, step)]


def spectral_effective_rank(
    eigenvalues: npt.NDArray[np.float64],
) -> float:
    """Effective number of factors: ``exp`` of the spectral entropy.

    Parameters
    ----------
    eigenvalues : numpy.ndarray
        Eigenvalues of a correlation matrix (non-negative; summing to ``N``).

    Returns
    -------
    float
        ``exp(-sum p_i log p_i)`` with ``p_i = lambda_i / sum(lambda)``. Equal to
        ``1`` when one eigenvalue dominates and ``N`` when all are equal.

    """
    ev = np.asarray(eigenvalues, dtype=np.float64)
    p = ev / ev.sum()
    p = p[p > 0.0]
    return float(np.exp(-(p * np.log(p)).sum()))


def collectivity_series(
    returns: npt.NDArray[np.float64],
    timestamps: npt.NDArray[np.int64],
    *,
    window: int,
    step: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Track market collectivity and effective rank over a rolling window.

    Parameters
    ----------
    returns : numpy.ndarray
        Aligned ``(N, T)`` return matrix from
        :func:`crypto_rmt.io.returns_matrix`.
    timestamps : numpy.ndarray
        Length-``T`` timestamps aligned to the return columns (from
        ``returns_matrix(..., return_timestamps=True)``).
    window : int
        Rolling window length in observations (e.g. ``720`` for ~30 days of
        hourly data).
    step : int
        Stride between windows in observations (e.g. ``168`` for weekly).

    Returns
    -------
    end_timestamps : numpy.ndarray
        Closing timestamp of each window.
    collectivity : numpy.ndarray
        ``lambda_max / N`` per window (variance share of the market mode).
    effective_rank : numpy.ndarray
        :func:`spectral_effective_rank` per window (effective number of
        factors).

    Notes
    -----
    Correlations are recomputed per window with
    :func:`crypto_rmt.rmt.correlation_matrix` (which standardizes internally),
    so only the eigenvalues are needed and :func:`numpy.linalg.eigvalsh` is used.

    """
    returns = np.asarray(returns, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.int64)
    n_assets, n_obs = returns.shape
    windows = rolling_windows(n_obs, window, step)

    end_ts = np.empty(len(windows), dtype=np.int64)
    collectivity = np.empty(len(windows), dtype=np.float64)
    effective_rank = np.empty(len(windows), dtype=np.float64)

    for i, (start, end) in enumerate(windows):
        eigenvalues = np.linalg.eigvalsh(correlation_matrix(returns[:, start:end]))
        collectivity[i] = eigenvalues.max() / n_assets
        effective_rank[i] = spectral_effective_rank(eigenvalues)
        end_ts[i] = timestamps[end - 1]

    return end_ts, collectivity, effective_rank
