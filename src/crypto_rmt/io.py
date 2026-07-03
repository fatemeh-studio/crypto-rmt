"""Data-loading layer for the crypto-RMT analysis.

This module owns *only* the ingestion and cleaning of the raw hourly price
files. It parses the on-disk JSON, applies a documented null policy, aligns
every asset to a common trailing window, and builds the z-scored log-return
matrix consumed by the analysis modules. No correlation, eigenvalue, or
plotting logic lives here.

Notes
-----
Each data file ``<TICKER>_Price_1h.txt`` contains a JSON array of
``{"t": <unix_seconds>, "v": <float | null>}`` records sampled strictly
hourly (3600 s, no gaps). Prices are stored as ``log10`` differences and then
z-scored, matching the original notebook exactly (do not switch to ``ln``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt

__all__ = [
    "TICKERS",
    "PRICE_FILE_TEMPLATE",
    "default_window",
    "load_prices",
    "align_window",
    "returns_matrix",
]

#: Canonical ticker order used throughout the analysis.
TICKERS: tuple[str, ...] = (
    "BTC",
    "BUSD",
    "DOUGH",
    "ETH",
    "GUSD",
    "HUSD",
    "LDO",
    "MATIC",
    "MCB",
    "MTL",
    "POLY",
    "PPT",
    "REN",
    "REP",
    "SAN",
    "SUSHI",
    "UBT",
    "UMA",
)

#: Filename pattern for the per-asset raw price files.
PRICE_FILE_TEMPLATE: str = "{ticker}_Price_1h.txt"

#: Decimal precision passed to :func:`numpy.round` when deriving the window.
#: ``-2`` rounds the shortest series length down to the nearest hundred.
_WINDOW_ROUND_DECIMALS: int = -2


def _forward_fill(series: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Forward-fill ``NaN`` entries in a 1-D price series.

    Parameters
    ----------
    series : numpy.ndarray
        One-dimensional array of prices that may contain ``NaN`` gaps.

    Returns
    -------
    numpy.ndarray
        Copy of ``series`` with each ``NaN`` replaced by the most recent
        preceding finite value.

    Notes
    -----
    This is the documented null policy: missing quotes are carried forward
    from the last known price rather than dropped, so the strict hourly grid
    (and therefore the returns alignment) is preserved. On the 12001-point
    analysis window every series is already gap-free, so this call is a
    **no-op** there; it exists to keep the loader reusable on the full history.
    """
    filled = np.asarray(series, dtype=np.float64).copy()
    mask = np.isnan(filled)
    if not mask.any():
        return filled
    idx = np.where(~mask, np.arange(filled.size), 0)
    np.maximum.accumulate(idx, out=idx)
    return filled[idx]


def default_window(prices: dict[str, npt.NDArray[np.float64]]) -> int:
    """Derive the common trailing window ``W`` from the loaded series.

    Parameters
    ----------
    prices : dict of str to numpy.ndarray
        Mapping from ticker to its full-length price array, as returned by
        :func:`load_prices`.

    Returns
    -------
    int
        ``round(min_len, -2) + 1`` where ``min_len`` is the length of the
        shortest series. For the shipped dataset the shortest series is LDO
        (12012 points), giving ``W = 12001``.

    Raises
    ------
    ValueError
        If ``prices`` is empty.
    """
    if not prices:
        raise ValueError("`prices` is empty; cannot derive a window length.")
    min_len = min(series.size for series in prices.values())
    return int(np.round(min_len, _WINDOW_ROUND_DECIMALS)) + 1


def load_prices(
    tickers: list[str] | tuple[str, ...],
    data_dir: str | Path,
) -> dict[str, npt.NDArray[np.float64]]:
    """Load full-length price series for the requested tickers.

    Replaces the 18 copy-pasted ``open``/``json.loads``/``append`` blocks of
    the original notebook with a single loop.

    Parameters
    ----------
    tickers : list of str or tuple of str
        Tickers to load. Each maps to ``<data_dir>/<TICKER>_Price_1h.txt``.
    data_dir : str or pathlib.Path
        Directory containing the raw price files.

    Returns
    -------
    dict of str to numpy.ndarray
        Mapping from ticker to its full-length ``float64`` price array. JSON
        ``null`` values are converted to :data:`numpy.nan`.

    Raises
    ------
    FileNotFoundError
        If a ticker's price file does not exist under ``data_dir``.

    Notes
    -----
    Nulls are only converted (``null -> np.nan``) here; the null *policy*
    (forward-fill) is applied later in :func:`returns_matrix`, after the
    analysis window has been selected.
    """
    data_path = Path(data_dir)
    prices: dict[str, npt.NDArray[np.float64]] = {}
    for ticker in tickers:
        file_path = data_path / PRICE_FILE_TEMPLATE.format(ticker=ticker)
        with file_path.open("r") as handle:
            records = json.loads(handle.read())
        values = [
            np.nan if record["v"] is None else float(record["v"]) for record in records
        ]
        prices[ticker] = np.asarray(values, dtype=np.float64)
    return prices


def align_window(
    series: npt.NDArray[np.float64],
    W: int,  # noqa: N803  (canonical RMT window symbol; part of the public API)
) -> npt.NDArray[np.float64]:
    """Return the last ``W`` points of a price series.

    Parameters
    ----------
    series : numpy.ndarray
        One-dimensional full-length price array.
    W : int
        Number of trailing points to keep (the common window length).

    Returns
    -------
    numpy.ndarray
        The final ``W`` elements of ``series``.

    Raises
    ------
    ValueError
        If ``series`` has fewer than ``W`` points.
    """
    series = np.asarray(series, dtype=np.float64)
    if series.size < W:
        raise ValueError(
            f"series has {series.size} points, fewer than the window W={W}."
        )
    return series[-W:]


def returns_matrix(
    prices: dict[str, npt.NDArray[np.float64]],
    tickers: list[str] | tuple[str, ...],
    W: int | None = None,  # noqa: N803  (canonical RMT window symbol)
) -> npt.NDArray[np.float64]:
    """Build the z-scored ``log10`` return matrix.

    For each asset: take the trailing window, forward-fill nulls, compute
    ``log10`` price differences, then z-score with ``(R - mean) / std``.
    Rows are stacked in the order given by ``tickers``.

    Parameters
    ----------
    prices : dict of str to numpy.ndarray
        Mapping from ticker to its full-length price array, as returned by
        :func:`load_prices`.
    tickers : list of str or tuple of str
        Row order of the output matrix.
    W : int, optional
        Common window length. Defaults to :func:`default_window`, i.e.
        ``round(min_len, -2) + 1`` over ``prices`` (12001 for the shipped
        dataset).

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N, W - 1)`` where ``N = len(tickers)``. Each row has
        zero mean and unit (population) standard deviation.

    Notes
    -----
    ``log10`` and the z-score are kept exactly as in the original notebook
    (do **not** switch to ``ln``). The standard deviation uses ``ddof=0``
    (NumPy default), so each row's std is exactly ``1``.
    """
    window = default_window(prices) if W is None else W

    rows: list[npt.NDArray[np.float64]] = []
    for ticker in tickers:
        windowed = align_window(prices[ticker], window)
        windowed = _forward_fill(windowed)
        log_returns = np.log10(windowed[1:]) - np.log10(windowed[:-1])
        standardized = (log_returns - np.mean(log_returns)) / np.std(log_returns)
        rows.append(standardized)
    return np.vstack(rows)
