"""Data-loading layer for the crypto-RMT analysis.

This module owns *only* the ingestion, alignment, and cleaning of the raw
hourly price files. It parses the on-disk JSON, aligns every asset onto the
hours they all share, and builds the z-scored ``log10`` return matrix consumed
by the analysis modules. No correlation, eigenvalue, or plotting logic lives
here.

Alignment policy
----------------
Each file ``<TICKER>_Price_1h.txt`` is a JSON array of
``{"t": <unix_seconds>, "v": <float | null>}`` records. Real exchange data has
sparse hourly gaps (maintenance windows, thin trading) that do **not** line up
across assets, so series cannot be aligned by row position -- doing so would
place different calendar hours in the same column and silently corrupt every
correlation. Instead :func:`align_prices` intersects timestamps across all
assets and keeps only the hours where every asset has a finite price (``null``
quotes are dropped, not filled). Returns are then computed close-to-close over
that common grid.

``log10`` price differences and the subsequent z-score match the original
notebook's return definition (do not switch to ``ln``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt

__all__ = [
    "TICKERS",
    "PRICE_FILE_TEMPLATE",
    "load_prices",
    "align_prices",
    "returns_matrix",
]

#: Canonical ticker order (liquid universe with full, comparable history).
TICKERS: tuple[str, ...] = (
    "BTC",
    "ETH",
    "BNB",
    "SOL",
    "ADA",
    "XRP",
    "AVAX",
    "DOT",
    "ATOM",
    "LTC",
    "DOGE",
    "LINK",
    "UNI",
    "AAVE",
    "CRV",
    "SUSHI",
    "SAND",
)

#: Filename pattern for the per-asset raw price files.
PRICE_FILE_TEMPLATE: str = "{ticker}_Price_1h.txt"

Series = tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]


def load_prices(
    tickers: list[str] | tuple[str, ...],
    data_dir: str | Path,
) -> dict[str, Series]:
    """Load the ``(timestamps, prices)`` series for each requested ticker.

    Parameters
    ----------
    tickers : list of str or tuple of str
        Tickers to load. Each maps to ``<data_dir>/<TICKER>_Price_1h.txt``.
    data_dir : str or pathlib.Path
        Directory containing the raw price files.

    Returns
    -------
    dict of str to (numpy.ndarray, numpy.ndarray)
        Mapping from ticker to ``(timestamps, prices)``: an ``int64`` array of
        unix-second timestamps and a ``float64`` array of prices. JSON ``null``
        prices are converted to :data:`numpy.nan`.

    Raises
    ------
    FileNotFoundError
        If a ticker's price file does not exist under ``data_dir``.

    """
    data_path = Path(data_dir)
    out: dict[str, Series] = {}
    for ticker in tickers:
        file_path = data_path / PRICE_FILE_TEMPLATE.format(ticker=ticker)
        records = json.loads(file_path.read_text())
        timestamps = np.fromiter(
            (int(record["t"]) for record in records),
            dtype=np.int64,
            count=len(records),
        )
        prices = np.fromiter(
            (
                np.nan if record["v"] is None else float(record["v"])
                for record in records
            ),
            dtype=np.float64,
            count=len(records),
        )
        out[ticker] = (timestamps, prices)
    return out


def align_prices(
    prices: dict[str, Series],
    tickers: list[str] | tuple[str, ...],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Align all assets onto their common, fully-observed hourly grid.

    Parameters
    ----------
    prices : dict of str to (numpy.ndarray, numpy.ndarray)
        ``(timestamps, prices)`` per ticker, as returned by :func:`load_prices`.
        Each timestamp array must be sorted ascending (as fetched).
    tickers : list of str or tuple of str
        Assets to align, and the row order of the output matrix.

    Returns
    -------
    timestamps : numpy.ndarray
        The ``int64`` timestamps present in *every* asset and finite in all of
        them, sorted ascending (length ``T``).
    matrix : numpy.ndarray
        Price matrix of shape ``(N, T)`` with ``N = len(tickers)``, row ``i``
        holding asset ``tickers[i]``'s prices at those common timestamps.

    Raises
    ------
    ValueError
        If ``tickers`` is empty or the assets share no common timestamp.

    Notes
    -----
    Alignment is by timestamp, not row position: the returned timestamps are the
    intersection across assets, restricted to hours where no asset has a
    ``null`` (``NaN``) price. This guarantees every column is the same calendar
    hour for all assets.

    """
    if not tickers:
        raise ValueError("`tickers` is empty; nothing to align.")

    common: set[int] = set(prices[tickers[0]][0].tolist())
    for ticker in tickers[1:]:
        common &= set(prices[ticker][0].tolist())
    if not common:
        raise ValueError("assets share no common timestamp.")
    common_ts = np.array(sorted(common), dtype=np.int64)

    rows = []
    for ticker in tickers:
        timestamps, series_prices = prices[ticker]
        idx = np.searchsorted(timestamps, common_ts)
        rows.append(series_prices[idx])
    matrix = np.asarray(rows, dtype=np.float64)

    finite = np.isfinite(matrix).all(axis=0)
    return common_ts[finite], matrix[:, finite]


def returns_matrix(
    prices: dict[str, Series],
    tickers: list[str] | tuple[str, ...],
    *,
    return_timestamps: bool = False,
) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Build the z-scored ``log10`` return matrix over the common grid.

    For the timestamp-aligned price matrix: compute ``log10`` price differences
    along time, then z-score each row with ``(R - mean) / std``.

    Parameters
    ----------
    prices : dict of str to (numpy.ndarray, numpy.ndarray)
        ``(timestamps, prices)`` per ticker, as returned by :func:`load_prices`.
    tickers : list of str or tuple of str
        Row order of the output matrix.
    return_timestamps : bool, optional
        If ``True``, also return the timestamps aligned to the returns (one per
        return column). Defaults to ``False``.

    Returns
    -------
    numpy.ndarray or (numpy.ndarray, numpy.ndarray)
        Array of shape ``(N, T - 1)`` whose rows have zero mean and unit
        (population) standard deviation. If ``return_timestamps`` is ``True``, a
        ``(matrix, timestamps)`` pair where ``timestamps`` has length ``T - 1``
        (the closing timestamp of each return interval).

    Notes
    -----
    ``log10`` and the population z-score (``ddof=0``) match the original
    notebook exactly. Prices are aligned by :func:`align_prices` first, so every
    return is computed over the same calendar interval for all assets.

    """
    timestamps, matrix = align_prices(prices, tickers)
    log_returns = np.diff(np.log10(matrix), axis=1)
    standardized = (
        log_returns - log_returns.mean(axis=1, keepdims=True)
    ) / log_returns.std(axis=1, keepdims=True)
    if return_timestamps:
        return standardized, timestamps[1:]
    return standardized
