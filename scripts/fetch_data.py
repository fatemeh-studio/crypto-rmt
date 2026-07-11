#!/usr/bin/env python3
"""Fetch hourly close-price series for the crypto-RMT universe from Binance.

Reproducibility utility for the crypto-rmt project. Pulls 1-hour klines for a
fixed, liquid universe of assets from Binance's public market-data API and
writes one JSON file per asset in the exact format consumed by
:mod:`crypto_rmt.io` (``[{"t": <unix_seconds>, "v": <close_price>}, ...]``).

The raw output files stay gitignored; committing *this script* is what makes
the dataset fully reproducible from a free, documented source.

Data source
-----------
Binance public spot klines, endpoint ``/api/v3/klines`` (no API key required).
The primary host is ``data-api.binance.vision`` (Binance's public data mirror,
which is not geo-restricted); the main ``api.binance.com`` cluster and a couple
of alternates are used as fallbacks. If every Binance host is blocked from your
network, use the CryptoCompare ``histohour`` fallback noted in the README.

Convention
----------
For each hourly bar we store ``t`` = bar **open time** (UTC, unix seconds) and
``v`` = bar **close** price. Returns are computed close-to-close downstream. The
``t`` field also gives Phase 2 the datetime axis it needs to annotate events.

Progress is printed one line per ticker, with a dot per fetched page, so a long
run is never opaque.

Usage
-----
    python scripts/fetch_data.py                      # full universe -> data/
    python scripts/fetch_data.py --out data --start 2021-01-01
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# --- Universe: liquid, recognizable, sector-diverse; all <TICKER>USDT on Binance.
#     Keep this tuple identical to crypto_rmt.io.TICKERS.
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

# Sector tags -- used later to annotate the cluster map and read the eigenvectors.
SECTORS: dict[str, str] = {
    "BTC": "L1",
    "ETH": "L1",
    "BNB": "L1/exchange",
    "SOL": "L1",
    "ADA": "L1",
    "XRP": "payments",
    "AVAX": "L1",
    "DOT": "L1/interop",
    "ATOM": "L1/interop",
    "LTC": "payments",
    "DOGE": "meme",
    "LINK": "oracle",
    "UNI": "DeFi/DEX",
    "AAVE": "DeFi/lending",
    "MKR": "DeFi/CDP",
    "CRV": "DeFi/stableswap",
    "SUSHI": "DeFi/DEX",
    "SAND": "gaming",
    "USDC": "stablecoin",
    "DAI": "stablecoin",
}

QUOTE = "USDT"
INTERVAL = "1h"
_HOUR_MS = 3_600_000
_LIMIT = 1000  # max klines returned per request by the klines endpoint

_HOSTS: tuple[str, ...] = (
    "https://data-api.binance.vision",  # public data mirror (not geo-blocked)
    "https://api.binance.com",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
)


def _get_json(path: str, hosts: list[str]) -> list:
    """GET ``path`` from the first reachable Binance host.

    Promotes a working host to the front of ``hosts`` so later calls try it
    first. Retries transient errors; treats HTTP 400 as an invalid symbol.

    Raises
    ------
    RuntimeError
        On an invalid symbol, or if every host fails.

    """
    last_err: Exception | None = None
    for i, host in enumerate(list(hosts)):
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    host + path, headers={"User-Agent": "crypto-rmt/1.0"}
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())
                if i:  # remember the host that worked
                    hosts.insert(0, hosts.pop(i))
                return data
            except urllib.error.HTTPError as e:
                if e.code in (429, 418):  # rate limited -> back off, retry host
                    time.sleep(2 * (attempt + 1))
                    last_err = e
                    continue
                if e.code == 400:  # invalid symbol -> not a host problem
                    raise RuntimeError(f"bad request for {path}: {e}") from e
                last_err = e
                break  # 451/403/5xx -> try the next host
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = e
                time.sleep(1 + attempt)
    raise RuntimeError(f"all Binance hosts failed for {path}: {last_err}")


def fetch_klines(
    symbol: str,
    start_ms: int,
    hosts: list[str],
    progress: bool = False,
) -> list[tuple[int, float]]:
    """Fetch every 1h ``(open_time_seconds, close)`` pair from ``start_ms`` to now.

    Parameters
    ----------
    symbol : str
        Trading pair, e.g. ``"BTCUSDT"``.
    start_ms : int
        Start of the fetch window, unix milliseconds (UTC).
    hosts : list of str
        Candidate Binance hosts, tried in order (see :func:`_get_json`).
    progress : bool, optional
        When ``True``, print one ``.`` per fetched page to stdout (flushed) so a
        long run shows live activity. Defaults to ``False``.

    Returns
    -------
    list of (int, float)
        ``(open_time_seconds, close_price)`` pairs, oldest first.

    """
    out: list[tuple[int, float]] = []
    cursor = start_ms
    while True:
        path = (
            f"/api/v3/klines?symbol={symbol}&interval={INTERVAL}"
            f"&startTime={cursor}&limit={_LIMIT}"
        )
        batch = _get_json(path, hosts)
        if not batch:
            break
        out.extend((int(k[0]) // 1000, float(k[4])) for k in batch)
        if progress:
            print(".", end="", flush=True)
        cursor = int(batch[-1][0]) + _HOUR_MS
        if len(batch) < _LIMIT:  # reached the most recent bar
            break
        time.sleep(0.25)  # be polite to the API
    return out


def write_series(records: list[tuple[int, float]], path: Path) -> None:
    """Write ``(t, v)`` pairs as JSON records for :mod:`crypto_rmt.io`."""
    path.write_text(json.dumps([{"t": t, "v": v} for t, v in records]))


def fetch_all(out_dir: str | Path, start: str) -> None:
    """Fetch the full universe into ``out_dir`` as ``<TICKER>_Price_1h.txt`` files.

    Prints a progress line per ticker: an index, the ticker, a dot per fetched
    page, then the bar count and covered date range (or a ``[skip]`` note).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    start_ms = (
        int(
            datetime.strptime(start, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
        * 1000
    )
    hosts = list(_HOSTS)
    total = len(TICKERS)
    for i, ticker in enumerate(TICKERS, start=1):
        symbol = f"{ticker}{QUOTE}"
        print(f"[{i:2d}/{total}] {ticker:5} ", end="", flush=True)
        try:
            records = fetch_klines(symbol, start_ms, hosts, progress=True)
        except RuntimeError as exc:
            print(f"[skip] {exc}")
            continue
        if not records:
            print(f"[skip] no data returned for {symbol}")
            continue
        write_series(records, out_path / f"{ticker}_Price_1h.txt")
        first = datetime.fromtimestamp(records[0][0], timezone.utc).date()
        last = datetime.fromtimestamp(records[-1][0], timezone.utc).date()
        print(f" {len(records):>6} bars  {first} -> {last}")


def main() -> None:
    """Parse CLI arguments and fetch the universe."""
    parser = argparse.ArgumentParser(
        description="Fetch hourly crypto prices for crypto-rmt."
    )
    parser.add_argument(
        "--out", default="data", help="output directory (default: data)"
    )
    parser.add_argument(
        "--start",
        default="2021-01-01",
        help="start date YYYY-MM-DD, UTC (default: 2021-01-01)",
    )
    args = parser.parse_args()
    fetch_all(args.out, args.start)


if __name__ == "__main__":
    main()
