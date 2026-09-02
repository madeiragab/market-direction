"""Download and cache the history of every ticker in the chosen universes."""
from __future__ import annotations

import argparse
import logging

import _bootstrap  # noqa: F401

from marketdir.config import BENCHMARK, UNIVERSE
from marketdir.data import download_ticker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", nargs="+", default=["BR", "US"])
    ap.add_argument("--force", action="store_true", help="ignore the cache and re-download")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    for market in args.markets:
        tickers = UNIVERSE[market] + [BENCHMARK[market]]
        for ticker in tickers:
            try:
                df = download_ticker(ticker, force=args.force)
                print(f"{market:>2} {ticker:<12} {len(df):>6} sessions  "
                      f"{str(df.index.min())[:10]} -> {str(df.index.max())[:10]}")
            except Exception as err:
                print(f"{market:>2} {ticker:<12} FAILED: {err}")


if __name__ == "__main__":
    main()
