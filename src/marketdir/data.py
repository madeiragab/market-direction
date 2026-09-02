"""OHLCV ingestion via yfinance, with a local parquet cache.

The cache exists for two reasons: to avoid hitting the API on every run and,
more importantly, to freeze the data used in a backtest so the result stays
reproducible.
"""
from __future__ import annotations

import logging
import time

import pandas as pd

from .config import BENCHMARK, DATA_RAW, START_DATE, UNIVERSE

log = logging.getLogger(__name__)

_COLUMNS = ["open", "high", "low", "close", "volume"]


def _cache_path(ticker: str):
    safe = ticker.replace("^", "_idx_").replace(".", "_")
    return DATA_RAW / f"{safe}.parquet"


def download_ticker(ticker: str, start: str = START_DATE, force: bool = False) -> pd.DataFrame:
    """Download (or read from cache) the daily history of one ticker.

    Returns a DataFrame indexed by date with open/high/low/close/volume columns.
    Prices come adjusted for dividends and splits (auto_adjust=True).
    """
    import yfinance as yf

    path = _cache_path(ticker)
    if path.exists() and not force:
        return pd.read_parquet(path)

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            raw = yf.download(
                ticker,
                start=start,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            break
        except Exception as err:  # flaky networking is the common failure here
            last_err = err
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"failed to download {ticker}") from last_err

    if raw is None or raw.empty:
        raise ValueError(f"yfinance returned empty data for {ticker}")

    # Even with a single ticker, yfinance may still return MultiIndex columns.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.rename(columns=str.lower)[_COLUMNS].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["close"] > 0]

    df.to_parquet(path)
    return df


def load_market(market: str, force: bool = False) -> pd.DataFrame:
    """Long panel for one market: one row per (date, ticker)."""
    frames = []
    for ticker in UNIVERSE[market]:
        try:
            df = download_ticker(ticker, force=force)
        except Exception as err:
            log.warning("skipping %s: %s", ticker, err)
            continue
        df = df.assign(ticker=ticker)
        frames.append(df.reset_index())

    if not frames:
        raise RuntimeError(f"no tickers loaded for market {market}")

    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_benchmark(market: str, force: bool = False) -> pd.DataFrame:
    """Reference index series for the market (Ibovespa or S&P 500)."""
    df = download_ticker(BENCHMARK[market], force=force)
    return df.reset_index()[["date", "close", "volume"]].rename(
        columns={"close": "bench_close", "volume": "bench_volume"}
    )
