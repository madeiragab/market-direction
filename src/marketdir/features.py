"""Feature engineering.

THE ONE NON-NEGOTIABLE RULE OF THIS FILE:
    every column produced here for row (t, ticker) may only depend on data
    timestamped <= t.

The target (defined in labels.py) uses t+1. If any feature can see t+1, the
whole backtest becomes fiction. scripts/check_leakage.py exists to prove
empirically that this rule holds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RETURN_HORIZONS = (1, 2, 3, 5, 10, 21, 63)
VOL_WINDOWS = (5, 10, 21, 63)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def _per_ticker_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features for a single ticker, already sorted by date."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    logret = np.log(close).diff()

    out["ret_1"] = logret
    for h in RETURN_HORIZONS:
        if h > 1:
            out[f"ret_{h}"] = np.log(close).diff(h)

    for w in VOL_WINDOWS:
        out[f"vol_{w}"] = logret.rolling(w, min_periods=w).std()

    # Volatility-normalized momentum: separates "went up a lot" from "went up a
    # lot relative to how much this ticker usually swings".
    out["ret_21_norm"] = out["ret_21"] / (out["vol_21"] * np.sqrt(21))
    out["ret_5_norm"] = out["ret_5"] / (out["vol_21"] * np.sqrt(5))

    # Short-term reversal against medium-term trend.
    out["mom_12_1"] = np.log(close).diff(252) - np.log(close).diff(21)

    for w in (20, 50, 200):
        sma = close.rolling(w, min_periods=w).mean()
        out[f"sma_ratio_{w}"] = close / sma - 1.0

    std20 = close.rolling(20, min_periods=20).std()
    sma20 = close.rolling(20, min_periods=20).mean()
    out["bb_pctb_20"] = (close - (sma20 - 2 * std20)) / (4 * std20)

    out["rsi_14"] = _rsi(close) / 100.0

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    out["macd_hist"] = (macd - macd.ewm(span=9, adjust=False, min_periods=9).mean()) / close

    atr = _atr(df)
    out["atr_pct"] = atr / close
    out["hl_range"] = (df["high"] - df["low"]) / close
    out["gap"] = df["open"] / close.shift(1) - 1.0

    # Where the close landed inside the day's bar: near the high suggests
    # aggressive buying into the end of the session.
    span = (df["high"] - df["low"]).replace(0.0, np.nan)
    out["clv"] = ((close - df["low"]) - (df["high"] - close)) / span

    volume = df["volume"].replace(0.0, np.nan)
    out["vol_ratio_21"] = np.log(volume / volume.rolling(21, min_periods=21).mean())
    dollar_vol = np.log((volume * close).clip(lower=1.0))
    out["dollar_vol_z"] = (
        dollar_vol - dollar_vol.rolling(63, min_periods=63).mean()
    ) / dollar_vol.rolling(63, min_periods=63).std()

    # Distance from the trailing one-year high.
    roll_max = close.rolling(252, min_periods=126).max()
    out["drawdown_252"] = close / roll_max - 1.0

    # Recent skew and volatility ratio: the tails move before the mean does.
    out["skew_63"] = logret.rolling(63, min_periods=63).skew()
    out["vol_ratio_5_63"] = out["vol_5"] / out["vol_63"]

    return out


def build_features(panel: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix for the whole panel.

    panel: columns date, ticker, open, high, low, close, volume.
    benchmark: columns date, bench_close, bench_volume.
    """
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    parts = []
    for ticker, grp in panel.groupby("ticker", sort=False):
        feats = _per_ticker_features(grp)
        feats.insert(0, "ticker", ticker)
        feats.insert(0, "date", grp["date"].to_numpy())
        feats["close"] = grp["close"].to_numpy()
        feats["open"] = grp["open"].to_numpy()
        parts.append(feats)

    out = pd.concat(parts, ignore_index=True)

    # ---- market context --------------------------------------------------
    bench = benchmark.sort_values("date").copy()
    bench_logret = np.log(bench["bench_close"]).diff()
    bench_feats = pd.DataFrame(
        {
            "date": bench["date"].to_numpy(),
            "bench_ret_1": bench_logret.to_numpy(),
            "bench_ret_5": np.log(bench["bench_close"]).diff(5).to_numpy(),
            "bench_ret_21": np.log(bench["bench_close"]).diff(21).to_numpy(),
            "bench_vol_21": bench_logret.rolling(21, min_periods=21).std().to_numpy(),
        }
    )
    bench_feats["bench_vol_ratio"] = (
        bench_logret.rolling(5, min_periods=5).std().to_numpy()
        / bench_feats["bench_vol_21"]
    )
    out = out.merge(bench_feats, on="date", how="left")

    # Relative strength against the index: is the ticker beating the market?
    out["rel_str_21"] = out["ret_21"] - out["bench_ret_21"]
    out["rel_str_5"] = out["ret_5"] - out["bench_ret_5"]

    # ---- cross section ---------------------------------------------------
    # Rank within the day, using only information from that day. Makes the model
    # comparable across tickers trading at very different scales.
    for col in ("ret_21", "ret_5", "vol_21", "rsi_14", "vol_ratio_21"):
        out[f"xs_{col}"] = out.groupby("date")[col].rank(pct=True)

    # ---- calendar --------------------------------------------------------
    dates = pd.to_datetime(out["date"])
    out["dow"] = dates.dt.dayofweek
    out["month"] = dates.dt.month
    out["is_month_end"] = (dates.dt.is_month_end).astype(int)

    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


NON_FEATURE_COLS = {"date", "ticker", "close", "open", "target", "fwd_ret", "fwd_ret_oc"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]
