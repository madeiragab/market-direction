"""Target definition.

The label on row (t, ticker) describes what happens AFTER t. This is the only
place in the project allowed to look into the future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_target(df: pd.DataFrame, mode: str = "close_to_close") -> pd.DataFrame:
    """Add fwd_ret (the trade's simple return) and target (1 = up).

    close_to_close: decide at the close of t, exit at the close of t+1.
        Assumes the closing auction is reachable.
    open_to_close: decide at the close of t, enter at the open of t+1 and exit
        at the close of t+1. Does not depend on the auction, but throws away the
        overnight return, which is usually a large share of the move.
    """
    out = df.sort_values(["ticker", "date"]).copy()
    g = out.groupby("ticker", sort=False)

    next_close = g["close"].shift(-1)
    next_open = g["open"].shift(-1)

    if mode == "close_to_close":
        out["fwd_ret"] = next_close / out["close"] - 1.0
    elif mode == "open_to_close":
        out["fwd_ret"] = next_close / next_open - 1.0
    else:
        raise ValueError(f"unknown target_mode: {mode}")

    out["target"] = (out["fwd_ret"] > 0).astype("int8")
    out.loc[out["fwd_ret"].isna(), "target"] = np.nan

    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def clean_panel(df: pd.DataFrame, feature_cols: list[str], max_nan_frac: float = 0.0) -> pd.DataFrame:
    """Drop rows without a target or with incomplete features.

    Warming up the long windows (SMA200, 12-1 momentum) eats the first year of
    every ticker. That is expected, not a bug.
    """
    out = df.dropna(subset=["target"]).copy()
    if max_nan_frac <= 0:
        out = out.dropna(subset=feature_cols)
    else:
        keep = out[feature_cols].isna().mean(axis=1) <= max_nan_frac
        out = out[keep]
    out["target"] = out["target"].astype(int)
    return out.reset_index(drop=True)
