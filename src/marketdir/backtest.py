"""Walk-forward: the honest core of the project.

A random split on a time series is the mistake that produces the 90% accuracy
found in blog tutorials. It lets the model train on Monday and Wednesday to
predict Tuesday -- information nobody has live.

Here the window expands: train on everything up to the cutoff date, test on the
next block, move forward. Every evaluated prediction is strictly out of sample,
and there is an embargo between train and test because the target of row t
resolves at t+1 and would otherwise touch the first day of the test block.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .models import BaseModel, CalibratedModel

log = logging.getLogger(__name__)


@dataclass
class Fold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int


def make_folds(dates: np.ndarray, cfg: BacktestConfig) -> list[tuple[int, int, int, int]]:
    """Produce (i_train_end, i_test_start, i_test_end) as unique-date indices."""
    n = len(dates)
    folds = []
    test_start = cfg.min_train_days
    while test_start < n:
        test_end = min(test_start + cfg.test_block_days, n)
        train_end = test_start - cfg.embargo_days
        if train_end <= 0:
            break
        folds.append((0, train_end, test_start, test_end))
        test_start = test_end
    return folds


def run_walk_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    models: list[BaseModel],
    cfg: BacktestConfig,
) -> tuple[pd.DataFrame, list[Fold]]:
    """Run the walk-forward and return every out-of-sample prediction.

    df must contain: date, ticker, target, fwd_ret and the feature columns.
    """
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    uniq_dates = np.sort(pd.unique(df["date"]))
    folds_idx = make_folds(uniq_dates, cfg)

    if not folds_idx:
        raise ValueError(
            f"history too short: {len(uniq_dates)} sessions, "
            f"min_train_days={cfg.min_train_days}"
        )

    date_col = df["date"].to_numpy()
    preds: list[pd.DataFrame] = []
    folds: list[Fold] = []
    importances: list[pd.Series] = []

    for k, (i0, i_train_end, i_test_start, i_test_end) in enumerate(folds_idx):
        train_lo, train_hi = uniq_dates[i0], uniq_dates[i_train_end - 1]
        test_lo, test_hi = uniq_dates[i_test_start], uniq_dates[i_test_end - 1]

        train_mask = (date_col >= train_lo) & (date_col <= train_hi)
        test_mask = (date_col >= test_lo) & (date_col <= test_hi)

        train = df[train_mask]
        test = df[test_mask]
        if len(train) < 1000 or test.empty:
            continue

        X_train = train[feature_cols]
        y_train = train["target"].to_numpy()
        X_test = test[feature_cols]

        folds.append(
            Fold(k, train_lo, train_hi, test_lo, test_hi, len(train), len(test))
        )
        log.info(
            "fold %02d | train %s..%s (%d) | test %s..%s (%d)",
            k, str(train_lo)[:10], str(train_hi)[:10], len(train),
            str(test_lo)[:10], str(test_hi)[:10], len(test),
        )

        fold_out = test[["date", "ticker", "target", "fwd_ret"]].copy()
        fold_out["fold"] = k

        for model in models:
            # Trivial baselines are not calibrated (see models.BaseModel).
            m = CalibratedModel(model, cfg.calibration_frac) if model.calibrate else model
            m.fit(X_train, y_train, dates=train["date"])
            fold_out[f"prob__{model.name}"] = m.predict_proba_up(X_test)

            if hasattr(model, "feature_importance"):
                imp = model.feature_importance()
                importances.append(imp.rename(f"fold_{k}"))

        preds.append(fold_out)

    out = pd.concat(preds, ignore_index=True)
    if importances:
        out.attrs["feature_importance"] = pd.concat(importances, axis=1).mean(axis=1).sort_values(
            ascending=False
        )
    return out, folds


# --------------------------------------------------------------------------
# Portfolio simulation
# --------------------------------------------------------------------------
def simulate_strategy(
    preds: pd.DataFrame,
    prob_col: str,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    """Turn probabilities into an equity curve, with cost.

    The rule: go long when the probability clears the threshold, go short (or
    stay out) when it falls below the mirrored threshold, and do nothing in the
    uncertain band in between. Equal weight across the positions open that day.

    Cost is charged on the CHANGE in weight, not on the weight itself: holding a
    position open does not pay commission again. That distinction decides
    whether the strategy survives or not.
    """
    p = preds.pivot_table(index="date", columns="ticker", values=prob_col)
    r = preds.pivot_table(index="date", columns="ticker", values="fwd_ret")
    r = r.reindex_like(p)

    long_sig = (p >= cfg.prob_threshold).astype(float)
    short_sig = (p <= 1 - cfg.prob_threshold).astype(float) if cfg.allow_short else 0.0
    signal = long_sig - short_sig
    signal = signal.where(p.notna(), 0.0)

    n_active = signal.abs().sum(axis=1).replace(0.0, np.nan)
    weights = signal.div(n_active, axis=0).fillna(0.0)

    gross = (weights * r.fillna(0.0)).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    cost = turnover * (cfg.cost_bps_per_side / 10_000.0)

    return pd.DataFrame(
        {
            "gross_return": gross,
            "cost": cost,
            "net_return": gross - cost,
            "turnover": turnover,
            "n_positions": signal.abs().sum(axis=1),
            "exposure": weights.abs().sum(axis=1),
        }
    )


def buy_and_hold(preds: pd.DataFrame) -> pd.Series:
    """Buy and hold the equally weighted portfolio of the universe.

    The competitor any strategy has to beat to justify existing.
    """
    r = preds.pivot_table(index="date", columns="ticker", values="fwd_ret")
    return r.mean(axis=1)
