"""Classification and strategy metrics.

Accuracy alone says nothing on a problem with an imbalanced class and a
transaction cost. Every evaluation here ships with its matching baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .config import TRADING_DAYS_PER_YEAR


def classification_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    out = {
        "n": int(len(y_true)),
        "base_rate_up": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, prob)),
    }
    # AUC and log loss need both classes to be present.
    if len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, prob))
        out["log_loss"] = float(log_loss(y_true, prob, labels=[0, 1]))
    else:
        out["auc"] = float("nan")
        out["log_loss"] = float("nan")
    return out


def binomial_pvalue(correct: int, n: int, p0: float = 0.5) -> float:
    """P(getting >= `correct` out of `n` right by pure chance). One-sided.

    This answers the question every pretty backtest skips: does that
    above-50% accuracy fit inside the noise?
    """
    from scipy.stats import binomtest

    return float(binomtest(correct, n, p0, alternative="greater").pvalue)


def _binomial_pvalue_normal(correct: int, n: int, p0: float = 0.5) -> float:
    """Normal approximation, used when scipy is not installed."""
    from math import erfc, sqrt

    if n == 0:
        return float("nan")
    z = (correct - n * p0) / sqrt(n * p0 * (1 - p0))
    return 0.5 * erfc(z / sqrt(2))


def safe_binomial_pvalue(correct: int, n: int, p0: float = 0.5) -> float:
    try:
        return binomial_pvalue(correct, n, p0)
    except Exception:
        return _binomial_pvalue_normal(correct, n, p0)


def strategy_metrics(daily_returns: pd.Series) -> dict[str, float]:
    """Equity-curve metrics derived from daily returns."""
    r = pd.Series(daily_returns).dropna()
    if r.empty:
        return {k: float("nan") for k in
                ("cagr", "vol_annual", "sharpe", "max_drawdown", "calmar", "hit_rate", "n_days")}

    equity = (1.0 + r).cumprod()
    years = len(r) / TRADING_DAYS_PER_YEAR
    cagr = equity.iloc[-1] ** (1 / years) - 1.0 if years > 0 else float("nan")
    vol = r.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (r.mean() / r.std()) * np.sqrt(TRADING_DAYS_PER_YEAR) if r.std() > 0 else float("nan")
    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min())

    return {
        "cagr": float(cagr),
        "vol_annual": float(vol),
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "calmar": float(cagr / abs(max_dd)) if max_dd < 0 else float("nan"),
        "hit_rate": float((r > 0).mean()),
        "n_days": int(len(r)),
    }


def block_bootstrap_sharpe(
    daily_returns: pd.Series,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 42,
) -> dict[str, float]:
    """Confidence interval for the Sharpe ratio via block bootstrap.

    Blocks preserve the autocorrelation of returns; a plain bootstrap ignores it
    and produces an interval that is far too narrow.
    """
    r = pd.Series(daily_returns).dropna().to_numpy()
    n = len(r)
    if n < 3 * block:
        return {"sharpe_ci_low": float("nan"), "sharpe_ci_high": float("nan"), "p_sharpe_le_0": float("nan")}

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    sharpes = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block, size=n_blocks)
        sample = np.concatenate([r[s:s + block] for s in starts])[:n]
        sd = sample.std()
        sharpes[i] = (sample.mean() / sd) * np.sqrt(TRADING_DAYS_PER_YEAR) if sd > 0 else 0.0

    return {
        "sharpe_ci_low": float(np.percentile(sharpes, 2.5)),
        "sharpe_ci_high": float(np.percentile(sharpes, 97.5)),
        "p_sharpe_le_0": float(np.mean(sharpes <= 0)),
    }
