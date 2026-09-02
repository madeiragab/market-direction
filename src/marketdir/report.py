"""Evaluation and report generation from the out-of-sample predictions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import buy_and_hold, simulate_strategy
from .config import BacktestConfig
from .metrics import (
    block_bootstrap_sharpe,
    classification_metrics,
    safe_binomial_pvalue,
    strategy_metrics,
)


def model_names(preds: pd.DataFrame) -> list[str]:
    return [c.removeprefix("prob__") for c in preds.columns if c.startswith("prob__")]


def evaluate(preds: pd.DataFrame, cfg: BacktestConfig) -> dict:
    """Classification and portfolio metrics for every model."""
    y = preds["target"].to_numpy()
    results: dict[str, dict] = {}

    for name in model_names(preds):
        prob = preds[f"prob__{name}"].to_numpy()
        cls = classification_metrics(y, prob)
        n_correct = int(((prob >= 0.5).astype(int) == y).sum())
        cls["p_value_vs_coin"] = safe_binomial_pvalue(n_correct, len(y))
        cls["p_value_vs_base_rate"] = safe_binomial_pvalue(
            n_correct, len(y), max(cls["base_rate_up"], 1 - cls["base_rate_up"])
        )

        sim = simulate_strategy(preds, f"prob__{name}", cfg)
        strat_net = strategy_metrics(sim["net_return"])
        strat_gross = strategy_metrics(sim["gross_return"])
        boot = block_bootstrap_sharpe(sim["net_return"], seed=cfg.seed)

        results[name] = {
            "classification": cls,
            "strategy_net": strat_net,
            "strategy_gross": strat_gross,
            "bootstrap": boot,
            "avg_turnover": float(sim["turnover"].mean()),
            "avg_positions": float(sim["n_positions"].mean()),
            "days_in_market": float((sim["n_positions"] > 0).mean()),
            "total_cost_drag": float(sim["cost"].sum()),
        }

    bh = buy_and_hold(preds)
    results["_buy_and_hold"] = {"strategy_net": strategy_metrics(bh)}
    return results


def reliability_curve(preds: pd.DataFrame, name: str, bins: int = 10) -> pd.DataFrame:
    """Predicted probability against observed frequency.

    This is the chart that separates a useful model from a guess with decimal
    places: if it says 60% and is right 52% of the time, the probability is
    decorative.
    """
    prob = preds[f"prob__{name}"]
    df = pd.DataFrame({"prob": prob, "y": preds["target"]})
    df["bin"] = pd.qcut(df["prob"], q=bins, duplicates="drop")
    out = df.groupby("bin", observed=True).agg(
        pred_mean=("prob", "mean"), obs_rate=("y", "mean"), n=("y", "size")
    )
    return out.reset_index(drop=True)


def per_fold_accuracy(preds: pd.DataFrame, name: str) -> pd.DataFrame:
    prob = preds[f"prob__{name}"]
    hit = (prob >= 0.5).astype(int) == preds["target"]
    return (
        preds.assign(hit=hit)
        .groupby("fold")
        .agg(accuracy=("hit", "mean"), n=("hit", "size"), date=("date", "min"))
        .reset_index()
    )


def per_ticker_accuracy(preds: pd.DataFrame, name: str) -> pd.DataFrame:
    prob = preds[f"prob__{name}"]
    hit = (prob >= 0.5).astype(int) == preds["target"]
    return (
        preds.assign(hit=hit)
        .groupby("ticker")
        .agg(accuracy=("hit", "mean"), n=("hit", "size"), base_rate=("target", "mean"))
        .sort_values("accuracy", ascending=False)
        .reset_index()
    )


# --------------------------------------------------------------------------
def _fmt_pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.2f}%"


def _fmt(x: float, nd: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def write_markdown(results: dict, preds: pd.DataFrame, cfg: BacktestConfig, path: Path,
                   folds: list, importance: pd.Series | None) -> None:
    names = [n for n in results if not n.startswith("_")]
    best = max(names, key=lambda n: results[n]["classification"]["accuracy"])

    lines: list[str] = []
    A = lines.append
    A(f"# Backtest -- {cfg.market} market\n")
    A(f"Target: next-day direction ({cfg.target_mode}). "
      f"Validation: expanding walk-forward, blocks of {cfg.test_block_days} sessions, "
      f"{cfg.embargo_days}-day embargo. "
      f"Cost: {cfg.cost_bps_per_side} bps per side. "
      f"Entry threshold: {cfg.prob_threshold}.\n")
    A(f"Out-of-sample period: {str(preds['date'].min())[:10]} to {str(preds['date'].max())[:10]} "
      f"({len(folds)} folds, {len(preds):,} predictions).\n")

    A("\n## Classification (all out of sample)\n")
    A("| model | accuracy | balanced acc. | AUC | Brier | p vs coin |")
    A("|---|---|---|---|---|---|")
    for n in names:
        c = results[n]["classification"]
        A(f"| {n} | {_fmt_pct(c['accuracy'])} | {_fmt_pct(c['balanced_accuracy'])} | "
          f"{_fmt(c['auc'])} | {_fmt(c['brier'])} | {c['p_value_vs_coin']:.4f} |")
    base = results[names[0]]["classification"]["base_rate_up"]
    A(f"\nNatural share of up days in the period: **{_fmt_pct(base)}**. "
      "Any accuracy below that loses to the rule of always guessing up.\n")

    A("\n## Strategy, net of cost\n")
    A("| model | CAGR | vol | Sharpe | Sharpe 95% CI | max DD | avg turnover | days in market |")
    A("|---|---|---|---|---|---|---|---|")
    for n in names:
        s = results[n]["strategy_net"]
        b = results[n]["bootstrap"]
        A(f"| {n} | {_fmt_pct(s['cagr'])} | {_fmt_pct(s['vol_annual'])} | {_fmt(s['sharpe'], 2)} | "
          f"[{_fmt(b['sharpe_ci_low'], 2)}, {_fmt(b['sharpe_ci_high'], 2)}] | "
          f"{_fmt_pct(s['max_drawdown'])} | {_fmt(results[n]['avg_turnover'], 2)} | "
          f"{_fmt_pct(results[n]['days_in_market'])} |")
    bh = results["_buy_and_hold"]["strategy_net"]
    A(f"| **buy & hold** | {_fmt_pct(bh['cagr'])} | {_fmt_pct(bh['vol_annual'])} | "
      f"{_fmt(bh['sharpe'], 2)} | -- | {_fmt_pct(bh['max_drawdown'])} | -- | 100.00% |")

    A("\n## Cost matters\n")
    A("| model | gross Sharpe | net Sharpe | lost to cost |")
    A("|---|---|---|---|")
    for n in names:
        g = results[n]["strategy_gross"]["sharpe"]
        s = results[n]["strategy_net"]["sharpe"]
        A(f"| {n} | {_fmt(g, 2)} | {_fmt(s, 2)} | {_fmt(g - s, 2)} |")

    A(f"\n## Calibration of the best model ({best})\n")
    A("If the left column and the middle column move together, the probability "
      "means something. If they drift apart, the number is decoration.\n")
    rc = reliability_curve(preds, best)
    A("| predicted probability bucket | observed frequency of up days | n |")
    A("|---|---|---|")
    for _, r in rc.iterrows():
        A(f"| {_fmt_pct(r['pred_mean'])} | {_fmt_pct(r['obs_rate'])} | {int(r['n']):,} |")

    A(f"\n## Accuracy per ticker ({best})\n")
    pt = per_ticker_accuracy(preds, best)
    A("| ticker | accuracy | up rate | n |")
    A("|---|---|---|---|")
    for _, r in pt.iterrows():
        A(f"| {r['ticker']} | {_fmt_pct(r['accuracy'])} | {_fmt_pct(r['base_rate'])} | {int(r['n']):,} |")

    if importance is not None and len(importance):
        A("\n## Features most used by LightGBM\n")
        A("| feature | mean importance |")
        A("|---|---|")
        for k, v in importance.head(20).items():
            A(f"| {k} | {v:.1f} |")

    A("\n## How to read these numbers\n")
    A("- 52-54% directional accuracy at D+1 is the realistic ceiling in the "
      "literature. A number far above that almost always means leakage, not talent.")
    A("- The p-value compares the accuracy against flipping a coin. Above 0.05, "
      "what you see fits inside chance.")
    A("- The Sharpe confidence interval comes from a block bootstrap, which "
      "preserves autocorrelation. If the interval crosses zero, the strategy "
      "proved nothing.")
    A("- Transaction cost is not a detail: it is the line that kills high-turnover "
      "strategies.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(results: dict, preds: pd.DataFrame, cfg: BacktestConfig, path: Path) -> dict:
    """Serialize everything the dashboard needs."""
    names = [n for n in results if not n.startswith("_")]
    best = max(names, key=lambda n: results[n]["classification"]["accuracy"])

    equity: dict[str, list] = {}
    for n in names:
        sim = simulate_strategy(preds, f"prob__{n}", cfg)
        eq = (1 + sim["net_return"]).cumprod()
        equity[n] = [round(float(v), 6) for v in eq.to_numpy()]
    bh = buy_and_hold(preds)
    bh_eq = (1 + bh).cumprod()
    equity["buy_and_hold"] = [round(float(v), 6) for v in bh_eq.to_numpy()]
    dates = [str(d)[:10] for d in bh_eq.index]

    payload = {
        "market": cfg.market,
        "config": {
            "target_mode": cfg.target_mode,
            "test_block_days": cfg.test_block_days,
            "min_train_days": cfg.min_train_days,
            "cost_bps_per_side": cfg.cost_bps_per_side,
            "prob_threshold": cfg.prob_threshold,
            "allow_short": cfg.allow_short,
        },
        "oos_start": str(preds["date"].min())[:10],
        "oos_end": str(preds["date"].max())[:10],
        "n_predictions": int(len(preds)),
        "best_model": best,
        "results": results,
        "equity": {"dates": dates, "series": equity},
        "reliability": reliability_curve(preds, best).round(4).to_dict(orient="records"),
        "per_fold": per_fold_accuracy(preds, best).assign(
            date=lambda d: d["date"].astype(str).str[:10]
        ).round(4).to_dict(orient="records"),
        "per_ticker": per_ticker_accuracy(preds, best).round(4).to_dict(orient="records"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return payload
