"""Report figures. No decoration: every chart answers one question."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import buy_and_hold, simulate_strategy
from .config import FIGURES, BacktestConfig
from .report import model_names, per_fold_accuracy, reliability_curve

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fig_equity(preds: pd.DataFrame, cfg: BacktestConfig) -> None:
    """Equity curve net of cost, against buy and hold."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name in model_names(preds):
        sim = simulate_strategy(preds, f"prob__{name}", cfg)
        eq = (1 + sim["net_return"]).cumprod()
        ax.plot(eq.index, eq.to_numpy(), lw=1.2, label=name)

    bh = (1 + buy_and_hold(preds)).cumprod()
    ax.plot(bh.index, bh.to_numpy(), lw=1.8, color="black", ls="--", label="buy & hold")

    ax.set_yscale("log")
    ax.set_title(f"Out-of-sample equity, net of cost -- {cfg.market}")
    ax.set_ylabel("multiple of initial capital (log scale)")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURES / f"equity_{cfg.market}.png")
    plt.close(fig)


def fig_reliability(preds: pd.DataFrame, best: str, cfg: BacktestConfig) -> None:
    """Reliability diagram: does the stated probability match the observed one?"""
    rc = reliability_curve(preds, best, bins=10)
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0.35, 0.65], [0.35, 0.65], ls="--", color="gray", lw=1, label="perfect calibration")
    ax.plot(rc["pred_mean"], rc["obs_rate"], "o-", lw=1.4, label=best)
    for _, r in rc.iterrows():
        ax.annotate(f"{int(r['n']):,}", (r["pred_mean"], r["obs_rate"]),
                    fontsize=6, xytext=(3, -8), textcoords="offset points", color="gray")
    ax.set_xlabel("predicted probability of an up day")
    ax.set_ylabel("observed frequency of up days")
    ax.set_title(f"Calibration -- {best} ({cfg.market})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / f"reliability_{cfg.market}.png")
    plt.close(fig)


def fig_fold_accuracy(preds: pd.DataFrame, best: str, cfg: BacktestConfig) -> None:
    """Accuracy per quarter. Shows whether the signal is stable or was a burst."""
    pf = per_fold_accuracy(preds, best)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    colors = ["#2e7d32" if a >= 0.5 else "#c62828" for a in pf["accuracy"]]
    ax.bar(range(len(pf)), pf["accuracy"] - 0.5, bottom=0.5, color=colors, width=0.8)
    ax.axhline(0.5, color="black", lw=1)
    ax.axhline(pf["accuracy"].mean(), color="#1565c0", lw=1, ls="--",
               label=f"mean = {pf['accuracy'].mean():.4f}")
    step = max(1, len(pf) // 12)
    ax.set_xticks(range(0, len(pf), step))
    ax.set_xticklabels([str(pf['date'].iloc[i])[:7] for i in range(0, len(pf), step)],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("fold accuracy")
    ax.set_title(f"Out-of-sample accuracy per test block -- {best} ({cfg.market})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / f"fold_accuracy_{cfg.market}.png")
    plt.close(fig)


def fig_threshold_sweep(preds: pd.DataFrame, best: str, cfg: BacktestConfig) -> None:
    """How the entry threshold trades selectivity against number of trades."""
    rows = []
    for th in np.arange(0.50, 0.66, 0.01):
        c = BacktestConfig(**{**cfg.__dict__, "prob_threshold": float(th)})
        sim = simulate_strategy(preds, f"prob__{best}", c)
        r = sim["net_return"].dropna()
        sharpe = (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else np.nan
        rows.append({"th": th, "sharpe": sharpe, "exposure": (sim["n_positions"] > 0).mean()})
    d = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(d["th"], d["sharpe"], "o-", lw=1.4, color="#1565c0", label="net Sharpe")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("probability threshold to open a position")
    ax.set_ylabel("net Sharpe")
    ax2 = ax.twinx()
    ax2.plot(d["th"], d["exposure"], "s--", lw=1, color="#ef6c00", ms=3,
             label="fraction of days in market")
    ax2.set_ylabel("fraction of days in market")
    ax2.grid(False)
    ax.set_title(f"Threshold sensitivity -- {best} ({cfg.market})")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES / f"threshold_sweep_{cfg.market}.png")
    plt.close(fig)


def fig_importance(preds: pd.DataFrame, cfg: BacktestConfig, top: int = 20) -> None:
    imp = preds.attrs.get("feature_importance")
    if imp is None or not len(imp):
        return
    imp = imp.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6, 0.28 * len(imp) + 1))
    ax.barh(range(len(imp)), imp.to_numpy(), color="#455a64")
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp.index, fontsize=7)
    ax.set_title(f"Mean feature importance (LightGBM) -- {cfg.market}")
    fig.tight_layout()
    fig.savefig(FIGURES / f"importance_{cfg.market}.png")
    plt.close(fig)


def make_all_figures(preds: pd.DataFrame, results: dict, cfg: BacktestConfig) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    names = [n for n in results if not n.startswith("_")]
    best = max(names, key=lambda n: results[n]["classification"]["accuracy"])

    fig_equity(preds, cfg)
    fig_reliability(preds, best, cfg)
    fig_fold_accuracy(preds, best, cfg)
    fig_threshold_sweep(preds, best, cfg)
    fig_importance(preds, cfg)
