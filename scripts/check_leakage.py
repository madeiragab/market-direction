"""Empirical proof that no future information leaks into the features.

Three tests:

1. TRUNCATION. Recompute the features using only history up to a date D and
   compare them against the features computed over the full history on that same
   day. If any column changes, some window is looking forward.

2. SHUFFLED TARGET. Train the model with labels permuted within each day.
   Out-of-sample accuracy must collapse to the majority class. If it rises, the
   pipeline is leaking through some other path.

3. TARGET ALIGNMENT. Verify by hand that fwd_ret on row t matches the return
   observed between t and t+1.
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from marketdir.data import load_benchmark, load_market
from marketdir.features import build_features, feature_columns
from marketdir.labels import add_target


def test_truncation(market: str, n_checks: int = 3, tol: float = 1e-9) -> bool:
    panel = load_market(market)
    bench = load_benchmark(market)
    full = build_features(panel, bench)

    dates = np.sort(pd.unique(full["date"]))
    cutoffs = dates[[int(len(dates) * f) for f in (0.5, 0.7, 0.9)][:n_checks]]
    cols = feature_columns(full)

    ok = True
    for cutoff in cutoffs:
        trunc_panel = panel[panel["date"] <= cutoff]
        trunc_bench = bench[bench["date"] <= cutoff]
        trunc = build_features(trunc_panel, trunc_bench)

        a = full[full["date"] == cutoff].set_index("ticker")[cols].sort_index()
        b = trunc[trunc["date"] == cutoff].set_index("ticker")[cols].sort_index()
        b = b.reindex(a.index)

        diff = (a - b).abs()
        worst = diff.max().max()
        bad = diff.max()[diff.max() > tol]

        status = "OK  " if worst <= tol or np.isnan(worst) else "FAIL"
        if status != "OK  ":
            ok = False
        print(f"  [{status}] cutoff {str(cutoff)[:10]}  largest difference = {worst:.3e}")
        if len(bad):
            print(f"          suspect columns: {list(bad.index)}")
    return ok


def test_target_alignment(market: str, n_samples: int = 5) -> bool:
    panel = load_market(market)
    bench = load_benchmark(market)
    feats = add_target(build_features(panel, bench), mode="close_to_close")

    ok = True
    rng = np.random.default_rng(0)
    for ticker in rng.choice(pd.unique(feats["ticker"]),
                             size=min(n_samples, feats["ticker"].nunique()), replace=False):
        g = feats[feats["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        i = len(g) // 2
        expected = g.loc[i + 1, "close"] / g.loc[i, "close"] - 1.0
        got = g.loc[i, "fwd_ret"]
        match = abs(expected - got) < 1e-12
        ok &= match
        print(f"  [{'OK  ' if match else 'FAIL'}] {ticker}: fwd_ret={got:+.6f} expected={expected:+.6f}")
    return ok


def test_shuffled_target(market: str) -> bool:
    from marketdir.backtest import run_walk_forward
    from marketdir.config import BacktestConfig
    from marketdir.models import LightGBMModel
    from marketdir.pipeline import build_dataset

    df, cols = build_dataset(market)
    rng = np.random.default_rng(0)
    df = df.copy()
    # Shuffle the target WITHIN each day: destroys any real relationship with the
    # features while preserving each session's up rate.
    df["target"] = df.groupby("date")["target"].transform(lambda s: rng.permutation(s.to_numpy()))

    cfg = BacktestConfig(market=market, test_block_days=126, min_train_days=1500)
    preds, _ = run_walk_forward(df, cols, [LightGBMModel()], cfg)
    acc = float(((preds["prob__lightgbm"] >= 0.5).astype(int) == preds["target"]).mean())
    base = float(preds["target"].mean())
    ok = abs(acc - max(base, 1 - base)) < 0.02
    print(f"  [{'OK  ' if ok else 'FAIL'}] accuracy with shuffled target = {acc:.4f} "
          f"(majority class = {max(base, 1 - base):.4f})")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="BR")
    ap.add_argument("--skip-shuffle", action="store_true")
    args = ap.parse_args()

    print(f"\n=== Leakage audit -- {args.market} market ===\n")
    print("1) History truncation")
    ok1 = test_truncation(args.market)
    print("\n2) Target alignment")
    ok2 = test_target_alignment(args.market)

    ok3 = True
    if not args.skip_shuffle:
        print("\n3) Shuffled target (trains for real, takes a while)")
        ok3 = test_shuffled_target(args.market)

    print("\n" + "=" * 60)
    print("RESULT:", "NO LEAKAGE DETECTED" if (ok1 and ok2 and ok3) else "*** LEAKAGE ***")
    print("""
The honest caveat none of the tests above covers: prices come adjusted for
dividends and splits (auto_adjust). Today's adjustment rewrites prices from
years ago, so the history carries a trace of future information. The effect on
daily direction is small, but it is real and cannot be removed without
point-in-time unadjusted data.""")


if __name__ == "__main__":
    main()
