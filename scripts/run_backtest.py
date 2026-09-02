"""Run the full walk-forward for one market and produce report + figures.

Typical use:
    python scripts/run_backtest.py --market BR
    python scripts/run_backtest.py --market US --threshold 0.53 --cost-bps 3
"""
from __future__ import annotations

import argparse
import logging
import time

import _bootstrap  # noqa: F401

from marketdir.backtest import run_walk_forward
from marketdir.config import DATA_PROCESSED, REPORTS, BacktestConfig
from marketdir.models import default_models
from marketdir.pipeline import build_dataset
from marketdir.report import evaluate, write_json, write_markdown


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="BR", choices=["BR", "US"])
    ap.add_argument("--target-mode", default="close_to_close",
                    choices=["close_to_close", "open_to_close"])
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--min-train-days", type=int, default=750)
    ap.add_argument("--test-block-days", type=int, default=63)
    ap.add_argument("--allow-short", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = BacktestConfig(
        market=args.market,
        target_mode=args.target_mode,
        prob_threshold=args.threshold,
        cost_bps_per_side=args.cost_bps,
        min_train_days=args.min_train_days,
        test_block_days=args.test_block_days,
        allow_short=args.allow_short,
    )

    t0 = time.time()
    print(f"building {cfg.market} panel...")
    df, cols = build_dataset(cfg.market, cfg.target_mode)
    print(f"  {len(df):,} rows | {df['ticker'].nunique()} tickers | {len(cols)} features")
    print(f"  {str(df['date'].min())[:10]} -> {str(df['date'].max())[:10]}")

    print("running walk-forward...")
    preds, folds = run_walk_forward(df, cols, default_models(cfg.seed), cfg)
    importance = preds.attrs.get("feature_importance")

    print("evaluating...")
    results = evaluate(preds, cfg)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    # DataFrame.attrs does not survive parquet; importance goes to its own csv.
    to_save = preds.copy()
    to_save.attrs.clear()
    to_save.to_parquet(DATA_PROCESSED / f"oos_predictions_{cfg.market}.parquet", index=False)
    if importance is not None:
        importance.rename("importance").to_csv(
            DATA_PROCESSED / f"feature_importance_{cfg.market}.csv"
        )

    md_path = REPORTS / f"backtest_{cfg.market}.md"
    json_path = REPORTS / f"backtest_{cfg.market}.json"
    write_markdown(results, preds, cfg, md_path, folds, importance)
    write_json(results, preds, cfg, json_path)

    try:
        from marketdir.figures import make_all_figures

        make_all_figures(preds, results, cfg)
    except Exception as err:  # figures are a nice-to-have, never fail the backtest
        print(f"  (figures failed: {err})")

    print(f"\ndone in {time.time() - t0:.0f}s")
    print(f"  {md_path}")
    print(f"  {json_path}")

    print("\n--- summary ---")
    for name, res in results.items():
        if name.startswith("_"):
            continue
        c = res["classification"]
        s = res["strategy_net"]
        sharpe = "     out" if res["days_in_market"] == 0 else f"{s['sharpe']:+8.2f}"
        print(f"{name:<14} acc={c['accuracy']:.4f}  auc={c['auc']:.4f}  "
              f"net_sharpe={sharpe}  cagr={s['cagr'] * 100:+7.2f}%  "
              f"exposure={res['days_in_market'] * 100:5.1f}%  "
              f"p={c['p_value_vs_coin']:.4f}")
    bh = results["_buy_and_hold"]["strategy_net"]
    print(f"{'buy_and_hold':<14} {'':<12}{'':<12}sharpe    ={bh['sharpe']:+.2f}  "
          f"cagr={bh['cagr'] * 100:+.2f}%")


if __name__ == "__main__":
    main()
