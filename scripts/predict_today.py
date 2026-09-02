"""Prediction for the next trading session.

Trains on all available history and scores the last row of every ticker -- the
row that has no target yet, because its target is tomorrow.

Unlike the backtest, there is no validation here: this is production. The
confidence number that comes out only means anything because the backtest
already measured, out of sample, how often that probability is right.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from marketdir.config import REPORTS, BacktestConfig
from marketdir.labels import add_target, clean_panel
from marketdir.models import CalibratedModel, LightGBMModel, LogisticModel
from marketdir.pipeline import build_features_only


def predict_market(market: str, model_name: str = "logistic", threshold: float = 0.55) -> pd.DataFrame:
    cfg = BacktestConfig(market=market, prob_threshold=threshold)

    feats, cols = build_features_only(market)
    feats = add_target(feats, mode=cfg.target_mode)

    train = clean_panel(feats, cols)

    # The last downloaded day is usually a partial bar of the session in
    # progress: zero volume, high equal to low, meaningless volume features.
    # Scoring that bar would mean predicting with data that does not exist yet,
    # so the cutoff is the last date with complete features.
    complete = feats.dropna(subset=cols)
    if complete.empty:
        raise RuntimeError("no ticker has complete features")

    last_date = complete["date"].max()
    live = complete[complete["date"] == last_date]

    if last_date != feats["date"].max():
        print(f"  (ignoring the incomplete bar of {str(feats['date'].max())[:10]})")

    base = LogisticModel() if model_name == "logistic" else LightGBMModel(seed=cfg.seed)
    model = CalibratedModel(base, cfg.calibration_frac)
    model.fit(train[cols], train["target"].to_numpy(), dates=train["date"])

    prob = model.predict_proba_up(live[cols])

    out = pd.DataFrame({
        "ticker": live["ticker"].to_numpy(),
        "last_close": live["close"].to_numpy(),
        "prob_up": prob,
    })
    out["signal"] = np.where(
        out["prob_up"] >= threshold, "BUY",
        np.where(out["prob_up"] <= 1 - threshold, "SELL", "out"),
    )
    out = out.sort_values("prob_up", ascending=False).reset_index(drop=True)
    out.attrs["as_of"] = str(last_date)[:10]
    out.attrs["n_train"] = len(train)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", nargs="+", default=["BR", "US"])
    ap.add_argument("--model", default="logistic", choices=["logistic", "lightgbm"])
    ap.add_argument("--threshold", type=float, default=0.55)
    args = ap.parse_args()

    payload: dict = {"generated_at": datetime.now().isoformat(timespec="seconds"),
                     "model": args.model, "threshold": args.threshold, "markets": {}}

    for market in args.markets:
        out = predict_market(market, args.model, args.threshold)
        print(f"\n=== {market} | close of {out.attrs['as_of']} "
              f"| trained on {out.attrs['n_train']:,} rows ===")
        print(f"{'ticker':<10} {'last close':>11} {'P(up)':>9}  signal")
        for _, r in out.iterrows():
            print(f"{r['ticker']:<10} {r['last_close']:>11.2f} {r['prob_up']:>8.1%}  {r['signal']}")

        payload["markets"][market] = {
            "as_of": out.attrs["as_of"],
            "n_train": out.attrs["n_train"],
            "predictions": out.round(4).to_dict(orient="records"),
        }

    path = REPORTS / "latest_predictions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"\n{path}")
    print("\nReminder: this project's backtest shows an edge of roughly one "
          "percentage point over a coin flip. These probabilities are a weak "
          "signal, not investment advice.")


if __name__ == "__main__":
    main()
