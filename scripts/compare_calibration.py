"""Pick the calibration method by measurement, not by opinion.

Compares three versions of the same base model out of sample:
    raw      -- the probability the classifier emits, untreated
    platt    -- two-parameter sigmoid fitted on the final slice of the train set
    isotonic -- isotonic regression, non-parametric

The criterion is not accuracy. Accuracy only looks at which side of 0.5 the
number landed on; calibration is about the value of the number itself. The
criteria are Brier, log loss and ECE, plus the most extreme probability each
method allows itself to emit -- because a method that says "99.9% up" backed by
six observations is lying with confidence.
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

import numpy as np

from marketdir.backtest import run_walk_forward
from marketdir.config import BacktestConfig
from marketdir.metrics import classification_metrics
from marketdir.models import BaseModel, CalibratedModel, LogisticModel
from marketdir.pipeline import build_dataset
from marketdir.report import reliability_curve


class CalibrationVariant(BaseModel):
    """One calibration variant of the same base model, exposed as a model.

    calibrate = False because the backtest must not wrap this in a second
    calibrator: the calibration (or its absence) is the object of the test.
    """

    calibrate = False

    def __init__(self, method: str | None, name: str):
        self.method = method
        self.name = name

    def fit(self, X, y, dates=None):
        base = LogisticModel()
        if self.method is None:
            self.inner_ = base.fit(X, y)
        else:
            self.inner_ = CalibratedModel(base, 0.2, method=self.method)
            self.inner_.fit(X, y, dates=dates)
        return self

    def predict_proba_up(self, X):
        return self.inner_.predict_proba_up(X)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="BR")
    args = ap.parse_args()

    cfg = BacktestConfig(market=args.market)
    df, cols = build_dataset(cfg.market, cfg.target_mode)

    variants = [
        CalibrationVariant(None, "raw"),
        CalibrationVariant("sigmoid", "platt"),
        CalibrationVariant("isotonic", "isotonic"),
    ]
    preds, folds = run_walk_forward(df, cols, variants, cfg)

    print(f"\n=== Calibration -- {cfg.market} | {len(preds):,} out-of-sample predictions ===\n")
    print(f"{'method':<10} {'accuracy':>9} {'AUC':>8} {'Brier':>9} {'log loss':>9} "
          f"{'p max':>8} {'p min':>8} {'ECE':>9}")

    y = preds["target"].to_numpy()
    for v in variants:
        prob = preds[f"prob__{v.name}"].to_numpy()
        c = classification_metrics(y, prob)
        rc = reliability_curve(preds, v.name, bins=10)
        ece = float(np.average((rc["pred_mean"] - rc["obs_rate"]).abs(), weights=rc["n"]))
        print(f"{v.name:<10} {c['accuracy']:>9.4f} {c['auc']:>8.4f} {c['brier']:>9.5f} "
              f"{c['log_loss']:>9.5f} {prob.max():>8.3f} {prob.min():>8.3f} {ece:>9.5f}")

    print("\nLowest Brier and lowest ECE win. An extreme probability backed by a "
          "small sample is a symptom, not a quality.")
    print("\nA note on AUC: it is invariant to monotone transformations within a "
          "fold, but folds are pooled here and each one calibrates with its own "
          "parameters. That reorders predictions across folds, so the pooled AUC "
          "shifts even with the same base model.")


if __name__ == "__main__":
    main()
