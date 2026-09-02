"""Models and baselines, all behind the same interface.

Baselines are not decoration. Without them there is no way to tell whether 53%
accuracy is signal or just the natural share of up days in the period.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class BaseModel:
    name = "base"
    # Trivial baselines skip the calibrator: isotonic regression would flatten
    # their fixed probabilities until they never cross the entry threshold, and
    # the baseline would degenerate into "stay out of the market" -- a useless
    # comparison.
    calibrate = True

    def fit(self, X: pd.DataFrame, y: np.ndarray, dates: pd.Series | None = None) -> "BaseModel":
        """`dates` aligns with X row by row. Models that don't need it ignore
        it; the calibrator uses it to cut the final slice of the training window
        without shuffling time."""
        raise NotImplementedError

    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
class AlwaysUp(BaseModel):
    """Always predicts up. Equivalent to buy and hold."""

    name = "always_up"
    calibrate = False

    def fit(self, X, y, dates=None):
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba_up(self, X):
        # Maximum confidence on purpose: this way it always opens a position and
        # the resulting strategy is buy & hold paying transaction cost.
        return np.full(len(X), 0.99)


class MajorityClass(BaseModel):
    """Predicts the most frequent class in the training window."""

    name = "majority"
    calibrate = False

    def fit(self, X, y, dates=None):
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba_up(self, X):
        return np.full(len(X), self.rate_)


class PrevSign(BaseModel):
    """Naive momentum: if it went up yesterday, it goes up tomorrow."""

    name = "prev_sign"
    calibrate = False

    def fit(self, X, y, dates=None):
        return self

    def predict_proba_up(self, X):
        r = X["ret_1"].to_numpy()
        return np.where(r > 0, 0.65, 0.35)


class RandomGuess(BaseModel):
    name = "random"
    calibrate = False

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, X, y, dates=None):
        return self

    def predict_proba_up(self, X):
        rng = np.random.default_rng(self.seed)
        return rng.uniform(0.0, 1.0, size=len(X))


# --------------------------------------------------------------------------
# Learned models
# --------------------------------------------------------------------------
@dataclass
class LogisticModel(BaseModel):
    """Logistic regression with standardization. Linear, fast and hard to
    overfit: the decent floor that boosting has to prove itself against."""

    C: float = 0.05
    name: str = "logistic"

    def fit(self, X, y, dates=None):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.cols_ = list(X.columns)
        self.pipe_ = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=self.C, max_iter=2000, solver="lbfgs"),
        )
        self.pipe_.fit(X, y)
        return self

    def predict_proba_up(self, X):
        return self.pipe_.predict_proba(X[self.cols_])[:, 1]


@dataclass
class LightGBMModel(BaseModel):
    """Gradient boosting with heavy regularization.

    The hyperparameters are deliberately conservative. Financial data has a low
    signal-to-noise ratio; a deep tree memorizes noise and the backtest looks
    great right up until it doesn't.
    """

    n_estimators: int = 400
    learning_rate: float = 0.02
    num_leaves: int = 15
    min_child_samples: int = 200
    subsample: float = 0.7
    colsample_bytree: float = 0.6
    reg_lambda: float = 10.0
    seed: int = 42
    name: str = "lightgbm"

    def fit(self, X, y, dates=None):
        import lightgbm as lgb

        self.cols_ = list(X.columns)
        self.model_ = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            subsample_freq=1,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            random_state=self.seed,
            n_jobs=-1,
            verbose=-1,
        )
        self.model_.fit(X, y)
        return self

    def predict_proba_up(self, X):
        return self.model_.predict_proba(X[self.cols_])[:, 1]

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model_.feature_importances_, index=self.cols_).sort_values(
            ascending=False
        )


class CalibratedModel(BaseModel):
    """Wraps a model and corrects its probabilities.

    Calibration uses the FINAL slice of the training window, never a random
    sample: mixing dates here leaks the future into the fit.

    The default method is Platt (sigmoid), not isotonic. Isotonic is more
    flexible, but with a weak signal it fits steps on top of handfuls of
    observations and emits "99.9% chance of an up day" from half a dozen points
    in the tail. A two-parameter sigmoid cannot do that.

    This matters because the system's useful output is not "up/down", it is
    "62% chance of up" -- and that number has to hold up 62% of the time.
    """

    def __init__(self, base: BaseModel, calibration_frac: float = 0.2, method: str = "sigmoid"):
        self.base = base
        self.calibration_frac = calibration_frac
        self.method = method
        self.name = f"{base.name}_cal"

    def fit(self, X, y, dates: pd.Series | None = None):
        n = len(X)
        n_cal = int(n * self.calibration_frac)
        if n_cal < 500 or dates is None:
            self.base.fit(X, y)
            self.calibrator_ = None
            return self

        # Split by DATE, not by row: the panel holds several tickers per day and
        # cutting mid-day would leave the same session on both sides.
        uniq = np.sort(pd.unique(dates))
        cut_idx = max(1, int(len(uniq) * (1 - self.calibration_frac)))
        cut_date = uniq[cut_idx]
        fit_mask = (dates < cut_date).to_numpy()
        cal_mask = ~fit_mask

        if cal_mask.sum() < 500 or len(np.unique(y[cal_mask])) < 2:
            self.base.fit(X, y)
            self.calibrator_ = None
            return self

        self.base.fit(X[fit_mask], y[fit_mask])
        raw = self.base.predict_proba_up(X[cal_mask])

        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            cal.fit(raw, y[cal_mask])
        else:
            from sklearn.linear_model import LogisticRegression

            cal = LogisticRegression()
            cal.fit(raw.reshape(-1, 1), y[cal_mask])

        self.calibrator_ = cal
        return self

    def predict_proba_up(self, X):
        raw = self.base.predict_proba_up(X)
        if self.calibrator_ is None:
            return raw
        if hasattr(self.calibrator_, "predict_proba"):
            return self.calibrator_.predict_proba(raw.reshape(-1, 1))[:, 1]
        return np.clip(self.calibrator_.predict(raw), 0.001, 0.999)


def default_models(seed: int = 42) -> list[BaseModel]:
    return [
        AlwaysUp(),
        PrevSign(),
        RandomGuess(seed=seed),
        LogisticModel(),
        LightGBMModel(seed=seed),
    ]
