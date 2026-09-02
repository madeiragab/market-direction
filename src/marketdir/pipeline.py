"""Glue between ingestion, features and labels."""
from __future__ import annotations

import pandas as pd

from .data import load_benchmark, load_market
from .features import build_features, feature_columns
from .labels import add_target, clean_panel


def build_dataset(market: str, target_mode: str = "close_to_close") -> tuple[pd.DataFrame, list[str]]:
    """Panel ready for modelling: features + target, with no incomplete rows."""
    panel = load_market(market)
    bench = load_benchmark(market)

    feats = build_features(panel, bench)
    feats = add_target(feats, mode=target_mode)

    cols = feature_columns(feats)
    clean = clean_panel(feats, cols)
    return clean, cols


def build_features_only(market: str) -> tuple[pd.DataFrame, list[str]]:
    """Same as above, but keeping the last row of every ticker.

    That is the row without a target -- exactly the one needed to predict
    tomorrow.
    """
    panel = load_market(market)
    bench = load_benchmark(market)
    feats = build_features(panel, bench)
    return feats, feature_columns(feats)
