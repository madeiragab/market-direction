"""Export each ticker's recent history together with the model's call.

For every session in the last few months the file stores the candle (open,
high, low, close, volume) and, where one exists, the probability the model gave
that day plus the following day's outcome. That is what makes it possible to
look at the bar and the call side by side, instead of trusting an aggregate
accuracy.

Every probability comes from the OUT-OF-SAMPLE prediction file produced by the
walk-forward. None of them was produced by a model that had already seen that
day.
"""
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

import pandas as pd

from marketdir.config import DATA_PROCESSED, REPORTS
from marketdir.data import load_market

DEFAULT_DAYS = 66  # ~3 months of sessions


def export_market(market: str, days: int, model: str) -> dict:
    panel = load_market(market)

    # The last bar is usually the session in progress: zero volume, high equal
    # to low. Drawing that as a candle misrepresents the day.
    panel = panel[panel["volume"] > 0]

    dates = sorted(panel["date"].unique())[-days:]
    panel = panel[panel["date"].isin(dates)]

    preds_path = DATA_PROCESSED / f"oos_predictions_{market}.parquet"
    prob_col = f"prob__{model}"
    if preds_path.exists():
        preds = pd.read_parquet(preds_path)
        if prob_col in preds.columns:
            preds = preds[["date", "ticker", prob_col, "target", "fwd_ret"]]
            panel = panel.merge(preds, on=["date", "ticker"], how="left")

    out: dict[str, list] = {}
    for ticker, grp in panel.groupby("ticker", sort=True):
        grp = grp.sort_values("date")
        rows = []
        for _, r in grp.iterrows():
            row = {
                "d": str(r["date"])[:10],
                "o": round(float(r["open"]), 2),
                "h": round(float(r["high"]), 2),
                "l": round(float(r["low"]), 2),
                "c": round(float(r["close"]), 2),
                "v": int(r["volume"]),
            }
            if prob_col in grp.columns and pd.notna(r.get(prob_col)):
                row["p"] = round(float(r[prob_col]), 4)
                row["y"] = int(r["target"])
                row["r"] = round(float(r["fwd_ret"]), 5)
            rows.append(row)
        out[ticker] = rows

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", nargs="+", default=["BR", "US"])
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS)
    ap.add_argument("--model", default="logistic")
    args = ap.parse_args()

    payload = {"model": args.model, "days": args.days, "markets": {}}
    for market in args.markets:
        data = export_market(market, args.days, args.model)
        payload["markets"][market] = data
        n_days = max((len(v) for v in data.values()), default=0)
        n_calls = sum(1 for v in data.values() for row in v if "p" in row)
        print(f"{market}: {len(data)} tickers | {n_days} sessions | {n_calls} model calls")

    path = REPORTS / "recent_history.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
