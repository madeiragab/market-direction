"""Build the self-contained HTML dashboard from the reports already produced.

Reads reports/backtest_<market>.json, reports/latest_predictions.json and
reports/recent_history.json, assembles a lean payload and injects it into the
template. The resulting file needs no network and no server: it opens straight
in a browser.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401

from marketdir.config import REPORTS, ROOT

TEMPLATE = ROOT / "dashboard" / "template.html"
OUTPUT = REPORTS / "dashboard.html"

# Series drawn on the equity chart. always_up is deliberately omitted: it traces
# the buy & hold curve exactly and would only clutter the read.
EQUITY_SERIES = ["logistic", "lightgbm", "prev_sign", "random", "buy_and_hold"]

# The equity chart is about 800px wide. Keeping 3,000 points per series only
# bloats the HTML and stalls rasterization: at two points per pixel the curve is
# pixel-identical.
MAX_POINTS = 700


def _downsample(values: list, step: int) -> list:
    """Keep one point every `step`, always preserving the last one.

    The last point matters because it is the final cumulative return -- exactly
    the number a reader looks for at the end of the curve.
    """
    out = values[::step]
    if out[-1] != values[-1]:
        out.append(values[-1])
    return out


def compact_market(payload: dict) -> dict:
    """Reduce the backtest JSON to what the dashboard actually draws."""
    eq = payload["equity"]
    series = {k: v for k, v in eq["series"].items() if k in EQUITY_SERIES}

    step = max(1, -(-len(eq["dates"]) // MAX_POINTS))
    if step > 1:
        eq = {
            "dates": _downsample(eq["dates"], step),
            "series": {k: _downsample(v, step) for k, v in series.items()},
        }
        series = eq["series"]

    return {
        "market": payload["market"],
        "config": payload["config"] | {"embargo_days": 1},
        "oos_start": payload["oos_start"],
        "oos_end": payload["oos_end"],
        "n_predictions": payload["n_predictions"],
        "n_folds": len(payload["per_fold"]),
        "best_model": payload["best_model"],
        "results": payload["results"],
        "equity": {"dates": eq["dates"], "series": series},
        "reliability": payload["reliability"],
        "per_fold": payload["per_fold"],
        "per_ticker": payload["per_ticker"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", nargs="+", default=["BR", "US"])
    ap.add_argument("--out", default=str(OUTPUT))
    args = ap.parse_args()

    markets: dict[str, dict] = {}
    for market in args.markets:
        path = REPORTS / f"backtest_{market}.json"
        if not path.exists():
            raise SystemExit(f"missing {path}. Run scripts/run_backtest.py --market {market}")
        markets[market] = compact_market(json.loads(path.read_text(encoding="utf-8")))

    latest_path = REPORTS / "latest_predictions.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8")) if latest_path.exists() else None

    hist_path = REPORTS / "recent_history.json"
    history = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.exists() else None

    data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "markets": markets,
        "latest": latest,
        "history": history,
    }

    html = TEMPLATE.read_text(encoding="utf-8")
    if "/*__PAYLOAD__*/" not in html:
        raise SystemExit("template is missing the /*__PAYLOAD__*/ marker")

    # No spaces in the separators: the payload is large and every byte ships
    # inside the HTML.
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # A literal </script> inside the JSON would close the tag early.
    blob = blob.replace("</", "<\\/")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace("/*__PAYLOAD__*/", blob), encoding="utf-8")

    size_kb = out.stat().st_size / 1024
    print(f"{out}  ({size_kb:,.0f} KB, {len(markets)} markets)")


if __name__ == "__main__":
    main()
