"""Central configuration.

Every constant that changes backtest behaviour lives here, so that a result
published in a report can be reproduced from a single file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# Universes. The .SA suffix is Yahoo's convention for B3 (Brazilian) tickers.
UNIVERSE_BR = [
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA",
    "B3SA3.SA", "WEGE3.SA", "RENT3.SA", "BBAS3.SA", "SUZB3.SA",
    "PRIO3.SA", "RADL3.SA", "EQTL3.SA", "LREN3.SA", "GGBR4.SA",
]

UNIVERSE_US = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "JPM", "JNJ", "XOM", "WMT", "V", "PG", "UNH", "HD",
]

BENCHMARK = {"BR": "^BVSP", "US": "^GSPC"}

UNIVERSE = {"BR": UNIVERSE_BR, "US": UNIVERSE_US}

START_DATE = "2010-01-01"


@dataclass(frozen=True)
class BacktestConfig:
    """Walk-forward and transaction-cost parameters."""

    market: str = "BR"

    # How the target return is measured.
    #   close_to_close: decide at the close of t, hold until the close of t+1.
    #     Standard in the literature; assumes execution in the closing auction.
    #   open_to_close: decide at the close of t, enter at the open of t+1 and
    #     exit at the close of t+1. Fully executable, no auction assumption, but
    #     it throws away the overnight return.
    target_mode: str = "close_to_close"

    # Expanding walk-forward: train on everything that came before, test on the
    # next block, roll the window forward.
    min_train_days: int = 750          # ~3 years before the first test block
    test_block_days: int = 63          # ~1 quarter per fold
    embargo_days: int = 1              # drop the day(s) on the boundary

    # Final slice of the training window used only to calibrate probabilities.
    calibration_frac: float = 0.2

    # Transaction cost in basis points per side (spread + slippage +
    # commission). 5 bps per side = 10 bps on a full round trip.
    cost_bps_per_side: float = 5.0

    # Probability threshold for opening a position. Outside the band
    # [1-threshold, threshold] the system stays out of the market.
    prob_threshold: float = 0.55

    # Allow short positions when the model predicts a down day.
    allow_short: bool = False

    seed: int = 42

    @property
    def cost_round_trip(self) -> float:
        return 2 * self.cost_bps_per_side / 10_000.0


TRADING_DAYS_PER_YEAR = 252
