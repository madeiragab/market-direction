# Backtest -- BR market

Target: next-day direction (close_to_close). Validation: expanding walk-forward, blocks of 63 sessions, 1-day embargo. Cost: 5.0 bps per side. Entry threshold: 0.55.

Out-of-sample period: 2014-07-28 to 2026-08-31 (45 folds, 40,169 predictions).


## Classification (all out of sample)

| model | accuracy | balanced acc. | AUC | Brier | p vs coin |
|---|---|---|---|---|---|
| always_up | 49.80% | 50.00% | 0.500 | 0.492 | 0.7877 |
| prev_sign | 49.04% | 49.03% | 0.490 | 0.275 | 0.9999 |
| random | 50.41% | 50.41% | 0.505 | 0.332 | 0.0498 |
| logistic | 50.60% | 50.59% | 0.510 | 0.250 | 0.0079 |
| lightgbm | 50.39% | 50.36% | 0.504 | 0.250 | 0.0598 |

Natural share of up days in the period: **49.80%**. Any accuracy below that loses to the rule of always guessing up.


## Strategy, net of cost

| model | CAGR | vol | Sharpe | Sharpe 95% CI | max DD | avg turnover | days in market |
|---|---|---|---|---|---|---|---|
| always_up | 16.44% | 24.69% | 0.74 | [0.13, 1.44] | -44.26% | 0.01 | 100.00% |
| prev_sign | -4.42% | 27.28% | -0.03 | [-0.73, 0.59] | -69.31% | 1.26 | 97.37% |
| random | 0.48% | 27.02% | 0.15 | [-0.41, 0.77] | -55.40% | 1.21 | 99.93% |
| logistic | -0.02% | 20.13% | 0.10 | [-0.32, 0.49] | -41.31% | 0.06 | 5.05% |
| lightgbm | -2.96% | 13.61% | -0.15 | [-0.56, 0.38] | -48.05% | 0.09 | 5.69% |
| **buy & hold** | 16.56% | 24.69% | 0.75 | -- | -44.26% | -- | 100.00% |

## Cost matters

| model | gross Sharpe | net Sharpe | lost to cost |
|---|---|---|---|
| always_up | 0.75 | 0.74 | 0.00 |
| prev_sign | 0.55 | -0.03 | 0.58 |
| random | 0.72 | 0.15 | 0.56 |
| logistic | 0.14 | 0.10 | 0.04 |
| lightgbm | -0.07 | -0.15 | 0.09 |

## Calibration of the best model (logistic)

If the left column and the middle column move together, the probability means something. If they drift apart, the number is decoration.

| predicted probability bucket | observed frequency of up days | n |
|---|---|---|
| 45.37% | 49.04% | 4,017 |
| 47.89% | 48.87% | 4,017 |
| 48.67% | 48.37% | 4,017 |
| 49.15% | 48.87% | 4,017 |
| 49.59% | 50.86% | 4,017 |
| 50.06% | 49.10% | 4,016 |
| 50.55% | 48.79% | 4,017 |
| 51.07% | 50.66% | 4,017 |
| 51.76% | 52.38% | 4,017 |
| 53.54% | 51.08% | 4,017 |

## Accuracy per ticker (logistic)

| ticker | accuracy | up rate | n |
|---|---|---|---|
| WEGE3.SA | 53.17% | 50.43% | 2,701 |
| RENT3.SA | 52.49% | 48.92% | 2,766 |
| PRIO3.SA | 51.76% | 49.11% | 2,645 |
| VALE3.SA | 51.20% | 49.98% | 2,709 |
| LREN3.SA | 51.18% | 49.26% | 2,710 |
| ABEV3.SA | 50.76% | 48.99% | 2,764 |
| EQTL3.SA | 50.72% | 50.25% | 2,764 |
| BBDC4.SA | 50.56% | 50.22% | 2,700 |
| B3SA3.SA | 50.54% | 49.24% | 2,699 |
| SUZB3.SA | 50.36% | 48.09% | 2,073 |
| BBAS3.SA | 49.81% | 50.19% | 2,580 |
| RADL3.SA | 49.71% | 50.36% | 2,764 |
| ITUB4.SA | 49.30% | 50.49% | 2,769 |
| GGBR4.SA | 48.88% | 49.06% | 2,764 |
| PETR4.SA | 48.64% | 52.01% | 2,761 |

## Features most used by LightGBM

| feature | mean importance |
|---|---|
| bench_ret_5 | 442.5 |
| bench_vol_21 | 437.9 |
| bench_ret_21 | 395.1 |
| bench_ret_1 | 392.1 |
| bench_vol_ratio | 387.4 |
| clv | 221.9 |
| month | 187.2 |
| gap | 156.9 |
| dollar_vol_z | 148.0 |
| ret_1 | 139.2 |
| mom_12_1 | 128.4 |
| ret_2 | 120.9 |
| dow | 119.7 |
| ret_3 | 115.1 |
| vol_5 | 108.7 |
| vol_ratio_5_63 | 101.9 |
| ret_63 | 100.2 |
| ret_21_norm | 99.2 |
| vol_ratio_21 | 94.7 |
| sma_ratio_200 | 89.4 |

## How to read these numbers

- 52-54% directional accuracy at D+1 is the realistic ceiling in the literature. A number far above that almost always means leakage, not talent.
- The p-value compares the accuracy against flipping a coin. Above 0.05, what you see fits inside chance.
- The Sharpe confidence interval comes from a block bootstrap, which preserves autocorrelation. If the interval crosses zero, the strategy proved nothing.
- Transaction cost is not a detail: it is the line that kills high-turnover strategies.