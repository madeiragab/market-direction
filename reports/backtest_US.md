# Backtest -- US market

Target: next-day direction (close_to_close). Validation: expanding walk-forward, blocks of 63 sessions, 1-day embargo. Cost: 5.0 bps per side. Entry threshold: 0.55.

Out-of-sample period: 2013-12-26 to 2026-08-31 (51 folds, 47,805 predictions).


## Classification (all out of sample)

| model | accuracy | balanced acc. | AUC | Brier | p vs coin |
|---|---|---|---|---|---|
| always_up | 52.77% | 50.00% | 0.500 | 0.463 | 0.0000 |
| prev_sign | 49.29% | 49.14% | 0.491 | 0.275 | 0.9990 |
| random | 50.24% | 50.25% | 0.502 | 0.334 | 0.1528 |
| logistic | 52.55% | 49.94% | 0.503 | 0.249 | 0.0000 |
| lightgbm | 52.63% | 50.07% | 0.494 | 0.250 | 0.0000 |

Natural share of up days in the period: **52.77%**. Any accuracy below that loses to the rule of always guessing up.


## Strategy, net of cost

| model | CAGR | vol | Sharpe | Sharpe 95% CI | max DD | avg turnover | days in market |
|---|---|---|---|---|---|---|---|
| always_up | 24.52% | 18.78% | 1.26 | [0.75, 1.87] | -31.24% | 0.00 | 100.00% |
| prev_sign | 2.19% | 19.64% | 0.21 | [-0.33, 0.82] | -40.55% | 1.18 | 96.93% |
| random | 4.73% | 19.77% | 0.33 | [-0.15, 0.85] | -39.10% | 1.20 | 100.00% |
| logistic | 4.14% | 16.41% | 0.33 | [-0.17, 0.86] | -47.79% | 0.28 | 20.71% |
| lightgbm | -0.89% | 11.04% | -0.03 | [-0.44, 0.43] | -37.36% | 0.26 | 18.26% |
| **buy & hold** | 24.52% | 18.77% | 1.26 | -- | -31.24% | -- | 100.00% |

## Cost matters

| model | gross Sharpe | net Sharpe | lost to cost |
|---|---|---|---|
| always_up | 1.26 | 1.26 | 0.00 |
| prev_sign | 0.97 | 0.21 | 0.76 |
| random | 1.10 | 0.33 | 0.76 |
| logistic | 0.54 | 0.33 | 0.21 |
| lightgbm | 0.27 | -0.03 | 0.30 |

## Calibration of the best model (always_up)

If the left column and the middle column move together, the probability means something. If they drift apart, the number is decoration.

| predicted probability bucket | observed frequency of up days | n |
|---|---|---|

## Accuracy per ticker (always_up)

| ticker | accuracy | up rate | n |
|---|---|---|---|
| V | 54.16% | 54.16% | 3,187 |
| NVDA | 54.03% | 54.03% | 3,187 |
| AAPL | 53.22% | 53.22% | 3,187 |
| GOOGL | 53.18% | 53.18% | 3,187 |
| MSFT | 53.09% | 53.09% | 3,187 |
| AMZN | 53.03% | 53.03% | 3,187 |
| WMT | 52.97% | 52.97% | 3,187 |
| HD | 52.75% | 52.75% | 3,187 |
| UNH | 52.65% | 52.65% | 3,187 |
| JPM | 52.65% | 52.65% | 3,187 |
| META | 52.56% | 52.56% | 3,187 |
| PG | 52.56% | 52.56% | 3,187 |
| JNJ | 52.37% | 52.37% | 3,187 |
| TSLA | 51.46% | 51.46% | 3,187 |
| XOM | 50.96% | 50.96% | 3,187 |

## Features most used by LightGBM

| feature | mean importance |
|---|---|
| bench_vol_21 | 537.6 |
| bench_ret_5 | 516.4 |
| bench_vol_ratio | 437.2 |
| bench_ret_1 | 432.4 |
| bench_ret_21 | 425.7 |
| gap | 197.1 |
| month | 178.0 |
| dow | 166.0 |
| clv | 142.2 |
| dollar_vol_z | 134.5 |
| ret_1 | 125.2 |
| ret_2 | 104.4 |
| mom_12_1 | 102.4 |
| ret_3 | 101.7 |
| ret_10 | 100.3 |
| vol_ratio_21 | 100.0 |
| vol_10 | 94.0 |
| ret_63 | 92.5 |
| hl_range | 86.7 |
| vol_63 | 86.4 |

## How to read these numbers

- 52-54% directional accuracy at D+1 is the realistic ceiling in the literature. A number far above that almost always means leakage, not talent.
- The p-value compares the accuracy against flipping a coin. Above 0.05, what you see fits inside chance.
- The Sharpe confidence interval comes from a block bootstrap, which preserves autocorrelation. If the interval crosses zero, the strategy proved nothing.
- Transaction cost is not a detail: it is the line that kills high-turnover strategies.