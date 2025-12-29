# Backtest Report

**Symbol:** BTCUSDT | **Interval:** 1m | **Duration:** 24.0h

**Period:** 2025-12-28T14:33:02Z -> 2025-12-29T14:33:02Z


## Quick Summary

| Metric | Value | Status |
|--------|-------|--------|
| **Accuracy** | 49.9% (718/1439) | ~ |
| **ECE** | 5.3% | + |
| **Confident Wrong** | 50.0% | - |
| **Direction Balance** | 0.96 | + |
| **Win Rate** | 49.4% | ~ |
| **Sharpe Ratio** | 7.0 | + |

## Warnings

- **Overconfident** - too many confident wrong predictions

## Direction Distribution

| | UP | DOWN | FLAT |
|---|---:|---:|---:|
| Predicted | 690 | 749 | 0 |
| Actual | 767 | 672 | 0 |

## Confusion Matrix

| pred \ actual | UP | DOWN | FLAT |
|---|---:|---:|---:|
| **UP** | 368 | 322 | 0 |
| **DOWN** | 399 | 350 | 0 |
| **FLAT** | 0 | 0 | 0 |

## Per-Model Performance

| Model | Accuracy | UP | DOWN | FLAT | Balance |
|-------|----------|---:|-----:|-----:|---------|
| TRENDVIC | 50.2% | 667 | 727 | 45 | 0.96 + |
| OSCILLATOR | 46.1% | 698 | 622 | 119 | 0.94 + |
| VOLUMEMETRIX | 46.4% | 768 | 671 | 0 | 0.93 + |

## Model Agreement

| Agreement | Count | Accuracy |
|-----------|------:|----------|
| Full | 83 | 51.8% |
| Partial | 0 | 0.0% |
| None | 1342 | 49.8% |

## Trading Simulation

- **Trades:** 1342 (Win: 663, Loss: 679)
- **Cumulative Return:** +0.6645%
- **Max Drawdown:** 1.6368%
- **Profit Factor:** 1.03
- **Sharpe:** 7.0 | **Sortino:** 7.3

## Accuracy Stability

- **Range:** 34.0% - 64.0%
- **Mean:** 50.0% ± 5.7%
- **Trend:** +0.001 -

## Feature Effectiveness (Top by correlation)

| Scope | Feature | Correlation | Effect Size | Count |
|-------|---------|------------:|------------:|------:|
| Overall | return_1 | -0.0677 | -0.1358 | 1439 |
| Overall | log_return_1 | -0.0677 | -0.1357 | 1439 |
| Overall | ma_slow | -0.0552 | -0.1106 | 1439 |
| Overall | close | -0.0548 | -0.1099 | 1439 |
| Overall | ma_fast | -0.0531 | -0.1064 | 1439 |
| TRENDVIC | rsi | 0.0601 | 0.1201 | 1439 |
| TRENDVIC | volume_z | -0.0368 | -0.0736 | 1439 |
| TRENDVIC | volume | -0.0315 | -0.0629 | 1439 |
| TRENDVIC | ma_delta | 0.0212 | 0.0424 | 1439 |
| TRENDVIC | return_1 | 0.0174 | 0.0348 | 1439 |
| OSCILLATOR | rsi | -0.0610 | -0.1224 | 1439 |
| OSCILLATOR | volume | 0.0471 | 0.0945 | 1439 |
| OSCILLATOR | volume_z | 0.0268 | 0.0537 | 1439 |
| OSCILLATOR | ma_delta | -0.0201 | -0.0404 | 1439 |
| OSCILLATOR | volatility | 0.0125 | 0.0250 | 1439 |
| VOLUMEMETRIX | volatility_z | -0.0609 | -0.1221 | 1439 |
| VOLUMEMETRIX | volatility | -0.0549 | -0.1100 | 1439 |
| VOLUMEMETRIX | ma_slow | -0.0346 | -0.0694 | 1439 |
| VOLUMEMETRIX | ma_fast | -0.0323 | -0.0647 | 1439 |
| VOLUMEMETRIX | close | -0.0317 | -0.0636 | 1439 |

## Statistical Significance

- **Accuracy:** 49.9% (718/1439)
- **95% CI:** 47.3% - 52.5%
- **p-value:** 0.6
- **vs Random:** - Not significantly different from random (50%)

## Accuracy by Session

| Session | Accuracy | Count |
|---------|----------|------:|
| ASIA | 48.5% | 480 |
| EUROPE | 50.6% | 360 |
| US | 50.6% | 599 |

- **Best hour:** 3:00 UTC (60.0%)
- **Worst hour:** 7:00 UTC (40.0%)

## Accuracy by Movement Size

| Size | Accuracy | Count | % of Total |
|------|----------|------:|------------|
| tiny | 51.0% | 457 | 31.8% |
| small | 50.0% | 724 | 50.3% |
| medium | 44.6% | 186 | 12.9% |
| large | 55.6% | 72 | 5.0% |

## Error Streaks (3+)

- **Count:** 96
- **Max length:** 7
- **Avg length:** 3.8
- **Common regime:** ranging (30 streaks)

## Confident Wrong Predictions (conf >= 65%)

- **Count:** 41
- **Avg confidence:** 72.0%

## Error Rate by Regime

| Regime | Errors | Total | Error Rate |
|--------|-------:|------:|------------|
| volatile | 129 | 229 | 56.3% |
| trending_down | 216 | 401 | 53.9% |
| ranging | 227 | 485 | 46.8% |
| trending_up | 149 | 324 | 46.0% |