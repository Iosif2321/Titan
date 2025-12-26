# Offline Replay Report
- symbol: BTCUSDT
- tfs: 1
- mode: smoke
- range: 2025-12-26T21:41:39.065000 -> 2025-12-26T23:41:39.065000
- candles: 119
- predictions: 357 facts: 354 updates: 354 pending_tail: 3
- checks: FAIL (4 errors)

## Failed Checks
- overconfidence_guard overall avg_conf=0.586 acc=0.237 ece=0.203
- overconfidence_guard 1:TRENDVIC avg_conf=0.595 acc=0.347 ece=0.156
- overconfidence_guard 1:OSCILLATOR avg_conf=0.580 acc=0.186 ece=0.268
- overconfidence_guard 1:VOLUMEMETRIX avg_conf=0.582 acc=0.178 ece=0.269

## Overall
- accuracy: 0.237 nonflat: 0.431 flat_rate: 0.449 avg_conf: 0.586
- calibration: ece=0.203 mce=0.941 brier=0.307
- confident_wrong_rate: 0.023
- adaptation: {'avg_lr_eff': 0.0009215349927975185, 'avg_anchor_lambda': 0.00010053534400319996, 'anchor_update_rate': 0.2937853107344633, 'avg_params_norm': 4.190838757664809, 'avg_anchor_norm': 4.242588701937928, 'avg_params_anchor_gap': 0.051770329604762286}

### Confusion (Overall)
pred\fact | UP | DOWN | FLAT
---|---:|---:|---:
UP | 53 | 73 | 0
DOWN | 38 | 31 | 0
FLAT | 89 | 70 | 0

### Calibration Bins (Overall)
bin | count | acc | avg_conf | avg_abs_ret_bps
---|---:|---:|---:|---:
0 | 0 | 0.000 | 0.000 | 0.000
1 | 0 | 0.000 | 0.000 | 0.000
2 | 0 | 0.000 | 0.000 | 0.000
3 | 0 | 0.000 | 0.000 | 0.000
4 | 0 | 0.000 | 0.000 | 0.000
5 | 85 | 0.494 | 0.579 | 2.324
6 | 81 | 0.444 | 0.639 | 2.882
7 | 20 | 0.250 | 0.746 | 3.081
8 | 8 | 0.125 | 0.846 | 3.078
9 | 1 | 0.000 | 0.941 | 6.294

### Top Confident-Wrong (Overall)
ts | tf | model | pred | fact | conf | ret_bps
---|---|---|---|---|---:|---:
1766781709636 | 1 | TRENDVIC | UP | DOWN | 0.941 | -6.294
1766781713050 | 1 | VOLUMEMETRIX | UP | DOWN | 0.887 | -4.564
1766781710650 | 1 | VOLUMEMETRIX | UP | DOWN | 0.882 | -2.102
1766781705927 | 1 | OSCILLATOR | DOWN | UP | 0.874 | 5.772
1766781705846 | 1 | TRENDVIC | UP | DOWN | 0.850 | -0.757
1766781705784 | 1 | TRENDVIC | UP | DOWN | 0.820 | -7.797
1766781712604 | 1 | VOLUMEMETRIX | UP | DOWN | 0.816 | -1.438
1766781712569 | 1 | TRENDVIC | UP | DOWN | 0.804 | -1.438

## Per Model
### 1:TRENDVIC
- accuracy: 0.347 nonflat: 0.526 flat_rate: 0.339 avg_conf: 0.595
- calibration: ece=0.156 mce=0.941 brier=0.284
- confident_wrong_rate: 0.034
- adaptation: {'avg_lr_eff': 0.0009593787313555022, 'avg_anchor_lambda': 0.00010015242852991336, 'anchor_update_rate': 0.3559322033898305, 'avg_params_norm': 4.223165321120155, 'avg_anchor_norm': 4.242616318040162, 'avg_params_anchor_gap': 0.019474882885809723}

Confusion:
pred\fact | UP | DOWN | FLAT
---|---:|---:|---:
UP | 21 | 19 | 0
DOWN | 18 | 20 | 0
FLAT | 21 | 19 | 0

Calibration bins:
bin | count | acc | avg_conf | avg_abs_ret_bps
---|---:|---:|---:|---:
0 | 0 | 0.000 | 0.000 | 0.000
1 | 0 | 0.000 | 0.000 | 0.000
2 | 0 | 0.000 | 0.000 | 0.000
3 | 0 | 0.000 | 0.000 | 0.000
4 | 0 | 0.000 | 0.000 | 0.000
5 | 31 | 0.645 | 0.579 | 2.231
6 | 38 | 0.474 | 0.632 | 2.881
7 | 5 | 0.600 | 0.737 | 2.050
8 | 3 | 0.000 | 0.825 | 3.331
9 | 1 | 0.000 | 0.941 | 6.294

Top confident-wrong:
ts | tf | model | pred | fact | conf | ret_bps
---|---|---|---|---|---:|---:
1766781709636 | 1 | TRENDVIC | UP | DOWN | 0.941 | -6.294
1766781705846 | 1 | TRENDVIC | UP | DOWN | 0.850 | -0.757
1766781705784 | 1 | TRENDVIC | UP | DOWN | 0.820 | -7.797
1766781712569 | 1 | TRENDVIC | UP | DOWN | 0.804 | -1.438
### 1:OSCILLATOR
- accuracy: 0.186 nonflat: 0.361 flat_rate: 0.483 avg_conf: 0.580
- calibration: ece=0.268 mce=0.592 brier=0.309
- confident_wrong_rate: 0.008
- adaptation: {'avg_lr_eff': 0.0009129191744543531, 'avg_anchor_lambda': 0.00010048176376236374, 'anchor_update_rate': 0.2627118644067797, 'avg_params_norm': 4.154870028168911, 'avg_anchor_norm': 4.242552684615076, 'avg_params_anchor_gap': 0.08768265644616265}

Confusion:
pred\fact | UP | DOWN | FLAT
---|---:|---:|---:
UP | 18 | 33 | 0
DOWN | 6 | 4 | 0
FLAT | 36 | 21 | 0

Calibration bins:
bin | count | acc | avg_conf | avg_abs_ret_bps
---|---:|---:|---:|---:
0 | 0 | 0.000 | 0.000 | 0.000
1 | 0 | 0.000 | 0.000 | 0.000
2 | 0 | 0.000 | 0.000 | 0.000
3 | 0 | 0.000 | 0.000 | 0.000
4 | 0 | 0.000 | 0.000 | 0.000
5 | 30 | 0.333 | 0.577 | 2.279
6 | 22 | 0.455 | 0.645 | 2.875
7 | 7 | 0.143 | 0.735 | 2.731
8 | 2 | 0.500 | 0.855 | 3.265
9 | 0 | 0.000 | 0.000 | 0.000

Top confident-wrong:
ts | tf | model | pred | fact | conf | ret_bps
---|---|---|---|---|---:|---:
1766781705927 | 1 | OSCILLATOR | DOWN | UP | 0.874 | 5.772
### 1:VOLUMEMETRIX
- accuracy: 0.178 nonflat: 0.375 flat_rate: 0.525 avg_conf: 0.582
- calibration: ece=0.269 mce=0.862 brier=0.336
- confident_wrong_rate: 0.025
- adaptation: {'avg_lr_eff': 0.0008923070725826989, 'avg_anchor_lambda': 0.00010097183971732303, 'anchor_update_rate': 0.2627118644067797, 'avg_params_norm': 4.19448092370535, 'avg_anchor_norm': 4.242597103158571, 'avg_params_anchor_gap': 0.04815344948231447}

Confusion:
pred\fact | UP | DOWN | FLAT
---|---:|---:|---:
UP | 14 | 21 | 0
DOWN | 14 | 7 | 0
FLAT | 32 | 30 | 0

Calibration bins:
bin | count | acc | avg_conf | avg_abs_ret_bps
---|---:|---:|---:|---:
0 | 0 | 0.000 | 0.000 | 0.000
1 | 0 | 0.000 | 0.000 | 0.000
2 | 0 | 0.000 | 0.000 | 0.000
3 | 0 | 0.000 | 0.000 | 0.000
4 | 0 | 0.000 | 0.000 | 0.000
5 | 24 | 0.500 | 0.579 | 2.500
6 | 21 | 0.381 | 0.643 | 2.891
7 | 8 | 0.125 | 0.760 | 4.033
8 | 3 | 0.000 | 0.862 | 2.701
9 | 0 | 0.000 | 0.000 | 0.000

Top confident-wrong:
ts | tf | model | pred | fact | conf | ret_bps
---|---|---|---|---|---:|---:
1766781713050 | 1 | VOLUMEMETRIX | UP | DOWN | 0.887 | -4.564
1766781710650 | 1 | VOLUMEMETRIX | UP | DOWN | 0.882 | -2.102
1766781712604 | 1 | VOLUMEMETRIX | UP | DOWN | 0.816 | -1.438

## Patterns
- 1:TRENDVIC: total=25 context=7 decision=18 coverage={'total': 25, 'context_fine': 0, 'context_coarse': 7, 'decision_fine': 0, 'decision_coarse': 18, 'unknown': 0}
- 1:OSCILLATOR: total=14 context=5 decision=9 coverage={'total': 14, 'context_fine': 0, 'context_coarse': 5, 'decision_fine': 0, 'decision_coarse': 9, 'unknown': 0}
- 1:VOLUMEMETRIX: total=13 context=4 decision=9 coverage={'total': 13, 'context_fine': 0, 'context_coarse': 4, 'decision_fine': 0, 'decision_coarse': 9, 'unknown': 0}
