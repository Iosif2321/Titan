# Offline Replay Report
- symbol: BTCUSDT
- tfs: 1
- mode: smoke
- range: 2025-12-27T14:00:51.440000 -> 2025-12-27T16:00:51.440000
- fact_flat_bps: 1.0
- candles: 119
- predictions: 357 facts: 354 updates: 354 pending_tail: 3
- checks: FAIL (4 errors)

## Failed Checks
- overconfidence_guard overall avg_conf=0.817 acc=0.249 ece=0.594
- overconfidence_guard 1:TRENDVIC avg_conf=0.703 acc=0.220 ece=0.516
- overconfidence_guard 1:OSCILLATOR avg_conf=0.882 acc=0.263 ece=0.657
- overconfidence_guard 1:VOLUMEMETRIX avg_conf=0.866 acc=0.263 ece=0.605

## Overall
- accuracy: 0.249 nonflat: 0.580 nonflat_swapped: 0.355 inversion_delta: -0.225 flat_rate: 0.610 avg_conf: 0.817
- calibration: ece=0.594 mce=0.675 brier=0.535
- confident_wrong_rate: 0.381
- adaptation: {'avg_lr_eff': 0.0009166751547526045, 'avg_anchor_lambda': 0.00010137624788330862, 'anchor_update_rate': 0.4265536723163842, 'avg_params_norm': 4.196372158001044, 'avg_anchor_norm': 4.242556863003641, 'avg_params_anchor_gap': 0.04622314209797969}

### Confusion (Overall)
pred\fact | UP | DOWN | FLAT
---|---:|---:|---:
UP | 40 | 19 | 101
DOWN | 30 | 40 | 107
FLAT | 5 | 4 | 8

### Calibration Bins (Overall)
bin | count | acc | avg_conf | avg_abs_ret_bps
---|---:|---:|---:|---:
0 | 0 | 0.000 | 0.000 | 0.000
1 | 0 | 0.000 | 0.000 | 0.000
2 | 0 | 0.000 | 0.000 | 0.000
3 | 0 | 0.000 | 0.000 | 0.000
4 | 0 | 0.000 | 0.000 | 0.000
5 | 28 | 0.143 | 0.574 | 0.562
6 | 52 | 0.115 | 0.648 | 0.802
7 | 65 | 0.200 | 0.749 | 0.635
8 | 41 | 0.293 | 0.852 | 1.118
9 | 151 | 0.298 | 0.973 | 0.965

### Top Confident-Wrong (Overall)
ts | tf | model | pred | fact | conf | ret_bps
---|---|---|---|---|---:|---:
1766840464304 | 1 | VOLUMEMETRIX | DOWN | FLAT | 1.000 | -0.708
1766840462679 | 1 | VOLUMEMETRIX | DOWN | FLAT | 1.000 | -0.994
1766840469290 | 1 | VOLUMEMETRIX | DOWN | FLAT | 1.000 | 0.011
1766840464148 | 1 | VOLUMEMETRIX | DOWN | UP | 1.000 | 1.006
1766840460597 | 1 | OSCILLATOR | DOWN | FLAT | 1.000 | 0.000
1766840464910 | 1 | VOLUMEMETRIX | UP | DOWN | 1.000 | -4.411
1766840462151 | 1 | OSCILLATOR | UP | FLAT | 0.999 | 0.343
1766840464579 | 1 | VOLUMEMETRIX | UP | FLAT | 0.999 | 0.206
1766840460526 | 1 | OSCILLATOR | DOWN | FLAT | 0.999 | 0.240
1766840462217 | 1 | OSCILLATOR | UP | FLAT | 0.999 | -0.011

## Per Model
### 1:TRENDVIC
- accuracy: 0.220 nonflat: 0.478 nonflat_swapped: 0.391 inversion_delta: -0.087 flat_rate: 0.610 avg_conf: 0.703
- calibration: ece=0.516 mce=0.615 brier=0.430
- confident_wrong_rate: 0.093
- adaptation: {'avg_lr_eff': 0.0009368729187688847, 'avg_anchor_lambda': 0.0001003413308461601, 'anchor_update_rate': 0.288135593220339, 'avg_params_norm': 4.2118193526546905, 'avg_anchor_norm': 4.242610139428458, 'avg_params_anchor_gap': 0.030899873370500364}

Confusion:
pred\fact | UP | DOWN | FLAT
---|---:|---:|---:
UP | 8 | 4 | 20
DOWN | 14 | 14 | 48
FLAT | 3 | 3 | 4

Calibration bins:
bin | count | acc | avg_conf | avg_abs_ret_bps
---|---:|---:|---:|---:
0 | 0 | 0.000 | 0.000 | 0.000
1 | 0 | 0.000 | 0.000 | 0.000
2 | 0 | 0.000 | 0.000 | 0.000
3 | 0 | 0.000 | 0.000 | 0.000
4 | 0 | 0.000 | 0.000 | 0.000
5 | 17 | 0.176 | 0.574 | 0.805
6 | 27 | 0.148 | 0.647 | 0.798
7 | 46 | 0.174 | 0.750 | 0.592
8 | 9 | 0.444 | 0.835 | 1.575
9 | 9 | 0.333 | 0.948 | 1.292

Top confident-wrong:
ts | tf | model | pred | fact | conf | ret_bps
---|---|---|---|---|---:|---:
1766840464270 | 1 | TRENDVIC | UP | FLAT | 0.978 | -0.708
1766840469208 | 1 | TRENDVIC | UP | FLAT | 0.944 | 0.011
1766840469967 | 1 | TRENDVIC | UP | FLAT | 0.940 | 0.011
1766840464400 | 1 | TRENDVIC | UP | DOWN | 0.940 | -1.211
1766840462640 | 1 | TRENDVIC | UP | FLAT | 0.922 | -0.994
1766840461267 | 1 | TRENDVIC | UP | DOWN | 0.902 | -3.747
1766840467891 | 1 | TRENDVIC | UP | DOWN | 0.897 | -1.337
1766840460280 | 1 | TRENDVIC | DOWN | UP | 0.856 | 1.131
1766840462133 | 1 | TRENDVIC | DOWN | FLAT | 0.823 | 0.343
1766840459980 | 1 | TRENDVIC | DOWN | UP | 0.817 | 2.755
### 1:OSCILLATOR
- accuracy: 0.263 nonflat: 0.587 nonflat_swapped: 0.413 inversion_delta: -0.174 flat_rate: 0.610 avg_conf: 0.882
- calibration: ece=0.657 mce=0.688 brier=0.607
- confident_wrong_rate: 0.568
- adaptation: {'avg_lr_eff': 0.0009014584435952928, 'avg_anchor_lambda': 0.00010210374371359634, 'anchor_update_rate': 0.5169491525423728, 'avg_params_norm': 4.170020375845108, 'avg_anchor_norm': 4.242487790372207, 'avg_params_anchor_gap': 0.07246741452709775}

Confusion:
pred\fact | UP | DOWN | FLAT
---|---:|---:|---:
UP | 14 | 8 | 41
DOWN | 11 | 13 | 27
FLAT | 0 | 0 | 4

Calibration bins:
bin | count | acc | avg_conf | avg_abs_ret_bps
---|---:|---:|---:|---:
0 | 0 | 0.000 | 0.000 | 0.000
1 | 0 | 0.000 | 0.000 | 0.000
2 | 0 | 0.000 | 0.000 | 0.000
3 | 0 | 0.000 | 0.000 | 0.000
4 | 0 | 0.000 | 0.000 | 0.000
5 | 5 | 0.000 | 0.570 | 0.034
6 | 11 | 0.000 | 0.657 | 0.877
7 | 6 | 0.333 | 0.744 | 1.473
8 | 19 | 0.211 | 0.851 | 0.805
9 | 73 | 0.288 | 0.976 | 0.981

Top confident-wrong:
ts | tf | model | pred | fact | conf | ret_bps
---|---|---|---|---|---:|---:
1766840460597 | 1 | OSCILLATOR | DOWN | FLAT | 1.000 | 0.000
1766840462151 | 1 | OSCILLATOR | UP | FLAT | 0.999 | 0.343
1766840460526 | 1 | OSCILLATOR | DOWN | FLAT | 0.999 | 0.240
1766840462217 | 1 | OSCILLATOR | UP | FLAT | 0.999 | -0.011
1766840464287 | 1 | OSCILLATOR | DOWN | FLAT | 0.999 | -0.708
1766840462065 | 1 | OSCILLATOR | UP | FLAT | 0.999 | 0.011
1766840461044 | 1 | OSCILLATOR | DOWN | FLAT | 0.999 | 0.011
1766840463171 | 1 | OSCILLATOR | UP | FLAT | 0.998 | 0.789
1766840461116 | 1 | OSCILLATOR | DOWN | FLAT | 0.998 | 0.000
1766840463267 | 1 | OSCILLATOR | UP | FLAT | 0.998 | 0.754
### 1:VOLUMEMETRIX
- accuracy: 0.263 nonflat: 0.674 nonflat_swapped: 0.261 inversion_delta: -0.413 flat_rate: 0.610 avg_conf: 0.866
- calibration: ece=0.605 mce=0.668 brier=0.561
- confident_wrong_rate: 0.483
- adaptation: {'avg_lr_eff': 0.000911694101893635, 'avg_anchor_lambda': 0.00010168366909016944, 'anchor_update_rate': 0.4745762711864407, 'avg_params_norm': 4.207276745503325, 'avg_anchor_norm': 4.242572659210251, 'avg_params_anchor_gap': 0.035302138396340936}

Confusion:
pred\fact | UP | DOWN | FLAT
---|---:|---:|---:
UP | 18 | 7 | 40
DOWN | 5 | 13 | 32
FLAT | 2 | 1 | 0

Calibration bins:
bin | count | acc | avg_conf | avg_abs_ret_bps
---|---:|---:|---:|---:
0 | 0 | 0.000 | 0.000 | 0.000
1 | 0 | 0.000 | 0.000 | 0.000
2 | 0 | 0.000 | 0.000 | 0.000
3 | 0 | 0.000 | 0.000 | 0.000
4 | 0 | 0.000 | 0.000 | 0.000
5 | 6 | 0.167 | 0.578 | 0.314
6 | 14 | 0.143 | 0.643 | 0.750
7 | 13 | 0.231 | 0.750 | 0.401
8 | 13 | 0.308 | 0.866 | 1.259
9 | 69 | 0.304 | 0.972 | 0.906

Top confident-wrong:
ts | tf | model | pred | fact | conf | ret_bps
---|---|---|---|---|---:|---:
1766840464304 | 1 | VOLUMEMETRIX | DOWN | FLAT | 1.000 | -0.708
1766840462679 | 1 | VOLUMEMETRIX | DOWN | FLAT | 1.000 | -0.994
1766840469290 | 1 | VOLUMEMETRIX | DOWN | FLAT | 1.000 | 0.011
1766840464148 | 1 | VOLUMEMETRIX | DOWN | UP | 1.000 | 1.006
1766840464910 | 1 | VOLUMEMETRIX | UP | DOWN | 1.000 | -4.411
1766840464579 | 1 | VOLUMEMETRIX | UP | FLAT | 0.999 | 0.206
1766840463514 | 1 | VOLUMEMETRIX | DOWN | FLAT | 0.999 | 0.011
1766840460170 | 1 | VOLUMEMETRIX | UP | DOWN | 0.998 | -1.851
1766840460616 | 1 | VOLUMEMETRIX | DOWN | FLAT | 0.997 | 0.000
1766840462169 | 1 | VOLUMEMETRIX | UP | FLAT | 0.996 | 0.343

## Patterns
- 1:TRENDVIC: total=29 context=8 decision=21 coverage={'total': 29, 'context_fine': 0, 'context_coarse': 8, 'decision_fine': 0, 'decision_coarse': 21, 'unknown': 0}
- 1:OSCILLATOR: total=16 context=7 decision=9 coverage={'total': 16, 'context_fine': 0, 'context_coarse': 7, 'decision_fine': 0, 'decision_coarse': 9, 'unknown': 0}
- 1:VOLUMEMETRIX: total=15 context=5 decision=10 coverage={'total': 15, 'context_fine': 0, 'context_coarse': 5, 'decision_fine': 0, 'decision_coarse': 10, 'unknown': 0}
