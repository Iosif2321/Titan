# Offline Replay Report
- symbol: BTCUSDT
- tfs: 1
- mode: train
- range: 2025-12-27T19:57:39.057000 -> 2025-12-27T23:57:39.057000
- fact_flat_bps: 0.2
- fact_flat_mode: adaptive target=0.100-0.250 bps_range=0.010-0.250 current=1=0.0100
- pred_flat_mode: adaptive target=0.050-0.150 delta_range=0.010-0.080 current=1=0.0120
- reward_mode: shaped
- candles: 239
- predictions: 717 facts: 714 updates: 714 pending_tail: 3
- checks: PASS (0 errors)

## Warnings
- overconfidence_guard overall avg_conf=0.734 acc=0.359 gap=0.375 ece=0.389
- calibration_low_a overall calib_a=0.200
- overconfidence_guard 1:TRENDVIC avg_conf=0.671 acc=0.345 gap=0.327 ece=0.335
- overconfidence_guard 1:OSCILLATOR avg_conf=0.734 acc=0.340 gap=0.394 ece=0.406
- calibration_low_a 1:OSCILLATOR calib_a=0.150
- overconfidence_guard 1:VOLUMEMETRIX avg_conf=0.796 acc=0.391 gap=0.405 ece=0.423
- calibration_low_a 1:VOLUMEMETRIX calib_a=0.200

## Overall
- accuracy: 0.359 nonflat: 0.497 nonflat_swapped: 0.400 inversion_delta: -0.096 flat_rate: 0.332 avg_conf: 0.734
- coverage: 0.905 action_accuracy: 0.367 pred_flat_rate: 0.095 fact_flat_rate: 0.332 flat_when_fact_nonflat_rate: 0.069
- calibration: ece=0.389 mce=0.584 brier=0.394
- confident_wrong_rate: 0.174
- adaptation: {'avg_lr_eff': 0.0008971868847169355, 'avg_anchor_lambda': 0.00010246490102995281, 'anchor_update_rate': 0.6148459383753502, 'avg_params_norm': 4.155259802452026, 'avg_anchor_norm': 4.242173261928162, 'avg_params_anchor_gap': 0.08692879313726666}

### Confusion (Overall)
pred\fact | UP | DOWN | FLAT
---|---:|---:|---:
UP | 121 | 78 | 94
DOWN | 113 | 116 | 124
FLAT | 24 | 25 | 19

### Calibration Bins (Overall)
bin | count | acc | avg_conf | avg_abs_ret_bps
---|---:|---:|---:|---:
0 | 0 | 0.000 | 0.000 | 0.000
1 | 0 | 0.000 | 0.000 | 0.000
2 | 0 | 0.000 | 0.000 | 0.000
3 | 0 | 0.000 | 0.000 | 0.000
4 | 0 | 0.000 | 0.000 | 0.000
5 | 55 | 0.327 | 0.574 | 0.680
6 | 208 | 0.351 | 0.655 | 0.532
7 | 171 | 0.339 | 0.749 | 0.501
8 | 96 | 0.458 | 0.839 | 0.373
9 | 116 | 0.379 | 0.964 | 0.664

### Calibration Evolution (Overall)
- a: initial=1.000 final=0.200 min=0.150 max=1.000
- b: initial=0.000 final=-0.026 min=-0.089 max=0.218
- collapse_step: n/a

### FLAT Diagnostics (Overall)
reason | count | share
---|---:|---:
GENUINE_UNCERTAINTY | 38 | 0.559
NEAR_THRESHOLD | 23 | 0.338
COMPRESSION_BY_LOW_A | 6 | 0.088
OTHER | 1 | 0.015

### Top Confident-Wrong (Overall)
ts | tf | model | pred | fact | conf | ret_bps | close_prev | close_curr | delta | calib_a | margin_raw | margin_cal
---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:
1766869073108 | 1 | VOLUMEMETRIX | UP | DOWN | 1.000 | -0.537 | 87484.700 | 87480.000 | -4.700 | 0.834 | 1.000 | 1.000
1766869072629 | 1 | VOLUMEMETRIX | UP | DOWN | 1.000 | -3.954 | 87502.500 | 87467.900 | -34.600 | 0.956 | 1.000 | 1.000
1766869070578 | 1 | VOLUMEMETRIX | UP | FLAT | 1.000 | 0.000 | 87505.900 | 87505.900 | 0.000 | 1.000 | 1.000 | 1.000
1766869075812 | 1 | VOLUMEMETRIX | UP | DOWN | 1.000 | -0.560 | 87516.900 | 87512.000 | -4.900 | 0.692 | 1.000 | 1.000
1766869075570 | 1 | VOLUMEMETRIX | UP | FLAT | 1.000 | 0.000 | 87516.900 | 87516.900 | 0.000 | 0.705 | 1.000 | 1.000
1766869070799 | 1 | VOLUMEMETRIX | UP | FLAT | 1.000 | 0.000 | 87505.900 | 87505.900 | 0.000 | 0.989 | 1.000 | 1.000
1766869071464 | 1 | VOLUMEMETRIX | UP | FLAT | 1.000 | 0.000 | 87505.900 | 87505.900 | 0.000 | 0.971 | 0.999 | 0.999
1766869071249 | 1 | VOLUMEMETRIX | UP | DOWN | 0.999 | -0.011 | 87506.000 | 87505.900 | -0.100 | 0.978 | 0.999 | 0.999
1766869077244 | 1 | VOLUMEMETRIX | UP | FLAT | 0.999 | 0.000 | 87522.000 | 87522.000 | 0.000 | 0.630 | 1.000 | 0.998
1766869077993 | 1 | VOLUMEMETRIX | UP | FLAT | 0.998 | 0.000 | 87524.700 | 87524.700 | 0.000 | 0.603 | 1.000 | 0.997

## Per Model
### 1:TRENDVIC
- accuracy: 0.345 nonflat: 0.465 nonflat_swapped: 0.384 inversion_delta: -0.082 flat_rate: 0.332 avg_conf: 0.671
- coverage: 0.866 action_accuracy: 0.359 pred_flat_rate: 0.134 fact_flat_rate: 0.332 flat_when_fact_nonflat_rate: 0.101
- calibration: ece=0.335 mce=0.503 brier=0.342
- confident_wrong_rate: 0.046
- adaptation: {'avg_lr_eff': 0.0009051691237816016, 'avg_anchor_lambda': 0.0001017998407037057, 'anchor_update_rate': 0.5294117647058824, 'avg_params_norm': 4.1973311687247294, 'avg_anchor_norm': 4.24247180558317, 'avg_params_anchor_gap': 0.045140636858442065}

Confusion:
pred\fact | UP | DOWN | FLAT
---|---:|---:|---:
UP | 30 | 18 | 18
DOWN | 43 | 44 | 53
FLAT | 13 | 11 | 8

Calibration bins:
bin | count | acc | avg_conf | avg_abs_ret_bps
---|---:|---:|---:|---:
0 | 0 | 0.000 | 0.000 | 0.000
1 | 0 | 0.000 | 0.000 | 0.000
2 | 0 | 0.000 | 0.000 | 0.000
3 | 0 | 0.000 | 0.000 | 0.000
4 | 0 | 0.000 | 0.000 | 0.000
5 | 25 | 0.320 | 0.576 | 0.692
6 | 110 | 0.345 | 0.658 | 0.480
7 | 49 | 0.347 | 0.735 | 0.366
8 | 9 | 0.556 | 0.843 | 0.269
9 | 13 | 0.462 | 0.964 | 1.084

Calibration evolution:
- a: initial=1.000 final=0.716 min=0.716 max=1.000
- b: initial=0.000 final=0.017 min=-0.006 max=0.054
- collapse_step: n/a

FLAT diagnostics:
reason | count | share
---|---:|---:
GENUINE_UNCERTAINTY | 28 | 0.875
NEAR_THRESHOLD | 4 | 0.125

Top confident-wrong:
ts | tf | model | pred | fact | conf | ret_bps | close_prev | close_curr | delta | calib_a | margin_raw | margin_cal
---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:
1766869084887 | 1 | TRENDVIC | UP | DOWN | 0.997 | -1.336 | 87558.300 | 87546.600 | -11.700 | 0.952 | 0.996 | 0.994
1766869072493 | 1 | TRENDVIC | UP | DOWN | 0.996 | -3.954 | 87502.500 | 87467.900 | -34.600 | 0.993 | 0.993 | 0.992
1766869090420 | 1 | TRENDVIC | UP | DOWN | 0.990 | -1.085 | 87545.900 | 87536.400 | -9.500 | 0.919 | 0.987 | 0.981
1766869118029 | 1 | TRENDVIC | UP | DOWN | 0.989 | -0.628 | 87596.100 | 87590.600 | -5.500 | 0.786 | 0.993 | 0.978
1766869100814 | 1 | TRENDVIC | UP | DOWN | 0.984 | -0.468 | 87564.100 | 87560.000 | -4.100 | 0.854 | 0.984 | 0.968
1766869095809 | 1 | TRENDVIC | UP | DOWN | 0.924 | -0.011 | 87520.000 | 87519.900 | -0.100 | 0.883 | 0.888 | 0.848
1766869112425 | 1 | TRENDVIC | UP | FLAT | 0.901 | 0.000 | 87573.700 | 87573.700 | 0.000 | 0.816 | 0.868 | 0.801
1766869083490 | 1 | TRENDVIC | UP | DOWN | 0.874 | -0.251 | 87557.200 | 87555.000 | -2.200 | 0.955 | 0.765 | 0.749
1766869093653 | 1 | TRENDVIC | UP | FLAT | 0.860 | 0.000 | 87527.000 | 87527.000 | 0.000 | 0.886 | 0.771 | 0.720
1766869070667 | 1 | TRENDVIC | DOWN | FLAT | 0.815 | 0.000 | 87505.900 | 87505.900 | 0.000 | 0.995 | 0.633 | 0.630
### 1:OSCILLATOR
- accuracy: 0.340 nonflat: 0.478 nonflat_swapped: 0.440 inversion_delta: -0.038 flat_rate: 0.332 avg_conf: 0.734
- coverage: 0.924 action_accuracy: 0.345 pred_flat_rate: 0.076 fact_flat_rate: 0.332 flat_when_fact_nonflat_rate: 0.055
- calibration: ece=0.406 mce=0.681 brier=0.400
- confident_wrong_rate: 0.181
- adaptation: {'avg_lr_eff': 0.0008682455867943665, 'avg_anchor_lambda': 0.00010346950749836878, 'anchor_update_rate': 0.7016806722689075, 'avg_params_norm': 4.091938136584278, 'avg_anchor_norm': 4.241719253266757, 'avg_params_anchor_gap': 0.14980934836432222}

Confusion:
pred\fact | UP | DOWN | FLAT
---|---:|---:|---:
UP | 39 | 30 | 30
DOWN | 40 | 37 | 44
FLAT | 7 | 6 | 5

Calibration bins:
bin | count | acc | avg_conf | avg_abs_ret_bps
---|---:|---:|---:|---:
0 | 0 | 0.000 | 0.000 | 0.000
1 | 0 | 0.000 | 0.000 | 0.000
2 | 0 | 0.000 | 0.000 | 0.000
3 | 0 | 0.000 | 0.000 | 0.000
4 | 0 | 0.000 | 0.000 | 0.000
5 | 18 | 0.333 | 0.577 | 0.764
6 | 60 | 0.333 | 0.658 | 0.506
7 | 70 | 0.300 | 0.754 | 0.578
8 | 49 | 0.469 | 0.838 | 0.450
9 | 23 | 0.261 | 0.942 | 0.608

Calibration evolution:
- a: initial=0.500 final=0.150 min=0.150 max=0.501
- b: initial=0.000 final=-0.020 min=-0.054 max=0.218
- collapse_step: n/a

FLAT diagnostics:
reason | count | share
---|---:|---:
NEAR_THRESHOLD | 8 | 0.444
COMPRESSION_BY_LOW_A | 6 | 0.333
GENUINE_UNCERTAINTY | 4 | 0.222

Top confident-wrong:
ts | tf | model | pred | fact | conf | ret_bps | close_prev | close_curr | delta | calib_a | margin_raw | margin_cal
---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:
1766869074568 | 1 | OSCILLATOR | DOWN | FLAT | 0.985 | 0.000 | 87525.200 | 87525.200 | 0.000 | 0.460 | 1.000 | 0.971
1766869074326 | 1 | OSCILLATOR | DOWN | UP | 0.971 | 0.651 | 87519.500 | 87525.200 | 5.700 | 0.469 | 0.999 | 0.942
1766869076702 | 1 | OSCILLATOR | DOWN | UP | 0.966 | 0.011 | 87534.900 | 87535.000 | 0.100 | 0.370 | 1.000 | 0.931
1766869070509 | 1 | OSCILLATOR | UP | FLAT | 0.964 | 0.000 | 87505.900 | 87505.900 | 0.000 | 0.501 | 0.997 | 0.928
1766869076444 | 1 | OSCILLATOR | DOWN | FLAT | 0.961 | 0.000 | 87534.900 | 87534.900 | 0.000 | 0.380 | 1.000 | 0.921
1766869073034 | 1 | OSCILLATOR | UP | DOWN | 0.959 | -0.537 | 87484.700 | 87480.000 | -4.700 | 0.476 | 0.997 | 0.917
1766869076223 | 1 | OSCILLATOR | DOWN | UP | 0.947 | 0.400 | 87531.400 | 87534.900 | 3.500 | 0.388 | 0.999 | 0.893
1766869081636 | 1 | OSCILLATOR | DOWN | UP | 0.937 | 0.217 | 87543.200 | 87545.100 | 1.900 | 0.304 | 1.000 | 0.875
1766869070728 | 1 | OSCILLATOR | UP | FLAT | 0.936 | 0.000 | 87505.900 | 87505.900 | 0.000 | 0.494 | 0.991 | 0.872
1766869081411 | 1 | OSCILLATOR | DOWN | FLAT | 0.932 | 0.000 | 87543.200 | 87543.200 | 0.000 | 0.313 | 1.000 | 0.863
### 1:VOLUMEMETRIX
- accuracy: 0.391 nonflat: 0.547 nonflat_swapped: 0.377 inversion_delta: -0.170 flat_rate: 0.332 avg_conf: 0.796
- coverage: 0.924 action_accuracy: 0.395 pred_flat_rate: 0.076 fact_flat_rate: 0.332 flat_when_fact_nonflat_rate: 0.050
- calibration: ece=0.423 mce=0.570 brier=0.435
- confident_wrong_rate: 0.294
- adaptation: {'avg_lr_eff': 0.0009181459435748413, 'avg_anchor_lambda': 0.0001021253548877835, 'anchor_update_rate': 0.6134453781512605, 'avg_params_norm': 4.176510102047067, 'avg_anchor_norm': 4.242328726934563, 'avg_params_anchor_gap': 0.0658363941890357}

Confusion:
pred\fact | UP | DOWN | FLAT
---|---:|---:|---:
UP | 52 | 30 | 46
DOWN | 30 | 35 | 27
FLAT | 4 | 8 | 6

Calibration bins:
bin | count | acc | avg_conf | avg_abs_ret_bps
---|---:|---:|---:|---:
0 | 0 | 0.000 | 0.000 | 0.000
1 | 0 | 0.000 | 0.000 | 0.000
2 | 0 | 0.000 | 0.000 | 0.000
3 | 0 | 0.000 | 0.000 | 0.000
4 | 0 | 0.000 | 0.000 | 0.000
5 | 12 | 0.333 | 0.567 | 0.530
6 | 38 | 0.395 | 0.640 | 0.724
7 | 52 | 0.385 | 0.756 | 0.524
8 | 38 | 0.421 | 0.841 | 0.298
9 | 80 | 0.400 | 0.970 | 0.611

Calibration evolution:
- a: initial=1.000 final=0.200 min=0.200 max=1.000
- b: initial=0.000 final=-0.026 min=-0.089 max=0.047
- collapse_step: n/a

FLAT diagnostics:
reason | count | share
---|---:|---:
NEAR_THRESHOLD | 11 | 0.611
GENUINE_UNCERTAINTY | 6 | 0.333
OTHER | 1 | 0.056

Top confident-wrong:
ts | tf | model | pred | fact | conf | ret_bps | close_prev | close_curr | delta | calib_a | margin_raw | margin_cal
---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:
1766869073108 | 1 | VOLUMEMETRIX | UP | DOWN | 1.000 | -0.537 | 87484.700 | 87480.000 | -4.700 | 0.834 | 1.000 | 1.000
1766869072629 | 1 | VOLUMEMETRIX | UP | DOWN | 1.000 | -3.954 | 87502.500 | 87467.900 | -34.600 | 0.956 | 1.000 | 1.000
1766869070578 | 1 | VOLUMEMETRIX | UP | FLAT | 1.000 | 0.000 | 87505.900 | 87505.900 | 0.000 | 1.000 | 1.000 | 1.000
1766869075812 | 1 | VOLUMEMETRIX | UP | DOWN | 1.000 | -0.560 | 87516.900 | 87512.000 | -4.900 | 0.692 | 1.000 | 1.000
1766869075570 | 1 | VOLUMEMETRIX | UP | FLAT | 1.000 | 0.000 | 87516.900 | 87516.900 | 0.000 | 0.705 | 1.000 | 1.000
1766869070799 | 1 | VOLUMEMETRIX | UP | FLAT | 1.000 | 0.000 | 87505.900 | 87505.900 | 0.000 | 0.989 | 1.000 | 1.000
1766869071464 | 1 | VOLUMEMETRIX | UP | FLAT | 1.000 | 0.000 | 87505.900 | 87505.900 | 0.000 | 0.971 | 0.999 | 0.999
1766869071249 | 1 | VOLUMEMETRIX | UP | DOWN | 0.999 | -0.011 | 87506.000 | 87505.900 | -0.100 | 0.978 | 0.999 | 0.999
1766869077244 | 1 | VOLUMEMETRIX | UP | FLAT | 0.999 | 0.000 | 87522.000 | 87522.000 | 0.000 | 0.630 | 1.000 | 0.998
1766869077993 | 1 | VOLUMEMETRIX | UP | FLAT | 0.998 | 0.000 | 87524.700 | 87524.700 | 0.000 | 0.603 | 1.000 | 0.997

## Patterns
- 1:TRENDVIC: total=30 context=8 decision=22 coverage={'total': 30, 'context_fine': 0, 'context_coarse': 8, 'decision_fine': 0, 'decision_coarse': 22, 'unknown': 0}
- 1:OSCILLATOR: total=18 context=7 decision=11 coverage={'total': 18, 'context_fine': 0, 'context_coarse': 7, 'decision_fine': 0, 'decision_coarse': 11, 'unknown': 0}
- 1:VOLUMEMETRIX: total=24 context=8 decision=16 coverage={'total': 24, 'context_fine': 0, 'context_coarse': 8, 'decision_fine': 0, 'decision_coarse': 16, 'unknown': 0}
