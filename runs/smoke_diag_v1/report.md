# Offline Replay Report
- symbol: BTCUSDT
- tfs: 1
- mode: smoke
- range: 2025-12-27T16:48:31.635000 -> 2025-12-27T18:48:31.635000
- fact_flat_bps: 1.0
- candles: 119
- predictions: 357 facts: 354 updates: 354 pending_tail: 3
- checks: PASS (0 errors)

## Warnings
- overconfidence_guard overall avg_conf=0.812 acc=0.167 gap=0.645 ece=0.718
- overconfidence_guard 1:TRENDVIC avg_conf=0.679 acc=0.212 gap=0.467 ece=0.587
- overconfidence_guard 1:OSCILLATOR avg_conf=0.887 acc=0.153 gap=0.734 ece=0.770
- overconfidence_guard 1:VOLUMEMETRIX avg_conf=0.870 acc=0.136 gap=0.734 ece=0.781

## Overall
- accuracy: 0.167 nonflat: 0.382 nonflat_swapped: 0.529 inversion_delta: 0.147 flat_rate: 0.712 avg_conf: 0.812
- coverage: 0.918 action_accuracy: 0.120 pred_flat_rate: 0.082 fact_flat_rate: 0.712 flat_when_fact_nonflat_rate: 0.025
- calibration: ece=0.718 mce=0.826 brier=0.636
- confident_wrong_rate: 0.463
- adaptation: {'avg_lr_eff': 0.0009255013677015621, 'avg_anchor_lambda': 0.00010010592428623783, 'anchor_update_rate': 0.1807909604519774, 'avg_params_norm': 4.196879534496499, 'avg_anchor_norm': 4.242608840880106, 'avg_params_anchor_gap': 0.04656674453333934}

### Confusion (Overall)
pred\fact | UP | DOWN | FLAT
---|---:|---:|---:
UP | 15 | 14 | 74
DOWN | 40 | 24 | 158
FLAT | 5 | 4 | 20

### Calibration Bins (Overall)
bin | count | acc | avg_conf | avg_abs_ret_bps
---|---:|---:|---:|---:
0 | 0 | 0.000 | 0.000 | 0.000
1 | 0 | 0.000 | 0.000 | 0.000
2 | 0 | 0.000 | 0.000 | 0.000
3 | 0 | 0.000 | 0.000 | 0.000
4 | 0 | 0.000 | 0.000 | 0.000
5 | 21 | 0.143 | 0.576 | 0.689
6 | 58 | 0.052 | 0.653 | 0.803
7 | 56 | 0.125 | 0.742 | 0.651
8 | 33 | 0.091 | 0.848 | 0.695
9 | 157 | 0.146 | 0.973 | 0.803

### Calibration Evolution (Overall)
- a: initial=1.000 final=0.300 min=0.300 max=1.001
- b: initial=0.000 final=0.062 min=-0.022 max=0.062
- collapse_step: n/a

### FLAT Diagnostics (Overall)
reason | count | share
---|---:|---:
GENUINE_UNCERTAINTY | 25 | 0.862
NEAR_THRESHOLD | 4 | 0.138

### Top Confident-Wrong (Overall)
ts | tf | model | pred | fact | conf | ret_bps | close_prev | close_curr | delta | calib_a | margin_raw | margin_cal
---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:
1766850521141 | 1 | VOLUMEMETRIX | DOWN | FLAT | 1.000 | 0.080 | 87478.800 | 87479.500 | 0.700 | 0.501 | 1.000 | 1.000
1766850519434 | 1 | VOLUMEMETRIX | UP | FLAT | 1.000 | 0.000 | 87418.800 | 87418.800 | 0.000 | 0.823 | 1.000 | 1.000
1766850519373 | 1 | VOLUMEMETRIX | UP | DOWN | 1.000 | -2.905 | 87444.200 | 87418.800 | -25.400 | 0.910 | 1.000 | 1.000
1766850519247 | 1 | VOLUMEMETRIX | UP | FLAT | 1.000 | 0.000 | 87431.700 | 87431.700 | 0.000 | 0.920 | 1.000 | 1.000
1766850521082 | 1 | VOLUMEMETRIX | DOWN | UP | 1.000 | 2.001 | 87461.300 | 87478.800 | 17.500 | 0.606 | 1.000 | 1.000
1766850520968 | 1 | VOLUMEMETRIX | DOWN | FLAT | 1.000 | 0.286 | 87443.500 | 87446.000 | 2.500 | 0.689 | 1.000 | 1.000
1766850521200 | 1 | VOLUMEMETRIX | DOWN | FLAT | 1.000 | -0.903 | 87479.500 | 87471.600 | -7.900 | 0.476 | 1.000 | 1.000
1766850519187 | 1 | VOLUMEMETRIX | UP | DOWN | 1.000 | -2.218 | 87451.100 | 87431.700 | -19.400 | 0.976 | 1.000 | 1.000
1766850521262 | 1 | VOLUMEMETRIX | DOWN | FLAT | 1.000 | 0.000 | 87471.600 | 87471.600 | 0.000 | 0.461 | 1.000 | 1.000
1766850519493 | 1 | VOLUMEMETRIX | UP | FLAT | 1.000 | 0.663 | 87418.800 | 87424.600 | 5.800 | 0.810 | 1.000 | 1.000

## Per Model
### 1:TRENDVIC
- accuracy: 0.212 nonflat: 0.353 nonflat_swapped: 0.500 inversion_delta: 0.147 flat_rate: 0.712 avg_conf: 0.679
- coverage: 0.847 action_accuracy: 0.120 pred_flat_rate: 0.153 fact_flat_rate: 0.712 flat_when_fact_nonflat_rate: 0.042
- calibration: ece=0.587 mce=0.845 brier=0.452
- confident_wrong_rate: 0.085
- adaptation: {'avg_lr_eff': 0.0009349530731190223, 'avg_anchor_lambda': 0.00010028915323259098, 'anchor_update_rate': 0.2542372881355932, 'avg_params_norm': 4.227493804876077, 'avg_anchor_norm': 4.242630251425478, 'avg_params_anchor_gap': 0.017648760998575777}

Confusion:
pred\fact | UP | DOWN | FLAT
---|---:|---:|---:
UP | 5 | 6 | 21
DOWN | 11 | 7 | 50
FLAT | 4 | 1 | 13

Calibration bins:
bin | count | acc | avg_conf | avg_abs_ret_bps
---|---:|---:|---:|---:
0 | 0 | 0.000 | 0.000 | 0.000
1 | 0 | 0.000 | 0.000 | 0.000
2 | 0 | 0.000 | 0.000 | 0.000
3 | 0 | 0.000 | 0.000 | 0.000
4 | 0 | 0.000 | 0.000 | 0.000
5 | 16 | 0.125 | 0.578 | 0.757
6 | 32 | 0.062 | 0.651 | 0.917
7 | 39 | 0.128 | 0.739 | 0.566
8 | 5 | 0.000 | 0.845 | 0.350
9 | 8 | 0.375 | 0.950 | 1.434

Calibration evolution:
- a: initial=1.000 final=0.920 min=0.920 max=1.001
- b: initial=0.000 final=0.019 min=-0.005 max=0.019
- collapse_step: n/a

FLAT diagnostics:
reason | count | share
---|---:|---:
GENUINE_UNCERTAINTY | 18 | 1.000

Top confident-wrong:
ts | tf | model | pred | fact | conf | ret_bps | close_prev | close_curr | delta | calib_a | margin_raw | margin_cal
---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:
1766850521107 | 1 | TRENDVIC | UP | FLAT | 0.997 | 0.080 | 87478.800 | 87479.500 | 0.700 | 0.981 | 0.994 | 0.994
1766850522801 | 1 | TRENDVIC | UP | DOWN | 0.994 | -4.227 | 87533.500 | 87496.500 | -37.000 | 0.964 | 0.991 | 0.989
1766850524814 | 1 | TRENDVIC | UP | FLAT | 0.964 | 0.000 | 87572.200 | 87572.200 | 0.000 | 0.928 | 0.943 | 0.929
1766850521719 | 1 | TRENDVIC | UP | FLAT | 0.932 | 0.834 | 87509.900 | 87517.200 | 7.300 | 0.975 | 0.872 | 0.864
1766850524927 | 1 | TRENDVIC | UP | FLAT | 0.915 | 0.011 | 87568.300 | 87568.400 | 0.100 | 0.926 | 0.855 | 0.829
1766850520937 | 1 | TRENDVIC | UP | FLAT | 0.879 | 0.286 | 87443.500 | 87446.000 | 2.500 | 0.983 | 0.767 | 0.759
1766850519214 | 1 | TRENDVIC | UP | FLAT | 0.873 | 0.000 | 87431.700 | 87431.700 | 0.000 | 1.001 | 0.747 | 0.747
1766850520377 | 1 | TRENDVIC | UP | DOWN | 0.840 | -1.464 | 87435.500 | 87422.700 | -12.800 | 0.992 | 0.685 | 0.681
1766850520645 | 1 | TRENDVIC | UP | FLAT | 0.824 | 0.000 | 87424.300 | 87424.300 | 0.000 | 0.984 | 0.657 | 0.648
1766850520761 | 1 | TRENDVIC | DOWN | FLAT | 0.810 | 0.000 | 87424.400 | 87424.400 | 0.000 | 0.983 | 0.626 | 0.620
### 1:OSCILLATOR
- accuracy: 0.153 nonflat: 0.441 nonflat_swapped: 0.500 inversion_delta: 0.059 flat_rate: 0.712 avg_conf: 0.887
- coverage: 0.958 action_accuracy: 0.133 pred_flat_rate: 0.042 fact_flat_rate: 0.712 flat_when_fact_nonflat_rate: 0.017
- calibration: ece=0.770 mce=0.850 brier=0.718
- confident_wrong_rate: 0.644
- adaptation: {'avg_lr_eff': 0.0009241627738569647, 'avg_anchor_lambda': 0.00010003905323291196, 'anchor_update_rate': 0.15254237288135594, 'avg_params_norm': 4.1748693225047635, 'avg_anchor_norm': 4.242593340962638, 'avg_params_anchor_gap': 0.06772401845788185}

Confusion:
pred\fact | UP | DOWN | FLAT
---|---:|---:|---:
UP | 6 | 4 | 26
DOWN | 13 | 9 | 55
FLAT | 1 | 1 | 3

Calibration bins:
bin | count | acc | avg_conf | avg_abs_ret_bps
---|---:|---:|---:|---:
0 | 0 | 0.000 | 0.000 | 0.000
1 | 0 | 0.000 | 0.000 | 0.000
2 | 0 | 0.000 | 0.000 | 0.000
3 | 0 | 0.000 | 0.000 | 0.000
4 | 0 | 0.000 | 0.000 | 0.000
5 | 4 | 0.250 | 0.573 | 0.411
6 | 12 | 0.083 | 0.663 | 0.661
7 | 8 | 0.000 | 0.750 | 0.870
8 | 7 | 0.000 | 0.850 | 1.057
9 | 82 | 0.159 | 0.974 | 0.724

Calibration evolution:
- a: initial=1.000 final=0.425 min=0.425 max=1.000
- b: initial=0.000 final=0.061 min=-0.016 max=0.061
- collapse_step: n/a

FLAT diagnostics:
reason | count | share
---|---:|---:
GENUINE_UNCERTAINTY | 5 | 1.000

Top confident-wrong:
ts | tf | model | pred | fact | conf | ret_bps | close_prev | close_curr | delta | calib_a | margin_raw | margin_cal
---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:
1766850519169 | 1 | OSCILLATOR | UP | DOWN | 1.000 | -2.218 | 87451.100 | 87431.700 | -19.400 | 0.980 | 1.000 | 1.000
1766850519107 | 1 | OSCILLATOR | UP | FLAT | 1.000 | 0.000 | 87451.100 | 87451.100 | 0.000 | 0.985 | 1.000 | 1.000
1766850519230 | 1 | OSCILLATOR | UP | FLAT | 1.000 | 0.000 | 87431.700 | 87431.700 | 0.000 | 0.932 | 1.000 | 0.999
1766850521852 | 1 | OSCILLATOR | DOWN | FLAT | 0.999 | 0.114 | 87530.900 | 87531.900 | 1.000 | 0.688 | 1.000 | 0.999
1766850521906 | 1 | OSCILLATOR | DOWN | FLAT | 0.999 | 0.457 | 87531.900 | 87535.900 | 4.000 | 0.681 | 1.000 | 0.999
1766850519044 | 1 | OSCILLATOR | UP | FLAT | 0.999 | -0.023 | 87451.300 | 87451.100 | -0.200 | 0.989 | 0.999 | 0.999
1766850521182 | 1 | OSCILLATOR | DOWN | FLAT | 0.999 | -0.903 | 87479.500 | 87471.600 | -7.900 | 0.799 | 1.000 | 0.998
1766850521964 | 1 | OSCILLATOR | DOWN | FLAT | 0.999 | -0.594 | 87535.900 | 87530.700 | -5.200 | 0.674 | 1.000 | 0.998
1766850521795 | 1 | OSCILLATOR | DOWN | UP | 0.999 | 1.565 | 87517.200 | 87530.900 | 13.700 | 0.734 | 1.000 | 0.998
1766850521124 | 1 | OSCILLATOR | DOWN | FLAT | 0.999 | 0.080 | 87478.800 | 87479.500 | 0.700 | 0.804 | 0.999 | 0.997
### 1:VOLUMEMETRIX
- accuracy: 0.136 nonflat: 0.353 nonflat_swapped: 0.588 inversion_delta: 0.235 flat_rate: 0.712 avg_conf: 0.870
- coverage: 0.949 action_accuracy: 0.107 pred_flat_rate: 0.051 fact_flat_rate: 0.712 flat_when_fact_nonflat_rate: 0.017
- calibration: ece=0.781 mce=0.870 brier=0.717
- confident_wrong_rate: 0.661
- adaptation: {'avg_lr_eff': 0.0009173882561286957, 'avg_anchor_lambda': 9.99895663932105e-05, 'anchor_update_rate': 0.13559322033898305, 'avg_params_norm': 4.188275476108651, 'avg_anchor_norm': 4.242602930252221, 'avg_params_anchor_gap': 0.054327454143560515}

Confusion:
pred\fact | UP | DOWN | FLAT
---|---:|---:|---:
UP | 4 | 4 | 27
DOWN | 16 | 8 | 53
FLAT | 0 | 2 | 4

Calibration bins:
bin | count | acc | avg_conf | avg_abs_ret_bps
---|---:|---:|---:|---:
0 | 0 | 0.000 | 0.000 | 0.000
1 | 0 | 0.000 | 0.000 | 0.000
2 | 0 | 0.000 | 0.000 | 0.000
3 | 0 | 0.000 | 0.000 | 0.000
4 | 0 | 0.000 | 0.000 | 0.000
5 | 1 | 0.000 | 0.558 | 0.709
6 | 14 | 0.000 | 0.648 | 0.664
7 | 9 | 0.222 | 0.747 | 0.825
8 | 21 | 0.143 | 0.847 | 0.656
9 | 67 | 0.104 | 0.974 | 0.824

Calibration evolution:
- a: initial=1.000 final=0.300 min=0.300 max=1.000
- b: initial=0.000 final=0.062 min=-0.022 max=0.062
- collapse_step: n/a

FLAT diagnostics:
reason | count | share
---|---:|---:
NEAR_THRESHOLD | 4 | 0.667
GENUINE_UNCERTAINTY | 2 | 0.333

Top confident-wrong:
ts | tf | model | pred | fact | conf | ret_bps | close_prev | close_curr | delta | calib_a | margin_raw | margin_cal
---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:
1766850521141 | 1 | VOLUMEMETRIX | DOWN | FLAT | 1.000 | 0.080 | 87478.800 | 87479.500 | 0.700 | 0.501 | 1.000 | 1.000
1766850519434 | 1 | VOLUMEMETRIX | UP | FLAT | 1.000 | 0.000 | 87418.800 | 87418.800 | 0.000 | 0.823 | 1.000 | 1.000
1766850519373 | 1 | VOLUMEMETRIX | UP | DOWN | 1.000 | -2.905 | 87444.200 | 87418.800 | -25.400 | 0.910 | 1.000 | 1.000
1766850519247 | 1 | VOLUMEMETRIX | UP | FLAT | 1.000 | 0.000 | 87431.700 | 87431.700 | 0.000 | 0.920 | 1.000 | 1.000
1766850521082 | 1 | VOLUMEMETRIX | DOWN | UP | 1.000 | 2.001 | 87461.300 | 87478.800 | 17.500 | 0.606 | 1.000 | 1.000
1766850520968 | 1 | VOLUMEMETRIX | DOWN | FLAT | 1.000 | 0.286 | 87443.500 | 87446.000 | 2.500 | 0.689 | 1.000 | 1.000
1766850521200 | 1 | VOLUMEMETRIX | DOWN | FLAT | 1.000 | -0.903 | 87479.500 | 87471.600 | -7.900 | 0.476 | 1.000 | 1.000
1766850519187 | 1 | VOLUMEMETRIX | UP | DOWN | 1.000 | -2.218 | 87451.100 | 87431.700 | -19.400 | 0.976 | 1.000 | 1.000
1766850521262 | 1 | VOLUMEMETRIX | DOWN | FLAT | 1.000 | 0.000 | 87471.600 | 87471.600 | 0.000 | 0.461 | 1.000 | 1.000
1766850519493 | 1 | VOLUMEMETRIX | UP | FLAT | 1.000 | 0.663 | 87418.800 | 87424.600 | 5.800 | 0.810 | 1.000 | 1.000

## Patterns
- 1:TRENDVIC: total=26 context=7 decision=19 coverage={'total': 26, 'context_fine': 0, 'context_coarse': 7, 'decision_fine': 0, 'decision_coarse': 19, 'unknown': 0}
- 1:OSCILLATOR: total=17 context=7 decision=10 coverage={'total': 17, 'context_fine': 0, 'context_coarse': 7, 'decision_fine': 0, 'decision_coarse': 10, 'unknown': 0}
- 1:VOLUMEMETRIX: total=17 context=7 decision=10 coverage={'total': 17, 'context_fine': 0, 'context_coarse': 7, 'decision_fine': 0, 'decision_coarse': 10, 'unknown': 0}
