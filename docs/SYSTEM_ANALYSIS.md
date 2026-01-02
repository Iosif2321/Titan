# Titan System Analysis

## 1. Feature Distribution by Model

### 1.1 Feature Summary

| Feature | TrendVIC | Oscillator | VolumeMetrix | ML Classifier | Total |
|---------|:--------:|:----------:|:------------:|:-------------:|:-----:|
| **Returns** |
| return_1 | | | + | + | 2 |
| log_return_1 | | | | + | 1 |
| price_momentum_3 | + | | | + | 2 |
| return_5 | | | | + | 1 |
| return_10 | | | | + | 1 |
| return_lag_1..5 | | | | + | 1 |
| **Volatility** |
| volatility | + | | + | | 2 |
| volatility_z | | + | | + | 2 |
| vol_ratio | | | | + | 1 |
| atr_pct | | | | + | 1 |
| high_low_range_pct | | | | + | 1 |
| **Volume** |
| volume_z | | | + | + | 2 |
| volume_trend | | | + | + | 2 |
| volume_change_pct | | | | + | 1 |
| **RSI** |
| rsi | | + | | + | 2 |
| rsi_momentum | | + | | + | 2 |
| rsi_oversold/overbought/neutral | | | | + | 1 |
| **Trend** |
| ma_delta | + | | + | | 2 |
| ma_delta_pct | | | | + | 1 |
| ema_10/20_spread_pct | | | | + | 1 |
| **Candle Structure** |
| body_ratio | + | | | + | 2 |
| body_pct | | | | + | 1 |
| candle_direction | + | | | + | 2 |
| upper_wick_ratio | | | + | + | 2 |
| lower_wick_ratio | | | + | + | 2 |
| close | + | | | | 1 |

### 1.2 Model Feature Counts

| Model | Primary Features | Secondary Features | Total |
|-------|-----------------|-------------------|-------|
| **TrendVIC** | ma_delta, volatility, close | body_ratio, candle_direction, price_momentum_3 | **6** |
| **Oscillator** | rsi, rsi_momentum | volatility_z | **3** |
| **VolumeMetrix** | volume_z, return_1, volatility | ma_delta, volume_trend, upper/lower_wick_ratio | **7** |
| **ML Classifier** | All 31 scale-invariant features | - | **31** |

### 1.3 Feature Usage Analysis

```
TrendVIC:
  - Primary signal: ma_delta (trend direction)
  - Scaling: volatility * close
  - Confirmation: body_ratio, candle_direction, price_momentum_3
  - Pattern context: volatility_z, body_ratio

Oscillator:
  - Primary signal: rsi (deviation from 50)
  - Momentum: rsi_momentum (confirmation/contradiction)
  - Penalty: volatility_z (high vol = weaker signal)
  - Pattern context: rsi, rsi_momentum, volatility_z

VolumeMetrix:
  - Primary signal: volume_z + return_1 (volume-price relationship)
  - Patterns: continuation, absorption, low_volume, normal
  - Confirmation: volume_trend, upper/lower_wick_ratio
  - Pattern context: volume_z, volatility_z, body_ratio

ML Classifier:
  - All 31 features (scale-invariant only)
  - Feature importance learned from data
  - No hardcoded logic
```

---

## 2. Calibration System Analysis

### 2.1 Current Architecture

```
Raw Model Output
      │
      ▼
┌─────────────────────────────────┐
│   ConfidenceCompressor          │
│   [0.5, 1.0] → [0.5, 0.70]     │
│   max_confidence = 0.70         │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│   Regime Confidence Multiplier  │
│   trending_up:    1.00          │
│   ranging:        0.95          │
│   trending_down:  0.85          │
│   volatile:       0.75          │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│   Pattern Context Adjustment    │
│   _apply_pattern_context_strength│
│   min scale = 0.7               │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│   OnlineCalibrator              │
│   Bin-based calibration         │
│   blend = 0.70                  │
└─────────────────────────────────┘
      │
      ▼
Final Confidence
```

### 2.2 Calibration Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| max_confidence | 0.70 | Caps all confidence at 70% |
| regime multipliers | 0.75-1.0 | Reduces confidence in bad regimes |
| pattern scale min | 0.70 | Prevents vicious cycle |
| calibration blend | 0.70 | How much to adjust based on history |
| calibration min_samples | 30 | Min samples before adjusting |

### 2.3 Current Results

| Confidence Bucket | Accuracy | Count | % of Total |
|-------------------|----------|-------|------------|
| 50-55% | 50.87% | 9165 | **90.9%** |
| 55-60% | 65.54% | 859 | 8.5% |
| 60-65% | 60.00% | 55 | 0.5% |
| 65-70% | 0% | 0 | 0% |

**Problem**: 91% of predictions are in lowest confidence bucket!

---

## 3. Issues Identified

### 3.1 Confidence Distribution Problem

The current system produces **too uniform** confidence distribution:
- 91% of predictions have confidence 50-55%
- Only 9% have higher confidence
- No predictions above 65%

**Root Cause**:
1. `ConfidenceCompressor` maps [0.5, 1.0] → [0.5, 0.70]
2. Model strengths are typically 0.05-0.30
3. Final confidence = 0.5 + strength * compression ≈ 0.52-0.58

### 3.2 Feature Overlap

Some features are used by multiple models but interpreted differently:
- `volatility_z`: Oscillator uses for penalty, VolumeMetrix for pattern
- `ma_delta`: TrendVIC uses for signal, VolumeMetrix for trend alignment

### 3.3 ML Classifier Underperformance

- Uses 31 features but only 37% accuracy
- Trained online during backtest (data leakage risk)
- No feature selection or importance weighting

---

## 4. Recommendations

### 4.1 Adaptive Calibration Improvements

```python
# Instead of fixed compression, use regime-based dynamic compression
REGIME_MAX_CONFIDENCE = {
    "trending_up": 0.75,    # Best regime - allow higher confidence
    "ranging": 0.70,        # Good regime
    "trending_down": 0.65,  # Problematic regime
    "volatile": 0.60,       # Worst regime - limit confidence
}

# Instead of linear compression, use sigmoid for better separation
def sigmoid_compress(strength, regime):
    max_conf = REGIME_MAX_CONFIDENCE.get(regime, 0.70)
    # Sigmoid: more separation in middle, compressed at extremes
    x = strength * 4 - 2  # Map [0, 1] to [-2, 2]
    sigmoid = 1 / (1 + exp(-x))
    return 0.5 + (max_conf - 0.5) * sigmoid
```

### 4.2 Feature Specialization

**TrendVIC** (Trend Following):
- Keep: ma_delta, body_ratio, candle_direction, price_momentum_3
- Add: ema_10_spread_pct, ema_20_spread_pct (trend strength)
- Remove: close (use ma_delta_pct instead)

**Oscillator** (Mean Reversion):
- Keep: rsi, rsi_momentum
- Add: rsi_oversold, rsi_overbought (zone indicators)
- Add: return_1 (recent move size for reversal potential)

**VolumeMetrix** (Volume-Price):
- Keep: volume_z, return_1, volume_trend
- Add: volume_change_pct (recent volume change)
- Add: atr_pct (volatility context)

### 4.3 ML Classifier Improvements

1. **Time-split training**: First 80% for training, last 20% for validation
2. **Feature selection**: Use top 15 features by importance
3. **Confidence thresholding**: Only output when probability > 0.55
4. **Ensemble weight reduction**: Lower ML weight when untrained

### 4.4 Confidence Boosting Strategy

To increase high-confidence predictions:

1. **Agreement Boost**: When 3+ models agree, boost confidence more
2. **Regime Alignment**: When prediction aligns with regime, boost
3. **Momentum Confirmation**: When multiple momentum indicators align

```python
def calculate_confidence_boost(models_agree, regime_aligned, momentum_aligned):
    boost = 0.0
    if models_agree >= 3:
        boost += 0.05
    if regime_aligned:
        boost += 0.03
    if momentum_aligned:
        boost += 0.02
    return min(boost, 0.10)  # Max 10% boost
```

---

## 5. Proposed Architecture Changes

### 5.1 Two-Stage Calibration

```
Stage 1: Model-Level Calibration
  - Each model outputs raw probability
  - Model-specific calibration based on historical accuracy
  - Output: calibrated probabilities per model

Stage 2: Ensemble Calibration
  - Combine calibrated model outputs
  - Apply agreement/regime bonuses
  - Final confidence capping
```

### 5.2 Feature Groups

```python
FEATURE_GROUPS = {
    "returns": ["return_1", "return_5", "return_10", "log_return_1"],
    "momentum": ["price_momentum_3", "rsi_momentum", "return_lag_1..5"],
    "volatility": ["volatility_z", "vol_ratio", "atr_pct"],
    "volume": ["volume_z", "volume_trend", "volume_change_pct"],
    "oscillators": ["rsi", "rsi_oversold", "rsi_overbought"],
    "trend": ["ma_delta_pct", "ema_10_spread_pct", "ema_20_spread_pct"],
    "candle": ["body_ratio", "body_pct", "upper_wick_ratio", "lower_wick_ratio"],
}

MODEL_FEATURE_GROUPS = {
    "TRENDVIC": ["trend", "momentum", "candle"],
    "OSCILLATOR": ["oscillators", "momentum", "volatility"],
    "VOLUMEMETRIX": ["volume", "returns", "candle"],
    "ML_CLASSIFIER": ["all"],  # Uses all features
}
```

---

## 6. Action Items

### Short-term (Sprint 16)
1. [ ] Implement regime-based max_confidence
2. [ ] Add sigmoid compression option
3. [ ] Increase agreement boost to 0.08

### Medium-term (Sprint 17)
1. [ ] Implement two-stage calibration
2. [ ] Add feature group analysis
3. [ ] ML classifier time-split training

### Long-term
1. [ ] Per-model calibration curves
2. [ ] Adaptive feature selection
3. [ ] Meta-model for confidence estimation

---

## 7. Current vs Target Metrics

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Ensemble Accuracy | 52.17% | 75%+ | -23% |
| Filtered Accuracy (≥55%) | 65.21% | 75%+ | -10% |
| High-conf Coverage | 9.07% | 30%+ | -21% |
| ECE | 1.95% | <5% | ✅ |
| p-value | 0.001 | <0.05 | ✅ |

**Key Insight**: Filtered accuracy is 65%, close to target. Need to increase high-confidence coverage from 9% to 30%+.
