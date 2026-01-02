# Titan Feature Analysis Results

**Date**: 2025-12-30
**Data**: BTCUSDT 1min, 7 days (10,139 records)

---

## 1. Model Performance Summary

| Model | Accuracy | Status |
|-------|----------|--------|
| **OSCILLATOR** | **51.1%** | BEST - Works! |
| TRENDVIC | 49.0% | Below random |
| VOLUMEMETRIX | 47.7% | Worst |

### Key Insight
On **1-minute timeframe**, mean reversion (Oscillator) works better than trend following (TrendVIC).
This confirms Chronos findings - shorter timeframes favor mean reversion.

---

## 2. Feature-Direction Correlation

Features that actually predict price direction:

| Feature | Correlation | Meaning |
|---------|-------------|---------|
| candle_direction | -0.060 | Bearish candle → UP next |
| return_1 | -0.051 | Down move → UP next |
| rsi_momentum | -0.045 | RSI falling → UP next |
| body_pct | +0.037 | Large body → UP next |
| atr_pct | +0.037 | High volatility → UP next |

**Interpretation**: Mean reversion dominates on 1-minute data!

---

## 3. Model-Specific Analysis

### 3.1 OSCILLATOR (51.1% - Best Model)

**What works:**
- RSI Q5 (high RSI): **53.6%** accuracy
- Volume_z Q5 (high volume): **54.2%** accuracy
- RSI + volume combination is powerful

**Current features** (good):
```python
["rsi", "rsi_momentum", "volatility_z"]
```

**Recommended additions:**
- `volume_z` (+0.027 effectiveness) ← **HIGH PRIORITY**
- `volume_change_pct` (+0.020)

**Optimal feature set:**
```python
OSCILLATOR_FEATURES = [
    "rsi",              # Primary signal
    "rsi_momentum",     # Reversal confirmation
    "rsi_oversold",     # Zone indicator
    "rsi_overbought",   # Zone indicator
    "volatility_z",     # High vol = weaker signal
    "volume_z",         # NEW: High volume confirms
    "return_1",         # Recent move (for reversal)
]
```

### 3.2 TRENDVIC (49.0% - Underperforming)

**Problem:** Most current features have ZERO effectiveness!

**Current features** (mostly useless):
```python
["ma_delta", "volatility", "close", "body_ratio", "candle_direction", "price_momentum_3"]
# close, ma_delta, volatility = 0.000 effectiveness!
```

**What actually helps:**
- `upper_wick_ratio` (-0.030): Low wick → better accuracy
- `body_ratio` Q5 (high): **50.3%** vs Q1: 46.6%
- `rsi` Q2: **51.6%** accuracy

**Recommended changes:**
```python
# REMOVE (ineffective)
- close           # 0.000 effectiveness
- volatility      # 0.000 effectiveness
- ma_delta        # 0.000 effectiveness

# ADD
+ upper_wick_ratio  # -0.030 effectiveness
+ rsi_neutral       # +0.022 effectiveness
+ ema_10_spread_pct # trend strength
+ return_5          # medium-term momentum
```

**Optimal feature set:**
```python
TRENDVIC_FEATURES = [
    "ma_delta_pct",      # Scale-invariant trend signal
    "ema_10_spread_pct", # Short-term trend
    "ema_20_spread_pct", # Medium-term trend
    "price_momentum_3",  # Rate of change
    "body_ratio",        # Candle confirmation
    "upper_wick_ratio",  # Rejection signal
    "return_5",          # 5-period momentum
]
```

### 3.3 VOLUMEMETRIX (47.7% - Needs Rework)

**Problem:** Most features have NEGATIVE effectiveness (model does WORSE when using them)!

**Current features** (problematic):
```python
["volume_z", "return_1", "volatility", "ma_delta", "volume_trend", "upper_wick_ratio", "lower_wick_ratio"]
# volume_z = -0.006 (slightly negative!)
# volume_trend = -0.006 (negative!)
# return_1 = +0.003 (almost zero)
```

**Root cause:** Volume-price analysis doesn't work well on 1-minute data without proper imbalance metrics.

**Missing critical feature:** `vol_imbalance` (volume UP vs DOWN)

**Recommended changes:**
```python
# REMOVE (ineffective or negative)
- volatility        # 0.000
- ma_delta          # 0.000
- volume_trend      # -0.006
- upper_wick_ratio  # -0.003

# ADD (from Chronos)
+ vol_imbalance_20  # CRITICAL - not yet implemented
+ volume_change_pct # -0.027 (informative)
+ body_pct          # -0.025
+ atr_pct           # volatility context
```

**Optimal feature set (after implementing vol_imbalance):**
```python
VOLUMEMETRIX_FEATURES = [
    "volume_z",           # Volume anomaly
    "vol_imbalance_20",   # NEW: UP vs DOWN volume
    "volume_change_pct",  # Recent volume change
    "return_1",           # Price movement
    "body_ratio",         # Candle structure
    "atr_pct",            # Volatility context
    "lower_wick_ratio",   # Buying pressure
]
```

---

## 4. Feature Quintile Analysis (Best Zones)

### Oscillator performs best when:
| Feature | Best Quintile | Accuracy |
|---------|---------------|----------|
| volume_z | Q5 (high) | **54.2%** |
| rsi | Q5 (high) | **53.6%** |
| volume_change_pct | Q5 (high) | **52.9%** |
| rsi_overbought | Q5 | **52.7%** |

**Insight:** Oscillator works best with **high RSI + high volume**

### TrendVIC performs best when:
| Feature | Best Quintile | Accuracy |
|---------|---------------|----------|
| rsi | Q2 (low-mid) | **51.6%** |
| upper_wick_ratio | Q3 (middle) | **51.5%** |
| rsi_neutral | Q5 | **51.1%** |
| body_ratio | Q5 (high) | **50.3%** |

**Insight:** TrendVIC works best with **neutral RSI + strong candles**

### VolumeMetrix performs best when:
| Feature | Best Quintile | Accuracy |
|---------|---------------|----------|
| candle_direction | Q2 | **50.3%** |
| body_pct | Q2 | **49.4%** |
| volume_change_pct | Q3 (middle) | **48.9%** |

**Insight:** VolumeMetrix struggles everywhere, needs fundamental rework

---

## 5. Critical Missing Features (from Chronos)

### 5.1 Volume Imbalance (CRITICAL for VolumeMetrix)
```python
# Implementation needed in FeatureStream
vol_up = volume * (return_1 > 0)
vol_down = volume * (return_1 < 0)
vol_imbalance_20 = (vol_up.rolling(20).sum() - vol_down.rolling(20).sum()) /
                   (vol_up.rolling(20).sum() + vol_down.rolling(20).sum())
```

### 5.2 Bollinger Bands (for Oscillator)
```python
sma_20 = close.rolling(20).mean()
std_20 = close.rolling(20).std()
bb_upper = sma_20 + 2 * std_20
bb_lower = sma_20 - 2 * std_20
bb_position = (close - bb_lower) / (bb_upper - bb_lower)  # 0-1 range
bb_width_pct = (bb_upper - bb_lower) / sma_20  # Volatility squeeze
```

### 5.3 MACD Percentage (for TrendVIC)
```python
ema_12 = close.ewm(span=12).mean()
ema_26 = close.ewm(span=26).mean()
macd_pct = (ema_12 - ema_26) / close * 100  # Scale-invariant
```

---

## 6. Implementation Priority

### Sprint 16: High Priority
1. **Add `volume_z` to Oscillator** - easy win, +3% potential
2. **Remove ineffective features from TrendVIC** (close, volatility, ma_delta)
3. **Add `vol_imbalance_20` to FeatureStream** - CRITICAL

### Sprint 17: Medium Priority
4. **Add Bollinger Bands** to FeatureStream (bb_position, bb_width_pct)
5. **Add MACD percentage** to FeatureStream
6. **Rework VolumeMetrix** with new features

### Sprint 18: Long-term
7. **Consider 5-minute timeframe** - trend following works better
8. **Add adaptive feature selection** per regime

---

## 7. Expected Results

| Metric | Current | After Sprint 16 | After Sprint 17 |
|--------|---------|-----------------|-----------------|
| Oscillator | 51.1% | 53-54% | 55%+ |
| TrendVIC | 49.0% | 50-51% | 52%+ |
| VolumeMetrix | 47.7% | 49-50% | 51%+ |
| **Ensemble** | ~49% | 51-52% | 53-55% |
| High-conf coverage | 9% | 15-20% | 25%+ |

---

## 8. Summary

### What we learned:
1. **Oscillator is the best model** on 1-minute data (mean reversion works)
2. **TrendVIC has dead features** - close, volatility, ma_delta don't help
3. **VolumeMetrix needs vol_imbalance** - core feature is missing
4. **High volume + extreme RSI = highest accuracy** (54%)

### Quick wins:
1. Add `volume_z` to Oscillator
2. Remove dead features from TrendVIC
3. Add `vol_imbalance` to FeatureStream

### Fundamental insight:
On 1-minute data, **mean reversion dominates**. For trend following to work better,
consider 5-minute or higher timeframes (as Chronos suggests).
