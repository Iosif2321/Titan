# Titan - Cryptocurrency Direction Prediction System

## Project Overview

Titan is a cryptocurrency price direction prediction system targeting 75%+ accuracy.

**IMPORTANT: This project is ONLY about PREDICTING price direction (UP/DOWN), NOT trading!**

### Key Principle
- **FLAT is NOT a third market direction** - it's model uncertainty
- **Actual direction is ALWAYS UP or DOWN** - price always moves
- Models should minimize FLAT predictions (uncertainty state)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI (cli.py)                            │
│  Commands: backtest, history, live                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    history.py / backtest.py                     │
│  - Load data (Bybit REST API)                                   │
│  - Gap validation                                               │
│  - Prefill warmup                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       FeatureStream                             │
│  Features: close, return_1, ma_fast/slow, volatility_z,        │
│            volume_z, rsi, ma_delta                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Models                                  │
│  TrendVIC │ Oscillator │ VolumeMetrix                          │
│     ▼           ▼            ▼                                  │
│  ModelOutput(prob_up, prob_down, state, metrics)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               Ensemble + RegimeDetector + Monitor               │
│  - Model weighting (adaptive by regime)                         │
│  - Performance monitoring per regime                            │
│  - Decision(direction, confidence, prob_up, prob_down)         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              BacktestStats + PredictionAnalyzer                 │
│  - Prediction vs actual comparison                              │
│  - Advanced analysis (temporal, magnitude, streaks)             │
│  - Statistical significance testing                             │
│  - Report generation                                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Current Status (Sprint 3 Complete)

| Metric | Baseline | After Sprint 2 | After Sprint 3 | Target |
|--------|----------|----------------|----------------|--------|
| Ensemble | 40.17% | 49.55% | **49.76%** | 75%+ |
| TrendVIC | 46.42% | 50.24% | **50.2%** | 75%+ |
| Oscillator | 17.03% | 45.80% | **46.1%** | 75%+ |
| VolumeMetrix | 35.16% | 46.28% | **46.4%** | 75%+ |
| ECE | ~10% | 3.72% | **5.2%** | <10% |
| FLAT rate | ~30% | 0% | **0%** | <5% |

### Key Insights from Analysis
- **Best session**: Europe (50.6%)
- **Worst session**: Asia (48.5%)
- **Best hour**: 3:00 UTC (60.0%)
- **Worst hour**: 7:00 UTC (40.0%)
- **Best movement size**: Large (55.9%)
- **Worst movement size**: Medium (44.6%)
- **Worst regime**: Volatile (56.2% error rate)
- **Best regime**: Trending Up (46.1% error rate)
- **Statistical significance**: p=0.6 (NOT significant vs random)

---

## Completed Sprints

### Sprint 0: Baseline
- Initial metrics collection
- Baseline accuracy ~40%

### Sprint 1: Foundation Fixes
- Corrected prefill calculation (21 → 40 bars)
- Added gap validation for candle continuity
- Dynamic Sharpe/Sortino annualization by interval
- Removed epsilon for actual direction (always UP/DOWN)

### Sprint 2: Model Improvements
- Minimized FLAT predictions to 0%
- Improved individual model accuracy
- Better confidence calibration (ECE 3.72%)

### Sprint 3: Regime-Based Adaptation
- Created `RegimeDetector` (4 regimes: trending_up, trending_down, ranging, volatile)
- Created `PerformanceMonitor` (rolling accuracy per model per regime)
- Created `AdaptiveWeightManager` (regime-aware weights)
- Integrated regime detection into Ensemble
- Added regime tracking to BacktestStats

### Sprint 3.5: Advanced Analysis System
- Created `TemporalAnalyzer` (by hour, session, day)
- Created `MagnitudeAnalyzer` (by movement size)
- Created `StreakAnalyzer` (error/correct streaks)
- Created `StatisticalValidator` (p-value, confidence intervals)
- Created `ErrorExplorer` (confident wrong, feature comparison)
- Integrated all into backtest reports

---

## Key Files

### Core System
- `titan/core/backtest.py` - Main backtest engine with analysis integration
- `titan/core/ensemble.py` - Model ensemble with regime adaptation
- `titan/core/features/stream.py` - Feature calculation (11 features)
- `titan/core/models/heuristic.py` - TrendVIC, Oscillator, VolumeMetrix models

### Regime & Adaptation (Sprint 3)
- `titan/core/regime.py` - RegimeDetector class
- `titan/core/monitor.py` - PerformanceMonitor class
- `titan/core/weights.py` - WeightManager + AdaptiveWeightManager

### Analysis System (Sprint 3.5)
- `titan/core/analysis.py` - Full analysis toolkit:
  - `PredictionDetail` - Individual prediction record
  - `TemporalAnalyzer` - Time-based analysis
  - `MagnitudeAnalyzer` - Movement size analysis
  - `StreakAnalyzer` - Consecutive prediction analysis
  - `StatisticalValidator` - Statistical tests
  - `ErrorExplorer` - Error investigation
  - `PredictionAnalyzer` - Combined analyzer

### Configuration
- `titan/core/config.py` - ConfigStore with all parameters
- `titan/core/calibration.py` - Online confidence calibration

### Documentation
- `docs/TESTING_SYSTEM_ANALYSIS.md` - Analysis of testing system and improvements

---

## Regime Detection Logic

```python
# RegimeDetector.detect()
vol_z = features.get("volatility_z", 0.0)
ma_delta = features.get("ma_delta", 0.0)
normalized_delta = abs(ma_delta) / (close + 1e-12)

if vol_z > 1.5:           # High volatility
    return "volatile"
if normalized_delta > 0.0003:  # Strong trend
    return "trending_up" if ma_delta > 0 else "trending_down"
return "ranging"          # Default: sideways market
```

## Regime Weights

```python
DEFAULT_REGIME_WEIGHTS = {
    "trending_up":   {"TRENDVIC": 0.50, "OSCILLATOR": 0.20, "VOLUMEMETRIX": 0.30},
    "trending_down": {"TRENDVIC": 0.50, "OSCILLATOR": 0.20, "VOLUMEMETRIX": 0.30},
    "ranging":       {"TRENDVIC": 0.20, "OSCILLATOR": 0.50, "VOLUMEMETRIX": 0.30},
    "volatile":      {"TRENDVIC": 0.25, "OSCILLATOR": 0.25, "VOLUMEMETRIX": 0.50},
}
```

---

## Running Tests

```bash
# 24-hour backtest
python -m titan.cli history --symbol BTCUSDT --interval 1 --hours 24

# 7-day backtest
python -m titan.cli history --symbol BTCUSDT --interval 1 --hours 168

# Live mode
python -m titan.cli live --symbol BTCUSDT --interval 1
```

---

## Report Output

The backtest generates comprehensive reports in `runs/history_*/`:
- `summary.json` - Full metrics in JSON
- `report.md` - Human-readable markdown report
- `predictions.jsonl` - Individual predictions

### Report Sections
1. Quick Summary (accuracy, ECE, sharpe)
2. Direction Distribution (predicted vs actual)
3. Confusion Matrix
4. Per-Model Performance
5. Model Agreement
6. Trading Simulation (for reference only)
7. Accuracy Stability
8. Feature Effectiveness
9. **Statistical Significance** (NEW)
10. **Accuracy by Session** (NEW)
11. **Accuracy by Movement Size** (NEW)
12. **Error Streaks** (NEW)
13. **Confident Wrong Predictions** (NEW)
14. **Error Rate by Regime** (NEW)

---

## Next Steps (Sprint 4+)

### Sprint 4: Selective High-Confidence
- Filter predictions by confidence for evaluation
- Track accuracy per confidence bucket
- Target: 60-65% on high-confidence predictions

### Sprint 5: Multi-Timeframe
- Use 1m + 5m + 15m data
- Timeframe consensus for stronger signals
- Target: 65-70%

### Sprint 6: ML Enhancement
- Add lightweight ML model (XGBoost/LightGBM)
- Online learning for adaptation
- Target: 70-75%

---

## Known Issues & Areas for Improvement

1. **Volatile regime has highest error rate (56.2%)** - Need better volatile handling
2. **Medium movements have lowest accuracy (44.6%)** - Focus area
3. **Not statistically significant (p=0.6)** - Need more data or better models
4. **Oscillator still weakest (46.1%)** - Needs improvement
5. **Error streaks up to 7** - Need streak-breaking logic

---

## Configuration Parameters

Key parameters in `config.py`:
```python
# Regime detection
regime.vol_z_high = 1.5
regime.trend_threshold = 0.0003

# Model thresholds
model.trendvic.dead_zone = 0.00005
model.oscillator.overbought = 70
model.oscillator.oversold = 30
model.volumemetrix.high_volume = 1.5

# Ensemble
ensemble.flat_threshold = 0.50
ensemble.min_confidence = 0.50
```

---

## Development Notes

- Always run 24h backtest after changes to verify no regression
- Check report.md for regime-specific performance
- Monitor ECE to ensure calibration stays <10%
- FLAT rate should stay at 0% (models always predict UP/DOWN)
- Use statistical significance to validate improvements
