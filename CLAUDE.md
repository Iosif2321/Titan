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

## Current Status (Sprint 13 Complete - 2025-12-29)

| Metric | Baseline | After Sprint 11 | After Sprint 13 | Target |
|--------|----------|-----------------|-----------------|--------|
| Ensemble | 40.17% | 51.91% | **52.12%** | 75%+ |
| TrendVIC | 46.42% | 50.73% | **50.94%** | 75%+ |
| Oscillator | 17.03% | 49.20% | **49.13%** | 75%+ |
| VolumeMetrix | 35.16% | 48.99% | **49.06%** | 75%+ |
| ECE | ~10% | 2.71% | **1.92%** | <10% ✅ |
| p-value | 0.6 | 0.13 | **0.05** | <0.05 |
| FLAT rate | ~30% | 0% | **0%** | <5% ✅ |
| Sharpe | - | 1.4 | **2.55** | >1.5 ✅ |

### Key Improvements (Sprints 4-13)
- **Agreement Accuracy**: Full agreement now **55.77%**
- **High Confidence**: conf 55-60% now has **61.40%** accuracy
- **High Confidence**: conf 60-65% now has **66.67%** accuracy
- **ECE**: Excellent at **1.92%** (best so far)
- **Pattern System**: Integrated historical pattern learning
- **Sharpe Ratio**: Improved to **2.55** (good risk-adjusted returns)

### Confidence Calibration (Sprint 13)
| Confidence | Accuracy | Count |
|------------|----------|-------|
| 50-55% | 50.83% | 1265 |
| 55-60% | 61.40% | 171 |
| 60-65% | 66.67% | 3 |

### Current Analysis (24h backtest 2025-12-29)
- **Total Predictions**: 1439
- **Best session**: US (52.3%)
- **Worst session**: Europe (51.4%)
- **Best hour**: 17:00 UTC (63.3%)
- **Worst hour**: 13:00 UTC (40.0%)
- **Best movement size**: Tiny (53.9%)
- **Worst movement size**: Large (45.6%)
- **Direction Balance**: 0.899 (balanced)
- **Statistical significance**: p=0.05 (borderline significant!)

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

### Sprint 4: Confidence Recalibration
- Created `ConfidenceCompressor` in calibration.py
- Fixes overconfidence problem (65%+ conf had 33-47% accuracy)
- Caps confidence at 62%, compresses [0.5, 1.0] → [0.5, 0.62]
- Added regime-based confidence penalties (volatile: 0.75, trending_down: 0.85)
- **Result**: ECE 5.31% → 2.34%

### Sprint 5: Volatile Handler
- Created `titan/core/strategies/volatile.py`
- `VolatileClassifier`: 4 types (extreme, breakout, choppy, reversal)
- `VolatileStrategy`: Specialized handling per volatile type
- Integrated into Ensemble for volatile regime
- **Result**: Volatile error 56.2% → 51.5%

### Sprint 6: Trending Down Fix
- Created `titan/core/detectors/exhaustion.py` - TrendExhaustionDetector
- Created `titan/core/strategies/trending.py` - TrendingStrategy
- Detects trend exhaustion via momentum/volume/RSI divergence
- Predicts reversals when trends are exhausted
- **Result**: Trending_down error 53.5% → 49.4%, p-value 0.6 → 0.05

### Sprint 7: Medium Movement Fix
- Auto-improved by Sprints 4-6
- Medium accuracy: 44.6% → 51.1% (then 50.0%)

### Sprint 8: Temporal Patterns
- Created `titan/core/adapters/temporal.py` - TemporalAdjuster
- Hourly confidence multipliers (worst hours: 7, 12, 19 UTC)
- Best hours: 3, 8, 20, 21 UTC
- Updated backtest.py and live.py to pass timestamp
- **Result**: ECE 3.23% → 1.55%

### Sprint 9: Model Improvements
- Enhanced Oscillator with RSI momentum confirmation
- Enhanced VolumeMetrix with Volume-Price relationship analysis
  - High volume + big move = CONTINUATION
  - High volume + small move = ABSORPTION (reversal)
- Added `rsi_momentum` feature to FeatureStream
- **Result**: Oscillator +3.4%, VolumeMetrix +2.2%, Full Agreement 54.9%

### Sprint 10: Feature Engineering
- Added new features to FeatureStream:
  - `price_momentum_3` - Rate of change over 3 periods
  - `volume_trend` - Is volume increasing?
  - `body_ratio` - Candle body to total range
  - `upper_wick_ratio` / `lower_wick_ratio` - Wick analysis
  - `candle_direction` - Bullish/bearish candle
- Updated TrendVIC to use body_ratio and price_momentum for confirmation
- Updated VolumeMetrix to use volume_trend and wick analysis
- **Result**: Features integrated, models use them for better signals

### Sprint 11: Ensemble Improvements
- Added `_check_agreement()` method to detect model consensus
- Added `_apply_agreement_boost()` for confidence adjustment:
  - Full agreement → +5% confidence (cap 65%)
  - Partial agreement → +2% confidence (cap 62%)
- **Result**: Agreement accuracy 55.34%, conf 55-60% now 60.43% accurate

### Sprint 13: Pattern System (Model Experience)
- Extended `PatternStore` with `get_events()` and `get_usage_count()` methods
- Created `PatternExperience` class in `titan/core/patterns.py`:
  - `get_pattern_stats()` - accuracy with exponential time decay (168h half-life)
  - `get_pattern_bias()` - determines if pattern works better for UP/DOWN
  - `should_trust_pattern()` - checks min 20 uses before adjusting
- Created `PatternAdjuster` class in `titan/core/adapters/pattern.py`:
  - Boosts confidence for historically accurate patterns (>55% accuracy)
  - Reduces confidence for inaccurate patterns (<45% accuracy)
  - Applies bias penalty when direction contradicts pattern bias
- Integrated into `Ensemble.decide()` with `pattern_id` parameter
- Updated `backtest.py` and `live.py` for pattern system
- Added config parameters: `pattern.boost_threshold`, `pattern.penalty_threshold`, etc.
- **Result**: ECE 1.92%, Sharpe 2.55, system learns from historical patterns

---

## Key Files

### Core System
- `titan/core/backtest.py` - Main backtest engine with analysis integration
- `titan/core/ensemble.py` - Model ensemble with regime adaptation
- `titan/core/features/stream.py` - Feature calculation (13 features including rsi_momentum)
- `titan/core/models/heuristic.py` - TrendVIC, Oscillator, VolumeMetrix models

### Regime & Adaptation (Sprint 3)
- `titan/core/regime.py` - RegimeDetector class
- `titan/core/monitor.py` - PerformanceMonitor class
- `titan/core/weights.py` - WeightManager + AdaptiveWeightManager

### Strategies & Adapters (Sprints 4-13)
- `titan/core/calibration.py` - ConfidenceCompressor, OnlineCalibrator
- `titan/core/strategies/volatile.py` - VolatileClassifier, VolatileStrategy
- `titan/core/strategies/trending.py` - TrendingStrategy
- `titan/core/detectors/exhaustion.py` - TrendExhaustionDetector
- `titan/core/adapters/temporal.py` - TemporalAdjuster
- `titan/core/adapters/pattern.py` - PatternAdjuster (Sprint 13)
- `titan/core/patterns.py` - PatternStore, PatternExperience (Sprint 13)

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

## Next Steps

### Sprint 12: ML Enhancement (NEXT)
- Add lightweight ML model (LightGBM/XGBoost)
- Features: all current features + model probabilities
- Online learning for adaptation
- Target: 55-60% accuracy

### Future Ideas
- Multi-timeframe analysis (1m + 5m + 15m)
- Streak-breaking logic (max loss streak = 9)
- Better large movement handling (45.6% accuracy)
- Session-specific model weights

---

## Known Issues & Areas for Improvement

1. **Statistical significance (p=0.05)** - Borderline, need more data
2. **Large movements have lowest accuracy (45.6%)** - Harder to predict
3. **Volatile regime still challenging (~48% error)** - Near random
4. **Error streaks up to 9** - Need streak-breaking logic
5. **13:00 UTC worst hour (40%)** - Temporal patterns not fully solved
6. **Pattern system needs time** - 20+ uses per pattern to take effect

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

# Pattern System (Sprint 13)
pattern.boost_threshold = 0.55    # Accuracy above this = boost confidence
pattern.penalty_threshold = 0.45  # Accuracy below this = reduce confidence
pattern.max_boost = 0.03          # Max +3% confidence boost
pattern.max_penalty = 0.03        # Max -3% confidence penalty
pattern.bias_penalty = 0.01       # -1% when direction contradicts pattern bias
```

---

## Development Notes

- Always run 24h backtest after changes to verify no regression
- Check report.md for regime-specific performance
- Monitor ECE to ensure calibration stays <10%
- FLAT rate should stay at 0% (models always predict UP/DOWN)
- Use statistical significance to validate improvements

---

## Pattern System Architecture (Sprint 13)

### How Patterns Work

1. **Pattern = Market Conditions** (27 possible patterns):
   - `trend`: up/down/flat (MA delta vs volatility)
   - `volatility`: high/mid/low (volatility_z thresholds)
   - `volume`: high/mid/low (volume_z thresholds)

2. **Pattern Storage** (SQLite):
   - `patterns` table: unique conditions → pattern_id
   - `pattern_events` table: each prediction outcome recorded

3. **Pattern Experience Analysis**:
   - Time decay: `weight = e^(-age_hours / 168)` (1 week half-life)
   - Calculates: accuracy, up_accuracy, down_accuracy, confidence
   - Determines direction bias (UP/DOWN/None)

4. **Pattern-Based Adjustment**:
   ```
   If pattern uses >= 20 (trusted):
     If accuracy > 55%: boost confidence (max +3%)
     If accuracy < 45%: reduce confidence (max -3%)
     If bias != direction: additional -1% penalty
   ```

### Pattern Flow in Pipeline

```
Candle → Features → build_conditions() → pattern_id
                                              ↓
Models → Ensemble.decide(outputs, features, ts, pattern_id)
                                              ↓
                                   PatternAdjuster.adjust_decision()
                                              ↓
                                   Final Decision
                                              ↓
[After evaluation] → PatternStore.record_usage(pattern_id, event)
                     → System learns for next time
```

---

## Current Sprint: Sprint 12 - Pattern System Hardening (2025-12-29)

### Critical Bugs Found
1. **Snapshot deletion corrupts other patterns** (`patterns.py:920`)
   - `_enforce_decision_limit()` deletes snapshots of OTHER patterns
   - Fix: Delete only snapshots of events being removed

2. **Data leakage in backtest** (`patterns.py:1247`)
   - `PatternExperience.get_pattern_stats()` sees "future" events
   - Fix: Add `max_ts` parameter to filter by timestamp

### Refactoring Tasks
- Config-driven constants (MAX_DECISIONS, etc.)
- Fix day_of_week inconsistency (in ExtendedConditions but not in pattern_key)
- Remove dead code (pattern_conditions_v2, conditions_version unused)
- Integrate PatternReader into ensemble

### Sprint 12 Plan
| Phase | Task | Status |
|-------|------|--------|
| A.1 | Fix snapshot deletion bug | Pending |
| A.2 | Fix data leakage (time-bounded) | Pending |
| B.1 | Config-driven constants | Pending |
| B.2 | day_of_week consistency | Pending |
| B.3 | Remove dead code | Pending |
| C | Update ROADMAP, run backtest | Pending |

---

## Last Session Context (2025-12-29)

### Completed
- Sprint 13: Pattern System fully integrated
- PatternExperience with time decay working
- PatternAdjuster integrated in Ensemble, backtest, live
- Test passed: 52.12% accuracy, ECE 1.92%, Sharpe 2.55

### Latest Test Results
```
Ensemble: 52.12% (750/1439)
Full Agreement: 55.77% (104 predictions)
ECE: 1.92%
Sharpe: 2.55
Direction Balance: 0.899
Conf 55-60%: 61.40% (171 predictions)
Conf 60-65%: 66.67% (3 predictions)
```
