# Titan - Cryptocurrency Direction Prediction System

## Project Overview

Titan is a cryptocurrency price direction prediction system targeting 75%+ accuracy.

**IMPORTANT: This project is ONLY about PREDICTING price direction (UP/DOWN), NOT trading!**

### Key Principle
- **FLAT is NOT a third market direction** - it's model uncertainty
- **Actual direction is ALWAYS UP or DOWN** - price always moves
- Models should minimize FLAT predictions (uncertainty state)

---

## Architecture (Legacy - Heuristic Models)

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

## NEW Architecture: Three-Head TFT Model (Sprint 23)

**Заменяем 3 эвристические модели на 3 ML-модели с reinforcement learning.**

### Общая архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                    THREE-HEAD TFT MODEL                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUTS:                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ Trend    │  │Oscillator│  │ Volume   │  │ Pattern History  │ │
│  │ Features │  │ Features │  │ Features │  │ (last 50 events) │ │
│  │ (12 dim) │  │ (10 dim) │  │ (8 dim)  │  │ + Aggregates     │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘ │
│       │             │             │                  │           │
│       ▼             ▼             ▼                  ▼           │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              SHARED TFT ENCODER (hidden=64)                 ││
│  │  - Variable Selection Network                               ││
│  │  - LSTM Encoder (seq_len=100)                              ││
│  │  - Multi-Head Attention (heads=4)                          ││
│  │  - Gated Residual Network                                  ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                   │
│       ┌──────────────────────┼──────────────────────┐           │
│       ▼                      ▼                      ▼           │
│  ┌─────────┐           ┌─────────┐           ┌─────────┐        │
│  │ TREND   │           │OSCILLAT │           │ VOLUME  │        │
│  │  HEAD   │           │  HEAD   │           │  HEAD   │        │
│  │(prob_up,│           │(prob_up,│           │(prob_up,│        │
│  │prob_down│           │prob_down│           │prob_down│        │
│  └─────────┘           └─────────┘           └─────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Ключевые решения

| Аспект | Решение | Обоснование |
|--------|---------|-------------|
| **Архитектура** | TFT (Temporal Fusion Transformer) | Читает большой контекст, interpretable |
| **GPU** | GTX 1060 (6GB) | hidden=64, heads=4, seq=100 |
| **Inputs** | Специализированные фичи для каждой головы | TrendML: MA, momentum; OscillatorML: RSI, BB; VolumeML: volume, MFI |
| **Training** | Hybrid: Offline pretrain + Online fine-tune | Баланс стабильности и адаптации |
| **Reward** | R2 (return-weighted) + streak bonus | `reward = return_pct * direction_match * streak_mult` |
| **Patterns** | Агрегаты + Attention к последним 50 событиям | Summary + deep analysis |

### Специализированные фичи

**TrendML (12 features):**
- ma_fast, ma_slow, ma_delta, ma_delta_pct
- ema_10_spread_pct, ema_20_spread_pct
- adx, price_momentum_3, return_5, return_10
- body_ratio, candle_direction

**OscillatorML (10 features):**
- rsi, rsi_momentum, rsi_oversold, rsi_overbought
- bb_position, stochastic_k, stochastic_d
- mfi, upper_wick_ratio, lower_wick_ratio

**VolumeML (8 features):**
- volume_z, volume_trend, volume_change_pct
- vol_imbalance_20, vol_ratio
- atr_pct, high_low_range_pct, body_pct

### Reward Function

```python
def calculate_reward(direction_match, return_pct, streak_length):
    streak_mult = 1.0 + 0.1 * min(streak_length, 5)  # max 1.5x
    reward = return_pct * direction_match * streak_mult
    return reward
```

### Pattern Integration

```python
# Каждый prediction получает:
pattern_input = {
    # Агрегаты (быстрый summary)
    "pattern_accuracy": 0.58,
    "pattern_up_accuracy": 0.62,
    "pattern_down_accuracy": 0.54,
    "pattern_uses": 100,
    "pattern_bias": "UP",

    # Attention к последним 50 событиям паттерна
    "pattern_events": [  # shape: (50, event_dim)
        {"hit": True, "return_pct": 0.0012, "confidence": 0.58, ...},
        ...
    ]
}
```

### Размеры для GTX 1060

| Параметр | Значение | Memory |
|----------|----------|--------|
| hidden_dim | 64 | ~1.5GB model |
| num_heads | 4 | |
| lstm_layers | 2 | |
| seq_len | 100 candles | |
| pattern_events | 50 | |
| batch_size | 32 | ~2GB training |
| **Total** | | **~3.5GB** ✅ |

### Файлы для создания

| Файл | Описание |
|------|----------|
| `titan/core/models/tft.py` | Three-Head TFT Model |
| `titan/core/models/heads.py` | Specialized prediction heads |
| `titan/core/training/trainer.py` | Hybrid training loop |
| `titan/core/training/reward.py` | Reward calculator |

---

## Current Status (Sprint 20 Complete - 2025-12-30)

| Metric | Sprint 18 | Sprint 20 | Target |
|--------|-----------|-----------|--------|
| Ensemble | 48.85% | **49.55%** | 75%+ |
| TrendVIC | 47.2% | **47.32%** | 75%+ |
| ML Classifier | 41.9% | **51.71%** ↑ | 75%+ |
| Oscillator | 44.3% | **44.68%** | 75%+ |
| VolumeMetrix | 46.8% | **46.91%** | 75%+ |
| ECE | 1.55% | **1.02%** ✅ | <10% |
| p-value | 0.3 | **0.6** | <0.05 |
| FLAT rate | 0% | **0%** ✅ | <5% |

### Sprint 20: Online Learning (2025-12-30)

Real-time model adaptation with SGD + RMSProp:

| Component | Description |
|-----------|-------------|
| **OnlineLearner** | SGD + RMSProp optimizer for weight updates |
| **MultiScaleEMA** | 3 time scales (short ~20, medium ~100, long ~1000) |
| **RewardCalculator** | Binary, confidence, return, risk-adjusted rewards |
| **OnlineAdapter** | Main interface for real-time adaptation |
| **Trend Detection** | Detects improving/degrading/stable model accuracy |

**EMA Tracking (per model):**
- TRENDVIC: short=0.53, medium=0.47, long=0.36, **trend=improving**
- OSCILLATOR: short=0.37, medium=0.46, long=0.58, trend=degrading
- VOLUMEMETRIX: short=0.49, medium=0.49, long=0.36, trend=stable
- ML_CLASSIFIER: short=0.49, medium=0.49, long=0.18, trend=stable

### Sprint 18: ML Hardening

Enhanced ML classifier with validation and calibration:

| Component | Description |
|-----------|-------------|
| **Walk-Forward CV** | Time-split cross-validation (no data leakage) |
| **Isotonic Calibration** | Calibrates probabilities to match empirical accuracy |
| **Feature Importance** | Identifies top/weak features automatically |
| **Feature Selection** | Removes features with importance < 1% |

**Top Features by Importance:**
1. `return_lag_3` (88) - Lagged return 3 periods ago
2. `volume_change_pct` (86) - Volume change percentage
3. `body_ratio` (84) - Candle body to range ratio

### Sprint 17: SessionAdapter (2025-12-30)

Implemented per-session (ASIA/EUROPE/US) model configuration using Thompson Sampling:

| Component | Description |
|-----------|-------------|
| **SessionMemory** | SQLite-based persistent storage for session stats |
| **SessionAdapter** | Thompson Sampling for parameter selection |
| **Decay Mechanism** | 168h half-life for forgetting old data |
| **Trust Blocks** | Min 50 samples + max 10% CI width |
| **Shrinkage** | `effective_acc = (global_acc * k + session_acc * n) / (k + n)` |
| **Calibration** | Per-session confidence temperature scaling |

**Key Config Parameters:**
```python
"session_adapter.enabled": True,
"session_adapter.min_samples": 50,       # Trust threshold
"session_adapter.half_life_hours": 168,  # 1 week decay
"session_adapter.prior_strength": 1000,  # Shrinkage strength
```

### Sprint 16: New Features (2025-12-30)
Added 4 new scale-invariant features:
- `bb_position` - Position within Bollinger Bands [0, 1]
- `vol_imbalance_20` - Volume imbalance over 20 periods
- `adx` - Average Directional Index (trend strength)
- `mfi` - Money Flow Index

### Key Finding
System is statistically significant (p=0.001) but accuracy plateaued around 51-52%.
**Session-based performance varies**: ASIA 50.5%, EUROPE 46.5%, US 49.8% → 4% gap justifies per-session adaptation.

### Sprint 14: LightGBM Classifier (2025-12-30)

| Component | Description |
|-----------|-------------|
| **ML_FEATURES** | 31 scale-invariant features (returns, RSI, volatility, candle structure) |
| **DirectionalClassifier** | LightGBM binary classifier (UP/DOWN only) |
| **Training** | Online learning: 500 min samples, retrain every 1000 samples |
| **Integration** | Added to Ensemble with 25% weight in all regimes |

### 7-Day Backtest Results (168h, 2025-12-30)
- **Total Predictions**: 10,079
- **Ensemble Accuracy**: 52.16% (statistically BETTER than random, p=0.001)
- **TrendVIC**: 47.03% accuracy, FLAT=466 (4.6%)
- **Oscillator**: 44.03% accuracy, FLAT=1645 (16.3%)
- **VolumeMetrix**: 41.79% accuracy, FLAT=1443 (14.3%)
- **ML_CLASSIFIER**: 36.41% accuracy (training online, needs improvement)
- **Full Agreement**: 55.16% (1066 predictions)
- **Direction Balance**: 0.85 (balanced)

### Accuracy by Session
| Session | Accuracy | Count |
|---------|----------|------:|
| ASIA | 53.1% | 3360 |
| US | 52.1% | 4200 |
| EUROPE | 51.0% | 2519 |

### Best/Worst Hours
- **Best hour**: 4:00 UTC (57.9%)
- **Worst hour**: 13:00 UTC (48.6%)

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

### Sprint 14: LightGBM Classifier (2025-12-30)
- Added 17 new scale-invariant features to `FeatureStream`:
  - Lagged returns (return_lag_1..5)
  - ATR percentage, high-low range percentage
  - EMA spreads (10 and 20 period)
  - Multi-period returns (5 and 10 period)
  - RSI zones (oversold, overbought, neutral)
  - Volume change percentage, body percentage, vol ratio
- Created `DirectionalClassifier` in `titan/core/models/ml.py`:
  - LightGBM binary classifier for UP/DOWN prediction
  - 31 scale-invariant features (ML_FEATURES)
  - Online training: accumulates samples, trains when 500+ samples
  - Never outputs FLAT (always UP or DOWN)
  - Includes save/load and feature importance methods
- Integrated into Ensemble with 25% weight across all regimes
- Updated `weights.py` with ML_CLASSIFIER in DEFAULT_REGIME_WEIGHTS
- Added config parameters: `ml.min_samples`, `ml.train_interval`, etc.
- **Result**: Ensemble 52.16%, p=0.001, statistically significant

### Sprint 15: Confidence Filtering (2025-12-30)
- Increased `confidence_compressor.max_confidence` from 0.62 to 0.70
- Added `confidence_filter.threshold` config parameter (default 0.55)
- Added filtered accuracy tracking to `BacktestStats`:
  - `filtered_total`, `filtered_correct`, `filtered_confidence_sum`
  - Coverage calculation (percentage of actionable predictions)
- Added "FILTERED ACCURACY" section to backtest output
- Added `filtered_accuracy` section to summary.json
- **Result**: Filtered accuracy **65.21%** at conf ≥55% threshold
  - Coverage: 9.07% (914/10,079 predictions)
  - Full agreement accuracy: 56.08%

### Sprint 16: New Features (2025-12-30)
- Added 4 new scale-invariant features:
  - `bb_position` - Position within Bollinger Bands [0, 1]
  - `vol_imbalance_20` - Volume imbalance over 20 periods
  - `adx` - Average Directional Index (trend strength)
  - `mfi` - Money Flow Index
- **Key Finding**: Session-based performance varies (ASIA 50.5%, EUROPE 46.5%, US 49.8%)
  - 4% gap justifies per-session adaptation → Sprint 17

### Sprint 17: SessionAdapter (2025-12-30)
- Created `SessionMemory` class in `titan/core/adapters/session.py`:
  - SQLite schema for persistent session stats storage
  - Tables: session_stats, global_stats, session_calibration, param_stats
  - Methods: get/update stats with decay, aggregation across regimes
- Created `SessionAdapter` class:
  - Thompson Sampling (Contextual Bandits) for parameter selection
  - Shrinkage formula: `effective_acc = (global_acc * k + session_acc * n) / (k + n)`
  - Decay mechanism: 168h half-life for forgetting old data
  - Trust blocks: min 50 samples + max 10% CI width
  - Per-session confidence calibration via temperature scaling
- Integrated into `backtest.py` and `live.py`:
  - Session-specific model weights
  - Outcome recording for learning
  - Session adapter summary in output
- Added config parameters in `config.py`:
  - `session_adapter.enabled`, `min_samples`, `half_life_hours`
  - `prior_strength`, `min_weight`, `max_weight`
- **Result**: SessionAdapter infrastructure complete, ready for learning over time

### Sprint 18: ML Hardening (2025-12-30)
- Enhanced `DirectionalClassifier` in `titan/core/models/ml.py`:
  - `walk_forward_validate()` - Time-split cross-validation (no data leakage)
  - `fit_calibrator()` - Isotonic regression calibration for probabilities
  - `analyze_features()` - Feature importance analysis
  - `select_features()` - Automatic feature selection (min 1% importance)
  - `get_training_stats()` - Returns CV accuracy, feature importance, etc.
- Updated `train()` to run validation and calibration automatically
- Updated `predict()` to apply isotonic calibration
- Updated `save()`/`load()` to persist calibrator and feature importance
- **Top Features**: return_lag_3, volume_change_pct, body_ratio, volatility_z
- **Weak Features**: log_return_1, return_lag_1, rsi_oversold, rsi_overbought, candle_direction
- **Result**: ML Classifier improved from 37.1% to 41.9% accuracy

---

## Key Files

### Core System
- `titan/core/backtest.py` - Main backtest engine with analysis integration
- `titan/core/ensemble.py` - Model ensemble with regime adaptation
- `titan/core/features/stream.py` - Feature calculation (30+ features including ML features)
- `titan/core/models/heuristic.py` - TrendVIC, Oscillator, VolumeMetrix models
- `titan/core/models/ml.py` - LightGBM DirectionalClassifier (Sprint 14)

### Regime & Adaptation (Sprint 3)
- `titan/core/regime.py` - RegimeDetector class
- `titan/core/monitor.py` - PerformanceMonitor class
- `titan/core/weights.py` - WeightManager + AdaptiveWeightManager

### Strategies & Adapters (Sprints 4-20)
- `titan/core/calibration.py` - ConfidenceCompressor, OnlineCalibrator
- `titan/core/strategies/volatile.py` - VolatileClassifier, VolatileStrategy
- `titan/core/strategies/trending.py` - TrendingStrategy
- `titan/core/detectors/exhaustion.py` - TrendExhaustionDetector
- `titan/core/adapters/temporal.py` - TemporalAdjuster
- `titan/core/adapters/pattern.py` - PatternAdjuster (Sprint 13)
- `titan/core/adapters/session.py` - SessionAdapter, SessionMemory (Sprint 17)
- `titan/core/patterns.py` - PatternStore, PatternExperience (Sprint 13)
- `titan/core/online.py` - OnlineLearner, MultiScaleEMA, OnlineAdapter (Sprint 20)

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
- `docs/SYSTEM_ANALYSIS.md` - Full system analysis (Sprint 15): features, calibration, recommendations

### Utilities
- `cleanup.py` - Cleanup script for runs/, databases, cache (--force, --dry-run, --interactive)

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

## Regime Weights (Sprint 14)

```python
DEFAULT_REGIME_WEIGHTS = {
    "trending_up":   {"TRENDVIC": 0.40, "OSCILLATOR": 0.15, "VOLUMEMETRIX": 0.20, "ML_CLASSIFIER": 0.25},
    "trending_down": {"TRENDVIC": 0.40, "OSCILLATOR": 0.15, "VOLUMEMETRIX": 0.20, "ML_CLASSIFIER": 0.25},
    "ranging":       {"TRENDVIC": 0.15, "OSCILLATOR": 0.40, "VOLUMEMETRIX": 0.20, "ML_CLASSIFIER": 0.25},
    "volatile":      {"TRENDVIC": 0.20, "OSCILLATOR": 0.20, "VOLUMEMETRIX": 0.35, "ML_CLASSIFIER": 0.25},
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

## Chronos Project Analysis (2025-12-30)

Analyzed C:\Projects\chronos - a production crypto prediction system achieving **66-74% accuracy** on 5-10 minute horizons.

### Key Chronos Achievements
| Metric | Chronos | Titan | Gap |
|--------|---------|-------|-----|
| **5min accuracy** | 66-69% | ~50% | **+16-19%** |
| **10min accuracy** | 72-74% | N/A | — |
| **60min accuracy** | 52% | 48.6% | +3.4% |
| **Online Learning** | ✅ Dual models | ❌ None | Critical |
| **ML Models** | LightGBM | Heuristics | Critical |

### Chronos Key Innovations

1. **Dual Model Architecture** - Two specialized models (UP detector + DOWN detector) with mutual communication
2. **Multi-Scale EMA Memory** - Short/Medium/Long-term pattern memory prevents catastrophic forgetting
3. **Scale-Invariant Features** - ALL features are relative (%, ratios, z-scores), never absolute prices
4. **Online Learning** - SGD + RMSProp updates every 60 seconds based on outcomes
5. **Confidence Filtering** - Only trade when confidence ≥65-70% → achieves 75%+ accuracy on filtered signals

### 75% Accuracy IS Achievable (Proven by Chronos)
```
5min horizon + confidence ≥ 0.70 → 76.9% accuracy (350 samples)
10min horizon + confidence ≥ 0.65 → 75.2% accuracy (444 samples)
```

## Strategic Plan (4 Directions)

After Sprint 16, the system has plateaued at ~51-52% accuracy. Four strategic directions identified:

### Direction 1: Session Adapter (PRIORITY)
**Per-session (ASIA/EUROPE/US) model configuration**

Rationale: 4% accuracy gap between sessions (ASIA 50.5% vs EUROPE 46.5%) justifies adaptation.

Components:
- **SessionMemory** (SQLite) - stores per-session stats
- **Weight Adaptation** - EMA updates with shrinkage to global
- **Parameter Selection** - Thompson Sampling for discrete params
- **Temperature Scaling** - per-session confidence calibration
- **Trust Blocks** - min 50 samples before updates, max 10% CI width

Key formulas:
```python
# Weight update with shrinkage
effective_acc = (global_acc * k + session_acc * n) / (k + n)
new_weight = α * raw_weight + (1 - α) * current_weight

# Thompson Sampling for params
alpha = stats.correct + 1
beta = stats.total - stats.correct + 1
sample = np.random.beta(alpha, beta)

# Decay (168h half-life)
decay = 0.5 ** (hours_since / 168)
```

### Direction 2: ML Hardening
- Time-split cross-validation (no data leakage)
- Proper calibration (isotonic regression)
- Stricter feature selection

### Direction 3: Data/Labeling
- ATR-relative thresholds (not fixed %)
- Noise filtering (ignore tiny moves)
- Spread/commission modeling

### Direction 4: Pattern Memory as Input
- Use pattern stats as direct ML features
- Not just confidence adjusters

---

## Reinforcement Learning Analysis

### Problem Formulation

**Our task:** Predict price direction (UP/DOWN) for next candle.

```
State (s):    features + session + regime
Action (a):   model weights / parameter selection
Reward (r):   +1 if prediction correct, 0 otherwise
```

**Critical question:** Is this single-step or sequential decision problem?

### RL Types Comparison

| RL Type | Description | Fits? | Why |
|---------|-------------|-------|-----|
| **Multi-Armed Bandits** | Choose from K actions, no state | Partial | Ignores context |
| **Contextual Bandits** | Action selection with context | **Perfect** | Context = features, single-step |
| **Q-Learning / DQN** | Tabular/neural Q(s,a) | Overkill | No sequential dependencies |
| **Policy Gradient / PPO** | Direct policy optimization | Overkill | Needs millions of samples |
| **Actor-Critic / A3C** | Value + policy combination | Overkill | Complexity without benefit |

### Why Contextual Bandits is Perfect

**Formal problem definition:**
```
At each step t:
1. Observe context x_t = (features, session, regime)
2. Select action a_t = (model weights, parameters)
3. Receive reward r_t = I(prediction correct)
4. NO state transition (s_{t+1} independent of a_t)
```

**Critical insight:** Our action (weight/param selection) **does NOT affect** next market state.

```
Full RL:      s_{t+1} = f(s_t, a_t)    # Action changes state
Our problem:  s_{t+1} = f(market)      # State determined by market
```

### Thompson Sampling vs Alternatives

| Algorithm | Description | Pros | Cons |
|-----------|-------------|------|------|
| **ε-greedy** | Random with prob ε | Simple | Requires ε tuning |
| **UCB** | Upper Confidence Bound | Deterministic | Hard for context |
| **Thompson Sampling** | Sample from posterior | Bayesian, adaptive | Needs prior |
| **LinUCB** | Linear model + UCB | Good for linear | Limited to linear |

**Why Thompson Sampling wins:**
1. Natural exploration/exploitation balance — no ε tuning
2. Bayesian uncertainty — more exploration when uncertain
3. Simple implementation — Beta distribution for binary outcomes
4. Works well with small data — prior helps at start

### Mathematical Foundation

```python
# Thompson Sampling for binary outcome:
# Maintain Beta(α, β) posterior for each parameter value

# After each outcome:
if hit:
    α += 1  # Success
else:
    β += 1  # Failure

# When selecting parameter:
for each value v:
    sample θ_v ~ Beta(α_v, β_v)
select v* = argmax(θ_v)

# Properties:
# Beta(1, 1) = uniform prior (know nothing)
# E[θ] = α / (α + β) = expected success rate
# Variance decreases with more data
```

### Why Full RL (DQN/PPO) is Unnecessary

**Argument 1: No sequential structure**
```
Full RL needed: Current action affects future states (games, robotics)
Our problem:    Each prediction is independent
                Weight choice for candle t doesn't affect candle t+1
```

**Argument 2: Data efficiency**

| Method | Samples to converge |
|--------|---------------------|
| Thompson Sampling | ~100-500 per arm |
| DQN | ~100,000 - 1,000,000 |
| PPO | ~1,000,000 - 10,000,000 |

We have ~10,000 candles/week. Thompson Sampling converges in days, DQN/PPO in months.

**Argument 3: Complexity vs Benefit**
```
Thompson Sampling: 50 lines of code, works out-of-box
DQN:               1000+ lines, requires tuning (lr, ε, replay, target network)
PPO:               2000+ lines, even more tuning
```

### When Would Full RL Be Needed?

**Scenario A: Portfolio Management**
```
Managing position (size, leverage):
- Action: [buy 10%, hold, sell 5%, ...]
- State: current position + features
- s_{t+1} depends on a_t (position changed)
→ Full RL makes sense
```

**Scenario B: Market Making**
```
Setting bid/ask prices:
- Action: [bid_price, ask_price, size]
- State: order book + inventory + features
- s_{t+1} strongly depends on a_t
→ Full RL required
```

**Our task doesn't fit either scenario.**

### Conclusion

```
┌─────────────────────────────────────────────────────────────┐
│  RECOMMENDATION: Thompson Sampling (Contextual Bandits)     │
├─────────────────────────────────────────────────────────────┤
│  ✅ Perfect fit for single-step prediction                  │
│  ✅ Simple implementation (~50 lines)                       │
│  ✅ No hyperparameter tuning needed                         │
│  ✅ Data-efficient (converges in 100-500 samples per arm)   │
│  ✅ Natural exploration/exploitation balance                │
├─────────────────────────────────────────────────────────────┤
│  ❌ Full RL (DQN/PPO) is overkill:                          │
│     - No sequential dependencies in our problem             │
│     - Requires 100x more data                               │
│     - 20x more complex implementation                       │
│     - No proven benefit for our task                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Next Steps

### Sprint 17: Session Adapter (NEXT)
Implement per-session adaptation:
1. Create `SessionMemory` SQLite schema
2. Implement `SessionAdapter` class with:
   - Weight updates (every 50 predictions)
   - Parameter selection (every 500 predictions)
   - Calibration updates (every 100 predictions)
3. Integrate into backtest.py and live.py
4. Add decay mechanism (168h half-life)
5. Add trust blocks (min 50 samples)

### Sprint 18: ML Hardening
- Time-split CV for training
- Calibration improvements
- Feature selection

### Sprint 19: 5-Minute Timeframe
- Switch from 1m to 5m candles
- Less noise, stronger signals
- Target: 55-60% accuracy

### Sprint 20: Online Learning (FINAL)
- Implement SGD + RMSProp weight updates
- Multi-scale EMA memory (short/medium/long)
- **Execute LAST** when system is debugged
- Target: Market adaptation

### Future Ideas
- Dual model architecture (UP/DOWN detectors)
- External data: Funding rate, Open Interest, Order book
- Multi-timeframe analysis (5m + 15m + 1h)
- Two-stage calibration (model-level + ensemble-level)

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

## Current Sprint: Sprint 17 - Session Adapter (2025-12-30)

### Completed Sprints ✅
- **Sprint 14**: LightGBM Classifier - 31 scale-invariant features
- **Sprint 15**: Confidence Filtering - 65.21% filtered accuracy
- **Sprint 16**: New features (bb_position, vol_imbalance_20, adx, mfi)

### Key Finding
System plateaued at ~51-52% accuracy. Session analysis shows 4% gap:
- ASIA: 50.5%, EUROPE: 46.5%, US: 49.8%
- Best hour: 6:00 UTC (60.8%), Worst: 4:00 UTC (42.5%)

### Sprint 17 Plan (Session Adapter)
| Phase | Task | Status |
|-------|------|--------|
| 1 | Create SessionMemory SQLite schema | Pending |
| 2 | Implement SessionAdapter class | Pending |
| 3 | Add Thompson Sampling for params | Pending |
| 4 | Add decay + trust blocks | Pending |
| 5 | Integrate into backtest/live | Pending |
| 6 | Run 7-day backtest | Pending |

### Target Metrics
- Per-session accuracy improvement: +2-3%
- Overall accuracy: ≥53%
- ECE: <3%

---

## Last Session Context (2025-12-30)

### Completed
- Sprint 14: LightGBM Classifier (31 features)
- Sprint 15: Confidence Filtering
- Sprint 16: New features (bb_position, vol_imbalance_20, adx, mfi)
- Strategic planning (4 directions identified)
- RL analysis (Thompson Sampling = Contextual Bandits)

### Latest Test Results (48h, 2025-12-30)
```
Ensemble: 49.2% (1417/2879)
ECE: 0.8%
Sharpe: -22.4
Direction Balance: 0.94 (balanced)
p-value: 0.3 (not significant on 48h)

Per-Model:
- TRENDVIC: 48.8%, Balance 1.00
- OSCILLATOR: 43.6%, Balance 0.98
- VOLUMEMETRIX: 45.2%, Balance 0.97
- ML_CLASSIFIER: 35.0%, Balance 0.65

Session Performance:
- ASIA: 50.5% (960)
- EUROPE: 46.5% (719) - worst
- US: 49.8% (1200)
```

### Strategic Direction Selected
**Session Adapter** - per-session (ASIA/EUROPE/US) model configuration with:
- Thompson Sampling for parameter selection
- EMA weight updates with shrinkage
- Temperature scaling for calibration
- Decay mechanism (168h half-life)
- Trust blocks (min 50 samples)

### RL Analysis Conclusion
Thompson Sampling in SessionAdapter = Contextual Bandits (a form of RL).
Full RL (DQN/PPO) is overkill for single-step prediction problem.
