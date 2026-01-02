# Chronos vs Titan: Полный сравнительный анализ

## 1. Сравнение результатов

| Метрика | Chronos | Titan | Gap |
|---------|---------|-------|-----|
| **5min direction accuracy** | 66-69% | N/A | — |
| **10min direction accuracy** | 72-74% | N/A | — |
| **1min accuracy** | ~50% (шум) | 52.17% | +2% |
| **Filtered accuracy (conf≥55%)** | ~75% | **65.21%** | -10% |
| **High-conf coverage** | 30-40% | **9.07%** | -21% |
| **Online learning** | ✅ SGD + Adagrad | ❌ Нет | Critical |

---

## 2. Сравнение Features

### 2.1 Chronos Features (Adaptive Feature Engineering)

**Core Features (всегда):**
```python
# Microstructure / Volatility proxies
- typical_price           # (high + low + close) / 3
- vwap_dist_pct          # Расстояние от VWAP в %
- spread_pct             # Спред в %
- vol_imbalance_{20,60,120}  # Volume imbalance по окнам
- realized_vol_{20,60,120}   # Realized volatility (log returns)

# Price features (все относительные!)
- return_lag_{20,50}     # Returns от N баров назад
- high_return_lag_{20,50}
- low_return_lag_{20,50}
- return_{20,50}         # Rolling returns
- volatility_{20,50,100} # Rolling volatility
- sma_dist_{20,50,100,200}  # Расстояние от SMA в %
- ema_dist_{20,50,100,200}  # Расстояние от EMA в %
- rsi_14                 # RSI
- macd_pct               # MACD в % от цены
- macd_signal_pct
- macd_hist_pct

# Volume features
- volume_ratio           # Текущий / средний объём
- volume_change_{20,50}  # Изменение объёма в %

# Volatility features (для high/extreme vol)
- parkinson_vol_{20,50}  # Parkinson volatility
- atr_pct_{20,50}        # ATR в % от цены
- breakout_up/down       # Breakout сигналы
- dist_to_high_{100}     # Расстояние до high/low
- dist_to_low_{100}
- range_pct_{100}        # Range в %

# Trend features (для bull/bear)
- green_streak_{20}      # Серия зелёных свечей
- red_streak_{20}
- higher_high            # Higher High / Lower Low
- lower_low
- hh_count_{20}
- ll_count_{20}
- sma_slope_{20}         # Наклон SMA

# Bollinger Bands (для normalized target)
- bb_dist_upper
- bb_dist_lower
- bb_width_pct
- bb_position            # Позиция внутри BB

# Log returns (для log_returns target)
- log_return_{20,50,100,200}
- cumulative_log_return_{20,50}
```

**Всего: ~60+ features** (адаптивно выбираются по режиму)

### 2.2 Titan Features (FeatureStream)

```python
# Base OHLCV
- close, open, high, low
- return_1, log_return_1
- ma_fast, ma_slow, ma_delta
- volatility, volatility_z
- volume, volume_z
- rsi, rsi_prev, rsi_momentum

# Sprint 10 features
- price_momentum_3
- volume_trend
- body_ratio
- upper_wick_ratio, lower_wick_ratio
- candle_direction

# Sprint 14 ML features
- return_lag_1..5        # 5 лагов
- atr_pct
- high_low_range_pct
- ma_delta_pct
- ema_10_spread_pct, ema_20_spread_pct
- return_5, return_10
- rsi_oversold, rsi_overbought, rsi_neutral
- volume_change_pct
- body_pct
- vol_ratio
```

**Всего: 37 features** (фиксированный набор)

### 2.3 Отсутствующие в Titan features (критические)

| Feature | Chronos | Titan | Важность |
|---------|---------|-------|----------|
| **vol_imbalance** | ✅ Объёмный дисбаланс UP/DOWN | ❌ | HIGH |
| **realized_vol** | ✅ Log-return volatility | ❌ | HIGH |
| **parkinson_vol** | ✅ High-Low volatility | ❌ | MEDIUM |
| **MACD (%)** | ✅ В % от цены | ❌ | MEDIUM |
| **Bollinger Bands** | ✅ 4 индикатора | ❌ | HIGH |
| **Breakout signals** | ✅ Бинарные | ❌ | MEDIUM |
| **Streak counting** | ✅ Green/red streaks | ❌ | LOW |
| **SMA slope** | ✅ Наклон тренда | ❌ | MEDIUM |
| **Адаптивные окна** | ✅ По волатильности | ❌ | HIGH |

---

## 3. Сравнение калибровки

### 3.1 Chronos Calibration

```python
# OnlineCalibrator
- Affine correction: delta_true ≈ a + b * delta_pred
- Per-horizon params (5, 15, 30, 60 min)
- Per-volatility params (low, normal, high, extreme)
- Exponential recency weights (decay=0.995)
- Adaptive clipping based on MAE
- Shrinkage factors:
  - MAE-based: shrink if MAE > target
  - Confidence-based: 0.5 + 0.5 * confidence
  - Volatility-based: {low: 0.7, normal: 0.85, high: 1.0, extreme: 1.1}
```

### 3.2 Titan Calibration

```python
# ConfidenceCompressor
- Linear compression: [0.5, 1.0] → [0.5, max_conf]
- max_confidence = 0.70 (фиксированный!)
- Regime multipliers: {trending_up: 1.0, volatile: 0.75}

# OnlineCalibrator
- Bin-based calibration (6 бинов)
- Simple accuracy tracking per bin
- Blend factor = 0.70
- Decay factor = 1.0 (нет экспоненциального затухания)
```

### 3.3 Ключевые различия

| Аспект | Chronos | Titan |
|--------|---------|-------|
| **Max confidence** | Адаптивный по режиму | Фиксированный 0.70 |
| **Compression** | Shrinkage + affine | Линейная |
| **Per-regime** | ✅ Отдельные params | ❌ Только multipliers |
| **Recency weighting** | ✅ Exponential decay | ❌ |
| **MAE tracking** | ✅ Adaptive clipping | ❌ |
| **Confidence bins** | ✅ 4 бина (low/medium/high/very_high) | 6 бинов |

---

## 4. Online Learning

### 4.1 Chronos OnlineLearner

```python
class OnlineLearner:
    """
    Architecture:
    1. SGDRegressor - learns residual correction
    2. ExponentialMovingStats - tracks error trends
    3. Direction calibration - Platt scaling style

    Updates:
    - Adagrad adaptive learning rates
    - Per-regime + per-horizon models
    - Global fallback model
    - 500 sample sliding window
    """

    def partial_fit(self, features, target):
        # Gradient: -2 * error * x + L2 regularization
        grad = -2 * error * x + 2 * self.l2_reg * self.weights

        # Adagrad: G = G + g²
        self.grad_accum += grad ** 2

        # Update: w -= lr * g / sqrt(G + eps)
        lr = self.base_lr / (np.sqrt(self.grad_accum) + 1e-8)
        self.weights -= lr * grad
```

### 4.2 Titan

❌ **Нет online learning!**

Titan использует только:
- LightGBM (batch training при накоплении 500+ samples)
- Retrain каждые 1000 samples
- Нет SGD/Adagrad updates

---

## 5. Рекомендации для Titan

### 5.1 Критические (Sprint 16)

#### A. Добавить недостающие features

```python
# В FeatureStream добавить:

# 1. Volume Imbalance (CRITICAL)
vol_up = volume * (ret_1 > 0)
vol_down = volume * (ret_1 < 0)
for w in [20, 60]:
    vol_imbalance_w = (vol_up.rolling(w).sum() - vol_down.rolling(w).sum()) /
                      (vol_up.rolling(w).sum() + vol_down.rolling(w).sum())

# 2. Realized Volatility (log returns)
for w in [20, 60]:
    realized_vol_w = log_return.rolling(w).std() * sqrt(60)

# 3. Bollinger Bands
sma_20 = close.rolling(20).mean()
std_20 = close.rolling(20).std()
bb_upper = sma_20 + 2 * std_20
bb_lower = sma_20 - 2 * std_20
bb_width_pct = (bb_upper - bb_lower) / sma_20
bb_position = (close - bb_lower) / (bb_upper - bb_lower)

# 4. MACD (percentage-based)
ema_12 = close.ewm(span=12).mean()
ema_26 = close.ewm(span=26).mean()
macd_pct = (ema_12 - ema_26) / close * 100
```

#### B. Адаптивная калибровка

```python
# Заменить фиксированный max_confidence на regime-based
REGIME_MAX_CONFIDENCE = {
    "trending_up": 0.75,    # Лучший режим
    "ranging": 0.70,        # Хороший
    "trending_down": 0.65,  # Проблемный
    "volatile": 0.60,       # Худший
}

# Sigmoid compression вместо линейной
def sigmoid_compress(strength, max_conf):
    x = strength * 4 - 2  # [0, 1] → [-2, 2]
    sigmoid = 1 / (1 + exp(-x))
    return 0.5 + (max_conf - 0.5) * sigmoid
```

### 5.2 Важные (Sprint 17)

#### C. Переход на 5-минутный таймфрейм

Chronos показывает:
- 1min accuracy ~50% (шум)
- 5min accuracy 66-69%
- 10min accuracy 72-74%

**Titan должен перейти на 5m!**

#### D. Добавить адаптивные окна

```python
# Окна по волатильности
WINDOW_CONFIGS = {
    'low': {'short': [5, 10], 'medium': [20, 30], 'long': [50, 100]},
    'normal': {'short': [10, 20], 'medium': [50, 100], 'long': [100, 200]},
    'high': {'short': [20, 50], 'medium': [100, 200], 'long': [200, 500]},
}
```

### 5.3 Долгосрочные (Sprint 18)

#### E. Online Learning

```python
class OnlineLearner:
    """Добавить SGD с Adagrad для online updates."""

    def __init__(self, n_features, lr=0.01):
        self.weights = np.zeros(n_features + 1)
        self.grad_accum = np.ones(n_features + 1) * 1e-8

    def partial_fit(self, features, target):
        x = np.concatenate([[1.0], features])  # bias
        pred = np.dot(self.weights, x)
        error = target - pred

        grad = -2 * error * x
        self.grad_accum += grad ** 2
        lr = self.base_lr / (np.sqrt(self.grad_accum) + 1e-8)
        self.weights -= lr * grad
```

---

## 6. План реализации

### Sprint 16: Adaptive Calibration + New Features
```
Week 1:
- [ ] Добавить vol_imbalance_{20,60}
- [ ] Добавить realized_vol_{20,60}
- [ ] Добавить Bollinger Bands (4 индикатора)
- [ ] Добавить macd_pct
- [ ] Regime-based max_confidence
- [ ] Sigmoid compression

Week 2:
- [ ] Тестирование на 7 днях
- [ ] Анализ confidence distribution
- [ ] Целевой coverage ≥20%
```

### Sprint 17: 5-Minute Timeframe
```
- [ ] Изменить interval с 1m на 5m
- [ ] Адаптировать window sizes
- [ ] Перетестировать все модели
- [ ] Целевая accuracy ≥55%
```

### Sprint 18: Online Learning
```
- [ ] Создать SGDRegressor
- [ ] Создать OnlineLearner
- [ ] Интегрировать в Ensemble
- [ ] Тестирование адаптации
```

---

## 7. Ожидаемые результаты

| Метрика | Текущее | После Sprint 16 | После Sprint 17 | Ultimate |
|---------|---------|-----------------|-----------------|----------|
| Ensemble Accuracy | 52.17% | 52-54% | 55-60% | 60%+ |
| Filtered Accuracy | 65.21% | 66-68% | 70-75% | **75%+** |
| Coverage | 9.07% | 20-25% | 30-40% | 40%+ |
| Features | 37 | 45+ | 50+ | 60+ |

---

## 8. Заключение

**Chronos достигает 75% accuracy благодаря:**
1. ✅ 60+ адаптивных features (vs 37 у Titan)
2. ✅ Online learning с Adagrad (vs batch training)
3. ✅ 5-10min timeframe (vs 1min)
4. ✅ Adaptive calibration per regime
5. ✅ Confidence filtering (≥70% → 75% acc)

**Titan может достичь аналогичных результатов при:**
1. Добавлении критических features (vol_imbalance, BB, MACD)
2. Переходе на 5min timeframe
3. Внедрении regime-based adaptive calibration
4. Реализации online learning (Sprint 18)
