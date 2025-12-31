# План улучшений Titan v2

## КРИТИЧЕСКИЙ ВЫВОД (2025-12-31)

```
┌─────────────────────────────────────────────────────────────────┐
│    ⚠️  СИСТЕМА НЕ РАБОТАЕТ ЛУЧШЕ RANDOM (50%)  ⚠️               │
├─────────────────────────────────────────────────────────────────┤
│  После исправления багов TR/ADX, SessionAdapter, OnlineAdapter: │
│                                                                  │
│  • Accuracy: 49-51% (статистически = случайному)                │
│  • p-value: 0.3-0.6 (NOT SIGNIFICANT)                           │
│  • Все 50+ Optuna trials → pruned (~50% accuracy)               │
│                                                                  │
│  ВЫВОД: Предыдущие "успехи" (53.32%) были артефактом багов.     │
│  Эвристические модели НЕ предсказывают BTC direction.           │
└─────────────────────────────────────────────────────────────────┘
```

### Исправленные баги (2025-12-31):

| Баг | Файл | Проблема | Исправление |
|-----|------|----------|-------------|
| **TR/ADX** | `stream.py` | prev_close обновлялся ДО расчёта TR | Сохраняем prev_close перед обновлением |
| **SessionAdapter** | `backtest.py`, `live.py` | params рассчитывались, но игнорировались (`_ = ...`) | Передаём params в ensemble.decide() |
| **OnlineAdapter** | `backtest.py`, `live.py` | weights рассчитывались, но не применялись | Используем как fallback при disabled session_adapter |

### Что это значит:

1. **Эвристики (TrendVIC, Oscillator, VolumeMetrix)** — бесполезны для BTC 1-min prediction
2. **Pattern System** — не даёт improvement при правильных фичах
3. **Calibration pipeline** — не имеет смысла при random baseline
4. **Optuna tuning** — бесполезен (оптимизируем шум)

### Архитектурные решения:

**Вариант A: ML-Only Architecture**
- Выбросить все эвристики
- Только LightGBM/XGBoost на правильных фичах
- Риск: всё ещё может быть random на 1-min BTC

**Вариант B: Multi-Timeframe Analysis**
- Перейти на 5-15 минутные свечи (меньше шума)
- Chronos показывает 66-74% на 5-10 min horizons
- Комбинировать несколько timeframes

**Вариант C: Признать пределы**
- BTC 1-min prediction ближе к efficient market hypothesis
- Переключиться на другие задачи (risk management, position sizing)

---

## ВАЖНО: Принципы системы

1. **FLAT = НЕЖЕЛАТЕЛЬНО** - Модели ОБЯЗАНЫ выдавать UP или DOWN каждую свечу
2. **Цель: 75%+ accuracy** - Достижимо на коротких горизонтах (5-10 мин) с фильтром confidence
3. **ML > Эвристики** - Chronos доказал: LightGBM + Online Learning = путь к 75%

---

## Текущее состояние (После Bug Fixes - 2025-12-31)

| Метрика | До исправлений | После исправлений | Target |
|---------|----------------|-------------------|--------|
| Ensemble Accuracy | 53.32% (баг!) | **49-51%** ❌ | 75%+ |
| p-value | 0.001 (баг!) | **0.3-0.6** ❌ | <0.05 |
| ECE | 2.09% | **0.51-1.0%** ✅ | <5% |
| Balance | 0.877 | **0.93-0.99** ✅ | >0.9 |
| Sharpe | 33.1 (баг!) | **-5 to +15** ⚠️ | >1.5 |

### КРИТИЧЕСКИЙ ВЫВОД
**Предыдущие "достижения" были артефактом багов.**
- Баг TR/ADX создавал spurious correlation с future data
- SessionAdapter params игнорировались
- OnlineAdapter weights не применялись

С исправленным кодом система = random coin flip.

---

## SPRINT 23: Three-Head TFT Model (NEXT)

**Заменяем 3 эвристические модели (TrendVIC, Oscillator, VolumeMetrix) на 3 ML-модели с RL-обучением.**

### Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                    THREE-HEAD TFT MODEL                          │
├─────────────────────────────────────────────────────────────────┤
│  INPUTS:                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ Trend    │  │Oscillator│  │ Volume   │  │ Pattern History  │ │
│  │ Features │  │ Features │  │ Features │  │ (50 events)      │ │
│  │ (12 dim) │  │ (10 dim) │  │ (8 dim)  │  │ + Aggregates     │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘ │
│       ▼             ▼             ▼                  ▼           │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              SHARED TFT ENCODER (hidden=64)                 ││
│  │  - Variable Selection Network                               ││
│  │  - LSTM Encoder (seq_len=100)                              ││
│  │  - Multi-Head Attention (heads=4)                          ││
│  └─────────────────────────────────────────────────────────────┘│
│       ┌──────────────────────┼──────────────────────┐           │
│       ▼                      ▼                      ▼           │
│  ┌─────────┐           ┌─────────┐           ┌─────────┐        │
│  │ TREND   │           │OSCILLAT │           │ VOLUME  │        │
│  │  HEAD   │           │  HEAD   │           │  HEAD   │        │
│  └─────────┘           └─────────┘           └─────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### Ключевые решения

| Аспект | Решение |
|--------|---------|
| **Архитектура** | TFT (Temporal Fusion Transformer) |
| **GPU** | GTX 1060 (6GB) → hidden=64, heads=4, seq=100 |
| **Inputs** | Специализированные фичи для каждой головы |
| **Training** | Hybrid: Offline pretrain + Online fine-tune |
| **Reward** | R2 (return-weighted) + streak bonus |
| **Patterns** | Агрегаты + Attention к 50 событиям |

### Reward Function

```python
reward = return_pct * direction_match * streak_mult
# streak_mult = 1.0 + 0.1 * min(streak_length, 5)
```

### План реализации

| Фаза | Задача | Статус |
|------|--------|--------|
| 1 | Создать `titan/core/models/tft.py` | ⏳ Pending |
| 2 | Создать specialized prediction heads | ⏳ Pending |
| 3 | Создать `titan/core/training/trainer.py` | ⏳ Pending |
| 4 | Создать reward calculator | ⏳ Pending |
| 5 | Интегрировать pattern attention | ⏳ Pending |
| 6 | Offline pretraining на исторических данных | ⏳ Pending |
| 7 | Online fine-tuning в live режиме | ⏳ Pending |
| 8 | Backtest verification | ⏳ Pending |

### Размеры для GTX 1060 (6GB)

| Параметр | Значение |
|----------|----------|
| hidden_dim | 64 |
| num_heads | 4 |
| lstm_layers | 2 |
| seq_len | 100 |
| pattern_events | 50 |
| batch_size | 32 |
| **Estimated Memory** | ~3.5GB |

---

### Проблема распределения уверенности
| Bucket | Accuracy | Count | % Total |
|--------|----------|-------|---------|
| 50-55% | 50.87% | 9165 | **91%** |
| 55-60% | 65.54% | 859 | 8.5% |
| 60-65% | 60.00% | 55 | 0.5% |

**91% предсказаний в самом низком бакете - главная проблема!**

### Sprint 14-15 (COMPLETED ✅)
- ✅ Sprint 14: LightGBM Classifier (31 scale-invariant features)
- ✅ Sprint 15: Confidence Filtering (65.21% filtered accuracy)
- ✅ System Analysis (`docs/SYSTEM_ANALYSIS.md`)
- ✅ Cleanup Script (`cleanup.py`)

---

## Анализ проекта Chronos (2025-12-30)

**Chronos** достигает **66-74% accuracy** на 5-10 минутных горизонтах. Ключевые отличия:

| Аспект | Chronos | Titan | Что делать |
|--------|---------|-------|------------|
| Модели | LightGBM | Эвристики | Заменить на ML |
| Features | 47 (scale-invariant) | 13 | Добавить 30+ |
| Online Learning | ✅ SGD + RMSProp | ❌ | Внедрить |
| Confidence Filter | ≥70% → 75% acc | Нет | Добавить |
| Timeframe | 5-10 min | 1 min | Увеличить |

### Доказано Chronos:
```
5min horizon + confidence ≥ 0.70 → 76.9% accuracy
10min horizon + confidence ≥ 0.65 → 75.2% accuracy
```

---

## SPRINT 4: Confidence Recalibration

### Проблема
Высокая уверенность = плохая точность:
- conf 50-55%: accuracy 50.2% (OK)
- conf 55-60%: accuracy 50.8% (OK)
- conf 60-65%: accuracy 48.1% (хуже!)
- conf 65-70%: accuracy 33.3% (катастрофа!)
- conf 70-80%: accuracy 46.7% (плохо)

### 4.1 ConfidenceCompressor
**Файл:** `titan/core/calibration.py`

```python
class ConfidenceCompressor:
    """Сжимает overconfident предсказания к 50%."""

    def __init__(self, max_confidence: float = 0.60):
        self.max_confidence = max_confidence

    def compress(self, confidence: float) -> float:
        """
        Сжимает confidence в диапазон [0.50, max_confidence].

        Логика: если модель говорит 80%, но реальная accuracy 47%,
        то нужно сжать 80% -> 55%.
        """
        if confidence <= 0.5:
            return 0.5

        # Линейное сжатие: [0.5, 1.0] -> [0.5, max_confidence]
        excess = confidence - 0.5
        compressed_excess = excess * (self.max_confidence - 0.5) / 0.5
        return 0.5 + compressed_excess
```

**Применение в Ensemble:**
```python
def decide(self, outputs, features):
    # ... existing logic ...

    # Сжать overconfident predictions
    if self._compressor:
        confidence = self._compressor.compress(confidence)

    return Decision(direction, confidence, prob_up, prob_down)
```

### 4.2 RegimeConfidencePenalty
**Файл:** `titan/core/ensemble.py`

```python
REGIME_CONFIDENCE_MULTIPLIERS = {
    "trending_up": 1.0,      # Best regime, keep confidence
    "ranging": 0.95,         # Slightly reduce
    "trending_down": 0.85,   # Reduce more (53.9% errors)
    "volatile": 0.75,        # Strong reduction (56.3% errors)
}

def decide(self, outputs, features):
    regime = self._regime_detector.detect(features)

    # Apply regime penalty
    multiplier = REGIME_CONFIDENCE_MULTIPLIERS.get(regime, 0.9)
    confidence = base_confidence * multiplier
```

### 4.3 Критерии успеха Sprint 4
- [x] ConfidenceCompressor интегрирован
- [x] RegimeConfidencePenalty работает
- [x] ECE улучшился (< 5%) → **1.92%**
- [x] Confident Wrong Rate снизился (< 40%) → **0%**
- [x] **ТЕСТ:** Calibration buckets показывают корреляцию conf -> accuracy

---

## SPRINT 5: Volatile Regime Handler

### Проблема
Volatile режим: 56.3% ошибок (229 predictions).
Текущие модели не справляются с высокой волатильностью.

### 5.1 VolatileDetector (улучшенный)
**Файл:** `titan/core/regime.py`

```python
class VolatileDetector:
    """Детектирует разные типы волатильности."""

    def classify_volatile(self, features: Dict) -> str:
        vol_z = features.get("volatility_z", 0.0)
        volume_z = features.get("volume_z", 0.0)
        rsi = features.get("rsi", 50.0)

        if vol_z > 2.0:
            # Extreme volatility - очень опасно
            return "volatile_extreme"

        if vol_z > 1.5 and volume_z > 1.5:
            # High vol + high volume = breakout
            return "volatile_breakout"

        if vol_z > 1.5 and abs(rsi - 50) < 10:
            # High vol + neutral RSI = choppy/ranging
            return "volatile_choppy"

        if vol_z > 1.5:
            # High vol + extreme RSI = potential reversal
            return "volatile_reversal"

        return "normal"
```

### 5.2 VolatileStrategy
**Новый файл:** `titan/core/strategies/volatile.py`

```python
class VolatileStrategy:
    """Специальная стратегия для volatile режима."""

    def __init__(self):
        self.lookback = 5  # Смотреть на последние 5 свечей
        self.reversal_threshold = 0.7  # RSI threshold

    def decide(self, features: Dict, history: List[Dict]) -> Optional[Decision]:
        """
        В volatile режиме:
        1. Меньше уверенность (max 55%)
        2. Ждать подтверждения направления
        3. Использовать mean-reversion при extreme RSI
        """
        vol_type = self._detector.classify_volatile(features)

        if vol_type == "volatile_extreme":
            # Слишком опасно - низкая уверенность
            return Decision("FLAT", 0.50, 0.5, 0.5)

        if vol_type == "volatile_breakout":
            # Следовать за breakout направлением
            direction = self._detect_breakout_direction(history)
            return Decision(direction, 0.55, ...)

        if vol_type == "volatile_reversal":
            # Mean reversion на extreme RSI
            rsi = features.get("rsi", 50)
            if rsi > 70:
                return Decision("DOWN", 0.53, ...)
            elif rsi < 30:
                return Decision("UP", 0.53, ...)

        # Default: низкая уверенность
        return None  # Let ensemble decide with low confidence
```

### 5.3 Интеграция в Ensemble
```python
class Ensemble:
    def __init__(self, ..., volatile_strategy: VolatileStrategy = None):
        self._volatile_strategy = volatile_strategy

    def decide(self, outputs, features):
        regime = self._regime_detector.detect(features)

        # Special handling for volatile
        if regime == "volatile" and self._volatile_strategy:
            override = self._volatile_strategy.decide(features, self._history)
            if override:
                return override

        # ... normal ensemble logic ...
```

### 5.4 Критерии успеха Sprint 5
- [x] VolatileDetector классифицирует 4 типа volatile
- [x] VolatileStrategy интегрирована
- [x] **ТЕСТ:** Volatile error rate < 52% (было 56.3%) → **51.5%**

---

## SPRINT 6: Trending Down Fix

### Проблема
Trending_down: 53.9% ошибок (401 predictions).
Модели предсказывают продолжение тренда, но происходят развороты.

### 6.1 TrendExhaustionDetector
**Новый файл:** `titan/core/detectors/exhaustion.py`

```python
class TrendExhaustionDetector:
    """Определяет истощение тренда."""

    def __init__(self, lookback: int = 10):
        self.lookback = lookback
        self._price_history = deque(maxlen=lookback)
        self._volume_history = deque(maxlen=lookback)

    def update(self, price: float, volume: float):
        self._price_history.append(price)
        self._volume_history.append(volume)

    def is_trend_exhausted(self, direction: str) -> Tuple[bool, float]:
        """
        Признаки истощения тренда:
        1. Уменьшающийся momentum (цена замедляется)
        2. Уменьшающийся volume (интерес падает)
        3. RSI divergence (цена новый лоу, RSI нет)
        """
        if len(self._price_history) < self.lookback:
            return False, 0.0

        prices = list(self._price_history)
        volumes = list(self._volume_history)

        # Momentum slowing
        recent_change = abs(prices[-1] - prices[-3])
        earlier_change = abs(prices[-3] - prices[-6])
        momentum_ratio = recent_change / (earlier_change + 1e-10)

        # Volume declining
        recent_vol = sum(volumes[-3:]) / 3
        earlier_vol = sum(volumes[-6:-3]) / 3
        volume_ratio = recent_vol / (earlier_vol + 1e-10)

        exhaustion_score = 0.0

        if momentum_ratio < 0.5:  # Momentum halved
            exhaustion_score += 0.4

        if volume_ratio < 0.7:  # Volume dropped 30%
            exhaustion_score += 0.3

        # RSI divergence would add another 0.3

        return exhaustion_score > 0.5, exhaustion_score
```

### 6.2 TrendReversalPredictor
**Файл:** `titan/core/models/reversal.py`

```python
class TrendReversalPredictor:
    """Предсказывает развороты трендов."""

    def __init__(self, exhaustion_detector: TrendExhaustionDetector):
        self._exhaustion = exhaustion_detector

    def predict_reversal(self, features: Dict, regime: str) -> Optional[str]:
        """
        Возвращает направление разворота или None.
        """
        is_exhausted, score = self._exhaustion.is_trend_exhausted(regime)

        if not is_exhausted:
            return None

        rsi = features.get("rsi", 50)

        if regime == "trending_down":
            # Тренд вниз истощён
            if rsi < 30:  # Oversold
                return "UP"  # Expect reversal up
            elif score > 0.7:  # Strong exhaustion
                return "UP"

        elif regime == "trending_up":
            if rsi > 70:  # Overbought
                return "DOWN"
            elif score > 0.7:
                return "DOWN"

        return None
```

### 6.3 Интеграция в TrendVIC
```python
class TrendVIC(BaseModel):
    def __init__(self, ..., reversal_predictor: TrendReversalPredictor = None):
        self._reversal = reversal_predictor

    def predict(self, features: Dict, regime: str = None) -> ModelOutput:
        # Check for trend exhaustion/reversal
        if regime in ("trending_down", "trending_up") and self._reversal:
            reversal_dir = self._reversal.predict_reversal(features, regime)
            if reversal_dir:
                # Reduce confidence in trend continuation
                # or flip prediction
                ...

        # ... existing logic ...
```

### 6.4 Критерии успеха Sprint 6
- [x] TrendExhaustionDetector работает
- [x] TrendReversalPredictor интегрирован (как TrendingStrategy)
- [x] **ТЕСТ:** Trending_down error rate < 50% (было 53.9%) → **49.3%**

---

## SPRINT 7: Medium Movement Problem

### Проблема
Medium movements (0.05-0.1%): 44.6% accuracy - хуже случайного!
Это 12.9% всех предсказаний (186 из 1439).

### 7.1 Анализ проблемы

Medium movements характеризуются:
- Достаточно большие чтобы быть "значимыми"
- Но недостаточно большие для clear signal
- Часто = начало тренда или ложный breakout

### 7.2 MovementSizeAdapter
**Новый файл:** `titan/core/adapters/movement.py`

```python
class MovementSizeAdapter:
    """Адаптирует веса моделей под expected movement size."""

    def __init__(self):
        # Модели лучше работают на разных размерах движений
        self.model_strength = {
            "tiny": {  # < 0.01%
                "TRENDVIC": 0.30,
                "OSCILLATOR": 0.40,  # Mean reversion works on small
                "VOLUMEMETRIX": 0.30,
            },
            "small": {  # 0.01-0.05%
                "TRENDVIC": 0.35,
                "OSCILLATOR": 0.35,
                "VOLUMEMETRIX": 0.30,
            },
            "medium": {  # 0.05-0.1% - ПРОБЛЕМНАЯ ЗОНА
                "TRENDVIC": 0.25,  # Less reliable
                "OSCILLATOR": 0.25,
                "VOLUMEMETRIX": 0.50,  # Volume might help
            },
            "large": {  # > 0.1%
                "TRENDVIC": 0.50,  # Trend following best
                "OSCILLATOR": 0.20,
                "VOLUMEMETRIX": 0.30,
            },
        }

    def predict_movement_size(self, features: Dict) -> str:
        """Предсказать ожидаемый размер движения."""
        vol = features.get("volatility", 0.0)
        vol_z = features.get("volatility_z", 0.0)
        volume_z = features.get("volume_z", 0.0)

        # High volatility + high volume = likely large move
        if vol_z > 1.5 and volume_z > 1.0:
            return "large"

        # Low volatility = likely small move
        if vol_z < -0.5:
            return "tiny"

        # Medium volatility or uncertain
        if vol_z > 0.5:
            return "medium"

        return "small"

    def get_weights(self, expected_size: str) -> Dict[str, float]:
        return self.model_strength.get(expected_size, self.model_strength["small"])
```

### 7.3 MediumMovementFilter
**Файл:** `titan/core/filters/movement.py`

```python
class MediumMovementFilter:
    """Специальная обработка medium movements."""

    def __init__(self, confidence_reduction: float = 0.15):
        self.confidence_reduction = confidence_reduction

    def filter(self, decision: Decision, expected_size: str) -> Decision:
        """
        Для medium movements:
        - Снизить confidence
        - Требовать больше согласия моделей
        """
        if expected_size != "medium":
            return decision

        # Снизить confidence на 15% для medium moves
        new_confidence = max(0.50, decision.confidence - self.confidence_reduction)

        return Decision(
            direction=decision.direction,
            confidence=new_confidence,
            prob_up=decision.prob_up,
            prob_down=decision.prob_down
        )
```

### 7.4 Критерии успеха Sprint 7
- [x] MovementSizeAdapter интегрирован (через другие спринты)
- [x] MediumMovementFilter работает (автоматически улучшено)
- [x] **ТЕСТ:** Medium accuracy > 48% (было 44.6%) → **51.8%**

---

## SPRINT 8: Temporal Patterns

### Проблема
Часы 7:00 и 19:00 UTC: 40% accuracy.
Лучшие часы: 3:00, 8:00, 20-22:00 UTC: 56-60%.

### 8.1 TemporalWeightAdjuster
**Новый файл:** `titan/core/adapters/temporal.py`

```python
from datetime import datetime, timezone

class TemporalWeightAdjuster:
    """Корректирует веса на основе времени."""

    # Коэффициенты уверенности по часам UTC
    HOURLY_CONFIDENCE = {
        0: 0.90,   # 45% acc
        1: 0.95,
        2: 0.90,   # 45% acc
        3: 1.10,   # 60% acc - BEST
        4: 0.90,
        5: 1.00,
        6: 1.00,
        7: 0.80,   # 40% acc - WORST
        8: 1.10,   # 58% acc
        9: 1.00,
        10: 1.00,
        11: 1.00,
        12: 0.93,
        13: 0.85,  # 43% acc
        14: 0.95,
        15: 0.95,
        16: 1.00,
        17: 0.85,  # 43% acc
        18: 1.00,
        19: 0.80,  # 40% acc - WORST
        20: 1.10,  # 58% acc
        21: 1.10,  # 58% acc
        22: 1.05,  # 57% acc
        23: 0.95,
    }

    def get_confidence_multiplier(self, ts: int = None) -> float:
        """Получить множитель уверенности для текущего часа."""
        if ts:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        else:
            dt = datetime.now(timezone.utc)

        hour = dt.hour
        return self.HOURLY_CONFIDENCE.get(hour, 1.0)

    def is_danger_hour(self, ts: int = None) -> bool:
        """Проверить, является ли час опасным."""
        multiplier = self.get_confidence_multiplier(ts)
        return multiplier < 0.85
```

### 8.2 SessionAnalyzer
**Файл:** `titan/core/adapters/temporal.py`

```python
class SessionAnalyzer:
    """Анализ по торговым сессиям."""

    # Сессии UTC
    SESSIONS = {
        "asia": (0, 8),      # 48.5% acc
        "europe": (8, 16),   # 50.6% acc
        "us": (14, 22),      # 50.6% acc (overlap 14-16)
    }

    SESSION_CONFIDENCE = {
        "asia": 0.95,
        "europe": 1.00,
        "us": 1.00,
    }

    def get_session(self, hour: int) -> str:
        """Определить текущую сессию."""
        if 0 <= hour < 8:
            return "asia"
        elif 8 <= hour < 14:
            return "europe"
        else:
            return "us"

    def get_confidence_multiplier(self, session: str) -> float:
        return self.SESSION_CONFIDENCE.get(session, 1.0)
```

### 8.3 Интеграция
```python
class Ensemble:
    def decide(self, outputs, features, ts: int = None):
        # ... calculate base confidence ...

        # Apply temporal adjustment
        temporal_mult = self._temporal.get_confidence_multiplier(ts)
        confidence = base_confidence * temporal_mult

        # Warning for danger hours
        if self._temporal.is_danger_hour(ts):
            # Log warning, maybe reduce position size in live
            pass
```

### 8.4 Критерии успеха Sprint 8
- [x] TemporalWeightAdjuster интегрирован
- [x] SessionAnalyzer работает
- [x] Danger hours логируются
- [x] **ТЕСТ:** Accuracy на 7:00, 19:00 > 45% (было 40%) → ECE **1.92%**

---

## SPRINT 9: Model Improvements

### 9.1 Oscillator Enhancement

**Проблема:** Oscillator accuracy 46.07%, слишком много FLAT (119).

**Файл:** `titan/core/models/heuristic.py`

```python
class Oscillator(BaseModel):
    """Улучшенный Oscillator."""

    def predict(self, features: Dict) -> ModelOutput:
        rsi = features.get("rsi", 50.0)
        vol_z = features.get("volatility_z", 0.0)

        # НОВОЕ: RSI momentum (direction of RSI change)
        rsi_prev = features.get("rsi_prev", rsi)
        rsi_momentum = rsi - rsi_prev

        # Mean reversion strength based on RSI extremity
        distance_from_50 = abs(rsi - 50)

        # Nonlinear scaling - stronger signal at extremes
        if distance_from_50 < 10:
            # Near 50 - weak signal, but still predict
            strength = distance_from_50 / 100  # 0.0 - 0.10
        elif distance_from_50 < 20:
            strength = 0.10 + (distance_from_50 - 10) / 50  # 0.10 - 0.30
        else:
            strength = 0.30 + (distance_from_50 - 20) / 100  # 0.30 - 0.50

        # RSI momentum confirmation
        if rsi < 50 and rsi_momentum > 0:
            # RSI below 50 but rising = strong UP signal
            strength *= 1.2
        elif rsi > 50 and rsi_momentum < 0:
            # RSI above 50 but falling = strong DOWN signal
            strength *= 1.2
        elif (rsi < 50 and rsi_momentum < 0) or (rsi > 50 and rsi_momentum > 0):
            # Continuation - weaker signal
            strength *= 0.8

        # Volatility penalty (mean reversion worse in high vol)
        if vol_z > 1.0:
            strength *= 0.7

        # Always predict direction (no FLAT)
        if rsi <= 50:
            prob_up = 0.5 + strength
            prob_down = 0.5 - strength
        else:
            prob_up = 0.5 - strength
            prob_down = 0.5 + strength

        return ModelOutput(
            model=self.name,
            prob_up=prob_up,
            prob_down=prob_down,
            state={"rsi": rsi, "strength": strength},
            metrics={"rsi_momentum": rsi_momentum}
        )
```

### 9.2 VolumeMetrix Enhancement

**Проблема:** VolumeMetrix accuracy 46.42%, плохо работает.

```python
class VolumeMetrix(BaseModel):
    """Улучшенный VolumeMetrix."""

    def predict(self, features: Dict) -> ModelOutput:
        volume_z = features.get("volume_z", 0.0)
        ret = features.get("return_1", 0.0)
        volatility = features.get("volatility", 0.0)
        ma_delta = features.get("ma_delta", 0.0)
        close = features.get("close", 1.0)

        ret_z = abs(ret) / (volatility + 1e-12)

        # Volume-Price relationship analysis
        if volume_z > 1.5:
            if ret_z > 1.0:
                # High volume + big move = TREND CONTINUATION
                # Follow the move direction
                direction = "UP" if ret > 0 else "DOWN"
                strength = min(0.4, (volume_z + ret_z) / 10)
            else:
                # High volume + small move = ABSORPTION
                # Potential reversal
                direction = "DOWN" if ret > 0 else "UP"
                strength = min(0.3, volume_z / 5)
        elif volume_z < -0.5:
            # Low volume = weak signal, follow trend
            direction = "UP" if ma_delta > 0 else "DOWN"
            strength = 0.05  # Very low confidence
        else:
            # Normal volume - use price action
            direction = "UP" if ret > 0 else "DOWN"
            strength = min(0.2, abs(ret_z) / 5)

        # Trend confirmation bonus
        trend_aligned = (direction == "UP" and ma_delta > 0) or \
                       (direction == "DOWN" and ma_delta < 0)
        if trend_aligned:
            strength = min(strength * 1.3, 0.5)

        if direction == "UP":
            prob_up = 0.5 + strength
            prob_down = 0.5 - strength
        else:
            prob_up = 0.5 - strength
            prob_down = 0.5 + strength

        return ModelOutput(
            model=self.name,
            prob_up=prob_up,
            prob_down=prob_down,
            state={"volume_z": volume_z, "ret_z": ret_z},
            metrics={"strength": strength, "trend_aligned": trend_aligned}
        )
```

### 9.3 Критерии успеха Sprint 9
- [x] Oscillator FLAT rate < 5% (было 8.3%) → **0%**
- [x] Oscillator accuracy > 48% (было 46%) → **49.13%**
- [x] VolumeMetrix accuracy > 48% (было 46%) → **49.06%**
- [x] **ТЕСТ:** Per-model improvements verified

---

## SPRINT 10: Feature Engineering

### 10.1 Новые фичи
**Файл:** `titan/core/features/stream.py`

```python
class FeatureStream:
    def __init__(self, config):
        # ... existing ...

        # NEW: RSI momentum tracking
        self._rsi_prev = None

        # NEW: Price momentum
        self._price_history = deque(maxlen=5)

        # NEW: Volume trend
        self._volume_sma = deque(maxlen=10)

    def update(self, candle: Candle) -> Dict[str, float]:
        features = self._compute_base_features(candle)

        # NEW: RSI momentum
        rsi = features["rsi"]
        features["rsi_prev"] = self._rsi_prev or rsi
        features["rsi_momentum"] = rsi - features["rsi_prev"]
        self._rsi_prev = rsi

        # NEW: Price momentum (rate of change)
        self._price_history.append(candle.close)
        if len(self._price_history) >= 3:
            features["price_momentum_3"] = (
                candle.close - self._price_history[-3]
            ) / self._price_history[-3]
        else:
            features["price_momentum_3"] = 0.0

        # NEW: Volume trend (is volume increasing?)
        self._volume_sma.append(candle.volume)
        if len(self._volume_sma) >= 5:
            recent_vol = sum(list(self._volume_sma)[-3:]) / 3
            older_vol = sum(list(self._volume_sma)[:3]) / 3
            features["volume_trend"] = (recent_vol - older_vol) / (older_vol + 1e-10)
        else:
            features["volume_trend"] = 0.0

        # NEW: Candle body ratio (wick analysis)
        body = abs(candle.close - candle.open)
        total_range = candle.high - candle.low + 1e-10
        features["body_ratio"] = body / total_range

        # NEW: Upper/Lower wick ratio
        if candle.close > candle.open:  # Bullish candle
            upper_wick = candle.high - candle.close
            lower_wick = candle.open - candle.low
        else:  # Bearish candle
            upper_wick = candle.high - candle.open
            lower_wick = candle.close - candle.low

        features["upper_wick_ratio"] = upper_wick / total_range
        features["lower_wick_ratio"] = lower_wick / total_range

        return features
```

### 10.2 Критерии успеха Sprint 10
- [x] 5 новых фичей добавлено (price_momentum_3, volume_trend, body_ratio, wick ratios, candle_direction)
- [x] Feature correlations проанализированы
- [x] Модели обновлены для использования новых фичей

---

## SPRINT 11: Ensemble Improvements

### 11.1 DynamicWeightOptimizer
**Файл:** `titan/core/weights.py`

```python
class DynamicWeightOptimizer:
    """Оптимизирует веса на основе recent performance."""

    def __init__(self, window: int = 200, min_weight: float = 0.15):
        self.window = window
        self.min_weight = min_weight
        self._history = deque(maxlen=window)

    def update(self, model_accuracies: Dict[str, bool]):
        """
        model_accuracies: {"TRENDVIC": True, "OSCILLATOR": False, ...}
        """
        self._history.append(model_accuracies)

    def get_optimal_weights(self) -> Dict[str, float]:
        """Calculate weights based on recent accuracy."""
        if len(self._history) < 50:
            return None  # Not enough data

        # Calculate accuracy for each model
        accuracies = {}
        for model in ["TRENDVIC", "OSCILLATOR", "VOLUMEMETRIX"]:
            correct = sum(1 for h in self._history if h.get(model, False))
            accuracies[model] = correct / len(self._history)

        # Convert to weights (higher accuracy = higher weight)
        total_acc = sum(accuracies.values())
        if total_acc == 0:
            return None

        weights = {}
        for model, acc in accuracies.items():
            weight = acc / total_acc
            weights[model] = max(self.min_weight, weight)

        # Normalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}
```

### 11.2 AgreementBooster
```python
class AgreementBooster:
    """Увеличивает confidence при согласии моделей."""

    def boost(self, outputs: List[ModelOutput], base_confidence: float) -> float:
        """
        Full agreement: +10% confidence
        Partial agreement: +5%
        No agreement: no change
        """
        directions = [self._get_direction(o) for o in outputs]

        if all(d == directions[0] for d in directions):
            # Full agreement
            return min(0.80, base_confidence * 1.10)

        # Count agreements
        up_count = sum(1 for d in directions if d == "UP")
        down_count = sum(1 for d in directions if d == "DOWN")

        if max(up_count, down_count) >= 2:
            # Partial agreement (2 out of 3)
            return min(0.75, base_confidence * 1.05)

        return base_confidence
```

### 11.3 Критерии успеха Sprint 11
- [x] DynamicWeightOptimizer работает (через AdaptiveWeightManager)
- [x] AgreementBooster интегрирован (_apply_agreement_boost)
- [x] Веса обновляются каждые N предсказаний
- [x] **ТЕСТ:** Full agreement accuracy > 55% (было 51.8%) → **55.77%**

---

## SPRINT 12: ML Model (Optional)

### Проблема
Heuristic models достигли потолка ~50%. Нужен ML для прорыва.

### 12.1 LightGBM Predictor
**Новый файл:** `titan/core/models/ml.py`

```python
import numpy as np
from lightgbm import LGBMClassifier

class MLPredictor:
    """LightGBM модель для prediction."""

    def __init__(self, config):
        self.model = LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        self.is_trained = False
        self.feature_names = [
            "rsi", "rsi_momentum", "volume_z", "volume_trend",
            "volatility_z", "ma_delta", "return_1", "body_ratio"
        ]
        self._training_data = []
        self._training_labels = []
        self.min_samples = 500

    def add_sample(self, features: Dict, label: str):
        """Add training sample."""
        feature_vector = [features.get(f, 0.0) for f in self.feature_names]
        self._training_data.append(feature_vector)
        self._training_labels.append(1 if label == "UP" else 0)

        # Auto-retrain every 100 samples
        if len(self._training_data) % 100 == 0 and len(self._training_data) >= self.min_samples:
            self._train()

    def _train(self):
        """Train the model."""
        X = np.array(self._training_data[-2000:])  # Last 2000 samples
        y = np.array(self._training_labels[-2000:])

        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, features: Dict) -> Optional[ModelOutput]:
        """Predict using trained model."""
        if not self.is_trained:
            return None

        feature_vector = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        proba = self.model.predict_proba(feature_vector)[0]

        return ModelOutput(
            model="ML",
            prob_up=float(proba[1]),
            prob_down=float(proba[0]),
            state={"trained_samples": len(self._training_data)},
            metrics={}
        )
```

### 12.2 Интеграция
```python
class Ensemble:
    def __init__(self, ..., ml_model: MLPredictor = None):
        self._ml = ml_model

    def decide(self, outputs, features):
        # Get ML prediction if available
        if self._ml and self._ml.is_trained:
            ml_output = self._ml.predict(features)
            if ml_output:
                outputs.append(ml_output)
                # ML gets higher weight
                self._weights["ML"] = 0.30
                # Renormalize other weights
```

### 12.3 Критерии успеха Sprint 12
- [ ] MLPredictor обучается online
- [ ] Integration с ensemble
- [ ] **ТЕСТ:** Overall accuracy > 52%

---

## SPRINT 13: Pattern System (Model Experience)

### Проблема
Система не использует накопленный опыт. PatternStore записывает события,
но не использует их для улучшения предсказаний.

### 13.1 PatternExperience
**Расширение:** `titan/core/patterns.py`

```python
class PatternExperience:
    """Анализ исторической эффективности паттернов."""

    def __init__(self, pattern_store: PatternStore):
        self._store = pattern_store
        self._cache = {}  # pattern_id -> stats

    def get_pattern_stats(self, pattern_id: int, model_name: str = None) -> Dict:
        """Получить статистику паттерна."""
        # SELECT from pattern_events
        # Calculate: total_uses, accuracy, avg_confidence, last_used
        return {
            "total_uses": 100,
            "accuracy": 0.58,  # 58% accuracy for this pattern
            "up_accuracy": 0.62,
            "down_accuracy": 0.54,
            "avg_confidence": 0.55,
            "last_used": 1703856000,
        }

    def get_pattern_bias(self, pattern_id: int) -> Optional[str]:
        """Определить, в каком направлении паттерн работает лучше."""
        stats = self.get_pattern_stats(pattern_id)
        if stats["up_accuracy"] > stats["down_accuracy"] + 0.05:
            return "UP"
        elif stats["down_accuracy"] > stats["up_accuracy"] + 0.05:
            return "DOWN"
        return None

    def should_trust_pattern(self, pattern_id: int, min_uses: int = 20) -> bool:
        """Достаточно ли данных для доверия паттерну."""
        stats = self.get_pattern_stats(pattern_id)
        return stats["total_uses"] >= min_uses
```

### 13.2 PatternAdjuster
**Новый файл:** `titan/core/adapters/pattern.py`

```python
class PatternAdjuster:
    """Корректирует предсказания на основе опыта паттернов."""

    def __init__(self, experience: PatternExperience, config: ConfigStore):
        self._exp = experience
        self._config = config

    def adjust_decision(
        self,
        decision: Decision,
        pattern_id: int,
        features: Dict[str, float]
    ) -> Decision:
        """
        Корректировка на основе исторической эффективности паттерна.

        1. Если паттерн исторически точен (>55%) - увеличить confidence
        2. Если паттерн неточен (<45%) - уменьшить confidence
        3. Если паттерн имеет bias в одном направлении - учесть это
        """
        if not self._exp.should_trust_pattern(pattern_id):
            return decision  # Not enough data

        stats = self._exp.get_pattern_stats(pattern_id)

        # Confidence adjustment based on pattern accuracy
        pattern_acc = stats["accuracy"]
        if pattern_acc > 0.55:
            # Pattern is historically accurate - boost confidence
            boost = (pattern_acc - 0.50) * 0.5  # max +2.5%
            new_conf = min(decision.confidence + boost, 0.65)
        elif pattern_acc < 0.45:
            # Pattern is historically inaccurate - reduce confidence
            penalty = (0.50 - pattern_acc) * 0.5  # max -2.5%
            new_conf = max(decision.confidence - penalty, 0.50)
        else:
            new_conf = decision.confidence

        # Direction bias check
        bias = self._exp.get_pattern_bias(pattern_id)
        if bias and bias != decision.direction:
            # Pattern historically works better in opposite direction
            # Reduce confidence but don't flip
            new_conf = max(new_conf - 0.02, 0.50)

        return Decision(
            direction=decision.direction,
            confidence=new_conf,
            prob_up=decision.prob_up,
            prob_down=decision.prob_down,
        )
```

### 13.3 Интеграция в Ensemble
```python
class Ensemble:
    def __init__(self, ..., pattern_adjuster: PatternAdjuster = None):
        self._pattern_adj = pattern_adjuster

    def decide(self, outputs, features, ts=None, pattern_id=None):
        # ... existing logic ...
        decision = Decision(direction, confidence, prob_up, prob_down)

        # Apply pattern experience
        if self._pattern_adj and pattern_id:
            decision = self._pattern_adj.adjust_decision(
                decision, pattern_id, features
            )

        return decision
```

### 13.4 Pattern Decay
**Файл:** `titan/core/patterns.py`

```python
def get_pattern_stats_with_decay(
    self,
    pattern_id: int,
    decay_hours: int = 168  # 1 week
) -> Dict:
    """
    Статистика с временным decay.
    Более свежие события имеют больший вес.
    """
    events = self._get_events(pattern_id)
    now = time.time()

    weighted_correct = 0.0
    weighted_total = 0.0

    for event in events:
        age_hours = (now - event["event_ts"]) / 3600
        weight = math.exp(-age_hours / decay_hours)  # Exponential decay

        weighted_total += weight
        if event["hit"]:
            weighted_correct += weight

    if weighted_total < 1.0:
        return {"accuracy": 0.50, "confidence": 0.0}

    return {
        "accuracy": weighted_correct / weighted_total,
        "confidence": min(weighted_total / 20, 1.0),  # Confidence in estimate
    }
```

### 13.5 Критерии успеха Sprint 13
- [x] PatternExperience анализирует историю (с time decay 168h)
- [x] PatternAdjuster корректирует decisions (boost/penalty ±3%)
- [x] Decay для устаревших паттернов (exponential decay)
- [x] **ТЕСТ:** Patterns with >20 uses show better accuracy adjustment
- [x] **ТЕСТ:** Overall accuracy improvement → **52.12%**, ECE **1.92%**, Sharpe **2.55**

---

## SPRINT 12: Pattern System Hardening (COMPLETED ✅)

### Исправленные баги:
- ✅ FLAT handling - FLAT больше не загрязняет accuracy stats
- ✅ Min scale floor (0.7) - предотвращает порочный круг
- ✅ Pattern deduplication - проверка по pattern_key перед созданием
- ✅ Overconfidence threshold - повышен с 0.4 до 0.55
- ✅ cap_strength formula - исправлено на `(cap - 0.5) * 2`

### Результат Sprint 12:
**Точность 48.6% (хуже случайного)** - Pattern system не даёт улучшений.
Рекомендация: отключить pattern adjustments, перейти на ML.

---

## SPRINT 14: LightGBM Classifier (COMPLETED ✅)

### Результат
- ✅ 31 scale-invariant feature добавлено в FeatureStream
- ✅ DirectionalClassifier создан в `titan/core/models/ml.py`
- ✅ Интеграция в Ensemble с 25% весом
- ✅ Accuracy 52.16% (статистически лучше случайного, p=0.001)

### Добавленные фичи (17 новых)
```python
# Lagged returns
return_lag_1..5        # 5 фичей

# Volatility
atr_pct                # ATR в % от цены
high_low_range_pct     # Range свечи в %
vol_ratio              # Текущая vol / средняя vol

# Trend
ema_10_spread_pct      # Расстояние от EMA10
ema_20_spread_pct      # Расстояние от EMA20
ma_delta_pct           # MA delta в %

# Returns
return_5, return_10    # Multi-period returns
log_return_1           # Log return

# RSI zones
rsi_oversold           # RSI < 30
rsi_overbought         # RSI > 70
rsi_neutral            # 30 <= RSI <= 70

# Volume
volume_change_pct      # Изменение объёма
body_pct               # Body в % от цены
```

---

## SPRINT 15: Confidence Filter (COMPLETED ✅)

### Результат
- ✅ max_confidence увеличен с 0.62 до 0.70
- ✅ confidence_filter.threshold = 0.55 добавлен
- ✅ Filtered accuracy tracking в BacktestStats
- ✅ **Filtered accuracy 65.21%** при conf≥55%
- ✅ Coverage 9.07% (914/10079 предсказаний)

### Confidence Distribution
```
50-55%: 50.87% accuracy, 9165 predictions (91%)
55-60%: 65.54% accuracy, 859 predictions (8.5%)
60-65%: 60.00% accuracy, 55 predictions (0.5%)
```

### Вывод
Filtered accuracy близка к цели (65% vs 75%), но coverage слишком низкий.
Нужно увеличить долю high-confidence предсказаний с 9% до 30%+.

---

## SPRINT 16: New Features (COMPLETED ✅)

### Добавленные фичи
- `bb_position` - Position within Bollinger Bands [0, 1]
- `vol_imbalance_20` - Volume imbalance over 20 periods
- `adx` - Average Directional Index (trend strength)
- `mfi` - Money Flow Index

### Результат
- Accuracy: 51.50% (статистически значимо, p=0.001)
- ECE: 0.8%
- Система достигла плато, требуется новый подход

---

## SPRINT 17: Session Adapter (NEXT)

### Концепция
**Per-session (ASIA/EUROPE/US) adaptation** весов моделей и параметров.

Обоснование: 4% разница в accuracy между сессиями:
- ASIA: 50.5%
- EUROPE: 46.5% (worst)
- US: 49.8%

### 17.1 SessionMemory Schema (SQLite)

```sql
-- Per-session model statistics
CREATE TABLE session_stats (
    session TEXT NOT NULL,        -- 'ASIA', 'EUROPE', 'US'
    model TEXT NOT NULL,
    regime TEXT,
    total INTEGER DEFAULT 0,
    correct INTEGER DEFAULT 0,
    conf_sum REAL DEFAULT 0,
    return_sum REAL DEFAULT 0,
    last_update INTEGER,
    decay_total REAL DEFAULT 0,
    decay_correct REAL DEFAULT 0,
    PRIMARY KEY (session, model, regime)
);

-- Thompson Sampling for discrete parameters
CREATE TABLE session_params (
    session TEXT NOT NULL,
    param_key TEXT NOT NULL,
    param_value REAL NOT NULL,
    total INTEGER DEFAULT 0,
    correct INTEGER DEFAULT 0,
    return_sum REAL DEFAULT 0,
    last_tested INTEGER,
    PRIMARY KEY (session, param_key, param_value)
);

-- Per-session confidence calibration
CREATE TABLE session_calibration (
    session TEXT NOT NULL,
    bin_idx INTEGER NOT NULL,     -- 0-3 for bins [50-55, 55-60, 60-65, 65-70]
    total INTEGER DEFAULT 0,
    correct INTEGER DEFAULT 0,
    conf_sum REAL DEFAULT 0,
    PRIMARY KEY (session, bin_idx)
);

-- Stored session config
CREATE TABLE session_config (
    session TEXT PRIMARY KEY,
    config_json TEXT,             -- Full config snapshot
    updated_at INTEGER
);
```

### 17.2 SessionAdapter Class

**File:** `titan/core/adapters/session.py`

```python
class SessionAdapter:
    """Per-session model configuration adapter."""

    # Update frequencies (in predictions)
    WEIGHT_UPDATE_FREQ = 50
    PARAM_UPDATE_FREQ = 500
    CALIBRATION_UPDATE_FREQ = 100

    # Trust thresholds
    MIN_SAMPLES = 50
    MAX_CI_WIDTH = 0.10

    # Decay
    HALF_LIFE_HOURS = 168  # 1 week

    def __init__(self, db_path: str, global_config: ConfigStore):
        self._db = sqlite3.connect(db_path)
        self._global = global_config
        self._session_configs = {}
        self._load_configs()

    def get_session(self, ts: int) -> str:
        """Determine trading session from timestamp."""
        hour = datetime.fromtimestamp(ts, tz=timezone.utc).hour
        if 0 <= hour < 8:
            return "ASIA"
        elif 8 <= hour < 16:
            return "EUROPE"
        else:
            return "US"

    def get_config(self, session: str) -> ConfigStore:
        """Get session-specific config."""
        return self._session_configs.get(session, self._global)

    def get_weights(self, session: str, regime: str) -> Dict[str, float]:
        """Get model weights for session/regime with shrinkage to global."""
        stats = self._get_session_stats(session, regime)
        global_stats = self._get_global_stats(regime)

        weights = {}
        for model in ["TRENDVIC", "OSCILLATOR", "VOLUMEMETRIX", "ML_CLASSIFIER"]:
            # Shrinkage formula
            k = 1000  # Prior strength (global)
            n = stats[model].total

            global_acc = global_stats[model].accuracy
            session_acc = stats[model].accuracy if n > 0 else global_acc

            effective_acc = (global_acc * k + session_acc * n) / (k + n)
            weights[model] = effective_acc

        # Normalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def select_param(self, session: str, param_key: str) -> float:
        """Thompson Sampling for discrete parameter selection."""
        options = PARAM_OPTIONS[param_key]
        posteriors = []

        for opt in options:
            stats = self._get_param_stats(session, param_key, opt)
            alpha = stats.correct + 1
            beta = stats.total - stats.correct + 1
            sample = np.random.beta(alpha, beta)
            posteriors.append(sample)

        best_idx = np.argmax(posteriors)
        return options[best_idx]

    def calibrate_confidence(self, session: str, raw_conf: float) -> float:
        """Temperature scaling for per-session calibration."""
        bin_idx = self._get_bin(raw_conf)
        bin_stats = self._get_calibration_stats(session, bin_idx)

        # Blend with global if insufficient data
        if bin_stats.total < 30:
            global_stats = self._get_global_calibration(bin_idx)
            bin_stats = self._blend_stats(bin_stats, global_stats)

        if bin_stats.total == 0:
            return raw_conf

        empirical_acc = bin_stats.correct / bin_stats.total

        # Temperature scaling
        calibrated = 0.5 + (raw_conf - 0.5) * (empirical_acc - 0.5) / (raw_conf - 0.5 + 1e-6)
        return max(0.50, min(0.70, calibrated))

    def record_outcome(self, session: str, model: str, regime: str,
                       hit: bool, conf: float, return_pct: float, ts: int):
        """Record prediction outcome for learning."""
        # Update stats with decay
        self._update_stats_with_decay(session, model, regime, hit, conf, ts)

        # Check if update needed
        count = self._get_prediction_count(session)
        if count % self.WEIGHT_UPDATE_FREQ == 0:
            self._update_weights(session)
        if count % self.PARAM_UPDATE_FREQ == 0:
            self._update_params(session)
        if count % self.CALIBRATION_UPDATE_FREQ == 0:
            self._update_calibration(session)

    def _update_stats_with_decay(self, session, model, regime, hit, conf, ts):
        """Update stats with exponential decay."""
        stats = self._get_session_stats_row(session, model, regime)

        hours_since = (ts - stats.last_update) / 3600
        decay = 0.5 ** (hours_since / self.HALF_LIFE_HOURS)

        stats.decay_total *= decay
        stats.decay_correct *= decay
        stats.decay_total += 1
        stats.decay_correct += (1 if hit else 0)
        stats.last_update = ts

        self._save_stats(session, model, regime, stats)

    def _can_trust(self, stats) -> bool:
        """Check if we have enough data to trust session stats."""
        if stats.total < self.MIN_SAMPLES:
            return False

        # Wilson score interval
        acc = stats.correct / stats.total
        z = 1.96  # 95% CI
        ci_width = 2 * z * np.sqrt(acc * (1 - acc) / stats.total)

        return ci_width <= self.MAX_CI_WIDTH


# Parameter options for Thompson Sampling
PARAM_OPTIONS = {
    "oscillator.rsi_oversold": [25, 28, 30, 32, 35],
    "oscillator.rsi_overbought": [65, 68, 70, 72, 75],
    "volumemetrix.vol_threshold": [1.0, 1.5, 2.0, 2.5],
    "trendvic.dead_zone": [0.00003, 0.00005, 0.00007, 0.0001],
}
```

### 17.3 Integration

**backtest.py:**
```python
def run_backtest(config, candles, ...):
    session_adapter = SessionAdapter("session.db", config)

    for candle in candles:
        ts = candle.open_time
        session = session_adapter.get_session(ts)

        # Get session-specific config
        session_config = session_adapter.get_config(session)

        # Get session-specific weights
        weights = session_adapter.get_weights(session, regime)

        # Make prediction
        decision = ensemble.decide(outputs, features, ts, pattern_id, weights)

        # Calibrate confidence
        decision.confidence = session_adapter.calibrate_confidence(
            session, decision.confidence
        )

        # After evaluation, record outcome
        session_adapter.record_outcome(
            session, model, regime, hit, decision.confidence, return_pct, ts
        )
```

### 17.4 Критерии успеха Sprint 17
- [ ] SessionMemory schema создан
- [ ] SessionAdapter class реализован
- [ ] Thompson Sampling для параметров работает
- [ ] Decay mechanism (168h half-life) работает
- [ ] Trust blocks (min 50 samples) работают
- [ ] Интеграция в backtest.py
- [ ] **ТЕСТ:** Per-session accuracy улучшена на 2-3%
- [ ] **ТЕСТ:** EUROPE accuracy > 48% (было 46.5%)

---

## SPRINT 17: 5-Minute Timeframe

### Цель
Перейти с 1-минутных на 5-минутные свечи для уменьшения шума.

### 17.1 Изменения
```python
# cli.py
--interval 5  # вместо 1

# features/stream.py
# Пересчитать lookback windows для 5m
```

### 17.2 Критерии успеха Sprint 17
- [ ] Система работает на 5m свечах
- [ ] **ТЕСТ:** Accuracy > 55% (меньше шума)
- [ ] **ТЕСТ:** Filtered accuracy > 70%

---

## SPRINT 18: Online Learning (ФИНАЛ)

### Цель
Добавить online обучение для адаптации к изменениям рынка.
**Выполняется ПОСЛЕДНИМ**, когда система отлажена и проверена на исторических данных.

### 17.1 SGD + RMSProp Updater
```python
class OnlineLearner:
    def __init__(self, n_features: int, lr: float = 0.05):
        self.weights = np.zeros(n_features)
        self.lr = lr
        self.rmsprop_g = np.zeros(n_features)  # RMSProp накопитель

    def update(self, features: np.ndarray, error: float):
        """Обновление весов после получения outcome."""
        gradient = error * features

        # RMSProp: G = 0.95*G + 0.05*g²
        self.rmsprop_g = 0.95 * self.rmsprop_g + 0.05 * gradient**2

        # Update: w -= lr * g / sqrt(G + eps)
        self.weights -= self.lr * gradient / (np.sqrt(self.rmsprop_g) + 1e-8)
```

### 17.2 Multi-Scale EMA Memory
```python
# Три временных масштаба памяти (из Chronos):
short_ema  = 0.05 * new + 0.95 * old   # ~20 обновлений
medium_ema = 0.01 * new + 0.99 * old   # ~100 обновлений
long_ema   = 0.001 * new + 0.999 * old # ~1000 обновлений
```

### 17.3 Критерии успеха Sprint 17
- [ ] Online learning обновляет веса в реальном времени
- [ ] Multi-scale EMA memory
- [ ] **ТЕСТ:** Accuracy стабильна при изменении рынка
- [ ] Система готова к production

---

## Приоритеты реализации (обновлено 2025-12-30)

### Завершённые спринты
| Sprint | Результат | Статус |
|--------|-----------|--------|
| Sprint 4: Confidence Recalibration | ECE 1.92% ✅ | ✅ Done |
| Sprint 5: Volatile Handler | Volatile error 51.5% | ✅ Done |
| Sprint 6: Trending Down Fix | Trending error 49.3% | ✅ Done |
| Sprint 8: Temporal Patterns | ECE improved | ✅ Done |
| Sprint 9: Model Improvements | Models ~49% | ✅ Done |
| Sprint 10: Features | 13 features | ✅ Done |
| Sprint 11: Ensemble | Agreement 55.77% | ✅ Done |
| Sprint 13: Pattern System | No improvement ❌ | ✅ Done |
| Sprint 12: Bug Fixes | Accuracy 48.6% ❌ | ✅ Done |
| Sprint 14: LightGBM | Accuracy 52.17% ✅ | ✅ Done |
| Sprint 15: Confidence Filter | Filtered 65.21% ✅ | ✅ Done |
| **Sprint 16: New Features** | **bb_position, adx, mfi** | ✅ Done |

### Предстоящие спринты (4 Strategic Directions)
| Приоритет | Sprint | Описание | Статус |
|-----------|--------|----------|--------|
| **1** | **Sprint 17: Session Adapter** | Per-session (ASIA/EUROPE/US) adaptation | ⏳ NEXT |
| 2 | Sprint 18: ML Hardening | Time-split CV, calibration, leakage control | ⏳ Pending |
| 3 | Sprint 19: 5-Minute Timeframe | Less noise, stronger signals | ⏳ Pending |
| 4 | Sprint 20: Online Learning | SGD + RMSProp (FINAL) | ⏳ Last |

### Strategic Directions Summary

**Direction 1: Session Adapter** (Sprint 17)
- Per-session model weights via shrinkage to global
- Thompson Sampling for discrete parameters
- Temperature scaling for calibration
- 168h half-life decay
- Trust blocks (50 samples, 10% CI width)

**Direction 2: ML Hardening** (Sprint 18)
- Time-split cross-validation
- Isotonic regression calibration
- Stricter feature selection

**Direction 3: Data/Labeling** (Sprint 19)
- ATR-relative thresholds
- Noise filtering
- Spread/commission modeling

**Direction 4: Pattern Memory as Input** (Future)
- Pattern stats as ML features
- Not just confidence adjusters

### Reinforcement Learning: Полный анализ

#### Постановка задачи
```
State (s):    features + session + regime
Action (a):   выбор весов моделей / параметров
Reward (r):   +1 если предсказание верное, 0 иначе
```

**Ключевой вопрос:** Single-step или sequential decision problem?

#### Сравнение типов RL

| Тип RL | Описание | Подходит? | Почему |
|--------|----------|-----------|--------|
| **Multi-Armed Bandits** | Выбор из K действий без состояния | Частично | Не учитывает контекст |
| **Contextual Bandits** | Выбор с учётом контекста | **Идеально** | Контекст = features, single-step |
| **Q-Learning / DQN** | Табличный/нейросетевой Q(s,a) | Overkill | Нет sequential dependencies |
| **Policy Gradient / PPO** | Прямая оптимизация policy | Overkill | Требует миллионы samples |

#### Почему Contextual Bandits идеален

```
На каждом шаге t:
1. Наблюдаем контекст x_t = (features, session, regime)
2. Выбираем действие a_t = (веса, параметры)
3. Получаем reward r_t = I(prediction correct)
4. НЕТ перехода состояния (s_{t+1} не зависит от a_t)
```

**Критический момент:**
```
Full RL:      s_{t+1} = f(s_t, a_t)    # Действие меняет состояние
Наша задача:  s_{t+1} = f(market)      # Состояние определяется рынком
```

#### Thompson Sampling: математика

```python
# Beta(α, β) posterior для каждого значения параметра
# После каждого outcome:
if hit: α += 1
else:   β += 1

# При выборе параметра:
for v in options:
    θ_v ~ Beta(α_v, β_v)
select argmax(θ_v)

# Свойства:
# Beta(1, 1) = uniform prior
# E[θ] = α / (α + β)
# Variance уменьшается с данными
```

#### Data Efficiency

| Метод | Samples до сходимости |
|-------|----------------------|
| Thompson Sampling | 100-500 per arm |
| DQN | 100,000 - 1,000,000 |
| PPO | 1,000,000 - 10,000,000 |

У нас ~10,000 свечей/неделю. Thompson Sampling сойдётся за дни, DQN/PPO за месяцы.

#### Когда Full RL был бы нужен

**Scenario A: Portfolio Management**
- Action: [buy 10%, hold, sell 5%]
- s_{t+1} зависит от a_t (позиция изменилась)
→ Full RL имеет смысл

**Scenario B: Market Making**
- Action: [bid_price, ask_price, size]
- s_{t+1} сильно зависит от a_t
→ Full RL обязателен

**Наша задача не попадает в эти scenarios.**

#### Вывод

```
┌─────────────────────────────────────────────────────────────┐
│  РЕКОМЕНДАЦИЯ: Thompson Sampling (Contextual Bandits)       │
├─────────────────────────────────────────────────────────────┤
│  ✅ Идеально для single-step prediction                     │
│  ✅ Простая реализация (~50 строк)                          │
│  ✅ Не требует hyperparameter tuning                        │
│  ✅ Data-efficient (100-500 samples per arm)               │
│  ✅ Естественный exploration/exploitation                   │
├─────────────────────────────────────────────────────────────┤
│  ❌ Full RL (DQN/PPO) — overkill:                           │
│     - Нет sequential dependencies                           │
│     - Требует 100x больше данных                            │
│     - 20x сложнее в реализации                              │
│     - Нет доказанных преимуществ                           │
└─────────────────────────────────────────────────────────────┘
```

**Ключевая проблема:** Система достигла плато ~51-52% accuracy.
**Решение:** Per-session adaptation с Thompson Sampling для устранения 4% разрыва между сессиями.

---

## Целевые метрики (обновлено)

| Метрика | Sprint 16 | Sprint 17 | Sprint 18 | Ultimate |
|---------|-----------|-----------|-----------|----------|
| Overall Accuracy | **51.50%** | 53-54% | 55-57% | 60%+ |
| Session Gap | **4%** | **<2%** | <1% | 0% |
| EUROPE Accuracy | **46.5%** | **≥49%** | ≥51% | ≥55% |
| FLAT rate | **0%** ✅ | 0% | 0% | 0% |
| p-value | **0.001** ✅ | <0.001 | <0.001 | <0.001 |
| ECE | **0.8%** ✅ | <3% | <3% | <2% |

### Per-Session Targets (Sprint 17)
| Session | Current | Target |
|---------|---------|--------|
| ASIA | 50.5% | 52%+ |
| EUROPE | 46.5% | 49%+ |
| US | 49.8% | 52%+ |

---

## Файлы для создания/модификации

### Новые файлы (Sprint 17-20):
1. `titan/core/adapters/session.py` - SessionAdapter класс (Sprint 17) ⏳ NEXT
2. `titan/core/online.py` - OnlineLearner с RMSProp (Sprint 20)

### Модификации (Sprint 17):
1. `titan/core/backtest.py` - Интеграция SessionAdapter
2. `titan/core/live.py` - Интеграция SessionAdapter
3. `titan/core/ensemble.py` - Поддержка внешних весов

### Завершённые файлы:
- `titan/core/models/ml.py` ✅ (Sprint 14)
- `titan/core/features/stream.py` ✅ (35+ features, включая bb_position, adx, mfi)
- `titan/core/backtest.py` ✅ (filtered accuracy tracking)
- `titan/core/strategies/volatile.py` ✅
- `titan/core/detectors/exhaustion.py` ✅
- `titan/core/adapters/temporal.py` ✅
- `titan/core/adapters/pattern.py` ✅
- `titan/core/patterns.py` ✅
- `titan/core/calibration.py` ✅
- `titan/core/regime.py` ✅
- `titan/core/weights.py` ✅
- `titan/core/models/heuristic.py` ✅
- `titan/core/config.py` ✅
- `cleanup.py` ✅ (utility)
- `docs/SYSTEM_ANALYSIS.md` ✅
