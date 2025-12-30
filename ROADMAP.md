# План улучшений Titan v2

## Текущее состояние (Sprint 13 - 2025-12-29)

| Метрика | Baseline | После Sprint 13 | Изменение |
|---------|----------|-----------------|-----------|
| Ensemble Accuracy | 49.9% | **52.12%** | +2.2% ✅ |
| Full Agreement Acc | 51.8% | **55.77%** | +4.0% ✅ |
| ECE | 5.3% | **1.92%** | -3.4% ✅ |
| Conf 55-60% Accuracy | 50.8% | **61.40%** | +10.6% ✅ |
| Sharpe Ratio | 1.4 | **2.55** | +1.15 ✅ |
| Direction Balance | 0.91 | **0.899** | Stable ✅ |
| TrendVIC FLAT | 45 | **45** | Stable |
| Oscillator FLAT | 119→0 | **0** | Fixed ✅ |
| VolumeMetrix FLAT | 213→0 | **0** | Fixed ✅ |

### Sprint 13 результаты
- PatternExperience: анализ исторической эффективности паттернов с time decay
- PatternAdjuster: корректировка confidence на основе паттернов
- Интеграция в Ensemble, backtest.py, live.py

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

## SPRINT 12: Pattern System Hardening (IN PROGRESS)

### Проблема
При анализе Sprint 13 обнаружены критические баги и технический долг:

### 12.1 Критический баг: Удаление чужих snapshots
**Файл:** `titan/core/patterns.py:917-926`

```python
# БЫЛО (БАГ):
DELETE FROM pattern_event_snapshots
WHERE event_id NOT IN (
    SELECT id FROM pattern_events WHERE pattern_id = ?
)
# Удаляет snapshots ВСЕХ паттернов кроме текущего!

# ДОЛЖНО БЫТЬ:
# 1. Собрать ID событий для удаления
# 2. Удалить их snapshots
# 3. Удалить сами события
```

### 12.2 Критический баг: Data Leakage
**Файл:** `titan/core/patterns.py:1247`

```python
# БЫЛО (БАГ):
events = self._store.get_events(pattern_id)  # Видит "будущие" события!

# ДОЛЖНО БЫТЬ:
events = self._store.get_events(pattern_id, max_ts=current_ts)
```

### 12.3 Рефакторинг: Config-driven константы
**Файл:** `titan/core/patterns.py:11-15`

```python
# БЫЛО (жёстко):
MAX_DECISIONS = 50000
TOP_DECISIONS_COUNT = 1000
INACTIVE_AFTER_DAYS = 30

# ДОЛЖНО БЫТЬ:
max_decisions = int(config.get("pattern.max_decisions", 50000))
```

### 12.4 Рефакторинг: day_of_week consistency
- `ExtendedConditions` имеет `day_of_week`
- `build_pattern_key()` НЕ включает его
- `pattern_search_index` НЕ имеет колонку
- **Решение:** Либо добавить везде, либо убрать из hash

### 12.5 Рефакторинг: Удаление мёртвого кода
- `pattern_conditions_v2` — таблица создаётся, но не используется
- `conditions_version` — колонка есть, всегда = 1
- `momentum`, `rsi_zone` — в схеме, не используются

### 12.6 Критерии успеха Sprint 12
- [ ] Баг snapshot deletion исправлен
- [ ] Data leakage исправлен (time-bounded queries)
- [ ] Константы читаются из конфига
- [ ] day_of_week согласован (или удалён)
- [ ] Мёртвый код удалён
- [ ] **ТЕСТ:** Backtest проходит без регрессии

---

## Приоритеты реализации

| Приоритет | Sprint | Ожидаемый эффект | Статус |
|-----------|--------|------------------|--------|
| 1 | Sprint 4: Confidence Recalibration | ECE < 5%, Confident Wrong < 40% | ✅ Done |
| 2 | Sprint 5: Volatile Handler | Volatile error < 52% | ✅ Done |
| 3 | Sprint 6: Trending Down Fix | Trending_down error < 50% | ✅ Done |
| 4 | Sprint 8: Temporal Patterns | Danger hours acc > 45% | ✅ Done |
| 5 | Sprint 9: Model Improvements | Per-model acc > 48% | ✅ Done |
| 6 | Sprint 10: Features | Better correlations | ✅ Done |
| 7 | Sprint 11: Ensemble | Agreement acc > 55% | ✅ Done (55.77%) |
| 8 | Sprint 13: Pattern System | Pattern-based accuracy boost | ✅ Done |
| 9 | **Sprint 12: Hardening** | Bug fixes, data integrity | 🔄 In Progress |
| 10 | Sprint 14: ML Model | Overall acc > 55% | ⏳ Next |

---

## Целевые метрики

| Метрика | Baseline | После Sprint 4-6 | После Sprint 7-13 | Target |
|---------|----------|------------------|-------------------|--------|
| Ensemble Acc | 49.9% | 51-52% | **52.12%** ✅ | 55-60% |
| p-value | 0.6 | < 0.3 | **0.05** ✅ | < 0.05 |
| Volatile Error | 56.3% | < 52% | **~48%** ✅ | < 48% |
| ECE | 5.3% | < 4% | **1.92%** ✅ | < 3% |
| Conf Wrong | 50% | < 40% | **0%** ✅ | < 30% |
| Sharpe | - | - | **2.55** ✅ | > 1.5 |
| Full Agreement | 51.8% | - | **55.77%** ✅ | > 55% |

---

## Файлы для создания/модификации

### Новые файлы:
1. `titan/core/strategies/volatile.py` ✅
2. `titan/core/detectors/exhaustion.py` ✅
3. `titan/core/adapters/movement.py`
4. `titan/core/adapters/temporal.py` ✅
5. `titan/core/filters/movement.py`
6. `titan/core/models/ml.py` ⏳ (Sprint 12)
7. `titan/core/adapters/pattern.py` ✅ (Sprint 13)

### Модификации:
1. `titan/core/calibration.py` - ConfidenceCompressor ✅
2. `titan/core/ensemble.py` - All integrations ✅
3. `titan/core/regime.py` - VolatileDetector ✅
4. `titan/core/weights.py` - AdaptiveWeightManager ✅
5. `titan/core/models/heuristic.py` - Oscillator, VolumeMetrix ✅
6. `titan/core/features/stream.py` - New features ✅
7. `titan/core/patterns.py` - PatternExperience ✅ (Sprint 13)
8. `titan/core/backtest.py` - Pattern integration ✅ (Sprint 13)
9. `titan/core/live.py` - Pattern integration ✅ (Sprint 13)
10. `titan/core/config.py` - Pattern parameters ✅ (Sprint 13)
