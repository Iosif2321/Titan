from typing import Dict, Optional, Sequence

from titan.core.config import ConfigStore
from titan.core.models.base import BaseModel
from titan.core.types import ModelOutput, PatternContext
from titan.core.utils import clamp, safe_div


def _pattern_state_summary(context: PatternContext) -> Dict[str, object]:
    return {
        "pattern_id": context.pattern_id,
        "pattern_key": context.pattern_key,
        "match_ratio": context.match_ratio,
        "accuracy": context.accuracy,
        "bias": context.bias,
        "trust_confidence": context.trust_confidence,
    }


def _apply_pattern_context_strength(
    strength: float,
    direction: str,
    context: Optional[PatternContext],
    feature_keys: Sequence[str],
) -> float:
    if not context:
        return strength

    scale = 1.0
    trust = context.trust_confidence

    if context.bias:
        if context.bias == direction:
            scale *= 1.0 + (0.05 * trust)
        else:
            scale *= 1.0 - (0.05 * trust)

    for key in feature_keys:
        insight = context.feature_insights.get(key)
        if not insight:
            continue
        acc = float(insight.get("accuracy", 0.5))
        if acc > 0.55:
            scale *= 1.0 + (0.05 * trust)
        elif acc < 0.45:
            scale *= 1.0 - (0.05 * trust)

    for key in ("session", "hour"):
        insight = context.temporal_insights.get(key)
        if not insight:
            continue
        acc = float(insight.get("accuracy", 0.5))
        if acc > 0.55:
            scale *= 1.0 + (0.03 * trust)
        elif acc < 0.45:
            scale *= 1.0 - (0.03 * trust)

    # Prevent excessive strength reduction - minimum scale 0.7
    # This avoids vicious cycle: low accuracy -> low strength -> FLAT -> lower accuracy
    scale = max(scale, 0.7)

    strength = clamp(strength * scale, 0.0, 0.5)

    if context.confidence_cap is not None:
        # confidence_cap = max allowed probability
        # prob = 0.5 + 0.5 * strength
        # cap_strength = (confidence_cap - 0.5) * 2
        cap_strength = max((context.confidence_cap - 0.5) * 2, 0.0)
        strength = min(strength, cap_strength)

    return strength


class TrendVIC(BaseModel):
    """Trend-following model using MA crossover with candle confirmation.

    Uses ma_delta as primary signal, with body_ratio and price_momentum
    for confirmation.
    """

    def __init__(self, config: ConfigStore) -> None:
        self.name = "TRENDVIC"
        self._config = config

    def predict(
        self,
        features: Dict[str, float],
        pattern_context: Optional[PatternContext] = None,
    ) -> ModelOutput:
        ma_delta = features.get("ma_delta", 0.0)
        volatility = features.get("volatility", 0.0)
        close = features.get("close", 0.0)
        body_ratio = features.get("body_ratio", 0.5)
        candle_direction = features.get("candle_direction", 0.0)
        price_momentum_3 = features.get("price_momentum_3", 0.0)

        scale = max(volatility * close, 1e-12)
        base_strength = clamp(abs(ma_delta) / scale, 0.0, 1.0)

        # Confirmation factor based on candle body and momentum
        confirmation = 1.0

        # Strong body (>70%) in same direction = confirmation
        if body_ratio > 0.7:
            if (ma_delta > 0 and candle_direction > 0) or \
               (ma_delta < 0 and candle_direction < 0):
                confirmation = 1.2  # Boost
            elif (ma_delta > 0 and candle_direction < 0) or \
                 (ma_delta < 0 and candle_direction > 0):
                confirmation = 0.8  # Penalty for contradiction

        # Price momentum alignment
        if (ma_delta > 0 and price_momentum_3 > 0.001) or \
           (ma_delta < 0 and price_momentum_3 < -0.001):
            confirmation *= 1.1  # Momentum confirms trend

        strength = clamp(base_strength * confirmation, 0.0, 1.0)

        if ma_delta >= 0:
            prob_up = 0.5 + 0.5 * strength
            prob_down = 1.0 - prob_up
            signal = "up"
        else:
            prob_down = 0.5 + 0.5 * strength
            prob_up = 1.0 - prob_down
            signal = "down"

        direction = "UP" if prob_up >= prob_down else "DOWN"
        strength = _apply_pattern_context_strength(
            strength,
            direction,
            pattern_context,
            feature_keys=("volatility_z", "body_ratio"),
        )
        if direction == "UP":
            prob_up = 0.5 + 0.5 * strength
            prob_down = 1.0 - prob_up
        else:
            prob_down = 0.5 + 0.5 * strength
            prob_up = 1.0 - prob_down

        state = {"signal": signal, "strength": strength, "confirmation": confirmation}
        if pattern_context:
            state["pattern"] = _pattern_state_summary(pattern_context)

        return ModelOutput(
            model_name=self.name,
            prob_up=prob_up,
            prob_down=prob_down,
            state=state,
            metrics={"ma_delta": ma_delta, "body_ratio": body_ratio},
        )


class Oscillator(BaseModel):
    """RSI-based mean reversion model with momentum confirmation.

    Uses RSI deviation from 50 combined with RSI momentum:
    - RSI < 50 AND rising -> strong UP signal (reversal starting)
    - RSI > 50 AND falling -> strong DOWN signal (reversal starting)
    - RSI momentum aligned with position -> weaker signal (continuation)
    """

    def __init__(self, config: ConfigStore) -> None:
        self.name = "OSCILLATOR"
        self._config = config

    def predict(
        self,
        features: Dict[str, float],
        pattern_context: Optional[PatternContext] = None,
    ) -> ModelOutput:
        rsi = features.get("rsi", 50.0)
        rsi_momentum = features.get("rsi_momentum", 0.0)
        vol_z = features.get("volatility_z", 0.0)

        # Distance from equilibrium (50)
        distance_from_50 = abs(rsi - 50.0)

        # Nonlinear strength scaling
        # Extreme RSI (near 0 or 100) = stronger signal
        if distance_from_50 < 10:
            # Near 50 - weak signal
            base_strength = distance_from_50 / 100.0  # 0.0 - 0.10
        elif distance_from_50 < 20:
            # Moderate deviation
            base_strength = 0.10 + (distance_from_50 - 10) / 50.0  # 0.10 - 0.30
        else:
            # Extreme RSI (< 30 or > 70)
            base_strength = 0.30 + (distance_from_50 - 20) / 100.0  # 0.30 - 0.50

        # RSI momentum confirmation/contradiction
        # Key insight: mean reversion works best when RSI is ALREADY reversing
        momentum_factor = 1.0
        if rsi < 50:
            # RSI oversold zone - expect UP
            if rsi_momentum > 0.5:
                # RSI is rising = reversal starting = STRONG signal
                momentum_factor = 1.3
            elif rsi_momentum < -0.5:
                # RSI still falling = not reversing yet = WEAK signal
                momentum_factor = 0.6
        else:
            # RSI overbought zone - expect DOWN
            if rsi_momentum < -0.5:
                # RSI is falling = reversal starting = STRONG signal
                momentum_factor = 1.3
            elif rsi_momentum > 0.5:
                # RSI still rising = not reversing yet = WEAK signal
                momentum_factor = 0.6

        # Apply momentum factor
        strength = base_strength * momentum_factor

        # Volatility penalty - mean reversion works worse in high volatility
        if vol_z > 1.5:
            strength *= 0.6
        elif vol_z > 1.0:
            strength *= 0.8

        # Clamp strength
        strength = clamp(strength, 0.05, 0.50)

        # Mean reversion direction
        if rsi <= 50:
            prob_up = 0.5 + strength
            prob_down = 0.5 - strength
            signal = "up"
        else:
            prob_up = 0.5 - strength
            prob_down = 0.5 + strength
            signal = "down"

        direction = "UP" if prob_up >= prob_down else "DOWN"
        strength = _apply_pattern_context_strength(
            strength,
            direction,
            pattern_context,
            feature_keys=("rsi", "rsi_momentum", "volatility_z"),
        )
        if direction == "UP":
            prob_up = 0.5 + strength
            prob_down = 0.5 - strength
        else:
            prob_up = 0.5 - strength
            prob_down = 0.5 + strength

        state = {
            "signal": signal,
            "strength": strength,
            "momentum_factor": momentum_factor,
        }
        if pattern_context:
            state["pattern"] = _pattern_state_summary(pattern_context)

        return ModelOutput(
            model_name=self.name,
            prob_up=prob_up,
            prob_down=prob_down,
            state=state,
            metrics={"rsi": rsi, "rsi_momentum": rsi_momentum},
        )


class VolumeMetrix(BaseModel):
    """Volume-Price relationship model with wick and trend analysis.

    Analyzes the relationship between volume and price movement:
    - High volume + big move = CONTINUATION (trend following)
    - High volume + small move = ABSORPTION (potential reversal)
    - Low volume = follow the trend with low confidence
    - Wick analysis for rejection signals
    """

    def __init__(self, config: ConfigStore) -> None:
        self.name = "VOLUMEMETRIX"
        self._config = config

    def predict(
        self,
        features: Dict[str, float],
        pattern_context: Optional[PatternContext] = None,
    ) -> ModelOutput:
        volume_z = features.get("volume_z", 0.0)
        ret = features.get("return_1", 0.0)
        volatility = features.get("volatility", 0.001)
        ma_delta = features.get("ma_delta", 0.0)
        volume_trend = features.get("volume_trend", 0.0)
        upper_wick_ratio = features.get("upper_wick_ratio", 0.0)
        lower_wick_ratio = features.get("lower_wick_ratio", 0.0)

        # Normalize return by volatility to get "relative" move size
        ret_z = safe_div(abs(ret), volatility, 0.0)

        # Determine volume-price pattern
        pattern = "normal"
        direction = "UP" if ret >= 0 else "DOWN"
        strength = 0.10  # default

        if volume_z > 1.5:
            # HIGH VOLUME scenarios
            if ret_z > 1.0:
                # High volume + big move = BREAKOUT/CONTINUATION
                pattern = "continuation"
                strength = clamp((volume_z + ret_z) / 8.0, 0.15, 0.40)
            else:
                # High volume + small move = ABSORPTION
                pattern = "absorption"
                strength = clamp(volume_z / 6.0, 0.10, 0.30)
                direction = "DOWN" if ret >= 0 else "UP"
        elif volume_z < -0.5:
            # LOW VOLUME - weak signal, follow trend
            pattern = "low_volume"
            strength = 0.05
            if ma_delta != 0:
                direction = "UP" if ma_delta > 0 else "DOWN"
        else:
            # NORMAL VOLUME - moderate signal from price action
            pattern = "normal"
            strength = clamp(ret_z / 6.0, 0.05, 0.20)

        # Sprint 10: Volume trend confirmation
        # Rising volume = strengthening move, falling volume = weakening
        if volume_trend > 0.2 and pattern == "continuation":
            strength = min(strength * 1.15, 0.45)  # Rising volume confirms
        elif volume_trend < -0.2 and pattern == "continuation":
            strength *= 0.85  # Falling volume = weakening trend

        # Sprint 10: Wick rejection analysis
        # Long upper wick (>40%) suggests selling pressure = bearish
        # Long lower wick (>40%) suggests buying pressure = bullish
        if upper_wick_ratio > 0.4:
            if direction == "UP":
                strength *= 0.8  # Upper wick contradicts UP
            else:
                strength = min(strength * 1.1, 0.45)  # Confirms DOWN

        if lower_wick_ratio > 0.4:
            if direction == "DOWN":
                strength *= 0.8  # Lower wick contradicts DOWN
            else:
                strength = min(strength * 1.1, 0.45)  # Confirms UP

        # Trend alignment bonus
        trend_aligned = (direction == "UP" and ma_delta > 0) or \
                       (direction == "DOWN" and ma_delta < 0)
        if trend_aligned:
            strength = min(strength * 1.2, 0.45)

        # Ensure minimum strength to avoid FLAT (threshold = 0.55)
        # prob = 0.5 + strength, need prob >= 0.55, so strength >= 0.05
        strength = max(strength, 0.05)

        # Build probabilities
        if direction == "UP":
            prob_up = 0.5 + strength
            prob_down = 0.5 - strength
            signal = "up"
        else:
            prob_up = 0.5 - strength
            prob_down = 0.5 + strength
            signal = "down"

        direction_label = "UP" if prob_up >= prob_down else "DOWN"
        strength = _apply_pattern_context_strength(
            strength,
            direction_label,
            pattern_context,
            feature_keys=("volume_z", "volatility_z", "body_ratio"),
        )
        if direction_label == "UP":
            prob_up = 0.5 + strength
            prob_down = 0.5 - strength
        else:
            prob_up = 0.5 - strength
            prob_down = 0.5 + strength

        state = {"signal": signal, "strength": strength, "pattern": pattern}
        if pattern_context:
            state["pattern"] = _pattern_state_summary(pattern_context)

        return ModelOutput(
            model_name=self.name,
            prob_up=prob_up,
            prob_down=prob_down,
            state=state,
            metrics={
                "volume_z": volume_z,
                "return_1": ret,
                "ret_z": ret_z,
                "trend_aligned": 1.0 if trend_aligned else 0.0,
            },
        )
