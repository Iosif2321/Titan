"""Volatile regime strategy.

Volatile regime has 55-56% error rate - the worst of all regimes.
This module provides specialized handling for volatile market conditions.
"""

from typing import Dict, Optional

from titan.core.config import ConfigStore
from titan.core.types import Decision


class VolatileClassifier:
    """Classifies different types of volatile conditions."""

    TYPES = ["volatile_extreme", "volatile_breakout", "volatile_choppy", "volatile_reversal"]

    def __init__(self, config: ConfigStore) -> None:
        self._config = config

    def classify(self, features: Dict[str, float]) -> str:
        """Classify the type of volatile condition.

        Returns:
            One of: volatile_extreme, volatile_breakout, volatile_choppy, volatile_reversal
        """
        vol_z = features.get("volatility_z", 0.0)
        volume_z = features.get("volume_z", 0.0)
        rsi = features.get("rsi", 50.0)
        ret = features.get("return_1", 0.0)

        # Extreme volatility - very dangerous, almost random
        extreme_threshold = float(self._config.get("volatile.extreme_threshold", 2.5))
        if vol_z > extreme_threshold:
            return "volatile_extreme"

        # High vol + high volume + big move = breakout
        breakout_vol_threshold = float(self._config.get("volatile.breakout_vol_threshold", 1.5))
        breakout_volume_threshold = float(self._config.get("volatile.breakout_volume_threshold", 1.0))
        if vol_z > breakout_vol_threshold and volume_z > breakout_volume_threshold:
            if abs(ret) > 0.0005:  # Significant move
                return "volatile_breakout"

        # High vol + extreme RSI = potential reversal
        rsi_extreme_high = float(self._config.get("volatile.rsi_extreme_high", 70))
        rsi_extreme_low = float(self._config.get("volatile.rsi_extreme_low", 30))
        if vol_z > 1.5 and (rsi > rsi_extreme_high or rsi < rsi_extreme_low):
            return "volatile_reversal"

        # High vol + neutral RSI = choppy/ranging
        return "volatile_choppy"


class VolatileStrategy:
    """Special strategy for volatile market conditions.

    In volatile conditions:
    1. Reduce confidence significantly
    2. Use mean-reversion for extreme RSI
    3. Follow breakouts with caution
    4. Avoid trading in choppy/extreme conditions
    """

    def __init__(self, config: ConfigStore) -> None:
        self._config = config
        self._classifier = VolatileClassifier(config)

    def _enabled(self) -> bool:
        return bool(self._config.get("volatile_strategy.enabled", True))

    def decide(self, features: Dict[str, float]) -> Optional[Decision]:
        """Make decision for volatile regime.

        Returns:
            Decision with adjusted confidence, or None to use default ensemble.
        """
        if not self._enabled():
            return None

        vol_type = self._classifier.classify(features)
        rsi = features.get("rsi", 50.0)
        ret = features.get("return_1", 0.0)
        ma_delta = features.get("ma_delta", 0.0)
        close = features.get("close", 1.0)

        if vol_type == "volatile_extreme":
            # Extreme volatility - near random, give very weak signal
            # Use last return direction with minimal confidence
            direction = "UP" if ret > 0 else "DOWN"
            return Decision(
                direction=direction,
                confidence=0.505,  # Almost 50-50
                prob_up=0.505 if direction == "UP" else 0.495,
                prob_down=0.495 if direction == "UP" else 0.505,
            )

        if vol_type == "volatile_breakout":
            # Breakout - follow the move direction with moderate confidence
            direction = "UP" if ret > 0 else "DOWN"
            # Slightly higher confidence for breakouts
            return Decision(
                direction=direction,
                confidence=0.54,
                prob_up=0.54 if direction == "UP" else 0.46,
                prob_down=0.46 if direction == "UP" else 0.54,
            )

        if vol_type == "volatile_reversal":
            # Potential reversal - use mean reversion
            if rsi > 70:
                # Overbought - expect DOWN
                strength = min((rsi - 70) / 60, 0.08)  # max +8% confidence
                return Decision(
                    direction="DOWN",
                    confidence=0.50 + strength,
                    prob_up=0.50 - strength,
                    prob_down=0.50 + strength,
                )
            elif rsi < 30:
                # Oversold - expect UP
                strength = min((30 - rsi) / 60, 0.08)
                return Decision(
                    direction="UP",
                    confidence=0.50 + strength,
                    prob_up=0.50 + strength,
                    prob_down=0.50 - strength,
                )

        # volatile_choppy - no clear signal, use weak trend following
        normalized_delta = ma_delta / (close + 1e-12)
        if abs(normalized_delta) > 0.0001:
            direction = "UP" if ma_delta > 0 else "DOWN"
        else:
            direction = "UP" if ret > 0 else "DOWN"

        return Decision(
            direction=direction,
            confidence=0.51,  # Very low confidence
            prob_up=0.51 if direction == "UP" else 0.49,
            prob_down=0.49 if direction == "UP" else 0.51,
        )

    def get_volatile_type(self, features: Dict[str, float]) -> str:
        """Get the classified volatile type for logging."""
        return self._classifier.classify(features)
