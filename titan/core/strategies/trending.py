"""Trending regime strategy.

Trending_down regime has 53-54% error rate - models predict continuation
but reversals happen. This strategy uses exhaustion detection to predict reversals.
"""

from typing import Dict, Optional

from titan.core.config import ConfigStore
from titan.core.detectors.exhaustion import TrendExhaustionDetector
from titan.core.types import Decision


class TrendingStrategy:
    """Strategy for trending market conditions.

    Uses trend exhaustion detection to predict when reversals are likely.
    When exhaustion is detected, reduces confidence in continuation or
    predicts reversal.
    """

    def __init__(self, config: ConfigStore) -> None:
        self._config = config
        self._exhaustion = TrendExhaustionDetector(config)

    def _enabled(self) -> bool:
        return bool(self._config.get("trending_strategy.enabled", True))

    def update(self, features: Dict[str, float]) -> None:
        """Update internal state with new features."""
        close = features.get("close", 0.0)
        volume = features.get("volume", 0.0)
        ret = features.get("return_1", 0.0)
        rsi = features.get("rsi", 50.0)

        self._exhaustion.update(close, volume, ret, rsi)

    def decide(
        self,
        features: Dict[str, float],
        regime: str,
        base_direction: str,
    ) -> Optional[Decision]:
        """Make decision for trending regime.

        Args:
            features: Current market features
            regime: Current regime (trending_up or trending_down)
            base_direction: Direction from base ensemble

        Returns:
            Modified decision if exhaustion detected, None otherwise.
        """
        if not self._enabled():
            return None

        if regime not in ("trending_up", "trending_down"):
            return None

        # Update exhaustion detector
        self.update(features)

        # Check for exhaustion
        is_exhausted, score, reason = self._exhaustion.is_exhausted(features, regime)

        rsi = features.get("rsi", 50.0)
        ma_delta = features.get("ma_delta", 0.0)

        if is_exhausted:
            # Trend is exhausted - predict reversal
            reversal_dir = self._exhaustion.predict_reversal(features, regime)

            if reversal_dir:
                # Confidence based on exhaustion score
                # Higher exhaustion = higher confidence in reversal
                confidence = 0.50 + min(score * 0.15, 0.08)  # max 58%

                return Decision(
                    direction=reversal_dir,
                    confidence=confidence,
                    prob_up=confidence if reversal_dir == "UP" else 1 - confidence,
                    prob_down=confidence if reversal_dir == "DOWN" else 1 - confidence,
                )

        # Check for extreme RSI without full exhaustion
        if regime == "trending_down" and rsi < 25:
            # Extremely oversold in downtrend - weak reversal signal
            return Decision(
                direction="UP",
                confidence=0.52,
                prob_up=0.52,
                prob_down=0.48,
            )
        elif regime == "trending_up" and rsi > 75:
            # Extremely overbought in uptrend - weak reversal signal
            return Decision(
                direction="DOWN",
                confidence=0.52,
                prob_up=0.48,
                prob_down=0.52,
            )

        # No exhaustion - let base ensemble decide
        # But reduce confidence in trending_down since it's error-prone
        if regime == "trending_down":
            # Trending down is problematic - lower confidence
            # Follow the trend direction but with reduced confidence
            trend_dir = "DOWN" if ma_delta < 0 else base_direction

            return Decision(
                direction=trend_dir,
                confidence=0.51,  # Low confidence
                prob_up=0.49 if trend_dir == "DOWN" else 0.51,
                prob_down=0.51 if trend_dir == "DOWN" else 0.49,
            )

        return None  # Use base ensemble for trending_up
