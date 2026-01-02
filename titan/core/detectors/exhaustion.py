"""Trend exhaustion detection.

Trending_down regime has 53-54% error rate - models predict continuation
but reversals happen. This module detects when a trend is likely exhausted.
"""

from collections import deque
from typing import Dict, Optional, Tuple

from titan.core.config import ConfigStore


class TrendExhaustionDetector:
    """Detects trend exhaustion for reversal prediction.

    Exhaustion signs:
    1. Momentum slowing (price change rate decreasing)
    2. Volume declining (less interest)
    3. RSI divergence (price makes new extreme but RSI doesn't)
    4. Price near support/resistance levels
    """

    def __init__(self, config: ConfigStore, lookback: int = 10) -> None:
        self._config = config
        self._lookback = lookback
        self._price_history: deque = deque(maxlen=lookback)
        self._volume_history: deque = deque(maxlen=lookback)
        self._return_history: deque = deque(maxlen=lookback)
        self._rsi_history: deque = deque(maxlen=lookback)

    def update(
        self,
        close: float,
        volume: float,
        ret: float,
        rsi: float,
    ) -> None:
        """Update with new candle data."""
        self._price_history.append(close)
        self._volume_history.append(volume)
        self._return_history.append(ret)
        self._rsi_history.append(rsi)

    def _momentum_exhaustion(self) -> float:
        """Check if momentum is declining.

        Returns score 0.0-1.0 where higher = more exhausted.
        """
        if len(self._return_history) < 6:
            return 0.0

        returns = list(self._return_history)

        # Compare recent momentum vs earlier momentum
        recent_momentum = sum(abs(r) for r in returns[-3:]) / 3
        earlier_momentum = sum(abs(r) for r in returns[-6:-3]) / 3

        if earlier_momentum < 1e-10:
            return 0.0

        # If recent momentum is less than half of earlier, that's exhaustion
        ratio = recent_momentum / earlier_momentum
        if ratio < 0.5:
            return 0.7  # Strong exhaustion
        elif ratio < 0.7:
            return 0.4  # Moderate exhaustion
        elif ratio < 0.9:
            return 0.2  # Slight exhaustion

        return 0.0

    def _volume_exhaustion(self) -> float:
        """Check if volume is declining.

        Returns score 0.0-1.0 where higher = more exhausted.
        """
        if len(self._volume_history) < 6:
            return 0.0

        volumes = list(self._volume_history)

        # Compare recent volume vs earlier volume
        recent_vol = sum(volumes[-3:]) / 3
        earlier_vol = sum(volumes[-6:-3]) / 3

        if earlier_vol < 1e-10:
            return 0.0

        # If recent volume is less than 70% of earlier, that's exhaustion
        ratio = recent_vol / earlier_vol
        if ratio < 0.5:
            return 0.5  # Strong volume decline
        elif ratio < 0.7:
            return 0.3  # Moderate decline
        elif ratio < 0.85:
            return 0.1  # Slight decline

        return 0.0

    def _rsi_divergence(self, regime: str) -> float:
        """Check for RSI divergence.

        In downtrend: price makes lower low but RSI makes higher low = bullish divergence
        In uptrend: price makes higher high but RSI makes lower high = bearish divergence

        Returns score 0.0-1.0 where higher = more divergence.
        """
        if len(self._rsi_history) < 6 or len(self._price_history) < 6:
            return 0.0

        prices = list(self._price_history)
        rsis = list(self._rsi_history)

        if regime == "trending_down":
            # Check for bullish divergence
            # Price making lower low?
            price_lower = prices[-1] < min(prices[-4:-1])
            # RSI making higher low?
            rsi_higher = rsis[-1] > min(rsis[-4:-1])

            if price_lower and rsi_higher:
                return 0.4  # Bullish divergence detected

        elif regime == "trending_up":
            # Check for bearish divergence
            price_higher = prices[-1] > max(prices[-4:-1])
            rsi_lower = rsis[-1] < max(rsis[-4:-1])

            if price_higher and rsi_lower:
                return 0.4  # Bearish divergence detected

        return 0.0

    def _rsi_extreme(self, rsi: float, regime: str) -> float:
        """Check if RSI is at extreme levels.

        Returns score 0.0-1.0.
        """
        if regime == "trending_down":
            # In downtrend, oversold RSI suggests reversal
            if rsi < 20:
                return 0.5
            elif rsi < 30:
                return 0.3
            elif rsi < 40:
                return 0.1

        elif regime == "trending_up":
            # In uptrend, overbought RSI suggests reversal
            if rsi > 80:
                return 0.5
            elif rsi > 70:
                return 0.3
            elif rsi > 60:
                return 0.1

        return 0.0

    def is_exhausted(
        self,
        features: Dict[str, float],
        regime: str,
    ) -> Tuple[bool, float, str]:
        """Check if trend is exhausted.

        Args:
            features: Current market features
            regime: Current regime (trending_up or trending_down)

        Returns:
            Tuple of (is_exhausted, exhaustion_score, reason)
        """
        if regime not in ("trending_up", "trending_down"):
            return False, 0.0, "not_trending"

        rsi = features.get("rsi", 50.0)

        # Calculate exhaustion scores
        momentum_score = self._momentum_exhaustion()
        volume_score = self._volume_exhaustion()
        divergence_score = self._rsi_divergence(regime)
        extreme_score = self._rsi_extreme(rsi, regime)

        # Weighted combination
        total_score = (
            momentum_score * 0.3
            + volume_score * 0.2
            + divergence_score * 0.25
            + extreme_score * 0.25
        )

        # Determine reason
        reasons = []
        if momentum_score > 0.3:
            reasons.append("momentum_slowing")
        if volume_score > 0.2:
            reasons.append("volume_declining")
        if divergence_score > 0:
            reasons.append("rsi_divergence")
        if extreme_score > 0.2:
            reasons.append("rsi_extreme")

        reason = "+".join(reasons) if reasons else "none"

        # Exhaustion threshold
        threshold = float(self._config.get("exhaustion.threshold", 0.35))
        is_exhausted = total_score >= threshold

        return is_exhausted, total_score, reason

    def predict_reversal(
        self,
        features: Dict[str, float],
        regime: str,
    ) -> Optional[str]:
        """Predict if reversal is likely.

        Returns:
            'UP' if reversal up expected, 'DOWN' if reversal down expected,
            None if no reversal predicted.
        """
        is_exhausted, score, reason = self.is_exhausted(features, regime)

        if not is_exhausted:
            return None

        # Predict reversal direction opposite to current trend
        if regime == "trending_down":
            return "UP"  # Exhausted downtrend -> expect UP
        elif regime == "trending_up":
            return "DOWN"  # Exhausted uptrend -> expect DOWN

        return None
