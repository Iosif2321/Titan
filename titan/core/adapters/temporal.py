"""Temporal adjustment module.

Adjusts predictions based on time patterns observed in historical data.
Some hours consistently have worse accuracy than others.
"""

from datetime import datetime, timezone
from typing import Optional

from titan.core.config import ConfigStore


class TemporalAdjuster:
    """Adjusts confidence based on time of day.

    Historical analysis shows some hours have significantly worse accuracy:
    - Worst hours: 7:00, 12:00, 19:00 UTC (~40-42% accuracy)
    - Best hours: 3:00, 8:00, 20:00, 21:00, 22:00 UTC (~58-60% accuracy)
    """

    # Confidence multipliers based on historical accuracy
    # Hours with <45% accuracy get penalty, >55% get bonus
    HOURLY_MULTIPLIERS = {
        0: 0.95,   # ~45% historically
        1: 0.97,
        2: 0.95,   # ~45%
        3: 1.05,   # ~60% - BEST
        4: 0.95,   # ~45%
        5: 1.00,
        6: 1.02,
        7: 0.88,   # ~40% - WORST
        8: 1.05,   # ~58%
        9: 1.02,
        10: 1.00,
        11: 1.00,
        12: 0.90,  # ~42% - BAD
        13: 0.92,  # ~43%
        14: 0.97,
        15: 0.97,
        16: 1.00,
        17: 0.92,  # ~43%
        18: 1.00,
        19: 0.88,  # ~40% - WORST
        20: 1.05,  # ~58%
        21: 1.05,  # ~58%
        22: 1.03,  # ~57%
        23: 0.97,
    }

    # Danger hours where we should be extra cautious
    DANGER_HOURS = {7, 12, 19}

    # Best hours where we can be more confident
    BEST_HOURS = {3, 8, 20, 21, 22}

    def __init__(self, config: ConfigStore) -> None:
        self._config = config

    def _enabled(self) -> bool:
        return bool(self._config.get("temporal_adjuster.enabled", True))

    def get_hour_utc(self, ts: Optional[int] = None) -> int:
        """Get hour in UTC from timestamp or current time."""
        if ts:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        else:
            dt = datetime.now(timezone.utc)
        return dt.hour

    def get_confidence_multiplier(self, ts: Optional[int] = None) -> float:
        """Get confidence multiplier for the given timestamp."""
        if not self._enabled():
            return 1.0

        hour = self.get_hour_utc(ts)
        return self.HOURLY_MULTIPLIERS.get(hour, 1.0)

    def is_danger_hour(self, ts: Optional[int] = None) -> bool:
        """Check if current hour is historically dangerous."""
        hour = self.get_hour_utc(ts)
        return hour in self.DANGER_HOURS

    def is_best_hour(self, ts: Optional[int] = None) -> bool:
        """Check if current hour is historically good."""
        hour = self.get_hour_utc(ts)
        return hour in self.BEST_HOURS

    def adjust_confidence(
        self,
        confidence: float,
        ts: Optional[int] = None,
    ) -> float:
        """Adjust confidence based on time of day.

        Args:
            confidence: Base confidence value
            ts: Unix timestamp (uses current time if None)

        Returns:
            Adjusted confidence, clamped to [0.50, 1.0]
        """
        if not self._enabled():
            return confidence

        multiplier = self.get_confidence_multiplier(ts)
        adjusted = confidence * multiplier

        # Don't let confidence drop below 50%
        return max(0.50, min(adjusted, 1.0))

    def should_reduce_position(self, ts: Optional[int] = None) -> bool:
        """Check if we should reduce position size for live trading.

        Returns True during danger hours where accuracy is historically bad.
        """
        return self.is_danger_hour(ts)
