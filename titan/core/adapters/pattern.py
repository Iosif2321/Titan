"""Pattern-based decision adjustment using historical experience.

Sprint 13: Uses accumulated pattern history to adjust confidence
based on how well each pattern has performed historically.
"""

from typing import TYPE_CHECKING, Dict, Optional

from titan.core.config import ConfigStore
from titan.core.types import Decision
from titan.core.utils import clamp

if TYPE_CHECKING:
    from titan.core.patterns import PatternExperience


class PatternAdjuster:
    """Adjusts predictions based on historical pattern performance.

    Uses PatternExperience to:
    1. Boost confidence for historically accurate patterns
    2. Reduce confidence for historically inaccurate patterns
    3. Consider directional bias (some patterns work better for UP/DOWN)
    """

    def __init__(
        self,
        experience: "PatternExperience",
        config: ConfigStore,
        min_uses: int = 20,
    ) -> None:
        """Initialize pattern adjuster.

        Args:
            experience: Pattern experience analyzer
            config: Configuration store
            min_uses: Minimum pattern uses before adjusting
        """
        self._exp = experience
        self._config = config
        self._min_uses = min_uses

    def adjust_decision(
        self,
        decision: Decision,
        pattern_id: int,
        features: Optional[Dict[str, float]] = None,
    ) -> Decision:
        """Adjust decision based on pattern's historical performance.

        Args:
            decision: The original ensemble decision
            pattern_id: ID of the current market pattern
            features: Current market features (optional, for future use)

        Returns:
            Adjusted decision with modified confidence
        """
        # Check if we have enough data for this pattern
        if not self._exp.should_trust_pattern(pattern_id, self._min_uses):
            return decision  # Not enough data

        stats = self._exp.get_pattern_stats(pattern_id)
        pattern_acc = stats["accuracy"]
        estimate_confidence = stats["confidence"]

        # Get config parameters
        boost_threshold = float(
            self._config.get("pattern.boost_threshold", 0.55)
        )
        penalty_threshold = float(
            self._config.get("pattern.penalty_threshold", 0.45)
        )
        max_boost = float(self._config.get("pattern.max_boost", 0.03))
        max_penalty = float(self._config.get("pattern.max_penalty", 0.03))

        new_conf = decision.confidence

        # Adjust based on pattern accuracy
        if pattern_acc > boost_threshold:
            # Pattern is historically accurate - boost confidence
            # Scale boost by how much above threshold and by estimate confidence
            boost = (pattern_acc - 0.50) * estimate_confidence * 0.5
            boost = min(boost, max_boost)
            new_conf = min(decision.confidence + boost, 0.65)

        elif pattern_acc < penalty_threshold:
            # Pattern is historically inaccurate - reduce confidence
            penalty = (0.50 - pattern_acc) * estimate_confidence * 0.5
            penalty = min(penalty, max_penalty)
            new_conf = max(decision.confidence - penalty, 0.50)

        # Check for directional bias
        bias = self._exp.get_pattern_bias(pattern_id)
        if bias and bias != decision.direction:
            # Pattern historically works better in opposite direction
            # Apply small penalty (but don't flip the decision)
            bias_penalty = float(
                self._config.get("pattern.bias_penalty", 0.01)
            )
            new_conf = max(new_conf - bias_penalty, 0.50)

        return Decision(
            direction=decision.direction,
            confidence=clamp(new_conf, 0.0, 1.0),
            prob_up=decision.prob_up,
            prob_down=decision.prob_down,
        )

    def get_pattern_summary(self, pattern_id: int) -> Dict:
        """Get a summary of pattern performance for logging/debugging."""
        stats = self._exp.get_pattern_stats(pattern_id)
        bias = self._exp.get_pattern_bias(pattern_id)
        trusted = self._exp.should_trust_pattern(pattern_id, self._min_uses)

        return {
            "pattern_id": pattern_id,
            "total_uses": stats["total_uses"],
            "accuracy": stats["accuracy"],
            "up_accuracy": stats["up_accuracy"],
            "down_accuracy": stats["down_accuracy"],
            "estimate_confidence": stats["confidence"],
            "bias": bias,
            "trusted": trusted,
        }
