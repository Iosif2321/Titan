"""Pattern-based decision adjustment using historical experience.

Sprint 13: Uses accumulated pattern history to adjust confidence
based on how well each pattern has performed historically.

Sprint 12 (Enhanced): PatternReader for fuzzy search and advanced insights.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from titan.core.config import ConfigStore
from titan.core.types import Decision, ModelOutput, PatternContext
from titan.core.utils import clamp

if TYPE_CHECKING:
    from titan.core.patterns import (
        ExtendedConditions,
        PatternAggregates,
        PatternExperience,
        PatternMatch,
        PatternStore,
    )


@dataclass
class PatternInsight:
    """Actionable insight for model decision-making."""
    pattern_id: int
    pattern_key: str
    model_name: Optional[str]
    is_trustworthy: bool
    trust_confidence: float

    # Directional insight
    has_directional_bias: bool
    favored_direction: Optional[str]
    bias_strength: float

    # Confidence insight
    is_overconfident: bool
    suggested_confidence_cap: Optional[float]

    # Accuracy insight
    accuracy: float
    up_accuracy: float
    down_accuracy: float

    # Status
    status: str
    match_ratio: float  # How well this pattern matches current conditions


@dataclass
class AdjustmentRecommendation:
    """Recommended adjustments to a proposed decision."""
    original_decision: str
    original_confidence: float
    recommended_decision: str
    recommended_confidence: float
    confidence_delta: float
    adjustment_reason: str
    pattern_accuracy: float
    pattern_bias: Optional[str]
    trust_level: float
    decision_changed: bool
    confidence_reduced: bool
    confidence_increased: bool


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
        model_name: Optional[str] = None,
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
        self._model_name = model_name

    def adjust_decision(
        self,
        decision: Decision,
        pattern_id: int,
        features: Optional[Dict[str, float]] = None,
        max_ts: Optional[int] = None,
    ) -> Decision:
        """Adjust decision based on pattern's historical performance.

        Args:
            decision: The original ensemble decision
            pattern_id: ID of the current market pattern
            features: Current market features (optional, for future use)
            max_ts: Optional timestamp ceiling to prevent data leakage in backtest

        Returns:
            Adjusted decision with modified confidence
        """
        # Check if we have enough data for this pattern
        # FIXED: Pass max_ts to prevent data leakage
        if not self._exp.should_trust_pattern(
            pattern_id,
            self._min_uses,
            max_ts=max_ts,
            model_name=self._model_name,
        ):
            return decision  # Not enough data

        stats = self._exp.get_pattern_stats(
            pattern_id, max_ts=max_ts, model_name=self._model_name
        )
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
        bias = self._exp.get_pattern_bias(
            pattern_id, max_ts=max_ts, model_name=self._model_name
        )
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
        stats = self._exp.get_pattern_stats(pattern_id, model_name=self._model_name)
        bias = self._exp.get_pattern_bias(pattern_id, model_name=self._model_name)
        trusted = self._exp.should_trust_pattern(
            pattern_id, self._min_uses, model_name=self._model_name
        )

        return {
            "pattern_id": pattern_id,
            "model_name": self._model_name,
            "total_uses": stats["total_uses"],
            "accuracy": stats["accuracy"],
            "up_accuracy": stats["up_accuracy"],
            "down_accuracy": stats["down_accuracy"],
            "estimate_confidence": stats["confidence"],
            "bias": bias,
            "trusted": trusted,
        }


class PatternModelAdjuster:
    """Adjust model outputs using pattern history (per-model)."""

    def __init__(
        self,
        reader: "PatternReader",
        config: ConfigStore,
        min_uses: int = 20,
    ) -> None:
        self._reader = reader
        self._config = config
        self._min_uses = min_uses
        self._min_match = float(config.get("pattern.min_match_ratio", 0.8))

    def adjust_model_output(
        self,
        output: ModelOutput,
        pattern_id: int,
        conditions: Optional[Dict[str, str]] = None,
        ts: Optional[int] = None,
    ) -> ModelOutput:
        """Adjust a model output using pattern history."""
        proposed_decision = "UP" if output.prob_up >= output.prob_down else "DOWN"
        proposed_conf = max(output.prob_up, output.prob_down)

        match_ratio = 1.0
        insight = self._reader.get_pattern_insight(
            pattern_id, model_name=output.model_name
        )

        if insight is None or not insight.is_trustworthy:
            if conditions and ts is not None:
                from titan.core.patterns import build_extended_conditions

                ext = build_extended_conditions(conditions, ts)
                matches = self._reader.find_matching_patterns(
                    ext, min_match=self._min_match
                )
                best = self._reader.get_best_match(matches, model_name=output.model_name)
                if best:
                    match_ratio = best.match_ratio
                    insight = self._reader.get_pattern_insight(
                        best.pattern_id,
                        match_ratio=match_ratio,
                        model_name=output.model_name,
                    )

        if insight is None:
            return output

        rec = self._reader.get_recommended_adjustment(
            insight.pattern_id,
            proposed_decision=proposed_decision,
            proposed_confidence=proposed_conf,
            match_ratio=match_ratio,
            model_name=output.model_name,
        )

        if (
            rec.recommended_decision == proposed_decision
            and abs(rec.confidence_delta) < 1e-9
        ):
            return output

        new_conf = clamp(rec.recommended_confidence, 0.5, 1.0)
        if rec.recommended_decision == "UP":
            prob_up = new_conf
            prob_down = 1.0 - new_conf
        else:
            prob_down = new_conf
            prob_up = 1.0 - new_conf

        state = dict(output.state)
        state["pattern_adjustment"] = {
            "pattern_id": insight.pattern_id,
            "pattern_key": insight.pattern_key,
            "match_ratio": match_ratio,
            "model_name": output.model_name,
            "pattern_accuracy": insight.accuracy,
            "pattern_bias": insight.favored_direction,
            "trust_confidence": insight.trust_confidence,
            "original_confidence": proposed_conf,
            "recommended_confidence": new_conf,
            "adjustment_reason": rec.adjustment_reason,
        }

        return ModelOutput(
            model_name=output.model_name,
            prob_up=clamp(prob_up, 0.0, 1.0),
            prob_down=clamp(prob_down, 0.0, 1.0),
            state=state,
            metrics=output.metrics,
        )


class PatternReader:
    """Read-only interface for models to query pattern history.

    Sprint 12: Enhanced pattern system with fuzzy search and actionable insights.

    Models use this to:
    - Find similar patterns using fuzzy matching (80% condition match)
    - Query what worked historically in this regime
    - See which confidence levels were accurate vs overconfident
    - Detect if pattern favors UP or DOWN
    - Get recommended adjustments for their decisions
    """

    def __init__(
        self,
        store: "PatternStore",
        config: ConfigStore,
        min_uses: int = 20,
    ) -> None:
        """Initialize pattern reader.

        Args:
            store: Pattern store with historical data
            config: Configuration store
            min_uses: Minimum uses before trusting a pattern
        """
        self._store = store
        self._config = config
        self._min_uses = min_uses
        self._min_match_ratio = float(config.get("pattern.min_match_ratio", 0.8))

    def build_context(
        self,
        pattern_id: int,
        features: Dict[str, float],
        ts: int,
        model_name: str,
        conditions: Optional[Dict[str, str]] = None,
    ) -> Optional[PatternContext]:
        """Build a PatternContext for model-aware prediction."""
        match_ratio = 1.0
        insight = self.get_pattern_insight(pattern_id, model_name=model_name)
        target_id = pattern_id

        if insight is None or not insight.is_trustworthy:
            if conditions:
                from titan.core.patterns import build_extended_conditions

                ext = build_extended_conditions(conditions, ts)
                matches = self.find_matching_patterns(
                    ext, min_match=self._min_match_ratio
                )
                best = self.get_best_match(matches, model_name=model_name)
                if best:
                    match_ratio = best.match_ratio
                    target_id = best.pattern_id
                    insight = self.get_pattern_insight(
                        target_id,
                        match_ratio=match_ratio,
                        model_name=model_name,
                    )

        if insight is None:
            return None

        feature_insights: Dict[str, Dict[str, float]] = {}
        from titan.core.patterns import get_feature_bucket, FEATURE_BUCKETS, get_trading_session

        all_feature_stats = self._store.get_feature_bucket_stats_all(
            target_id, model_name
        )
        for feature in FEATURE_BUCKETS.keys():
            bucket = get_feature_bucket(feature, features.get(feature))
            if bucket is None:
                continue
            bucket_stats = all_feature_stats.get(feature, {}).get(bucket)
            if bucket_stats:
                feature_insights[feature] = {
                    "bucket": bucket,
                    "count": bucket_stats.get("count", 0.0),
                    "accuracy": bucket_stats.get("accuracy", 0.5),
                    "avg_return": bucket_stats.get("avg_return", 0.0),
                }

        temporal_insights: Dict[str, Dict[str, float]] = {}
        from datetime import datetime

        dt = datetime.utcfromtimestamp(ts)
        hour = dt.hour
        session = get_trading_session(hour)
        day_of_week = dt.weekday()

        hour_stats = self._store.get_temporal_stats_by_hour(
            target_id, model_name, hour
        )
        if hour_stats:
            temporal_insights["hour"] = {
                "count": hour_stats.get("count", 0.0),
                "accuracy": hour_stats.get("accuracy", 0.5),
                "avg_return": hour_stats.get("avg_return", 0.0),
            }

        session_stats = self._store.get_temporal_stats_by_session(
            target_id, model_name, session
        )
        if session_stats:
            temporal_insights["session"] = {
                "count": session_stats.get("count", 0.0),
                "accuracy": session_stats.get("accuracy", 0.5),
                "avg_return": session_stats.get("avg_return", 0.0),
            }

        day_stats = self._store.get_temporal_stats_by_day(
            target_id, model_name, day_of_week
        )
        if day_stats:
            temporal_insights["day_of_week"] = {
                "count": day_stats.get("count", 0.0),
                "accuracy": day_stats.get("accuracy", 0.5),
                "avg_return": day_stats.get("avg_return", 0.0),
            }

        return PatternContext(
            pattern_id=target_id,
            pattern_key=insight.pattern_key,
            model_name=model_name,
            match_ratio=match_ratio,
            accuracy=insight.accuracy,
            up_accuracy=insight.up_accuracy,
            down_accuracy=insight.down_accuracy,
            bias=insight.favored_direction,
            trust_confidence=insight.trust_confidence,
            overconfident=insight.is_overconfident,
            confidence_cap=insight.suggested_confidence_cap,
            feature_insights=feature_insights,
            temporal_insights=temporal_insights,
        )

    def find_matching_patterns(
        self,
        conditions: "ExtendedConditions",
        min_match: float = 0.8,
        limit: int = 5,
    ) -> List["PatternMatch"]:
        """Find patterns matching current conditions using fuzzy search.

        Args:
            conditions: Current market conditions
            min_match: Minimum match ratio (default 0.8 = 80%)
            limit: Maximum number of patterns to return

        Returns:
            List of PatternMatch sorted by match quality
        """
        matches = self._store.find_similar_patterns(
            conditions,
            min_match=min_match,
            include_inactive=True,
            limit=limit,
        )

        # Reactivate any inactive patterns that were found
        for match in matches:
            if match.status == "inactive":
                self._store.reactivate_pattern(match.pattern_id)
                match.status = "active"  # Update in result

        return matches

    def get_pattern_insight(
        self,
        pattern_id: int,
        match_ratio: float = 1.0,
        model_name: Optional[str] = None,
    ) -> Optional[PatternInsight]:
        """Get actionable insight for a pattern.

        Args:
            pattern_id: Target pattern
            match_ratio: How well this pattern matches (from fuzzy search)

        Returns:
            PatternInsight with recommendations, or None if pattern not found
        """
        aggregates = None
        if model_name:
            aggregates = self._store.get_model_aggregates(pattern_id, model_name)
        if not aggregates:
            aggregates = self._store.get_aggregates(pattern_id)
        if not aggregates:
            return None

        info = self._store.get_pattern_info(pattern_id)
        pattern_key = info["pattern_key"] if info else ""
        status = info["status"] if info else "active"

        # Determine if trustworthy
        is_trustworthy = aggregates.total_uses >= self._min_uses
        trust_confidence = min(aggregates.total_uses / 50.0, 1.0)

        # Determine directional bias
        up_acc = aggregates.up_accuracy
        down_acc = aggregates.down_accuracy
        bias_threshold = float(self._config.get("pattern.bias_threshold", 0.05))

        has_bias = abs(up_acc - down_acc) > bias_threshold and is_trustworthy
        favored_dir = None
        bias_strength = 0.0

        if has_bias:
            if up_acc > down_acc:
                favored_dir = "UP"
                bias_strength = up_acc - down_acc
            else:
                favored_dir = "DOWN"
                bias_strength = down_acc - up_acc

        # Check for overconfidence - only penalize if SIGNIFICANTLY worse than expected
        # conf_wrong_rate > 0.55 means >55% of high-conf predictions are wrong
        # This is bad even for a 50% accuracy pattern (would expect ~50% wrong)
        is_overconfident = (
            aggregates.conf_wrong_rate > 0.55
            and aggregates.high_conf_count >= 20
        )
        suggested_cap = None
        if is_overconfident:
            suggested_cap = 0.60  # Lower confidence cap for overconfident patterns

        return PatternInsight(
            pattern_id=pattern_id,
            pattern_key=pattern_key,
            model_name=model_name,
            is_trustworthy=is_trustworthy,
            trust_confidence=trust_confidence,
            has_directional_bias=has_bias,
            favored_direction=favored_dir,
            bias_strength=bias_strength,
            is_overconfident=is_overconfident,
            suggested_confidence_cap=suggested_cap,
            accuracy=aggregates.accuracy,
            up_accuracy=up_acc,
            down_accuracy=down_acc,
            status=status,
            match_ratio=match_ratio,
        )

    def get_recommended_adjustment(
        self,
        pattern_id: int,
        proposed_decision: str,
        proposed_confidence: float,
        match_ratio: float = 1.0,
        model_name: Optional[str] = None,
    ) -> AdjustmentRecommendation:
        """Get recommended adjustment for a proposed decision.

        This is the primary method models should use to adjust their
        predictions based on pattern history.

        Args:
            pattern_id: Current pattern
            proposed_decision: Model's proposed decision ('UP', 'DOWN', 'FLAT')
            proposed_confidence: Model's proposed confidence
            match_ratio: How well pattern matches (from fuzzy search)

        Returns:
            AdjustmentRecommendation with suggested modifications
        """
        insight = self.get_pattern_insight(
            pattern_id, match_ratio, model_name=model_name
        )

        # Default: no adjustment
        if insight is None or not insight.is_trustworthy:
            return AdjustmentRecommendation(
                original_decision=proposed_decision,
                original_confidence=proposed_confidence,
                recommended_decision=proposed_decision,
                recommended_confidence=proposed_confidence,
                confidence_delta=0.0,
                adjustment_reason="Insufficient pattern history",
                pattern_accuracy=0.5,
                pattern_bias=None,
                trust_level=0.0 if insight is None else insight.trust_confidence,
                decision_changed=False,
                confidence_reduced=False,
                confidence_increased=False,
            )

        # Get config parameters
        boost_threshold = float(self._config.get("pattern.boost_threshold", 0.55))
        penalty_threshold = float(self._config.get("pattern.penalty_threshold", 0.45))
        max_boost = float(self._config.get("pattern_reader.max_confidence_boost", 0.03))
        max_penalty = float(self._config.get("pattern_reader.max_confidence_penalty", 0.05))
        bias_flip_threshold = float(self._config.get("pattern_reader.bias_flip_threshold", 0.15))

        new_conf = proposed_confidence
        new_decision = proposed_decision
        reason_parts = []

        # Scale adjustments by match ratio (fuzzy matches get smaller adjustments)
        scale = match_ratio

        # Accuracy-based adjustment
        if insight.accuracy > boost_threshold:
            boost = (insight.accuracy - 0.50) * insight.trust_confidence * 0.5 * scale
            boost = min(boost, max_boost)
            new_conf = min(new_conf + boost, 0.65)
            reason_parts.append(f"Pattern accuracy {insight.accuracy:.1%} → boost")

        elif insight.accuracy < penalty_threshold:
            penalty = (0.50 - insight.accuracy) * insight.trust_confidence * 0.5 * scale
            penalty = min(penalty, max_penalty)
            new_conf = max(new_conf - penalty, 0.50)
            reason_parts.append(f"Pattern accuracy {insight.accuracy:.1%} → penalty")

        # Directional bias adjustment
        if insight.has_directional_bias and insight.favored_direction:
            if insight.favored_direction != proposed_decision:
                # Pattern favors opposite direction
                if insight.bias_strength > bias_flip_threshold:
                    # Strong bias - consider flipping decision
                    new_decision = insight.favored_direction
                    new_conf = max(new_conf - 0.05, 0.52)  # Reduce confidence on flip
                    reason_parts.append(f"Strong bias toward {insight.favored_direction} → flip")
                else:
                    # Moderate bias - just penalize
                    bias_penalty = float(self._config.get("pattern.bias_penalty", 0.01))
                    new_conf = max(new_conf - bias_penalty * scale, 0.50)
                    reason_parts.append(f"Bias toward {insight.favored_direction} → penalty")

        # Overconfidence cap
        if insight.is_overconfident and insight.suggested_confidence_cap:
            if new_conf > insight.suggested_confidence_cap:
                new_conf = insight.suggested_confidence_cap
                reason_parts.append("Overconfident pattern → cap")

        reason = "; ".join(reason_parts) if reason_parts else "No adjustment needed"
        confidence_delta = new_conf - proposed_confidence

        return AdjustmentRecommendation(
            original_decision=proposed_decision,
            original_confidence=proposed_confidence,
            recommended_decision=new_decision,
            recommended_confidence=clamp(new_conf, 0.0, 1.0),
            confidence_delta=confidence_delta,
            adjustment_reason=reason,
            pattern_accuracy=insight.accuracy,
            pattern_bias=insight.favored_direction,
            trust_level=insight.trust_confidence,
            decision_changed=(new_decision != proposed_decision),
            confidence_reduced=(confidence_delta < 0),
            confidence_increased=(confidence_delta > 0),
        )

    def get_best_match(
        self,
        matches: List["PatternMatch"],
        model_name: Optional[str] = None,
    ) -> Optional["PatternMatch"]:
        """Select the best pattern from a list of fuzzy matches.

        Prefers:
        1. Active over inactive
        2. Higher match ratio
        3. More historical usage

        Args:
            matches: List of pattern matches from fuzzy search

        Returns:
            Best match or None if list is empty
        """
        if not matches:
            return None

        # Get aggregates for all matches
        pattern_ids = [m.pattern_id for m in matches]
        if model_name:
            aggregates = self._store.get_model_aggregates_batch(pattern_ids, model_name)
        else:
            aggregates = self._store.get_aggregates_batch(pattern_ids)

        # Score each match
        best_match = None
        best_score = -1.0

        for match in matches:
            agg = aggregates.get(match.pattern_id)
            if not agg:
                continue

            # Score: match_ratio * (status_bonus) * log(usage + 1)
            status_bonus = 1.5 if match.status == "active" else 1.0
            usage_factor = 1.0 + (0.1 * min(agg.total_uses, 100))  # Cap at 100

            score = match.match_ratio * status_bonus * usage_factor

            if score > best_score:
                best_score = score
                best_match = match

        return best_match

    def is_pattern_favorable(
        self,
        pattern_id: int,
        direction: str,
        model_name: Optional[str] = None,
    ) -> Tuple[bool, float]:
        """Check if pattern historically favors a direction.

        Args:
            pattern_id: Target pattern
            direction: Direction to check ('UP' or 'DOWN')

        Returns:
            (is_favorable, historical_accuracy_for_direction)
        """
        aggregates = None
        if model_name:
            aggregates = self._store.get_model_aggregates(pattern_id, model_name)
        if not aggregates:
            aggregates = self._store.get_aggregates(pattern_id)
        if not aggregates or aggregates.total_uses < self._min_uses:
            return False, 0.5

        if direction == "UP":
            accuracy = aggregates.up_accuracy
        else:
            accuracy = aggregates.down_accuracy

        favorable_threshold = float(self._config.get("pattern.boost_threshold", 0.55))
        is_favorable = accuracy > favorable_threshold

        return is_favorable, accuracy

    def get_confidence_cap(
        self,
        pattern_id: int,
        model_name: Optional[str] = None,
    ) -> Optional[float]:
        """Get recommended confidence cap for pattern.

        Returns suggested maximum confidence based on historical
        calibration data, or None if pattern has good calibration.

        Args:
            pattern_id: Target pattern

        Returns:
            Suggested max confidence or None
        """
        insight = self.get_pattern_insight(pattern_id, model_name=model_name)
        if insight and insight.is_overconfident:
            return insight.suggested_confidence_cap
        return None
