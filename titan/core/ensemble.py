from typing import TYPE_CHECKING, Dict, List, Optional, Union

from titan.core.adapters.temporal import TemporalAdjuster
from titan.core.calibration import ConfidenceCompressor
from titan.core.config import ConfigStore
from titan.core.strategies.trending import TrendingStrategy
from titan.core.strategies.volatile import VolatileStrategy
from titan.core.types import Decision, ModelOutput
from titan.core.utils import clamp
from titan.core.weights import AdaptiveWeightManager, WeightManager

if TYPE_CHECKING:
    from titan.core.adapters.pattern import PatternAdjuster
    from titan.core.fusion import TransformerFusion
    from titan.core.regime import RegimeDetector


class Ensemble:
    """Combines multiple model outputs into a single decision.

    Supports regime-aware weighting when using AdaptiveWeightManager.
    Now includes confidence compression to fix overconfidence problem.
    Includes specialized handling for volatile and trending regimes.
    Includes temporal adjustments for hour-based accuracy patterns.
    Sprint 11: Includes agreement boost for when models agree.
    Sprint 13: Includes pattern experience for historical performance.
    """

    def __init__(
        self,
        config: ConfigStore,
        weights: Union[WeightManager, AdaptiveWeightManager],
        regime_detector: Optional["RegimeDetector"] = None,
        pattern_adjuster: Optional["PatternAdjuster"] = None,
        fusion: Optional["TransformerFusion"] = None,
    ) -> None:
        self._config = config
        self._weights = weights
        self._regime_detector = regime_detector
        self._pattern_adjuster = pattern_adjuster
        self._fusion = fusion
        self._last_regime: Optional[str] = None
        self._compressor = ConfidenceCompressor(config)
        self._volatile_strategy = VolatileStrategy(config)
        self._trending_strategy = TrendingStrategy(config)
        self._temporal = TemporalAdjuster(config)

    def _check_agreement(self, outputs: List[ModelOutput]) -> str:
        """Check model agreement level.

        Returns:
            'full' - all models agree
            'partial' - majority agrees (2 out of 3)
            'none' - no clear agreement
        """
        if len(outputs) < 2:
            return "none"

        directions = []
        for o in outputs:
            if o.prob_up > o.prob_down:
                directions.append("UP")
            else:
                directions.append("DOWN")

        up_count = sum(1 for d in directions if d == "UP")
        down_count = len(directions) - up_count

        if up_count == len(directions) or down_count == len(directions):
            return "full"
        elif up_count >= 2 or down_count >= 2:
            return "partial"
        return "none"

    def _apply_agreement_boost(
        self, confidence: float, outputs: List[ModelOutput]
    ) -> float:
        """Boost confidence when models agree.

        Sprint 11: Full agreement → +5% confidence
                   Partial agreement → +2% confidence
        """
        agreement = self._check_agreement(outputs)

        full_boost = float(self._config.get("ensemble.agreement_full_boost", 0.05))
        partial_boost = float(self._config.get("ensemble.agreement_partial_boost", 0.02))

        if agreement == "full":
            return min(confidence + full_boost, 0.65)  # Cap at 65%
        elif agreement == "partial":
            return min(confidence + partial_boost, 0.62)  # Cap at 62%
        return confidence

    def decide(
        self,
        outputs: List[ModelOutput],
        features: Optional[Dict[str, float]] = None,
        ts: Optional[int] = None,
        pattern_id: Optional[int] = None,
        override_weights: Optional[Dict[str, float]] = None,
        override_params: Optional[Dict[str, float]] = None,
    ) -> Decision:
        """Make ensemble decision from model outputs.

        Args:
            outputs: List of model predictions
            features: Current market features (used for regime detection)
            ts: Unix timestamp for temporal adjustments
            pattern_id: Current pattern ID for experience-based adjustment
            override_weights: Optional weights to use instead of default (Sprint 17: SessionAdapter)
            override_params: Optional params to override config (Sprint 17: SessionAdapter Thompson Sampling)

        Returns:
            Combined decision with direction, confidence, and probabilities
        """
        # Helper to get param with override support
        def get_param(key: str, default: float) -> float:
            if override_params and key in override_params:
                return float(override_params[key])
            return float(self._config.get(key, default))
        # Detect regime if features available
        regime: Optional[str] = None
        if features and self._regime_detector:
            regime = self._regime_detector.detect(features)
            self._last_regime = regime

        # Special handling for volatile regime (55%+ error rate)
        if regime == "volatile" and features:
            volatile_decision = self._volatile_strategy.decide(features)
            if volatile_decision:
                # Apply compression and regime penalty even to volatile decisions
                final_conf = self._compressor.compress(volatile_decision.confidence)
                final_conf = self._compressor.apply_regime_penalty(final_conf, regime)
                return Decision(
                    direction=volatile_decision.direction,
                    confidence=clamp(final_conf, 0.0, 1.0),
                    prob_up=volatile_decision.prob_up,
                    prob_down=volatile_decision.prob_down,
                )

        # Special handling for trending_down regime (53%+ error rate)
        # Uses exhaustion detection to predict reversals
        if regime == "trending_down" and features:
            # Update trending strategy with current features
            self._trending_strategy.update(features)

            # Get base direction hint (simple majority from outputs)
            up_count = sum(1 for o in outputs if o.prob_up > o.prob_down)
            base_dir = "UP" if up_count > len(outputs) / 2 else "DOWN"

            trending_decision = self._trending_strategy.decide(features, regime, base_dir)
            if trending_decision:
                final_conf = self._compressor.compress(trending_decision.confidence)
                final_conf = self._compressor.apply_regime_penalty(final_conf, regime)
                return Decision(
                    direction=trending_decision.direction,
                    confidence=clamp(final_conf, 0.0, 1.0),
                    prob_up=trending_decision.prob_up,
                    prob_down=trending_decision.prob_down,
                )

        # Sprint 21: Use TransformerFusion if available and enabled
        if self._fusion and self._fusion.enabled:
            fusion_up, fusion_down = self._fusion.forward(outputs, features)
            prob_up = fusion_up
            prob_down = fusion_down
        else:
            # Get weights - use override if provided (Sprint 17: SessionAdapter)
            # Otherwise use regime-aware if using AdaptiveWeightManager
            if override_weights is not None:
                weights = override_weights
            elif isinstance(self._weights, AdaptiveWeightManager) and regime:
                weights = self._weights.get_model_weights(regime)
            else:
                weights = self._weights.get_model_weights()

            flat_threshold = get_param("ensemble.flat_threshold", 0.55)
            min_margin = get_param("ensemble.min_margin", 0.05)

            # Additional volatility-based adjustments (legacy behavior)
            if features:
                vol_z = float(features.get("volatility_z", 0.0))
                vol_high = float(self._config.get("ensemble.vol_z_high", 1.0))
                if vol_z >= vol_high:
                    scale = clamp(
                        float(self._config.get("ensemble.trendvic_high_scale", 0.6)),
                        0.0,
                        1.0,
                    )
                    if "TRENDVIC" in weights:
                        weights["TRENDVIC"] = max(weights["TRENDVIC"] * scale, 0.0)
                    flat_threshold = clamp(
                        flat_threshold
                        + float(self._config.get("ensemble.high_vol_flat_add", 0.0)),
                        0.0,
                        1.0,
                    )
                    min_margin = max(
                        min_margin
                        + float(self._config.get("ensemble.high_vol_margin_add", 0.0)),
                        0.0,
                    )

            weighted_up = 0.0
            weighted_down = 0.0
            total_weight = 0.0

            for output in outputs:
                weight = float(weights.get(output.model_name, 1.0))
                weighted_up += weight * output.prob_up
                weighted_down += weight * output.prob_down
                total_weight += weight

            if total_weight <= 0:
                total_weight = 1.0

            prob_up = weighted_up / total_weight
            prob_down = weighted_down / total_weight

        # Define thresholds (needed for direction logic below)
        # Use get_param to respect override_params from SessionAdapter
        flat_threshold = get_param("ensemble.flat_threshold", 0.55)
        min_margin = get_param("ensemble.min_margin", 0.05)
        confidence = max(prob_up, prob_down)
        margin = abs(prob_up - prob_down)

        # Always return UP or DOWN - FLAT only when explicitly uncertain
        # FLAT is undesirable; even a weak signal is better than abstaining
        if confidence >= flat_threshold and margin >= min_margin:
            # High confidence decision
            direction = "UP" if prob_up >= prob_down else "DOWN"
        elif margin >= 0.01:
            # Low confidence but there's a clear direction - use it
            direction = "UP" if prob_up >= prob_down else "DOWN"
        else:
            # Truly tied (margin < 1%) - still pick one, FLAT is last resort
            # Use prob_up/prob_down to break ties (works with both fusion and weighted)
            direction = "UP" if prob_up >= prob_down else "DOWN"

        # Apply confidence compression to fix overconfidence problem
        # High confidence (65%+) historically has LOWER accuracy (33-47%)
        final_confidence = self._compressor.compress(confidence)

        # Sprint 11: Apply agreement boost (when models agree, boost confidence)
        final_confidence = self._apply_agreement_boost(final_confidence, outputs)

        # Apply regime-based penalty (volatile/trending_down are problematic)
        final_confidence = self._compressor.apply_regime_penalty(final_confidence, regime)

        # Apply temporal adjustment (some hours have historically worse accuracy)
        final_confidence = self._temporal.adjust_confidence(final_confidence, ts)

        # Build decision
        decision = Decision(
            direction=direction,
            confidence=clamp(final_confidence, 0.0, 1.0),
            prob_up=clamp(prob_up, 0.0, 1.0),
            prob_down=clamp(prob_down, 0.0, 1.0),
        )

        # Sprint 13: Apply pattern experience adjustment
        # FIXED: Pass max_ts=ts to prevent data leakage in backtest
        if self._pattern_adjuster and pattern_id is not None:
            decision = self._pattern_adjuster.adjust_decision(
                decision, pattern_id, features, max_ts=ts
            )

        return decision

    def get_last_regime(self) -> Optional[str]:
        """Get the last detected market regime.

        Returns:
            Regime name or None if not detected
        """
        return self._last_regime
