from typing import TYPE_CHECKING, Dict, List, Optional, Union

from titan.core.config import ConfigStore
from titan.core.types import Decision, ModelOutput
from titan.core.utils import clamp
from titan.core.weights import AdaptiveWeightManager, WeightManager

if TYPE_CHECKING:
    from titan.core.regime import RegimeDetector


class Ensemble:
    """Combines multiple model outputs into a single decision.

    Supports regime-aware weighting when using AdaptiveWeightManager.
    """

    def __init__(
        self,
        config: ConfigStore,
        weights: Union[WeightManager, AdaptiveWeightManager],
        regime_detector: Optional["RegimeDetector"] = None,
    ) -> None:
        self._config = config
        self._weights = weights
        self._regime_detector = regime_detector
        self._last_regime: Optional[str] = None

    def decide(
        self,
        outputs: List[ModelOutput],
        features: Optional[Dict[str, float]] = None,
    ) -> Decision:
        """Make ensemble decision from model outputs.

        Args:
            outputs: List of model predictions
            features: Current market features (used for regime detection)

        Returns:
            Combined decision with direction, confidence, and probabilities
        """
        # Detect regime if features available
        regime: Optional[str] = None
        if features and self._regime_detector:
            regime = self._regime_detector.detect(features)
            self._last_regime = regime

        # Get weights - regime-aware if using AdaptiveWeightManager
        if isinstance(self._weights, AdaptiveWeightManager) and regime:
            weights = self._weights.get_model_weights(regime)
        else:
            weights = self._weights.get_model_weights()

        flat_threshold = float(self._config.get("ensemble.flat_threshold", 0.55))
        min_margin = float(self._config.get("ensemble.min_margin", 0.05))

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
            # Use the raw weighted values to break ties
            direction = "UP" if weighted_up >= weighted_down else "DOWN"

        return Decision(
            direction=direction,
            confidence=clamp(confidence, 0.0, 1.0),
            prob_up=clamp(prob_up, 0.0, 1.0),
            prob_down=clamp(prob_down, 0.0, 1.0),
        )

    def get_last_regime(self) -> Optional[str]:
        """Get the last detected market regime.

        Returns:
            Regime name or None if not detected
        """
        return self._last_regime
