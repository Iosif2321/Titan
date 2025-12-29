from typing import TYPE_CHECKING, Dict, Optional

from titan.core.state_store import StateStore

if TYPE_CHECKING:
    from titan.core.monitor import PerformanceMonitor
    from titan.core.regime import RegimeDetector


DEFAULT_MODEL_WEIGHTS: Dict[str, float] = {
    "TRENDVIC": 1.0,
    "OSCILLATOR": 1.0,
    "VOLUMEMETRIX": 1.0,
}

# Base regime weights (before performance adjustment)
DEFAULT_REGIME_WEIGHTS: Dict[str, Dict[str, float]] = {
    "trending_up": {"TRENDVIC": 0.50, "OSCILLATOR": 0.20, "VOLUMEMETRIX": 0.30},
    "trending_down": {"TRENDVIC": 0.50, "OSCILLATOR": 0.20, "VOLUMEMETRIX": 0.30},
    "ranging": {"TRENDVIC": 0.20, "OSCILLATOR": 0.50, "VOLUMEMETRIX": 0.30},
    "volatile": {"TRENDVIC": 0.25, "OSCILLATOR": 0.25, "VOLUMEMETRIX": 0.50},
}


class WeightManager:
    def __init__(self, state_store: StateStore) -> None:
        self._state = state_store

    def get_model_weights(self) -> Dict[str, float]:
        stored = self._state.get("weights.models")
        if stored is None:
            return dict(DEFAULT_MODEL_WEIGHTS)
        merged = dict(DEFAULT_MODEL_WEIGHTS)
        merged.update(stored)
        return merged

    def set_model_weights(self, weights: Dict[str, float]) -> None:
        self._state.set("weights.models", dict(weights))


class AdaptiveWeightManager(WeightManager):
    """Weight manager that adapts weights based on regime and performance.

    Combines:
        1. Base regime weights (different models excel in different regimes)
        2. Performance adjustments (boost models that are doing well)

    The final weight for a model in a regime is:
        weight = regime_base_weight * performance_multiplier
    """

    def __init__(
        self,
        state_store: StateStore,
        regime_detector: Optional["RegimeDetector"] = None,
        monitor: Optional["PerformanceMonitor"] = None,
        performance_blend: float = 0.3,
        min_weight: float = 0.1,
    ) -> None:
        """Initialize adaptive weight manager.

        Args:
            state_store: State persistence
            regime_detector: For detecting current market regime
            monitor: For tracking model performance
            performance_blend: How much to weight performance vs base (0-1)
                0 = use only base regime weights
                1 = use only performance-based weights
            min_weight: Minimum weight for any model (prevents zeroing out)
        """
        super().__init__(state_store)
        self._regime_detector = regime_detector
        self._monitor = monitor
        self._performance_blend = performance_blend
        self._min_weight = min_weight
        self._current_regime: Optional[str] = None

    def get_model_weights(self, regime: Optional[str] = None) -> Dict[str, float]:
        """Get weights adapted for current regime and performance.

        Args:
            regime: Market regime (if None, uses last detected or defaults)

        Returns:
            Dictionary mapping model names to weights
        """
        # Use provided regime or fall back to stored/default
        effective_regime = regime or self._current_regime

        if effective_regime is None:
            # No regime info - return base weights
            return super().get_model_weights()

        # Get base regime weights
        base_weights = DEFAULT_REGIME_WEIGHTS.get(
            effective_regime,
            {"TRENDVIC": 0.33, "OSCILLATOR": 0.33, "VOLUMEMETRIX": 0.34}
        )

        # If no monitor, return base regime weights
        if self._monitor is None:
            return dict(base_weights)

        # Get performance-based weights
        perf_weights = self._monitor.get_optimal_weights(effective_regime)

        if not perf_weights:
            return dict(base_weights)

        # Blend base and performance weights
        blended = {}
        for model in base_weights:
            base_w = base_weights.get(model, 0.33)
            perf_w = perf_weights.get(model, 0.33)

            # Weighted average
            combined = (
                (1 - self._performance_blend) * base_w
                + self._performance_blend * perf_w
            )

            # Ensure minimum weight
            blended[model] = max(combined, self._min_weight)

        # Normalize to sum to ~1.0
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}

        return blended

    def update_regime(self, features: Dict[str, float]) -> str:
        """Update current regime based on features.

        Args:
            features: Current market features

        Returns:
            Detected regime name
        """
        if self._regime_detector is None:
            return "unknown"

        self._current_regime = self._regime_detector.detect(features)
        return self._current_regime

    def get_current_regime(self) -> Optional[str]:
        """Get the current detected regime."""
        return self._current_regime

    def get_regime_weights_debug(self) -> Dict[str, Dict[str, float]]:
        """Get weights for all regimes (for debugging/logging)."""
        result = {}
        for regime in DEFAULT_REGIME_WEIGHTS:
            result[regime] = self.get_model_weights(regime)
        return result
