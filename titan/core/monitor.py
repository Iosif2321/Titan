"""Performance monitoring for adaptive model weighting."""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional


@dataclass
class PredictionResult:
    """Single prediction result for monitoring."""

    model_name: str
    predicted: str  # "UP", "DOWN", "FLAT"
    actual: str  # "UP", "DOWN"
    regime: str  # "trending_up", "trending_down", "ranging", "volatile"
    confidence: float
    is_correct: bool = field(init=False)

    def __post_init__(self) -> None:
        # FLAT predictions are never "correct" - they abstained
        if self.predicted == "FLAT":
            self.is_correct = False
        else:
            self.is_correct = self.predicted == self.actual


class PerformanceMonitor:
    """Monitors model performance in real-time for adaptive weighting.

    Tracks:
        - Rolling accuracy per model
        - Accuracy per model per regime
        - Recent predictions for analysis

    This data is used by AdaptiveWeightManager to adjust model weights
    based on how well each model performs in the current market regime.
    """

    def __init__(self, window: int = 100) -> None:
        """Initialize monitor.

        Args:
            window: Rolling window size for accuracy calculation
        """
        self._window = window

        # Per-model rolling predictions
        self._model_history: Dict[str, Deque[PredictionResult]] = {}

        # Per-model per-regime predictions
        self._regime_history: Dict[str, Dict[str, Deque[PredictionResult]]] = {}

        # All recent predictions (for debugging/analysis)
        self._all_predictions: Deque[PredictionResult] = deque(maxlen=window * 3)

    def update(
        self,
        model_name: str,
        predicted: str,
        actual: str,
        regime: str,
        confidence: float = 0.5,
    ) -> None:
        """Record a prediction result.

        Args:
            model_name: Name of the model (e.g., "TRENDVIC")
            predicted: Predicted direction ("UP", "DOWN", "FLAT")
            actual: Actual direction ("UP", "DOWN")
            regime: Current market regime
            confidence: Model's confidence in prediction
        """
        result = PredictionResult(
            model_name=model_name,
            predicted=predicted,
            actual=actual,
            regime=regime,
            confidence=confidence,
        )

        # Update model history
        if model_name not in self._model_history:
            self._model_history[model_name] = deque(maxlen=self._window)
        self._model_history[model_name].append(result)

        # Update regime history
        if model_name not in self._regime_history:
            self._regime_history[model_name] = {}
        if regime not in self._regime_history[model_name]:
            self._regime_history[model_name][regime] = deque(maxlen=self._window)
        self._regime_history[model_name][regime].append(result)

        # Update all predictions
        self._all_predictions.append(result)

    def get_model_accuracy(self, model_name: str) -> float:
        """Get rolling accuracy for a model.

        Args:
            model_name: Name of the model

        Returns:
            Accuracy as fraction (0.0 to 1.0), or 0.5 if no data
        """
        if model_name not in self._model_history:
            return 0.5  # Default when no data

        history = self._model_history[model_name]
        if not history:
            return 0.5

        # Only count non-FLAT predictions for accuracy
        valid_predictions = [r for r in history if r.predicted != "FLAT"]
        if not valid_predictions:
            return 0.5

        correct = sum(1 for r in valid_predictions if r.is_correct)
        return correct / len(valid_predictions)

    def get_regime_accuracy(self, model_name: str, regime: str) -> float:
        """Get accuracy for a model in a specific regime.

        Args:
            model_name: Name of the model
            regime: Market regime

        Returns:
            Accuracy as fraction (0.0 to 1.0), or 0.5 if no data
        """
        if model_name not in self._regime_history:
            return 0.5
        if regime not in self._regime_history[model_name]:
            return 0.5

        history = self._regime_history[model_name][regime]
        if not history:
            return 0.5

        # Only count non-FLAT predictions
        valid_predictions = [r for r in history if r.predicted != "FLAT"]
        if not valid_predictions:
            return 0.5

        correct = sum(1 for r in valid_predictions if r.is_correct)
        return correct / len(valid_predictions)

    def get_all_model_accuracies(self) -> Dict[str, float]:
        """Get rolling accuracy for all tracked models.

        Returns:
            Dictionary mapping model names to accuracies
        """
        return {
            model: self.get_model_accuracy(model)
            for model in self._model_history
        }

    def get_regime_accuracies(self, regime: str) -> Dict[str, float]:
        """Get accuracy for all models in a specific regime.

        Args:
            regime: Market regime

        Returns:
            Dictionary mapping model names to accuracies in that regime
        """
        result = {}
        for model_name in self._model_history:
            result[model_name] = self.get_regime_accuracy(model_name, regime)
        return result

    def get_optimal_weights(self, regime: str) -> Dict[str, float]:
        """Calculate optimal weights based on regime performance.

        Models with higher accuracy in the current regime get higher weights.

        Args:
            regime: Current market regime

        Returns:
            Dictionary mapping model names to suggested weights (sum = 1.0)
        """
        accuracies = self.get_regime_accuracies(regime)

        if not accuracies:
            return {}

        # Convert accuracies to weights
        # Use (accuracy - 0.3) to penalize poor models more
        # Models below 30% accuracy get near-zero weight
        adjusted = {
            model: max(acc - 0.3, 0.05)
            for model, acc in accuracies.items()
        }

        total = sum(adjusted.values())
        if total <= 0:
            # Fallback to equal weights
            n = len(accuracies)
            return {model: 1.0 / n for model in accuracies}

        return {model: val / total for model, val in adjusted.items()}

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for debugging/logging.

        Returns:
            Dictionary with detailed performance statistics
        """
        stats: Dict[str, Any] = {
            "total_predictions": len(self._all_predictions),
            "models": {},
        }

        for model_name in self._model_history:
            model_stats = {
                "overall_accuracy": self.get_model_accuracy(model_name),
                "prediction_count": len(self._model_history[model_name]),
                "regime_accuracies": {},
            }

            if model_name in self._regime_history:
                for regime, history in self._regime_history[model_name].items():
                    valid = [r for r in history if r.predicted != "FLAT"]
                    if valid:
                        correct = sum(1 for r in valid if r.is_correct)
                        model_stats["regime_accuracies"][regime] = {
                            "accuracy": correct / len(valid),
                            "count": len(valid),
                        }

            stats["models"][model_name] = model_stats

        return stats

    def reset(self) -> None:
        """Clear all history. Use when restarting or after major config changes."""
        self._model_history.clear()
        self._regime_history.clear()
        self._all_predictions.clear()
