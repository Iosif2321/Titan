"""Sprint 20: Online Learning with SGD + RMSProp.

This module provides online learning capabilities for real-time model adaptation.
Uses Contextual Bandits approach with Thompson Sampling for exploration.

Key components:
- OnlineLearner: SGD + RMSProp optimizer for weight updates
- MultiScaleEMA: Multi-scale exponential moving average memory
- RewardCalculator: Converts outcomes to reward signals
- OnlineAdapter: Combines all components for real-time adaptation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore

from titan.core.config import ConfigStore


# Model names for weight learning
MODEL_NAMES = ["TRENDVIC", "OSCILLATOR", "VOLUMEMETRIX", "ML_CLASSIFIER"]


@dataclass
class EMAState:
    """State for exponential moving average."""
    value: float = 0.5
    count: int = 0


@dataclass
class OnlineStats:
    """Statistics for online learning."""
    total_updates: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    weight_updates: int = 0
    exploration_count: int = 0
    exploitation_count: int = 0


class MultiScaleEMA:
    """Multi-scale exponential moving average memory.

    Tracks statistics at three time scales:
    - Short: ~20 updates (alpha=0.05)
    - Medium: ~100 updates (alpha=0.01)
    - Long: ~1000 updates (alpha=0.001)

    This allows the system to adapt quickly to recent changes
    while maintaining long-term stability.
    """

    # EMA decay rates (from Chronos)
    SHORT_ALPHA = 0.05   # ~20 updates half-life
    MEDIUM_ALPHA = 0.01  # ~100 updates half-life
    LONG_ALPHA = 0.001   # ~1000 updates half-life

    def __init__(self) -> None:
        self._short: Dict[str, EMAState] = {}
        self._medium: Dict[str, EMAState] = {}
        self._long: Dict[str, EMAState] = {}

    def update(self, key: str, value: float) -> None:
        """Update all EMA scales with new value."""
        # Initialize if needed
        if key not in self._short:
            self._short[key] = EMAState(value=value, count=1)
            self._medium[key] = EMAState(value=value, count=1)
            self._long[key] = EMAState(value=value, count=1)
            return

        # Update each scale
        self._short[key].value = (
            self.SHORT_ALPHA * value +
            (1 - self.SHORT_ALPHA) * self._short[key].value
        )
        self._short[key].count += 1

        self._medium[key].value = (
            self.MEDIUM_ALPHA * value +
            (1 - self.MEDIUM_ALPHA) * self._medium[key].value
        )
        self._medium[key].count += 1

        self._long[key].value = (
            self.LONG_ALPHA * value +
            (1 - self.LONG_ALPHA) * self._long[key].value
        )
        self._long[key].count += 1

    def get(self, key: str, scale: str = "medium") -> float:
        """Get EMA value for key at specified scale."""
        if scale == "short":
            return self._short.get(key, EMAState()).value
        elif scale == "long":
            return self._long.get(key, EMAState()).value
        else:  # medium (default)
            return self._medium.get(key, EMAState()).value

    def get_all_scales(self, key: str) -> Tuple[float, float, float]:
        """Get values at all three scales."""
        return (
            self._short.get(key, EMAState()).value,
            self._medium.get(key, EMAState()).value,
            self._long.get(key, EMAState()).value,
        )

    def get_trend(self, key: str) -> str:
        """Detect trend by comparing scales.

        Returns:
            'improving' if short > medium > long
            'degrading' if short < medium < long
            'stable' otherwise
        """
        short, medium, long = self.get_all_scales(key)

        if short > medium > long:
            return "improving"
        elif short < medium < long:
            return "degrading"
        return "stable"

    def get_count(self, key: str) -> int:
        """Get update count for key."""
        return self._medium.get(key, EMAState()).count

    def summary(self) -> Dict[str, object]:
        """Get summary of all tracked keys."""
        result = {}
        for key in self._medium:
            short, medium, long = self.get_all_scales(key)
            result[key] = {
                "short": round(short, 4),
                "medium": round(medium, 4),
                "long": round(long, 4),
                "trend": self.get_trend(key),
                "count": self.get_count(key),
            }
        return result


class OnlineLearner:
    """Online learner with SGD + RMSProp optimizer.

    Updates model weights based on prediction outcomes using
    stochastic gradient descent with RMSProp adaptive learning rate.

    The learning signal is:
    - reward = +1 if prediction correct
    - reward = -1 if prediction wrong
    - gradient = reward * (model_weight - 1/n_models)

    This pushes weights toward models that are currently performing well.
    """

    def __init__(
        self,
        n_models: int = 4,
        learning_rate: float = 0.01,
        rmsprop_decay: float = 0.95,
        min_weight: float = 0.10,
        max_weight: float = 0.50,
    ) -> None:
        if not HAS_NUMPY:
            raise ImportError("numpy required for OnlineLearner")

        self._n_models = n_models
        self._lr = learning_rate
        self._rmsprop_decay = rmsprop_decay
        self._min_weight = min_weight
        self._max_weight = max_weight

        # Initialize weights uniformly
        self._weights = np.ones(n_models) / n_models

        # RMSProp accumulators (one per model)
        self._rmsprop_g = np.zeros(n_models)

        # Statistics
        self._update_count = 0

    def update(
        self,
        model_hits: List[bool],
        model_confs: List[float],
    ) -> np.ndarray:
        """Update weights based on model performance.

        Args:
            model_hits: List of whether each model was correct
            model_confs: List of each model's confidence

        Returns:
            Updated weights as numpy array
        """
        if len(model_hits) != self._n_models:
            return self._weights

        # Convert to numpy
        hits = np.array([1.0 if h else -1.0 for h in model_hits])
        confs = np.array(model_confs)

        # Reward signal: hit * confidence (confident correct = high reward)
        rewards = hits * confs

        # Gradient: push weight toward models with positive reward
        # gradient[i] = reward[i] * (weight[i] - mean_weight)
        mean_weight = 1.0 / self._n_models
        gradient = rewards * (self._weights - mean_weight)

        # RMSProp: accumulate squared gradient
        self._rmsprop_g = (
            self._rmsprop_decay * self._rmsprop_g +
            (1 - self._rmsprop_decay) * gradient ** 2
        )

        # Update weights: w += lr * gradient / sqrt(G + eps)
        # Note: += because we want to increase weight for positive reward
        adaptive_lr = self._lr / (np.sqrt(self._rmsprop_g) + 1e-8)
        self._weights += adaptive_lr * gradient

        # Apply bounds
        self._weights = np.clip(self._weights, self._min_weight, self._max_weight)

        # Normalize to sum to 1
        self._weights = self._weights / np.sum(self._weights)

        self._update_count += 1
        return self._weights

    def get_weights(self) -> Dict[str, float]:
        """Get current weights as dict."""
        return {
            name: float(self._weights[i])
            for i, name in enumerate(MODEL_NAMES[:self._n_models])
        }

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Set weights from dict."""
        for i, name in enumerate(MODEL_NAMES[:self._n_models]):
            if name in weights:
                self._weights[i] = weights[name]
        # Normalize
        self._weights = self._weights / np.sum(self._weights)

    @property
    def update_count(self) -> int:
        return self._update_count


class RewardCalculator:
    """Calculates reward signals from prediction outcomes.

    Reward types:
    - Binary: +1 correct, -1 wrong
    - Confidence-weighted: reward * confidence
    - Return-weighted: reward * abs(return_pct)
    - Risk-adjusted: penalizes high-confidence wrong predictions
    """

    def __init__(
        self,
        reward_type: str = "confidence",
        wrong_penalty_mult: float = 1.5,
    ) -> None:
        self._reward_type = reward_type
        self._wrong_penalty_mult = wrong_penalty_mult

    def calculate(
        self,
        hit: bool,
        confidence: float,
        return_pct: float = 0.0,
    ) -> float:
        """Calculate reward for a single prediction.

        Args:
            hit: Whether prediction was correct
            confidence: Model's confidence [0.5, 1.0]
            return_pct: Actual return percentage

        Returns:
            Reward signal
        """
        base_reward = 1.0 if hit else -1.0

        if self._reward_type == "binary":
            return base_reward

        elif self._reward_type == "confidence":
            # Weight by confidence
            return base_reward * confidence

        elif self._reward_type == "return":
            # Weight by return magnitude
            return_weight = min(abs(return_pct) * 100, 2.0)  # Cap at 2x
            return base_reward * return_weight

        elif self._reward_type == "risk_adjusted":
            # Penalize high-confidence wrong predictions
            if hit:
                return confidence
            else:
                # Wrong prediction penalty increases with confidence
                return -confidence * self._wrong_penalty_mult

        return base_reward


class OnlineAdapter:
    """Combines online learning components for real-time adaptation.

    This is the main interface for online learning in backtest/live.

    Components:
    - OnlineLearner: Updates model weights
    - MultiScaleEMA: Tracks accuracy at multiple time scales
    - RewardCalculator: Computes learning signals
    """

    def __init__(
        self,
        config: Optional[ConfigStore] = None,
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled and HAS_NUMPY
        self._config = config

        if not self._enabled:
            return

        # Get config values
        lr = 0.01
        min_weight = 0.10
        max_weight = 0.50
        reward_type = "confidence"

        if config:
            lr = float(config.get("online.learning_rate", 0.01))
            min_weight = float(config.get("online.min_weight", 0.10))
            max_weight = float(config.get("online.max_weight", 0.50))
            reward_type = str(config.get("online.reward_type", "confidence"))

        # Initialize components
        self._learner = OnlineLearner(
            n_models=len(MODEL_NAMES),
            learning_rate=lr,
            min_weight=min_weight,
            max_weight=max_weight,
        )
        self._ema = MultiScaleEMA()
        self._reward_calc = RewardCalculator(reward_type=reward_type)

        # Statistics
        self._stats = OnlineStats()

    def record_outcome(
        self,
        model_decisions: Dict[str, str],
        actual_direction: str,
        model_confs: Dict[str, float],
        ensemble_hit: bool,
        return_pct: float = 0.0,
    ) -> Dict[str, float]:
        """Record prediction outcome and update weights.

        Args:
            model_decisions: Dict of model_name -> predicted direction
            actual_direction: Actual direction ("UP" or "DOWN")
            model_confs: Dict of model_name -> confidence
            ensemble_hit: Whether ensemble prediction was correct
            return_pct: Actual return percentage

        Returns:
            Updated model weights
        """
        if not self._enabled:
            return {}

        # Calculate hits for each model
        model_hits = []
        confs = []
        for name in MODEL_NAMES:
            predicted = model_decisions.get(name, "")
            hit = predicted == actual_direction
            model_hits.append(hit)
            confs.append(model_confs.get(name, 0.5))

            # Update EMA for this model's accuracy
            self._ema.update(f"{name}_accuracy", 1.0 if hit else 0.0)

        # Update ensemble accuracy EMA
        self._ema.update("ensemble_accuracy", 1.0 if ensemble_hit else 0.0)

        # Calculate reward and update stats
        reward = self._reward_calc.calculate(
            hit=ensemble_hit,
            confidence=max(confs),
            return_pct=return_pct,
        )
        self._stats.total_updates += 1
        self._stats.total_reward += reward
        self._stats.avg_reward = self._stats.total_reward / self._stats.total_updates

        # Update weights
        new_weights = self._learner.update(model_hits, confs)
        self._stats.weight_updates += 1

        return self._learner.get_weights()

    def get_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        if not self._enabled:
            return {}
        return self._learner.get_weights()

    def set_initial_weights(self, weights: Dict[str, float]) -> None:
        """Set initial weights (e.g., from previous run)."""
        if self._enabled:
            self._learner.set_weights(weights)

    def get_model_trend(self, model_name: str) -> str:
        """Get accuracy trend for a model."""
        if not self._enabled:
            return "unknown"
        return self._ema.get_trend(f"{model_name}_accuracy")

    def get_model_accuracy(
        self, model_name: str, scale: str = "medium"
    ) -> float:
        """Get EMA accuracy for a model at specified scale."""
        if not self._enabled:
            return 0.5
        return self._ema.get(f"{model_name}_accuracy", scale)

    def should_explore(self, exploration_rate: float = 0.1) -> bool:
        """Decide whether to explore (random) or exploit (best weights).

        Uses epsilon-greedy strategy with decaying exploration.
        """
        if not self._enabled or not HAS_NUMPY:
            return False

        # Decay exploration over time
        decay = 0.999 ** self._stats.total_updates
        effective_rate = exploration_rate * decay

        if np.random.random() < effective_rate:
            self._stats.exploration_count += 1
            return True

        self._stats.exploitation_count += 1
        return False

    def summary(self) -> Dict[str, object]:
        """Get summary of online learning state."""
        if not self._enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "weights": self._learner.get_weights(),
            "update_count": self._stats.total_updates,
            "avg_reward": round(self._stats.avg_reward, 4),
            "weight_updates": self._stats.weight_updates,
            "exploration_rate": (
                self._stats.exploration_count /
                max(1, self._stats.exploration_count + self._stats.exploitation_count)
            ),
            "ema_summary": self._ema.summary(),
        }


def create_online_adapter(
    config: Optional[ConfigStore] = None,
    enabled: bool = True,
) -> OnlineAdapter:
    """Factory function to create OnlineAdapter."""
    return OnlineAdapter(config=config, enabled=enabled)
