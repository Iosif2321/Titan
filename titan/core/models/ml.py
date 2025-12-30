"""Sprint 14: LightGBM-based directional classifier.

This module provides ML models for predicting price direction (UP/DOWN).
Key principle: NO FLAT predictions - model always outputs UP or DOWN.
"""

from typing import Dict, List, Optional, Tuple
import pickle
from pathlib import Path

import warnings

try:
    import numpy as np
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    np = None  # type: ignore
    LGBMClassifier = None  # type: ignore

from titan.core.models.base import BaseModel
from titan.core.types import ModelOutput


# Scale-invariant features to use for ML model
# Excludes absolute values like close, open, high, low, volume, ma_fast, ma_slow
ML_FEATURES = [
    # Returns and momentum (scale-invariant)
    "return_1",
    "log_return_1",
    "price_momentum_3",
    "return_5",
    "return_10",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_4",
    "return_lag_5",
    # Volatility (z-scores and ratios)
    "volatility_z",
    "vol_ratio",
    "atr_pct",
    "high_low_range_pct",
    # Volume (z-scores and ratios)
    "volume_z",
    "volume_trend",
    "volume_change_pct",
    # RSI (bounded 0-100, scale-invariant)
    "rsi",
    "rsi_momentum",
    "rsi_oversold",
    "rsi_overbought",
    "rsi_neutral",
    # Price structure (ratios, scale-invariant)
    "ma_delta_pct",
    "ema_10_spread_pct",
    "ema_20_spread_pct",
    # Candle structure (ratios)
    "body_ratio",
    "body_pct",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "candle_direction",
]


class DirectionalClassifier(BaseModel):
    """LightGBM classifier for UP/DOWN prediction.

    This classifier:
    - Uses only scale-invariant features
    - Always outputs UP or DOWN (no FLAT)
    - Can be trained on historical data
    - Provides probability estimates for confidence
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_samples: int = 500,
    ) -> None:
        self.name = "ML_CLASSIFIER"
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._num_leaves = num_leaves
        self._min_samples = min_samples

        self._model: Optional[LGBMClassifier] = None
        self._is_trained = False
        self._training_samples = 0

        # Training data accumulator
        self._X_train: List[List[float]] = []
        self._y_train: List[int] = []

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def training_samples(self) -> int:
        return self._training_samples

    def _features_to_array(self, features: Dict[str, float]) -> List[float]:
        """Convert feature dict to array in consistent order."""
        return [features.get(f, 0.0) for f in ML_FEATURES]

    def add_training_sample(self, features: Dict[str, float], direction: str) -> None:
        """Add a training sample.

        Args:
            features: Feature dictionary
            direction: "UP" or "DOWN" (FLAT is ignored)
        """
        if direction not in ("UP", "DOWN"):
            return  # Ignore FLAT

        feature_vector = self._features_to_array(features)
        label = 1 if direction == "UP" else 0

        self._X_train.append(feature_vector)
        self._y_train.append(label)
        self._training_samples = len(self._X_train)

    def train(self, max_samples: int = 10000) -> bool:
        """Train the model on accumulated samples.

        Args:
            max_samples: Maximum number of recent samples to use

        Returns:
            True if training successful, False otherwise
        """
        if not HAS_LIGHTGBM:
            return False

        if len(self._X_train) < self._min_samples:
            return False

        # Use only recent samples
        X = np.array(self._X_train[-max_samples:])
        y = np.array(self._y_train[-max_samples:])

        # Check class balance
        n_up = np.sum(y)
        n_down = len(y) - n_up
        if n_up < 10 or n_down < 10:
            return False  # Need at least 10 samples per class

        self._model = LGBMClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            num_leaves=self._num_leaves,
            objective="binary",
            metric="binary_logloss",
            verbose=-1,
            n_jobs=1,
            random_state=42,
        )

        try:
            self._model.fit(X, y)
            self._is_trained = True
            return True
        except Exception:
            self._is_trained = False
            return False

    def predict(
        self,
        features: Dict[str, float],
        pattern_context=None,
    ) -> ModelOutput:
        """Predict direction from features.

        Always returns UP or DOWN, never FLAT.
        """
        if not self._is_trained or self._model is None or not HAS_LIGHTGBM:
            # Fallback: random-ish prediction based on momentum
            momentum = features.get("price_momentum_3", 0.0)
            if momentum >= 0:
                prob_up = 0.52
            else:
                prob_up = 0.48
            prob_down = 1.0 - prob_up

            return ModelOutput(
                model_name=self.name,
                prob_up=prob_up,
                prob_down=prob_down,
                state={"trained": False, "samples": self._training_samples},
                metrics={},
            )

        # Get prediction from model (suppress sklearn feature names warning)
        feature_vector = np.array([self._features_to_array(features)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            prob = self._model.predict_proba(feature_vector)[0]

        prob_up = float(prob[1])
        prob_down = float(prob[0])

        # Ensure no exact 0.5 (prevents FLAT)
        if abs(prob_up - 0.5) < 0.001:
            # Tiny nudge based on momentum
            momentum = features.get("price_momentum_3", 0.0)
            if momentum >= 0:
                prob_up = 0.501
            else:
                prob_up = 0.499
            prob_down = 1.0 - prob_up

        return ModelOutput(
            model_name=self.name,
            prob_up=prob_up,
            prob_down=prob_down,
            state={
                "trained": True,
                "samples": self._training_samples,
            },
            metrics={
                "confidence": max(prob_up, prob_down),
            },
        )

    def save(self, path: str) -> bool:
        """Save model to file."""
        if not self._is_trained or self._model is None:
            return False

        try:
            with open(path, "wb") as f:
                pickle.dump({
                    "model": self._model,
                    "n_estimators": self._n_estimators,
                    "max_depth": self._max_depth,
                    "learning_rate": self._learning_rate,
                    "num_leaves": self._num_leaves,
                    "training_samples": self._training_samples,
                }, f)
            return True
        except Exception:
            return False

    def load(self, path: str) -> bool:
        """Load model from file."""
        if not HAS_LIGHTGBM:
            return False

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            self._model = data["model"]
            self._n_estimators = data.get("n_estimators", 100)
            self._max_depth = data.get("max_depth", 6)
            self._learning_rate = data.get("learning_rate", 0.05)
            self._num_leaves = data.get("num_leaves", 31)
            self._training_samples = data.get("training_samples", 0)
            self._is_trained = True
            return True
        except Exception:
            self._is_trained = False
            return False

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self._is_trained or self._model is None:
            return {}

        importance = self._model.feature_importances_
        # Convert numpy types to Python floats for JSON serialization
        return {f: float(v) for f, v in zip(ML_FEATURES, importance)}


def create_ml_classifier() -> Optional[DirectionalClassifier]:
    """Factory function to create ML classifier if dependencies available."""
    if not HAS_LIGHTGBM:
        return None
    return DirectionalClassifier()
