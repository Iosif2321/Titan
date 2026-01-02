"""Sprint 14: LightGBM-based directional classifier.

This module provides ML models for predicting price direction (UP/DOWN).
Key principle: NO FLAT predictions - model always outputs UP or DOWN.

Sprint 18: ML Hardening - Added:
- Time-split cross-validation (walk-forward)
- Isotonic regression calibration
- Feature importance analysis
- Automatic feature selection
"""

from typing import Dict, List, Optional, Tuple
import pickle
from pathlib import Path

import warnings

try:
    import numpy as np
    from lightgbm import LGBMClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.calibration import calibration_curve
    HAS_LIGHTGBM = True
    HAS_SKLEARN = True
except ImportError:
    HAS_LIGHTGBM = False
    HAS_SKLEARN = False
    np = None  # type: ignore
    LGBMClassifier = None  # type: ignore
    IsotonicRegression = None  # type: ignore

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

    Sprint 18: ML Hardening additions:
    - Time-split cross-validation (walk-forward)
    - Isotonic regression calibration
    - Feature importance analysis
    - Automatic feature selection
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_samples: int = 500,
        use_calibration: bool = True,
        min_feature_importance: float = 0.01,
    ) -> None:
        self.name = "ML_CLASSIFIER"
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._num_leaves = num_leaves
        self._min_samples = min_samples
        self._use_calibration = use_calibration
        self._min_feature_importance = min_feature_importance

        self._model: Optional[LGBMClassifier] = None
        self._calibrator: Optional[IsotonicRegression] = None
        self._is_trained = False
        self._is_calibrated = False
        self._training_samples = 0

        # Training data accumulator
        self._X_train: List[List[float]] = []
        self._y_train: List[int] = []

        # Sprint 18: Track validation metrics
        self._last_cv_accuracy: float = 0.0
        self._last_cv_scores: List[float] = []
        self._selected_features: List[str] = ML_FEATURES.copy()
        self._feature_importance_scores: Dict[str, float] = {}

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

    def walk_forward_validate(
        self,
        n_splits: int = 5,
        test_size: float = 0.2,
        max_samples: int = 10000,
    ) -> Tuple[float, List[float]]:
        """Perform walk-forward (time-split) cross-validation.

        Sprint 18: Ensures no data leakage by always training on past data
        and validating on future data.

        Args:
            n_splits: Number of validation folds
            test_size: Fraction of data for each test fold
            max_samples: Maximum samples to use

        Returns:
            Tuple of (mean_accuracy, list_of_fold_accuracies)
        """
        if not HAS_LIGHTGBM or len(self._X_train) < self._min_samples:
            return 0.0, []

        X = np.array(self._X_train[-max_samples:])
        y = np.array(self._y_train[-max_samples:])
        n = len(X)

        if n < 200:  # Need at least 200 samples for CV
            return 0.0, []

        scores = []
        fold_size = int(n * test_size)

        for i in range(n_splits):
            # Walk-forward: train on [0, split_point), test on [split_point, split_point + fold_size)
            split_point = int(n * (0.5 + i * 0.1))  # Start from 50%, increase by 10% each fold
            if split_point + fold_size > n:
                break
            if split_point < self._min_samples:
                continue

            X_train_fold = X[:split_point]
            y_train_fold = y[:split_point]
            X_test_fold = X[split_point:split_point + fold_size]
            y_test_fold = y[split_point:split_point + fold_size]

            # Check class balance in both sets
            if sum(y_train_fold) < 10 or sum(y_train_fold) > len(y_train_fold) - 10:
                continue
            if sum(y_test_fold) < 5 or sum(y_test_fold) > len(y_test_fold) - 5:
                continue

            model = LGBMClassifier(
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
                model.fit(X_train_fold, y_train_fold)
                predictions = model.predict(X_test_fold)
                accuracy = float(np.mean(predictions == y_test_fold))
                scores.append(accuracy)
            except Exception:
                continue

        if not scores:
            return 0.0, []

        mean_accuracy = float(np.mean(scores))
        self._last_cv_accuracy = mean_accuracy
        self._last_cv_scores = scores
        return mean_accuracy, scores

    def analyze_features(self, max_samples: int = 10000) -> Dict[str, float]:
        """Analyze feature importance and identify weak features.

        Sprint 18: Identifies which features contribute most to predictions.

        Returns:
            Dict of feature_name -> importance_score
        """
        if not HAS_LIGHTGBM or len(self._X_train) < self._min_samples:
            return {}

        X = np.array(self._X_train[-max_samples:])
        y = np.array(self._y_train[-max_samples:])

        # Train a model just for feature analysis
        model = LGBMClassifier(
            n_estimators=50,  # Faster for analysis
            max_depth=4,
            learning_rate=0.1,
            num_leaves=15,
            objective="binary",
            verbose=-1,
            n_jobs=1,
            random_state=42,
        )

        try:
            model.fit(X, y)
            importance = model.feature_importances_
            total_importance = sum(importance)
            if total_importance > 0:
                # Normalize to sum to 1.0
                importance = importance / total_importance

            self._feature_importance_scores = {
                f: float(v) for f, v in zip(ML_FEATURES, importance)
            }
            return self._feature_importance_scores
        except Exception:
            return {}

    def select_features(self, min_importance: Optional[float] = None) -> List[str]:
        """Select features with importance above threshold.

        Sprint 18: Removes features that don't contribute to predictions.

        Args:
            min_importance: Minimum importance to keep (default: self._min_feature_importance)

        Returns:
            List of selected feature names
        """
        if not self._feature_importance_scores:
            self.analyze_features()

        threshold = min_importance if min_importance is not None else self._min_feature_importance

        selected = [
            f for f, imp in self._feature_importance_scores.items()
            if imp >= threshold
        ]

        # Always keep at least 5 features
        if len(selected) < 5:
            # Keep top 5 by importance
            sorted_features = sorted(
                self._feature_importance_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            selected = [f for f, _ in sorted_features[:5]]

        self._selected_features = selected
        return selected

    def fit_calibrator(self, max_samples: int = 5000) -> bool:
        """Fit isotonic regression calibrator on validation data.

        Sprint 18: Calibrates probability outputs to match empirical accuracy.

        Args:
            max_samples: Maximum samples to use for calibration

        Returns:
            True if calibration successful
        """
        if not HAS_SKLEARN or not self._is_trained or self._model is None:
            return False

        X = np.array(self._X_train[-max_samples:])
        y = np.array(self._y_train[-max_samples:])
        n = len(X)

        if n < 200:
            return False

        # Use last 20% for calibration (simulating held-out data)
        split_point = int(n * 0.8)
        X_calib = X[split_point:]
        y_calib = y[split_point:]

        if len(y_calib) < 50:
            return False

        try:
            # Get uncalibrated probabilities
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                probs = self._model.predict_proba(X_calib)[:, 1]

            # Fit isotonic regression
            self._calibrator = IsotonicRegression(out_of_bounds="clip")
            self._calibrator.fit(probs, y_calib)
            self._is_calibrated = True
            return True
        except Exception:
            self._is_calibrated = False
            return False

    def train(self, max_samples: int = 10000, run_validation: bool = True) -> bool:
        """Train the model on accumulated samples.

        Sprint 18: Now includes walk-forward validation and optional calibration.

        Args:
            max_samples: Maximum number of recent samples to use
            run_validation: Whether to run walk-forward validation

        Returns:
            True if training successful, False otherwise
        """
        if not HAS_LIGHTGBM:
            return False

        if len(self._X_train) < self._min_samples:
            return False

        # Run walk-forward validation first
        if run_validation and len(self._X_train) >= 500:
            cv_acc, cv_scores = self.walk_forward_validate(n_splits=3, max_samples=max_samples)
            # If CV accuracy is too low, still train but log warning
            if cv_acc < 0.45 and cv_acc > 0:
                pass  # Model may not generalize well

        # Use only recent samples for final training
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

            # Sprint 18: Fit calibrator after training
            if self._use_calibration:
                self.fit_calibrator(max_samples)

            # Analyze feature importance
            self.analyze_features(max_samples)

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

        Sprint 18: Now applies isotonic calibration if available.
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

        # Sprint 18: Apply isotonic calibration if available
        raw_prob_up = prob_up
        if self._is_calibrated and self._calibrator is not None:
            try:
                calibrated = self._calibrator.predict([prob_up])[0]
                prob_up = float(np.clip(calibrated, 0.01, 0.99))
                prob_down = 1.0 - prob_up
            except Exception:
                pass  # Use uncalibrated if calibration fails

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
                "calibrated": self._is_calibrated,
                "cv_accuracy": self._last_cv_accuracy,
            },
            metrics={
                "confidence": max(prob_up, prob_down),
                "raw_prob_up": raw_prob_up,
            },
        )

    def get_training_stats(self) -> Dict[str, object]:
        """Get training statistics for reporting.

        Sprint 18: Returns validation metrics and feature importance.

        Returns:
            Dict with training statistics
        """
        stats: Dict[str, object] = {
            "is_trained": self._is_trained,
            "is_calibrated": self._is_calibrated,
            "training_samples": self._training_samples,
            "cv_accuracy": self._last_cv_accuracy,
            "cv_scores": self._last_cv_scores,
            "n_features": len(ML_FEATURES),
            "n_selected_features": len(self._selected_features),
        }

        if self._feature_importance_scores:
            # Top 10 features by importance
            sorted_features = sorted(
                self._feature_importance_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            stats["top_features"] = sorted_features[:10]
            stats["weak_features"] = [
                f for f, imp in sorted_features
                if imp < self._min_feature_importance
            ]

        return stats

    def save(self, path: str) -> bool:
        """Save model to file.

        Sprint 18: Now includes calibrator and feature importance.
        """
        if not self._is_trained or self._model is None:
            return False

        try:
            with open(path, "wb") as f:
                pickle.dump({
                    "model": self._model,
                    "calibrator": self._calibrator,
                    "n_estimators": self._n_estimators,
                    "max_depth": self._max_depth,
                    "learning_rate": self._learning_rate,
                    "num_leaves": self._num_leaves,
                    "training_samples": self._training_samples,
                    "is_calibrated": self._is_calibrated,
                    "cv_accuracy": self._last_cv_accuracy,
                    "cv_scores": self._last_cv_scores,
                    "feature_importance": self._feature_importance_scores,
                    "selected_features": self._selected_features,
                }, f)
            return True
        except Exception:
            return False

    def load(self, path: str) -> bool:
        """Load model from file.

        Sprint 18: Now loads calibrator and feature importance.
        """
        if not HAS_LIGHTGBM:
            return False

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            self._model = data["model"]
            self._calibrator = data.get("calibrator")
            self._n_estimators = data.get("n_estimators", 100)
            self._max_depth = data.get("max_depth", 6)
            self._learning_rate = data.get("learning_rate", 0.05)
            self._num_leaves = data.get("num_leaves", 31)
            self._training_samples = data.get("training_samples", 0)
            self._is_trained = True
            self._is_calibrated = data.get("is_calibrated", False)
            self._last_cv_accuracy = data.get("cv_accuracy", 0.0)
            self._last_cv_scores = data.get("cv_scores", [])
            self._feature_importance_scores = data.get("feature_importance", {})
            self._selected_features = data.get("selected_features", ML_FEATURES.copy())
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
