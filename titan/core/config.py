from typing import Any, Dict

from titan.core.state_store import StateStore


DEFAULT_CONFIG: Dict[str, Any] = {
    "feature.fast_window": 5,
    "feature.slow_window": 20,
    "feature.rsi_window": 14,
    "feature.vol_window": 20,
    "feature.volume_window": 20,
    "model.flat_threshold": 0.55,
    "model.rsi_oversold": 30,
    "model.rsi_overbought": 70,
    "model.volume_z_max": 3.0,
    "pattern.trend_eps_mult": 0.5,
    "pattern.vol_z_high": 1.0,
    "pattern.vol_z_low": -1.0,
    "pattern.volume_z_high": 1.0,
    "pattern.volume_z_low": -1.0,
    # Sprint 13: Pattern experience adjustment parameters
    "pattern.boost_threshold": 0.55,
    "pattern.penalty_threshold": 0.45,
    "pattern.max_boost": 0.03,
    "pattern.max_penalty": 0.03,
    "pattern.bias_penalty": 0.01,
    "pattern.bias_threshold": 0.05,
    # Sprint 12: Enhanced pattern system
    "pattern.conditions_version": 2,
    "pattern.hour_bucket_size": 1,  # Each hour separately (0-23)
    "pattern.snapshot_rate": 1.0,  # Store full snapshots for ALL events
    "pattern.high_conf_threshold": 0.65,
    "pattern.max_decisions": 50000,  # Max decisions per pattern
    "pattern.top_decisions_count": 1000,  # Always keep top N brightest
    "pattern.min_match_ratio": 0.8,  # 80% condition match for fuzzy search
    "pattern.inactive_after_days": 30,  # Move to inactive after 30 days unused
    "pattern.delete_after_days": 90,  # Delete after 90 days in inactive
    "pattern.model_adjuster_enabled": False,
    "pattern.include_ensemble_in_global": False,
    # PatternReader adjustments
    "pattern_reader.max_confidence_boost": 0.03,
    "pattern_reader.max_confidence_penalty": 0.05,
    "pattern_reader.bias_flip_threshold": 0.15,
    # Sprint 14: ML Classifier (LightGBM)
    "ml.enabled": True,
    "ml.min_samples": 500,  # Minimum samples before training
    "ml.train_interval": 1000,  # Retrain every N samples
    "ml.n_estimators": 100,
    "ml.max_depth": 6,
    "ml.learning_rate": 0.05,
    "ml.num_leaves": 31,
    # Sprint 15: Confidence Filtering
    "confidence_compressor.max_confidence": 0.70,  # Increased from 0.62
    "confidence_filter.threshold": 0.55,  # Min confidence for "actionable" predictions
    "confidence_filter.enabled": True,
    "ensemble.flat_threshold": 0.55,
    "ensemble.min_margin": 0.05,
    "ensemble.vol_z_high": 1.0,
    "ensemble.trendvic_high_scale": 0.6,
    "ensemble.high_vol_flat_add": 0.05,
    "ensemble.high_vol_margin_add": 0.02,
    "calibration.enabled": True,
    "calibration.min_samples": 30,
    "calibration.blend": 0.7,
    "calibration.decay": 1.0,
    "weights.min_weight": 0.1,
}


class ConfigStore:
    def __init__(self, state_store: StateStore) -> None:
        self._state = state_store

    def ensure_defaults(self) -> None:
        for key, value in DEFAULT_CONFIG.items():
            self._state.set_if_missing(f"config.{key}", value)

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(f"config.{key}", default)

    def set(self, key: str, value: Any) -> None:
        self._state.set(f"config.{key}", value)
