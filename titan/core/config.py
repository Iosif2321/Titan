from typing import Any, Dict

from titan.core.state_store import StateStore


DEFAULT_CONFIG: Dict[str, Any] = {
    # Sprint 22: Optuna-optimized parameters (53.32% accuracy - Trial 50/50)
    "feature.fast_window": 3,  # Was 5 - faster response
    "feature.slow_window": 15,  # Was 20 - optimized trend detection
    "feature.rsi_window": 17,  # Was 14 - optimized RSI
    "feature.vol_window": 20,
    "feature.volume_window": 20,
    "model.flat_threshold": 0.566,  # Optuna: 0.5661
    "model.rsi_oversold": 27,  # Optuna optimized
    "model.rsi_overbought": 67,  # Optuna optimized
    "model.volume_z_max": 3.0,
    "pattern.trend_eps_mult": 0.5,
    "pattern.vol_z_high": 1.0,
    "pattern.vol_z_low": -1.0,
    "pattern.volume_z_high": 1.0,
    "pattern.volume_z_low": -1.0,
    # Sprint 13: Pattern experience adjustment parameters (Optuna Trial 50)
    "pattern.boost_threshold": 0.551,  # Optuna: 0.5509
    "pattern.penalty_threshold": 0.413,  # Optuna: 0.4134
    "pattern.max_boost": 0.033,  # Optuna: 0.0325
    "pattern.max_penalty": 0.045,  # Optuna: 0.0451
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
    # Sprint 14: ML Classifier (LightGBM) - Optuna Trial 50
    "ml.enabled": True,
    "ml.min_samples": 500,  # Minimum samples before training
    "ml.train_interval": 1000,  # Retrain every N samples
    "ml.n_estimators": 196,  # Optuna optimized
    "ml.max_depth": 7,  # Optuna optimized
    "ml.learning_rate": 0.065,  # Optuna: 0.0654
    "ml.num_leaves": 31,  # Optuna optimized
    # Sprint 15: Confidence Filtering (Optuna Trial 50)
    "confidence_compressor.max_confidence": 0.667,  # Optuna: 0.6666
    "confidence_filter.threshold": 0.565,  # Optuna: 0.5649
    "confidence_filter.enabled": True,
    # Sprint 17: Session Adapter
    "session_adapter.enabled": True,
    "session_adapter.weight_update_freq": 50,
    "session_adapter.param_update_freq": 500,
    "session_adapter.calibration_update_freq": 100,
    "session_adapter.min_samples": 50,
    "session_adapter.max_ci_width": 0.10,
    "session_adapter.half_life_hours": 168.0,
    "session_adapter.prior_strength": 1000,
    "session_adapter.min_weight": 0.10,
    "session_adapter.max_weight": 0.50,
    # Sprint 20: Online Learning (Optuna Trial 50)
    "online.enabled": True,
    "online.learning_rate": 0.049,  # Optuna: 0.0490
    "online.min_weight": 0.072,  # Optuna: 0.0718
    "online.max_weight": 0.484,  # Optuna: 0.4835
    "online.reward_type": "confidence",  # binary, confidence, return, risk_adjusted
    "online.exploration_rate": 0.1,
    "ensemble.flat_threshold": 0.571,  # Optuna: 0.5711
    "ensemble.min_margin": 0.062,  # Optuna: 0.0619
    "ensemble.vol_z_high": 1.0,
    "ensemble.trendvic_high_scale": 0.6,
    "ensemble.high_vol_flat_add": 0.05,
    "ensemble.high_vol_margin_add": 0.02,
    "calibration.enabled": True,
    "calibration.min_samples": 30,
    "calibration.blend": 0.7,
    "calibration.decay": 1.0,
    "weights.min_weight": 0.1,
    # Sprint 21: Transformer Fusion (disabled by default - needs more training data)
    "fusion.enabled": False,
    "fusion.hidden_dim": 32,
    "fusion.num_heads": 2,
    "fusion.dropout": 0.2,
    "fusion.learning_rate": 0.001,
    "fusion.l2_lambda": 0.01,
    "fusion.warmup_steps": 100,
    "fusion.min_samples": 200,
    "fusion.val_split": 0.2,
    "fusion.gradient_clip": 1.0,
    "fusion.early_stopping_patience": 50,
    "fusion.early_stopping_delta": 0.001,
    # Hyperparameter Tuning
    "tuner.n_trials": 100,
    "tuner.timeout_per_trial": 300,
    "tuner.study_name": "titan_optimization",
    "tuner.pruner": "median",  # median, hyperband, none
    "tuner.objective": "accuracy",  # accuracy, sharpe, multi
    "tuner.ece_constraint": 0.05,
    "tuner.min_predictions": 100,
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
