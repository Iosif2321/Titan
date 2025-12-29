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
