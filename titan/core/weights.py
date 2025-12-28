from typing import Dict

from titan.core.state_store import StateStore


DEFAULT_MODEL_WEIGHTS: Dict[str, float] = {
    "TRENDVIC": 1.0,
    "OSCILLATOR": 1.0,
    "VOLUMEMETRIX": 1.0,
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
