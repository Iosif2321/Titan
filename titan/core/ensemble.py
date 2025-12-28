from typing import List

from titan.core.config import ConfigStore
from titan.core.types import Decision, ModelOutput
from titan.core.utils import clamp
from titan.core.weights import WeightManager


class Ensemble:
    def __init__(self, config: ConfigStore, weights: WeightManager) -> None:
        self._config = config
        self._weights = weights

    def decide(self, outputs: List[ModelOutput]) -> Decision:
        weights = self._weights.get_model_weights()
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

        flat_threshold = float(self._config.get("ensemble.flat_threshold", 0.55))
        min_margin = float(self._config.get("ensemble.min_margin", 0.05))
        direction = "FLAT"
        if confidence >= flat_threshold and margin >= min_margin:
            direction = "UP" if prob_up >= prob_down else "DOWN"

        return Decision(
            direction=direction,
            confidence=clamp(confidence, 0.0, 1.0),
            prob_up=clamp(prob_up, 0.0, 1.0),
            prob_down=clamp(prob_down, 0.0, 1.0),
        )
