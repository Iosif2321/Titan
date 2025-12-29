from typing import Dict, List, Optional, Tuple

from titan.core.config import ConfigStore
from titan.core.state_store import StateStore
from titan.core.utils import clamp


class OnlineCalibrator:
    def __init__(
        self,
        config: ConfigStore,
        state_store: Optional[StateStore] = None,
        state_key: str = "calibration.state",
    ) -> None:
        self._config = config
        self._state_store = state_store
        self._state_key = state_key
        self._bins = self._load_bins()

    def _default_bins(self) -> List[Dict[str, float]]:
        raw_bins = self._config.get(
            "calibration.bins",
            [
                [0.50, 0.55],
                [0.55, 0.60],
                [0.60, 0.65],
                [0.65, 0.70],
                [0.70, 0.80],
                [0.80, 1.01],
            ],
        )
        bins: List[Dict[str, float]] = []
        for low, high in raw_bins:
            bins.append({"low": float(low), "high": float(high), "correct": 0.0, "total": 0.0})
        return bins

    def _load_bins(self) -> List[Dict[str, float]]:
        if self._state_store is None:
            return self._default_bins()
        stored = self._state_store.get(self._state_key)
        if not isinstance(stored, dict):
            return self._default_bins()
        bins = stored.get("bins")
        if not isinstance(bins, list):
            return self._default_bins()
        cleaned: List[Dict[str, float]] = []
        for item in bins:
            if not isinstance(item, dict):
                continue
            try:
                cleaned.append(
                    {
                        "low": float(item["low"]),
                        "high": float(item["high"]),
                        "correct": float(item.get("correct", 0.0)),
                        "total": float(item.get("total", 0.0)),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        return cleaned or self._default_bins()

    def _save_bins(self) -> None:
        if self._state_store is None:
            return
        self._state_store.set(self._state_key, {"bins": self._bins})

    def _enabled(self) -> bool:
        return bool(self._config.get("calibration.enabled", True))

    def _bin_index(self, confidence: float) -> int:
        for idx, item in enumerate(self._bins):
            if confidence >= item["low"] and confidence < item["high"]:
                return idx
        return max(len(self._bins) - 1, 0)

    def calibrate(self, confidence: float) -> float:
        if not self._enabled():
            return clamp(confidence, 0.0, 1.0)
        idx = self._bin_index(confidence)
        bin_info = self._bins[idx]
        total = bin_info["total"]
        min_samples = float(self._config.get("calibration.min_samples", 30))
        if total < min_samples or total <= 0:
            return clamp(confidence, 0.0, 1.0)
        accuracy = bin_info["correct"] / total
        blend = clamp(float(self._config.get("calibration.blend", 0.7)), 0.0, 1.0)
        return clamp(confidence + (accuracy - confidence) * blend, 0.0, 1.0)

    def update(self, confidence: float, is_correct: bool) -> None:
        if not self._enabled():
            return
        decay = clamp(float(self._config.get("calibration.decay", 1.0)), 0.0, 1.0)
        if decay < 1.0:
            for item in self._bins:
                item["correct"] *= decay
                item["total"] *= decay
        idx = self._bin_index(confidence)
        self._bins[idx]["total"] += 1.0
        if is_correct:
            self._bins[idx]["correct"] += 1.0
        self._save_bins()

    def summary(self) -> Dict[str, object]:
        summary_bins = []
        for item in self._bins:
            total = item["total"]
            accuracy = (item["correct"] / total) if total else 0.0
            summary_bins.append(
                {
                    "low": item["low"],
                    "high": item["high"],
                    "total": round(total, 3),
                    "accuracy": round(accuracy, 4),
                }
            )
        return {"bins": summary_bins}
