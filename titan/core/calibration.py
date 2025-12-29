from typing import Dict, List, Optional, Tuple

from titan.core.config import ConfigStore
from titan.core.state_store import StateStore
from titan.core.utils import clamp


# Regime confidence multipliers - reduce confidence in problematic regimes
REGIME_CONFIDENCE_MULTIPLIERS = {
    "trending_up": 1.0,      # Best regime (54% acc), keep confidence
    "ranging": 0.95,         # Good regime (53.2% acc), slight reduction
    "trending_down": 0.85,   # Problematic (46.1% acc), reduce more
    "volatile": 0.75,        # Worst regime (43.7% acc), strong reduction
}


class ConfidenceCompressor:
    """Compresses overconfident predictions towards 50%.

    Problem: High confidence (65%+) correlates with LOWER accuracy (33-47%).
    Solution: Compress confidence into a smaller range [0.50, max_confidence].
    """

    def __init__(self, config: ConfigStore) -> None:
        self._config = config

    def _enabled(self) -> bool:
        return bool(self._config.get("confidence_compressor.enabled", True))

    def _max_confidence(self) -> float:
        return float(self._config.get("confidence_compressor.max_confidence", 0.62))

    def compress(self, confidence: float) -> float:
        """Compress confidence into range [0.50, max_confidence].

        Maps [0.5, 1.0] -> [0.5, max_confidence] linearly.
        """
        if not self._enabled():
            return confidence

        if confidence <= 0.5:
            return 0.5

        max_conf = self._max_confidence()
        # Linear compression: [0.5, 1.0] -> [0.5, max_confidence]
        excess = confidence - 0.5
        compressed_excess = excess * (max_conf - 0.5) / 0.5
        return clamp(0.5 + compressed_excess, 0.5, max_conf)

    def apply_regime_penalty(self, confidence: float, regime: Optional[str]) -> float:
        """Apply confidence penalty based on regime.

        Reduces confidence in regimes where accuracy is historically low.
        """
        if not regime:
            return confidence

        multiplier = REGIME_CONFIDENCE_MULTIPLIERS.get(regime, 1.0)
        # Don't let confidence drop below 50%
        return max(0.50, confidence * multiplier)


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
