"""Market regime detection for adaptive model weighting."""

from typing import Dict, Optional

from titan.core.config import ConfigStore


class RegimeDetector:
    """Detects current market regime based on features.

    Regimes:
        - trending_up: Strong upward trend (positive ma_delta, low volatility)
        - trending_down: Strong downward trend (negative ma_delta, low volatility)
        - ranging: Sideways market (small ma_delta, low volatility)
        - volatile: High volatility regardless of direction

    Each model performs differently in different regimes:
        - TrendVIC: Best in trending_up/trending_down
        - Oscillator: Best in ranging (mean reversion works)
        - VolumeMetrix: Best in volatile (volume signals matter more)
    """

    REGIMES = ["trending_up", "trending_down", "ranging", "volatile"]

    def __init__(self, config: ConfigStore) -> None:
        self._config = config

    def detect(self, features: Dict[str, float]) -> str:
        """Detect market regime from current features.

        Args:
            features: Dictionary with volatility_z, ma_delta, rsi, etc.

        Returns:
            One of: "trending_up", "trending_down", "ranging", "volatile"
        """
        vol_z = features.get("volatility_z", 0.0)
        ma_delta = features.get("ma_delta", 0.0)
        volatility = features.get("volatility", 0.0)
        close = features.get("close", 1.0)

        # Thresholds from config
        vol_z_high = float(self._config.get("regime.vol_z_high", 1.5))
        trend_threshold = float(self._config.get("regime.trend_threshold", 0.0003))

        # Normalize ma_delta by close price for comparison across price levels
        normalized_delta = abs(ma_delta) / (close + 1e-12)

        # Priority 1: High volatility regime
        if vol_z > vol_z_high:
            return "volatile"

        # Priority 2: Trending regime (strong directional movement)
        if normalized_delta > trend_threshold:
            return "trending_up" if ma_delta > 0 else "trending_down"

        # Default: Ranging (sideways, low volatility)
        return "ranging"

    def get_regime_description(self, regime: str) -> str:
        """Get human-readable description of regime."""
        descriptions = {
            "trending_up": "Upward trend - TrendVIC favored",
            "trending_down": "Downward trend - TrendVIC favored",
            "ranging": "Sideways market - Oscillator favored",
            "volatile": "High volatility - VolumeMetrix favored",
        }
        return descriptions.get(regime, "Unknown regime")

    def get_regime_weights(self, regime: str) -> Dict[str, float]:
        """Get recommended model weights for a regime.

        These are base weights that can be further adjusted by PerformanceMonitor.

        Args:
            regime: Current market regime

        Returns:
            Dictionary mapping model names to weights
        """
        # Default regime weights from config or hardcoded defaults
        default_weights = {
            "trending_up": {
                "TRENDVIC": float(self._config.get("regime.weights.trending_up.trendvic", 0.50)),
                "OSCILLATOR": float(self._config.get("regime.weights.trending_up.oscillator", 0.20)),
                "VOLUMEMETRIX": float(self._config.get("regime.weights.trending_up.volumemetrix", 0.30)),
            },
            "trending_down": {
                "TRENDVIC": float(self._config.get("regime.weights.trending_down.trendvic", 0.50)),
                "OSCILLATOR": float(self._config.get("regime.weights.trending_down.oscillator", 0.20)),
                "VOLUMEMETRIX": float(self._config.get("regime.weights.trending_down.volumemetrix", 0.30)),
            },
            "ranging": {
                "TRENDVIC": float(self._config.get("regime.weights.ranging.trendvic", 0.20)),
                "OSCILLATOR": float(self._config.get("regime.weights.ranging.oscillator", 0.50)),
                "VOLUMEMETRIX": float(self._config.get("regime.weights.ranging.volumemetrix", 0.30)),
            },
            "volatile": {
                "TRENDVIC": float(self._config.get("regime.weights.volatile.trendvic", 0.25)),
                "OSCILLATOR": float(self._config.get("regime.weights.volatile.oscillator", 0.25)),
                "VOLUMEMETRIX": float(self._config.get("regime.weights.volatile.volumemetrix", 0.50)),
            },
        }

        return default_weights.get(regime, {
            "TRENDVIC": 0.33,
            "OSCILLATOR": 0.33,
            "VOLUMEMETRIX": 0.34,
        })
