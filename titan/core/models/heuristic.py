from typing import Dict

from titan.core.config import ConfigStore
from titan.core.models.base import BaseModel
from titan.core.types import ModelOutput
from titan.core.utils import clamp, safe_div


class TrendVIC(BaseModel):
    def __init__(self, config: ConfigStore) -> None:
        self.name = "TRENDVIC"
        self._config = config

    def predict(self, features: Dict[str, float]) -> ModelOutput:
        ma_delta = features.get("ma_delta", 0.0)
        volatility = features.get("volatility", 0.0)
        close = features.get("close", 0.0)
        scale = max(volatility * close, 1e-12)
        strength = clamp(abs(ma_delta) / scale, 0.0, 1.0)

        if ma_delta >= 0:
            prob_up = 0.5 + 0.5 * strength
            prob_down = 1.0 - prob_up
            signal = "up"
        else:
            prob_down = 0.5 + 0.5 * strength
            prob_up = 1.0 - prob_down
            signal = "down"

        return ModelOutput(
            model_name=self.name,
            prob_up=prob_up,
            prob_down=prob_down,
            state={"signal": signal, "strength": strength},
            metrics={"ma_delta": ma_delta, "volatility": volatility},
        )


class Oscillator(BaseModel):
    """RSI-based mean reversion model.

    Always predicts UP or DOWN (never FLAT) based on RSI deviation from 50.
    RSI < 50 -> expect rise (UP), RSI > 50 -> expect fall (DOWN).
    """

    def __init__(self, config: ConfigStore) -> None:
        self.name = "OSCILLATOR"
        self._config = config

    def predict(self, features: Dict[str, float]) -> ModelOutput:
        rsi = features.get("rsi", 50.0)

        # Mean reversion: RSI < 50 -> expect UP, RSI > 50 -> expect DOWN
        # Strength proportional to deviation from 50
        deviation = (rsi - 50.0) / 50.0  # -1 to +1

        # Non-linear strength: amplify extreme values
        # abs(deviation)^0.7 makes weak signals slightly stronger
        strength = abs(deviation) ** 0.7

        # Minimum strength to avoid pure 0.5/0.5 (which would be FLAT)
        min_strength = 0.05
        strength = max(strength, min_strength)
        strength = min(strength, 1.0)

        # RSI < 50: expect price to rise (mean reversion UP)
        # RSI > 50: expect price to fall (mean reversion DOWN)
        if rsi <= 50:
            prob_up = 0.5 + 0.5 * strength
            prob_down = 1.0 - prob_up
            signal = "up"
        else:
            prob_down = 0.5 + 0.5 * strength
            prob_up = 1.0 - prob_down
            signal = "down"

        return ModelOutput(
            model_name=self.name,
            prob_up=prob_up,
            prob_down=prob_down,
            state={"signal": signal, "strength": strength, "deviation": deviation},
            metrics={"rsi": rsi},
        )


class VolumeMetrix(BaseModel):
    """Volume-informed momentum model.

    Always predicts UP or DOWN (never FLAT) based on return_1 direction.
    Strength is modulated by volume_z - higher volume = higher confidence.
    """

    def __init__(self, config: ConfigStore) -> None:
        self.name = "VOLUMEMETRIX"
        self._config = config

    def predict(self, features: Dict[str, float]) -> ModelOutput:
        volume_z = features.get("volume_z", 0.0)
        ret = features.get("return_1", 0.0)
        max_z = float(self._config.get("model.volume_z_max", 3.0))

        # Base strength from volume - higher volume = more confidence
        strength = clamp(safe_div(abs(volume_z), max_z, 0.0), 0.0, 1.0)

        # Minimum strength to avoid FLAT (ensures max(prob) >= 0.55)
        min_strength = 0.15
        strength = max(strength, min_strength)

        # Direction from return_1 - always UP or DOWN
        if ret >= 0:
            prob_up = 0.5 + 0.5 * strength
            prob_down = 1.0 - prob_up
            signal = "up"
        else:
            prob_down = 0.5 + 0.5 * strength
            prob_up = 1.0 - prob_down
            signal = "down"

        return ModelOutput(
            model_name=self.name,
            prob_up=prob_up,
            prob_down=prob_down,
            state={"signal": signal, "strength": strength},
            metrics={"volume_z": volume_z, "return_1": ret},
        )
