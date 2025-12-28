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
    def __init__(self, config: ConfigStore) -> None:
        self.name = "OSCILLATOR"
        self._config = config

    def predict(self, features: Dict[str, float]) -> ModelOutput:
        rsi = features.get("rsi", 0.0)
        oversold = float(self._config.get("model.rsi_oversold", 30))
        overbought = float(self._config.get("model.rsi_overbought", 70))

        prob_up = 0.5
        prob_down = 0.5
        signal = "flat"
        strength = 0.0

        if rsi <= oversold:
            strength = clamp(safe_div(oversold - rsi, oversold, 0.0), 0.0, 1.0)
            prob_up = 0.5 + 0.5 * strength
            prob_down = 1.0 - prob_up
            signal = "up"
        elif rsi >= overbought:
            strength = clamp(safe_div(rsi - overbought, 100.0 - overbought, 0.0), 0.0, 1.0)
            prob_down = 0.5 + 0.5 * strength
            prob_up = 1.0 - prob_down
            signal = "down"

        return ModelOutput(
            model_name=self.name,
            prob_up=prob_up,
            prob_down=prob_down,
            state={"signal": signal, "strength": strength},
            metrics={"rsi": rsi},
        )


class VolumeMetrix(BaseModel):
    def __init__(self, config: ConfigStore) -> None:
        self.name = "VOLUMEMETRIX"
        self._config = config

    def predict(self, features: Dict[str, float]) -> ModelOutput:
        volume_z = features.get("volume_z", 0.0)
        ret = features.get("return_1", 0.0)
        max_z = float(self._config.get("model.volume_z_max", 3.0))
        strength = clamp(safe_div(abs(volume_z), max_z, 0.0), 0.0, 1.0)

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
