from collections import deque
from typing import Deque, Optional

from .config import PredictorConfig
from .features import FeatureBuilder
from .model import TwoHeadLinearModel
from .types import Candle, Direction, Prediction


def decide_direction(p_up: float, p_down: float, config: PredictorConfig) -> Direction:
    if (p_up < config.flat_max_prob and p_down < config.flat_max_prob) or abs(
        p_up - p_down
    ) < config.flat_max_delta:
        return Direction.FLAT
    return Direction.UP if p_up > p_down else Direction.DOWN


class Predictor:
    def __init__(
        self,
        feature_builder: FeatureBuilder,
        model: TwoHeadLinearModel,
        config: PredictorConfig,
    ) -> None:
        self.feature_builder = feature_builder
        self.model = model
        self.config = config
        self.candles: Deque[Candle] = deque(maxlen=feature_builder.spec.lookback)
        self.last_ts: Optional[int] = None

    def ingest(self, candle: Candle) -> Optional[Prediction]:
        if self.last_ts is not None and candle.start_ts < self.last_ts:
            return None

        if self.last_ts is not None and candle.start_ts == self.last_ts:
            if self.candles:
                self.candles[-1] = candle
            else:
                self.candles.append(candle)
        else:
            self.last_ts = candle.start_ts
            self.candles.append(candle)
        features = self.feature_builder.build(list(self.candles))
        if features is None:
            return None

        p_up, p_down = self.model.predict(features)
        direction = decide_direction(p_up, p_down, self.config)
        return Prediction(
            candle_ts=candle.start_ts,
            p_up=p_up,
            p_down=p_down,
            direction=direction,
        )
