from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from .config import PredictorConfig
from .features import FeatureBuilder
from .model import TwoHeadLinearModel
from .types import Candle, Direction, FeatureVector, Prediction
from .utils import interval_to_ms


def decide_direction(p_up: float, p_down: float, config: PredictorConfig) -> Direction:
    if (p_up < config.flat_max_prob and p_down < config.flat_max_prob) or abs(
        p_up - p_down
    ) < config.flat_max_delta:
        return Direction.FLAT
    return Direction.UP if p_up > p_down else Direction.DOWN


@dataclass(frozen=True)
class PredictionBundle:
    prediction: Prediction
    features: FeatureVector


class Predictor:
    def __init__(
        self,
        feature_builder: FeatureBuilder,
        model: TwoHeadLinearModel,
        config: PredictorConfig,
        model_id: str,
        interval: str,
    ) -> None:
        self.feature_builder = feature_builder
        self.model = model
        self.config = config
        self.candles: Deque[Candle] = deque(maxlen=feature_builder.spec.lookback)
        self.last_ts: Optional[int] = None
        self.model_id = model_id
        self.interval = interval

    def warm_start(self, candles: list[Candle]) -> None:
        if not candles:
            return
        self.candles.clear()
        for candle in candles[-self.feature_builder.spec.lookback :]:
            self.candles.append(candle)
        self.last_ts = self.candles[-1].start_ts

    def ingest(self, candle: Candle) -> Optional[PredictionBundle]:
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

        p_up, p_down = self.model.predict(features.values)
        direction = decide_direction(p_up, p_down, self.config)
        confidence = max(p_up, p_down)
        target_ts = candle.start_ts + interval_to_ms(self.interval)
        prediction = Prediction(
            candle_ts=candle.start_ts,
            target_ts=target_ts,
            model_id=self.model_id,
            interval=self.interval,
            p_up=p_up,
            p_down=p_down,
            direction=direction,
            confidence=confidence,
            meta={
                "flat_max_prob": self.config.flat_max_prob,
                "flat_max_delta": self.config.flat_max_delta,
                "delta": abs(p_up - p_down),
                "feature_schema": features.schema_version,
            },
        )
        return PredictionBundle(prediction=prediction, features=features)
