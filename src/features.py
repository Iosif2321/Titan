from dataclasses import dataclass
from typing import List, Optional

import jax.numpy as jnp
import numpy as np

from .config import FeatureConfig
from .types import Candle


@dataclass(frozen=True)
class FeatureSpec:
    lookback: int
    n_returns: int
    n_extra: int
    input_size: int


class FeatureBuilder:
    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        n_returns = max(config.lookback - 1, 0)
        n_extra = 3
        self.spec = FeatureSpec(
            lookback=config.lookback,
            n_returns=n_returns,
            n_extra=n_extra,
            input_size=n_returns + n_extra,
        )

    def build(self, candles: List[Candle]) -> Optional[jnp.ndarray]:
        if len(candles) < self.config.lookback:
            return None

        window = candles[-self.config.lookback :]
        opens = np.array([c.open for c in window], dtype=np.float32)
        highs = np.array([c.high for c in window], dtype=np.float32)
        lows = np.array([c.low for c in window], dtype=np.float32)
        closes = np.array([c.close for c in window], dtype=np.float32)
        volumes = np.array([c.volume for c in window], dtype=np.float32)

        returns = np.log(closes[1:] / closes[:-1])
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        returns = np.clip(returns, -self.config.clip_return, self.config.clip_return)
        returns = returns * self.config.return_scale

        eps = self.config.epsilon
        range_pct = (highs[-1] - lows[-1]) / max(closes[-1], eps)
        body_pct = (closes[-1] - opens[-1]) / max(opens[-1], eps)
        vol_z = (volumes[-1] - volumes.mean()) / (volumes.std() + eps)

        extras = np.array(
            [range_pct * 100.0, body_pct * 100.0, vol_z], dtype=np.float32
        )

        features = np.concatenate([returns.astype(np.float32), extras])
        return jnp.asarray(features, dtype=jnp.float32)
