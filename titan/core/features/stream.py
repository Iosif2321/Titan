import math
from collections import deque
from typing import Dict, Optional

from titan.core.config import ConfigStore
from titan.core.data.schema import Candle


class RollingStats:
    def __init__(self, maxlen: int) -> None:
        self._values = deque(maxlen=maxlen)
        self._sum = 0.0
        self._sumsq = 0.0

    @property
    def maxlen(self) -> int:
        return self._values.maxlen  # type: ignore[return-value]

    def append(self, value: float) -> None:
        if len(self._values) == self.maxlen:
            old = self._values.popleft()
            self._sum -= old
            self._sumsq -= old * old
        self._values.append(value)
        self._sum += value
        self._sumsq += value * value

    def mean(self) -> Optional[float]:
        if not self._values:
            return None
        return self._sum / len(self._values)

    def std(self) -> Optional[float]:
        n = len(self._values)
        if n < 2:
            return None
        mean = self._sum / n
        var = (self._sumsq / n) - (mean * mean)
        return math.sqrt(max(var, 0.0))

    def ready(self) -> bool:
        return len(self._values) >= self.maxlen


class RollingSum:
    def __init__(self, maxlen: int) -> None:
        self._values = deque(maxlen=maxlen)
        self._sum = 0.0

    @property
    def maxlen(self) -> int:
        return self._values.maxlen  # type: ignore[return-value]

    def append(self, value: float) -> None:
        if len(self._values) == self.maxlen:
            old = self._values.popleft()
            self._sum -= old
        self._values.append(value)
        self._sum += value

    def mean(self) -> Optional[float]:
        if not self._values:
            return None
        return self._sum / len(self._values)

    def ready(self) -> bool:
        return len(self._values) >= self.maxlen


class FeatureStream:
    def __init__(self, config: ConfigStore) -> None:
        self._config = config
        self._fast_window = int(config.get("feature.fast_window", 5))
        self._slow_window = int(config.get("feature.slow_window", 20))
        self._rsi_window = int(config.get("feature.rsi_window", 14))
        self._vol_window = int(config.get("feature.vol_window", 20))
        self._volume_window = int(config.get("feature.volume_window", 20))

        self._price_fast = RollingStats(self._fast_window)
        self._price_slow = RollingStats(self._slow_window)
        self._returns = RollingStats(self._vol_window)
        self._volume = RollingStats(self._volume_window)
        self._volatility = RollingStats(self._vol_window)
        self._gains = RollingSum(self._rsi_window)
        self._losses = RollingSum(self._rsi_window)

        self._prev_close: Optional[float] = None

    def update(self, candle: Candle) -> Optional[Dict[str, float]]:
        if self._prev_close is None:
            self._prev_close = candle.close
            self._price_fast.append(candle.close)
            self._price_slow.append(candle.close)
            self._volume.append(candle.volume)
            return None

        ret = (candle.close / self._prev_close) - 1.0
        log_ret = math.log(candle.close / self._prev_close)
        self._prev_close = candle.close

        self._price_fast.append(candle.close)
        self._price_slow.append(candle.close)
        self._returns.append(ret)
        self._volume.append(candle.volume)

        gain = max(ret, 0.0)
        loss = max(-ret, 0.0)
        self._gains.append(gain)
        self._losses.append(loss)

        ma_fast = self._price_fast.mean() if self._price_fast.ready() else None
        ma_slow = self._price_slow.mean() if self._price_slow.ready() else None
        ma_delta = None
        if ma_fast is not None and ma_slow is not None:
            ma_delta = ma_fast - ma_slow

        volatility = self._returns.std() if self._returns.ready() else None
        if volatility is not None:
            self._volatility.append(volatility)
        vol_z = None
        vol_mean = self._volatility.mean()
        vol_std = self._volatility.std()
        if volatility is not None and vol_mean is not None and vol_std:
            vol_z = (volatility - vol_mean) / vol_std

        volume_mean = self._volume.mean()
        volume_std = self._volume.std()
        volume_z = None
        if volume_mean is not None and volume_std:
            volume_z = (candle.volume - volume_mean) / volume_std

        rsi = None
        if self._gains.ready() and self._losses.ready():
            avg_gain = self._gains.mean() or 0.0
            avg_loss = self._losses.mean() or 0.0
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))

        if not (
            self._price_slow.ready()
            and self._returns.ready()
            and self._volume.ready()
            and self._gains.ready()
            and self._losses.ready()
        ):
            return None

        return {
            "close": candle.close,
            "return_1": ret,
            "log_return_1": log_ret,
            "ma_fast": ma_fast or 0.0,
            "ma_slow": ma_slow or 0.0,
            "ma_delta": ma_delta or 0.0,
            "volatility": volatility or 0.0,
            "volatility_z": vol_z or 0.0,
            "volume": candle.volume,
            "volume_z": volume_z or 0.0,
            "rsi": rsi or 0.0,
        }


def build_conditions(features: Dict[str, float], config: ConfigStore) -> Dict[str, str]:
    trend_eps_mult = float(config.get("pattern.trend_eps_mult", 0.5))
    trend_eps = abs(features.get("volatility", 0.0) * features.get("close", 0.0)) * trend_eps_mult
    ma_delta = features.get("ma_delta", 0.0)

    if ma_delta > trend_eps:
        trend = "up"
    elif ma_delta < -trend_eps:
        trend = "down"
    else:
        trend = "flat"

    vol_z = features.get("volatility_z", 0.0)
    vol_high = float(config.get("pattern.vol_z_high", 1.0))
    vol_low = float(config.get("pattern.vol_z_low", -1.0))
    if vol_z >= vol_high:
        volatility = "high"
    elif vol_z <= vol_low:
        volatility = "low"
    else:
        volatility = "mid"

    volume_z = features.get("volume_z", 0.0)
    volume_high = float(config.get("pattern.volume_z_high", 1.0))
    volume_low = float(config.get("pattern.volume_z_low", -1.0))
    if volume_z >= volume_high:
        volume = "high"
    elif volume_z <= volume_low:
        volume = "low"
    else:
        volume = "mid"

    return {
        "trend": trend,
        "volatility": volatility,
        "volume": volume,
    }
