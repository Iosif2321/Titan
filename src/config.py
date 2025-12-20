from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    symbol: str = "BTCUSDT"
    interval: str = "1"
    ws_url: str = "wss://stream.bybit.com/v5/public/spot"
    use_confirmed_only: bool = True
    reconnect_delay: float = 5.0


@dataclass(frozen=True)
class FeatureConfig:
    lookback: int = 60
    return_scale: float = 100.0
    clip_return: float = 0.02
    epsilon: float = 1e-8


@dataclass(frozen=True)
class PredictorConfig:
    flat_max_prob: float = 0.55
    flat_max_delta: float = 0.05


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
