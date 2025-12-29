import time
from typing import Optional


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def utc_now_ts() -> int:
    return int(time.time())


def safe_div(num: float, denom: float, default: float = 0.0) -> float:
    if denom == 0:
        return default
    return num / denom


def maybe(value: Optional[float], default: float) -> float:
    if value is None:
        return default
    return value
