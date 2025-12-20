from dataclasses import dataclass
from enum import Enum


class Direction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    FLAT = "FLAT"


@dataclass(frozen=True)
class Candle:
    start_ts: int
    end_ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    confirmed: bool


@dataclass(frozen=True)
class Prediction:
    candle_ts: int
    p_up: float
    p_down: float
    direction: Direction
