import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class Candle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def iso_time(self) -> str:
        return dt.datetime.utcfromtimestamp(self.ts).isoformat() + "Z"


def parse_timestamp(value: str) -> int:
    ts = int(float(value))
    if ts > 10_000_000_000:
        ts //= 1000
    return ts
