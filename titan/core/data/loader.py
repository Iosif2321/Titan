import csv
from typing import Iterator, Optional

from titan.core.data.schema import Candle, parse_timestamp


class CsvCandleReader:
    def __init__(self, path: str, limit: Optional[int] = None) -> None:
        self._path = path
        self._limit = limit

    def __iter__(self) -> Iterator[Candle]:
        with open(self._path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            count = 0
            for row in reader:
                yield Candle(
                    ts=parse_timestamp(row["timestamp"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
                count += 1
                if self._limit is not None and count >= self._limit:
                    break
