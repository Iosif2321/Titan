import json
from pathlib import Path
from typing import Any, Dict

from .types import Candle, Prediction


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._fp = path.open("a", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        self._fp.write(payload + "\n")
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()


class JsonlRecorder:
    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._candles = JsonlWriter(directory / "candles.jsonl")
        self._predictions = JsonlWriter(directory / "predictions.jsonl")

    def record_candle(self, candle: Candle) -> None:
        self._candles.write(
            {
                "ts_start": candle.start_ts,
                "ts_end": candle.end_ts,
                "o": candle.open,
                "h": candle.high,
                "l": candle.low,
                "c": candle.close,
                "volume": candle.volume,
                "confirmed": candle.confirmed,
            }
        )

    def record_prediction(self, prediction: Prediction) -> None:
        self._predictions.write(
            {
                "candle_ts": prediction.candle_ts,
                "p_up": prediction.p_up,
                "p_down": prediction.p_down,
                "direction": prediction.direction.value,
            }
        )

    def close(self) -> None:
        self._candles.close()
        self._predictions.close()
