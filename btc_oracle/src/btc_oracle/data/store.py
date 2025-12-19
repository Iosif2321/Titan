"""In-memory хранилище свечей и прогнозов."""

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple

from btc_oracle.core.types import Candle, Decision


class DataStore:
    """Простое in-memory хранилище, совместимое с API приложения."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._candles: DefaultDict[Tuple[str, str], List[Candle]] = defaultdict(list)
        self._predictions: DefaultDict[Tuple[str, int], List[Dict]] = defaultdict(list)

    def add_candle(self, candle: Candle, symbol: str, timeframe: str) -> None:
        """Добавить или обновить свечу."""
        key = (symbol, timeframe)
        bucket = self._candles[key]

        inserted = False
        for idx, existing in enumerate(bucket):
            if existing.timestamp == candle.timestamp:
                bucket[idx] = candle
                inserted = True
                break
            if existing.timestamp > candle.timestamp:
                bucket.insert(idx, candle)
                inserted = True
                break

        if not inserted:
            bucket.append(candle)

    def add_candles_batch(self, candles: List[Candle], symbol: str, timeframe: str) -> None:
        """Добавить батч свечей."""
        for candle in candles:
            self.add_candle(candle, symbol, timeframe)

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        end_ts: Optional[datetime] = None,
        window_size: int = 100,
    ) -> List[Candle]:
        """Вернуть последнее окно свечей до end_ts (включительно)."""
        key = (symbol, timeframe)
        bucket = self._candles.get(key, [])
        end_ms = int(end_ts.timestamp() * 1000) if end_ts else None

        filtered = (
            [c for c in bucket if end_ms is None or c.timestamp <= end_ms]
            if bucket
            else []
        )
        return filtered[-window_size:]

    def get_candle_at(self, symbol: str, timeframe: str, ts: datetime) -> Optional[Candle]:
        """Получить свечу по точному времени."""
        target_ms = int(ts.timestamp() * 1000)
        for candle in self._candles.get((symbol, timeframe), []):
            if candle.timestamp == target_ms:
                return candle
        return None

    def add_prediction(self, decision: Decision) -> None:
        """Сохранить прогноз."""
        record = {
            "ts": int(decision.ts.timestamp() * 1000),
            "symbol": decision.symbol,
            "horizon_min": decision.horizon_min,
            "label": decision.label.value,
            "reason_code": decision.reason_code.value,
            "p_up": decision.p_up,
            "p_down": decision.p_down,
            "p_flat": decision.p_flat,
            "flat_score": decision.flat_score,
            "uncertainty_score": decision.uncertainty_score,
            "consensus": decision.consensus,
            "latency_ms": decision.latency_ms,
            "memory": decision.memory,
            "matured_at": None,
            "truth_label": None,
            "truth_magnitude": None,
            "reward": None,
        }
        self._predictions[(decision.symbol, decision.horizon_min)].append(record)

    def get_latest_prediction(self, symbol: str, horizon_min: int) -> Optional[Dict]:
        """Последний прогноз по символу и горизонту."""
        preds = self._predictions.get((symbol, horizon_min), [])
        return preds[-1] if preds else None

    def get_pending_predictions(
        self,
        symbol: str,
        horizon_min: int,
        current_ts: int,
    ) -> List[Dict]:
        """Вернуть прогнозы, которые должны созреть к текущему времени."""
        pending: list[Dict] = []
        for pred in self._predictions.get((symbol, horizon_min), []):
            if pred["matured_at"] is not None:
                continue
            maturity_ts = pred["ts"] + horizon_min * 60 * 1000
            if maturity_ts <= current_ts:
                pending.append(pred)
        return pending

    def mark_prediction_matured(
        self,
        ts: int,
        symbol: str,
        horizon_min: int,
        truth_label: str,
        truth_magnitude: float,
        reward: float,
    ) -> None:
        """Отметить прогноз как созревший."""
        for pred in self._predictions.get((symbol, horizon_min), []):
            if pred["ts"] == ts:
                pred["truth_label"] = truth_label
                pred["truth_magnitude"] = truth_magnitude
                pred["reward"] = reward
                pred["matured_at"] = int(datetime.now().timestamp() * 1000)
                return
