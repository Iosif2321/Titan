"""Утилиты для чтения данных из DataStore."""

from datetime import datetime, timedelta
from typing import List, Optional

from btc_oracle.core.types import Candle
from btc_oracle.data.store import DataStore


class DataRepository:
    """Интерфейс для выборок свечей и прогонов."""

    def __init__(self, store: DataStore):
        self.store = store

    def get_feature_window(
        self,
        symbol: str,
        timeframe: str,
        end_ts: datetime,
        window_size: int = 100,
    ) -> List[Candle]:
        """Вернуть окно свечей, заканчивающееся end_ts."""
        return self.store.get_candles(
            symbol=symbol,
            timeframe=timeframe,
            end_ts=end_ts,
            window_size=window_size,
        )

    def get_truth_candle(
        self,
        symbol: str,
        timeframe: str,
        forecast_ts: datetime,
        horizon_min: int,
    ) -> Optional[Candle]:
        """Получить свечу на горизонте прогнозирования."""
        target_ts = forecast_ts + timedelta(minutes=horizon_min)
        return self.store.get_candle_at(symbol, timeframe, target_ts)
