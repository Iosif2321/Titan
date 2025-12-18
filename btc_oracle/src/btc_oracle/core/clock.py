"""Управление "созреванием" прогнозов по горизонтам."""

from datetime import datetime, timedelta
from typing import Optional

from btc_oracle.core.types import Label


class HorizonClock:
    """Отслеживание времени создания и созревания прогнозов."""
    
    def __init__(self, horizons: list[int]):
        """
        Args:
            horizons: список горизонтов в минутах
        """
        self.horizons = sorted(horizons)
    
    def is_mature(self, forecast_ts: datetime, current_ts: datetime, horizon_min: int) -> bool:
        """Проверка, созрел ли прогноз для горизонта."""
        elapsed = (current_ts - forecast_ts).total_seconds() / 60
        return elapsed >= horizon_min
    
    def get_mature_horizons(
        self, forecast_ts: datetime, current_ts: datetime
    ) -> list[int]:
        """Получить список созревших горизонтов."""
        mature = []
        for horizon in self.horizons:
            if self.is_mature(forecast_ts, current_ts, horizon):
                mature.append(horizon)
        return mature
    
    def get_truth_timestamp(self, forecast_ts: datetime, horizon_min: int) -> datetime:
        """Получить timestamp, когда прогноз должен созреть."""
        return forecast_ts + timedelta(minutes=horizon_min)
    
    def get_next_maturity(self, forecast_ts: datetime, current_ts: datetime) -> Optional[datetime]:
        """Получить следующий момент созревания."""
        for horizon in self.horizons:
            maturity_ts = self.get_truth_timestamp(forecast_ts, horizon)
            if maturity_ts > current_ts:
                return maturity_ts
        return None

