"""Энкодер паттернов (дискретизация и хеширование)."""

import hashlib
from typing import Optional

import numpy as np

from btc_oracle.core.types import Features, PatternKey


class PatternEncoder:
    """Энкодер для преобразования признаков в PatternID."""
    
    def __init__(self, bins: int = 20):
        """
        Args:
            bins: количество бинов для дискретизации
        """
        self.bins = bins
        self._quantiles_cache: dict[int, np.ndarray] = {}  # кэш квантилей для адаптации
    
    def encode(
        self,
        features: Features,
        horizon: int,
        timeframe: str = "1m",
    ) -> PatternKey:
        """
        Закодировать признаки в PatternID.
        
        Args:
            features: вектор признаков
            horizon: горизонт в минутах
            timeframe: таймфрейм
        
        Returns:
            PatternKey
        """
        # Дискретизация признаков
        discretized = self._discretize(features.vector)
        
        # Добавляем контекст (vol_regime, session и т.д.)
        context = self._get_context(features, horizon)
        
        # Создаём токены
        tokens = list(discretized) + context
        
        # Хешируем в uint64
        pattern_id = self._hash_tokens(tokens)
        
        return PatternKey(
            timeframe=timeframe,
            horizon=horizon,
            pattern_id=pattern_id,
        )
    
    def _discretize(self, vector: np.ndarray) -> np.ndarray:
        """Дискретизация признаков в бины."""
        # Заменяем NaN на 0 (или среднее)
        vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Простая равномерная дискретизация
        # В реальности можно использовать адаптивные квантили
        discretized = np.clip(
            (vector + 1) / 2 * self.bins,  # нормализуем в [0, bins]
            0,
            self.bins - 1,
        ).astype(np.int32)
        
        return discretized
    
    def _get_context(self, features: Features, horizon: int) -> list[int]:
        """Получить контекстные токены."""
        context = []
        
        # Vol regime (упрощённо по ATR)
        atr = features.meta.get("atr", 0.0)
        close = features.meta.get("close", 1.0)
        atr_rel = atr / close if close > 0 else 0.0
        
        if atr_rel < 0.01:
            vol_regime = 0  # низкая волатильность
        elif atr_rel < 0.03:
            vol_regime = 1  # средняя
        else:
            vol_regime = 2  # высокая
        
        context.append(vol_regime)
        
        # Session (упрощённо по времени)
        hour = features.ts.hour
        if 0 <= hour < 8:
            session = 0  # азиатская
        elif 8 <= hour < 16:
            session = 1  # европейская
        else:
            session = 2  # американская
        
        context.append(session)
        
        # Horizon bucket
        if horizon <= 5:
            horizon_bucket = 0
        elif horizon <= 15:
            horizon_bucket = 1
        elif horizon <= 60:
            horizon_bucket = 2
        else:
            horizon_bucket = 3
        
        context.append(horizon_bucket)
        
        return context
    
    def _hash_tokens(self, tokens: list) -> int:
        """Хеширование токенов в uint64."""
        # Преобразуем в строку и хешируем
        token_str = ",".join(str(t) for t in tokens)
        hash_obj = hashlib.sha256(token_str.encode())
        # Берём первые 8 байт для uint64
        return int.from_bytes(hash_obj.digest()[:8], byteorder="big")

