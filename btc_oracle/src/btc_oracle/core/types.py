"""Основные типы данных для системы прогнозирования."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np


class Label(str, Enum):
    """4 состояния прогноза."""
    UP = "UP"
    DOWN = "DOWN"
    FLAT = "FLAT"
    UNCERTAIN = "UNCERTAIN"


class ReasonCode(str, Enum):
    """Коды причин для решений."""
    TINY_MOVE_EXPECTED = "TINY_MOVE_EXPECTED"
    HIGH_EPISTEMIC_UNCERTAINTY = "HIGH_EPISTEMIC_UNCERTAINTY"
    LOW_ENSEMBLE_CONSENSUS = "LOW_ENSEMBLE_CONSENSUS"
    MEMORY_VS_NEURAL_CONFLICT = "MEMORY_VS_NEURAL_CONFLICT"
    INSUFFICIENT_PATTERN_SUPPORT = "INSUFFICIENT_PATTERN_SUPPORT"
    STRONG_DIRECTIONAL_SIGNAL = "STRONG_DIRECTIONAL_SIGNAL"
    WEAK_DIRECTIONAL_SIGNAL = "WEAK_DIRECTIONAL_SIGNAL"
    MEMORY_HIGH_CONFIDENCE = "MEMORY_HIGH_CONFIDENCE"


@dataclass
class Candle:
    """Свеча OHLCV."""
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    confirmed: bool = True  # закрыта ли свеча
    
    @property
    def timestamp(self) -> int:
        """Unix timestamp в миллисекундах."""
        return int(self.ts.timestamp() * 1000)


@dataclass
class Features:
    """Вектор признаков для модели."""
    ts: datetime
    timeframe: str
    vector: np.ndarray  # shape: (feature_dim,)
    meta: dict  # дополнительные метаданные (ATR, volume_regime и т.д.)
    
    @property
    def feature_dim(self) -> int:
        """Размерность вектора признаков."""
        return len(self.vector)


@dataclass
class PatternKey:
    """Ключ паттерна."""
    timeframe: str
    horizon: int  # в минутах
    pattern_id: int  # 128-bit internal id (int)
    market_hash: bytes  # 256-bit SimHash
    context_hash: bytes  # 256-bit SimHash
    regime_key: int  # compact regime id


@dataclass
class NeuralOpinion:
    """Мнение нейросетевого ансамбля."""
    p_up: float  # вероятность UP
    p_down: float  # вероятность DOWN
    p_flat: float  # вероятность FLAT
    u_dir: float  # uncertainty для direction
    u_mag: float  # uncertainty для magnitude/flat
    consensus: float  # консенсус ансамбля (0-1)
    disagreement: float  # разногласие ансамбля (0-1)
    
    @property
    def p_dir(self) -> tuple[float, float]:
        """Вероятности направления (UP, DOWN)."""
        total = self.p_up + self.p_down
        if total > 0:
            return (self.p_up / total, self.p_down / total)
        return (0.5, 0.5)


@dataclass
class MemoryOpinion:
    """Мнение памяти паттернов."""
    p_up_mem: float
    p_down_mem: float
    p_flat_mem: float
    credibility: float  # доверие к памяти (0-1)
    n: int  # количество примеров паттерна
    pattern_id: int
    
    @property
    def p_dir_mem(self) -> tuple[float, float]:
        """Вероятности направления из памяти."""
        total = self.p_up_mem + self.p_down_mem
        if total > 0:
            return (self.p_up_mem / total, self.p_down_mem / total)
        return (0.5, 0.5)


@dataclass
class FusedForecast:
    """Объединённый прогноз (Neural + Memory)."""
    p_up: float
    p_down: float
    p_flat: float
    u_dir: float
    u_mag: float
    flat_score: float  # оценка "малости" движения (не uncertainty!)
    uncertainty_score: float  # общая неопределённость модели
    consensus: float
    memory_support: Optional[dict] = None  # информация о поддержке памяти
    
    @property
    def p_dir(self) -> tuple[float, float]:
        """Нормализованные вероятности направления."""
        total = self.p_up + self.p_down
        if total > 0:
            return (self.p_up / total, self.p_down / total)
        return (0.5, 0.5)


@dataclass
class Decision:
    """Финальное решение системы."""
    label: Label
    reason_code: ReasonCode
    ts: datetime
    symbol: str
    horizon_min: int
    p_up: float
    p_down: float
    p_flat: float
    flat_score: float
    uncertainty_score: float
    consensus: float
    memory: Optional[dict] = None
    latency_ms: float = 0.0
    debug: Optional[dict] = None
    
    def to_dict(self) -> dict:
        """Преобразование в словарь для API."""
        return {
            "ts": self.ts.isoformat() + "Z",
            "symbol": self.symbol,
            "horizon_min": self.horizon_min,
            "label": self.label.value,
            "p_up": self.p_up,
            "p_down": self.p_down,
            "p_flat": self.p_flat,
            "flat_score": self.flat_score,
            "uncertainty_score": self.uncertainty_score,
            "consensus": self.consensus,
            "memory": self.memory,
            "reason_code": self.reason_code.value,
            "latency_ms": self.latency_ms,
        }

