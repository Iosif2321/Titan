"""Статистика паттернов (Beta распределения)."""

from datetime import datetime, timedelta
from typing import Optional

from btc_oracle.core.types import Label


class PatternStats:
    """Статистика одного паттерна."""
    
    def __init__(
        self,
        pattern_id: int,
        timeframe: str,
        horizon: int,
        alpha_up: float = 1.0,
        beta_down: float = 1.0,
        alpha_flat: float = 1.0,
        beta_not_flat: float = 1.0,
        n: int = 0,
        last_seen: Optional[datetime] = None,
        decay_factor: float = 1.0,
        cooldown_until: Optional[datetime] = None,
    ):
        """
        Args:
            pattern_id: идентификатор паттерна
            timeframe: таймфрейм
            horizon: горизонт в минутах
            alpha_up, beta_down: параметры Beta для направления
            alpha_flat, beta_not_flat: параметры Beta для FLAT
            n: количество наблюдений
            last_seen: последнее время наблюдения
            decay_factor: фактор затухания (для decay)
            cooldown_until: время окончания cooldown
        """
        self.pattern_id = pattern_id
        self.timeframe = timeframe
        self.horizon = horizon
        self.alpha_up = alpha_up
        self.beta_down = beta_down
        self.alpha_flat = alpha_flat
        self.beta_not_flat = beta_not_flat
        self.n = n
        self.last_seen = last_seen or datetime.now()
        self.decay_factor = decay_factor
        self.cooldown_until = cooldown_until
    
    @property
    def p_up_mem(self) -> float:
        """Вероятность UP из памяти."""
        total = self.alpha_up + self.beta_down
        if total == 0:
            return 0.5
        return self.alpha_up / total
    
    @property
    def p_down_mem(self) -> float:
        """Вероятность DOWN из памяти."""
        total = self.alpha_up + self.beta_down
        if total == 0:
            return 0.5
        return self.beta_down / total
    
    @property
    def p_flat_mem(self) -> float:
        """Вероятность FLAT из памяти."""
        total = self.alpha_flat + self.beta_not_flat
        if total == 0:
            return 0.5
        return self.alpha_flat / total
    
    @property
    def credibility(self) -> float:
        """Доверие к памяти (на основе n и decay)."""
        # Базовое доверие растёт с n, но уменьшается с decay
        base_cred = min(self.n / 10.0, 1.0)  # максимум при n >= 10
        return base_cred * self.decay_factor
    
    @property
    def is_in_cooldown(self) -> bool:
        """Проверка, находится ли паттерн в cooldown."""
        if self.cooldown_until is None:
            return False
        return datetime.now() < self.cooldown_until
    
    def update(
        self,
        truth: Label,
        magnitude: float,
        decay_half_life_hours: float = 168,
    ) -> None:
        """
        Обновить статистику паттерна.
        
        Args:
            truth: истинный лейбл (UP, DOWN, FLAT)
            magnitude: величина движения (|Δ|/ATR)
            decay_half_life_hours: период полураспада для decay
        """
        # Применяем decay перед обновлением
        self._apply_decay(decay_half_life_hours)
        
        # Обновляем направление
        if truth == Label.UP:
            self.alpha_up += 1
        elif truth == Label.DOWN:
            self.beta_down += 1
        # FLAT не обновляет direction stats
        
        # Обновляем FLAT
        if truth == Label.FLAT:
            self.alpha_flat += 1
        else:
            self.beta_not_flat += 1
        
        self.n += 1
        self.last_seen = datetime.now()
    
    def _apply_decay(self, half_life_hours: float) -> None:
        """Применить затухание к статистике."""
        if self.last_seen is None:
            return
        
        hours_passed = (datetime.now() - self.last_seen).total_seconds() / 3600
        if hours_passed <= 0:
            return
        
        # Экспоненциальное затухание
        decay_rate = 0.5 ** (hours_passed / half_life_hours)
        self.decay_factor *= decay_rate
        
        # Уменьшаем параметры Beta пропорционально
        # Но не ниже prior (1.0)
        reduction = 1.0 - decay_rate
        self.alpha_up = max(1.0, self.alpha_up * (1 - reduction * 0.1))
        self.beta_down = max(1.0, self.beta_down * (1 - reduction * 0.1))
        self.alpha_flat = max(1.0, self.alpha_flat * (1 - reduction * 0.1))
        self.beta_not_flat = max(1.0, self.beta_not_flat * (1 - reduction * 0.1))
    
    def record_error(self, cooldown_duration_hours: float = 24) -> None:
        """Записать ошибку и активировать cooldown при необходимости."""
        # Упрощённая логика: если серия ошибок, активируем cooldown
        # В реальности нужно отслеживать streak ошибок отдельно
        if self.cooldown_until is None or datetime.now() > self.cooldown_until:
            # Активируем cooldown
            self.cooldown_until = datetime.now() + timedelta(hours=cooldown_duration_hours)
    
    def to_dict(self) -> dict:
        """Преобразование в словарь для сохранения."""
        return {
            "pattern_id": self.pattern_id,
            "timeframe": self.timeframe,
            "horizon": self.horizon,
            "alpha_up": self.alpha_up,
            "beta_down": self.beta_down,
            "alpha_flat": self.alpha_flat,
            "beta_not_flat": self.beta_not_flat,
            "n": self.n,
            "last_seen": int(self.last_seen.timestamp() * 1000) if self.last_seen else None,
            "decay_factor": self.decay_factor,
            "cooldown_until": int(self.cooldown_until.timestamp() * 1000) if self.cooldown_until else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "PatternStats":
        """Создание из словаря."""
        return cls(
            pattern_id=data["pattern_id"],
            timeframe=data["timeframe"],
            horizon=data["horizon"],
            alpha_up=data["alpha_up"],
            beta_down=data["beta_down"],
            alpha_flat=data["alpha_flat"],
            beta_not_flat=data["beta_not_flat"],
            n=data["n"],
            last_seen=datetime.fromtimestamp(data["last_seen"] / 1000) if data.get("last_seen") else None,
            decay_factor=data.get("decay_factor", 1.0),
            cooldown_until=datetime.fromtimestamp(data["cooldown_until"] / 1000) if data.get("cooldown_until") else None,
        )

