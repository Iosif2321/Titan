"""Replay buffer для онлайн обучения."""

import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from btc_oracle.core.types import Features, Label


@dataclass
class ReplaySample:
    """Пример для replay buffer."""
    features: Features
    truth: Label
    magnitude: float
    priority: float = 1.0
    timestamp: Optional[float] = None


class ReplayBuffer:
    """Replay buffer с приоритетной выборкой."""
    
    def __init__(
        self,
        buffer_size: int = 10000,
        recent_ratio: float = 0.5,
        priority_alpha: float = 0.6,
        priority_beta: float = 0.4,
    ):
        """
        Args:
            buffer_size: размер буфера
            recent_ratio: доля свежих примеров
            priority_alpha: экспонента для приоритетов
            priority_beta: экспонента для importance sampling
        """
        self.buffer_size = buffer_size
        self.recent_ratio = recent_ratio
        self.priority_alpha = priority_alpha
        self.priority_beta = priority_beta
        
        # Разделяем на recent и old
        self.recent_buffer: deque = deque(maxlen=int(buffer_size * recent_ratio))
        self.old_buffer: deque = deque(maxlen=int(buffer_size * (1 - recent_ratio)))
        
        self.max_priority = 1.0
    
    def add(
        self,
        features: Features,
        truth: Label,
        magnitude: float,
        priority: Optional[float] = None,
    ) -> None:
        """Добавить пример в буфер."""
        if priority is None:
            priority = 1.0
        
        sample = ReplaySample(
            features=features,
            truth=truth,
            magnitude=magnitude,
            priority=priority,
        )
        
        # Добавляем в recent
        self.recent_buffer.append(sample)
        
        # Периодически перемещаем в old
        if len(self.recent_buffer) >= self.recent_buffer.maxlen:
            # Перемещаем старые из recent в old
            while len(self.recent_buffer) > int(self.recent_buffer.maxlen * 0.8):
                old_sample = self.recent_buffer.popleft()
                self.old_buffer.append(old_sample)
        
        # Обновляем max priority
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int) -> list[ReplaySample]:
        """
        Выбрать батч примеров.
        
        Args:
            batch_size: размер батча
        
        Returns:
            список примеров
        """
        # Смешиваем recent и old
        recent_count = int(batch_size * self.recent_ratio)
        old_count = batch_size - recent_count
        
        samples = []
        
        # Выбираем из recent
        if len(self.recent_buffer) > 0:
            recent_samples = random.sample(
                list(self.recent_buffer),
                min(recent_count, len(self.recent_buffer)),
            )
            samples.extend(recent_samples)
        
        # Выбираем из old
        if len(self.old_buffer) > 0:
            old_samples = random.sample(
                list(self.old_buffer),
                min(old_count, len(self.old_buffer)),
            )
            samples.extend(old_samples)
        
        # Если недостаточно, дополняем из recent
        while len(samples) < batch_size and len(self.recent_buffer) > 0:
            samples.append(random.choice(list(self.recent_buffer)))
        
        return samples
    
    def update_priority(self, sample: ReplaySample, new_priority: float) -> None:
        """Обновить приоритет примера."""
        sample.priority = new_priority
        self.max_priority = max(self.max_priority, new_priority)
    
    def __len__(self) -> int:
        """Размер буфера."""
        return len(self.recent_buffer) + len(self.old_buffer)

