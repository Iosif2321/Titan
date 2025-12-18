"""Онлайн нормализация признаков."""

from typing import Optional

import numpy as np


class OnlineNormalizer:
    """Онлайн нормализатор с экспоненциальным скользящим средним."""
    
    def __init__(self, alpha: float = 0.01, epsilon: float = 1e-8):
        """
        Args:
            alpha: коэффициент обновления EMA (меньше = медленнее адаптация)
            epsilon: маленькое значение для избежания деления на ноль
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.mean: Optional[np.ndarray] = None
        self.var: Optional[np.ndarray] = None
        self.n = 0
    
    def fit(self, X: np.ndarray) -> "OnlineNormalizer":
        """Инициализация на батче данных."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        self.mean = np.mean(X, axis=0)
        self.var = np.var(X, axis=0)
        self.n = len(X)
        return self
    
    def partial_fit(self, x: np.ndarray) -> None:
        """Обновление статистик одним примером."""
        if self.mean is None:
            self.mean = x.copy()
            self.var = np.zeros_like(x)
            self.n = 1
            return
        
        # Экспоненциальное обновление
        delta = x - self.mean
        self.mean += self.alpha * delta
        self.var = (1 - self.alpha) * (self.var + self.alpha * delta ** 2)
        self.n += 1
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Нормализация признаков."""
        if self.mean is None:
            return x
        
        std = np.sqrt(self.var + self.epsilon)
        return (x - self.mean) / std
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit и transform."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, x_norm: np.ndarray) -> np.ndarray:
        """Обратная нормализация."""
        if self.mean is None:
            return x_norm
        
        std = np.sqrt(self.var + self.epsilon)
        return x_norm * std + self.mean

