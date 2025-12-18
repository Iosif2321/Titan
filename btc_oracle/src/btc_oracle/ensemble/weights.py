"""Управление весами моделей ансамбля (EMA)."""

from typing import Optional

from btc_oracle.core.log import get_logger

logger = get_logger(__name__)


class WeightManager:
    """Менеджер весов моделей ансамбля."""
    
    def __init__(self, num_models: int, ema_alpha: float = 0.1, uncertainty_penalty_lambda: float = 0.3):
        """
        Args:
            num_models: количество моделей
            ema_alpha: коэффициент EMA для обновления весов
            uncertainty_penalty_lambda: штраф за uncertainty
        """
        self.num_models = num_models
        self.ema_alpha = ema_alpha
        self.uncertainty_penalty_lambda = uncertainty_penalty_lambda
        
        # Инициализируем равные веса
        self.ema_weights = [1.0 / num_models] * num_models
    
    def get_effective_weights(self, uncertainties: list[float]) -> list[float]:
        """
        Получить эффективные веса с учётом uncertainty.
        
        Args:
            uncertainties: список uncertainty для каждой модели
        
        Returns:
            список эффективных весов
        """
        if len(uncertainties) != self.num_models:
            raise ValueError(f"Expected {self.num_models} uncertainties, got {len(uncertainties)}")
        
        effective_weights = []
        for i, (w, u) in enumerate(zip(self.ema_weights, uncertainties)):
            # Штрафуем за uncertainty
            w_eff = w * (1 - self.uncertainty_penalty_lambda * u)
            effective_weights.append(max(0.0, w_eff))  # не отрицательные
        
        # Нормализуем
        total = sum(effective_weights)
        if total > 0:
            return [w / total for w in effective_weights]
        else:
            # Fallback на равные веса
            return [1.0 / self.num_models] * self.num_models
    
    def update_weights(self, rewards: list[float]) -> None:
        """
        Обновить EMA веса на основе наград.
        
        Args:
            rewards: список наград для каждой модели
        """
        if len(rewards) != self.num_models:
            raise ValueError(f"Expected {self.num_models} rewards, got {len(rewards)}")
        
        # Нормализуем награды в [0, 1] для стабильности
        min_reward = min(rewards)
        max_reward = max(rewards)
        if max_reward > min_reward:
            normalized = [(r - min_reward) / (max_reward - min_reward) for r in rewards]
        else:
            normalized = [0.5] * self.num_models
        
        # Обновляем EMA веса
        for i in range(self.num_models):
            # Больше награда → больше вес
            target_weight = normalized[i]
            self.ema_weights[i] = (
                self.ema_alpha * target_weight + (1 - self.ema_alpha) * self.ema_weights[i]
            )
        
        # Нормализуем веса
        total = sum(self.ema_weights)
        if total > 0:
            self.ema_weights = [w / total for w in self.ema_weights]
        
        logger.debug("Updated model weights", weights=self.ema_weights, rewards=rewards)
    
    def get_weights(self) -> list[float]:
        """Получить текущие EMA веса."""
        return self.ema_weights.copy()

