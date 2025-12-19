"""Базовый интерфейс модели."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from btc_oracle.core.types import Features


class BaseModel(nn.Module, ABC):
    """Базовый класс для моделей прогнозирования."""
    
    def __init__(self, input_dim: int):
        """
        Args:
            input_dim: размерность входного вектора признаков
        """
        super().__init__()
        self.input_dim = input_dim
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            словарь с ключами:
            - "direction_logits": logits для направления (shape: [batch, 2])
            - "flat_logits": logits для FLAT (shape: [batch, 1] или [batch, 2])
        """
        pass
    
    def predict_proba(self, features: Features) -> dict[str, float]:
        """
        Предсказать вероятности для одного примера.
        
        Args:
            features: вектор признаков
        
        Returns:
            словарь с вероятностями
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(features.vector, dtype=torch.float32).unsqueeze(0)
            output = self.forward(x)
            
            # Direction probabilities
            dir_logits = output["direction_logits"]
            dir_probs = torch.softmax(dir_logits, dim=-1)
            p_up = float(dir_probs[0, 0])
            p_down = float(dir_probs[0, 1])
            
            # Flat probability
            flat_logits = output["flat_logits"]
            if flat_logits.shape[-1] == 1:
                p_flat = float(torch.sigmoid(flat_logits[0, 0]))
            else:
                flat_probs = torch.softmax(flat_logits, dim=-1)
                p_flat = float(flat_probs[0, 0])
            
            return {
                "p_up": p_up,
                "p_down": p_down,
                "p_flat": p_flat,
            }

