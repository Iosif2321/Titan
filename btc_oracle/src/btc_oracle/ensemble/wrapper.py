"""Обёртка для модели с MC Dropout inference."""

import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from btc_oracle.core.types import Features, NeuralOpinion
from btc_oracle.models.base import BaseModel


class ModelWrapper:
    """Обёртка модели с MC Dropout для оценки uncertainty."""
    
    def __init__(
        self,
        model: BaseModel,
        model_id: str,
        mc_passes: int = 10,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: модель для обёртки
            model_id: идентификатор модели
            mc_passes: количество MC Dropout прогонов
            device: устройство (CPU/GPU)
        """
        self.model = model
        self.model_id = model_id
        self.mc_passes = mc_passes
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def predict_with_uncertainty(
        self,
        features: Features,
        enable_dropout: bool = True,
    ) -> NeuralOpinion:
        """
        Предсказать с оценкой uncertainty через MC Dropout.
        
        Args:
            features: вектор признаков
            enable_dropout: включать ли dropout для MC
        
        Returns:
            NeuralOpinion с вероятностями и uncertainty
        """
        start_time = time.time()
        
        # Подготавливаем вход
        x = torch.tensor(features.vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # MC Dropout прогоны
        p_up_samples = []
        p_down_samples = []
        p_flat_samples = []
        
        with torch.no_grad():
            for _ in range(self.mc_passes):
                if enable_dropout:
                    # Включаем dropout для оценки uncertainty
                    self.model.train()
                else:
                    self.model.eval()
                
                output = self.model(x)
                
                # Direction
                dir_logits = output["direction_logits"]
                dir_probs = torch.softmax(dir_logits, dim=-1)
                p_up_samples.append(float(dir_probs[0, 0]))
                p_down_samples.append(float(dir_probs[0, 1]))
                
                # Flat
                flat_logits = output["flat_logits"]
                if flat_logits.shape[-1] == 1:
                    p_flat_samples.append(float(torch.sigmoid(flat_logits[0, 0])))
                else:
                    flat_probs = torch.softmax(flat_logits, dim=-1)
                    p_flat_samples.append(float(flat_probs[0, 0]))
        
        self.model.eval()
        
        # Статистики
        p_up_mean = float(np.mean(p_up_samples))
        p_down_mean = float(np.mean(p_down_samples))
        p_flat_mean = float(np.mean(p_flat_samples))
        
        # Uncertainty как стандартное отклонение
        u_dir = float(np.std(p_up_samples))
        u_mag = float(np.std(p_flat_samples))
        
        # Защита от NaN
        if np.isnan(u_dir): u_dir = 1.0
        if np.isnan(u_mag): u_mag = 1.0
        if np.isnan(p_up_mean): p_up_mean = 0.5
        if np.isnan(p_down_mean): p_down_mean = 0.5
        if np.isnan(p_flat_mean): p_flat_mean = 0.5
        
        # Для ансамбля consensus/disagreement будут вычисляться в aggregator
        # Здесь возвращаем placeholder
        consensus = 1.0 - u_dir  # упрощённо
        disagreement = u_dir
        
        latency_ms = (time.time() - start_time) * 1000
        
        return NeuralOpinion(
            p_up=p_up_mean,
            p_down=p_down_mean,
            p_flat=p_flat_mean,
            u_dir=u_dir,
            u_mag=u_mag,
            consensus=consensus,
            disagreement=disagreement,
        )

