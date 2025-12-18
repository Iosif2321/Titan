"""Continual learning (anti-forgetting)."""

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from btc_oracle.models.base import BaseModel


class ContinualLearner:
    """Защита от катастрофического забывания."""
    
    def __init__(
        self,
        kl_weight: float = 0.1,
        ewc_enabled: bool = False,
        ewc_lambda: float = 0.4,
    ):
        """
        Args:
            kl_weight: вес KL divergence к предыдущей модели
            ewc_enabled: использовать ли EWC (Elastic Weight Consolidation)
            ewc_lambda: вес EWC регуляризации
        """
        self.kl_weight = kl_weight
        self.ewc_enabled = ewc_enabled
        self.ewc_lambda = ewc_lambda
        self.previous_model: Optional[BaseModel] = None
        self.ewc_fisher: Optional[dict] = None
    
    def set_previous_model(self, model: BaseModel) -> None:
        """Установить предыдущую модель для KL/EWC."""
        self.previous_model = copy.deepcopy(model)
        self.previous_model.eval()
    
    def compute_kl_loss(
        self,
        current_model: BaseModel,
        features_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Вычислить KL divergence к предыдущей модели.
        
        Args:
            current_model: текущая модель
            features_batch: батч признаков
        
        Returns:
            KL loss
        """
        if self.previous_model is None:
            return torch.tensor(0.0, device=features_batch.device)
        
        # Предсказания текущей модели
        current_output = current_model(features_batch)
        current_dir_logits = current_output["direction_logits"]
        current_dir_probs = F.softmax(current_dir_logits, dim=-1)
        
        # Предсказания предыдущей модели
        with torch.no_grad():
            prev_output = self.previous_model(features_batch)
            prev_dir_logits = prev_output["direction_logits"]
            prev_dir_probs = F.softmax(prev_dir_logits, dim=-1)
        
        # KL divergence: KL(prev || current)
        kl_loss = F.kl_div(
            F.log_softmax(current_dir_logits, dim=-1),
            prev_dir_probs,
            reduction="batchmean",
        )
        
        return kl_loss * self.kl_weight
    
    def compute_ewc_loss(self, current_model: BaseModel) -> torch.Tensor:
        """
        Вычислить EWC регуляризацию.
        
        Args:
            current_model: текущая модель
        
        Returns:
            EWC loss
        """
        if not self.ewc_enabled or self.ewc_fisher is None:
            return torch.tensor(0.0)
        
        ewc_loss = torch.tensor(0.0, device=next(current_model.parameters()).device)
        
        for name, param in current_model.named_parameters():
            if name in self.ewc_fisher:
                fisher = self.ewc_fisher[name]["fisher"]
                prev_param = self.ewc_fisher[name]["param"]
                
                # EWC: sum(F * (theta - theta*)^2)
                ewc_loss += (fisher * (param - prev_param) ** 2).sum()
        
        return ewc_loss * self.ewc_lambda
    
    def update_fisher(self, model: BaseModel, dataloader: torch.utils.data.DataLoader) -> None:
        """
        Обновить Fisher information matrix для EWC.
        
        Args:
            model: модель
            dataloader: загрузчик данных
        """
        if not self.ewc_enabled:
            return
        
        model.eval()
        fisher = {}
        
        for name, param in model.named_parameters():
            fisher[name] = torch.zeros_like(param)
        
        # Вычисляем Fisher information
        for batch in dataloader:
            model.zero_grad()
            output = model(batch)
            # Упрощённо: используем direction logits
            logits = output["direction_logits"]
            probs = F.softmax(logits, dim=-1)
            
            # Fisher = E[grad^2]
            for i in range(logits.size(0)):
                prob = probs[i]
                # Берём градиент по каждому классу
                for c in range(logits.size(1)):
                    model.zero_grad()
                    logits[i, c].backward(retain_graph=True)
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            fisher[name] += prob[c] * (param.grad ** 2)
        
        # Сохраняем Fisher и параметры
        self.ewc_fisher = {}
        for name, param in model.named_parameters():
            self.ewc_fisher[name] = {
                "fisher": fisher[name] / len(dataloader),
                "param": param.data.clone(),
            }

