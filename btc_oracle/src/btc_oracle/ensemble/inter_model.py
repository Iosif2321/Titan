"""Обмен решениями между моделями (distillation)."""

from collections import deque
from typing import Optional

import torch
import torch.nn.functional as F

from btc_oracle.core.types import Decision, Features
from btc_oracle.models.base import BaseModel


class InterModelLearner:
    """Обучение моделей друг у друга через distillation."""
    
    def __init__(
        self,
        top_k_leaders: int = 2,
        min_leader_confidence: float = 0.7,
        distillation_weight: float = 0.3,
    ):
        """
        Args:
            top_k_leaders: количество лидеров для distillation
            min_leader_confidence: минимум уверенности лидера
            distillation_weight: вес KL divergence в loss
        """
        self.top_k_leaders = top_k_leaders
        self.min_leader_confidence = min_leader_confidence
        self.distillation_weight = distillation_weight
        
        # История скоров для определения лидеров
        self.model_scores: dict[str, deque] = {}
        self.window_size = 100
    
    def record_score(self, model_id: str, score: float) -> None:
        """Записать скор модели."""
        if model_id not in self.model_scores:
            self.model_scores[model_id] = deque(maxlen=self.window_size)
        self.model_scores[model_id].append(score)
    
    def get_leaders(self) -> list[str]:
        """Получить список лидеров (top-K по среднему скору)."""
        if not self.model_scores:
            return []
        
        # Вычисляем средние скоры
        avg_scores = {
            model_id: sum(scores) / len(scores)
            for model_id, scores in self.model_scores.items()
            if len(scores) > 0
        }
        
        # Сортируем по убыванию
        sorted_models = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Берём top-K
        leaders = [model_id for model_id, score in sorted_models[:self.top_k_leaders]]
        return leaders
    
    def compute_distillation_loss(
        self,
        student_model: BaseModel,
        leader_models: list[BaseModel],
        features_batch: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Вычислить loss distillation от лидеров к студенту.
        
        Args:
            student_model: модель-студент
            leader_models: список моделей-лидеров
            features_batch: батч признаков [batch, feature_dim]
            temperature: температура для softmax
        
        Returns:
            KL divergence loss
        """
        if not leader_models:
            return torch.tensor(0.0, device=features_batch.device)
        
        # Предсказания студента
        student_output = student_model(features_batch)
        student_dir_logits = student_output["direction_logits"] / temperature
        student_dir_probs = F.softmax(student_dir_logits, dim=-1)
        
        # Предсказания лидеров (усреднённые)
        leader_dir_probs_list = []
        for leader_model in leader_models:
            leader_output = leader_model(features_batch)
            leader_dir_logits = leader_output["direction_logits"] / temperature
            leader_dir_probs = F.softmax(leader_dir_logits, dim=-1)
            leader_dir_probs_list.append(leader_dir_probs)
        
        # Усредняем предсказания лидеров
        leader_dir_probs = torch.stack(leader_dir_probs_list).mean(dim=0)
        
        # KL divergence: KL(leader || student)
        kl_loss = F.kl_div(
            F.log_softmax(student_dir_logits, dim=-1),
            leader_dir_probs,
            reduction="batchmean",
        )
        
        return kl_loss * self.distillation_weight
    
    def is_conflict_case(
        self,
        decision: Decision,
        memory_opinion: Optional[dict],
    ) -> bool:
        """
        Проверить, является ли случай конфликтным (высокий приоритет для replay).
        
        Args:
            decision: решение модели
            memory_opinion: мнение памяти (опционально)
        
        Returns:
            True если конфликтный случай
        """
        # Низкий consensus
        if decision.consensus < 0.6:
            return True
        
        # Конфликт нейро vs память
        if memory_opinion and memory_opinion.get("weight", 0) > 0.3:
            mem_p_up = memory_opinion.get("p_up_mem", 0.5)
            neural_p_up = decision.p_up
            if abs(mem_p_up - neural_p_up) > 0.3:
                return True
        
        return False

