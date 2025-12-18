"""Онлайн обучение ансамбля."""

import time
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from btc_oracle.core.types import Features, Label
from btc_oracle.ensemble.inter_model import InterModelLearner
from btc_oracle.models.base import BaseModel
from btc_oracle.training.continual import ContinualLearner
from btc_oracle.training.replay import ReplayBuffer, ReplaySample


class OnlineTrainer:
    """Тренер для онлайн обучения моделей."""
    
    def __init__(
        self,
        model: BaseModel,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        gradient_clip_norm: float = 1.0,
        replay_buffer: Optional[ReplayBuffer] = None,
        inter_model_learner: Optional[InterModelLearner] = None,
        continual_learner: Optional[ContinualLearner] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: модель для обучения
            learning_rate: learning rate
            batch_size: размер батча
            gradient_clip_norm: норма для gradient clipping
            replay_buffer: replay buffer
            inter_model_learner: learner для distillation
            continual_learner: learner для anti-forgetting
            device: устройство
        """
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_clip_norm = gradient_clip_norm
        self.replay_buffer = replay_buffer
        self.inter_model_learner = inter_model_learner
        self.continual_learner = continual_learner
        self.device = device or torch.device("cpu")
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.to(self.device)
    
    def train_step(
        self,
        leader_models: Optional[list[BaseModel]] = None,
    ) -> dict[str, float]:
        """
        Выполнить один шаг обучения.
        
        Args:
            leader_models: модели-лидеры для distillation
        
        Returns:
            словарь с метриками обучения
        """
        if self.replay_buffer is None or len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0, "direction_loss": 0.0, "flat_loss": 0.0}
        
        # Выбираем батч
        samples = self.replay_buffer.sample(self.batch_size)
        
        # Подготавливаем данные
        features_batch = torch.stack([
            torch.tensor(s.features.vector, dtype=torch.float32)
            for s in samples
        ]).to(self.device)
        
        # Truth labels
        truth_labels_dir = torch.tensor([
            0 if s.truth == Label.UP else 1 if s.truth == Label.DOWN else 0
            for s in samples
        ], dtype=torch.long).to(self.device)
        
        truth_labels_flat = torch.tensor([
            1 if s.truth == Label.FLAT else 0
            for s in samples
        ], dtype=torch.float32).to(self.device)
        
        # Forward pass
        self.model.train()
        output = self.model(features_batch)
        
        # Direction loss (cross-entropy)
        direction_logits = output["direction_logits"]
        direction_loss = nn.functional.cross_entropy(direction_logits, truth_labels_dir)
        
        # Flat loss (binary cross-entropy)
        flat_logits = output["flat_logits"]
        if flat_logits.shape[-1] == 1:
            flat_loss = nn.functional.binary_cross_entropy_with_logits(
                flat_logits.squeeze(-1),
                truth_labels_flat,
            )
        else:
            flat_loss = nn.functional.cross_entropy(
                flat_logits,
                truth_labels_flat.long(),
            )
        
        # Общий loss
        total_loss = direction_loss + flat_loss
        
        # Distillation loss
        if self.inter_model_learner and leader_models:
            distill_loss = self.inter_model_learner.compute_distillation_loss(
                self.model,
                leader_models,
                features_batch,
            )
            total_loss += distill_loss
        else:
            distill_loss = torch.tensor(0.0)
        
        # Continual learning loss (KL/EWC)
        if self.continual_learner:
            kl_loss = self.continual_learner.compute_kl_loss(self.model, features_batch)
            ewc_loss = self.continual_learner.compute_ewc_loss(self.model)
            total_loss += kl_loss + ewc_loss
        else:
            kl_loss = torch.tensor(0.0)
            ewc_loss = torch.tensor(0.0)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.gradient_clip_norm,
        )
        
        self.optimizer.step()
        
        return {
            "loss": float(total_loss),
            "direction_loss": float(direction_loss),
            "flat_loss": float(flat_loss),
            "distill_loss": float(distill_loss),
            "kl_loss": float(kl_loss),
            "ewc_loss": float(ewc_loss),
        }

