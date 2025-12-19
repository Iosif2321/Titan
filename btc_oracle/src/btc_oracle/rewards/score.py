"""Вычисление наград и штрафов."""

from typing import Optional

import numpy as np

from btc_oracle.core.types import Decision, Label
from btc_oracle.rewards.bins import MagnitudeBinner
from btc_oracle.rewards.streaks import MultiLevelStreakTracker


class RewardScorer:
    """Вычисление наград и штрафов для моделей."""
    
    def __init__(
        self,
        bins_config: dict,
        confidence_penalty_alpha: float = 1.5,
        target_coverage: float = 0.7,
        uncertain_penalty_base: float = 0.1,
        uncertain_penalty_large_move: float = 0.5,
        streak_tracker: Optional[MultiLevelStreakTracker] = None,
    ):
        """
        Args:
            bins_config: конфигурация бинов
            confidence_penalty_alpha: усилитель штрафа за уверенность
            target_coverage: целевой coverage (доля не-UNCERTAIN)
            uncertain_penalty_base: базовый штраф за UNCERTAIN
            uncertain_penalty_large_move: штраф за пропуск большого движения
            streak_tracker: трекер streaks
        """
        self.binner = MagnitudeBinner(bins_config)
        self.confidence_penalty_alpha = confidence_penalty_alpha
        self.target_coverage = target_coverage
        self.uncertain_penalty_base = uncertain_penalty_base
        self.uncertain_penalty_large_move = uncertain_penalty_large_move
        self.streak_tracker = streak_tracker or MultiLevelStreakTracker()
    
    def compute_reward(
        self,
        decision: Decision,
        truth: Label,
        magnitude: float,
        model_id: Optional[str] = None,
        horizon: Optional[int] = None,
    ) -> float:
        """
        Вычислить награду/штраф для решения.
        
        Args:
            decision: решение модели
            truth: истинный лейбл
            magnitude: величина движения
            model_id: ID модели (для streak)
            horizon: горизонт (для streak)
        
        Returns:
            награда (положительная) или штраф (отрицательная)
        """
        # Получаем бин и вес
        _, bin_weight = self.binner.get_bin(magnitude)
        
        # Проверяем правильность
        is_correct = (decision.label == truth)
        
        # Если UNCERTAIN, это abstain
        if decision.label == Label.UNCERTAIN:
            return self._compute_uncertain_penalty(truth, magnitude, bin_weight)
        
        # Если правильный прогноз
        if is_correct:
            # Награда пропорциональна уверенности и весу бина
            confidence = max(decision.p_up, decision.p_down)
            reward = bin_weight * (confidence ** self.confidence_penalty_alpha)
            return reward
        
        # Если неправильный прогноз
        # Штраф пропорционален уверенности (уверенная ошибка штрафуется сильнее)
        confidence = max(decision.p_up, decision.p_down)
        penalty = -bin_weight * (confidence ** self.confidence_penalty_alpha)
        
        # Усиливаем штраф за streak ошибок
        if model_id:
            streak_mult = self.streak_tracker.get_model_multiplier(model_id)
            penalty *= streak_mult
        
        if horizon:
            horizon_mult = self.streak_tracker.get_horizon_multiplier(horizon)
            penalty *= horizon_mult
        
        return penalty
    
    def _compute_uncertain_penalty(
        self,
        truth: Label,
        magnitude: float,
        bin_weight: float,
    ) -> float:
        """Вычислить штраф за UNCERTAIN."""
        # Базовый штраф
        penalty = -self.uncertain_penalty_base
        
        # Если пропустили большое движение с очевидным направлением
        _, bin_weight_actual = self.binner.get_bin(magnitude)
        if bin_weight_actual >= 1.5 and truth != Label.FLAT:
            penalty -= self.uncertain_penalty_large_move * bin_weight_actual
        
        return penalty
    
    def compute_logloss(
        self,
        decision: Decision,
        truth: Label,
    ) -> float:
        """
        Вычислить logloss для оценки качества вероятностей.
        
        Args:
            decision: решение
            truth: истинный лейбл
        
        Returns:
            logloss
        """
        if truth == Label.UP:
            p_true = decision.p_up
        elif truth == Label.DOWN:
            p_true = decision.p_down
        elif truth == Label.FLAT:
            p_true = decision.p_flat
        else:
            # UNCERTAIN не является truth, но можем считать как равномерное распределение
            p_true = 1.0 / 3.0
        
        # Если UNCERTAIN, считаем как равномерное распределение
        if decision.label == Label.UNCERTAIN:
            # Штраф за неопределённость
            return -np.log(p_true + 1e-10) + 0.5
        
        return -np.log(p_true + 1e-10)
    
    def compute_brier_score(
        self,
        decision: Decision,
        truth: Label,
    ) -> float:
        """
        Вычислить Brier score.
        
        Args:
            decision: решение
            truth: истинный лейбл
        
        Returns:
            Brier score
        """
        # One-hot encoding truth
        y_up = 1.0 if truth == Label.UP else 0.0
        y_down = 1.0 if truth == Label.DOWN else 0.0
        y_flat = 1.0 if truth == Label.FLAT else 0.0
        
        # Brier score = mean((p - y)^2)
        brier = (
            (decision.p_up - y_up) ** 2 +
            (decision.p_down - y_down) ** 2 +
            (decision.p_flat - y_flat) ** 2
        ) / 3.0
        
        return brier

