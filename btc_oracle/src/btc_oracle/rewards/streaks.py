"""3-уровневые streak trackers."""

from typing import Optional


class StreakTracker:
    """Трекер серий (streaks)."""
    
    def __init__(self):
        self.current_streak = 0
        self.is_positive = True  # True для успехов, False для ошибок
        self.max_streak = 0
    
    def record(self, success: bool) -> None:
        """Записать результат."""
        if success == self.is_positive:
            self.current_streak += 1
            self.max_streak = max(self.max_streak, self.current_streak)
        else:
            # Меняем направление
            self.current_streak = 1
            self.is_positive = success
    
    def get_multiplier(self, base_multiplier: float, max_multiplier: float = 3.0) -> float:
        """Получить множитель на основе streak."""
        if not self.is_positive:  # streak ошибок
            multiplier = 1.0 + (self.current_streak - 1) * (base_multiplier - 1.0)
            return min(multiplier, max_multiplier)
        else:
            return 1.0  # streak успехов не увеличивает штраф


class MultiLevelStreakTracker:
    """3-уровневый трекер streaks."""
    
    def __init__(
        self,
        model_error_multiplier: float = 1.2,
        group_error_multiplier: float = 1.1,
        horizon_error_multiplier: float = 1.15,
        max_streak_multiplier: float = 3.0,
    ):
        """
        Args:
            model_error_multiplier: базовый множитель для ошибок модели
            group_error_multiplier: множитель для ошибок группы
            horizon_error_multiplier: множитель для ошибок горизонта
            max_streak_multiplier: максимум множителя
        """
        self.model_error_multiplier = model_error_multiplier
        self.group_error_multiplier = group_error_multiplier
        self.horizon_error_multiplier = horizon_error_multiplier
        self.max_streak_multiplier = max_streak_multiplier
        
        # Трекеры для каждого уровня
        self.model_streaks: dict[str, StreakTracker] = {}
        self.group_streaks: dict[str, StreakTracker] = {}
        self.horizon_streaks: dict[int, StreakTracker] = {}
    
    def record_model_result(self, model_id: str, success: bool) -> None:
        """Записать результат модели."""
        if model_id not in self.model_streaks:
            self.model_streaks[model_id] = StreakTracker()
        self.model_streaks[model_id].record(success)
    
    def record_group_result(self, group_id: str, success: bool) -> None:
        """Записать результат группы."""
        if group_id not in self.group_streaks:
            self.group_streaks[group_id] = StreakTracker()
        self.group_streaks[group_id].record(success)
    
    def record_horizon_result(self, horizon: int, success: bool) -> None:
        """Записать результат горизонта."""
        if horizon not in self.horizon_streaks:
            self.horizon_streaks[horizon] = StreakTracker()
        self.horizon_streaks[horizon].record(success)
    
    def get_model_multiplier(self, model_id: str) -> float:
        """Получить множитель для модели."""
        if model_id not in self.model_streaks:
            return 1.0
        return self.model_streaks[model_id].get_multiplier(
            self.model_error_multiplier,
            self.max_streak_multiplier,
        )
    
    def get_group_multiplier(self, group_id: str) -> float:
        """Получить множитель для группы."""
        if group_id not in self.group_streaks:
            return 1.0
        return self.group_streaks[group_id].get_multiplier(
            self.group_error_multiplier,
            self.max_streak_multiplier,
        )
    
    def get_horizon_multiplier(self, horizon: int) -> float:
        """Получить множитель для горизонта."""
        if horizon not in self.horizon_streaks:
            return 1.0
        return self.horizon_streaks[horizon].get_multiplier(
            self.horizon_error_multiplier,
            self.max_streak_multiplier,
        )

