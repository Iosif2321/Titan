"""Детекция drift."""

from collections import deque
from typing import Optional

import numpy as np

from btc_oracle.core.log import get_logger

logger = get_logger(__name__)


class DriftDetector:
    """Детектор drift для триггера адаптации."""
    
    def __init__(
        self,
        window_size: int = 1000,
        threshold: float = 0.15,
    ):
        """
        Args:
            window_size: размер окна для сравнения
            threshold: порог для триггера drift
        """
        self.window_size = window_size
        self.threshold = threshold
        
        # История метрик
        self.recent_metrics: deque = deque(maxlen=window_size)
        self.baseline_metrics: deque = deque(maxlen=window_size)
    
    def record_metric(self, metric: float, is_baseline: bool = False) -> None:
        """Записать метрику."""
        if is_baseline:
            self.baseline_metrics.append(metric)
        else:
            self.recent_metrics.append(metric)
    
    def detect_drift(self) -> bool:
        """Обнаружить drift."""
        if len(self.recent_metrics) < self.window_size // 2:
            return False
        
        if len(self.baseline_metrics) < self.window_size // 2:
            return False
        
        # Сравниваем распределения метрик
        recent_mean = np.mean(list(self.recent_metrics))
        baseline_mean = np.mean(list(self.baseline_metrics))
        
        # Простая метрика: разница средних
        drift_score = abs(recent_mean - baseline_mean) / (baseline_mean + 1e-10)
        
        if drift_score > self.threshold:
            logger.warning(
                "Drift detected",
                drift_score=drift_score,
                threshold=self.threshold,
            )
            return True
        
        return False
    
    def update_baseline(self) -> None:
        """Обновить baseline на текущие метрики."""
        self.baseline_metrics = deque(list(self.recent_metrics), maxlen=self.window_size)
        logger.info("Baseline updated")

