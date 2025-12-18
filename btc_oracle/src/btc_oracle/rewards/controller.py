"""Контроллер для авто-регулировки параметров."""

from typing import Optional

from btc_oracle.core.config import Config
from btc_oracle.core.log import get_logger

logger = get_logger(__name__)


class RewardController:
    """Контроллер для управления параметрами на основе наград."""
    
    def __init__(self, config: Config):
        """
        Args:
            config: конфигурация системы
        """
        self.config = config
        self.latency_history: list[float] = []
        self.coverage_history: list[float] = []
    
    def update_mc_passes(
        self,
        current_latency_ms: float,
        current_mc_passes: int,
    ) -> int:
        """
        Адаптивно обновить количество MC passes на основе латентности.
        
        Args:
            current_latency_ms: текущая латентность
            current_mc_passes: текущее количество MC passes
        
        Returns:
            новое количество MC passes
        """
        if not self.config.performance.adaptive_mc_passes:
            return current_mc_passes
        
        max_latency = self.config.performance.max_inference_time_ms
        
        # Если превышаем лимит, уменьшаем passes
        if current_latency_ms > max_latency * 0.8:  # 80% лимита
            new_passes = max(
                self.config.performance.mc_passes_min,
                int(current_mc_passes * 0.8),
            )
            logger.info(
                "Reducing MC passes due to latency",
                old_passes=current_mc_passes,
                new_passes=new_passes,
                latency_ms=current_latency_ms,
            )
            return new_passes
        
        # Если латентность низкая, можно увеличить
        if current_latency_ms < max_latency * 0.3:
            new_passes = min(
                self.config.performance.mc_passes_max,
                int(current_mc_passes * 1.1),
            )
            return new_passes
        
        return current_mc_passes
    
    def update_coverage_thresholds(
        self,
        current_coverage: float,
    ) -> dict[str, float]:
        """
        Обновить пороги для достижения target coverage.
        
        Args:
            current_coverage: текущий coverage (доля не-UNCERTAIN)
        
        Returns:
            словарь с обновлёнными порогами
        """
        target = self.config.rewards.target_coverage
        
        # Если coverage слишком низкий, снижаем пороги uncertainty
        if current_coverage < target * 0.9:
            # Снижаем порог uncertainty чтобы больше прогнозов проходило
            new_u_thr = self.config.uncertainty.u_thr * 1.1
            new_consensus_thr = self.config.uncertainty.consensus_thr * 0.95
            logger.info(
                "Lowering uncertainty thresholds to increase coverage",
                old_u_thr=self.config.uncertainty.u_thr,
                new_u_thr=new_u_thr,
                current_coverage=current_coverage,
            )
            return {
                "u_thr": min(new_u_thr, 0.8),  # ограничиваем
                "consensus_thr": max(new_consensus_thr, 0.3),
            }
        
        # Если coverage слишком высокий, повышаем пороги
        if current_coverage > target * 1.1:
            new_u_thr = self.config.uncertainty.u_thr * 0.95
            new_consensus_thr = self.config.uncertainty.consensus_thr * 1.05
            return {
                "u_thr": max(new_u_thr, 0.1),
                "consensus_thr": min(new_consensus_thr, 0.9),
            }
        
        return {}

