"""Двухканальный fusion Neural + Memory."""

from typing import Optional

from btc_oracle.core.types import FusedForecast, MemoryOpinion, NeuralOpinion


class FusionEngine:
    """Движок fusion для объединения Neural и Memory мнений."""
    
    def __init__(
        self,
        memory_credibility_min: float = 0.3,
        neural_confidence_weight: float = 0.7,
    ):
        """
        Args:
            memory_credibility_min: минимум credibility для использования памяти
            neural_confidence_weight: базовый вес нейросети
        """
        self.memory_credibility_min = memory_credibility_min
        self.neural_confidence_weight = neural_confidence_weight
    
    def fuse(
        self,
        neural: NeuralOpinion,
        memory: Optional[MemoryOpinion] = None,
    ) -> FusedForecast:
        """
        Объединить мнения Neural и Memory.
        
        Args:
            neural: мнение нейросетевого ансамбля
            memory: мнение памяти паттернов (опционально)
        
        Returns:
            объединённый прогноз
        """
        # Вычисляем веса доказательств
        w_mem = 0.0
        if memory and memory.credibility >= self.memory_credibility_min:
            w_mem = memory.credibility
        
        w_net = (1 - neural.u_dir) * self.neural_confidence_weight
        
        # Нормализуем веса
        total_weight = w_net + w_mem
        if total_weight > 0:
            w_net /= total_weight
            w_mem /= total_weight
        else:
            w_net = 1.0
            w_mem = 0.0
        
        # Fusion для Direction (двухканально)
        if memory and w_mem > 0:
            p_up_final = w_net * neural.p_up + w_mem * memory.p_up_mem
            p_down_final = w_net * neural.p_down + w_mem * memory.p_down_mem
        else:
            p_up_final = neural.p_up
            p_down_final = neural.p_down
        
        # Нормализуем direction probabilities
        dir_total = p_up_final + p_down_final
        if dir_total > 0:
            p_up_final /= dir_total
            p_down_final /= dir_total
        else:
            p_up_final = 0.5
            p_down_final = 0.5
        
        # Fusion для Flat (отдельный канал)
        if memory and w_mem > 0:
            p_flat_final = w_net * neural.p_flat + w_mem * memory.p_flat_mem
        else:
            p_flat_final = neural.p_flat
        
        # Ограничиваем вероятности
        p_up_final = max(0.0, min(1.0, p_up_final))
        p_down_final = max(0.0, min(1.0, p_down_final))
        p_flat_final = max(0.0, min(1.0, p_flat_final))
        
        # Uncertainty остаётся от нейросети (память не добавляет uncertainty)
        u_dir = neural.u_dir
        u_mag = neural.u_mag
        
        # Flat score = p_flat (это оценка "малости" движения, не uncertainty)
        flat_score = p_flat_final
        
        # Uncertainty score комбинирует u_dir и consensus
        uncertainty_score = (u_dir + (1 - neural.consensus)) / 2.0
        
        # Memory support info
        memory_support = None
        if memory:
            memory_support = {
                "pattern_id": memory.pattern_id,
                "n": memory.n,
                "credibility": memory.credibility,
                "weight": w_mem,
            }
        
        return FusedForecast(
            p_up=p_up_final,
            p_down=p_down_final,
            p_flat=p_flat_final,
            u_dir=u_dir,
            u_mag=u_mag,
            flat_score=flat_score,
            uncertainty_score=uncertainty_score,
            consensus=neural.consensus,
            memory_support=memory_support,
        )

