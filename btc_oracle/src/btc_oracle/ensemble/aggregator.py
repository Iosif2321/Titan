"""Агрегатор мнений ансамбля моделей."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from btc_oracle.core.types import Features, NeuralOpinion
from btc_oracle.ensemble.weights import WeightManager
from btc_oracle.ensemble.wrapper import ModelWrapper


class EnsembleAggregator:
    """Агрегатор для объединения мнений ансамбля."""
    
    def __init__(
        self,
        models: list[ModelWrapper],
        weight_manager: WeightManager,
        parallel: bool = True,
        max_workers: Optional[int] = None,
    ):
        """
        Args:
            models: список обёрток моделей
            weight_manager: менеджер весов
            parallel: использовать ли параллельный инференс
            max_workers: максимальное количество потоков (если parallel=True)
        """
        self.models = models
        self.weight_manager = weight_manager
        self.parallel = parallel
        self.max_workers = max_workers or len(models)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers) if parallel else None
    
    async def aggregate(
        self,
        features: Features,
        enable_dropout: bool = True,
    ) -> NeuralOpinion:
        """
        Агрегировать мнения ансамбля.
        
        Args:
            features: вектор признаков
            enable_dropout: включать ли dropout для MC
        
        Returns:
            объединённое мнение ансамбля
        """
        if self.parallel:
            # Параллельный инференс
            opinions = await self._predict_parallel(features, enable_dropout)
        else:
            # Последовательный
            opinions = [m.predict_with_uncertainty(features, enable_dropout) for m in self.models]
        
        # Вычисляем эффективные веса
        uncertainties = [op.u_dir for op in opinions]
        weights = self.weight_manager.get_effective_weights(uncertainties)
        
        # Взвешенное усреднение вероятностей
        p_up = sum(w * op.p_up for w, op in zip(weights, opinions))
        p_down = sum(w * op.p_down for w, op in zip(weights, opinions))
        p_flat = sum(w * op.p_flat for w, op in zip(weights, opinions))
        
        # Uncertainty ансамбля
        # Комбинируем: средний uncertainty + disagreement
        u_dir_mean = sum(w * op.u_dir for w, op in zip(weights, opinions))
        u_mag_mean = sum(w * op.u_mag for w, op in zip(weights, opinions))
        
        # Disagreement (разброс между моделями)
        p_up_values = [op.p_up for op in opinions]
        p_down_values = [op.p_down for op in opinions]
        p_flat_values = [op.p_flat for op in opinions]
        
        disagreement_dir = float(np.std(p_up_values))
        disagreement_mag = float(np.std(p_flat_values))
        
        # Общий uncertainty score
        u_dir = u_dir_mean + disagreement_dir
        u_mag = u_mag_mean + disagreement_mag
        
        # Consensus (насколько модели согласны)
        # Высокий consensus = низкий disagreement
        consensus = max(0.0, 1.0 - disagreement_dir - disagreement_mag)
        
        # Общий disagreement
        disagreement = (disagreement_dir + disagreement_mag) / 2.0
        
        return NeuralOpinion(
            p_up=p_up,
            p_down=p_down,
            p_flat=p_flat,
            u_dir=u_dir,
            u_mag=u_mag,
            consensus=consensus,
            disagreement=disagreement,
        )
    
    async def _predict_parallel(
        self,
        features: Features,
        enable_dropout: bool,
    ) -> list[NeuralOpinion]:
        """Параллельный инференс моделей."""
        loop = asyncio.get_event_loop()
        
        # Запускаем в thread pool (PyTorch не полностью async)
        futures = [
            loop.run_in_executor(
                self.executor,
                lambda m=m: m.predict_with_uncertainty(features, enable_dropout),
            )
            for m in self.models
        ]
        
        opinions = await asyncio.gather(*futures)
        return list(opinions)
    
    def close(self):
        """Закрыть executor."""
        if self.executor:
            self.executor.shutdown(wait=False)

