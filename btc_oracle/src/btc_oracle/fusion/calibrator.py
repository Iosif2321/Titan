"""Калибровка вероятностей (temperature scaling)."""

from typing import Optional

from btc_oracle.core.types import FusedForecast


class TemperatureCalibrator:
    """Калибратор с temperature scaling."""
    
    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: начальная температура
        """
        self.temperature = temperature
        self.update_count = 0
    
    def calibrate(self, forecast: FusedForecast) -> FusedForecast:
        """
        Применить калибровку к прогнозу.
        
        Args:
            forecast: исходный прогноз
        
        Returns:
            откалиброванный прогноз
        """
        # Temperature scaling для direction probabilities
        # p_calibrated = softmax(logits / temperature)
        # Но у нас уже вероятности, поэтому используем обратное преобразование
        
        # Для простоты MVP: если temperature != 1.0, применяем степенное преобразование
        if self.temperature != 1.0:
            # Упрощённая версия: p^t / (p^t + (1-p)^t)
            # Для direction
            p_up_cal = (forecast.p_up ** (1.0 / self.temperature)) / (
                (forecast.p_up ** (1.0 / self.temperature)) +
                (forecast.p_down ** (1.0 / self.temperature)) +
                1e-10
            )
            p_down_cal = (forecast.p_down ** (1.0 / self.temperature)) / (
                (forecast.p_up ** (1.0 / self.temperature)) +
                (forecast.p_down ** (1.0 / self.temperature)) +
                1e-10
            )
            
            # Для flat
            p_flat_cal = (forecast.p_flat ** (1.0 / self.temperature)) / (
                (forecast.p_flat ** (1.0 / self.temperature)) +
                ((1 - forecast.p_flat) ** (1.0 / self.temperature)) +
                1e-10
            )
        else:
            p_up_cal = forecast.p_up
            p_down_cal = forecast.p_down
            p_flat_cal = forecast.p_flat
        
        # Обновляем счётчик
        self.update_count += 1
        
        return FusedForecast(
            p_up=p_up_cal,
            p_down=p_down_cal,
            p_flat=p_flat_cal,
            u_dir=forecast.u_dir,
            u_mag=forecast.u_mag,
            flat_score=p_flat_cal,  # flat_score тоже калибруется
            uncertainty_score=forecast.uncertainty_score,
            consensus=forecast.consensus,
            memory_support=forecast.memory_support,
        )
    
    def update_temperature(self, new_temperature: float) -> None:
        """Обновить температуру."""
        self.temperature = max(0.1, min(10.0, new_temperature))  # ограничиваем

