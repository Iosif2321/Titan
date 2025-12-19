"""Бины величины движения для наград."""

from enum import Enum
from typing import Optional


class MagnitudeBin(str, Enum):
    """Бины величины движения."""
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    EXTREME = "extreme"


class MagnitudeBinner:
    """Классификатор величины движения по бинам."""
    
    def __init__(self, bins_config: dict):
        """
        Args:
            bins_config: конфигурация бинов из config.rewards.bins
        """
        self.bins = bins_config
    
    def get_bin(self, magnitude: float) -> tuple[MagnitudeBin, float]:
        """
        Получить бин для величины движения.
        
        Args:
            magnitude: величина движения (|Δ|/ATR)
        
        Returns:
            (bin, weight)
        """
        if magnitude < self.bins["tiny"]["max"]:
            return MagnitudeBin.TINY, self.bins["tiny"]["weight"]
        elif magnitude < self.bins["small"]["max"]:
            return MagnitudeBin.SMALL, self.bins["small"]["weight"]
        elif magnitude < self.bins["medium"]["max"]:
            return MagnitudeBin.MEDIUM, self.bins["medium"]["weight"]
        elif magnitude < self.bins["large"]["max"]:
            return MagnitudeBin.LARGE, self.bins["large"]["weight"]
        else:
            return MagnitudeBin.EXTREME, self.bins["extreme"]["weight"]

