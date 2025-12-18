"""Вычисление truth labels на основе величины движения."""

from typing import Optional

from btc_oracle.core.types import Candle, Label


def compute_truth_label(
    forecast_candle: Candle,
    truth_candle: Candle,
    atr: float,
    m_flat_threshold: float = 0.05,
) -> tuple[Label, float]:
    """
    Вычислить truth label на основе величины движения.
    
    Args:
        forecast_candle: свеча на момент прогноза
        truth_candle: свеча через horizon минут
        atr: Average True Range на момент прогноза
        m_flat_threshold: порог для FLAT (|Δ|/ATR)
    
    Returns:
        (truth_label, magnitude)
    """
    # Доходность
    r = (truth_candle.close - forecast_candle.close) / forecast_candle.close
    
    # Величина движения (нормализованная на ATR)
    if atr > 0:
        m = abs(r) / (atr / forecast_candle.close)  # ATR уже в абсолютных единицах
    else:
        m = abs(r) / 0.01  # fallback если ATR = 0
    
    # Определяем truth
    if m < m_flat_threshold:
        return Label.FLAT, m
    elif r > 0:
        return Label.UP, m
    else:
        return Label.DOWN, m


def get_truth_candle_magnitude(
    forecast_candle: Candle,
    truth_candle: Optional[Candle],
    atr: float,
) -> Optional[float]:
    """
    Получить величину движения для truth свечи.
    
    Args:
        forecast_candle: свеча на момент прогноза
        truth_candle: свеча через horizon минут (может быть None)
        atr: ATR на момент прогноза
    
    Returns:
        magnitude или None если truth_candle отсутствует
    """
    if truth_candle is None:
        return None
    
    r = (truth_candle.close - forecast_candle.close) / forecast_candle.close
    
    if atr > 0:
        m = abs(r) / (atr / forecast_candle.close)
    else:
        m = abs(r) / 0.01
    
    return m

