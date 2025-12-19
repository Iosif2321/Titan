"""Технические индикаторы для признаков."""

from typing import Optional

import numpy as np
import pandas as pd


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range."""
    if len(high) < period + 1:
        return np.full(len(high), np.nan)
    
    tr1 = np.abs(high[1:] - close[:-1])
    tr2 = np.abs(low[1:] - close[:-1])
    tr3 = np.abs(high[1:] - low[1:])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.full(len(high), np.nan)
    atr[period] = np.mean(tr[:period])
    
    for i in range(period + 1, len(high)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i - 1]) / period
    
    return atr


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index."""
    if len(close) < period + 1:
        return np.full(len(close), np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close), np.nan)
    avg_loss = np.full(len(close), np.nan)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(
    close: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD (Moving Average Convergence Divergence)."""
    if len(close) < slow + signal:
        nan_arr = np.full(len(close), np.nan)
        return nan_arr, nan_arr, nan_arr
    
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    if len(values) < period:
        return np.full(len(values), np.nan)
    
    ema = np.full(len(values), np.nan)
    multiplier = 2.0 / (period + 1)
    
    ema[period - 1] = np.mean(values[:period])
    
    for i in range(period, len(values)):
        ema[i] = (values[i] - ema[i - 1]) * multiplier + ema[i - 1]
    
    return ema


def calculate_bollinger_bands(
    close: np.ndarray,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands."""
    if len(close) < period:
        nan_arr = np.full(len(close), np.nan)
        return nan_arr, nan_arr, nan_arr
    
    sma = _sma(close, period)
    std = _rolling_std(close, period)
    
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    
    return upper, sma, lower


def _sma(values: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    if len(values) < period:
        return np.full(len(values), np.nan)
    
    sma = np.full(len(values), np.nan)
    for i in range(period - 1, len(values)):
        sma[i] = np.mean(values[i - period + 1:i + 1])
    
    return sma


def _rolling_std(values: np.ndarray, period: int) -> np.ndarray:
    """Rolling Standard Deviation."""
    if len(values) < period:
        return np.full(len(values), np.nan)
    
    std = np.full(len(values), np.nan)
    for i in range(period - 1, len(values)):
        std[i] = np.std(values[i - period + 1:i + 1])
    
    return std


def calculate_volume_profile(
    volume: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    bins: int = 10,
) -> np.ndarray:
    """Профиль объёма (нормализованный)."""
    if len(volume) == 0:
        return np.array([])
    
    price_range = high.max() - low.min()
    if price_range == 0:
        return np.full(bins, 1.0 / bins)
    
    # Упрощённая версия: распределение объёма по бинам цены
    volumes = np.zeros(bins)
    for i in range(len(volume)):
        bin_idx = int(((high[i] + low[i]) / 2 - low.min()) / price_range * bins)
        bin_idx = max(0, min(bins - 1, bin_idx))
        volumes[bin_idx] += volume[i]
    
    total = volumes.sum()
    if total > 0:
        return volumes / total
    return np.full(bins, 1.0 / bins)


def calculate_price_features(close: np.ndarray) -> dict[str, float]:
    """Базовые статистики цены."""
    if len(close) < 2:
        return {
            "returns_mean": 0.0,
            "returns_std": 0.0,
            "returns_skew": 0.0,
            "returns_kurt": 0.0,
        }
    
    returns = np.diff(close) / close[:-1]
    
    return {
        "returns_mean": float(np.mean(returns)),
        "returns_std": float(np.std(returns)),
        "returns_skew": float(_skewness(returns)),
        "returns_kurt": float(_kurtosis(returns)),
    }


def _skewness(values: np.ndarray) -> float:
    """Skewness."""
    if len(values) < 3:
        return 0.0
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return 0.0
    return float(np.mean(((values - mean) / std) ** 3))


def _kurtosis(values: np.ndarray) -> float:
    """Kurtosis."""
    if len(values) < 4:
        return 0.0
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return 0.0
    return float(np.mean(((values - mean) / std) ** 4)) - 3.0

