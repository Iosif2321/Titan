"""Пайплайн построения признаков из свечей."""

from typing import Optional

import numpy as np

from btc_oracle.core.types import Candle, Features
from btc_oracle.features.indicators import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_price_features,
    calculate_rsi,
    calculate_volume_profile,
)
from btc_oracle.features.normalizer import OnlineNormalizer


class FeaturePipeline:
    VOLUME_PROFILE_BINS = 5

    """Пайплайн для построения признаков."""
    
    def __init__(self, window_size: int = 100, normalize: bool = True):
        """
        Args:
            window_size: размер окна для индикаторов
            normalize: применять ли нормализацию
        """
        self.window_size = window_size
        self.normalize = normalize
        self.normalizer: Optional[OnlineNormalizer] = OnlineNormalizer() if normalize else None
    
    def build_features(self, candles: list[Candle]) -> Features:
        """
        Построить вектор признаков из окна свечей.
        
        Args:
            candles: список свечей (последняя - текущая)
        
        Returns:
            объект Features
        """
        if len(candles) < self.window_size:
            # Дополняем первыми значениями если недостаточно данных
            first_candle = candles[0] if candles else None
            if first_candle:
                padding = [first_candle] * (self.window_size - len(candles))
                candles = padding + candles
        
        # Извлекаем массивы
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])
        opens = np.array([c.open for c in candles])
        
        # Вычисляем индикаторы
        features_list = []
        
        # ATR
        atr = calculate_atr(highs, lows, closes)
        if not np.isnan(atr[-1]):
            features_list.append(atr[-1])
            features_list.append(atr[-1] / closes[-1])  # нормализованный ATR
        else:
            features_list.extend([0.0, 0.0])
        
        # RSI
        rsi = calculate_rsi(closes)
        if not np.isnan(rsi[-1]):
            features_list.append(rsi[-1] / 100.0)  # нормализуем в [0, 1]
        else:
            features_list.append(0.5)
        
        # MACD
        macd, signal, histogram = calculate_macd(closes)
        if not np.isnan(macd[-1]):
            features_list.append(macd[-1] / closes[-1])  # нормализованный
            features_list.append(signal[-1] / closes[-1])
            features_list.append(histogram[-1] / closes[-1])
        else:
            features_list.extend([0.0, 0.0, 0.0])
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(closes)
        if not np.isnan(bb_upper[-1]):
            bb_width = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
            bb_position = (closes[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1] + 1e-10)
            features_list.append(bb_width)
            features_list.append(np.clip(bb_position, 0, 1))
        else:
            features_list.extend([0.0, 0.5])
        
        # Volume Profile (упрощённо - последние N свечей)
        vol_profile = calculate_volume_profile(volumes[-20:], highs[-20:], lows[-20:], bins=self.VOLUME_PROFILE_BINS)
        features_list.extend(vol_profile.tolist())
        
        # Price features (статистики доходности)
        price_features = calculate_price_features(closes[-20:])
        features_list.extend([
            price_features["returns_mean"],
            price_features["returns_std"],
            price_features["returns_skew"],
            price_features["returns_kurt"],
        ])
        
        # Дополнительные признаки
        # Returns на разных горизонтах
        if len(closes) >= 5:
            returns_1 = (closes[-1] - closes[-2]) / closes[-2]
            returns_5 = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 else returns_1
            features_list.extend([returns_1, returns_5])
        else:
            features_list.extend([0.0, 0.0])
        
        # Volume ratio
        if len(volumes) >= 20:
            vol_ratio = volumes[-1] / (np.mean(volumes[-20:-1]) + 1e-10)
            features_list.append(min(vol_ratio, 5.0))  # ограничиваем
        else:
            features_list.append(1.0)
        
        # High-Low spread
        hl_spread = (highs[-1] - lows[-1]) / closes[-1]
        features_list.append(hl_spread)
        
        # Собираем вектор
        feature_vector = np.array(features_list, dtype=np.float32)
        # Заменяем NaN на 0
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Метаданные
        meta = {
            "atr": float(atr[-1]) if not np.isnan(atr[-1]) else 0.0,
            "volume": float(volumes[-1]),
            "close": float(closes[-1]),
        }
        
        # Нормализация
        if self.normalize and self.normalizer:
            if self.normalizer.mean is None:
                # Инициализируем на текущем векторе
                self.normalizer.fit(feature_vector.reshape(1, -1))
            feature_vector = self.normalizer.transform(feature_vector)
        
        return Features(
            ts=candles[-1].ts,
            timeframe="1m",  # можно сделать параметром
            vector=feature_vector,
            meta=meta,
        )
    
    @classmethod
    def feature_names(cls) -> list[str]:
        """Poryadok priznakov, podavaemykh v model."""
        names = [
            'atr',
            'atr_norm',
            'rsi',
            'macd',
            'macd_signal',
            'macd_hist',
            'bb_width',
            'bb_position',
        ]
        names.extend([f'vol_profile_{i}' for i in range(cls.VOLUME_PROFILE_BINS)])
        names.extend([
            'returns_mean',
            'returns_std',
            'returns_skew',
            'returns_kurt',
            'return_1',
            'return_5',
            'vol_ratio',
            'hl_spread',
        ])
        return names

    def update_normalizer(self, feature_vector: np.ndarray) -> None:
        """Обновить нормализатор одним примером."""
        if self.normalizer:
            self.normalizer.partial_fit(feature_vector)

