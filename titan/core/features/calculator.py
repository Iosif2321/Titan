"""Unified feature calculator for train and inference.

Sprint 23: P0 fix - ensures train/inference features are identical.

This module is the SINGLE SOURCE OF TRUTH for all feature calculations.
Both TFTDataset (training) and FeatureStream (inference) must use this module.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# =============================================================================
# FEATURE NAMES (canonical order - DO NOT CHANGE without migration)
# =============================================================================

# Features that need z-score normalization (prices)
PRICE_FEATURES = ["close", "ma_fast", "ma_slow", "ma_delta"]

# Features that are 0-100 and need /100 normalization
OSCILLATOR_FEATURES = ["rsi", "stochastic_k", "stochastic_d", "mfi", "adx"]

# Features that are already scale-invariant (no normalization needed)
SCALE_INVARIANT_FEATURES = [
    "return_1", "ma_delta_pct", "ema_10_spread_pct", "ema_20_spread_pct",
    "volatility_z", "volume_z", "volume_trend", "volume_change_pct",
    "rsi_momentum", "rsi_oversold", "rsi_overbought",
    "price_momentum_3", "return_5", "return_10",
    "body_ratio", "candle_direction", "upper_wick_ratio", "lower_wick_ratio",
    "body_pct", "high_low_range_pct", "atr_pct", "bb_position",
    "vol_ratio", "vol_imbalance_20",
]

# All features in canonical order (sorted for consistency)
ALL_FEATURES = sorted(set(
    PRICE_FEATURES + OSCILLATOR_FEATURES + SCALE_INVARIANT_FEATURES
))


# =============================================================================
# BATCH FEATURE CALCULATION (for TFTDataset)
# =============================================================================

def compute_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features from OHLCV DataFrame.

    Args:
        df: DataFrame with columns [timestamp, open, high, low, close, volume]

    Returns:
        DataFrame with all computed features (NOT normalized yet)
    """
    df = df.copy()

    # Basic features
    df["return_1"] = df["close"].pct_change().fillna(0)

    # Moving averages
    df["ma_fast"] = df["close"].rolling(5).mean()
    df["ma_slow"] = df["close"].rolling(20).mean()
    df["ma_delta"] = df["ma_fast"] - df["ma_slow"]
    df["ma_delta_pct"] = df["ma_delta"] / (df["close"] + 1e-10)

    # EMAs
    df["ema_10"] = df["close"].ewm(span=10).mean()
    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_10_spread_pct"] = (df["close"] - df["ema_10"]) / (df["close"] + 1e-10)
    df["ema_20_spread_pct"] = (df["close"] - df["ema_20"]) / (df["close"] + 1e-10)

    # Volatility
    df["volatility"] = df["return_1"].rolling(20).std()
    vol_mean = df["volatility"].rolling(100).mean()
    vol_std = df["volatility"].rolling(100).std()
    df["volatility_z"] = (df["volatility"] - vol_mean) / (vol_std + 1e-10)

    # Volume
    df["volume_ma"] = df["volume"].rolling(20).mean()
    df["volume_std"] = df["volume"].rolling(20).std()
    df["volume_z"] = (df["volume"] - df["volume_ma"]) / (df["volume_std"] + 1e-10)
    df["volume_trend"] = (df["volume"] > df["volume_ma"]).astype(float)
    df["volume_change_pct"] = df["volume"].pct_change().fillna(0).clip(-2, 2)

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi_momentum"] = df["rsi"].diff()
    df["rsi_oversold"] = (df["rsi"] < 30).astype(float)
    df["rsi_overbought"] = (df["rsi"] > 70).astype(float)

    # Price momentum
    df["price_momentum_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)
    df["return_10"] = df["close"].pct_change(10)

    # Candle features
    df["body"] = abs(df["close"] - df["open"])
    df["range"] = df["high"] - df["low"]
    df["body_ratio"] = df["body"] / (df["range"] + 1e-10)
    df["candle_direction"] = ((df["close"] >= df["open"]).astype(float) * 2) - 1  # -1 or +1
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["upper_wick_ratio"] = df["upper_wick"] / (df["range"] + 1e-10)
    df["lower_wick_ratio"] = df["lower_wick"] / (df["range"] + 1e-10)
    df["body_pct"] = df["body"] / (df["close"] + 1e-10)
    df["high_low_range_pct"] = df["range"] / (df["close"] + 1e-10)

    # ATR
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1))
        )
    )
    df["atr"] = df["tr"].rolling(14).mean()
    df["atr_pct"] = df["atr"] / (df["close"] + 1e-10)

    # ADX (simplified)
    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    df["adx"] = (plus_dm.rolling(14).mean() + minus_dm.rolling(14).mean()) / 2

    # Bollinger Bands position
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_position"] = (df["close"] - bb_mid) / (2 * bb_std + 1e-10)

    # Stochastic
    low_14 = df["low"].rolling(14).min()
    high_14 = df["high"].rolling(14).max()
    df["stochastic_k"] = (df["close"] - low_14) / (high_14 - low_14 + 1e-10) * 100
    df["stochastic_d"] = df["stochastic_k"].rolling(3).mean()

    # MFI (Money Flow Index)
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    money_flow = typical_price * df["volume"]
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    df["mfi"] = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-10)))

    # Volume ratio and imbalance
    df["vol_ratio"] = df["volume"] / (df["volume_ma"] + 1e-10)
    df["vol_imbalance_20"] = df["volume"].rolling(20).apply(
        lambda x: (x[x > x.mean()].sum() - x[x <= x.mean()].sum()) / (x.sum() + 1e-10),
        raw=False
    )

    return df


def normalize_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize features for neural network training.

    Normalization rules:
    - Price features (close, ma_fast, ma_slow, ma_delta): rolling z-score
    - Oscillators (rsi, stochastic, mfi, adx): divide by 100
    - Scale-invariant features: no change (already normalized)

    Args:
        df: DataFrame with computed features

    Returns:
        DataFrame with normalized features
    """
    df = df.copy()

    # Price features -> rolling z-scores
    # Sprint 23 FIX: Use window=50 (not 20) to avoid ma_slow - close_mean = 0
    # ma_slow uses rolling(20), so using rolling(20) here makes them identical
    close_mean = df["close"].rolling(50).mean()
    close_std = df["close"].rolling(50).std() + 1e-10

    df["close"] = (df["close"] - close_mean) / close_std
    df["ma_fast"] = (df["ma_fast"] - close_mean) / close_std
    df["ma_slow"] = (df["ma_slow"] - close_mean) / close_std
    df["ma_delta"] = df["ma_delta"] / close_std

    # Oscillators 0-100 -> 0-1
    df["rsi"] = df["rsi"] / 100.0
    df["stochastic_k"] = df["stochastic_k"] / 100.0
    df["stochastic_d"] = df["stochastic_d"] / 100.0
    df["mfi"] = df["mfi"] / 100.0
    df["adx"] = df["adx"] / 100.0

    # RSI momentum -> z-score
    rsi_mom_std = df["rsi_momentum"].rolling(50).std() + 1e-10
    df["rsi_momentum"] = df["rsi_momentum"] / rsi_mom_std

    # Fill NaN
    df = df.fillna(0)

    return df


def compute_and_normalize_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features and normalize in one call.

    Args:
        df: DataFrame with columns [timestamp, open, high, low, close, volume]

    Returns:
        DataFrame with all features computed and normalized
    """
    df = compute_features_batch(df)
    df = normalize_features_batch(df)
    return df


# =============================================================================
# STREAMING FEATURE CALCULATION (for FeatureStream)
# =============================================================================

class StreamingNormalizer:
    """Rolling statistics for streaming normalization.

    Maintains rolling windows for z-score calculation in streaming mode.
    """

    def __init__(self, window: int = 50):
        # Sprint 23 FIX: Use window=50 to match batch normalization
        # (must differ from ma_slow's 20-period to avoid zero normalization)
        self.window = window
        self._close_history: List[float] = []
        self._rsi_mom_history: List[float] = []

    def update_close(self, close: float) -> None:
        """Update close price history."""
        self._close_history.append(close)
        if len(self._close_history) > self.window:
            self._close_history.pop(0)

    def update_rsi_momentum(self, rsi_mom: float) -> None:
        """Update RSI momentum history."""
        self._rsi_mom_history.append(rsi_mom)
        if len(self._rsi_mom_history) > 50:  # RSI momentum uses 50-period
            self._rsi_mom_history.pop(0)

    def get_close_stats(self) -> Tuple[float, float]:
        """Get rolling mean and std of close prices (ddof=1 to match pandas)."""
        if len(self._close_history) < 2:
            return 0.0, 1.0
        mean = np.mean(self._close_history)
        std = np.std(self._close_history, ddof=1) + 1e-10  # sample std like pandas
        return mean, std

    def get_rsi_mom_std(self) -> float:
        """Get rolling std of RSI momentum (ddof=1 to match pandas)."""
        if len(self._rsi_mom_history) < 2:
            return 1.0
        return np.std(self._rsi_mom_history, ddof=1) + 1e-10  # sample std like pandas

    def normalize_price(self, value: float, close_mean: float, close_std: float) -> float:
        """Normalize a price-related feature."""
        return (value - close_mean) / close_std

    def normalize_oscillator(self, value: float) -> float:
        """Normalize 0-100 oscillator to 0-1."""
        return value / 100.0

    def normalize_rsi_momentum(self, value: float) -> float:
        """Normalize RSI momentum using rolling std."""
        return value / self.get_rsi_mom_std()


def normalize_features_streaming(
    features: Dict[str, float],
    normalizer: StreamingNormalizer,
) -> Dict[str, float]:
    """Normalize a single feature dict for streaming inference.

    Args:
        features: Dict of raw feature values
        normalizer: StreamingNormalizer with updated history

    Returns:
        Dict of normalized feature values
    """
    normalized = features.copy()

    # Get close stats
    close_mean, close_std = normalizer.get_close_stats()

    # Price features -> z-scores
    if "close" in normalized:
        normalized["close"] = normalizer.normalize_price(features["close"], close_mean, close_std)
    if "ma_fast" in normalized:
        normalized["ma_fast"] = normalizer.normalize_price(features["ma_fast"], close_mean, close_std)
    if "ma_slow" in normalized:
        normalized["ma_slow"] = normalizer.normalize_price(features["ma_slow"], close_mean, close_std)
    if "ma_delta" in normalized:
        normalized["ma_delta"] = features["ma_delta"] / close_std

    # Oscillators -> /100
    for osc in ["rsi", "stochastic_k", "stochastic_d", "mfi", "adx"]:
        if osc in normalized:
            normalized[osc] = normalizer.normalize_oscillator(features[osc])

    # RSI momentum -> z-score
    if "rsi_momentum" in normalized:
        normalized["rsi_momentum"] = normalizer.normalize_rsi_momentum(features["rsi_momentum"])

    # volume_change_pct -> clip
    if "volume_change_pct" in normalized:
        normalized["volume_change_pct"] = np.clip(features["volume_change_pct"], -2, 2)

    return normalized


# =============================================================================
# VALIDATION
# =============================================================================

def validate_features(features: Dict[str, float]) -> List[str]:
    """Check for missing or invalid features.

    Args:
        features: Dict of feature values

    Returns:
        List of warning messages (empty if all OK)
    """
    warnings = []

    for name in ALL_FEATURES:
        if name not in features:
            warnings.append(f"Missing feature: {name}")
        elif not np.isfinite(features[name]):
            warnings.append(f"Invalid value for {name}: {features[name]}")

    return warnings


def compare_feature_dicts(
    train_features: Dict[str, float],
    inference_features: Dict[str, float],
    rtol: float = 0.1,
) -> List[str]:
    """Compare train and inference features for consistency.

    Args:
        train_features: Features from TFTDataset
        inference_features: Features from FeatureStream
        rtol: Relative tolerance for comparison

    Returns:
        List of mismatch messages (empty if all OK)
    """
    mismatches = []

    all_keys = set(train_features.keys()) | set(inference_features.keys())

    for key in sorted(all_keys):
        if key not in train_features:
            mismatches.append(f"{key}: missing in train")
        elif key not in inference_features:
            mismatches.append(f"{key}: missing in inference")
        else:
            train_val = train_features[key]
            inf_val = inference_features[key]
            if abs(train_val - inf_val) > rtol * max(abs(train_val), abs(inf_val), 1e-10):
                mismatches.append(f"{key}: train={train_val:.4f}, inference={inf_val:.4f}")

    return mismatches
