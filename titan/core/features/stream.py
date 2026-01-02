import math
from collections import deque
from typing import Dict, Optional

from titan.core.config import ConfigStore
from titan.core.data.schema import Candle
from titan.core.features.calculator import (
    StreamingNormalizer,
    normalize_features_streaming,
)


class RollingStats:
    def __init__(self, maxlen: int) -> None:
        self._values = deque(maxlen=maxlen)
        self._sum = 0.0
        self._sumsq = 0.0

    @property
    def maxlen(self) -> int:
        return self._values.maxlen  # type: ignore[return-value]

    def append(self, value: float) -> None:
        if len(self._values) == self.maxlen:
            old = self._values.popleft()
            self._sum -= old
            self._sumsq -= old * old
        self._values.append(value)
        self._sum += value
        self._sumsq += value * value

    def mean(self) -> Optional[float]:
        if not self._values:
            return None
        return self._sum / len(self._values)

    def std(self) -> Optional[float]:
        """Sample standard deviation (ddof=1) to match pandas rolling.std()."""
        n = len(self._values)
        if n < 2:
            return None
        mean = self._sum / n
        # Sample variance (ddof=1): divide by (n-1) instead of n
        # Formula: sum((x - mean)^2) / (n - 1) = (sumsq - n*mean^2) / (n - 1)
        sample_var = (self._sumsq - n * mean * mean) / (n - 1)
        return math.sqrt(max(sample_var, 0.0))

    def ready(self) -> bool:
        return len(self._values) >= self.maxlen


class RollingSum:
    def __init__(self, maxlen: int) -> None:
        self._values = deque(maxlen=maxlen)
        self._sum = 0.0

    @property
    def maxlen(self) -> int:
        return self._values.maxlen  # type: ignore[return-value]

    def append(self, value: float) -> None:
        if len(self._values) == self.maxlen:
            old = self._values.popleft()
            self._sum -= old
        self._values.append(value)
        self._sum += value

    def mean(self) -> Optional[float]:
        if not self._values:
            return None
        return self._sum / len(self._values)

    def ready(self) -> bool:
        return len(self._values) >= self.maxlen


class FeatureStream:
    def __init__(self, config: ConfigStore) -> None:
        self._config = config
        self._fast_window = int(config.get("feature.fast_window", 5))
        self._slow_window = int(config.get("feature.slow_window", 20))
        self._rsi_window = int(config.get("feature.rsi_window", 14))
        self._vol_window = int(config.get("feature.vol_window", 20))
        self._volume_window = int(config.get("feature.volume_window", 20))

        self._price_fast = RollingStats(self._fast_window)
        self._price_slow = RollingStats(self._slow_window)
        self._returns = RollingStats(self._vol_window)
        self._volume = RollingStats(self._volume_window)
        self._volatility = RollingStats(100)  # 100-period for volatility_z (aligned with batch)
        self._gains = RollingSum(self._rsi_window)
        self._losses = RollingSum(self._rsi_window)

        self._prev_close: Optional[float] = None
        self._prev_rsi: Optional[float] = None

        # Sprint 10: New feature tracking
        self._price_history: deque = deque(maxlen=6)  # For return_5 (need 6 elements: current + 5 previous)
        self._volume_history: deque = deque(maxlen=10)  # For volume trend

        # Sprint 14: Extended features for ML
        self._return_history: deque = deque(maxlen=10)  # For lagged returns
        self._high_history: deque = deque(maxlen=20)  # For ATR
        self._low_history: deque = deque(maxlen=20)  # For ATR
        self._atr_history: deque = deque(maxlen=20)  # ATR rolling
        self._ema_10: Optional[float] = None  # EMA(10)
        self._ema_20: Optional[float] = None  # EMA(20)

        # Sprint 16: New predictive features
        self._up_volume: deque = deque(maxlen=20)  # Volume on UP candles
        self._down_volume: deque = deque(maxlen=20)  # Volume on DOWN candles
        self._typical_price_history: deque = deque(maxlen=20)  # For MFI
        self._mf_positive: deque = deque(maxlen=14)  # Positive money flow
        self._mf_negative: deque = deque(maxlen=14)  # Negative money flow
        self._tr_history: deque = deque(maxlen=14)  # True range for ADX
        self._plus_dm_history: deque = deque(maxlen=14)  # +DM for ADX
        self._minus_dm_history: deque = deque(maxlen=14)  # -DM for ADX
        self._dx_history: deque = deque(maxlen=14)  # DX for ADX smoothing
        self._prev_high: Optional[float] = None
        self._prev_low: Optional[float] = None
        self._prev_typical_price: Optional[float] = None

        # Sprint 23: Stochastic indicator
        self._stoch_high_14: deque = deque(maxlen=14)  # 14-period high
        self._stoch_low_14: deque = deque(maxlen=14)  # 14-period low
        self._stoch_k_history: deque = deque(maxlen=3)  # For %D smoothing

        # Sprint 23: Streaming normalizer for train/inference consistency
        # FIX: Use window=50 to match batch normalization (must differ from ma_slow's 20)
        self._normalizer = StreamingNormalizer(window=50)
        self._normalize_output = bool(config.get("feature.normalize_output", True))

    def update(self, candle: Candle) -> Optional[Dict[str, float]]:
        if self._prev_close is None:
            self._prev_close = candle.close
            self._price_fast.append(candle.close)
            self._price_slow.append(candle.close)
            self._volume.append(candle.volume)
            self._price_history.append(candle.close)
            self._volume_history.append(candle.volume)
            # Sprint 14: Initialize extended tracking
            self._high_history.append(candle.high)
            self._low_history.append(candle.low)
            self._ema_10 = candle.close
            self._ema_20 = candle.close
            # Sprint 16: Initialize new feature tracking
            self._prev_high = candle.high
            self._prev_low = candle.low
            self._prev_typical_price = (candle.high + candle.low + candle.close) / 3
            return None

        ret = (candle.close / self._prev_close) - 1.0
        log_ret = math.log(candle.close / self._prev_close)
        # BUG FIX: Save prev_close for TR calculation BEFORE updating
        prev_close_for_tr = self._prev_close
        self._prev_close = candle.close

        self._price_fast.append(candle.close)
        self._price_slow.append(candle.close)
        self._returns.append(ret)
        self._volume.append(candle.volume)

        # Sprint 10: Track history for new features
        self._price_history.append(candle.close)
        self._volume_history.append(candle.volume)

        # Sprint 14: Track extended history
        self._return_history.append(ret)
        self._high_history.append(candle.high)
        self._low_history.append(candle.low)

        # Sprint 14: Update EMAs
        alpha_10 = 2.0 / (10 + 1)
        alpha_20 = 2.0 / (20 + 1)
        if self._ema_10 is not None:
            self._ema_10 = alpha_10 * candle.close + (1 - alpha_10) * self._ema_10
        else:
            self._ema_10 = candle.close
        if self._ema_20 is not None:
            self._ema_20 = alpha_20 * candle.close + (1 - alpha_20) * self._ema_20
        else:
            self._ema_20 = candle.close

        # Sprint 16: Update tracking for new features
        # Volume imbalance - track volume on up/down candles
        if candle.close >= candle.open:
            self._up_volume.append(candle.volume)
            self._down_volume.append(0.0)
        else:
            self._up_volume.append(0.0)
            self._down_volume.append(candle.volume)

        # Money Flow tracking for MFI
        typical_price = (candle.high + candle.low + candle.close) / 3
        raw_money_flow = typical_price * candle.volume
        if self._prev_typical_price is not None:
            if typical_price > self._prev_typical_price:
                self._mf_positive.append(raw_money_flow)
                self._mf_negative.append(0.0)
            else:
                self._mf_positive.append(0.0)
                self._mf_negative.append(raw_money_flow)
        self._prev_typical_price = typical_price

        # ADX components - True Range, +DM, -DM
        # BUG FIX: Use prev_close_for_tr (the ACTUAL previous close, not current)
        if self._prev_high is not None and self._prev_low is not None:
            tr = max(
                candle.high - candle.low,
                abs(candle.high - prev_close_for_tr),
                abs(candle.low - prev_close_for_tr)
            )
            self._tr_history.append(tr)

            # +DM and -DM
            up_move = candle.high - self._prev_high
            down_move = self._prev_low - candle.low
            if up_move > down_move and up_move > 0:
                self._plus_dm_history.append(up_move)
                self._minus_dm_history.append(0.0)
            elif down_move > up_move and down_move > 0:
                self._plus_dm_history.append(0.0)
                self._minus_dm_history.append(down_move)
            else:
                self._plus_dm_history.append(0.0)
                self._minus_dm_history.append(0.0)

        self._prev_high = candle.high
        self._prev_low = candle.low

        # RSI: use price delta (diff(close)) to match batch calculation
        # batch uses: delta = df["close"].diff()
        price_delta = candle.close - prev_close_for_tr
        gain = max(price_delta, 0.0)
        loss = max(-price_delta, 0.0)
        self._gains.append(gain)
        self._losses.append(loss)

        ma_fast = self._price_fast.mean() if self._price_fast.ready() else None
        ma_slow = self._price_slow.mean() if self._price_slow.ready() else None
        ma_delta = None
        if ma_fast is not None and ma_slow is not None:
            ma_delta = ma_fast - ma_slow

        volatility = self._returns.std() if self._returns.ready() else None
        if volatility is not None:
            self._volatility.append(volatility)
        vol_z = None
        vol_mean = self._volatility.mean()
        vol_std = self._volatility.std()
        if volatility is not None and vol_mean is not None and vol_std:
            vol_z = (volatility - vol_mean) / vol_std

        volume_mean = self._volume.mean()
        volume_std = self._volume.std()
        volume_z = None
        if volume_mean is not None and volume_std:
            volume_z = (candle.volume - volume_mean) / volume_std

        rsi = None
        if self._gains.ready() and self._losses.ready():
            avg_gain = self._gains.mean() or 0.0
            avg_loss = self._losses.mean() or 0.0
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))

        # Sprint 23 BUG FIX: Update normalizer BEFORE readiness check
        # Otherwise first ~20 candles are not added to close_history,
        # causing batch/stream parity issues (normalizer 20 candles behind)
        self._normalizer.update_close(candle.close)

        if not (
            self._price_slow.ready()
            and self._returns.ready()
            and self._volume.ready()
            and self._gains.ready()
            and self._losses.ready()
        ):
            return None

        # Compute RSI momentum (change in RSI)
        rsi_val = rsi or 50.0
        rsi_prev = self._prev_rsi if self._prev_rsi is not None else rsi_val
        rsi_momentum = rsi_val - rsi_prev
        self._prev_rsi = rsi_val

        # Sprint 10: Compute new features

        # 1. Price momentum (rate of change over 3 periods)
        price_momentum_3 = 0.0
        if len(self._price_history) >= 4:
            old_price = self._price_history[-4]  # 3 periods ago
            if old_price > 0:
                price_momentum_3 = (candle.close - old_price) / old_price

        # 2. Volume trend (is current volume above its moving average? 0 or 1)
        # Aligned with batch: volume_trend = (volume > volume_ma).astype(float)
        volume_trend = 1.0 if candle.volume > (volume_mean or 0) else 0.0

        # 3. Candle body ratio (body size / total range)
        total_range = candle.high - candle.low
        body = abs(candle.close - candle.open)
        body_ratio = body / (total_range + 1e-10)

        # 4. Wick ratios (upper and lower wicks)
        if candle.close >= candle.open:  # Bullish candle
            upper_wick = candle.high - candle.close
            lower_wick = candle.open - candle.low
        else:  # Bearish candle
            upper_wick = candle.high - candle.open
            lower_wick = candle.close - candle.low

        upper_wick_ratio = upper_wick / (total_range + 1e-10)
        lower_wick_ratio = lower_wick / (total_range + 1e-10)

        # 5. Candle direction (1 = bullish, -1 = bearish)
        candle_direction = 1.0 if candle.close >= candle.open else -1.0

        # Sprint 14: Scale-invariant features for ML
        # 6. Lagged returns (return_lag_1 through return_lag_5)
        return_lags = [0.0] * 5
        for i in range(min(5, len(self._return_history))):
            if i < len(self._return_history):
                return_lags[i] = self._return_history[-(i + 1)]

        # 7. ATR as percentage of price
        # Aligned with batch: uses proper True Range from _tr_history
        # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
        atr_pct = 0.0
        if len(self._tr_history) >= 14:
            atr = sum(self._tr_history) / 14
            atr_pct = atr / candle.close if candle.close > 0 else 0.0

        # 8. High-Low range as percentage of close
        high_low_range_pct = total_range / candle.close if candle.close > 0 else 0.0

        # 9. MA delta as percentage of price (scale-invariant)
        ma_delta_pct = (ma_delta or 0.0) / candle.close if candle.close > 0 else 0.0

        # 10. EMA spreads as percentage
        ema_10_spread_pct = 0.0
        ema_20_spread_pct = 0.0
        if self._ema_10 is not None and candle.close > 0:
            ema_10_spread_pct = (candle.close - self._ema_10) / candle.close
        if self._ema_20 is not None and candle.close > 0:
            ema_20_spread_pct = (candle.close - self._ema_20) / candle.close

        # 11. Multi-period returns (5 and 10 periods)
        # Aligned with batch: pct_change(N) = (close - close[N]) / close[N]
        return_5 = 0.0
        return_10 = 0.0
        # _price_history has maxlen=6, so when full, index 0 is 5 bars ago
        if len(self._price_history) >= 6:
            old_price_5 = self._price_history[0]  # 5 periods ago (oldest in 6-element buffer)
            if old_price_5 > 0:
                return_5 = (candle.close - old_price_5) / old_price_5
        # For return_10, we need price from 10 bars ago
        # Compound the returns: (1+r1)*(1+r2)*...*(1+r10) - 1
        if len(self._return_history) >= 10:
            compound = 1.0
            for r in self._return_history:
                compound *= (1.0 + r)
            return_10 = compound - 1.0

        # 12. RSI zones (oversold/overbought indicators)
        rsi_oversold = 1.0 if rsi_val < 30 else 0.0
        rsi_overbought = 1.0 if rsi_val > 70 else 0.0
        rsi_neutral = 1.0 if 40 <= rsi_val <= 60 else 0.0

        # 13. Volume change percentage
        # Aligned with batch: clip to [-2, 2]
        volume_change_pct = 0.0
        if len(self._volume_history) >= 2:
            prev_vol = self._volume_history[-2]
            if prev_vol > 0:
                volume_change_pct = (candle.volume - prev_vol) / prev_vol
                volume_change_pct = max(-2.0, min(2.0, volume_change_pct))  # clip

        # 14. Body size as percentage of close
        body_pct = body / candle.close if candle.close > 0 else 0.0

        # 15. Volume ratio (current volume vs moving average)
        # Aligned with batch: vol_ratio = volume / volume_ma
        vol_ratio = 0.0
        if volume_mean and volume_mean > 0:
            vol_ratio = candle.volume / volume_mean

        # Sprint 16: New predictive features
        # 16. Volume Imbalance (volumes above mean vs below mean, over 20 periods)
        # Aligned with batch: (vol[vol > mean].sum() - vol[vol <= mean].sum()) / total
        vol_imbalance_20 = 0.0
        if self._volume.ready():
            # Access internal values from RollingStats
            vol_list = list(self._volume._values)
            vol_mean_local = sum(vol_list) / len(vol_list) if vol_list else 0
            above_mean = sum(v for v in vol_list if v > vol_mean_local)
            below_mean = sum(v for v in vol_list if v <= vol_mean_local)
            total_vol = above_mean + below_mean
            if total_vol > 0:
                vol_imbalance_20 = (above_mean - below_mean) / total_vol

        # 17. Bollinger Band Position (where price is within the bands)
        # Aligned with batch: bb_position = (close - bb_mid) / (2 * bb_std)
        # This gives values typically in [-1, 1] range (can exceed during extremes)
        bb_position = 0.0  # Default: at middle of bands
        bb_std = self._price_slow.std()
        if bb_std is not None and bb_std > 0 and ma_slow is not None:
            bb_position = (candle.close - ma_slow) / (2 * bb_std + 1e-10)

        # 18. ADX (simplified: average of +DM and -DM)
        # Aligned with batch: adx = (plus_dm.rolling(14).mean() + minus_dm.rolling(14).mean()) / 2
        adx = 0.0
        if len(self._plus_dm_history) >= 14:
            plus_dm_avg = sum(self._plus_dm_history) / 14
            minus_dm_avg = sum(self._minus_dm_history) / 14
            adx = (plus_dm_avg + minus_dm_avg) / 2

        # 19. MFI (Money Flow Index) - volume-weighted RSI, 0-100
        mfi = 50.0  # Default: neutral
        if len(self._mf_positive) >= 14:
            pos_flow = sum(self._mf_positive)
            neg_flow = sum(self._mf_negative)
            if neg_flow > 0:
                money_ratio = pos_flow / neg_flow
                mfi = 100 - (100 / (1 + money_ratio))
            elif pos_flow > 0:
                mfi = 100.0

        # 20. Stochastic %K and %D (Sprint 23)
        self._stoch_high_14.append(candle.high)
        self._stoch_low_14.append(candle.low)

        stochastic_k = 50.0  # Default: neutral
        stochastic_d = 50.0
        if len(self._stoch_high_14) >= 14:
            high_14 = max(self._stoch_high_14)
            low_14 = min(self._stoch_low_14)
            if high_14 > low_14:
                stochastic_k = (candle.close - low_14) / (high_14 - low_14) * 100
            self._stoch_k_history.append(stochastic_k)
            if len(self._stoch_k_history) >= 3:
                stochastic_d = sum(self._stoch_k_history) / 3

        # Update normalizer RSI momentum (close already updated earlier)
        if rsi is not None:
            self._normalizer.update_rsi_momentum(rsi_momentum)

        features = {
            "close": candle.close,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "return_1": ret,
            "log_return_1": log_ret,
            "ma_fast": ma_fast or 0.0,
            "ma_slow": ma_slow or 0.0,
            "ma_delta": ma_delta or 0.0,
            "volatility": volatility or 0.0,
            "volatility_z": vol_z or 0.0,
            "volume": candle.volume,
            "volume_z": volume_z or 0.0,
            "rsi": rsi or 0.0,
            "rsi_prev": rsi_prev,
            "rsi_momentum": rsi_momentum,
            # Sprint 10: New features
            "price_momentum_3": price_momentum_3,
            "volume_trend": volume_trend,
            "body_ratio": body_ratio,
            "upper_wick_ratio": upper_wick_ratio,
            "lower_wick_ratio": lower_wick_ratio,
            "candle_direction": candle_direction,
            # Sprint 14: Scale-invariant ML features (17 new)
            "return_lag_1": return_lags[0],
            "return_lag_2": return_lags[1],
            "return_lag_3": return_lags[2],
            "return_lag_4": return_lags[3],
            "return_lag_5": return_lags[4],
            "atr_pct": atr_pct,
            "high_low_range_pct": high_low_range_pct,
            "ma_delta_pct": ma_delta_pct,
            "ema_10_spread_pct": ema_10_spread_pct,
            "ema_20_spread_pct": ema_20_spread_pct,
            "return_5": return_5,
            "return_10": return_10,
            "rsi_oversold": rsi_oversold,
            "rsi_overbought": rsi_overbought,
            "rsi_neutral": rsi_neutral,
            "volume_change_pct": volume_change_pct,
            "body_pct": body_pct,
            "vol_ratio": vol_ratio,
            # Sprint 16: New predictive features
            "vol_imbalance_20": vol_imbalance_20,
            "bb_position": bb_position,
            "adx": adx,
            "mfi": mfi,
            # Sprint 23: Stochastic
            "stochastic_k": stochastic_k,
            "stochastic_d": stochastic_d,
        }

        # Sprint 23: Apply normalization for train/inference consistency
        if self._normalize_output:
            features = normalize_features_streaming(features, self._normalizer)

        return features


def build_conditions(
    features: Dict[str, float],
    config: ConfigStore,
    ts: Optional[int] = None,
) -> Dict[str, str]:
    """Build pattern conditions from features.

    Args:
        features: Current market features
        config: Configuration store
        ts: Optional timestamp for extended conditions (hour, session, day_of_week)

    Returns:
        Conditions dict with trend, volatility, volume, and optionally
        hour, session, day_of_week if ts is provided.
    """
    trend_eps_mult = float(config.get("pattern.trend_eps_mult", 0.5))
    trend_eps = abs(features.get("volatility", 0.0) * features.get("close", 0.0)) * trend_eps_mult
    ma_delta = features.get("ma_delta", 0.0)

    if ma_delta > trend_eps:
        trend = "up"
    elif ma_delta < -trend_eps:
        trend = "down"
    else:
        trend = "flat"

    vol_z = features.get("volatility_z", 0.0)
    vol_high = float(config.get("pattern.vol_z_high", 1.0))
    vol_low = float(config.get("pattern.vol_z_low", -1.0))
    if vol_z >= vol_high:
        volatility = "high"
    elif vol_z <= vol_low:
        volatility = "low"
    else:
        volatility = "mid"

    volume_z = features.get("volume_z", 0.0)
    volume_high = float(config.get("pattern.volume_z_high", 1.0))
    volume_low = float(config.get("pattern.volume_z_low", -1.0))
    if volume_z >= volume_high:
        volume = "high"
    elif volume_z <= volume_low:
        volume = "low"
    else:
        volume = "mid"

    conditions: Dict[str, str] = {
        "trend": trend,
        "volatility": volatility,
        "volume": volume,
    }

    # Add extended conditions if timestamp provided
    if ts is not None:
        from datetime import datetime
        from titan.core.patterns import get_trading_session

        dt = datetime.utcfromtimestamp(ts)
        hour_bucket = int(config.get("pattern.hour_bucket_size", 1))
        hour = dt.hour
        if hour_bucket > 1:
            hour = (hour // hour_bucket) * hour_bucket
        conditions["hour"] = str(hour)
        conditions["session"] = get_trading_session(hour)
        conditions["day_of_week"] = str(dt.weekday())

    return conditions
