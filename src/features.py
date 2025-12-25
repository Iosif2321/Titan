from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .config import FeatureConfig
from .indicators import (
    cci,
    macd,
    mfi,
    obv,
    parabolic_sar,
    returns_bps,
    rsi,
    sma,
    stochastic,
    volume_zscore,
)
from .types import Candle

MODEL_TREND = "TRENDVIC"
MODEL_OSC = "OSCILLATOR"
MODEL_VOL = "VOLUMEMETRIX"
MODEL_TYPES = [MODEL_TREND, MODEL_OSC, MODEL_VOL]


@dataclass(frozen=True)
class FeatureSpec:
    input_size: int
    feature_names: List[str]
    lookback: int
    required_lookback: int
    schema_version: str
    model_type: str


@dataclass(frozen=True)
class FeatureBundle:
    values: np.ndarray
    context_coarse: Dict[str, str]
    context_fine: Dict[str, str]


class FeatureBuilder:
    def __init__(self, config: FeatureConfig, model_type: str) -> None:
        self.config = config
        self.model_type = model_type
        if model_type == MODEL_TREND:
            self._builder = TrendvicFeatureBuilder(config)
        elif model_type == MODEL_OSC:
            self._builder = OscillatorFeatureBuilder(config)
        elif model_type == MODEL_VOL:
            self._builder = VolumeMetrixFeatureBuilder(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        self.spec = self._builder.spec

    def build(self, candles: List[Candle]) -> Optional[FeatureBundle]:
        return self._builder.build(candles)


class TrendvicFeatureBuilder:
    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self._required = max(
            config.lookback,
            max(config.ma_periods) + 1,
            config.macd_slow + config.macd_signal,
        )
        feature_names = [
            "ret_bps_scaled",
            "range_pct",
            "body_pct",
            "vol_z",
        ]
        for period in config.ma_periods:
            feature_names.append(f"ma_ratio_{period}")
        feature_names.extend(["macd", "macd_signal", "macd_hist", "sar_rel"])
        self.spec = FeatureSpec(
            input_size=len(feature_names),
            feature_names=feature_names,
            lookback=config.lookback,
            required_lookback=self._required,
            schema_version=config.schema_version,
            model_type=MODEL_TREND,
        )

    def build(self, candles: List[Candle]) -> Optional[FeatureBundle]:
        if len(candles) < self._required:
            return None
        window = candles[-self.config.lookback :]
        opens = np.array([c.open for c in window], dtype=np.float32)
        highs = np.array([c.high for c in window], dtype=np.float32)
        lows = np.array([c.low for c in window], dtype=np.float32)
        closes = np.array([c.close for c in window], dtype=np.float32)
        volumes = np.array([c.volume for c in window], dtype=np.float32)

        ret = returns_bps(closes)
        ret_scaled = float(ret[-1] / 100.0) if ret.size else 0.0
        range_pct = float((highs[-1] - lows[-1]) / max(closes[-1], 1e-12) * 100.0)
        body_pct = float((closes[-1] - opens[-1]) / max(opens[-1], 1e-12) * 100.0)
        vol_window = volumes[-self.config.vol_z_window :]
        vol_z = volume_zscore(vol_window)

        ma_ratios = []
        for period in self.config.ma_periods:
            ma_val = sma(closes, period)
            ma_ratios.append(((closes[-1] - ma_val) / max(ma_val, 1e-12)) * 100.0)

        macd_line, macd_signal, macd_hist = macd(
            closes, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal
        )
        macd_scale = max(closes[-1], 1e-12)
        macd_line = macd_line / macd_scale * 100.0
        macd_signal = macd_signal / macd_scale * 100.0
        macd_hist = macd_hist / macd_scale * 100.0

        sar, sar_up = parabolic_sar(highs, lows, self.config.sar_step, self.config.sar_max)
        sar_rel = (sar - closes[-1]) / max(closes[-1], 1e-12) * 100.0

        values = [ret_scaled, range_pct, body_pct, vol_z]
        values.extend(ma_ratios)
        values.extend([macd_line, macd_signal, macd_hist, sar_rel])
        features = np.array(values, dtype=np.float32)

        ma_fast = sma(closes, self.config.ma_periods[0])
        ma_slow = sma(closes, self.config.ma_periods[-1])
        if closes.size > self.config.ma_periods[0]:
            ma_fast_prev = sma(closes[:-1], self.config.ma_periods[0])
        else:
            ma_fast_prev = ma_fast
        slope_bps = (ma_fast - ma_fast_prev) / max(ma_fast_prev, 1e-12) * 10_000.0
        if slope_bps > self.config.ma_slope_bps:
            ma_slope_zone = "UP"
        elif slope_bps < -self.config.ma_slope_bps:
            ma_slope_zone = "DOWN"
        else:
            ma_slope_zone = "FLAT"

        if abs(macd_hist) < 1e-6:
            macd_sign = "ZERO"
        else:
            macd_sign = "POS" if macd_hist > 0 else "NEG"

        sar_side = "BELOW" if sar < closes[-1] else "ABOVE"

        ret_bps = float(ret[-1]) if ret.size else 0.0
        if abs(ret_bps) <= self.config.ret_bin_bps:
            ret_bin = "MID"
        else:
            ret_bin = "POS" if ret_bps > 0 else "NEG"

        ret_vol = float(np.std(ret)) if ret.size else 0.0
        if ret_vol < self.config.vol_low_bps:
            vol_bin = "LOW"
        elif ret_vol > self.config.vol_high_bps:
            vol_bin = "HIGH"
        else:
            vol_bin = "MID"

        context_fine = {
            "ma_fast_gt_slow": "1" if ma_fast > ma_slow else "0",
            "ma_slope_zone": ma_slope_zone,
            "macd_hist_sign": macd_sign,
            "sar_side": sar_side,
            "last_ret_bin": ret_bin,
            "vol_bin": vol_bin,
        }
        context_coarse = {
            "ma_fast_gt_slow": context_fine["ma_fast_gt_slow"],
            "macd_hist_sign": context_fine["macd_hist_sign"],
            "sar_side": context_fine["sar_side"],
        }
        return FeatureBundle(values=features, context_coarse=context_coarse, context_fine=context_fine)


class OscillatorFeatureBuilder:
    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self._required = max(
            config.lookback,
            config.rsi_period + 1,
            config.stoch_k + config.stoch_d,
            config.cci_period,
        )
        feature_names = ["rsi", "stoch_k", "stoch_d", "cci"]
        self.spec = FeatureSpec(
            input_size=len(feature_names),
            feature_names=feature_names,
            lookback=config.lookback,
            required_lookback=self._required,
            schema_version=config.schema_version,
            model_type=MODEL_OSC,
        )

    def build(self, candles: List[Candle]) -> Optional[FeatureBundle]:
        if len(candles) < self._required:
            return None
        window = candles[-self.config.lookback :]
        highs = np.array([c.high for c in window], dtype=np.float32)
        lows = np.array([c.low for c in window], dtype=np.float32)
        closes = np.array([c.close for c in window], dtype=np.float32)

        rsi_val = rsi(closes, self.config.rsi_period)
        stoch_k, stoch_d = stochastic(
            highs, lows, closes, self.config.stoch_k, self.config.stoch_d, self.config.stoch_smooth
        )
        cci_val = cci(highs, lows, closes, self.config.cci_period)

        rsi_scaled = (rsi_val - 50.0) / 50.0
        stoch_k_scaled = (stoch_k - 50.0) / 50.0
        stoch_d_scaled = (stoch_d - 50.0) / 50.0
        cci_scaled = float(np.clip(cci_val / 200.0, -1.0, 1.0))

        features = np.array(
            [rsi_scaled, stoch_k_scaled, stoch_d_scaled, cci_scaled],
            dtype=np.float32,
        )

        rsi_zone = "LOW" if rsi_val <= 30 else "HIGH" if rsi_val >= 70 else "MID"
        stoch_zone = "LOW" if stoch_k <= 20 else "HIGH" if stoch_k >= 80 else "MID"
        cci_zone = "LOW" if cci_val <= -100 else "HIGH" if cci_val >= 100 else "MID"
        context_fine = {"rsi_zone": rsi_zone, "stoch_zone": stoch_zone, "cci_zone": cci_zone}
        context_coarse = {"rsi_zone": rsi_zone, "stoch_zone": stoch_zone}
        return FeatureBundle(values=features, context_coarse=context_coarse, context_fine=context_fine)


class VolumeMetrixFeatureBuilder:
    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self._required = max(
            config.lookback,
            config.mfi_period + 1,
            config.obv_slope_period + 1,
        )
        feature_names = [
            "ret_bps_scaled",
            "range_pct",
            "body_pct",
            "vol_z",
            "obv_slope",
            "obv_z",
            "mfi",
        ]
        self.spec = FeatureSpec(
            input_size=len(feature_names),
            feature_names=feature_names,
            lookback=config.lookback,
            required_lookback=self._required,
            schema_version=config.schema_version,
            model_type=MODEL_VOL,
        )

    def build(self, candles: List[Candle]) -> Optional[FeatureBundle]:
        if len(candles) < self._required:
            return None
        window = candles[-self.config.lookback :]
        opens = np.array([c.open for c in window], dtype=np.float32)
        highs = np.array([c.high for c in window], dtype=np.float32)
        lows = np.array([c.low for c in window], dtype=np.float32)
        closes = np.array([c.close for c in window], dtype=np.float32)
        volumes = np.array([c.volume for c in window], dtype=np.float32)

        ret = returns_bps(closes)
        ret_scaled = float(ret[-1] / 100.0) if ret.size else 0.0
        range_pct = float((highs[-1] - lows[-1]) / max(closes[-1], 1e-12) * 100.0)
        body_pct = float((closes[-1] - opens[-1]) / max(opens[-1], 1e-12) * 100.0)
        vol_window = volumes[-self.config.vol_z_window :]
        vol_z = volume_zscore(vol_window)

        obv_series = obv(closes, volumes)
        obv_slope = obv_series[-1] - obv_series[-self.config.obv_slope_period]
        vol_mean = float(np.mean(volumes)) if volumes.size else 1.0
        obv_slope_scaled = float(obv_slope / max(vol_mean, 1e-12))
        obv_z = float((obv_series[-1] - np.mean(obv_series)) / (np.std(obv_series) + 1e-12))

        mfi_val = mfi(highs, lows, closes, volumes, self.config.mfi_period)
        mfi_scaled = (mfi_val - 50.0) / 50.0

        features = np.array(
            [ret_scaled, range_pct, body_pct, vol_z, obv_slope_scaled, obv_z, mfi_scaled],
            dtype=np.float32,
        )

        mfi_zone = "LOW" if mfi_val <= 20 else "HIGH" if mfi_val >= 80 else "MID"
        if obv_slope > 0:
            obv_sign = "POS"
        elif obv_slope < 0:
            obv_sign = "NEG"
        else:
            obv_sign = "ZERO"
        if vol_z <= -1:
            vol_zone = "LOW"
        elif vol_z >= 1:
            vol_zone = "HIGH"
        else:
            vol_zone = "MID"
        if range_pct < self.config.range_low_pct:
            range_bin = "LOW"
        elif range_pct > self.config.range_high_pct:
            range_bin = "HIGH"
        else:
            range_bin = "MID"
        context_fine = {
            "mfi_zone": mfi_zone,
            "obv_trend": obv_sign,
            "vol_z": vol_zone,
            "range_bin": range_bin,
        }
        context_coarse = {"mfi_zone": mfi_zone, "vol_z": vol_zone}
        return FeatureBundle(values=features, context_coarse=context_coarse, context_fine=context_fine)
