from __future__ import annotations

from typing import Tuple

import numpy as np


def safe_div(n: float, d: float, default: float = 0.0) -> float:
    if d == 0:
        return default
    return n / d


def sma(values: np.ndarray, period: int) -> float:
    if period <= 0 or values.size < period:
        return float(values[-1]) if values.size else 0.0
    return float(np.mean(values[-period:]))


def ema_series(values: np.ndarray, period: int) -> np.ndarray:
    if period <= 0 or values.size == 0:
        return np.array([], dtype=np.float32)
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(values, dtype=np.float32)
    out[0] = values[0]
    for i in range(1, values.size):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def ema_last(values: np.ndarray, period: int) -> float:
    series = ema_series(values, period)
    if series.size == 0:
        return 0.0
    return float(series[-1])


def macd(values: np.ndarray, fast: int, slow: int, signal: int) -> Tuple[float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0
    ema_fast = ema_series(values, fast)
    ema_slow = ema_series(values, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema_series(macd_line, signal)
    hist = macd_line - signal_line
    return float(macd_line[-1]), float(signal_line[-1]), float(hist[-1])


def rsi(closes: np.ndarray, period: int) -> float:
    if closes.size < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.clip(deltas, 0.0, None)
    losses = -np.clip(deltas, None, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def stochastic(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    k_period: int,
    d_period: int,
    smooth_k: int,
) -> Tuple[float, float]:
    if closes.size < k_period + d_period:
        return 50.0, 50.0
    raw_k_values = []
    for idx in range(d_period):
        end = closes.size - idx
        start = end - k_period
        window_high = highs[start:end]
        window_low = lows[start:end]
        window_close = closes[end - 1]
        low_min = float(np.min(window_low))
        high_max = float(np.max(window_high))
        denom = high_max - low_min
        if denom == 0:
            k_val = 50.0
        else:
            k_val = 100.0 * (window_close - low_min) / denom
        raw_k_values.append(k_val)
    raw_k_values = raw_k_values[::-1]
    if smooth_k > 1 and len(raw_k_values) >= smooth_k:
        k_smooth = float(np.mean(raw_k_values[-smooth_k:]))
    else:
        k_smooth = float(raw_k_values[-1])
    d_val = float(np.mean(raw_k_values[-d_period:]))
    return k_smooth, d_val


def cci(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> float:
    if closes.size < period:
        return 0.0
    tp = (highs + lows + closes) / 3.0
    tp_slice = tp[-period:]
    mean_tp = np.mean(tp_slice)
    dev = np.mean(np.abs(tp_slice - mean_tp))
    if dev == 0:
        return 0.0
    return float((tp_slice[-1] - mean_tp) / (0.015 * dev))


def obv(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    if closes.size == 0:
        return np.array([], dtype=np.float32)
    out = np.zeros_like(closes, dtype=np.float32)
    for i in range(1, closes.size):
        if closes[i] > closes[i - 1]:
            out[i] = out[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            out[i] = out[i - 1] - volumes[i]
        else:
            out[i] = out[i - 1]
    return out


def mfi(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray, period: int) -> float:
    if closes.size < period + 1:
        return 50.0
    tp = (highs + lows + closes) / 3.0
    raw_flow = tp * volumes
    pos_flow = []
    neg_flow = []
    for i in range(1, tp.size):
        if tp[i] > tp[i - 1]:
            pos_flow.append(raw_flow[i])
            neg_flow.append(0.0)
        elif tp[i] < tp[i - 1]:
            pos_flow.append(0.0)
            neg_flow.append(raw_flow[i])
        else:
            pos_flow.append(0.0)
            neg_flow.append(0.0)
    pos = np.sum(pos_flow[-period:])
    neg = np.sum(neg_flow[-period:])
    if neg == 0:
        return 100.0 if pos > 0 else 50.0
    mfr = pos / neg
    return 100.0 - (100.0 / (1.0 + mfr))


def parabolic_sar(highs: np.ndarray, lows: np.ndarray, step: float, max_step: float) -> Tuple[float, bool]:
    if highs.size < 2:
        return float(lows[-1] if lows.size else 0.0), True
    uptrend = highs[1] >= highs[0]
    sar = lows[0] if uptrend else highs[0]
    ep = highs[0] if uptrend else lows[0]
    af = step

    for i in range(1, highs.size):
        sar = sar + af * (ep - sar)
        if uptrend:
            sar = min(sar, lows[i - 1], lows[i])
            if highs[i] > ep:
                ep = highs[i]
                af = min(af + step, max_step)
            if lows[i] < sar:
                uptrend = False
                sar = ep
                ep = lows[i]
                af = step
        else:
            sar = max(sar, highs[i - 1], highs[i])
            if lows[i] < ep:
                ep = lows[i]
                af = min(af + step, max_step)
            if highs[i] > sar:
                uptrend = True
                sar = ep
                ep = highs[i]
                af = step
    return float(sar), uptrend


def volume_zscore(volumes: np.ndarray) -> float:
    if volumes.size == 0:
        return 0.0
    mean = float(np.mean(volumes))
    std = float(np.std(volumes))
    if std == 0:
        return 0.0
    return float((volumes[-1] - mean) / std)


def returns_bps(closes: np.ndarray) -> np.ndarray:
    if closes.size < 2:
        return np.array([], dtype=np.float32)
    prev = closes[:-1]
    curr = closes[1:]
    ret = (curr - prev) / np.maximum(prev, 1e-12) * 10_000.0
    return ret.astype(np.float32)
