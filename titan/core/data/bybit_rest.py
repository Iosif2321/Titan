import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from titan.core.data.schema import Candle


@dataclass
class CandleGap:
    """Represents a gap in candle data."""
    expected_ts: int
    actual_ts: int
    missing_count: int


def validate_candle_continuity(
    candles: List[Candle],
    interval_sec: int,
) -> Tuple[List[CandleGap], int]:
    """Check for gaps in candle sequence.

    Args:
        candles: Sorted list of candles
        interval_sec: Expected interval between candles in seconds

    Returns:
        Tuple of (list of gaps, total missing candles count)
    """
    if len(candles) < 2:
        return [], 0

    gaps: List[CandleGap] = []
    total_missing = 0

    for i in range(1, len(candles)):
        expected_ts = candles[i - 1].ts + interval_sec
        actual_ts = candles[i].ts
        delta = actual_ts - expected_ts

        if delta > 0:
            # There's a gap
            missing_count = delta // interval_sec
            total_missing += missing_count
            gaps.append(CandleGap(
                expected_ts=expected_ts,
                actual_ts=actual_ts,
                missing_count=missing_count,
            ))

    return gaps, total_missing


_INTERVAL_SECONDS: Dict[str, int] = {
    "1": 60,
    "3": 180,
    "5": 300,
    "15": 900,
    "30": 1800,
    "60": 3600,
    "120": 7200,
    "240": 14400,
    "360": 21600,
    "720": 43200,
    "D": 86400,
    "W": 604800,
}


def interval_to_ms(interval: str) -> int:
    if interval.isdigit():
        return int(interval) * 60_000
    interval = interval.upper()
    if interval in _INTERVAL_SECONDS:
        return _INTERVAL_SECONDS[interval] * 1000
    raise ValueError(f"Unsupported interval: {interval}")


def fetch_klines(
    symbol: str,
    interval: str,
    start_ts: int,
    end_ts: int,
    limit: int = 1000,
    sleep_sec: float = 0.2,
) -> List[Candle]:
    if start_ts >= end_ts:
        return []

    start_ms = start_ts * 1000
    end_ms = end_ts * 1000
    interval_ms = interval_to_ms(interval)

    seen: Dict[int, Candle] = {}
    cursor_end = end_ms
    last_oldest: Optional[int] = None

    while cursor_end >= start_ms:
        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": interval,
            "end": str(cursor_end),
            "limit": str(limit),
        }
        url = "https://api.bybit.com/v5/market/kline?" + urllib.parse.urlencode(params)

        with urllib.request.urlopen(url, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        if payload.get("retCode") != 0:
            raise RuntimeError(f"Bybit error: {payload}")

        items = payload.get("result", {}).get("list", [])
        if not items:
            break

        oldest: Optional[int] = None
        for item in items:
            try:
                ts_ms = int(float(item[0]))
                if oldest is None or ts_ms < oldest:
                    oldest = ts_ms
                if ts_ms < start_ms or ts_ms > end_ms:
                    continue
                ts = ts_ms // 1000
                candle = Candle(
                    ts=ts,
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5]),
                )
                seen[ts] = candle
            except (TypeError, ValueError, IndexError):
                continue

        if oldest is None:
            break
        if last_oldest is not None and oldest >= last_oldest:
            break
        last_oldest = oldest
        cursor_end = oldest - interval_ms

        if sleep_sec:
            time.sleep(sleep_sec)

    candles = list(seen.values())
    candles.sort(key=lambda c: c.ts)
    return candles
