from __future__ import annotations

import time
from typing import List, Optional

import requests

from .config import RestConfig
from .types import Candle
from .utils import interval_to_ms


def fetch_klines(
    symbol: str, interval: str, limit: int, rest_config: RestConfig
) -> List[Candle]:
    params = {
        "category": "spot",
        "symbol": symbol,
        "interval": interval,
        "limit": str(limit),
    }
    resp = requests.get(rest_config.kline_url, params=params, timeout=rest_config.timeout_s)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("retCode") != 0:
        raise RuntimeError(
            f"Bybit REST error: retCode={payload.get('retCode')} retMsg={payload.get('retMsg')}"
        )

    lst = payload.get("result", {}).get("list", [])
    candles: List[Candle] = []
    for item in reversed(lst):
        if not isinstance(item, list) or len(item) < 6:
            continue
        start_ts = int(item[0])
        candles.append(
            Candle(
                start_ts=start_ts,
                end_ts=start_ts + interval_to_ms(interval) - 1,
                open=float(item[1]),
                high=float(item[2]),
                low=float(item[3]),
                close=float(item[4]),
                volume=float(item[5]),
                confirmed=True,
                tf=interval,
            )
        )

    dedup: List[Candle] = []
    last_ts: Optional[int] = None
    for candle in candles:
        if last_ts is not None and candle.start_ts == last_ts:
            dedup[-1] = candle
        else:
            dedup.append(candle)
            last_ts = candle.start_ts

    interval_ms = interval_to_ms(interval)
    now_ms = int(time.time() * 1000)
    current_start = (now_ms // interval_ms) * interval_ms
    return [c for c in dedup if c.start_ts < current_start]
