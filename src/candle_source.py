from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import requests

from .config import RestConfig
from .types import Candle
from .utils import interval_to_ms, now_ms


def _cache_path(cache_dir: Path, symbol: str, tf: str, start_ms: int, end_ms: int) -> Path:
    safe_symbol = symbol.replace("/", "_")
    name = f"{safe_symbol}_{tf}_{start_ms}_{end_ms}.jsonl"
    return cache_dir / name


def _write_jsonl(path: Path, candles: List[Candle], symbol: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for candle in candles:
            handle.write(
                json.dumps(
                    {
                        "symbol": symbol,
                        "tf": candle.tf,
                        "ts_start": candle.start_ts,
                        "ts_end": candle.end_ts,
                        "o": candle.open,
                        "h": candle.high,
                        "l": candle.low,
                        "c": candle.close,
                        "volume": candle.volume,
                        "confirmed": candle.confirmed,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    tmp_path.replace(path)


def _read_jsonl(path: Path, symbol: Optional[str], tf: Optional[str]) -> List[Candle]:
    candles: List[Candle] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if symbol and record.get("symbol") not in (None, symbol):
                continue
            if tf and record.get("tf") not in (None, tf):
                continue
            try:
                start_ts = int(record.get("ts_start"))
                end_ts = int(record.get("ts_end", start_ts))
                candles.append(
                    Candle(
                        start_ts=start_ts,
                        end_ts=end_ts,
                        open=float(record.get("o")),
                        high=float(record.get("h")),
                        low=float(record.get("l")),
                        close=float(record.get("c")),
                        volume=float(record.get("volume")),
                        confirmed=bool(record.get("confirmed", True)),
                        tf=str(record.get("tf") or tf or ""),
                    )
                )
            except (TypeError, ValueError):
                continue
    candles.sort(key=lambda c: c.start_ts)
    return candles


class CandleSource:
    def load(self, symbol: str, tf: str, start_ms: int, end_ms: int) -> List[Candle]:
        raise NotImplementedError


@dataclass
class BybitRestCandleSource(CandleSource):
    rest_config: RestConfig
    cache_dir: Path = Path("cache")
    page_limit: int = 1000
    throttle_s: float = 0.05

    def load(self, symbol: str, tf: str, start_ms: int, end_ms: int) -> List[Candle]:
        cache_path = _cache_path(self.cache_dir, symbol, tf, start_ms, end_ms)
        if cache_path.exists():
            return _read_jsonl(cache_path, symbol, tf)
        candles = self._fetch_range(symbol, tf, start_ms, end_ms)
        _write_jsonl(cache_path, candles, symbol)
        return candles

    def _fetch_range(self, symbol: str, tf: str, start_ms: int, end_ms: int) -> List[Candle]:
        tf_ms = interval_to_ms(tf)
        if end_ms <= start_ms:
            return []
        candles: List[Candle] = []
        seen = set()
        current_end = end_ms
        now_cutoff = (now_ms() // tf_ms) * tf_ms
        while current_end > start_ms:
            page_start = max(start_ms, current_end - tf_ms * self.page_limit)
            page = self._fetch_page(symbol, tf, page_start, current_end)
            if not page:
                break
            page.sort(key=lambda c: c.start_ts)
            oldest = page[0].start_ts
            for candle in page:
                if candle.start_ts < start_ms or candle.start_ts > end_ms:
                    continue
                if candle.start_ts >= now_cutoff:
                    continue
                if candle.start_ts in seen:
                    continue
                seen.add(candle.start_ts)
                candles.append(candle)
            if oldest >= current_end:
                break
            current_end = oldest - tf_ms
            if self.throttle_s > 0:
                time.sleep(self.throttle_s)
        candles.sort(key=lambda c: c.start_ts)
        return candles

    def _fetch_page(self, symbol: str, tf: str, start_ms: int, end_ms: int) -> List[Candle]:
        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": tf,
            "limit": str(self.page_limit),
            "start": str(start_ms),
            "end": str(end_ms),
        }
        resp = requests.get(self.rest_config.kline_url, params=params, timeout=self.rest_config.timeout_s)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("retCode") != 0:
            raise RuntimeError(
                f"Bybit REST error: retCode={payload.get('retCode')} retMsg={payload.get('retMsg')}"
            )
        lst = payload.get("result", {}).get("list", [])
        candles: List[Candle] = []
        for item in lst:
            if not isinstance(item, list) or len(item) < 6:
                continue
            start_ts = int(item[0])
            candles.append(
                Candle(
                    start_ts=start_ts,
                    end_ts=start_ts + interval_to_ms(tf) - 1,
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5]),
                    confirmed=True,
                    tf=tf,
                )
            )
        return candles


@dataclass
class JsonlCandleSource(CandleSource):
    path: Path

    def load(self, symbol: str, tf: str, start_ms: int, end_ms: int) -> List[Candle]:
        candles = _read_jsonl(self.path, symbol, tf)
        return [c for c in candles if start_ms <= c.start_ts <= end_ms]
