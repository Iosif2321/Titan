import asyncio
import json
from typing import AsyncIterator, Iterable, List, Optional

from titan.core.data.schema import Candle, parse_timestamp


class BybitSpotWebSocket:
    def __init__(
        self,
        symbol: str,
        interval: str = "1",
        url: str = "wss://stream.bybit.com/v5/public/spot",
        ping_interval: int = 20,
        ping_timeout: int = 20,
    ) -> None:
        self.symbol = symbol
        self.interval = interval
        self.url = url
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._last_ts: Optional[int] = None

    async def stream(self) -> AsyncIterator[Candle]:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError(
                "websockets is required for live mode. Install with: pip install websockets"
            ) from exc

        backoff = 1
        while True:
            try:
                async with websockets.connect(
                    self.url,
                    ping_interval=self._ping_interval,
                    ping_timeout=self._ping_timeout,
                ) as ws:
                    sub = {"op": "subscribe", "args": [f"kline.{self.interval}.{self.symbol}"]}
                    await ws.send(json.dumps(sub))
                    self._last_ts = None

                    async for message in ws:
                        candles = _parse_message(message)
                        for candle in candles:
                            if self._last_ts is not None and candle.ts <= self._last_ts:
                                continue
                            self._last_ts = candle.ts
                            yield candle

                backoff = 1
            except Exception:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)


def _parse_message(message: str) -> List[Candle]:
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return []

    if not isinstance(payload, dict):
        return []

    if payload.get("op") in {"pong", "ping"}:
        return []

    data = payload.get("data")
    if data is None:
        return []

    if isinstance(data, dict):
        items: Iterable[dict] = [data]
    elif isinstance(data, list):
        items = data
    else:
        return []

    candles: List[Candle] = []
    for item in items:
        if not isinstance(item, dict):
            continue

        confirm = item.get("confirm")
        if confirm is False:
            continue

        ts_value = item.get("start") or item.get("timestamp") or item.get("ts")
        if ts_value is None:
            continue

        try:
            candle = Candle(
                ts=parse_timestamp(str(ts_value)),
                open=float(item.get("open", 0.0)),
                high=float(item.get("high", 0.0)),
                low=float(item.get("low", 0.0)),
                close=float(item.get("close", 0.0)),
                volume=float(item.get("volume", item.get("vol", 0.0))),
            )
        except (TypeError, ValueError):
            continue

        candles.append(candle)

    return candles
