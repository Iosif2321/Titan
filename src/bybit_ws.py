import asyncio
import json
import logging
from typing import AsyncIterator, List

import websockets

from .config import DataConfig
from .types import Candle


def _parse_kline_message(raw: str) -> List[Candle]:
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        return []

    topic = msg.get("topic", "")
    if not isinstance(topic, str) or not topic.startswith("kline."):
        return []

    data = msg.get("data", [])
    if not isinstance(data, list):
        return []

    candles: List[Candle] = []
    for item in data:
        try:
            candles.append(
                Candle(
                    start_ts=int(item["start"]),
                    end_ts=int(item["end"]),
                    open=float(item["open"]),
                    high=float(item["high"]),
                    low=float(item["low"]),
                    close=float(item["close"]),
                    volume=float(item["volume"]),
                    confirmed=bool(item.get("confirm", False)),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return candles


async def stream_candles(config: DataConfig) -> AsyncIterator[Candle]:
    topic = f"kline.{config.interval}.{config.symbol}"
    sub_msg = {"op": "subscribe", "args": [topic]}

    while True:
        try:
            async with websockets.connect(
                config.ws_url, ping_interval=20, ping_timeout=20
            ) as ws:
                await ws.send(json.dumps(sub_msg))

                async for raw in ws:
                    candles = _parse_kline_message(raw)
                    for candle in candles:
                        if config.use_confirmed_only and not candle.confirmed:
                            continue
                        yield candle
        except Exception as exc:
            logging.warning("WebSocket error: %s; reconnecting...", exc)
            await asyncio.sleep(config.reconnect_delay)
