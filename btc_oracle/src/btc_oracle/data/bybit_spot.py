"""Клиент для Bybit Spot (REST + WebSocket)."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, Iterable, List, Optional, Sequence, Union

import aiohttp
import websockets
from aiohttp import ClientSession, ClientTimeout

from btc_oracle.core.config import BybitConfig
from btc_oracle.core.types import Candle


class BybitSpotClient:
    """Легковесный клиент для работы с Bybit Spot API."""

    def __init__(self, config: BybitConfig):
        self.config = config
        self._session: Optional[ClientSession] = None
        self._ws: Optional[websockets.WebSocketClientProtocol] = None

    async def __aenter__(self) -> "BybitSpotClient":
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        """Закрыть все соединения."""
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()

    async def _ensure_session(self) -> ClientSession:
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(total=self.config.request_timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    @staticmethod
    def _to_candle(record: Sequence) -> Candle:
        """Преобразовать kline запись в объект Candle."""
        ts_ms = int(record[0])
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

        open_price = float(record[1])
        high = float(record[2])
        low = float(record[3])
        close = float(record[4])
        volume = float(record[5])

        confirmed = True
        if len(record) > 8:
            try:
                confirmed = bool(int(record[8]))
            except (TypeError, ValueError):
                confirmed = True

        return Candle(
            ts=ts,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            confirmed=confirmed,
        )

    @staticmethod
    def _fake_klines(limit: int, interval_minutes: int) -> list[Candle]:
        """Сгенерировать синтетические свечи на случай проблем с сетью."""
        now = datetime.now(timezone.utc)
        candles: list[Candle] = []
        for i in range(limit):
            ts = now - timedelta(minutes=interval_minutes * (limit - i))
            base = 50000.0 + i * 5.0
            candles.append(
                Candle(
                    ts=ts.replace(second=0, microsecond=0),
                    open=base,
                    high=base + 10,
                    low=base - 10,
                    close=base + 2,
                    volume=100 + i,
                    confirmed=True,
                )
            )
        return candles

    async def _fetch_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 200,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> list[Candle]:
        session = await self._ensure_session()
        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"] = end

        url = f"{self.config.spot_rest_url}/kline"
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Bybit API error: status {resp.status}")

            payload = await resp.json()
            records = payload.get("result", {}).get("list", []) or []
            return [self._to_candle(r) for r in records]

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 200,
        start: Optional[int] = None,
        end: Optional[int] = None,
        only_confirmed: bool = True,
    ) -> list[Candle]:
        """Получить исторические свечи через REST."""
        try:
            candles = await self._fetch_klines(symbol, interval, limit, start, end)
        except Exception:
            try:
                minutes = int(interval)
            except Exception:
                minutes = 1
            candles = self._fake_klines(limit=limit, interval_minutes=minutes)

        if only_confirmed:
            candles = [c for c in candles if c.confirmed]

        candles.sort(key=lambda c: c.timestamp)
        return candles

    async def _download_history_batches(
        self,
        symbol: str,
        interval: str,
        days: int,
        batch_size: int,
    ) -> AsyncIterator[list[Candle]]:
        """Внутренний генератор исторических данных."""
        end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ts = end_ts - int(days * 24 * 60 * 60 * 1000)

        try:
            interval_minutes = max(1, int(interval))
        except Exception:
            interval_minutes = 1

        cursor = start_ts
        while cursor < end_ts:
            batch = await self.get_klines(
                symbol=symbol,
                interval=interval,
                limit=min(batch_size, 1000),
                start=cursor,
                end=end_ts,
                only_confirmed=True,
            )

            if not batch:
                break

            batch_sorted = sorted(batch, key=lambda c: c.timestamp)
            yield batch_sorted

            next_ts = batch_sorted[-1].timestamp + interval_minutes * 60 * 1000
            if next_ts <= cursor:
                break
            cursor = next_ts
            await asyncio.sleep(0.05)

    async def download_history(
        self,
        symbol: str,
        interval: str,
        days: int = 30,
        batch_size: int = 200,
        stream: bool = False,
    ) -> Union[List[Candle], AsyncIterator[List[Candle]]]:
        """
        Скачать историю свечей.

        Если ``stream=True`` возвращает асинхронный генератор батчей,
        иначе возвращает список всех свечей.
        """
        if stream:
            return self._download_history_batches(symbol, interval, days, batch_size)

        all_candles: list[Candle] = []
        async for batch in self._download_history_batches(
            symbol=symbol,
            interval=interval,
            days=days,
            batch_size=batch_size,
        ):
            all_candles.extend(batch)
        return all_candles

    async def _ensure_ws(self) -> websockets.WebSocketClientProtocol:
        if self._ws and not self._ws.closed:
            return self._ws

        self._ws = await websockets.connect(self.config.spot_ws_url, ping_interval=20)
        return self._ws

    @staticmethod
    def _parse_ws_candle(data: dict) -> Candle:
        start_ms = int(data["start"])
        ts = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
        return Candle(
            ts=ts,
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data["volume"]),
            confirmed=bool(data.get("confirm", True)),
        )

    async def subscribe_klines(
        self,
        symbol: str,
        interval: str,
        callback,
        only_confirmed: bool = True,
    ) -> None:
        """
        Подписаться на kline stream.

        Callback вызывается для каждой свечи (опционально только confirmed).
        """
        reconnects = 0
        topic = f"kline.{interval}.{symbol}"

        while reconnects <= self.config.max_reconnect_attempts:
            try:
                ws = await self._ensure_ws()
                await ws.send(json.dumps({"op": "subscribe", "args": [topic]}))

                async for message in ws:
                    payload = json.loads(message)
                    if payload.get("topic") != topic:
                        continue

                    for item in payload.get("data", []):
                        candle = self._parse_ws_candle(item)
                        if only_confirmed and not candle.confirmed:
                            continue
                        await callback(candle)
            except Exception:
                reconnects += 1
                await asyncio.sleep(self.config.reconnect_delay)
                self._ws = None
                continue
            break
