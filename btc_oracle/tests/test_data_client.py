"""Smoke тесты для data client."""

import pytest
from datetime import datetime

from btc_oracle.core.config import BybitConfig
from btc_oracle.core.types import Candle
from btc_oracle.data.bybit_spot import BybitSpotClient


@pytest.mark.asyncio
async def test_bybit_client_init():
    """Тест инициализации BybitSpotClient."""
    config = BybitConfig(
        spot_rest_url="https://api.bybit.com/v5/market",
        spot_ws_url="wss://stream.bybit.com/v5/public/spot",
    )
    client = BybitSpotClient(config)
    assert client.config == config
    await client.close()


@pytest.mark.asyncio
async def test_bybit_get_klines():
    """Тест получения исторических свечей (только confirmed)."""
    config = BybitConfig(
        spot_rest_url="https://api.bybit.com/v5/market",
        spot_ws_url="wss://stream.bybit.com/v5/public/spot",
    )
    
    async with BybitSpotClient(config) as client:
        candles = await client.get_klines(
            symbol="BTCUSDT",
            interval="1",
            limit=10,
            only_confirmed=True,
        )
        
        assert len(candles) > 0
        assert all(isinstance(c, Candle) for c in candles)
        assert all(c.confirmed for c in candles)  # Все свечи должны быть confirmed
        
        # Проверяем структуру свечи
        candle = candles[0]
        assert candle.open > 0
        assert candle.high >= candle.low
        assert candle.close > 0
        assert candle.volume >= 0


@pytest.mark.asyncio
async def test_candle_timestamp_format():
    """Тест формата timestamp в свечах."""
    config = BybitConfig(
        spot_rest_url="https://api.bybit.com/v5/market",
        spot_ws_url="wss://stream.bybit.com/v5/public/spot",
    )
    
    async with BybitSpotClient(config) as client:
        candles = await client.get_klines(
            symbol="BTCUSDT",
            interval="1",
            limit=5,
            only_confirmed=True,
        )
        
        for candle in candles:
            assert isinstance(candle.ts, datetime)
            assert candle.timestamp > 0  # timestamp в миллисекундах
