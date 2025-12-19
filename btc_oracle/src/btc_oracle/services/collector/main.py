"""Collector service: собирает confirmed candles от Bybit и пишет в БД."""

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from btc_oracle.core.config import Config, load_config
from btc_oracle.core.log import get_logger, setup_logging
from btc_oracle.core.types import Candle
from btc_oracle.data.bybit_spot import BybitSpotClient
from btc_oracle.db import AsyncSessionLocal, Candle as CandleModel, init_db
from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert

logger = get_logger(__name__)


async def save_candle(session, candle: Candle, symbol: str, timeframe: str):
    """Сохранить свечу в БД."""
    stmt = insert(CandleModel).values(
        time=datetime.fromtimestamp(candle.timestamp / 1000, tz=timezone.utc),
        symbol=symbol,
        timeframe=timeframe,
        open=candle.open,
        high=candle.high,
        low=candle.low,
        close=candle.close,
        volume=candle.volume,
        confirmed=1 if candle.confirmed else 0,
    ).on_conflict_do_update(
        index_elements=["time", "symbol", "timeframe"],
        set_={
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
            "confirmed": 1 if candle.confirmed else 0,
        },
    )
    await session.execute(stmt)
    await session.commit()


async def get_candle_count(session, symbol: str) -> int:
    """Получить количество свечей в БД."""
    stmt = select(func.count()).select_from(CandleModel).where(CandleModel.symbol == symbol)
    result = await session.execute(stmt)
    return result.scalar() or 0


async def ensure_history(
    client: BybitSpotClient,
    symbol: str,
    timeframe: str,
    days: int = 365,
):
    """Обеспечить наличие исторических данных."""
    async with AsyncSessionLocal() as session:
        count = await get_candle_count(session, symbol)

        # Приводим timeframe к:
        # - interval_api: строка для Bybit (например "1", "5", "15", "60")
        # - timeframe_db: строка для хранения в БД (например "1m", "5m")
        interval_api = timeframe.replace("m", "") if timeframe.endswith("m") else timeframe
        timeframe_db = timeframe if timeframe.endswith("m") else f"{timeframe}m"

        # Оценка target_count на основе минут на свечу
        try:
            minutes_per_candle = int(interval_api)
        except Exception:
            minutes_per_candle = 1
        target_count = int(days * 24 * 60 / max(1, minutes_per_candle))
        
        if count >= target_count * 0.9:  # 90% достаточно
            logger.info(f"History sufficient: {count} candles (target: {target_count})")
            return
        
        logger.info(f"Fetching history: {count}/{target_count} candles")
        
        fetched = 0
        try:
            async for batch in client.download_history(
                symbol=symbol,
                interval=interval_api,
                days=days,
            ):
                async with AsyncSessionLocal() as batch_session:
                    for candle in batch:
                        await save_candle(batch_session, candle, symbol, timeframe_db)
                    fetched += len(batch)
                    logger.info(f"Fetched {fetched} candles so far")
        except Exception as e:
            # Не валим сервис целиком: даже без истории можно продолжить live сбор.
            logger.error(f"History download failed, continuing with live stream: {e}", exc_info=True)


async def run_collector(config: Config):
    """Главный цикл collector service."""
    logger.info("Starting collector service")
    
    # Инициализация БД
    await init_db()
    
    # Проверяем историю
    async with BybitSpotClient(config.bybit) as client:
        await ensure_history(client, config.symbol, config.timeframe, days=365)
    
    # Подписываемся на live свечи
    async with BybitSpotClient(config.bybit) as client:
        async def on_new_candle(candle: Candle):
            """Callback для новой confirmed свечи."""
            async with AsyncSessionLocal() as session:
                await save_candle(session, candle, config.symbol, config.timeframe)
                logger.debug(f"Saved candle: {candle.ts} close={candle.close:.2f}")
        
        # Подписываемся на свечи
        await client.subscribe_klines(
            symbol=config.symbol,
            interval=config.timeframe.replace("m", ""),
            callback=on_new_candle,
        )


def main():
    """Точка входа для collector service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collector service for Titan Oracle")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Config file")
    parser.add_argument("--ensure-days", type=int, default=365, help="Days of history to ensure")
    
    args = parser.parse_args()
    
    config_path = Path(__file__).parent.parent.parent.parent.parent / args.config
    config = load_config(config_path)
    
    setup_logging(
        level=config.logging.level,
        structured=config.logging.structured,
        log_file=Path(config.logging.log_file) if config.logging.log_file else None,
    )
    
    try:
        asyncio.run(run_collector(config))
    except KeyboardInterrupt:
        logger.info("Collector stopped")


if __name__ == "__main__":
    main()
