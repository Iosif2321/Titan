"""Скрипт для скачивания исторических данных."""

import asyncio
import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from btc_oracle.core.config import Config, load_config
from btc_oracle.core.log import get_logger, setup_logging
from btc_oracle.data.bybit_spot import BybitSpotClient
from btc_oracle.data.store import DataStore


async def main():
    """Главная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download historical data from Bybit")
    parser.add_argument("--days", type=int, default=30, help="Number of days to download")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol")
    parser.add_argument("--interval", type=str, default="1", help="Interval (1, 5, 15, 60)")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Config file")
    
    args = parser.parse_args()
    
    # Загружаем конфигурацию
    config = load_config(Path(args.config))
    setup_logging(level=config.logging.level)
    
    logger = get_logger(__name__)
    logger.info("Starting download", days=args.days, symbol=args.symbol)
    
    # Создаём хранилище
    store = DataStore(Path(config.storage.db_path))
    
    # Скачиваем данные
    async with BybitSpotClient(config.bybit) as client:
        candles = await client.download_history(
            symbol=args.symbol,
            interval=args.interval,
            days=args.days,
        )
        
        # Сохраняем
        store.add_candles_batch(candles, args.symbol, args.interval + "m")
        logger.info("Download complete", candles_count=len(candles))


if __name__ == "__main__":
    asyncio.run(main())

