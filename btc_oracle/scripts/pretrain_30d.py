"""Скрипт для pretrain на 30 днях данных (simulated-online)."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from btc_oracle.core.config import Config, load_config
from btc_oracle.core.log import get_logger, setup_logging


async def main():
    """Главная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pretrain on 30 days of data")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Config file")
    parser.add_argument("--skip", action="store_true", help="Skip pretraining")
    
    args = parser.parse_args()
    
    config = load_config(Path(args.config))
    setup_logging(level=config.logging.level)
    
    logger = get_logger(__name__)
    
    if args.skip:
        logger.info("Skipping pretraining")
        return
    
    logger.info("Starting pretraining (simulated-online)")
    # TODO: Реализовать simulated-online проход по данным
    logger.info("Pretraining complete")


if __name__ == "__main__":
    asyncio.run(main())

