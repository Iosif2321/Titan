"""Настройка SQLAlchemy async сессий для TimescaleDB."""

import os
import sys
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from btc_oracle.core.log import get_logger

logger = get_logger(__name__)

# Загружаем переменные окружения
_ROOT = Path(__file__).resolve().parents[3]
_env_file = _ROOT / ".env.local"
if _env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(_env_file))

# Параметры подключения к БД
POSTGRES_USER = os.getenv("POSTGRES_USER", "titan")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "titan_password")
POSTGRES_DB = os.getenv("POSTGRES_DB", "titan_db")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "127.0.0.1")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5433")  # Порт 5433 по умолчанию

DATABASE_URL = f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Создаём async engine
engine_kwargs = {
    "echo": False,
    "future": True,
    "pool_pre_ping": True,
}
if os.getenv("PYTEST_RUNNING") or os.getenv("PYTEST_CURRENT_TEST") or "pytest" in sys.modules:
    engine_kwargs["poolclass"] = NullPool

engine = create_async_engine(DATABASE_URL, **engine_kwargs)

# Session factory
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db():
    """Dependency для FastAPI (async generator)."""
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    """
    Инициализация БД: создание таблиц и hypertable.
    
    Вызывается при старте приложения.
    """
    from sqlalchemy import text
    
    from btc_oracle.db.models import Base
    
    async with engine.begin() as conn:
        # Создаём все таблицы
        await conn.run_sync(Base.metadata.create_all)
        
        # Создаём hypertable для candles (TimescaleDB)
        try:
            await conn.execute(
                text("SELECT create_hypertable('candles', 'time', if_not_exists => TRUE);")
            )
            logger.info("Hypertable 'candles' created or already exists")
        except Exception as e:
            logger.warning(f"Could not create hypertable (may already exist): {e}")
        
        logger.info("Database initialized successfully")
