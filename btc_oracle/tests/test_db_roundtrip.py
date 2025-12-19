"""Smoke тесты для DB roundtrip (сохранение/чтение данных)."""

import pytest
import asyncio
from datetime import datetime, timezone
from pathlib import Path

from btc_oracle.core.types import Candle as CandleType, Decision, Label, ReasonCode
from btc_oracle.db import AsyncSessionLocal, Candle, Prediction, init_db
from sqlalchemy import select


@pytest.fixture(scope="module")
async def db_setup():
    """Инициализация БД для тестов."""
    try:
        await init_db()
    except Exception as exc:
        pytest.skip(f"DB not available: {exc}")
    yield
    # Cleanup можно добавить здесь


@pytest.mark.asyncio
async def test_candle_save_and_read(db_setup):
    """Тест сохранения и чтения свечи."""
    test_candle = CandleType(
        ts=datetime.now(timezone.utc),
        open=50000.0,
        high=51000.0,
        low=49000.0,
        close=50500.0,
        volume=100.5,
        confirmed=True,
    )
    
    async with AsyncSessionLocal() as session:
        # Сохраняем свечу
        db_candle = Candle(
            time=test_candle.ts,
            symbol="BTCUSDT",
            timeframe="1m",
            open=test_candle.open,
            high=test_candle.high,
            low=test_candle.low,
            close=test_candle.close,
            volume=test_candle.volume,
            confirmed=1 if test_candle.confirmed else 0,
        )
        session.add(db_candle)
        await session.commit()
        
        # Читаем свечу
        query = select(Candle).where(
            Candle.symbol == "BTCUSDT",
            Candle.timeframe == "1m",
            Candle.time == test_candle.ts,
        )
        result = await session.execute(query)
        retrieved = result.scalar_one_or_none()
        
        assert retrieved is not None
        assert retrieved.symbol == "BTCUSDT"
        assert retrieved.close == 50500.0
        assert retrieved.confirmed == 1


@pytest.mark.asyncio
async def test_prediction_save_and_read(db_setup):
    """Тест сохранения и чтения прогноза."""
    test_time = datetime.now(timezone.utc)
    
    async with AsyncSessionLocal() as session:
        # Сохраняем прогноз
        db_pred = Prediction(
            time=test_time,
            symbol="BTCUSDT",
            horizon_minutes=5,
            label="UP",
            reason_code="CONFIDENT",
            p_up=0.7,
            p_down=0.2,
            p_flat=0.1,
            flat_score=0.1,
            uncertainty_score=0.15,
            consensus=0.8,
            latency_ms=50.0,
            created_at=test_time,
        )
        session.add(db_pred)
        await session.commit()
        
        # Читаем прогноз
        query = select(Prediction).where(
            Prediction.symbol == "BTCUSDT",
            Prediction.horizon_minutes == 5,
            Prediction.time == test_time,
        )
        result = await session.execute(query)
        retrieved = result.scalar_one_or_none()
        
        assert retrieved is not None
        assert retrieved.label == "UP"
        assert retrieved.p_up == 0.7
        assert retrieved.consensus == 0.8


@pytest.mark.asyncio
async def test_prediction_maturation(db_setup):
    """Тест созревания прогноза (matured_at)."""
    test_time = datetime.now(timezone.utc)
    
    async with AsyncSessionLocal() as session:
        # Создаём прогноз
        db_pred = Prediction(
            time=test_time,
            symbol="BTCUSDT",
            horizon_minutes=5,
            label="UP",
            reason_code="CONFIDENT",
            p_up=0.7,
            p_down=0.2,
            p_flat=0.1,
            flat_score=0.1,
            uncertainty_score=0.15,
            consensus=0.8,
            latency_ms=50.0,
            created_at=test_time,
        )
        session.add(db_pred)
        await session.commit()
        
        # Отмечаем как созревший
        db_pred.truth_label = "UP"
        db_pred.truth_magnitude = 0.05
        db_pred.reward = 1.0
        db_pred.matured_at = datetime.now(timezone.utc)
        await session.commit()
        
        # Проверяем
        query = select(Prediction).where(
            Prediction.time == test_time,
            Prediction.symbol == "BTCUSDT",
        )
        result = await session.execute(query)
        retrieved = result.scalar_one()
        
        assert retrieved.matured_at is not None
        assert retrieved.truth_label == "UP"
        assert retrieved.reward == 1.0
