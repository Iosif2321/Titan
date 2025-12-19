"""Минимальные smoke тесты для сервисов."""

import pytest
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.mark.skip(reason="Requires running services")
def test_server_health():
    """Тест health check сервера."""
    import urllib.request
    
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as resp:
            assert resp.status == 200
            data = resp.read()
            assert b"ok" in data.lower()
    except Exception:
        pytest.skip("Server not running")


@pytest.mark.skip(reason="Requires database")
def test_database_connection():
    """Тест подключения к БД."""
    import asyncio
    from btc_oracle.db import AsyncSessionLocal
    from sqlalchemy import text
    
    async def test():
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
            return True
    
    try:
        result = asyncio.run(test())
        assert result is True
    except Exception:
        pytest.skip("Database not available")


def test_cli_import():
    """Тест импорта CLI модуля."""
    from btc_oracle.cli import cli
    assert cli is not None


def test_services_import():
    """Тест импорта сервисов."""
    from btc_oracle.services.collector.main import main as collector_main
    from btc_oracle.services.inferencer.main import main as inferencer_main
    from btc_oracle.services.trainer.main import main as trainer_main
    from btc_oracle.services.server.main import app
    
    assert collector_main is not None
    assert inferencer_main is not None
    assert trainer_main is not None
    assert app is not None
