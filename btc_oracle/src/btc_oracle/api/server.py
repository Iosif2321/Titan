"""FastAPI сервер для прогнозов и метрик."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from btc_oracle.core.config import Config, load_config
from btc_oracle.core.log import get_logger, setup_logging
from btc_oracle.data.store import DataStore

logger = get_logger(__name__)

app = FastAPI(title="BTC Oracle API", version="0.1.0")

# Настройка шаблонов и статики
base_dir = Path(__file__).parent
templates = Jinja2Templates(directory=str(base_dir / "templates"))
app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")

# Глобальное состояние (в реальности лучше использовать dependency injection)
_config: Optional[Config] = None
_store: Optional[DataStore] = None
_latest_predictions: dict[tuple[str, int], dict] = {}  # (symbol, horizon) -> prediction


@app.on_event("startup")
async def startup():
    """Инициализация при старте."""
    global _config, _store
    
    config_path = Path("config/default.yaml")
    _config = load_config(config_path)
    
    _store = DataStore(Path(_config.storage.db_path))
    
    setup_logging(
        level=_config.logging.level,
        structured=_config.logging.structured,
        log_file=Path(_config.logging.log_file) if _config.logging.log_file else None,
    )
    
    logger.info("API server started")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Главная страница дашборда."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/predictions/latest")
async def get_latest_predictions(
    symbol: str = "BTCUSDT",
    horizon_min: Optional[int] = None,
):
    """Получить последние прогнозы."""
    if _store is None:
        raise HTTPException(status_code=500, detail="Store not initialized")
    
    if horizon_min:
        prediction = _store.get_latest_prediction(symbol, horizon_min)
        if prediction:
            return prediction
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    # Все горизонты
    predictions = {}
    for horizon in _config.horizons if _config else [1, 5, 15, 60]:
        pred = _store.get_latest_prediction(symbol, horizon)
        if pred:
            predictions[f"{horizon}min"] = pred
    
    return predictions


@app.get("/metrics")
async def get_metrics():
    """Получить метрики системы."""
    # В реальности здесь должны быть реальные метрики из MetricsCollector
    return {
        "status": "ok",
        "metrics": {
            "coverage": 0.0,
            "logloss": 0.0,
            "brier_score": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
        },
    }


@app.get("/patterns/top")
async def get_top_patterns(limit: int = 10):
    """Получить топ паттернов по количеству наблюдений."""
    # В реальности нужно реализовать запрос к pattern_stats
    return {
        "patterns": [],
        "limit": limit,
    }


def main():
    """Точка входа для запуска API сервера."""
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

