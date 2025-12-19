"""Server service: FastAPI + WebSocket для UI и API."""

import asyncio
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Set

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from btc_oracle.core.config import Config, load_config
from btc_oracle.core.log import get_logger, setup_logging
from btc_oracle.core.types import Candle as CandleType
from btc_oracle.core.timeframes import candle_close_time, timeframe_to_minutes
from btc_oracle.data.bybit_spot import BybitSpotClient
from btc_oracle.db import AsyncSessionLocal, Candle, FeatureSnapshot, ModelPrediction, Prediction, init_db, get_db
from btc_oracle.patterns.store import PatternStore
from sqlalchemy import select, func, and_, case

logger = get_logger(__name__)

app = FastAPI(title="Titan Oracle API", version="0.1.0")

# Настройка шаблонов и статики
# NOTE: `server/main.py` находится в `btc_oracle/services/server/`, а UI-шаблоны — в `btc_oracle/api/`.
# Поэтому поднимаемся к пакету `btc_oracle/` и строим путь оттуда.
_pkg_dir = Path(__file__).resolve().parents[2]  # .../src/btc_oracle
base_dir = _pkg_dir / "api"
templates = Jinja2Templates(directory=str(base_dir / "templates"))
if (base_dir / "static").exists():
    app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")

# Глобальное состояние
_config: Optional[Config] = None
_pattern_store: Optional[PatternStore] = None
_ws_connections: Set[WebSocket] = set()
_last_candle_time: Optional[datetime] = None
_last_prediction_time: Optional[datetime] = None
_last_price: Optional[float] = None
_last_price_time: Optional[datetime] = None
_last_partial_candle: Optional[CandleType] = None
_last_partial_candle_updated: Optional[datetime] = None
_last_partial_candle_close: Optional[datetime] = None

# region agent log helper
_DBG_STATE: set[str] = set()


def _agent_dbg(hypothesisId: str, location: str, message: str, data: dict, *, runId: str = "run1"):
    try:
        payload = {
            "sessionId": "debug-session",
            "runId": runId,
            "hypothesisId": hypothesisId,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(r"c:\Projects\Titan\.cursor\debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


# endregion


def _prediction_to_ui_dict(pred: Prediction) -> dict:
    """Сериализация Prediction в формат, который ожидает dashboard (index.html)."""
    return {
        "ts": _format_ts(pred.time),
        "label": pred.label,
        "reason_code": pred.reason_code,
        "p_up": float(pred.p_up),
        "p_down": float(pred.p_down),
        "p_flat": float(pred.p_flat),
        "flat_score": float(pred.flat_score),
        "uncertainty_score": float(pred.uncertainty_score),
        "consensus": float(pred.consensus),
        "latency_ms": float(pred.latency_ms),
        "truth_label": pred.truth_label,
        "reward": float(pred.reward) if pred.reward is not None else None,
        "matured_at": _format_ts(pred.matured_at) if pred.matured_at else None,
        # `memory` опционально — dashboard корректно обработает отсутствие ключа.
    }

async def _get_close_price(
    session,
    symbol: str,
    timeframe: str,
    close_ts: datetime,
    tf_minutes: int,
) -> Optional[float]:
    """Получить цену закрытия свечи по времени закрытия."""
    if close_ts.tzinfo is None:
        close_ts = close_ts.replace(tzinfo=timezone.utc)
    exact_query = select(Candle.close).where(
        Candle.symbol == symbol,
        Candle.timeframe == timeframe,
        Candle.confirmed == 1,
        Candle.time == close_ts,
    )
    exact_result = await session.execute(exact_query)
    price = exact_result.scalar_one_or_none()
    if price is not None:
        return float(price)

    tolerance = timedelta(minutes=max(1, tf_minutes))
    fallback_query = (
        select(Candle.close)
        .where(
            Candle.symbol == symbol,
            Candle.timeframe == timeframe,
            Candle.confirmed == 1,
            Candle.time >= close_ts - tolerance,
            Candle.time <= close_ts + tolerance,
        )
        .order_by(Candle.time.asc())
        .limit(1)
    )
    fallback_result = await session.execute(fallback_query)
    price = fallback_result.scalar_one_or_none()
    return float(price) if price is not None else None

def _model_pred_to_ui_dict(pred: ModelPrediction) -> dict:
    """Сериализация ModelPrediction для UI."""
    return {
        "ts": _format_ts(pred.time),
        "model_id": pred.model_id,
        "label": pred.label,
        "reason_code": pred.reason_code,
        "p_up": float(pred.p_up),
        "p_down": float(pred.p_down),
        "p_flat": float(pred.p_flat),
        "u_dir": float(pred.u_dir),
        "u_mag": float(pred.u_mag),
        "consensus": float(pred.consensus),
        "disagreement": float(pred.disagreement),
        "weight": float(pred.weight),
    }


async def _fetch_latest_predictions(symbol: str, horizon_min: Optional[int]) -> dict:
    """Получить последние прогнозы в UI-формате."""
    if _config is None:
        horizons = [1, 5, 15, 60]
    else:
        horizons = list(_config.horizons)
    tf = _config.timeframe if _config else "1m"
    tf_minutes = timeframe_to_minutes(tf)

    async def _latest_aligned(session, horizon: int) -> Optional[Prediction]:
        query = (
            select(Prediction)
            .where(Prediction.symbol == symbol, Prediction.horizon_minutes == int(horizon))
            .order_by(Prediction.time.desc())
            .limit(50)
        )
        result = await session.execute(query)
        preds = result.scalars().all()
        for pred in preds:
            if _is_aligned(pred.time, int(horizon)):
                return pred
        return None

    async for session in get_db():
        if horizon_min is not None:
            pred = await _latest_aligned(session, horizon_min)
            if pred is None:
                raise HTTPException(status_code=404, detail="Prediction not found")
            price_pred = await _get_close_price(session, symbol, tf, pred.time, tf_minutes)
            price_fact = await _get_close_price(
                session,
                symbol,
                tf,
                pred.time + timedelta(minutes=int(horizon_min)),
                tf_minutes,
            )
            payload = _prediction_to_ui_dict(pred)
            payload["price_pred"] = price_pred
            payload["price_fact"] = price_fact
            return payload

        out: dict[str, dict] = {}
        for h in horizons:
            pred = await _latest_aligned(session, int(h))
            if pred is not None:
                price_pred = await _get_close_price(session, symbol, tf, pred.time, tf_minutes)
                price_fact = await _get_close_price(
                    session,
                    symbol,
                    tf,
                    pred.time + timedelta(minutes=int(h)),
                    tf_minutes,
                )
                payload = _prediction_to_ui_dict(pred)
                payload["price_pred"] = price_pred
                payload["price_fact"] = price_fact
                out[f"{int(h)}min"] = payload

        # region agent log
        if not out and "server_latest_empty" not in _DBG_STATE:
            _DBG_STATE.add("server_latest_empty")
            _agent_dbg(
                "H1",
                "services/server/main.py:_fetch_latest_predictions",
                "latest_predictions_empty",
                {"symbol": symbol, "horizons": horizons},
            )
        if out and "server_latest_nonempty" not in _DBG_STATE:
            _DBG_STATE.add("server_latest_nonempty")
            _agent_dbg(
                "H1",
                "services/server/main.py:_fetch_latest_predictions",
                "latest_predictions_nonempty",
                {"symbol": symbol, "keys": sorted(out.keys())[:10]},
            )
        # endregion
        return out


async def _fetch_latest_model_predictions(symbol: str, horizon_min: Optional[int]) -> dict:
    """Получить последние прогнозы отдельных моделей в UI-формате."""
    if _config is None:
        horizons = [1, 5, 15, 60]
    else:
        horizons = list(_config.horizons)

    async def _latest_model_batch(session, horizon: int) -> tuple[Optional[datetime], list[ModelPrediction]]:
        query = (
            select(ModelPrediction)
            .where(ModelPrediction.symbol == symbol, ModelPrediction.horizon_minutes == int(horizon))
            .order_by(ModelPrediction.time.desc())
            .limit(200)
        )
        result = await session.execute(query)
        rows = result.scalars().all()
        latest_time = None
        for row in rows:
            if _is_aligned(row.time, int(horizon)):
                latest_time = row.time
                break
        if latest_time is None:
            return None, []
        batch = [row for row in rows if row.time == latest_time]
        if not batch:
            exact = await session.execute(
                select(ModelPrediction).where(
                    ModelPrediction.symbol == symbol,
                    ModelPrediction.horizon_minutes == int(horizon),
                    ModelPrediction.time == latest_time,
                )
            )
            batch = exact.scalars().all()
        return latest_time, batch

    async for session in get_db():
        if horizon_min is not None:
            latest_time, batch = await _latest_model_batch(session, horizon_min)
            if not batch or latest_time is None:
                raise HTTPException(status_code=404, detail="Model predictions not found")
            models = sorted(batch, key=lambda r: r.model_id)
            return {
                "ts": _format_ts(latest_time),
                "horizon_min": int(horizon_min),
                "models": [_model_pred_to_ui_dict(row) for row in models],
            }

        out: dict[str, dict] = {}
        for h in horizons:
            latest_time, batch = await _latest_model_batch(session, int(h))
            if not batch or latest_time is None:
                continue
            models = sorted(batch, key=lambda r: r.model_id)
            out[f"{int(h)}min"] = {
                "ts": _format_ts(latest_time),
                "horizon_min": int(h),
                "models": [_model_pred_to_ui_dict(row) for row in models],
            }
        return out


async def _fetch_latest_inputs(symbol: str, timeframe: Optional[str]) -> dict:
    """Получить последний снимок входов и окно свечей."""
    tf = timeframe or (_config.timeframe if _config else '1m')
    tf_minutes = timeframe_to_minutes(tf)

    async for session in get_db():
        query = (
            select(FeatureSnapshot)
            .where(FeatureSnapshot.symbol == symbol, FeatureSnapshot.timeframe == tf)
            .order_by(FeatureSnapshot.time.desc())
            .limit(1)
        )
        result = await session.execute(query)
        snapshot = result.scalars().first()
        if snapshot is None:
            raise HTTPException(status_code=404, detail='Feature snapshot not found')

        candles_query = (
            select(Candle)
            .where(
                Candle.symbol == symbol,
                Candle.timeframe == tf,
                Candle.confirmed == 1,
                Candle.time <= snapshot.time,
            )
            .order_by(Candle.time.desc())
            .limit(snapshot.window_size)
        )
        candles_result = await session.execute(candles_query)
        candle_rows = list(reversed(candles_result.scalars().all()))

        names = snapshot.feature_names or []
        vector = snapshot.feature_vector or []
        features = []
        if len(names) == len(vector):
            features = [
                {
                    'name': name,
                    'value': float(value),
                }
                for name, value in zip(names, vector)
            ]
        else:
            features = [
                {
                    'name': f'f_{i}',
                    'value': float(value),
                }
                for i, value in enumerate(vector)
            ]

        candles = [
            {
                'ts': _format_ts(c.time),
                'close_ts': _format_ts(candle_close_time(c.time, tf_minutes)),
                'open': float(c.open),
                'high': float(c.high),
                'low': float(c.low),
                'close': float(c.close),
                'volume': float(c.volume),
            }
            for c in candle_rows
        ]

        window_start = None
        window_end = None
        if candle_rows:
            window_start = _format_ts(candle_close_time(candle_rows[0].time, tf_minutes))
            window_end = _format_ts(candle_close_time(candle_rows[-1].time, tf_minutes))

        market = {}
        if _last_price is not None and _last_price_time is not None:
            market = {
                'price': float(_last_price),
                'ts': _format_ts(_last_price_time),
            }

        live_candle = None
        if _last_partial_candle is not None and _last_partial_candle_updated is not None:
            live_candle = {
                'ts': _format_ts(_last_partial_candle.ts),
                'close_ts': _format_ts(_last_partial_candle_close) if _last_partial_candle_close else None,
                'open': float(_last_partial_candle.open),
                'high': float(_last_partial_candle.high),
                'low': float(_last_partial_candle.low),
                'close': float(_last_partial_candle.close),
                'volume': float(_last_partial_candle.volume),
                'age_sec': (datetime.now(timezone.utc) - _last_partial_candle_updated).total_seconds(),
            }

        return {
            'ts': _format_ts(snapshot.time),
            'symbol': snapshot.symbol,
            'timeframe': snapshot.timeframe,
            'timeframe_minutes': tf_minutes,
            'window_size': snapshot.window_size,
            'feature_dim': snapshot.feature_dim,
            'feature_meta': snapshot.feature_meta or {},
            'window_start_ts': window_start,
            'window_end_ts': window_end,
            'features': features,
            'candles': candles,
            'market': market,
            'live_candle': live_candle,
        }


@app.on_event("startup")
async def startup():
    """Инициализация при старте."""
    global _config
    global _pattern_store
    
    config_path = Path(__file__).parent.parent.parent.parent.parent / "config" / "default.yaml"
    _config = load_config(config_path)
    
    # Инициализация БД
    await init_db()
    
    setup_logging(
        level=_config.logging.level,
        structured=_config.logging.structured,
        log_file=Path(_config.logging.log_file) if _config.logging.log_file else None,
    )

    _pattern_store = PatternStore(_config.patterns)

    # region agent log
    _agent_dbg(
        "H2",
        "services/server/main.py:startup",
        "startup_loaded_config",
        {
            "symbol": _config.symbol,
            "timeframe": _config.timeframe,
            "horizons": list(_config.horizons),
        },
    )
    # endregion
    
    # Запускаем фоновые задачи для WebSocket
    asyncio.create_task(_monitor_candles())
    asyncio.create_task(_monitor_predictions())
    asyncio.create_task(_monitor_partial_candles())
    asyncio.create_task(_monitor_price())
    
    logger.info("API server started")


@app.on_event("shutdown")
async def shutdown():
    """Завершение работы сервиса."""
    if _pattern_store is not None:
        _pattern_store.close()


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Главная страница дашборда."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/models", response_class=HTMLResponse)
async def models_root(request: Request):
    """Страница прогнозов по моделям."""
    return templates.TemplateResponse("models.html", {"request": request})

@app.get("/inputs", response_class=HTMLResponse)
async def inputs_root(request: Request):
    """Страница входных данных модели."""
    return templates.TemplateResponse("inputs.html", {"request": request})

@app.get("/patterns", response_class=HTMLResponse)
async def patterns_root(request: Request):
    """Страница мониторинга паттернов."""
    return templates.TemplateResponse("patterns.html", {"request": request})




@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/predictions/latest")
async def ui_latest_predictions(
    symbol: str = "BTCUSDT",
    horizon_min: Optional[int] = None,
):
    """Endpoint для dashboard (совместимость с шаблоном `index.html`)."""
    return await _fetch_latest_predictions(symbol=symbol, horizon_min=horizon_min)


@app.get("/api/predictions/latest")
async def get_latest_predictions(
    symbol: str = "BTCUSDT",
    horizon_min: Optional[int] = None,
):
    """Получить последние прогнозы."""
    return await _fetch_latest_predictions(symbol=symbol, horizon_min=horizon_min)

@app.get("/api/models/latest")
async def get_latest_model_predictions(
    symbol: str = "BTCUSDT",
    horizon_min: Optional[int] = None,
):
    """Получить последние прогнозы отдельных моделей."""
    return await _fetch_latest_model_predictions(symbol=symbol, horizon_min=horizon_min)

@app.get("/api/inputs/latest")
async def get_latest_inputs(
    symbol: str = "BTCUSDT",
    timeframe: Optional[str] = None,
):
    """Получить последний снимок входных данных модели."""
    return await _fetch_latest_inputs(symbol=symbol, timeframe=timeframe)




@app.get("/api/config")
async def get_config():
    """Получить текущую конфигурацию (horizons/timeframe)."""
    if _config is None:
        num_models = 3
        return {
            "symbol": "BTCUSDT",
            "timeframe": "1m",
            "horizons": [1, 5, 10, 15, 30, 60],
            "num_models": num_models,
            "model_ids": [f"model_{i}" for i in range(num_models)],
        }
    num_models = _config.ensemble.num_models
    return {
        "symbol": _config.symbol,
        "timeframe": _config.timeframe,
        "horizons": list(_config.horizons),
        "num_models": num_models,
        "model_ids": [f"model_{i}" for i in range(num_models)],
    }


def _format_ts(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)
    return ts.isoformat().replace("+00:00", "Z")

def _parse_pattern_id(pattern_id: str) -> int:
    text = pattern_id.strip().lower()
    if text.startswith("0x"):
        return int(text, 16)
    if any(ch in "abcdef" for ch in text):
        return int(text, 16)
    return int(text)


def _is_aligned(ts: datetime, horizon_min: int) -> bool:
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc)
    if ts.second != 0 or ts.microsecond != 0:
        return False
    if horizon_min <= 1:
        return True
    if horizon_min >= 60:
        return ts.minute == 0
    return ts.minute % int(horizon_min) == 0


@app.get("/api/metrics")
async def get_metrics():
    """Получить метрики системы."""
    async for session in get_db():
        symbol = _config.symbol if _config else "BTCUSDT"
        # Подсчитываем метрики из predictions
        total_query = select(func.count()).select_from(Prediction).where(
            Prediction.symbol == symbol,
            Prediction.truth_label.isnot(None),
        )
        total_result = await session.execute(total_query)
        total = total_result.scalar() or 0
        
        correct_query = select(func.count()).select_from(Prediction).where(
            Prediction.symbol == symbol,
            Prediction.truth_label.isnot(None),
            Prediction.label == Prediction.truth_label,
        )
        correct_result = await session.execute(correct_query)
        correct = correct_result.scalar() or 0
        
        accuracy = (correct / total * 100) if total > 0 else 0.0

        non_uncertain_query = select(
            func.count()
        ).select_from(Prediction).where(
            Prediction.symbol == symbol,
            Prediction.truth_label.isnot(None),
            Prediction.label != "UNCERTAIN",
        )
        non_uncertain_result = await session.execute(non_uncertain_query)
        non_uncertain = non_uncertain_result.scalar() or 0
        coverage = (non_uncertain / total * 100) if total > 0 else 0.0

        dir_condition = and_(
            Prediction.label.in_(["UP", "DOWN"]),
            Prediction.truth_label.in_(["UP", "DOWN"]),
        )
        dir_total_query = select(func.count()).select_from(Prediction).where(
            Prediction.symbol == symbol,
            Prediction.truth_label.isnot(None),
            dir_condition,
        )
        dir_total_result = await session.execute(dir_total_query)
        dir_total = dir_total_result.scalar() or 0

        dir_correct_query = select(func.count()).select_from(Prediction).where(
            Prediction.symbol == symbol,
            Prediction.truth_label.isnot(None),
            dir_condition,
            Prediction.label == Prediction.truth_label,
        )
        dir_correct_result = await session.execute(dir_correct_query)
        dir_correct = dir_correct_result.scalar() or 0
        dir_accuracy = (dir_correct / dir_total * 100) if dir_total > 0 else 0.0

        conf_query = select(func.avg(func.greatest(
            Prediction.p_up,
            Prediction.p_down,
            Prediction.p_flat,
        ))).where(
            Prediction.symbol == symbol,
            Prediction.truth_label.isnot(None),
        )
        conf_result = await session.execute(conf_query)
        avg_confidence = float(conf_result.scalar() or 0.0)

        unc_query = select(func.avg(Prediction.uncertainty_score)).where(
            Prediction.symbol == symbol,
            Prediction.truth_label.isnot(None),
        )
        unc_result = await session.execute(unc_query)
        avg_uncertainty = float(unc_result.scalar() or 0.0)

        per_horizon_query = (
            select(
                Prediction.horizon_minutes.label("horizon"),
                func.count().label("total"),
                func.sum(case((Prediction.label == Prediction.truth_label, 1), else_=0)).label("correct"),
                func.sum(case((Prediction.label != "UNCERTAIN", 1), else_=0)).label("non_uncertain"),
                func.sum(case((dir_condition, 1), else_=0)).label("dir_total"),
                func.sum(
                    case((and_(dir_condition, Prediction.label == Prediction.truth_label), 1), else_=0)
                ).label("dir_correct"),
                func.avg(func.greatest(
                    Prediction.p_up,
                    Prediction.p_down,
                    Prediction.p_flat,
                )).label("avg_confidence"),
                func.avg(Prediction.uncertainty_score).label("avg_uncertainty"),
            )
            .where(
                Prediction.symbol == symbol,
                Prediction.truth_label.isnot(None),
            )
            .group_by(Prediction.horizon_minutes)
            .order_by(Prediction.horizon_minutes.asc())
        )
        per_horizon_result = await session.execute(per_horizon_query)
        per_horizon_rows = per_horizon_result.all()
        per_horizon = []
        for row in per_horizon_rows:
            total_h = row.total or 0
            dir_total_h = row.dir_total or 0
            per_horizon.append(
                {
                    "horizon": int(row.horizon),
                    "total": int(total_h),
                    "correct": int(row.correct or 0),
                    "accuracy": (float(row.correct or 0) / total_h * 100) if total_h > 0 else 0.0,
                    "coverage": (float(row.non_uncertain or 0) / total_h * 100) if total_h > 0 else 0.0,
                    "dir_accuracy": (float(row.dir_correct or 0) / dir_total_h * 100) if dir_total_h > 0 else 0.0,
                    "avg_confidence": float(row.avg_confidence or 0.0),
                    "avg_uncertainty": float(row.avg_uncertainty or 0.0),
                }
            )
        
        return {
            "status": "ok",
            "metrics": {
                "total_predictions": total,
                "correct_predictions": correct,
                "accuracy": accuracy,
                "coverage": coverage,
                "directional_accuracy": dir_accuracy,
                "avg_confidence": avg_confidence,
                "avg_uncertainty": avg_uncertainty,
                "logloss": 0.0,
                "brier_score": 0.0,
            },
            "per_horizon": per_horizon,
        }


@app.get("/api/patterns/top")
async def get_top_patterns(limit: int = 10):
    """Получить топ паттернов по количеству наблюдений."""
    if _pattern_store is None:
        return {"patterns": [], "limit": limit}
    patterns = _pattern_store.get_top_patterns(limit=limit, sort_by="n")
    return {
        "patterns": patterns,
        "limit": limit,
    }


@app.get("/api/patterns/summary")
async def get_patterns_summary():
    """Получить сводку по паттернам."""
    if _pattern_store is None:
        return {"summary": {}}
    return {"summary": _pattern_store.get_patterns_summary()}


@app.get("/api/patterns/list")
async def get_patterns_list(
    limit: int = 200,
    offset: int = 0,
    sort_by: str = "n",
    order: str = "desc",
    status: Optional[str] = None,
    timeframe: Optional[str] = None,
    horizon: Optional[int] = None,
    contrarian: Optional[bool] = None,
    search: Optional[str] = None,
):
    """Получить страницу списка паттернов."""
    if _pattern_store is None:
        return {"patterns": [], "total": 0, "offset": offset, "limit": limit}
    return _pattern_store.list_patterns(
        offset=offset,
        limit=limit,
        sort_by=sort_by,
        order=order,
        status=status,
        timeframe=timeframe,
        horizon=horizon,
        contrarian=contrarian,
        search=search,
    )


@app.get("/api/patterns/detail")
async def get_pattern_detail(
    pattern_id: str,
    ring_limit: int = 200,
    extremes_limit: int = 20,
    include_vectors: bool = False,
):
    """Получить детали паттерна."""
    if _pattern_store is None:
        raise HTTPException(status_code=404, detail="Pattern store not available")
    try:
        pid = _parse_pattern_id(pattern_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid pattern_id") from exc
    detail = _pattern_store.get_pattern_detail(
        pid,
        ring_limit=ring_limit,
        extremes_limit=extremes_limit,
        include_vectors=include_vectors,
    )
    if detail is None:
        raise HTTPException(status_code=404, detail="Pattern not found")
    return detail


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint для live updates."""
    await websocket.accept()
    _ws_connections.add(websocket)
    logger.info(f"WebSocket client connected. Total: {len(_ws_connections)}")
    
    try:
        # Отправляем приветственное сообщение
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to Titan Oracle",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        # Держим соединение открытым
        while True:
            # Ожидаем сообщения от клиента (ping/pong или подписки)
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                except json.JSONDecodeError:
                    pass
            except asyncio.TimeoutError:
                # Отправляем ping для keepalive
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        _ws_connections.discard(websocket)
        logger.info(f"WebSocket client removed. Total: {len(_ws_connections)}")


async def _broadcast(event_type: str, data: dict):
    """Отправить событие всем подключенным WebSocket клиентам."""
    if not _ws_connections:
        return
    
    message = {
        "type": event_type,
        "data": data,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    disconnected = set()
    for ws in _ws_connections:
        try:
            await ws.send_json(message)
        except Exception as e:
            logger.debug(f"Failed to send to WebSocket: {e}")
            disconnected.add(ws)
    
    # Удаляем отключённые соединения
    for ws in disconnected:
        _ws_connections.discard(ws)


async def _monitor_candles():
    """Мониторинг новых свечей из БД."""
    global _last_candle_time
    
    if not _config:
        return
    
    while True:
        try:
            async with AsyncSessionLocal() as session:
                query = select(Candle).where(
                    Candle.symbol == _config.symbol,
                    Candle.timeframe == _config.timeframe,
                    Candle.confirmed == 1,
                )
                
                if _last_candle_time:
                    query = query.where(Candle.time > _last_candle_time)
                
                query = query.order_by(Candle.time.desc()).limit(1)
                
                result = await session.execute(query)
                candle = result.scalar_one_or_none()
                
                if candle:
                    _last_candle_time = candle.time
                    
                    await _broadcast("candle", {
                        "time": candle.time.isoformat(),
                        "symbol": candle.symbol,
                        "timeframe": candle.timeframe,
                        "open": candle.open,
                        "high": candle.high,
                        "low": candle.low,
                        "close": candle.close,
                        "volume": candle.volume,
                        "confirmed": bool(candle.confirmed),
                    })
        
        except Exception as e:
            logger.error(f"Error monitoring candles: {e}", exc_info=True)
        
        await asyncio.sleep(1.0)  # Проверяем каждую секунду


async def _monitor_predictions():
    """Мониторинг новых прогнозов из БД."""
    global _last_prediction_time
    
    if not _config:
        return
    
    while True:
        try:
            async with AsyncSessionLocal() as session:
                query = select(Prediction).where(
                    Prediction.symbol == _config.symbol,
                )
                
                if _last_prediction_time:
                    query = query.where(Prediction.time > _last_prediction_time)
                
                query = query.order_by(Prediction.time.desc()).limit(10)
                
                result = await session.execute(query)
                predictions = result.scalars().all()
                
                for pred in reversed(predictions):  # Старые → новые
                    if not _last_prediction_time or pred.time > _last_prediction_time:
                        _last_prediction_time = pred.time
                        
                        await _broadcast("prediction", {
                            "time": pred.time.isoformat(),
                            "symbol": pred.symbol,
                            "horizon_minutes": pred.horizon_minutes,
                            "label": pred.label,
                            "reason_code": pred.reason_code,
                            "p_up": pred.p_up,
                            "p_down": pred.p_down,
                            "p_flat": pred.p_flat,
                            "consensus": pred.consensus,
                            "uncertainty_score": pred.uncertainty_score,
                        })
                
                # Проверяем созревшие прогнозы (results)
                if predictions:
                    for pred in predictions:
                        if pred.matured_at and pred.truth_label:
                            await _broadcast("result", {
                                "time": pred.time.isoformat(),
                                "symbol": pred.symbol,
                                "horizon_minutes": pred.horizon_minutes,
                                "predicted": pred.label,
                                "actual": pred.truth_label,
                                "magnitude": pred.truth_magnitude,
                                "reward": pred.reward,
                            })
        
        except Exception as e:
            logger.error(f"Error monitoring predictions: {e}", exc_info=True)
        
        await asyncio.sleep(2.0)  # Проверяем каждые 2 секунды


async def _monitor_partial_candles():
    """Monitor partial (unconfirmed) candle updates for UI."""
    global _last_partial_candle
    global _last_partial_candle_updated
    global _last_partial_candle_close

    if not _config:
        return

    interval = _config.timeframe.replace("m", "") if _config.timeframe.endswith("m") else _config.timeframe
    tf_minutes = timeframe_to_minutes(_config.timeframe)

    while True:
        try:
            async with BybitSpotClient(_config.bybit) as client:
                async def on_candle(candle: CandleType):
                    if candle.confirmed:
                        return
                    _last_partial_candle = candle
                    _last_partial_candle_updated = datetime.now(timezone.utc)
                    _last_partial_candle_close = candle_close_time(candle.ts, tf_minutes)

                await client.subscribe_klines(
                    symbol=_config.symbol,
                    interval=interval,
                    callback=on_candle,
                    only_confirmed=False,
                )
        except Exception as e:
            logger.error(f"Error monitoring partial candles: {e}", exc_info=True)
            await asyncio.sleep(5.0)


async def _monitor_price():
    """Мониторинг real-time цены через Bybit WebSocket."""
    if not _config:
        return
    
    while True:
        try:
            async with BybitSpotClient(_config.bybit) as client:
                async for price in client.stream_ticker(_config.symbol):
                    global _last_price
                    global _last_price_time
                    _last_price = float(price)
                    _last_price_time = datetime.now(timezone.utc)
                    await _broadcast("price", {
                        "symbol": _config.symbol,
                        "price": price,
                    })
        except Exception as e:
            logger.error(f"Error monitoring price: {e}", exc_info=True)
            await asyncio.sleep(5.0)  # Пауза при ошибке


def main():
    """Точка входа для запуска API сервера."""
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
