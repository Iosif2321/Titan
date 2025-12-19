"""Inferencer service: читает новые свечи, делает прогнозы, пишет predictions."""

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from btc_oracle.core.config import Config, load_config
from btc_oracle.core.log import get_logger, setup_logging
from btc_oracle.core.types import Candle as CandleType, Features, FusedForecast
from btc_oracle.core.timeframes import candle_close_time, timeframe_to_minutes
from btc_oracle.db import AsyncSessionLocal, Candle, FeatureSnapshot, ModelPrediction, Prediction, init_db
from btc_oracle.ensemble.aggregator import EnsembleAggregator
from btc_oracle.ensemble.weights import WeightManager
from btc_oracle.ensemble.wrapper import ModelWrapper
from btc_oracle.features.pipeline import FeaturePipeline
from btc_oracle.fusion.calibrator import TemperatureCalibrator
from btc_oracle.fusion.fuse import FusionEngine
from btc_oracle.fusion.policy import DecisionPolicy
from btc_oracle.models.lstm import LSTMModel
from btc_oracle.patterns.encoder import PatternEncoder
from btc_oracle.patterns.store import PatternStore
from sqlalchemy import select, and_
from sqlalchemy.dialects.postgresql import insert

logger = get_logger(__name__)

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


def _is_horizon_tick(ts: datetime, horizon_min: int) -> bool:
    if ts.second != 0 or ts.microsecond != 0:
        return False
    if horizon_min <= 1:
        return True
    minute = ts.minute
    if horizon_min >= 60:
        return minute == 0
    return minute % int(horizon_min) == 0


class InferencerService:
    """Сервис инференса для прогнозирования."""
    
    def __init__(self, config: Config):
        """Инициализация сервиса."""
        self.config = config
        
        # Инициализация компонентов
        self.pattern_store = PatternStore(config.patterns)
        self.pattern_encoder = PatternEncoder(config.patterns)
        self.feature_pipeline = FeaturePipeline(window_size=100)
        self.base_tf_min = timeframe_to_minutes(config.timeframe)
        
        # Модели ансамбля
        self.models = []
        self.model_wrappers = []
        self.weight_manager = WeightManager(
            num_models=config.ensemble.num_models,
            ema_alpha=config.ensemble.ema_alpha,
            uncertainty_penalty_lambda=config.ensemble.uncertainty_penalty_lambda,
        )
        
        self._init_models()
        
        # Ансамбль
        self.aggregator = EnsembleAggregator(
            models=self.model_wrappers,
            weight_manager=self.weight_manager,
            parallel=True,
        )
        
        # Fusion и Policy
        self.fusion_engine = FusionEngine(
            memory_credibility_min=config.fusion.memory_credibility_min,
            neural_confidence_weight=config.fusion.neural_confidence_weight,
        )
        self.calibrator = TemperatureCalibrator(
            temperature=config.calibration.temperature_initial,
        )
        self.policy = DecisionPolicy(config)
        
        # Трекер последней обработанной свечи
        self.last_processed_ts: datetime | None = None
    
    def _init_models(self):
        """Инициализировать модели ансамбля."""
        feature_dim = len(self.feature_pipeline.feature_names())
        
        for i in range(self.config.ensemble.num_models):
            model = LSTMModel(
                input_dim=feature_dim,
                hidden_dim=64,
                num_layers=2,
                dropout=0.2,
            )
            wrapper = ModelWrapper(
                model=model,
                model_id=f"model_{i}",
                mc_passes=self.config.ensemble.mc_passes,
            )
            self.models.append(model)
            self.model_wrappers.append(wrapper)
        
        logger.info(f"Initialized {len(self.models)} models")
    
    async def process_new_candles(self):
        """Обработать новые свечи из БД."""
        async with AsyncSessionLocal() as session:
            # Получаем новые свечи (после last_processed_ts)
            query = select(Candle).where(
                Candle.symbol == self.config.symbol,
                Candle.timeframe == self.config.timeframe,
                Candle.confirmed == 1,
            )
            
            if self.last_processed_ts:
                query = query.where(Candle.time > self.last_processed_ts)
            
            query = query.order_by(Candle.time.asc()).limit(10)
            
            result = await session.execute(query)
            new_candles = result.scalars().all()
            
            if not new_candles:
                return
            
            for db_candle in new_candles:
                # Преобразуем в CandleType
                candle = CandleType(
                    ts=db_candle.time,
                    open=db_candle.open,
                    high=db_candle.high,
                    low=db_candle.low,
                    close=db_candle.close,
                    volume=db_candle.volume,
                    confirmed=True,
                )
                
                # Обрабатываем свечу
                await self._process_candle(session, candle)
                
                # Обновляем last_processed_ts
                self.last_processed_ts = db_candle.time
    
    async def _process_candle(self, session, candle: CandleType):
        """Обработать одну свечу и сделать прогноз."""
        start_time = time.time()
        candle_open_ts = candle.ts.replace(second=0, microsecond=0)
        candle_close_ts = candle_close_time(candle_open_ts, self.base_tf_min)
        
        # Получаем окно для признаков
        # TODO: использовать DataRepository когда он будет работать с async БД
        window_candles = await self._get_feature_window(session, candle.ts, window_size=100)
        
        if len(window_candles) < 50:
            logger.debug("Not enough data for prediction")
            return
        
        # Строим признаки
        features = self.feature_pipeline.build_features(window_candles)
        features.ts = candle_close_ts
        features.timeframe = self.config.timeframe
        await self._save_feature_snapshot(session, candle_close_ts, features)

        
        # Получаем мнения Neural и Memory
        neural_opinion, model_opinions, model_weights = await self.aggregator.aggregate_with_details(
            features,
            enable_dropout=True,
        )

        horizons = sorted(self.config.horizons)
        for horizon in horizons:
            if not _is_horizon_tick(candle_close_ts, horizon):
                continue

            pattern_key = self.pattern_encoder.encode(
                features,
                horizon=horizon,
                timeframe=self.config.timeframe,
                context=neural_opinion,
            )

            # region agent log
            if "inferencer_first_pattern_key" not in _DBG_STATE:
                _DBG_STATE.add("inferencer_first_pattern_key")
                _agent_dbg(
                    "H3",
                    "services/inferencer/main.py:_process_candle",
                    "pattern_key_computed",
                    {
                "ts": candle_close_ts.isoformat(),
                        "timeframe": self.config.timeframe,
                        "horizon": int(horizon),
                        "pattern_id": int(pattern_key.pattern_id),
                    },
                )
            # endregion

            memory_opinion = self.pattern_store.get_opinion(
                pattern_key,
                min_samples=self.config.patterns.min_pattern_samples,
                features=features,
            )

            # region agent log
            if memory_opinion is None and "inferencer_memory_none" not in _DBG_STATE:
                _DBG_STATE.add("inferencer_memory_none")
                _agent_dbg(
                    "H3",
                    "services/inferencer/main.py:_process_candle",
                    "memory_opinion_none",
                    {"pattern_id": int(pattern_key.pattern_id)},
                )
            if memory_opinion is not None and "inferencer_memory_some" not in _DBG_STATE:
                _DBG_STATE.add("inferencer_memory_some")
                _agent_dbg(
                    "H3",
                    "services/inferencer/main.py:_process_candle",
                    "memory_opinion_present",
                    {
                        "pattern_id": int(pattern_key.pattern_id),
                        "n": int(memory_opinion.n),
                        "credibility": float(memory_opinion.credibility),
                    },
                )
            # endregion
            
            # Fusion
            fused = self.fusion_engine.fuse(neural_opinion, memory_opinion)
            
            # Calibration
            if self.config.calibration.enabled:
                fused = self.calibrator.calibrate(fused)
            
            # Policy
            latency_ms = (time.time() - start_time) * 1000
            decision = self.policy.decide(
                forecast=fused,
                symbol=self.config.symbol,
                horizon_min=horizon,
                ts=candle_close_ts,
                latency_ms=latency_ms,
            )
            await self._save_model_predictions(
                session,
                candle_close_ts,
                horizon,
                model_opinions,
                model_weights,
            )

            
            # Сохраняем прогноз в БД
            await self._save_prediction(session, decision)
            
            logger.info(
                "Prediction made",
                label=decision.label.value,
                reason=decision.reason_code.value,
                latency_ms=latency_ms,
                horizon=horizon,
            )
    
    async def _get_feature_window(
        self,
        session,
        end_ts: datetime,
        window_size: int,
    ) -> list[CandleType]:
        """Получить окно свечей для признаков."""
        query = select(Candle).where(
            Candle.symbol == self.config.symbol,
            Candle.timeframe == self.config.timeframe,
            Candle.time <= end_ts,
            Candle.confirmed == 1,
        ).order_by(Candle.time.desc()).limit(window_size)
        
        result = await session.execute(query)
        db_candles = result.scalars().all()
        
        # Преобразуем и разворачиваем (старые → новые)
        candles = []
        for db_candle in reversed(db_candles):
            candle = CandleType(
                ts=db_candle.time,
                open=db_candle.open,
                high=db_candle.high,
                low=db_candle.low,
                close=db_candle.close,
                volume=db_candle.volume,
                confirmed=True,
            )
            candles.append(candle)
        
        return candles
    
    async def _save_feature_snapshot(self, session, ts: datetime, features: Features) -> None:
        """Сохранить снимок входных признаков."""
        names = self.feature_pipeline.feature_names()
        vector = features.vector.tolist()
        if len(names) != len(vector):
            names = [f"f_{i}" for i in range(len(vector))]

        stmt = insert(FeatureSnapshot).values(
            time=ts,
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            window_size=self.feature_pipeline.window_size,
            feature_dim=len(vector),
            feature_names=names,
            feature_vector=vector,
            feature_meta=features.meta,
            created_at=datetime.now(timezone.utc),
        ).on_conflict_do_update(
            index_elements=["time", "symbol", "timeframe"],
            set_={
                "window_size": insert(FeatureSnapshot).excluded.window_size,
                "feature_dim": insert(FeatureSnapshot).excluded.feature_dim,
                "feature_names": insert(FeatureSnapshot).excluded.feature_names,
                "feature_vector": insert(FeatureSnapshot).excluded.feature_vector,
                "feature_meta": insert(FeatureSnapshot).excluded.feature_meta,
                "created_at": insert(FeatureSnapshot).excluded.created_at,
            },
        )
        await session.execute(stmt)
        await session.commit()


    async def _save_prediction(self, session, decision):
        """Сохранить прогноз в БД."""
        stmt = insert(Prediction).values(
            time=decision.ts,
            symbol=decision.symbol,
            horizon_minutes=decision.horizon_min,
            label=decision.label.value,
            reason_code=decision.reason_code.value,
            p_up=decision.p_up,
            p_down=decision.p_down,
            p_flat=decision.p_flat,
            flat_score=decision.flat_score,
            uncertainty_score=decision.uncertainty_score,
            consensus=decision.consensus,
            latency_ms=decision.latency_ms,
            created_at=datetime.now(timezone.utc),
        ).on_conflict_do_update(
            index_elements=["time", "symbol", "horizon_minutes"],
            set_={
                "label": decision.label.value,
                "reason_code": decision.reason_code.value,
                "p_up": decision.p_up,
                "p_down": decision.p_down,
                "p_flat": decision.p_flat,
                "flat_score": decision.flat_score,
                "uncertainty_score": decision.uncertainty_score,
                "consensus": decision.consensus,
                "latency_ms": decision.latency_ms,
            },
        )
        await session.execute(stmt)
        await session.commit()

        # region agent log
        if "inferencer_first_saved" not in _DBG_STATE:
            _DBG_STATE.add("inferencer_first_saved")
            _agent_dbg(
                "H4",
                "services/inferencer/main.py:_save_prediction",
                "prediction_saved",
                {
                    "ts": decision.ts.isoformat(),
                    "symbol": decision.symbol,
                    "horizon_minutes": int(decision.horizon_min),
                    "label": decision.label.value,
                    "latency_ms": float(decision.latency_ms),
                },
            )
        # endregion

    async def _save_model_predictions(
        self,
        session,
        ts: datetime,
        horizon: int,
        model_opinions: list,
        model_weights: list[float],
    ) -> None:
        if not model_opinions:
            return

        rows = []
        for wrapper, opinion, weight in zip(self.model_wrappers, model_opinions, model_weights):
            model_forecast = FusedForecast(
                p_up=opinion.p_up,
                p_down=opinion.p_down,
                p_flat=opinion.p_flat,
                u_dir=opinion.u_dir,
                u_mag=opinion.u_mag,
                flat_score=opinion.p_flat,
                uncertainty_score=max(opinion.u_dir, opinion.u_mag),
                consensus=opinion.consensus,
                memory_support=None,
            )
            decision = self.policy.decide(
                forecast=model_forecast,
                symbol=self.config.symbol,
                horizon_min=horizon,
                ts=ts,
                latency_ms=0.0,
            )
            rows.append(
                {
                    "time": ts,
                    "symbol": self.config.symbol,
                    "horizon_minutes": int(horizon),
                    "model_id": wrapper.model_id,
                    "label": decision.label.value,
                    "reason_code": decision.reason_code.value,
                    "p_up": float(opinion.p_up),
                    "p_down": float(opinion.p_down),
                    "p_flat": float(opinion.p_flat),
                    "u_dir": float(opinion.u_dir),
                    "u_mag": float(opinion.u_mag),
                    "consensus": float(opinion.consensus),
                    "disagreement": float(opinion.disagreement),
                    "weight": float(weight),
                    "created_at": datetime.now(timezone.utc),
                }
            )

        stmt = insert(ModelPrediction).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["time", "symbol", "horizon_minutes", "model_id"],
            set_={
                "label": stmt.excluded.label,
                "reason_code": stmt.excluded.reason_code,
                "p_up": stmt.excluded.p_up,
                "p_down": stmt.excluded.p_down,
                "p_flat": stmt.excluded.p_flat,
                "u_dir": stmt.excluded.u_dir,
                "u_mag": stmt.excluded.u_mag,
                "consensus": stmt.excluded.consensus,
                "disagreement": stmt.excluded.disagreement,
                "weight": stmt.excluded.weight,
                "created_at": stmt.excluded.created_at,
            },
        )
        await session.execute(stmt)
        await session.commit()



async def run_inferencer(config: Config):
    """Главный цикл inferencer service."""
    logger.info("Starting inferencer service")
    
    # Инициализация БД
    await init_db()
    
    service = InferencerService(config)
    
    # Основной цикл: проверяем новые свечи каждую секунду
    while True:
        try:
            await service.process_new_candles()
            await asyncio.sleep(1.0)  # Проверяем каждую секунду
        except Exception as e:
            logger.error(f"Error in inferencer loop: {e}", exc_info=True)
            # region agent log
            _agent_dbg(
                "H4",
                "services/inferencer/main.py:run_inferencer",
                "inferencer_loop_exception",
                {"error": str(e)},
            )
            # endregion
            await asyncio.sleep(5.0)  # Пауза при ошибке


def main():
    """Точка входа для inferencer service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Inferencer service for Titan Oracle")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Config file")
    
    args = parser.parse_args()
    
    config_path = Path(__file__).parent.parent.parent.parent.parent / args.config
    config = load_config(config_path)
    
    setup_logging(
        level=config.logging.level,
        structured=config.logging.structured,
        log_file=Path(config.logging.log_file) if config.logging.log_file else None,
    )
    
    try:
        asyncio.run(run_inferencer(config))
    except KeyboardInterrupt:
        logger.info("Inferencer stopped")


if __name__ == "__main__":
    main()
