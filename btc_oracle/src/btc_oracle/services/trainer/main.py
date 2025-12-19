"""Trainer service: обрабатывает matured predictions, обучает модели."""

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from btc_oracle.core.config import Config, load_config
from btc_oracle.core.log import get_logger, setup_logging
from btc_oracle.core.types import Candle as CandleType, Label
from btc_oracle.core.timeframes import timeframe_to_minutes
from btc_oracle.db import AsyncSessionLocal, Candle, Prediction, init_db
from btc_oracle.features.pipeline import FeaturePipeline
from btc_oracle.patterns.encoder import PatternEncoder
from btc_oracle.patterns.store import DecisionRecord, PatternStore
from btc_oracle.rewards.score import RewardScorer
from btc_oracle.training.labels import compute_truth_label
from btc_oracle.training.online import OnlineTrainer
from btc_oracle.training.replay import ReplayBuffer
from sqlalchemy import select, and_

logger = get_logger(__name__)


class TrainerService:
    """Сервис обучения моделей."""
    
    def __init__(self, config: Config):
        """Инициализация сервиса."""
        self.config = config
        
        # Инициализация компонентов
        self.pattern_store = PatternStore(config.patterns)
        self.pattern_encoder = PatternEncoder(config.patterns)
        self.feature_pipeline = FeaturePipeline(window_size=100)
        self.base_tf_min = timeframe_to_minutes(config.timeframe)
        
        # Rewards
        bins_config = {
            "tiny": config.rewards.bins.tiny,
            "small": config.rewards.bins.small,
            "medium": config.rewards.bins.medium,
            "large": config.rewards.bins.large,
            "extreme": config.rewards.bins.extreme,
        }
        self.reward_scorer = RewardScorer(
            bins_config=bins_config,
            confidence_penalty_alpha=config.rewards.confidence_penalty_alpha,
            target_coverage=config.rewards.target_coverage,
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            buffer_size=config.training.replay.buffer_size,
            recent_ratio=config.training.replay.recent_ratio,
        )
        
        # TODO: Инициализировать OnlineTrainer когда будут готовы модели
        # self.online_trainer = OnlineTrainer(...)
    
    async def process_matured_predictions(self):
        """Обработать созревшие прогнозы."""
        async with AsyncSessionLocal() as session:
            current_time = datetime.now(timezone.utc)
            
            for horizon in self.config.horizons:
                # Находим прогнозы, которые созрели (matured_at IS NULL и прошло horizon минут)
                horizon_delta = timedelta(minutes=horizon)
                cutoff_time = current_time - horizon_delta
                
                query = select(Prediction).where(
                    Prediction.symbol == self.config.symbol,
                    Prediction.horizon_minutes == horizon,
                    Prediction.matured_at.is_(None),
                    Prediction.time <= cutoff_time,
                ).limit(100)  # Обрабатываем батчами
                
                result = await session.execute(query)
                predictions = result.scalars().all()
                
                if not predictions:
                    continue
                
                logger.info(f"Processing {len(predictions)} matured predictions for {horizon}m")
                
                for pred in predictions:
                    await self._process_prediction(session, pred, horizon)
    
    async def _process_prediction(self, session, pred: Prediction, horizon: int):
        """Обработать один созревший прогноз."""
        # Получаем свечу на момент прогноза
        forecast_candle = await self._get_candle_at(session, pred.time)
        
        # Получаем truth свечу (через horizon минут)
        truth_time = pred.time + timedelta(minutes=horizon)
        truth_candle = await self._get_candle_at(session, truth_time)
        
        if not forecast_candle or not truth_candle:
            logger.warning(f"Missing candles for prediction at {pred.time}")
            return
        
        # Вычисляем truth label
        # TODO: получить реальный ATR из features
        atr = 0.01  # упрощённо
        truth_label, magnitude = compute_truth_label(
            forecast_candle,
            truth_candle,
            atr,
            m_flat_threshold=self.config.flat.m_flat,
        )
        
        # Вычисляем reward/penalty
        from btc_oracle.core.types import Decision
        
        decision = Decision(
            label=Label(pred.label),
            reason_code=pred.reason_code,  # type: ignore
            ts=pred.time,
            symbol=pred.symbol,
            horizon_min=pred.horizon_minutes,
            p_up=pred.p_up,
            p_down=pred.p_down,
            p_flat=pred.p_flat,
            flat_score=pred.flat_score,
            uncertainty_score=pred.uncertainty_score,
            consensus=pred.consensus,
        )
        
        reward = self.reward_scorer.compute_reward(
            decision=decision,
            truth=truth_label,
            magnitude=magnitude,
        )
        
        # Обновляем прогноз в БД
        pred.truth_label = truth_label.value
        pred.truth_magnitude = magnitude
        pred.reward = reward
        pred.matured_at = datetime.now(timezone.utc)
        
        await session.commit()
        
        # Обновляем pattern store
        feature_window = await self._get_feature_window(
            session,
            pred.time,
            window_size=self.feature_pipeline.window_size,
        )
        if feature_window:
            features = self.feature_pipeline.build_features(feature_window)
            features.ts = pred.time
            features.timeframe = self.config.timeframe
            context = {
                "p_up": float(pred.p_up),
                "p_down": float(pred.p_down),
                "p_flat": float(pred.p_flat),
                "u_dir": float(pred.uncertainty_score),
                "u_mag": float(pred.uncertainty_score),
                "consensus": float(pred.consensus),
                "disagreement": float(1.0 - pred.consensus),
            }
            pattern_key = self.pattern_encoder.encode(
                features,
                horizon=horizon,
                timeframe=self.config.timeframe,
                context=context,
            )
            record = DecisionRecord(
                ts_ms=int(pred.time.timestamp() * 1000),
                tf_id=horizon,
                model_id=0,
                head_id=0,
                pred_class=_label_to_int(decision.label),
                actual_class=_label_to_int(truth_label),
                flags=_build_flags(decision.label, truth_label),
                p_up=float(decision.p_up),
                p_down=float(decision.p_down),
                confidence=_decision_confidence(decision),
                reward=float(reward),
                outcome_margin=float(magnitude),
            )
            candle_blob = _build_candle_blob(forecast_candle, truth_candle)
            self.pattern_store.record_decision(
                pattern_key,
                record,
                features=features,
                candle_blob=candle_blob,
            )
        else:
            logger.debug("Missing feature window for pattern update", ts=pred.time, horizon=horizon)
        
        logger.debug(
            f"Processed prediction: {pred.time} -> {truth_label.value}, "
            f"magnitude={magnitude:.4f}, reward={reward:.4f}"
        )
    
    async def _get_candle_at(self, session, ts: datetime) -> CandleType | None:
        """Получить свечу на конкретный момент времени."""
        tolerance = timedelta(minutes=max(1, self.base_tf_min))
        
        query = select(Candle).where(
            Candle.symbol == self.config.symbol,
            Candle.timeframe == self.config.timeframe,
            Candle.time >= ts - tolerance,
            Candle.time <= ts + tolerance,
        ).order_by(Candle.time.asc()).limit(1)
        
        result = await session.execute(query)
        db_candle = result.scalar_one_or_none()
        
        if not db_candle:
            return None
        
        return CandleType(
            ts=db_candle.time,
            open=db_candle.open,
            high=db_candle.high,
            low=db_candle.low,
            close=db_candle.close,
            volume=db_candle.volume,
            confirmed=True,
        )

    async def _get_feature_window(
        self,
        session,
        end_ts: datetime,
        window_size: int,
    ) -> list[CandleType]:
        """Получить окно свечей для features."""
        query = select(Candle).where(
            Candle.symbol == self.config.symbol,
            Candle.timeframe == self.config.timeframe,
            Candle.time <= end_ts,
            Candle.confirmed == 1,
        ).order_by(Candle.time.desc()).limit(window_size)

        result = await session.execute(query)
        db_candles = result.scalars().all()

        candles = []
        for db_candle in reversed(db_candles):
            candles.append(
                CandleType(
                    ts=db_candle.time,
                    open=db_candle.open,
                    high=db_candle.high,
                    low=db_candle.low,
                    close=db_candle.close,
                    volume=db_candle.volume,
                    confirmed=True,
                )
            )
        return candles

    
    async def train_step(self):
        """Выполнить один шаг обучения."""
        if not self.config.training.online.enabled:
            return
        
        # TODO: Реализовать online training когда будут готовы модели
        # if len(self.replay_buffer) >= self.config.training.online.batch_size:
        #     self.online_trainer.train_step(...)
        pass


_FLAG_MISDIRECTION = 1 << 0
_FLAG_CONFLICT = 1 << 1
_FLAG_UNCERTAIN = 1 << 2
_FLAG_FLAT = 1 << 3


def _label_to_int(label: Label) -> int:
    if label == Label.UP:
        return 1
    if label == Label.DOWN:
        return 2
    if label == Label.FLAT:
        return 3
    if label == Label.UNCERTAIN:
        return 4
    return 0


def _build_flags(pred_label: Label, truth_label: Label) -> int:
    flags = 0
    if pred_label == Label.UNCERTAIN:
        flags |= _FLAG_UNCERTAIN
    if pred_label == Label.FLAT or truth_label == Label.FLAT:
        flags |= _FLAG_FLAT
    if pred_label in (Label.UP, Label.DOWN) and truth_label in (Label.UP, Label.DOWN):
        if pred_label != truth_label:
            flags |= _FLAG_MISDIRECTION
    return flags


def _decision_confidence(decision) -> float:
    if decision.label == Label.UP:
        return float(decision.p_up)
    if decision.label == Label.DOWN:
        return float(decision.p_down)
    if decision.label == Label.FLAT:
        return float(decision.p_flat)
    if decision.label == Label.UNCERTAIN:
        return float(max(0.0, 1.0 - decision.uncertainty_score))
    return float(max(decision.p_up, decision.p_down, decision.p_flat))


def _build_candle_blob(forecast: CandleType, truth: CandleType) -> bytes:
    data = np.array(
        [
            forecast.open,
            forecast.high,
            forecast.low,
            forecast.close,
            forecast.volume,
            truth.open,
            truth.high,
            truth.low,
            truth.close,
            truth.volume,
        ],
        dtype=np.float32,
    )
    return data.tobytes()


async def run_trainer(config: Config):
    """Главный цикл trainer service."""
    logger.info("Starting trainer service")
    
    # Инициализация БД
    await init_db()
    
    service = TrainerService(config)
    
    # Основной цикл: обрабатываем matured predictions каждые 5 секунд
    while True:
        try:
            await service.process_matured_predictions()
            await service.train_step()
            await asyncio.sleep(5.0)
        except Exception as e:
            logger.error(f"Error in trainer loop: {e}", exc_info=True)
            await asyncio.sleep(10.0)


def main():
    """Точка входа для trainer service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Trainer service for Titan Oracle")
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
        asyncio.run(run_trainer(config))
    except KeyboardInterrupt:
        logger.info("Trainer stopped")


if __name__ == "__main__":
    main()
