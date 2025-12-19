"""Главный live loop для системы прогнозирования."""

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from btc_oracle.core.config import Config, load_config
from btc_oracle.core.log import get_logger, setup_logging
from btc_oracle.core.types import Candle, Decision, Features, Label
from btc_oracle.core.timeframes import candle_close_time, timeframe_to_minutes
from btc_oracle.data.bybit_spot import BybitSpotClient
from btc_oracle.data.repository import DataRepository
from btc_oracle.data.store import DataStore
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
from btc_oracle.rewards.score import RewardScorer
from btc_oracle.training.labels import compute_truth_label

logger = get_logger(__name__)


def _is_horizon_tick(ts: datetime, horizon_min: int) -> bool:
    if ts.second != 0 or ts.microsecond != 0:
        return False
    if horizon_min <= 1:
        return True
    minute = ts.minute
    if horizon_min >= 60:
        return minute == 0
    return minute % int(horizon_min) == 0


class OracleApp:
    """Главное приложение системы прогнозирования."""
    
    def __init__(self, config: Config):
        """
        Args:
            config: конфигурация системы
        """
        self.config = config
        
        # Инициализация компонентов
        self.store = DataStore(Path(config.storage.db_path))
        self.repository = DataRepository(self.store)
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
        
        # Инициализация моделей
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
            uncertain_penalty_base=config.rewards.uncertain_penalty_base,
            uncertain_penalty_large_move=config.rewards.uncertain_penalty_large_move,
        )
    
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
    
    async def process_candle(self, candle: Candle) -> Optional[Decision]:
        """
        Обработать новую свечу и выдать прогноз.
        
        Args:
            candle: новая свеча
        
        Returns:
            Decision или None если недостаточно данных
        """
        start_time = time.time()
        candle_open_ts = candle.ts.replace(second=0, microsecond=0)
        candle_close_ts = candle_close_time(candle_open_ts, self.base_tf_min)
        
        # Сохраняем свечу
        self.store.add_candle(candle, self.config.symbol, self.config.timeframe)
        
        # Получаем окно для признаков
        feature_window = self.repository.get_feature_window(
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            end_ts=candle.ts,
            window_size=100,
        )
        
        if len(feature_window) < 50:  # минимум данных
            return None
        
        # Строим признаки
        features = self.feature_pipeline.build_features(feature_window)
        features.ts = candle_close_ts
        features.timeframe = self.config.timeframe
        
        # Neural opinion + детали по моделям
        fused_opinion, model_opinions, model_weights = await self.aggregator.aggregate_with_details(
            features, enable_dropout=True
        )

        horizons = sorted(self.config.horizons)
        last_decision: Optional[Decision] = None
        
        for horizon in horizons:
            if not _is_horizon_tick(candle_close_ts, horizon):
                continue

            pattern_key = self.pattern_encoder.encode(
                features,
                horizon=horizon,
                timeframe=self.config.timeframe,
                context=fused_opinion,
            )
            memory_opinion = self.pattern_store.get_opinion(
                pattern_key,
                min_samples=self.config.patterns.min_pattern_samples,
                features=features,
            )
            
            # Fusion
            fused = self.fusion_engine.fuse(fused_opinion, memory_opinion)
            
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
            
            # Сохраняем прогноз и индивидуальные мнения моделей
            per_model_payload = [
                {
                    "model_id": f"model_{idx}",
                    "p_up": op.p_up,
                    "p_down": op.p_down,
                    "p_flat": op.p_flat,
                    "u_dir": op.u_dir,
                    "u_mag": op.u_mag,
                }
                for idx, op in enumerate(model_opinions)
            ]
            self.store.add_prediction(
                decision,
                model_opinions=per_model_payload,
                model_weights=model_weights,
            )
            last_decision = decision
            
            logger.info(
                "Prediction made",
                label=decision.label.value,
                reason=decision.reason_code.value,
                latency_ms=latency_ms,
                horizon=horizon,
            )
        
        return last_decision
    
    async def process_matured_predictions(self):
        """Обработать созревшие прогнозы."""
        current_ts = int(datetime.now().timestamp() * 1000)
        
        for horizon in self.config.horizons:
            pending = self.store.get_pending_predictions(
                symbol=self.config.symbol,
                horizon_min=horizon,
                current_ts=current_ts,
            )
            
            for pred_data in pending:
                # Получаем truth
                forecast_ts = datetime.fromtimestamp(pred_data["ts"] / 1000)
                feature_window = self.repository.get_feature_window(
                    symbol=self.config.symbol,
                    timeframe=self.config.timeframe,
                    end_ts=forecast_ts,
                    window_size=1,
                )
                forecast_candle = feature_window[0] if feature_window else None
                
                truth_candle = self.repository.get_truth_candle(
                    symbol=self.config.symbol,
                    timeframe=self.config.timeframe,
                    forecast_ts=forecast_ts,
                    horizon_min=horizon,
                )
                
                if forecast_candle and truth_candle:
                    # Вычисляем truth
                    atr = 0.01  # упрощённо, в реальности из features.meta
                    truth, magnitude = compute_truth_label(
                        forecast_candle,
                        truth_candle,
                        atr,
                        m_flat_threshold=self.config.flat.m_flat,
                    )
                    
                    from btc_oracle.core.types import Decision, FusedForecast

                    agg_decision = Decision(
                        label=Label(pred_data["label"]),
                        reason_code=pred_data["reason_code"],  # type: ignore
                        ts=forecast_ts,
                        symbol=pred_data["symbol"],
                        horizon_min=horizon,
                        p_up=pred_data["p_up"],
                        p_down=pred_data["p_down"],
                        p_flat=pred_data["p_flat"],
                        flat_score=pred_data["flat_score"],
                        uncertainty_score=pred_data["uncertainty_score"],
                        consensus=pred_data["consensus"],
                    )
                    reward = self.reward_scorer.compute_reward(
                        decision=agg_decision,
                        truth=truth,
                        magnitude=magnitude,
                    )

                    model_rewards: list[float] = []
                    if pred_data.get("model_opinions"):
                        for idx, model_payload in enumerate(pred_data["model_opinions"]):
                            single_forecast = FusedForecast(
                                p_up=model_payload["p_up"],
                                p_down=model_payload["p_down"],
                                p_flat=model_payload["p_flat"],
                                u_dir=model_payload["u_dir"],
                                u_mag=model_payload["u_mag"],
                                flat_score=model_payload["p_flat"],
                                uncertainty_score=model_payload["u_dir"],
                                consensus=pred_data.get("model_weights", [1.0])[idx]
                                if pred_data.get("model_weights")
                                else pred_data["consensus"],
                                memory_support=None,
                            )
                            single_decision = self.policy.decide(
                                forecast=single_forecast,
                                symbol=self.config.symbol,
                                horizon_min=horizon,
                                ts=forecast_ts,
                                latency_ms=pred_data["latency_ms"],
                            )
                            model_reward = self.reward_scorer.compute_reward(
                                decision=single_decision,
                                truth=truth,
                                magnitude=magnitude,
                                model_id=model_payload["model_id"],
                                horizon=horizon,
                            )
                            model_rewards.append(model_reward)

                        if model_rewards:
                            self.weight_manager.update_weights(model_rewards)

                    self.store.mark_prediction_matured(
                        ts=pred_data["ts"],
                        symbol=self.config.symbol,
                        horizon_min=horizon,
                        truth_label=truth.value,
                        truth_magnitude=magnitude,
                        reward=reward,
                        model_rewards=model_rewards if model_rewards else None,
                    )
    
    async def run_live(self):
        """Запустить live loop."""
        logger.info("Starting live loop")
        
        async with BybitSpotClient(self.config.bybit) as client:
            async def on_new_candle(candle: Candle):
                await self.process_candle(candle)
                await self.process_matured_predictions()
            
            # Подписываемся на свечи
            await client.subscribe_klines(
                symbol=self.config.symbol,
                interval="1",  # 1 минута
                callback=on_new_candle,
            )


def main():
    """Точка входа."""
    import sys
    
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config/default.yaml")
    config = load_config(config_path)
    
    setup_logging(
        level=config.logging.level,
        structured=config.logging.structured,
        log_file=Path(config.logging.log_file) if config.logging.log_file else None,
    )
    
    app = OracleApp(config)
    
    try:
        asyncio.run(app.run_live())
    except KeyboardInterrupt:
        logger.info("Shutting down")

