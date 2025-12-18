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
from btc_oracle.training.labels import compute_truth_label

logger = get_logger(__name__)


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
        self.pattern_store = PatternStore(
            Path(config.storage.db_path),
            discretization_bins=config.patterns.discretization_bins,
        )
        self.pattern_encoder = PatternEncoder(bins=config.patterns.discretization_bins)
        self.feature_pipeline = FeaturePipeline(window_size=100)
        
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
    
    def _init_models(self):
        """Инициализировать модели ансамбля."""
        # Определяем размерность признаков (примерно)
        feature_dim = 30  # из pipeline
        
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
        
        # Параллельно получаем мнения Neural и Memory
        neural_task = self.aggregator.aggregate(features, enable_dropout=True)
        
        # Memory opinion
        pattern_key = self.pattern_encoder.encode(
            features,
            horizon=self.config.horizons[0],  # для первого горизонта
            timeframe=self.config.timeframe,
        )
        memory_opinion = self.pattern_store.get_opinion(
            pattern_key,
            min_samples=self.config.patterns.min_pattern_samples,
        )
        
        # Ждём neural
        neural_opinion = await neural_task
        
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
            horizon_min=self.config.horizons[0],
            ts=candle.ts,
            latency_ms=latency_ms,
        )
        
        # Сохраняем прогноз
        self.store.add_prediction(decision)
        
        logger.info(
            "Prediction made",
            label=decision.label.value,
            reason=decision.reason_code.value,
            latency_ms=latency_ms,
        )
        
        return decision
    
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
                    
                    # Обновляем прогноз
                    reward = 0.0  # будет вычисляться в rewards
                    self.store.mark_prediction_matured(
                        ts=pred_data["ts"],
                        symbol=self.config.symbol,
                        horizon_min=horizon,
                        truth_label=truth,
                        truth_magnitude=magnitude,
                        reward=reward,
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

