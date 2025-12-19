"""Загрузка и валидация конфигурации."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class PerformanceConfig(BaseModel):
    """Конфигурация производительности."""
    max_inference_time_ms: int = 10000
    max_api_latency_ms: int = 500
    adaptive_mc_passes: bool = True
    mc_passes_initial: int = 10
    mc_passes_min: int = 3
    mc_passes_max: int = 20


class FlatConfig(BaseModel):
    """Конфигурация FLAT."""
    m_flat: float = 0.05
    p_thr: float = 0.6
    u_mag_max: float = 0.3


class UncertaintyConfig(BaseModel):
    """Конфигурация UNCERTAIN."""
    u_thr: float = 0.4
    consensus_thr: float = 0.65


class DecisionConfig(BaseModel):
    """Decision configuration."""
    dir_p_thr: float = 0.55
    dir_gap_min: float = 0.02


class EnsembleConfig(BaseModel):
    """Конфигурация ансамбля."""
    num_models: int = 3
    mc_passes: int = 10
    ema_alpha: float = 0.1
    uncertainty_penalty_lambda: float = 0.3


class PatternStorageConfig(BaseModel):
    """Конфигурация хранилища паттернов."""
    lmdb_path: str = "artifacts/patterns/patterns.lmdb"
    log_dir: str = "artifacts/patterns/logs"
    disk_cap_gb: float = 5.0


class PatternHashConfig(BaseModel):
    """Конфигурация хеширования паттернов."""
    type: str = "simhash"
    bits: int = 256
    bands: int = 8
    band_bits: int = 32
    dual_hash: bool = True
    seed: int = 1337


class PatternSearchConfig(BaseModel):
    """Конфигурация поиска паттернов."""
    similarity_min: float = 0.90
    adaptive_step: float = 0.02
    target_candidates_min: int = 20
    target_candidates_max: int = 50
    full_vector_metric: str = "cosine"
    regime_filter: bool = True


class PatternLimitsConfig(BaseModel):
    """Лимиты паттернов."""
    ring_buffer: int = 10000
    aggregates_max: int = 1000000
    extremes_max: int = 1000


class PatternCullingConfig(BaseModel):
    """Конфигурация отсева паттернов."""
    n_min: int = 500
    p0_scope: str = "tf+regime"
    delta: float = 0.02
    contrarian: bool = True


class PatternsConfig(BaseModel):
    """Конфигурация памяти паттернов."""
    beta_prior_alpha: float = 1.0
    beta_prior_beta: float = 1.0
    beta_prior_alpha_flat: float = 1.0
    beta_prior_beta_flat: float = 1.0
    decay_half_life_hours: float = 168
    cooldown_errors_threshold: int = 5
    cooldown_duration_hours: float = 24
    min_pattern_samples: int = 3

    storage: PatternStorageConfig = PatternStorageConfig()
    hash: PatternHashConfig = PatternHashConfig()
    search: PatternSearchConfig = PatternSearchConfig()
    limits: PatternLimitsConfig = PatternLimitsConfig()
    culling: PatternCullingConfig = PatternCullingConfig()


class FusionConfig(BaseModel):
    """Конфигурация fusion."""
    memory_credibility_min: float = 0.3
    neural_confidence_weight: float = 0.7


class CalibrationConfig(BaseModel):
    """Конфигурация калибровки."""
    enabled: bool = True
    method: str = "temperature"
    temperature_initial: float = 1.0
    update_frequency: int = 100


class RewardBinsConfig(BaseModel):
    """Конфигурация бинов наград."""
    tiny: dict = Field(default={"max": 0.05, "weight": 0.2})
    small: dict = Field(default={"max": 0.25, "weight": 0.5})
    medium: dict = Field(default={"max": 0.75, "weight": 1.0})
    large: dict = Field(default={"max": 1.5, "weight": 1.5})
    extreme: dict = Field(default={"max": None, "weight": 2.0})


class RewardsConfig(BaseModel):
    """Конфигурация наград."""
    bins: RewardBinsConfig = RewardBinsConfig()
    confidence_penalty_alpha: float = 1.5
    streak: dict = Field(default={
        "model_error_multiplier": 1.2,
        "group_error_multiplier": 1.1,
        "horizon_error_multiplier": 1.15,
        "max_streak_multiplier": 3.0,
    })
    target_coverage: float = 0.7
    uncertain_penalty_base: float = 0.1
    uncertain_penalty_large_move: float = 0.5


class OnlineTrainingConfig(BaseModel):
    """Конфигурация онлайн обучения."""
    enabled: bool = True
    steps_per_candle: int = 1
    batch_size: int = 32
    learning_rate: float = 0.001
    learning_rate_min: float = 0.0001
    learning_rate_decay: float = 0.95
    gradient_clip_norm: float = 1.0
    update_frequency: int = 5


class ReplayConfig(BaseModel):
    """Конфигурация replay buffer."""
    buffer_size: int = 10000
    recent_ratio: float = 0.5
    priority_alpha: float = 0.6
    priority_beta: float = 0.4


class DistillationConfig(BaseModel):
    """Конфигурация distillation."""
    enabled: bool = True
    top_k_leaders: int = 2
    weight: float = 0.3
    min_leader_confidence: float = 0.7


class ContinualConfig(BaseModel):
    """Конфигурация continual learning."""
    anti_forgetting: bool = True
    kl_weight: float = 0.1
    ewc_enabled: bool = False
    ewc_lambda: float = 0.4


class TrainingConfig(BaseModel):
    """Конфигурация обучения."""
    online: OnlineTrainingConfig = OnlineTrainingConfig()
    replay: ReplayConfig = ReplayConfig()
    distillation: DistillationConfig = DistillationConfig()
    continual: ContinualConfig = ContinualConfig()


class DriftConfig(BaseModel):
    """Конфигурация drift detection."""
    enabled: bool = True
    window_size: int = 1000
    threshold: float = 0.15
    action: str = "increase_learning_rate"


class StorageConfig(BaseModel):
    """Конфигурация хранилища."""
    db_path: str = "data/oracle.db"
    checkpoint_dir: str = "checkpoints"
    checkpoint_frequency: int = 100
    keep_checkpoints: int = 5


class LoggingConfig(BaseModel):
    """Конфигурация логирования."""
    level: str = "INFO"
    structured: bool = True
    log_file: str = "logs/oracle.log"
    metrics_file: str = "logs/metrics.jsonl"


class BybitConfig(BaseModel):
    """Конфигурация Bybit API."""
    spot_ws_url: str = "wss://stream.bybit.com/v5/public/spot"
    spot_rest_url: str = "https://api.bybit.com/v5/market"
    reconnect_delay: int = 5
    max_reconnect_attempts: int = 10
    request_timeout: int = 10


@dataclass
class Config:
    """Главная конфигурация системы."""
    symbol: str = "BTCUSDT"
    timeframe: str = "1m"
    horizons: list[int] = field(default_factory=lambda: [1, 5, 10, 15, 30, 60])
    
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    flat: FlatConfig = field(default_factory=FlatConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    patterns: PatternsConfig = field(default_factory=PatternsConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    rewards: RewardsConfig = field(default_factory=RewardsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    bybit: BybitConfig = field(default_factory=BybitConfig)
    
    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Загрузка конфигурации из YAML файла."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        config = cls(
            symbol=data.get("symbol", "BTCUSDT"),
            timeframe=data.get("timeframe", "1m"),
            horizons=data.get("horizons", [1, 5, 10, 15, 30, 60]),
        )
        
        # Загружаем вложенные конфигурации
        if "performance" in data:
            config.performance = PerformanceConfig(**data["performance"])
        if "flat" in data:
            config.flat = FlatConfig(**data["flat"])
        if "uncertainty" in data:
            config.uncertainty = UncertaintyConfig(**data["uncertainty"])
        if "decision" in data:
            config.decision = DecisionConfig(**data["decision"])
        if "ensemble" in data:
            config.ensemble = EnsembleConfig(**data["ensemble"])
        if "patterns" in data:
            config.patterns = PatternsConfig(**data["patterns"])
        if "fusion" in data:
            config.fusion = FusionConfig(**data["fusion"])
        if "calibration" in data:
            config.calibration = CalibrationConfig(**data["calibration"])
        if "rewards" in data:
            config.rewards = RewardsConfig(**data["rewards"])
        if "training" in data:
            config.training = TrainingConfig(**data["training"])
        if "drift" in data:
            config.drift = DriftConfig(**data["drift"])
        if "storage" in data:
            config.storage = StorageConfig(**data["storage"])
        if "logging" in data:
            config.logging = LoggingConfig(**data["logging"])
        if "bybit" in data:
            config.bybit = BybitConfig(**data["bybit"])
        
        return config
    
    @classmethod
    def default(cls) -> "Config":
        """Создание конфигурации по умолчанию."""
        return cls()


def load_config(config_path: Optional[Path] = None) -> Config:
    """Загрузка конфигурации из файла или создание по умолчанию."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "default.yaml"
    
    if config_path.exists():
        return Config.from_yaml(config_path)
    else:
        return Config.default()

