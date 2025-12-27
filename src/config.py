from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict


def _normalize_config(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _normalize_config(v) for k, v in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _normalize_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_config(v) for v in value]
    return value


@dataclass
class DataConfig:
    symbol: str = "BTCUSDT"
    tfs: list[str] = field(default_factory=lambda: ["1"])
    ws_url: str = "wss://stream.bybit.com/v5/public/spot"
    use_confirmed_only: bool = True
    reconnect_delay: float = 1.0
    ping_interval: float = 20.0


@dataclass
class RestConfig:
    kline_url: str = "https://api.bybit.com/v5/market/kline"
    timeout_s: float = 10.0
    history_limit: int = 200
    enable_history: bool = True


@dataclass
class FeatureConfig:
    lookback: int = 120
    ma_periods: list[int] = field(default_factory=lambda: [5, 10, 20])
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    sar_step: float = 0.02
    sar_max: float = 0.2
    rsi_period: int = 14
    stoch_k: int = 14
    stoch_d: int = 3
    stoch_smooth: int = 3
    cci_period: int = 20
    mfi_period: int = 14
    obv_slope_period: int = 5
    vol_z_window: int = 30
    range_window: int = 14
    ret_bin_bps: float = 5.0
    ma_slope_bps: float = 2.0
    vol_low_bps: float = 5.0
    vol_high_bps: float = 20.0
    range_low_pct: float = 0.05
    range_high_pct: float = 0.3
    schema_version: str = "v2"


@dataclass
class DecisionConfig:
    flat_max_prob: float = 0.55
    flat_max_delta: float = 0.02
    use_context_priors: bool = True
    context_trust_min: float = 0.15
    context_flat_gain: float = 0.12
    context_dir_gain: float = 0.08
    anti_flat_prob_boost: float = 0.08
    anti_flat_delta_boost: float = 0.05


@dataclass
class FactConfig:
    fact_flat_bps: float = 1.0


@dataclass
class RewardConfig:
    reward_correct: float = 1.0
    reward_wrong_dir: float = -1.0
    reward_flat_miss: float = -0.5
    reward_dir_in_flat: float = -0.5


@dataclass
class ModelInitConfig:
    init_mode: str = "heuristic"
    seed: int = 42
    logit_scale: float = 3.0
    weight_clip: float = 5.0


@dataclass
class TrainingConfig:
    ema_decay: float = 0.999
    anchor_decay: float = 0.9999
    anchor_lambda_base: float = 1e-4
    anchor_gain: float = 0.2
    lr_gain: float = 0.5
    lr_min_mult: float = 0.5
    lr_max_mult: float = 2.0
    anchor_min_mult: float = 0.5
    anchor_max_mult: float = 2.0
    flat_train_weight: float = 0.25
    class_balance_strength: float = 0.5
    class_balance_ema: float = 0.98
    class_balance_min: float = 0.5
    class_balance_max: float = 2.0
    class_balance_floor: float = 0.05
    calib_lr: float = 0.005
    calib_a_min: float = 0.30
    calib_a_max: float = 2.0
    calib_b_min: float = -1.0
    calib_b_max: float = 1.0
    calib_l2_a: float = 0.01
    calib_l2_b: float = 0.001
    calib_flat_bps: float = 1.0
    calib_flat_weight: float = 0.25
    calibration_bins: int = 10
    perf_lr_gain: float = 0.4
    perf_lr_min_mult: float = 0.5
    perf_lr_max_mult: float = 1.5
    perf_lr_baseline: float = 0.5
    perf_lr_min_samples: int = 50


@dataclass
class ModelLRConfig:
    lr_trend: float = 0.001
    lr_osc: float = 0.001
    lr_vol: float = 0.001


@dataclass
class PatternConfig:
    db_path: Path = Path("state/patterns.db")
    ema_decay: float = 0.995
    event_ttl_days: int = 14
    max_events: int = 100_000
    max_patterns: int = 200_000
    support_k: int = 20
    recency_tau_hours: float = 72.0
    anti_min_support: int = 20
    anti_win_threshold: float = 0.35
    anti_trust_threshold: float = 0.15
    maintenance_seconds: int = 600
    fine_trust_threshold: float = 0.2


@dataclass
class PersistenceConfig:
    state_db: Path = Path("state/state.db")
    autosave_seconds: int = 300
    autosave_updates: int = 50


@dataclass
class OutputConfig:
    out_dir: Path | None = None


@dataclass
class DashboardConfig:
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    history_size: int = 200


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class ModelConfig:
    direction_flip_by_model: Dict[str, bool] = field(
        default_factory=lambda: {
            "TRENDVIC": False,
            "OSCILLATOR": True,
            "VOLUMEMETRIX": True,
        }
    )


@dataclass
class AppConfig:
    data: DataConfig
    rest: RestConfig
    features: FeatureConfig
    decision: DecisionConfig
    facts: FactConfig
    reward: RewardConfig
    model: ModelConfig
    model_init: ModelInitConfig
    training: TrainingConfig
    lrs: ModelLRConfig
    patterns: PatternConfig
    persistence: PersistenceConfig
    output: OutputConfig
    dashboard: DashboardConfig
    logging: LoggingConfig


def config_to_dict(config: AppConfig) -> Dict[str, Any]:
    return _normalize_config(config)
