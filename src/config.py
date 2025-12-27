from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict
import json


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
    pred_flat_mode: str = "fixed"
    pred_flat_target_lo: float = 0.05
    pred_flat_target_hi: float = 0.15
    pred_flat_delta_min: float = 0.01
    pred_flat_delta_max: float = 0.08
    pred_flat_adjust_rate: float = 0.05
    pred_flat_ema_decay: float = 0.02
    pred_flat_min_action_acc: float = 0.45
    use_context_priors: bool = True
    context_trust_min: float = 0.15
    context_flat_gain: float = 0.12
    context_dir_gain: float = 0.08
    anti_flat_prob_boost: float = 0.08
    anti_flat_delta_boost: float = 0.05


@dataclass
class FactConfig:
    fact_flat_mode: str = "fixed"
    fact_flat_bps: float = 0.20
    fact_flat_window: int = 300
    fact_flat_update_every: int = 20
    fact_flat_min_samples: int = 50
    fact_flat_target_lo: float = 0.10
    fact_flat_target_hi: float = 0.25
    fact_flat_p_start: float = 0.15
    fact_flat_p_step: float = 0.02
    fact_flat_p_min: float = 0.02
    fact_flat_p_max: float = 0.35
    fact_flat_bps_min: float = 0.01
    fact_flat_bps_max: float = 0.25
    fact_flat_smooth_beta: float = 0.20


@dataclass
class RewardConfig:
    reward_correct: float = 1.0
    reward_wrong_dir: float = -1.0
    reward_flat_miss: float = -0.5
    reward_dir_in_flat: float = -0.5
    reward_mode: str = "classic"
    shaped_R_max: float = 6.0
    shaped_R_big: float = 1.0
    shaped_alpha: float = 0.70
    shaped_x_cap: float = 6.0
    shaped_wrong_near: float = -1.0
    shaped_wrong_far: float = -1.5
    shaped_flat_miss_near: float = -0.6
    shaped_flat_miss_far: float = -1.2
    shaped_flat_correct: float = 0.10
    shaped_dir_in_flat_base: float = -0.25
    shaped_dir_in_flat_slope: float = -0.55
    shaped_micro_x_max: float = 1.20
    shaped_micro_share_thr: float = 0.55
    shaped_micro_scale: float = 0.70
    shaped_reward_clip_min: float = -2.0
    shaped_reward_clip_max: float = 6.0


@dataclass
class ModelInitConfig:
    init_mode: str = "heuristic"
    seed: int = 42
    logit_scale: float = 3.0
    weight_clip: float = 5.0


@dataclass
class PerModelCalibConfig:
    """
    Per-model calibration configuration overrides.
    Allows fine-tuning calibration parameters for specific models
    (e.g., TRENDVIC, OSCILLATOR, VOLUMEMETRIX).
    """

    # Base calibration parameter overrides
    lr: float | None = None
    a_min: float | None = None
    a_max: float | None = None
    b_min: float | None = None
    b_max: float | None = None
    l2_a: float | None = None
    l2_b: float | None = None

    # Adaptive lr parameter overrides
    lr_min: float | None = None
    lr_max: float | None = None
    ece_target: float | None = None
    ece_good_threshold: float | None = None
    ece_bad_threshold: float | None = None
    lr_increase_factor: float | None = None
    lr_decrease_factor: float | None = None
    adaptation_interval: int | None = None

    # Initial state overrides
    init_a: float | None = None
    init_b: float | None = None


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

    # Adaptive calibration lr defaults
    calib_lr_min: float = 0.0001
    calib_lr_max: float = 0.02
    calib_ece_target: float = 0.05
    calib_ece_good_threshold: float = 0.10
    calib_ece_bad_threshold: float = 0.25
    calib_lr_increase_factor: float = 1.5
    calib_lr_decrease_factor: float = 0.9
    calib_adaptation_interval: int = 20
    calib_min_samples_for_adaptation: int = 30
    calib_ece_window_size: int = 50

    # Per-model calibration configs
    calib_per_model: Dict[str, PerModelCalibConfig] = field(default_factory=dict)


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


def load_calibration_config(
    config_path: str | Path, base_training: TrainingConfig
) -> TrainingConfig:
    """
    Load calibration configuration from JSON file and apply to TrainingConfig.

    JSON structure:
    {
        "global": {
            "calib_lr": 0.005,
            "calib_lr_min": 0.0001,
            "calib_lr_max": 0.02,
            "calib_ece_target": 0.05,
            "calib_ece_good_threshold": 0.10,
            "calib_ece_bad_threshold": 0.25,
            "calib_lr_increase_factor": 1.5,
            "calib_lr_decrease_factor": 0.9,
            "calib_adaptation_interval": 20,
            "calib_min_samples_for_adaptation": 30,
            "calib_ece_window_size": 50,
            "calib_a_min": 0.30,
            "calib_a_max": 2.0,
            "calib_b_min": -1.0,
            "calib_b_max": 1.0,
            "calib_l2_a": 0.01,
            "calib_l2_b": 0.001
        },
        "per_model": {
            "OSCILLATOR": {
                "lr": 0.010,
                "a_min": 0.15,
                "a_max": 3.0,
                "lr_max": 0.04,
                "ece_target": 0.10,
                "ece_bad_threshold": 0.35
            },
            "VOLUMEMETRIX": {
                "lr": 0.007,
                "a_min": 0.25,
                "ece_target": 0.08
            }
        }
    }

    Args:
        config_path: Path to JSON config file
        base_training: Base TrainingConfig to modify

    Returns:
        Modified TrainingConfig with calibration overrides applied
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration config not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Start with a copy of base config
    from copy import deepcopy
    config = deepcopy(base_training)

    # Apply global calibration overrides
    global_overrides = data.get("global", {})
    for key, value in global_overrides.items():
        # Skip comments and metadata (keys starting with _)
        if key.startswith("_"):
            continue

        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown calibration parameter: {key}")

    # Apply per-model configs
    per_model_data = data.get("per_model", {})
    config.calib_per_model = {}

    for model_name, model_overrides in per_model_data.items():
        # Skip comments and metadata (keys starting with _)
        if model_name.startswith("_"):
            continue

        if not isinstance(model_overrides, dict):
            raise ValueError(f"Per-model config for {model_name} must be a dict")

        # Create PerModelCalibConfig from dict
        per_model_cfg = PerModelCalibConfig(
            lr=model_overrides.get("lr"),
            a_min=model_overrides.get("a_min"),
            a_max=model_overrides.get("a_max"),
            b_min=model_overrides.get("b_min"),
            b_max=model_overrides.get("b_max"),
            l2_a=model_overrides.get("l2_a"),
            l2_b=model_overrides.get("l2_b"),
            lr_min=model_overrides.get("lr_min"),
            lr_max=model_overrides.get("lr_max"),
            ece_target=model_overrides.get("ece_target"),
            ece_good_threshold=model_overrides.get("ece_good_threshold"),
            ece_bad_threshold=model_overrides.get("ece_bad_threshold"),
            lr_increase_factor=model_overrides.get("lr_increase_factor"),
            lr_decrease_factor=model_overrides.get("lr_decrease_factor"),
            adaptation_interval=model_overrides.get("adaptation_interval"),
            init_a=model_overrides.get("init_a"),
            init_b=model_overrides.get("init_b"),
        )
        config.calib_per_model[model_name] = per_model_cfg

    return config
