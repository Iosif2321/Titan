import argparse
import asyncio
import contextlib
import logging
from pathlib import Path

import jax

from .analysis_cycle import AnalysisConfig, PeriodicAnalyzer
from .bybit_rest import fetch_klines
from .bybit_ws import stream_candles
from .config import (
    AppConfig,
    DashboardConfig,
    DataConfig,
    DecisionConfig,
    FeatureConfig,
    LoggingConfig,
    ModelInitConfig,
    ModelLRConfig,
    OutputConfig,
    PatternConfig,
    PersistenceConfig,
    RestConfig,
    TrainingConfig,
)
from .config_manager import ConfigManager
from .engine import MultiTimeframeEngine
from .pattern_store import PatternStore
from .recording import JsonlRecorder
from .runtime_state import RuntimeState
from .state_store import ModelStateStore
from .utils import now_ms, parse_tfs

try:
    from .dashboard.server import start_dashboard
except Exception:  # pragma: no cover - optional UI dependency
    start_dashboard = None


def build_arg_parser() -> argparse.ArgumentParser:
    data_defaults = DataConfig()
    rest_defaults = RestConfig()
    feature_defaults = FeatureConfig()
    decision_defaults = DecisionConfig()
    training_defaults = TrainingConfig()
    lr_defaults = ModelLRConfig()
    pattern_defaults = PatternConfig()
    persistence_defaults = PersistenceConfig()
    dashboard_defaults = DashboardConfig()

    parser = argparse.ArgumentParser(
        description="BTCUSDT spot kline predictor (UP/DOWN/FLAT) with online updates"
    )
    parser.add_argument("--symbol", default=data_defaults.symbol)
    parser.add_argument("--tfs", default=",".join(data_defaults.tfs))
    parser.add_argument("--ws-url", default=data_defaults.ws_url)
    parser.add_argument(
        "--include-unconfirmed",
        action="store_true",
        help="Use interim candles before close",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=data_defaults.reconnect_delay,
    )
    parser.add_argument(
        "--ping-interval",
        type=float,
        default=data_defaults.ping_interval,
    )
    parser.add_argument("--history-limit", type=int, default=rest_defaults.history_limit)
    parser.add_argument("--no-history", action="store_true")

    parser.add_argument("--lookback", type=int, default=feature_defaults.lookback)

    parser.add_argument(
        "--flat-max-prob", type=float, default=decision_defaults.flat_max_prob
    )
    parser.add_argument(
        "--flat-max-delta", type=float, default=decision_defaults.flat_max_delta
    )
    parser.add_argument("--flat-bps", type=float, default=training_defaults.flat_bps)
    parser.add_argument(
        "--flat-update-weight",
        type=float,
        default=training_defaults.flat_update_weight,
    )
    parser.add_argument(
        "--reward-dir-correct",
        type=float,
        default=training_defaults.reward_dir_correct,
    )
    parser.add_argument(
        "--reward-dir-wrong",
        type=float,
        default=training_defaults.reward_dir_wrong,
    )
    parser.add_argument(
        "--reward-flat-correct",
        type=float,
        default=training_defaults.reward_flat_correct,
    )
    parser.add_argument(
        "--reward-flat-wrong",
        type=float,
        default=training_defaults.reward_flat_wrong,
    )
    parser.add_argument(
        "--flat-penalty",
        type=float,
        default=training_defaults.flat_penalty,
    )
    parser.add_argument(
        "--class-balance-strength",
        type=float,
        default=training_defaults.class_balance_strength,
    )
    parser.add_argument(
        "--class-balance-ema",
        type=float,
        default=training_defaults.class_balance_ema,
    )
    parser.add_argument(
        "--class-balance-min",
        type=float,
        default=training_defaults.class_balance_min,
    )
    parser.add_argument(
        "--class-balance-max",
        type=float,
        default=training_defaults.class_balance_max,
    )
    parser.add_argument(
        "--class-balance-floor",
        type=float,
        default=training_defaults.class_balance_floor,
    )
    parser.add_argument("--temp-init", type=float, default=training_defaults.temp_init)
    parser.add_argument("--temp-min", type=float, default=training_defaults.temp_min)
    parser.add_argument("--temp-max", type=float, default=training_defaults.temp_max)
    parser.add_argument("--temp-lr", type=float, default=training_defaults.temp_lr)
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=training_defaults.calibration_bins,
    )
    parser.add_argument(
        "--perf-lr-gain",
        type=float,
        default=training_defaults.perf_lr_gain,
    )
    parser.add_argument(
        "--perf-lr-min-mult",
        type=float,
        default=training_defaults.perf_lr_min_mult,
    )
    parser.add_argument(
        "--perf-lr-max-mult",
        type=float,
        default=training_defaults.perf_lr_max_mult,
    )
    parser.add_argument(
        "--perf-lr-baseline",
        type=float,
        default=training_defaults.perf_lr_baseline,
    )
    parser.add_argument(
        "--perf-lr-min-samples",
        type=int,
        default=training_defaults.perf_lr_min_samples,
    )

    parser.add_argument(
        "--init-mode",
        default=ModelInitConfig().init_mode,
        choices=["heuristic", "random"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logit-scale", type=float, default=3.0)
    parser.add_argument("--weight-clip", type=float, default=5.0)

    parser.add_argument("--ema-decay", type=float, default=training_defaults.ema_decay)
    parser.add_argument("--anchor-decay", type=float, default=training_defaults.anchor_decay)
    parser.add_argument(
        "--anchor-lambda-base", type=float, default=training_defaults.anchor_lambda_base
    )
    parser.add_argument("--anchor-gain", type=float, default=training_defaults.anchor_gain)
    parser.add_argument("--lr-gain", type=float, default=training_defaults.lr_gain)
    parser.add_argument("--lr-min-mult", type=float, default=training_defaults.lr_min_mult)
    parser.add_argument("--lr-max-mult", type=float, default=training_defaults.lr_max_mult)
    parser.add_argument(
        "--anchor-min-mult", type=float, default=training_defaults.anchor_min_mult
    )
    parser.add_argument(
        "--anchor-max-mult", type=float, default=training_defaults.anchor_max_mult
    )

    parser.add_argument("--lr-trend", type=float, default=lr_defaults.lr_trend)
    parser.add_argument("--lr-osc", type=float, default=lr_defaults.lr_osc)
    parser.add_argument("--lr-vol", type=float, default=lr_defaults.lr_vol)

    parser.add_argument(
        "--pattern-db", default=str(pattern_defaults.db_path), help="Path to pattern SQLite db"
    )
    parser.add_argument(
        "--pattern-ema-decay", type=float, default=pattern_defaults.ema_decay
    )
    parser.add_argument(
        "--event-ttl-days", type=int, default=pattern_defaults.event_ttl_days
    )
    parser.add_argument("--max-events", type=int, default=pattern_defaults.max_events)
    parser.add_argument("--max-patterns", type=int, default=pattern_defaults.max_patterns)
    parser.add_argument("--support-k", type=int, default=pattern_defaults.support_k)
    parser.add_argument(
        "--recency-tau-hours", type=float, default=pattern_defaults.recency_tau_hours
    )
    parser.add_argument(
        "--anti-min-support", type=int, default=pattern_defaults.anti_min_support
    )
    parser.add_argument(
        "--anti-win-threshold", type=float, default=pattern_defaults.anti_win_threshold
    )
    parser.add_argument(
        "--anti-trust-threshold", type=float, default=pattern_defaults.anti_trust_threshold
    )
    parser.add_argument(
        "--maintenance-seconds", type=int, default=pattern_defaults.maintenance_seconds
    )

    parser.add_argument("--state-db", default=str(persistence_defaults.state_db))
    parser.add_argument(
        "--autosave-seconds", type=int, default=persistence_defaults.autosave_seconds
    )
    parser.add_argument(
        "--autosave-updates", type=int, default=persistence_defaults.autosave_updates
    )

    parser.add_argument(
        "--save-jsonl-dir",
        default=None,
        help="Directory for JSONL output (candles.jsonl, predictions.jsonl, facts.jsonl, updates.jsonl)",
    )
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--dashboard-host", default=dashboard_defaults.host)
    parser.add_argument("--dashboard-port", type=int, default=dashboard_defaults.port)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--analysis-inline",
        action="store_true",
        help="Run periodic analysis loop inside the main process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print JAX backend/devices and exit",
    )

    return parser


async def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info("JAX backend=%s devices=%s", jax.default_backend(), jax.devices())
    if args.dry_run:
        return

    data_config = DataConfig(
        symbol=args.symbol,
        tfs=parse_tfs(args.tfs),
        ws_url=args.ws_url,
        use_confirmed_only=not args.include_unconfirmed,
        reconnect_delay=args.reconnect_delay,
        ping_interval=args.ping_interval,
    )
    rest_config = RestConfig(
        history_limit=args.history_limit,
        enable_history=not args.no_history,
    )
    feature_config = FeatureConfig(
        lookback=args.lookback,
    )
    decision_config = DecisionConfig(
        flat_max_prob=args.flat_max_prob,
        flat_max_delta=args.flat_max_delta,
    )
    model_init = ModelInitConfig(
        init_mode=args.init_mode,
        seed=args.seed,
        logit_scale=args.logit_scale,
        weight_clip=args.weight_clip,
    )
    training = TrainingConfig(
        ema_decay=args.ema_decay,
        anchor_decay=args.anchor_decay,
        anchor_lambda_base=args.anchor_lambda_base,
        anchor_gain=args.anchor_gain,
        lr_gain=args.lr_gain,
        lr_min_mult=args.lr_min_mult,
        lr_max_mult=args.lr_max_mult,
        anchor_min_mult=args.anchor_min_mult,
        anchor_max_mult=args.anchor_max_mult,
        flat_bps=args.flat_bps,
        flat_update_weight=args.flat_update_weight,
        reward_dir_correct=args.reward_dir_correct,
        reward_dir_wrong=args.reward_dir_wrong,
        reward_flat_correct=args.reward_flat_correct,
        reward_flat_wrong=args.reward_flat_wrong,
        flat_penalty=args.flat_penalty,
        class_balance_strength=args.class_balance_strength,
        class_balance_ema=args.class_balance_ema,
        class_balance_min=args.class_balance_min,
        class_balance_max=args.class_balance_max,
        class_balance_floor=args.class_balance_floor,
        temp_init=args.temp_init,
        temp_min=args.temp_min,
        temp_max=args.temp_max,
        temp_lr=args.temp_lr,
        calibration_bins=args.calibration_bins,
        perf_lr_gain=args.perf_lr_gain,
        perf_lr_min_mult=args.perf_lr_min_mult,
        perf_lr_max_mult=args.perf_lr_max_mult,
        perf_lr_baseline=args.perf_lr_baseline,
        perf_lr_min_samples=args.perf_lr_min_samples,
    )
    lrs = ModelLRConfig(
        lr_trend=args.lr_trend,
        lr_osc=args.lr_osc,
        lr_vol=args.lr_vol,
    )
    patterns = PatternConfig(
        db_path=Path(args.pattern_db),
        ema_decay=args.pattern_ema_decay,
        event_ttl_days=args.event_ttl_days,
        max_events=args.max_events,
        max_patterns=args.max_patterns,
        support_k=args.support_k,
        recency_tau_hours=args.recency_tau_hours,
        anti_min_support=args.anti_min_support,
        anti_win_threshold=args.anti_win_threshold,
        anti_trust_threshold=args.anti_trust_threshold,
        maintenance_seconds=args.maintenance_seconds,
    )
    persistence = PersistenceConfig(
        state_db=Path(args.state_db),
        autosave_seconds=args.autosave_seconds,
        autosave_updates=args.autosave_updates,
    )
    output = OutputConfig(out_dir=Path(args.save_jsonl_dir) if args.save_jsonl_dir else None)
    dashboard = DashboardConfig(
        enabled=args.dashboard,
        host=args.dashboard_host,
        port=args.dashboard_port,
        history_size=200,
    )
    logging_config = LoggingConfig(level=args.log_level)

    app_config = AppConfig(
        data=data_config,
        rest=rest_config,
        features=feature_config,
        decision=decision_config,
        model_init=model_init,
        training=training,
        lrs=lrs,
        patterns=patterns,
        persistence=persistence,
        output=output,
        dashboard=dashboard,
        logging=logging_config,
    )

    patterns.db_path.parent.mkdir(parents=True, exist_ok=True)
    persistence.state_db.parent.mkdir(parents=True, exist_ok=True)
    pattern_store = PatternStore(str(patterns.db_path), patterns)
    state_store = ModelStateStore(str(persistence.state_db))
    engine = MultiTimeframeEngine(
        tfs=data_config.tfs,
        feature_config=feature_config,
        model_init=model_init,
        training=training,
        decision=decision_config,
        lrs=lrs,
        pattern_store=pattern_store,
        state_store=state_store,
    )
    engine.load_states()

    recorder = JsonlRecorder(output.out_dir) if output.out_dir else None
    runtime_state = RuntimeState(history_size=dashboard.history_size)
    runtime_state.update_jax_info(
        jax.default_backend(), [str(device) for device in jax.devices()]
    )
    config_manager = ConfigManager(feature_config, decision_config, training, lrs, patterns)

    analysis = None
    analysis_task = None
    if args.analysis_inline:
        analysis = PeriodicAnalyzer(
            AnalysisConfig(),
            recorder.record_analysis if recorder else None,
            pattern_store=pattern_store,
        )
        analysis_task = asyncio.create_task(analysis.run())

    def record_update(update):
        if recorder is not None:
            recorder.record_update(update)
        if analysis is not None:
            analysis.observe_update(update)

    dashboard_runner = None
    if dashboard.enabled:
        if start_dashboard is None:
            logging.warning("Dashboard requested but aiohttp is unavailable.")
        else:
            dashboard_runner = await start_dashboard(
                runtime_state, config_manager, pattern_store, dashboard.host, dashboard.port
            )
            logging.info("Dashboard started at http://%s:%d", dashboard.host, dashboard.port)

    if rest_config.enable_history:
        for tf in data_config.tfs:
            try:
                required = engine.buffers[tf].maxlen
                history = fetch_klines(
                    data_config.symbol,
                    tf,
                    limit=max(rest_config.history_limit, required + 2),
                    rest_config=rest_config,
                )
                if history:
                    engine.warm_start(tf, history)
                    engine.bootstrap_predictions(tf, recorder.record_prediction if recorder else None, runtime_state)
            except Exception as exc:
                logging.warning("REST warmup failed for tf=%s: %s (continue WS only)", tf, exc)

    try:
        async for candle in stream_candles(data_config):
            runtime_state.update_ws_status(True)
            processed = engine.process_candle(
                candle,
                recorder.record_prediction if recorder else None,
                recorder.record_fact if recorder else None,
                record_update,
                runtime_state,
                autosave_seconds=persistence.autosave_seconds,
                autosave_updates=persistence.autosave_updates,
            )
            if processed:
                runtime_state.update_candle(candle)
                if recorder is not None:
                    recorder.record_candle(candle)
    finally:
        if recorder is not None:
            recorder.close()
        if analysis is not None:
            analysis.stop()
        if analysis_task is not None:
            analysis_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await analysis_task
        if dashboard_runner is not None:
            await dashboard_runner.cleanup()
        pattern_store.close()
        state_store.close()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        logging.info("Stopped")


if __name__ == "__main__":
    main()
