import asyncio
import json
import os
import time
from typing import Dict, Optional

from titan.core.adapters.pattern import PatternAdjuster, PatternModelAdjuster, PatternReader
from titan.core.adapters.session import SessionAdapter
from titan.core.online import create_online_adapter
from titan.core.backtest import BacktestStats, DetailWriter, _evaluate, _model_decision, _tune_weights
from titan.core.calibration import OnlineCalibrator
from titan.core.config import ConfigStore
from titan.core.data.bybit_rest import fetch_klines, interval_to_ms
from titan.core.data.bybit_ws import BybitSpotWebSocket
from titan.core.data.store import CandleStore
from titan.core.features.stream import FeatureStream, build_conditions
from titan.core.models.heuristic import Oscillator, TrendVIC, VolumeMetrix
from titan.core.ensemble import Ensemble
from titan.core.patterns import PatternExperience, PatternStore
from titan.core.regime import RegimeDetector
from titan.core.state_store import StateStore
from titan.core.types import Decision, PatternContext, PredictionRecord
from titan.core.weights import WeightManager


def _apply_overrides(config_store: ConfigStore, overrides: Dict[str, object]) -> None:
    for key, value in overrides.items():
        config_store.set(key, value)


async def _run_blocking(func, *args):
    if hasattr(asyncio, "to_thread"):
        return await asyncio.to_thread(func, *args)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, func, *args)


async def live_loop(
    symbol: str,
    interval: str,
    db_path: str,
    out_dir: str,
    max_predictions: Optional[int],
    tune_weights: bool,
    store_candles: bool,
    overrides: Optional[Dict[str, object]] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    state_store = StateStore(db_path)
    config_store = ConfigStore(state_store)
    config_store.ensure_defaults()
    if overrides:
        _apply_overrides(config_store, overrides)

    weight_manager = WeightManager(state_store)
    # Sprint 12: Pass config for config-driven pattern behavior
    pattern_store = PatternStore(db_path, config=config_store)
    candle_store = CandleStore(db_path) if store_candles else None

    # Sprint 13: Create pattern experience and adjuster
    pattern_experience = PatternExperience(pattern_store)
    pattern_adjuster = PatternAdjuster(
        pattern_experience, config_store, model_name="ENSEMBLE"
    )
    pattern_reader = PatternReader(pattern_store, config_store)
    pattern_model_adjuster = PatternModelAdjuster(pattern_reader, config_store)
    model_adjuster_enabled = bool(
        config_store.get("pattern.model_adjuster_enabled", False)
    )

    # Sprint 17: Create session adapter for per-session adaptation
    session_db_path = db_path.replace(".db", "_session.db") if db_path.endswith(".db") else db_path + "_session"
    session_adapter_enabled = bool(config_store.get("session_adapter.enabled", True))
    session_adapter = SessionAdapter(
        session_db_path,
        config_store,
        enabled=session_adapter_enabled,
    )
    if session_adapter_enabled:
        print(f"[Live] Session Adapter enabled (Thompson Sampling)")

    # Sprint 20: Create online learning adapter
    online_enabled = bool(config_store.get("online.enabled", True))
    online_adapter = create_online_adapter(config_store, enabled=online_enabled)
    if online_enabled:
        print(f"[Live] Online Learning enabled (SGD + RMSProp)")

    # Create regime detector for session weights
    regime_detector = RegimeDetector(config_store)

    models = [
        TrendVIC(config_store),
        Oscillator(config_store),
        VolumeMetrix(config_store),
    ]

    feature_stream = FeatureStream(config_store)
    ensemble = Ensemble(config_store, weight_manager, pattern_adjuster=pattern_adjuster)
    calibrator = OnlineCalibrator(config_store, state_store)

    stats = BacktestStats([model.name for model in models])
    details_path = os.path.join(out_dir, "predictions.jsonl")
    detail_writer = DetailWriter(details_path)

    pending: Optional[PredictionRecord] = None
    evaluated = 0
    processed_candles = 0
    gap_count = 0
    backfilled_candles = 0
    duplicate_count = 0
    out_of_order_count = 0
    interval_sec = int(interval_to_ms(interval) / 1000)
    last_ts: Optional[int] = None

    ws = BybitSpotWebSocket(symbol, interval=interval)

    async def _process_candle(candle) -> bool:
        nonlocal pending, evaluated, processed_candles

        processed_candles += 1

        if candle_store is not None:
            candle_store.insert_many(symbol, [candle])

        features = feature_stream.update(candle)
        if features is None:
            return False

        if pending is not None:
            outcome = _evaluate(pending, candle.close)
            raw_confidence = max(pending.decision.prob_up, pending.decision.prob_down)
            calibrator.update(
                raw_confidence,
                pending.decision.direction == outcome.actual_direction,
            )
            model_decisions: Dict[str, str] = {}

            for output in pending.outputs:
                model_direction = _model_decision(output, config_store)
                model_decisions[output.model_name] = model_direction
                event = {
                    "price": pending.price,
                    "model_state": output.state,
                    "forecast": {
                        "prob_up": output.prob_up,
                        "prob_down": output.prob_down,
                        "direction": model_direction,
                    },
                    "metrics": output.metrics,
                    "outcome": {
                        "actual_direction": outcome.actual_direction,
                        "price_delta": outcome.price_delta,
                        "return_pct": outcome.return_pct,
                        "hit": model_direction == outcome.actual_direction,
                    },
                }
                # Sprint 12: Pass features_snapshot for 100% storage
                pattern_store.record_usage(
                    pending.pattern_id,
                    output.model_name,
                    event,
                    event_ts=pending.ts,
                    features_snapshot=pending.features,
                )

            ensemble_event = {
                "price": pending.price,
                "model_state": {"ensemble": True},
                "forecast": {
                    "prob_up": pending.decision.prob_up,
                    "prob_down": pending.decision.prob_down,
                    "direction": pending.decision.direction,
                },
                "metrics": {},
                "outcome": {
                    "actual_direction": outcome.actual_direction,
                    "price_delta": outcome.price_delta,
                    "return_pct": outcome.return_pct,
                    "hit": pending.decision.direction == outcome.actual_direction,
                },
            }
            pattern_store.record_usage(
                pending.pattern_id,
                "ENSEMBLE",
                ensemble_event,
                event_ts=pending.ts,
                features_snapshot=pending.features,
            )

            stats.update(pending, outcome, model_decisions)
            detail_writer.write(pending, outcome, model_decisions)
            evaluated += 1

            # Sprint 17: Record outcome to session adapter for learning
            if session_adapter_enabled:
                session = session_adapter.get_session(pending.ts)
                pending_regime = regime_detector.detect(pending.features)
                is_hit = pending.decision.direction == outcome.actual_direction
                for output in pending.outputs:
                    model_dir = model_decisions[output.model_name]
                    model_hit = model_dir == outcome.actual_direction
                    session_adapter.record_outcome(
                        session=session,
                        model=output.model_name,
                        regime=pending_regime,
                        hit=model_hit,
                        conf=max(output.prob_up, output.prob_down),
                        return_pct=outcome.return_pct,
                        ts=pending.ts,
                    )
                # Also record ensemble outcome
                session_adapter.record_outcome(
                    session=session,
                    model="ENSEMBLE",
                    regime=pending_regime,
                    hit=is_hit,
                    conf=pending.decision.confidence,
                    return_pct=outcome.return_pct,
                    ts=pending.ts,
                )

            # Sprint 20: Record outcome to online learning adapter
            if online_enabled:
                model_confs = {
                    output.model_name: max(output.prob_up, output.prob_down)
                    for output in pending.outputs
                }
                online_adapter.record_outcome(
                    model_decisions=model_decisions,
                    actual_direction=outcome.actual_direction,
                    model_confs=model_confs,
                    ensemble_hit=pending.decision.direction == outcome.actual_direction,
                    return_pct=outcome.return_pct,
                )

            if tune_weights:
                tuned = _tune_weights(stats, config_store)
                weight_manager.set_model_weights(tuned)

            if max_predictions is not None and evaluated >= max_predictions:
                return True

        # Get pattern_id first so we can pass it to ensemble.decide()
        # Sprint 12: Pass ts for extended conditions (hour, session, day_of_week)
        conditions = build_conditions(features, config_store, ts=candle.ts)
        pattern_id = pattern_store.get_or_create(conditions, ts=candle.ts)

        pattern_contexts: Dict[str, Optional[PatternContext]] = {}
        for model in models:
            pattern_contexts[model.name] = pattern_reader.build_context(
                pattern_id,
                features,
                ts=candle.ts,
                model_name=model.name,
                conditions=conditions,
            )

        outputs = [
            model.predict(features, pattern_context=pattern_contexts.get(model.name))
            for model in models
        ]
        if model_adjuster_enabled:
            outputs = [
                pattern_model_adjuster.adjust_model_output(
                    output, pattern_id, conditions=conditions, ts=candle.ts
                )
                for output in outputs
            ]

        # Sprint 17: Get session-specific weights if enabled
        override_weights = None
        current_session = None
        session_params = None
        if session_adapter_enabled:
            current_session = session_adapter.get_session(candle.ts)
            current_regime = regime_detector.detect(features)
            override_weights = session_adapter.get_weights(current_session, current_regime)
            # BUG FIX: Actually use the Thompson Sampling params
            session_params = session_adapter.get_all_params(current_session)
        elif online_enabled:
            # BUG FIX: Use online-learned weights when session adapter is disabled
            online_weights = online_adapter.get_weights()
            if online_weights:
                override_weights = online_weights

        # Sprint 13: Pass pattern_id for experience-based adjustment
        decision = ensemble.decide(
            outputs, features, ts=candle.ts, pattern_id=pattern_id,
            override_weights=override_weights,
            override_params=session_params,
        )

        # Apply calibration: first online calibrator, then session-specific
        calibrated_conf = calibrator.calibrate(decision.confidence)
        if session_adapter_enabled and current_session:
            calibrated_conf = session_adapter.calibrate_confidence(current_session, calibrated_conf)

        decision = Decision(
            direction=decision.direction,
            confidence=calibrated_conf,
            prob_up=decision.prob_up,
            prob_down=decision.prob_down,
        )

        pending = PredictionRecord(
            ts=candle.ts,
            price=candle.close,
            pattern_id=pattern_id,
            features=features,
            outputs=outputs,
            decision=decision,
        )

        print(
            f"{candle.iso_time()} decision={decision.direction} "
            f"conf={decision.confidence:.3f}"
        )
        return False

    try:
        async for candle in ws.stream():
            if last_ts is not None:
                if candle.ts <= last_ts:
                    if candle.ts == last_ts:
                        duplicate_count += 1
                    else:
                        out_of_order_count += 1
                    continue

                if candle.ts > last_ts + interval_sec:
                    gap_count += 1
                    gap_start = last_ts + interval_sec
                    gap_end = candle.ts - interval_sec
                    try:
                        backfilled = await _run_blocking(
                            fetch_klines, symbol, interval, gap_start, gap_end
                        )
                    except Exception as exc:
                        print(f"Backfill failed: {exc}")
                        backfilled = []

                    if backfilled:
                        backfilled_candles += len(backfilled)

                    for filled in backfilled:
                        if last_ts is not None and filled.ts <= last_ts:
                            continue
                        last_ts = filled.ts
                        if await _process_candle(filled):
                            return

            last_ts = candle.ts
            if await _process_candle(candle):
                break
    finally:
        detail_writer.close()

        summary = stats.summary()
        summary["online_calibration"] = calibrator.summary()
        summary["run_meta"] = {
            "source": "bybit_ws",
            "symbol": symbol,
            "interval": interval,
            "db_path": db_path,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "max_predictions": max_predictions,
            "tune_weights": tune_weights,
            "store_candles": store_candles,
            "data_quality": {
                "candles_processed": processed_candles,
                "gaps_detected": gap_count,
                "backfilled_candles": backfilled_candles,
                "ws_duplicates": duplicate_count,
                "ws_out_of_order": out_of_order_count,
            },
        }
        if overrides:
            summary["run_meta"]["config_overrides"] = overrides

        summary["weights"] = weight_manager.get_model_weights()

        # Sprint 12: Run pattern lifecycle maintenance
        lifecycle = pattern_store.run_lifecycle_maintenance()
        summary["pattern_lifecycle"] = lifecycle

        # Sprint 17: Session Adapter summary
        if session_adapter_enabled:
            summary["session_adapter"] = {
                "enabled": True,
                "sessions": {
                    session: session_adapter.get_session_summary(session)
                    for session in ["ASIA", "EUROPE", "US"]
                },
            }
            session_adapter.close()
        else:
            summary["session_adapter"] = {"enabled": False}

        # Sprint 20: Online Learning summary
        summary["online_learning"] = online_adapter.summary()

        summary_path = os.path.join(out_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=True, indent=2))


def run_live(
    symbol: str,
    interval: str,
    db_path: str,
    out_dir: str,
    max_predictions: Optional[int],
    tune_weights: bool,
    store_candles: bool,
    overrides: Optional[Dict[str, object]] = None,
) -> None:
    asyncio.run(
        live_loop(
            symbol=symbol,
            interval=interval,
            db_path=db_path,
            out_dir=out_dir,
            max_predictions=max_predictions,
            tune_weights=tune_weights,
            store_candles=store_candles,
            overrides=overrides,
        )
    )
