import asyncio
import json
import os
import time
from typing import Dict, Optional

from titan.core.backtest import BacktestStats, DetailWriter, _evaluate, _model_decision, _tune_weights
from titan.core.config import ConfigStore
from titan.core.data.bybit_rest import fetch_klines, interval_to_ms
from titan.core.data.bybit_ws import BybitSpotWebSocket
from titan.core.data.store import CandleStore
from titan.core.features.stream import FeatureStream, build_conditions
from titan.core.models.heuristic import Oscillator, TrendVIC, VolumeMetrix
from titan.core.ensemble import Ensemble
from titan.core.patterns import PatternStore
from titan.core.state_store import StateStore
from titan.core.types import PredictionRecord
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
    pattern_store = PatternStore(db_path)
    candle_store = CandleStore(db_path) if store_candles else None

    models = [
        TrendVIC(config_store),
        Oscillator(config_store),
        VolumeMetrix(config_store),
    ]

    feature_stream = FeatureStream(config_store)
    ensemble = Ensemble(config_store, weight_manager)

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
            model_decisions: Dict[str, str] = {}

            for output in pending.outputs:
                model_direction = _model_decision(output, config_store)
                model_decisions[output.model_name] = model_direction
                event = {
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
                pattern_store.record_usage(
                    pending.pattern_id,
                    output.model_name,
                    event,
                    event_ts=pending.ts,
                )

            stats.update(pending, outcome, model_decisions)
            detail_writer.write(pending, outcome, model_decisions)
            evaluated += 1

            if tune_weights:
                tuned = _tune_weights(stats, config_store)
                weight_manager.set_model_weights(tuned)

            if max_predictions is not None and evaluated >= max_predictions:
                return True

        outputs = [model.predict(features) for model in models]
        decision = ensemble.decide(outputs)
        conditions = build_conditions(features, config_store)
        pattern_id = pattern_store.get_or_create(conditions)

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
