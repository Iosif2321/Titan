from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, Deque, Dict, Optional

from .types import Candle, Fact, Prediction, UpdateEvent


class RuntimeState:
    def __init__(self, stream_queue_size: int = 100, history_size: int = 200) -> None:
        self.latest_candles: Dict[str, Candle] = {}
        self.latest_predictions: Dict[str, Prediction] = {}
        self.latest_facts: Dict[str, Fact] = {}
        self.latest_updates: Dict[str, UpdateEvent] = {}
        self.metrics: Dict[str, Any] = {}
        self.model_stats: Dict[str, Any] = {}
        self.ws_status: Dict[str, Any] = {"connected": False, "last_error": None}
        self.jax_info: Dict[str, Any] = {"backend": None, "devices": []}
        self.recent_events: Deque[Dict[str, Any]] = deque(maxlen=history_size)
        self._queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=stream_queue_size)

    def _push_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        event = {"type": event_type, "data": payload}
        self.recent_events.appendleft(event)
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            pass

    def update_jax_info(self, backend: str, devices: list[str]) -> None:
        self.jax_info = {"backend": backend, "devices": devices}
        self._push_event("jax", self.jax_info)

    def update_candle(self, candle: Candle) -> None:
        self.latest_candles[candle.tf] = candle
        self._push_event("candle", _candle_to_dict(candle))

    def update_prediction(self, prediction: Prediction) -> None:
        key = f"{prediction.tf}:{prediction.model_id}"
        self.latest_predictions[key] = prediction
        self._push_event("prediction", _prediction_to_dict(prediction))

    def update_fact(self, fact: Fact) -> None:
        self.latest_facts[fact.tf] = fact
        self._push_event("fact", _fact_to_dict(fact))

    def update_update(self, update: UpdateEvent) -> None:
        key = f"{update.tf}:{update.model_id}"
        self.latest_updates[key] = update
        self._push_event("update", _update_to_dict(update))

    def update_metrics(self, model_key: str, metrics: Dict[str, Any]) -> None:
        self.metrics[model_key] = metrics
        payload = {"model_key": model_key, **metrics}
        self._push_event("metrics", payload)

    def update_model_stats(self, model_key: str, stats: Dict[str, Any]) -> None:
        self.model_stats[model_key] = stats

    def update_ws_status(self, connected: bool, last_error: Optional[str] = None) -> None:
        self.ws_status = {"connected": connected, "last_error": last_error}
        self._push_event("ws_status", self.ws_status)

    async def next_event(self) -> Dict[str, Any]:
        return await self._queue.get()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "candles": {k: _candle_to_dict(v) for k, v in self.latest_candles.items()},
            "predictions": {
                k: _prediction_to_dict(v) for k, v in self.latest_predictions.items()
            },
            "facts": {k: _fact_to_dict(v) for k, v in self.latest_facts.items()},
            "updates": {k: _update_to_dict(v) for k, v in self.latest_updates.items()},
            "metrics": self.metrics,
            "ws": self.ws_status,
            "jax": self.jax_info,
            "recent_events": list(self.recent_events),
        }

    def models_snapshot(self) -> Dict[str, Any]:
        return {"models": list(self.model_stats.values())}


def _candle_to_dict(candle: Optional[Candle]) -> Optional[Dict[str, Any]]:
    if candle is None:
        return None
    return {
        "ts_start": candle.start_ts,
        "ts_end": candle.end_ts,
        "o": candle.open,
        "h": candle.high,
        "l": candle.low,
        "c": candle.close,
        "volume": candle.volume,
        "confirmed": candle.confirmed,
        "tf": candle.tf,
    }


def _prediction_to_dict(pred: Optional[Prediction]) -> Optional[Dict[str, Any]]:
    if pred is None:
        return None
    payload = {
        "ts": pred.ts,
        "tf": pred.tf,
        "model_id": pred.model_id,
        "model_type": pred.model_type,
        "candle_ts": pred.candle_ts,
        "target_ts": pred.target_ts,
        "logits_up": pred.logits_up,
        "logits_down": pred.logits_down,
        "p_up": pred.p_up,
        "p_down": pred.p_down,
        "direction": pred.direction.value,
        "confidence": pred.confidence,
        "used_ema": pred.used_ema,
        "context_key_used": pred.context_key_used,
        "decision_key_used": pred.decision_key_used,
        "trust_ctx": pred.trust_ctx,
        "trust_dec": pred.trust_dec,
        "prior_ctx": pred.prior_ctx,
        "prior_win_dec": pred.prior_win_dec,
        "flat_thresholds": pred.flat_thresholds,
        "notes": pred.notes,
        "meta": pred.meta,
    }
    if isinstance(pred.meta, dict):
        for key in (
            "p_up_raw",
            "p_down_raw",
            "p_up_cal",
            "p_down_cal",
            "conf_raw",
            "conf_cal",
            "calib_a",
            "calib_b",
            "calib_n",
        ):
            if key in pred.meta:
                payload[key] = pred.meta[key]
    return payload


def _fact_to_dict(fact: Optional[Fact]) -> Optional[Dict[str, Any]]:
    if fact is None:
        return None
    return {
        "tf": fact.tf,
        "prev_ts": fact.prev_ts,
        "curr_ts": fact.curr_ts,
        "close_prev": fact.close_prev,
        "close_curr": fact.close_curr,
        "ret_bps": fact.ret_bps,
        "fact_flat_bps": fact.fact_flat_bps,
        "direction": fact.direction.value,
    }


def _update_to_dict(update: Optional[UpdateEvent]) -> Optional[Dict[str, Any]]:
    if update is None:
        return None
    payload = {
        "ts": update.ts,
        "tf": update.tf,
        "model_id": update.model_id,
        "model_type": update.model_type,
        "target_ts": update.target_ts,
        "candle_ts": update.candle_ts,
        "pred_dir": update.pred_dir.value,
        "pred_conf": update.pred_conf,
        "fact_dir": update.fact_dir.value,
        "ret_bps": update.ret_bps,
        "reward": update.reward,
        "loss_task": update.loss_task,
        "loss_total": update.loss_total,
        "lr_eff": update.lr_eff,
        "anchor_lambda_eff": update.anchor_lambda_eff,
        "weight_norms": update.weight_norms,
        "anchor_update_applied": update.anchor_update_applied,
        "notes": update.notes,
        "meta": update.meta,
    }
    if isinstance(update.meta, dict):
        for key in (
            "p_up_raw",
            "p_down_raw",
            "p_up_cal",
            "p_down_cal",
            "conf_raw",
            "conf_cal",
            "calib_a",
            "calib_b",
            "calib_n",
        ):
            if key in update.meta:
                payload[key] = update.meta[key]
    return payload
