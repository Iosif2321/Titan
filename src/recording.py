import json
from pathlib import Path
from typing import Any, Dict

from .types import Candle, Fact, Prediction, UpdateEvent


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._fp = path.open("a", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        self._fp.write(payload + "\n")
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()


class JsonlRecorder:
    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._candles = JsonlWriter(directory / "candles.jsonl")
        self._predictions = JsonlWriter(directory / "predictions.jsonl")
        self._facts = JsonlWriter(directory / "facts.jsonl")
        self._updates = JsonlWriter(directory / "updates.jsonl")
        self._analysis = JsonlWriter(directory / "analysis.jsonl")

    def record_candle(self, candle: Candle) -> None:
        self._candles.write(
            {
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
        )

    def record_prediction(self, prediction: Prediction) -> None:
        payload = {
            "ts": prediction.ts,
            "tf": prediction.tf,
            "model_id": prediction.model_id,
            "model_type": prediction.model_type,
            "candle_ts": prediction.candle_ts,
            "target_ts": prediction.target_ts,
            "logits_up": prediction.logits_up,
            "logits_down": prediction.logits_down,
            "p_up": prediction.p_up,
            "p_down": prediction.p_down,
            "dir": prediction.direction.value,
            "confidence": prediction.confidence,
            "used_ema": prediction.used_ema,
            "context_key_used": prediction.context_key_used,
            "decision_key_used": prediction.decision_key_used,
            "trust_ctx": prediction.trust_ctx,
            "trust_dec": prediction.trust_dec,
            "prior_ctx": prediction.prior_ctx,
            "prior_win_dec": prediction.prior_win_dec,
            "flat_thresholds": prediction.flat_thresholds,
            "notes": prediction.notes,
            "meta": prediction.meta,
        }
        if isinstance(prediction.meta, dict):
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
                if key in prediction.meta:
                    payload[key] = prediction.meta[key]
        self._predictions.write(payload)

    def record_fact(self, fact: Fact) -> None:
        self._facts.write(
            {
                "tf": fact.tf,
                "prev_ts": fact.prev_ts,
                "curr_ts": fact.curr_ts,
                "close_prev": fact.close_prev,
                "close_curr": fact.close_curr,
                "ret_bps": fact.ret_bps,
                "abs_ret_bps": abs(fact.ret_bps),
                "fact_flat_bps": fact.fact_flat_bps,
                "fact_dir": fact.direction.value,
            }
        )

    def record_update(self, update: UpdateEvent) -> None:
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
            "calib_a": update.calib_a,
            "calib_b": update.calib_b,
            "calib_n": update.calib_n,
            "p_up_raw": update.p_up_raw,
            "p_down_raw": update.p_down_raw,
            "p_up_cal": update.p_up_cal,
            "p_down_cal": update.p_down_cal,
            "margin_raw": update.margin_raw,
            "margin_cal": update.margin_cal,
            "close_prev": update.close_prev,
            "close_curr": update.close_curr,
            "delta": update.delta,
            "features": update.features,
            "fact_flat_bps": update.fact_flat_bps,
            "abs_ret_bps": update.abs_ret_bps,
            "x_ret": update.x_ret,
            "pred_flat_delta": update.pred_flat_delta,
            "reward_raw": update.reward_raw,
            "micro_share": update.micro_share,
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
        self._updates.write(payload)

    def record_analysis(self, payload: Dict[str, Any]) -> None:
        self._analysis.write(payload)

    def close(self) -> None:
        self._candles.close()
        self._predictions.close()
        self._facts.close()
        self._updates.close()
        self._analysis.close()
