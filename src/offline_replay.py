from __future__ import annotations

import argparse
import csv
import json
import os
import logging
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .candle_source import BybitRestCandleSource, JsonlCandleSource
from .config import (
    DecisionConfig,
    FactConfig,
    FeatureConfig,
    ModelInitConfig,
    ModelConfig,
    ModelLRConfig,
    PatternConfig,
    PersistenceConfig,
    RewardConfig,
    RestConfig,
    TrainingConfig,
)
from .features import FeatureBuilder, MODEL_OSC, MODEL_TREND, MODEL_VOL
from .jax_utils import ensure_jax_backend
from .metrics import CalibrationMetrics
from .pattern_store import NullPatternStore, PatternStore
from .recording import JsonlRecorder
from .state_store import ModelStateStore
from .types import Candle, Direction, Prediction, UpdateEvent
from .utils import interval_to_ms, now_ms, parse_tfs

if TYPE_CHECKING:
    from .engine import MultiTimeframeEngine

CONFIDENT_WRONG_THRESHOLD = 0.8
MAX_CONFIDENT_EXAMPLES = 10
MIN_GUARD_SAMPLES = 20
MIN_DIR_SHARE = 0.02
MAX_DIR_SHARE = 0.98
CALIB_A_WARN = 0.20
CALIB_A_ERROR = 0.10
CALIB_B_WARN = 0.30
PRED_FLAT_WARN = 0.35
PRED_FLAT_ERROR = 0.50
OVERCONF_ECE_WARNING = 0.15
OVERCONF_ECE_ERROR = 0.20
OVERCONF_GAP_ERROR = 0.20
LOOKAHEAD_SAMPLES = 50
SMOKE_MAX_CANDLES = 500
TOP_FEATURES = 12


def _parse_ts(value: str) -> int:
    if value is None:
        raise ValueError("timestamp is required")
    raw = value.strip()
    if raw.isdigit():
        ts = int(raw)
        if ts < 1_000_000_000_000:
            ts *= 1000
        return ts
    text = raw.replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return int(dt.timestamp() * 1000)


def _iso_local(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0).isoformat()


def _resolve_time_range(
    start: Optional[str], end: Optional[str], minutes: Optional[int]
) -> Tuple[int, int]:
    if minutes is not None:
        end_ms = _parse_ts(end) if end else now_ms()
        start_ms = end_ms - int(minutes) * 60_000
        return start_ms, end_ms
    if not start or not end:
        raise ValueError("Provide --start/--end or --minutes")
    return _parse_ts(start), _parse_ts(end)


def _select_indices(candles: List[Candle], min_index: int, count: int) -> List[int]:
    length = len(candles)
    if length <= min_index + 1 or min_index >= length:
        return []
    indices = set()
    for idx in range(min_index, min(min_index + 5, length)):
        indices.add(idx)
    last_start = max(min_index, length - 5)
    for idx in range(last_start, length):
        indices.add(idx)
    if count > 0:
        available = length - min_index
        step = max(1, available // count)
        for i in range(count):
            idx = min_index + i * step
            if idx < length:
                indices.add(idx)
    vol_candidates: List[Tuple[float, int]] = []
    for idx in range(min_index + 1, length):
        prev = candles[idx - 1].close
        curr = candles[idx].close
        if prev > 0 and curr > 0:
            vol = abs(math.log(curr / prev))
            vol_candidates.append((vol, idx))
    if vol_candidates:
        vol_candidates.sort(key=lambda item: item[0], reverse=True)
        top_k = min(len(vol_candidates), max(5, count // 5))
        for _, idx in vol_candidates[:top_k]:
            indices.add(idx)
    return sorted(indices)


@dataclass
class ConfidentExample:
    ts: int
    tf: str
    model_id: str
    pred_dir: str
    fact_dir: str
    conf: float
    ret_bps: float
    close_prev: Optional[float] = None
    close_curr: Optional[float] = None
    delta: Optional[float] = None
    p_up_raw: Optional[float] = None
    p_down_raw: Optional[float] = None
    p_up_cal: Optional[float] = None
    p_down_cal: Optional[float] = None
    margin_raw: Optional[float] = None
    margin_cal: Optional[float] = None
    calib_a: Optional[float] = None
    calib_b: Optional[float] = None
    top_features: Optional[Dict[str, float]] = None


class OfflineStats:
    def __init__(
        self,
        bins: int,
        flat_max_delta: Optional[float] = None,
        feature_names_by_key: Optional[Dict[Tuple[str, str], List[str]]] = None,
        diagnostic: bool = False,
    ) -> None:
        self.total = 0
        self.correct = 0
        self.nonflat_total = 0
        self.nonflat_correct = 0
        self.nonflat_swapped_correct = 0
        self.flat_total = 0
        self.flat_correct = 0
        self.pred_flat_total = 0
        self.pred_flat_when_fact_nonflat = 0
        self.action_total = 0
        self.action_correct = 0
        self.pred_dir_counts = {d.value: 0 for d in Direction}
        self.fact_dir_counts = {d.value: 0 for d in Direction}
        self.confusion = {d.value: {f.value: 0 for f in Direction} for d in Direction}
        self.conf_sum = 0.0
        self.calibration = CalibrationMetrics(bins=bins, window=10_000)
        self.calib_a_history: List[float] = []
        self.calib_b_history: List[float] = []
        self.calib_n_history: List[int] = []
        self.collapse_step: Optional[int] = None
        self.flat_reasons: Dict[str, int] = {}
        self.confident_wrong = 0
        self.confident_wrong_examples: List[ConfidentExample] = []
        self.lr_eff_sum = 0.0
        self.anchor_lambda_sum = 0.0
        self.anchor_update_count = 0
        self.params_norm_sum = 0.0
        self.anchor_norm_sum = 0.0
        self.params_anchor_gap_sum = 0.0
        self.flat_max_delta = flat_max_delta
        self.feature_names_by_key = feature_names_by_key or {}
        self.diagnostic = diagnostic

    def observe(self, update: UpdateEvent) -> None:
        pred_dir = update.pred_dir.value
        fact_dir = update.fact_dir.value
        correct = pred_dir == fact_dir
        self.total += 1
        self.correct += int(correct)
        self.pred_dir_counts[pred_dir] += 1
        self.fact_dir_counts[fact_dir] += 1
        self.confusion[pred_dir][fact_dir] += 1
        if pred_dir == Direction.FLAT.value:
            self.pred_flat_total += 1
            if fact_dir != Direction.FLAT.value:
                self.pred_flat_when_fact_nonflat += 1
        else:
            self.action_total += 1
            self.action_correct += int(correct)
        if fact_dir == Direction.FLAT.value:
            self.flat_total += 1
            self.flat_correct += int(correct)
        else:
            self.nonflat_total += 1
            self.nonflat_correct += int(correct)
            swapped = pred_dir
            if pred_dir == Direction.UP.value:
                swapped = Direction.DOWN.value
            elif pred_dir == Direction.DOWN.value:
                swapped = Direction.UP.value
            self.nonflat_swapped_correct += int(swapped == fact_dir)
        if pred_dir != Direction.FLAT.value:
            self.calibration.update(update.pred_conf, correct, abs(update.ret_bps))
        if update.calib_a is not None:
            self.calib_a_history.append(float(update.calib_a))
            if update.calib_b is not None:
                self.calib_b_history.append(float(update.calib_b))
            if update.calib_n is not None:
                self.calib_n_history.append(int(update.calib_n))
            if update.calib_a <= CALIB_A_ERROR and self.collapse_step is None:
                self.collapse_step = self.total
        if pred_dir == Direction.FLAT.value:
            reason = _flat_reason(update, self.flat_max_delta)
            if reason:
                self.flat_reasons[reason] = self.flat_reasons.get(reason, 0) + 1
        self.conf_sum += update.pred_conf
        if update.pred_conf >= CONFIDENT_WRONG_THRESHOLD and not correct:
            self.confident_wrong += 1
            self._add_confident_example(update)
        self.lr_eff_sum += update.lr_eff
        self.anchor_lambda_sum += update.anchor_lambda_eff
        self.anchor_update_count += int(update.anchor_update_applied)
        params_norm = _norm_value(update.weight_norms.get("params"))
        anchor_norm = _norm_value(update.weight_norms.get("anchor"))
        self.params_norm_sum += params_norm
        self.anchor_norm_sum += anchor_norm
        self.params_anchor_gap_sum += abs(params_norm - anchor_norm)

    def _add_confident_example(self, update: UpdateEvent) -> None:
        top_features = self._extract_top_features(update)
        example = ConfidentExample(
            ts=update.ts,
            tf=update.tf,
            model_id=update.model_id,
            pred_dir=update.pred_dir.value,
            fact_dir=update.fact_dir.value,
            conf=update.pred_conf,
            ret_bps=update.ret_bps,
            close_prev=update.close_prev,
            close_curr=update.close_curr,
            delta=update.delta,
            p_up_raw=update.p_up_raw,
            p_down_raw=update.p_down_raw,
            p_up_cal=update.p_up_cal,
            p_down_cal=update.p_down_cal,
            margin_raw=update.margin_raw,
            margin_cal=update.margin_cal,
            calib_a=update.calib_a,
            calib_b=update.calib_b,
            top_features=top_features,
        )
        self.confident_wrong_examples.append(example)
        self.confident_wrong_examples.sort(key=lambda e: e.conf, reverse=True)
        if len(self.confident_wrong_examples) > MAX_CONFIDENT_EXAMPLES:
            self.confident_wrong_examples = self.confident_wrong_examples[:MAX_CONFIDENT_EXAMPLES]

    def _extract_top_features(self, update: UpdateEvent) -> Optional[Dict[str, float]]:
        if not self.diagnostic:
            return None
        if not update.features:
            return None
        names = self.feature_names_by_key.get((update.tf, update.model_id))
        if not names:
            return None
        pairs = list(zip(names, update.features))
        pairs.sort(key=lambda item: abs(float(item[1])), reverse=True)
        top = pairs[:TOP_FEATURES]
        return {name: float(value) for name, value in top}

    def summary(self) -> Dict[str, object]:
        total = self.total
        accuracy = self.correct / total if total else 0.0
        nonflat_accuracy = (
            self.nonflat_correct / self.nonflat_total if self.nonflat_total else 0.0
        )
        nonflat_swapped_accuracy = (
            self.nonflat_swapped_correct / self.nonflat_total if self.nonflat_total else 0.0
        )
        inversion_delta = nonflat_swapped_accuracy - nonflat_accuracy
        flat_rate = self.flat_total / total if total else 0.0
        pred_flat_rate = self.pred_flat_total / total if total else 0.0
        fact_flat_rate = self.fact_dir_counts.get(Direction.FLAT.value, 0) / total if total else 0.0
        coverage = self.action_total / total if total else 0.0
        action_accuracy = self.action_correct / self.action_total if self.action_total else 0.0
        flat_when_fact_nonflat_rate = (
            self.pred_flat_when_fact_nonflat / total if total else 0.0
        )
        avg_conf = self.conf_sum / total if total else 0.0
        calib = self.calibration.snapshot()
        calibration_evolution = None
        if self.calib_a_history:
            b_history = self.calib_b_history
            b_initial = b_history[0] if b_history else None
            b_final = b_history[-1] if b_history else None
            b_min = min(b_history) if b_history else None
            b_max = max(b_history) if b_history else None
            calibration_evolution = {
                "a_initial": self.calib_a_history[0],
                "a_final": self.calib_a_history[-1],
                "a_min": min(self.calib_a_history),
                "a_max": max(self.calib_a_history),
                "b_initial": b_initial,
                "b_final": b_final,
                "b_min": b_min,
                "b_max": b_max,
                "collapse_step": self.collapse_step,
            }
        return {
            "total": total,
            "accuracy": accuracy,
            "accuracy_nonflat": nonflat_accuracy,
            "accuracy_nonflat_swapped": nonflat_swapped_accuracy,
            "inversion_delta": inversion_delta,
            "nonflat_total": self.nonflat_total,
            "flat_rate": flat_rate,
            "pred_flat_rate": pred_flat_rate,
            "fact_flat_rate": fact_flat_rate,
            "coverage": coverage,
            "action_accuracy": action_accuracy,
            "flat_when_fact_nonflat_rate": flat_when_fact_nonflat_rate,
            "avg_conf": avg_conf,
            "confusion": self.confusion,
            "pred_dir_counts": self.pred_dir_counts,
            "fact_dir_counts": self.fact_dir_counts,
            "calibration": calib,
            "calibration_evolution": calibration_evolution,
            "flat_diagnostics": {"reasons": self.flat_reasons},
            "confident_wrong_rate": self.confident_wrong / total if total else 0.0,
            "confident_wrong_examples": [
                {
                    "ts": ex.ts,
                    "tf": ex.tf,
                    "model_id": ex.model_id,
                    "pred_dir": ex.pred_dir,
                    "fact_dir": ex.fact_dir,
                    "conf": ex.conf,
                    "ret_bps": ex.ret_bps,
                    "close_prev": ex.close_prev,
                    "close_curr": ex.close_curr,
                    "delta": ex.delta,
                    "p_up_raw": ex.p_up_raw,
                    "p_down_raw": ex.p_down_raw,
                    "p_up_cal": ex.p_up_cal,
                    "p_down_cal": ex.p_down_cal,
                    "margin_raw": ex.margin_raw,
                    "margin_cal": ex.margin_cal,
                    "calib_a": ex.calib_a,
                    "calib_b": ex.calib_b,
                    "top_features": ex.top_features,
                }
                for ex in self.confident_wrong_examples
            ],
            "adaptation": {
                "avg_lr_eff": self.lr_eff_sum / total if total else 0.0,
                "avg_anchor_lambda": self.anchor_lambda_sum / total if total else 0.0,
                "anchor_update_rate": self.anchor_update_count / total if total else 0.0,
                "avg_params_norm": self.params_norm_sum / total if total else 0.0,
                "avg_anchor_norm": self.anchor_norm_sum / total if total else 0.0,
                "avg_params_anchor_gap": self.params_anchor_gap_sum / total if total else 0.0,
            },
        }


class OfflineCollector:
    def __init__(
        self,
        bins: int,
        flat_max_delta: Optional[float] = None,
        feature_names_by_key: Optional[Dict[Tuple[str, str], List[str]]] = None,
        diagnostic: bool = False,
    ) -> None:
        self.bins = bins
        self.flat_max_delta = flat_max_delta
        self.feature_names_by_key = feature_names_by_key or {}
        self.diagnostic = diagnostic
        self.overall = OfflineStats(
            bins,
            flat_max_delta=flat_max_delta,
            feature_names_by_key=self.feature_names_by_key,
            diagnostic=diagnostic,
        )
        self.by_model: Dict[str, OfflineStats] = {}
        self.predictions_total = 0
        self.facts_total = 0
        self.updates_total = 0
        self.errors: List[str] = []

    def on_prediction(self, pred: Prediction) -> None:
        self.predictions_total += 1
        tf_ms = interval_to_ms(pred.tf)
        expected = pred.candle_ts + tf_ms
        if pred.target_ts != expected:
            self.errors.append(
                f"target_ts_mismatch tf={pred.tf} model={pred.model_id} "
                f"candle_ts={pred.candle_ts} target_ts={pred.target_ts} expected={expected}"
            )

    def on_fact(self) -> None:
        self.facts_total += 1

    def on_update(self, update: UpdateEvent) -> None:
        self.updates_total += 1
        if not _finite(update.pred_conf):
            self.errors.append(f"nan_pred_conf tf={update.tf} model={update.model_id}")
        if not _finite(update.loss_task) or not _finite(update.loss_total):
            self.errors.append(f"nan_loss tf={update.tf} model={update.model_id}")
        for key, value in update.weight_norms.items():
            if not _finite(value):
                self.errors.append(f"nan_weight_norm {key} tf={update.tf} model={update.model_id}")
        self.overall.observe(update)
        key = f"{update.tf}:{update.model_id}"
        stats = self.by_model.setdefault(
            key,
            OfflineStats(
                self.bins,
                flat_max_delta=self.flat_max_delta,
                feature_names_by_key=self.feature_names_by_key,
                diagnostic=self.diagnostic,
            ),
        )
        stats.observe(update)

    def summary(self) -> Dict[str, object]:
        return {
            "predictions_total": self.predictions_total,
            "facts_total": self.facts_total,
            "updates_total": self.updates_total,
        }


def _iter_floats(value) -> List[float]:
    floats: List[float] = []
    if isinstance(value, dict):
        for item in value.values():
            floats.extend(_iter_floats(item))
        return floats
    if isinstance(value, (list, tuple)):
        for item in value:
            floats.extend(_iter_floats(item))
        return floats
    try:
        floats.append(float(value))
    except (TypeError, ValueError):
        pass
    return floats


def _finite(value) -> bool:
    values = _iter_floats(value)
    if not values:
        return False
    return all(math.isfinite(v) for v in values)


def _norm_value(value) -> float:
    values = _iter_floats(value)
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values))


def _load_candles(
    source,
    symbol: str,
    tfs: List[str],
    start_ms: int,
    end_ms: int,
    warmup_ms_by_tf: Optional[Dict[str, int]] = None,
) -> Dict[str, List[Candle]]:
    candles_by_tf: Dict[str, List[Candle]] = {}
    for tf in tfs:
        warmup_ms = warmup_ms_by_tf.get(tf, 0) if warmup_ms_by_tf else 0
        tf_start = start_ms - warmup_ms
        candles = source.load(symbol, tf, tf_start, end_ms)
        candles_by_tf[tf] = candles
    return candles_by_tf


def _merge_candles(candles_by_tf: Dict[str, List[Candle]]) -> List[Candle]:
    merged: List[Candle] = []
    for candles in candles_by_tf.values():
        merged.extend(candles)
    merged.sort(key=lambda c: (c.start_ts, c.tf))
    return merged


def _build_segments(start_ms: int, end_ms: int, train_min: int, eval_min: int) -> List[Tuple[bool, int, int]]:
    segments: List[Tuple[bool, int, int]] = []
    cursor = start_ms
    while cursor < end_ms:
        train_end = min(end_ms, cursor + train_min * 60_000)
        segments.append((True, cursor, train_end))
        cursor = train_end
        if cursor >= end_ms:
            break
        eval_end = min(end_ms, cursor + eval_min * 60_000)
        segments.append((False, cursor, eval_end))
        cursor = eval_end
    return segments


def _warmup_ms_by_tf(tfs: List[str], feature_config: FeatureConfig) -> Dict[str, int]:
    builders = [
        FeatureBuilder(feature_config, MODEL_TREND),
        FeatureBuilder(feature_config, MODEL_OSC),
        FeatureBuilder(feature_config, MODEL_VOL),
    ]
    max_required = max(builder.spec.required_lookback for builder in builders)
    warmup_candles = max(0, max_required - 1)
    return {tf: warmup_candles * interval_to_ms(tf) for tf in tfs}


def _split_candles(
    candles_by_tf: Dict[str, List[Candle]],
    start_ms: int,
) -> Tuple[Dict[str, List[Candle]], Dict[str, List[Candle]]]:
    warmup_by_tf: Dict[str, List[Candle]] = {}
    main_by_tf: Dict[str, List[Candle]] = {}
    for tf, candles in candles_by_tf.items():
        warmup_by_tf[tf] = [c for c in candles if c.start_ts < start_ms]
        main_by_tf[tf] = [c for c in candles if c.start_ts >= start_ms]
    return warmup_by_tf, main_by_tf


def _lookahead_samples(
    engine: "MultiTimeframeEngine",
    candles_by_tf: Dict[str, List[Candle]],
    start_ms: int,
    lookahead_samples: int,
) -> Dict[str, List[int]]:
    samples: Dict[str, List[int]] = {}
    for tf, candles in candles_by_tf.items():
        runners = engine.runners.get(tf, [])
        if not runners:
            samples[tf] = []
            continue
        max_required = max(runner.feature_builder.spec.required_lookback for runner in runners)
        first_idx = next(
            (idx for idx, candle in enumerate(candles) if candle.start_ts >= start_ms),
            len(candles),
        )
        min_index = max(max_required - 1, first_idx)
        samples[tf] = _select_indices(candles, min_index, lookahead_samples)
    return samples


def _check_lookahead(
    engine: "MultiTimeframeEngine",
    candles_by_tf: Dict[str, List[Candle]],
    samples: Dict[str, List[int]],
    captured: Dict[Tuple[str, str, int], np.ndarray],
    errors: List[str],
) -> None:
    for tf, indices in samples.items():
        if not indices:
            continue
        candles = candles_by_tf[tf]
        for runner in engine.runners.get(tf, []):
            for idx in indices:
                if idx >= len(candles):
                    continue
                candle_ts = candles[idx].start_ts
                key = (tf, runner.model_id, candle_ts)
                if key not in captured:
                    continue
                recomputed = runner.feature_builder.build(candles[: idx + 1])
                if recomputed is None:
                    continue
                stored = captured[key]
                if stored.shape != recomputed.values.shape or not np.allclose(
                    stored, recomputed.values, atol=1e-6, rtol=1e-5
                ):
                    errors.append(
                        f"lookahead_mismatch tf={tf} model={runner.model_id} candle_ts={candle_ts}"
                    )


def _flat_reason(update: UpdateEvent, flat_max_delta: Optional[float]) -> Optional[str]:
    if update.margin_raw is None and update.margin_cal is None:
        return None
    margin_raw = update.margin_raw
    margin_cal = update.margin_cal
    if margin_raw is not None and margin_raw < 0.10:
        return "GENUINE_UNCERTAINTY"
    if (
        margin_raw is not None
        and update.calib_a is not None
        and update.calib_a < CALIB_A_WARN
        and margin_raw >= 0.10
    ):
        return "COMPRESSION_BY_LOW_A"
    if margin_cal is not None and flat_max_delta is not None:
        if margin_cal >= 0.8 * flat_max_delta:
            return "NEAR_THRESHOLD"
    return "OTHER"


def _guard_direction_shares(
    stats: OfflineStats,
    label: str,
    errors: List[str],
    warnings: List[str],
    strict: bool,
) -> None:
    if stats.total < MIN_GUARD_SAMPLES:
        return
    counts = stats.pred_dir_counts
    total = stats.total
    shares = {k: v / total for k, v in counts.items()}
    if any(share < MIN_DIR_SHARE or share > MAX_DIR_SHARE for share in shares.values()):
        msg = f"collapse_guard {label} shares={shares}"
        if strict:
            errors.append(msg)
        else:
            warnings.append(msg)


def _guard_overconfidence(
    stats: OfflineStats,
    label: str,
    errors: List[str],
    warnings: List[str],
    strict: bool,
) -> None:
    if stats.total < MIN_GUARD_SAMPLES:
        return
    summary = stats.summary()
    avg_conf = float(summary.get("avg_conf", 0.0))
    acc = float(summary.get("accuracy", 0.0))
    ece = float(summary.get("calibration", {}).get("ece", 0.0))
    gap = avg_conf - acc
    should_warn = gap > OVERCONF_GAP_ERROR or ece > OVERCONF_ECE_WARNING
    if not should_warn:
        return
    is_error = strict and (gap > OVERCONF_GAP_ERROR or ece > OVERCONF_ECE_ERROR)
    msg = (
        f"overconfidence_guard {label} avg_conf={avg_conf:.3f} "
        f"acc={acc:.3f} gap={gap:.3f} ece={ece:.3f}"
    )
    if is_error:
        errors.append(msg)
    else:
        warnings.append(msg)


def _guard_calibration_health(
    stats: OfflineStats,
    label: str,
    errors: List[str],
    warnings: List[str],
    strict: bool,
) -> None:
    if stats.total < MIN_GUARD_SAMPLES or not stats.calib_a_history:
        return
    last_a = stats.calib_a_history[-1]
    last_b = stats.calib_b_history[-1] if stats.calib_b_history else None
    if last_a <= CALIB_A_ERROR:
        msg = f"calibration_collapse {label} calib_a={last_a:.3f}"
        if strict:
            errors.append(msg)
        else:
            warnings.append(msg)
    elif last_a <= CALIB_A_WARN:
        warnings.append(f"calibration_low_a {label} calib_a={last_a:.3f}")
    if last_b is not None and abs(last_b) > CALIB_B_WARN:
        warnings.append(f"calibration_high_b {label} calib_b={last_b:.3f}")


def _guard_flat_epidemic(
    stats: OfflineStats,
    label: str,
    errors: List[str],
    warnings: List[str],
    strict: bool,
) -> None:
    if stats.total < MIN_GUARD_SAMPLES:
        return
    summary = stats.summary()
    pred_flat_rate = float(summary.get("pred_flat_rate", 0.0))
    if pred_flat_rate >= PRED_FLAT_ERROR:
        msg = f"flat_epidemic {label} pred_flat_rate={pred_flat_rate:.3f}"
        if strict:
            errors.append(msg)
        else:
            warnings.append(msg)
    elif pred_flat_rate >= PRED_FLAT_WARN:
        warnings.append(f"flat_epidemic {label} pred_flat_rate={pred_flat_rate:.3f}")
    flat_reasons = stats.flat_reasons
    flat_total = sum(flat_reasons.values())
    if flat_total > 0:
        compression = flat_reasons.get("COMPRESSION_BY_LOW_A", 0)
        if compression / flat_total > 0.5:
            warnings.append(
                f"flat_caused_by_low_a {label} share={compression / flat_total:.3f}"
            )


def _write_report(run_dir: Path, summary: Dict[str, object]) -> None:
    report_path = run_dir / "report.md"
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# Offline Replay Report")
    run = summary.get("run", {})
    lines.append(f"- symbol: {run.get('symbol')}")
    lines.append(f"- tfs: {', '.join(run.get('tfs', []))}")
    lines.append(f"- mode: {run.get('mode')}")
    lines.append(f"- range: {run.get('start')} -> {run.get('end')}")
    if run.get("fact_flat_bps") is not None:
        lines.append(f"- fact_flat_bps: {run.get('fact_flat_bps')}")
    lines.append(f"- candles: {run.get('candles_total')}")
    counts = summary.get("counts", {})
    lines.append(
        f"- predictions: {counts.get('predictions_total')} facts: {counts.get('facts_total')} "
        f"updates: {counts.get('updates_total')} pending_tail: {counts.get('pending_tail')}"
    )
    errors = summary.get("checks", {}).get("errors", [])
    warnings = summary.get("checks", {}).get("warnings", [])
    lines.append(f"- checks: {'PASS' if not errors else 'FAIL'} ({len(errors)} errors)")
    if errors:
        lines.append("")
        lines.append("## Failed Checks")
        for err in errors[:20]:
            lines.append(f"- {err}")
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        for warning in warnings[:20]:
            lines.append(f"- {warning}")

    lines.append("")
    lines.append("## Overall")
    overall = summary.get("overall", {})
    lines.append(
        f"- accuracy: {overall.get('accuracy'):.3f} nonflat: {overall.get('accuracy_nonflat'):.3f} "
        f"nonflat_swapped: {overall.get('accuracy_nonflat_swapped'):.3f} "
        f"inversion_delta: {overall.get('inversion_delta'):.3f} "
        f"flat_rate: {overall.get('flat_rate'):.3f} avg_conf: {overall.get('avg_conf'):.3f}"
    )
    lines.append(
        f"- coverage: {_fmt_optional(overall.get('coverage'))} "
        f"action_accuracy: {_fmt_optional(overall.get('action_accuracy'))} "
        f"pred_flat_rate: {_fmt_optional(overall.get('pred_flat_rate'))} "
        f"fact_flat_rate: {_fmt_optional(overall.get('fact_flat_rate'))} "
        f"flat_when_fact_nonflat_rate: {_fmt_optional(overall.get('flat_when_fact_nonflat_rate'))}"
    )
    calib = overall.get("calibration", {})
    lines.append(
        f"- calibration: ece={calib.get('ece'):.3f} mce={calib.get('mce'):.3f} "
        f"brier={calib.get('brier'):.3f}"
    )
    lines.append(f"- confident_wrong_rate: {overall.get('confident_wrong_rate'):.3f}")
    lines.append(f"- adaptation: {overall.get('adaptation')}")
    lines.append("")
    lines.append("### Confusion (Overall)")
    lines.extend(_render_confusion(overall.get("confusion", {})))
    lines.append("")
    lines.append("### Calibration Bins (Overall)")
    lines.extend(_render_bins(calib.get("bin_stats", [])))
    lines.append("")
    lines.append("### Calibration Evolution (Overall)")
    lines.extend(_render_calibration_evolution(overall.get("calibration_evolution")))
    lines.append("")
    lines.append("### FLAT Diagnostics (Overall)")
    lines.extend(_render_flat_diagnostics(overall.get("flat_diagnostics")))
    lines.append("")
    lines.append("### Top Confident-Wrong (Overall)")
    lines.extend(_render_confident_examples(overall.get("confident_wrong_examples", [])))

    lines.append("")
    lines.append("## Per Model")
    for key, payload in summary.get("models", {}).items():
        lines.append(f"### {key}")
        lines.append(
            f"- accuracy: {payload.get('accuracy'):.3f} nonflat: {payload.get('accuracy_nonflat'):.3f} "
            f"nonflat_swapped: {payload.get('accuracy_nonflat_swapped'):.3f} "
            f"inversion_delta: {payload.get('inversion_delta'):.3f} "
            f"flat_rate: {payload.get('flat_rate'):.3f} avg_conf: {payload.get('avg_conf'):.3f}"
        )
        lines.append(
            f"- coverage: {_fmt_optional(payload.get('coverage'))} "
            f"action_accuracy: {_fmt_optional(payload.get('action_accuracy'))} "
            f"pred_flat_rate: {_fmt_optional(payload.get('pred_flat_rate'))} "
            f"fact_flat_rate: {_fmt_optional(payload.get('fact_flat_rate'))} "
            f"flat_when_fact_nonflat_rate: {_fmt_optional(payload.get('flat_when_fact_nonflat_rate'))}"
        )
        calib = payload.get("calibration", {})
        lines.append(
            f"- calibration: ece={calib.get('ece'):.3f} mce={calib.get('mce'):.3f} "
            f"brier={calib.get('brier'):.3f}"
        )
        lines.append(f"- confident_wrong_rate: {payload.get('confident_wrong_rate'):.3f}")
        lines.append(f"- adaptation: {payload.get('adaptation')}")
        lines.append("")
        lines.append("Confusion:")
        lines.extend(_render_confusion(payload.get("confusion", {})))
        lines.append("")
        lines.append("Calibration bins:")
        lines.extend(_render_bins(calib.get("bin_stats", [])))
        lines.append("")
        lines.append("Calibration evolution:")
        lines.extend(_render_calibration_evolution(payload.get("calibration_evolution")))
        lines.append("")
        lines.append("FLAT diagnostics:")
        lines.extend(_render_flat_diagnostics(payload.get("flat_diagnostics")))
        lines.append("")
        lines.append("Top confident-wrong:")
        lines.extend(_render_confident_examples(payload.get("confident_wrong_examples", [])))

    lines.append("")
    lines.append("## Patterns")
    patterns = summary.get("patterns", {})
    if not patterns:
        lines.append("- disabled")
    else:
        for key, payload in patterns.items():
            summary_row = payload.get("summary", {})
            coverage = payload.get("coverage", {})
            lines.append(
                f"- {key}: total={summary_row.get('total', 0)} "
                f"context={summary_row.get('context', 0)} decision={summary_row.get('decision', 0)} "
                f"coverage={coverage}"
            )
            anti = payload.get("top", {}).get("anti", [])
            if anti:
                lines.append("  top anti-patterns:")
                for item in anti[:5]:
                    lines.append(f"  - {item.get('pattern_key')} count={item.get('count')} win={item.get('ema_win')}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_confusion(confusion: Dict[str, Dict[str, int]]) -> List[str]:
    rows = ["pred\\fact | UP | DOWN | FLAT", "---|---:|---:|---:"]
    for pred in ["UP", "DOWN", "FLAT"]:
        row = confusion.get(pred, {})
        rows.append(f"{pred} | {row.get('UP', 0)} | {row.get('DOWN', 0)} | {row.get('FLAT', 0)}")
    return rows


def _render_bins(bin_stats: List[Dict[str, object]]) -> List[str]:
    if not bin_stats:
        return ["- no bins"]
    lines = ["bin | count | acc | avg_conf | avg_abs_ret_bps", "---|---:|---:|---:|---:"]
    for idx, stat in enumerate(bin_stats):
        lines.append(
            f"{idx} | {stat.get('count', 0)} | {stat.get('accuracy', 0.0):.3f} "
            f"| {stat.get('avg_conf', 0.0):.3f} | {stat.get('avg_abs_ret_bps', 0.0):.3f}"
        )
    return lines


def _render_calibration_evolution(payload: Optional[Dict[str, object]]) -> List[str]:
    if not payload:
        return ["- no history"]
    collapse_step = payload.get("collapse_step")
    collapse_text = str(collapse_step) if collapse_step is not None else "n/a"
    lines = [
        f"- a: initial={_fmt_optional(payload.get('a_initial'))} "
        f"final={_fmt_optional(payload.get('a_final'))} "
        f"min={_fmt_optional(payload.get('a_min'))} max={_fmt_optional(payload.get('a_max'))}",
        f"- b: initial={_fmt_optional(payload.get('b_initial'))} "
        f"final={_fmt_optional(payload.get('b_final'))} "
        f"min={_fmt_optional(payload.get('b_min'))} max={_fmt_optional(payload.get('b_max'))}",
        f"- collapse_step: {collapse_text}",
    ]
    return lines


def _render_flat_diagnostics(payload: Optional[Dict[str, object]]) -> List[str]:
    if not payload:
        return ["- none"]
    reasons = payload.get("reasons", {}) if isinstance(payload, dict) else {}
    if not reasons:
        return ["- none"]
    total = sum(reasons.values())
    if total <= 0:
        return ["- none"]
    lines = ["reason | count | share", "---|---:|---:"]
    for reason, count in sorted(reasons.items(), key=lambda item: item[1], reverse=True):
        share = count / total if total else 0.0
        lines.append(f"{reason} | {count} | {share:.3f}")
    return lines


def _render_confident_examples(examples: List[Dict[str, object]]) -> List[str]:
    if not examples:
        return ["- none"]
    lines = [
        "ts | tf | model | pred | fact | conf | ret_bps | close_prev | close_curr | delta | calib_a | margin_raw | margin_cal",
        "---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:",
    ]
    for item in examples:
        lines.append(
            f"{item.get('ts')} | {item.get('tf')} | {item.get('model_id')} | "
            f"{item.get('pred_dir')} | {item.get('fact_dir')} | "
            f"{_fmt_optional(item.get('conf'))} | {_fmt_optional(item.get('ret_bps'))} | "
            f"{_fmt_optional(item.get('close_prev'))} | {_fmt_optional(item.get('close_curr'))} | "
            f"{_fmt_optional(item.get('delta'))} | {_fmt_optional(item.get('calib_a'))} | "
            f"{_fmt_optional(item.get('margin_raw'))} | {_fmt_optional(item.get('margin_cal'))}"
        )
    return lines


def _fmt_optional(value: object, precision: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline replay for BTCUSDT predictor")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--tf", default=None)
    parser.add_argument("--tfs", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--minutes", type=int, default=None)
    parser.add_argument("--mode", choices=["smoke", "eval-only", "train", "walkforward"], default="train")
    parser.add_argument("--train-min", type=int, default=60)
    parser.add_argument("--eval-min", type=int, default=20)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument(
        "--clean-run",
        action="store_true",
        help="Remove run-dir if it exists before running",
    )
    parser.add_argument("--state-db", default=None)
    parser.add_argument("--pattern-db", default=None)
    parser.add_argument("--no-patterns", action="store_true")
    parser.add_argument("--no-anchor", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--fact-flat-bps", type=float, default=FactConfig().fact_flat_bps)
    parser.add_argument("--candles-jsonl", default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--strict", action="store_true", help="Treat guard warnings as errors")
    parser.add_argument(
        "--stop-on-collapse",
        action="store_true",
        help="Stop replay if calibration collapses",
    )
    parser.add_argument(
        "--collapse-a-threshold",
        type=float,
        default=CALIB_A_ERROR,
        help="Calibration a threshold for stop-on-collapse",
    )
    parser.add_argument(
        "--lookahead-samples",
        type=int,
        default=LOOKAHEAD_SAMPLES,
        help="Number of lookahead samples for feature validation",
    )
    parser.add_argument(
        "--min-guard-samples",
        type=int,
        default=MIN_GUARD_SAMPLES,
        help="Minimum updates before guard checks apply",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Write out/updates_diagnostics.csv",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Enable extended diagnostic output",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    global MIN_GUARD_SAMPLES
    MIN_GUARD_SAMPLES = args.min_guard_samples
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    if "TITAN_ENTRYPOINT" not in os.environ:
        os.environ["TITAN_ENTRYPOINT"] = "src.offline_replay"
    backend, devices = ensure_jax_backend()
    logging.info("JAX backend=%s devices=%s", backend, devices)

    from .engine import MultiTimeframeEngine, RunnerConfig

    tfs = parse_tfs(args.tfs or args.tf or "1")
    start_ms, end_ms = _resolve_time_range(args.start, args.end, args.minutes)
    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / f"offline_{now_ms()}"
    if args.clean_run and run_dir.exists():
        if run_dir.is_dir():
            shutil.rmtree(run_dir)
        else:
            run_dir.unlink()
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir = run_dir / "out"

    state_db = Path(args.state_db) if args.state_db else run_dir / "state.db"
    pattern_db = Path(args.pattern_db) if args.pattern_db else run_dir / "patterns.db"

    rest_config = RestConfig()
    if args.candles_jsonl:
        source = JsonlCandleSource(Path(args.candles_jsonl))
    else:
        source = BybitRestCandleSource(rest_config=rest_config)

    feature_config = FeatureConfig()
    decision_config = DecisionConfig()
    fact_config = FactConfig(fact_flat_bps=args.fact_flat_bps)
    reward_config = RewardConfig()
    model_config = ModelConfig()
    training = TrainingConfig()
    model_init = ModelInitConfig()
    lrs = ModelLRConfig()

    pattern_config = PatternConfig(db_path=pattern_db)
    persistence = PersistenceConfig(state_db=state_db)
    output = out_dir
    if args.no_patterns:
        pattern_store = NullPatternStore(pattern_config)
    else:
        pattern_db.parent.mkdir(parents=True, exist_ok=True)
        pattern_store = PatternStore(str(pattern_db), pattern_config)

    state_db.parent.mkdir(parents=True, exist_ok=True)
    state_store = ModelStateStore(str(state_db))

    base_update_patterns = not args.no_patterns
    base_use_patterns = not args.no_patterns
    base_enable_anchor = not args.no_anchor
    use_ema = not args.no_ema

    training_enabled = args.mode != "eval-only"
    runner_config = RunnerConfig(
        use_ema=use_ema,
        use_patterns=base_use_patterns,
        update_patterns=base_update_patterns and training_enabled,
        enable_training=training_enabled,
        enable_anchor=base_enable_anchor,
        enable_calibration_update=training_enabled,
    )

    engine = MultiTimeframeEngine(
        tfs=tfs,
        feature_config=feature_config,
        fact_config=fact_config,
        reward_config=reward_config,
        model_config=model_config,
        model_init=model_init,
        training=training,
        decision=decision_config,
        lrs=lrs,
        pattern_store=pattern_store,
        state_store=state_store,
        runner_config=runner_config,
    )
    engine.load_states()
    feature_names_by_key: Dict[Tuple[str, str], List[str]] = {}
    for tf, runners in engine.runners.items():
        for runner in runners:
            feature_names_by_key[(tf, runner.model_id)] = runner.feature_builder.spec.feature_names

    warmup_ms_by_tf = _warmup_ms_by_tf(tfs, feature_config)
    candles_by_tf_full = _load_candles(
        source, args.symbol, tfs, start_ms, end_ms, warmup_ms_by_tf
    )
    warmup_by_tf, candles_by_tf = _split_candles(candles_by_tf_full, start_ms)
    for tf, warmup in warmup_by_tf.items():
        if warmup:
            engine.warm_start(tf, warmup)

    merged = _merge_candles(candles_by_tf)
    if args.mode == "smoke" and len(merged) > SMOKE_MAX_CANDLES:
        merged = merged[:SMOKE_MAX_CANDLES]

    recorder = JsonlRecorder(output)
    collector = OfflineCollector(
        bins=training.calibration_bins,
        flat_max_delta=decision_config.flat_max_delta,
        feature_names_by_key=feature_names_by_key,
        diagnostic=args.diagnostic,
    )

    lookahead_samples = max(0, int(args.lookahead_samples))
    samples = _lookahead_samples(engine, candles_by_tf_full, start_ms, lookahead_samples)
    captured_features: Dict[Tuple[str, str, int], np.ndarray] = {}
    sample_sets = {
        tf: {candles_by_tf_full[tf][idx].start_ts for idx in indices}
        for tf, indices in samples.items()
    }
    csv_writer = None
    csv_fp = None
    if args.export_csv:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "updates_diagnostics.csv"
        csv_fp = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_fp)
        csv_writer.writerow(
            [
                "ts",
                "iso",
                "tf",
                "model_id",
                "pred_dir",
                "fact_dir",
                "correct",
                "pred_conf",
                "ret_bps",
                "loss_task",
                "loss_total",
                "lr_eff",
                "anchor_lambda_eff",
                "calib_a",
                "calib_b",
                "calib_n",
                "p_up_raw",
                "p_down_raw",
                "p_up_cal",
                "p_down_cal",
                "margin_raw",
                "margin_cal",
                "close_prev",
                "close_curr",
                "delta",
                "flat_reason",
            ]
        )
    stop_reason: Optional[str] = None
    stop_step: Optional[int] = None
    stop_requested = False

    def record_prediction(pred: Prediction) -> None:
        collector.on_prediction(pred)
        recorder.record_prediction(pred)

    def record_fact(_fact) -> None:
        collector.on_fact()
        recorder.record_fact(_fact)

    def record_update(update: UpdateEvent) -> None:
        nonlocal stop_reason, stop_step, stop_requested
        collector.on_update(update)
        if csv_writer is not None:
            flat_reason = ""
            if update.pred_dir == Direction.FLAT:
                flat_reason = _flat_reason(update, decision_config.flat_max_delta) or ""
            csv_writer.writerow(
                [
                    update.ts,
                    _iso_local(update.ts),
                    update.tf,
                    update.model_id,
                    update.pred_dir.value,
                    update.fact_dir.value,
                    update.pred_dir == update.fact_dir,
                    update.pred_conf,
                    update.ret_bps,
                    update.loss_task,
                    update.loss_total,
                    update.lr_eff,
                    update.anchor_lambda_eff,
                    update.calib_a,
                    update.calib_b,
                    update.calib_n,
                    update.p_up_raw,
                    update.p_down_raw,
                    update.p_up_cal,
                    update.p_down_cal,
                    update.margin_raw,
                    update.margin_cal,
                    update.close_prev,
                    update.close_curr,
                    update.delta,
                    flat_reason,
                ]
            )
        if (
            args.stop_on_collapse
            and stop_reason is None
            and update.calib_a is not None
            and update.calib_a <= args.collapse_a_threshold
        ):
            stop_step = collector.updates_total
            stop_reason = (
                f"stop_on_collapse tf={update.tf} model={update.model_id} "
                f"calib_a={update.calib_a:.3f} step={stop_step}"
            )
            stop_requested = True
        recorder.record_update(update)

    if args.mode == "walkforward":
        segments = _build_segments(start_ms, end_ms, args.train_min, args.eval_min)
        segment_idx = 0
        current_mode = segments[0][0] if segments else True
        engine.set_runtime_flags(
            enable_training=current_mode,
            update_patterns=current_mode and base_update_patterns,
            enable_calibration_update=current_mode,
            enable_anchor=current_mode and base_enable_anchor,
        )
    else:
        segments = []
        segment_idx = 0
        current_mode = runner_config.enable_training

    for candle in merged:
        if segments:
            while segment_idx < len(segments) and candle.start_ts >= segments[segment_idx][2]:
                segment_idx += 1
                if segment_idx >= len(segments):
                    break
                current_mode = segments[segment_idx][0]
                engine.set_runtime_flags(
                    enable_training=current_mode,
                    update_patterns=current_mode and base_update_patterns,
                    enable_calibration_update=current_mode,
                    enable_anchor=current_mode and base_enable_anchor,
                )
        engine.process_candle(
            candle,
            record_prediction,
            record_fact,
            record_update,
            runtime_state=None,
            autosave_seconds=persistence.autosave_seconds,
            autosave_updates=persistence.autosave_updates,
        )
        tf = candle.tf
        if tf in sample_sets and candle.start_ts in sample_sets[tf]:
            target_ts = candle.start_ts + interval_to_ms(tf)
            for runner in engine.runners.get(tf, []):
                pending = runner.pending.get(target_ts)
                if pending is not None:
                    captured_features[(tf, runner.model_id, candle.start_ts)] = pending.features.copy()
        if stop_requested:
            break

    for tf, runners in engine.runners.items():
        for runner in runners:
            runner.maybe_save(now_ms(), autosave_seconds=0, autosave_updates=1)

    pending_tail = sum(len(runner.pending) for runners in engine.runners.values() for runner in runners)
    collector_errors = list(collector.errors)
    counts = collector.summary()
    if counts["updates_total"] != counts["facts_total"]:
        collector_errors.append("facts_updates_mismatch")
    if counts["updates_total"] > counts["predictions_total"]:
        collector_errors.append("updates_exceed_predictions")
    if counts["predictions_total"] - counts["updates_total"] != pending_tail:
        collector_errors.append("pending_tail_mismatch")

    _check_lookahead(engine, candles_by_tf_full, samples, captured_features, collector_errors)

    warnings: List[str] = []
    _guard_direction_shares(collector.overall, "overall", collector_errors, warnings, args.strict)
    _guard_overconfidence(collector.overall, "overall", collector_errors, warnings, args.strict)
    _guard_calibration_health(collector.overall, "overall", collector_errors, warnings, args.strict)
    _guard_flat_epidemic(collector.overall, "overall", collector_errors, warnings, args.strict)
    for key, stats in collector.by_model.items():
        _guard_direction_shares(stats, key, collector_errors, warnings, args.strict)
        _guard_overconfidence(stats, key, collector_errors, warnings, args.strict)
        _guard_calibration_health(stats, key, collector_errors, warnings, args.strict)
        _guard_flat_epidemic(stats, key, collector_errors, warnings, args.strict)

    patterns_report: Dict[str, object] = {}
    if not args.no_patterns and isinstance(pattern_store, PatternStore):
        for tf in tfs:
            for runner in engine.runners.get(tf, []):
                model_key = f"{tf}:{runner.model_id}"
                patterns_report[model_key] = {
                    "summary": pattern_store.stats_summary(tf, runner.model_id),
                    "coverage": pattern_store.coverage_summary(tf, runner.model_id),
                    "top": pattern_store.top_patterns(tf, runner.model_id, limit=5),
                }

    for key, stats in collector.by_model.items():
        summary_stats = stats.summary()
        nonflat_total = int(summary_stats.get("nonflat_total", 0))
        inversion_delta = float(summary_stats.get("inversion_delta", 0.0))
        if nonflat_total >= 50 and inversion_delta >= 0.10:
            warning_msg = (
                f"inversion_warning {key} delta={inversion_delta:.3f} "
                f"nonflat_total={nonflat_total}"
            )
            warnings.append(warning_msg)
            logging.warning("Inversion detected: %s", warning_msg)
    if stop_reason:
        warnings.append(stop_reason)

    summary = {
        "run": {
            "symbol": args.symbol,
            "tfs": tfs,
            "mode": args.mode,
            "start": _iso_local(start_ms),
            "end": _iso_local(end_ms),
            "candles_total": len(merged),
            "run_dir": str(run_dir),
            "fact_flat_bps": fact_config.fact_flat_bps,
            "stopped_early": bool(stop_reason),
            "stop_reason": stop_reason,
            "stop_step": stop_step,
        },
        "counts": {
            **counts,
            "pending_tail": pending_tail,
        },
        "overall": collector.overall.summary(),
        "models": {k: v.summary() for k, v in collector.by_model.items()},
        "patterns": patterns_report,
        "checks": {"errors": collector_errors, "warnings": warnings},
        "guards": {
            "min_dir_share": MIN_DIR_SHARE,
            "max_dir_share": MAX_DIR_SHARE,
            "min_guard_samples": MIN_GUARD_SAMPLES,
            "pred_flat_warn": PRED_FLAT_WARN,
            "pred_flat_error": PRED_FLAT_ERROR,
            "calib_a_warn": CALIB_A_WARN,
            "calib_a_error": CALIB_A_ERROR,
            "calib_b_warn": CALIB_B_WARN,
            "overconf_ece_warn": OVERCONF_ECE_WARNING,
            "overconf_ece_error": OVERCONF_ECE_ERROR,
            "overconf_gap_error": OVERCONF_GAP_ERROR,
        },
    }

    _write_report(run_dir, summary)

    recorder.close()
    if csv_fp is not None:
        csv_fp.close()
    pattern_store.close()
    state_store.close()

    logging.info("Report saved: %s", run_dir / "report.md")
    if collector_errors:
        raise SystemExit(2)
    if args.strict and warnings:
        raise SystemExit(2)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
