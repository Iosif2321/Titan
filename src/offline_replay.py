from __future__ import annotations

import argparse
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
MIN_GUARD_SAMPLES = 50
MIN_DIR_SHARE = 0.02
MAX_DIR_SHARE = 0.98
OVERCONF_GAP = 0.15
OVERCONF_ECE = 0.2
LOOKAHEAD_SAMPLES = 10
SMOKE_MAX_CANDLES = 500


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


def _select_indices(length: int, min_index: int, count: int) -> List[int]:
    if length <= min_index + 1:
        return []
    available = length - min_index
    step = max(1, available // count)
    indices = [min_index + i * step for i in range(count)]
    return [idx for idx in indices if idx < length]


@dataclass
class ConfidentExample:
    ts: int
    tf: str
    model_id: str
    pred_dir: str
    fact_dir: str
    conf: float
    ret_bps: float


class OfflineStats:
    def __init__(self, bins: int) -> None:
        self.total = 0
        self.correct = 0
        self.nonflat_total = 0
        self.nonflat_correct = 0
        self.nonflat_swapped_correct = 0
        self.flat_total = 0
        self.flat_correct = 0
        self.pred_dir_counts = {d.value: 0 for d in Direction}
        self.fact_dir_counts = {d.value: 0 for d in Direction}
        self.confusion = {d.value: {f.value: 0 for f in Direction} for d in Direction}
        self.conf_sum = 0.0
        self.calibration = CalibrationMetrics(bins=bins, window=10_000)
        self.confident_wrong = 0
        self.confident_wrong_examples: List[ConfidentExample] = []
        self.lr_eff_sum = 0.0
        self.anchor_lambda_sum = 0.0
        self.anchor_update_count = 0
        self.params_norm_sum = 0.0
        self.anchor_norm_sum = 0.0
        self.params_anchor_gap_sum = 0.0

    def observe(self, update: UpdateEvent) -> None:
        pred_dir = update.pred_dir.value
        fact_dir = update.fact_dir.value
        correct = pred_dir == fact_dir
        self.total += 1
        self.correct += int(correct)
        self.pred_dir_counts[pred_dir] += 1
        self.fact_dir_counts[fact_dir] += 1
        self.confusion[pred_dir][fact_dir] += 1
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
        example = ConfidentExample(
            ts=update.ts,
            tf=update.tf,
            model_id=update.model_id,
            pred_dir=update.pred_dir.value,
            fact_dir=update.fact_dir.value,
            conf=update.pred_conf,
            ret_bps=update.ret_bps,
        )
        self.confident_wrong_examples.append(example)
        self.confident_wrong_examples.sort(key=lambda e: e.conf, reverse=True)
        if len(self.confident_wrong_examples) > MAX_CONFIDENT_EXAMPLES:
            self.confident_wrong_examples = self.confident_wrong_examples[:MAX_CONFIDENT_EXAMPLES]

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
        avg_conf = self.conf_sum / total if total else 0.0
        calib = self.calibration.snapshot()
        return {
            "total": total,
            "accuracy": accuracy,
            "accuracy_nonflat": nonflat_accuracy,
            "accuracy_nonflat_swapped": nonflat_swapped_accuracy,
            "inversion_delta": inversion_delta,
            "nonflat_total": self.nonflat_total,
            "flat_rate": flat_rate,
            "avg_conf": avg_conf,
            "confusion": self.confusion,
            "pred_dir_counts": self.pred_dir_counts,
            "fact_dir_counts": self.fact_dir_counts,
            "calibration": calib,
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
    def __init__(self, bins: int) -> None:
        self.bins = bins
        self.overall = OfflineStats(bins)
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
        stats = self.by_model.setdefault(key, OfflineStats(self.bins))
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
    engine: "MultiTimeframeEngine", candles_by_tf: Dict[str, List[Candle]], start_ms: int
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
        samples[tf] = _select_indices(len(candles), min_index, LOOKAHEAD_SAMPLES)
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


def _guard_direction_shares(stats: OfflineStats, label: str, errors: List[str]) -> None:
    if stats.total < MIN_GUARD_SAMPLES:
        return
    counts = stats.pred_dir_counts
    total = stats.total
    shares = {k: v / total for k, v in counts.items()}
    if any(share < MIN_DIR_SHARE or share > MAX_DIR_SHARE for share in shares.values()):
        errors.append(f"collapse_guard {label} shares={shares}")


def _guard_overconfidence(stats: OfflineStats, label: str, errors: List[str]) -> None:
    if stats.total < MIN_GUARD_SAMPLES:
        return
    summary = stats.summary()
    avg_conf = float(summary.get("avg_conf", 0.0))
    acc = float(summary.get("accuracy", 0.0))
    ece = float(summary.get("calibration", {}).get("ece", 0.0))
    if avg_conf - acc > OVERCONF_GAP or ece > OVERCONF_ECE:
        errors.append(
            f"overconfidence_guard {label} avg_conf={avg_conf:.3f} acc={acc:.3f} ece={ece:.3f}"
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


def _render_confident_examples(examples: List[Dict[str, object]]) -> List[str]:
    if not examples:
        return ["- none"]
    lines = ["ts | tf | model | pred | fact | conf | ret_bps", "---|---|---|---|---|---:|---:"]
    for item in examples:
        lines.append(
            f"{item.get('ts')} | {item.get('tf')} | {item.get('model_id')} | "
            f"{item.get('pred_dir')} | {item.get('fact_dir')} | "
            f"{item.get('conf'):.3f} | {item.get('ret_bps'):.3f}"
        )
    return lines


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
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
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
    collector = OfflineCollector(bins=training.calibration_bins)

    samples = _lookahead_samples(engine, candles_by_tf_full, start_ms)
    captured_features: Dict[Tuple[str, str, int], np.ndarray] = {}
    sample_sets = {
        tf: {candles_by_tf_full[tf][idx].start_ts for idx in indices}
        for tf, indices in samples.items()
    }

    def record_prediction(pred: Prediction) -> None:
        collector.on_prediction(pred)
        recorder.record_prediction(pred)

    def record_fact(_fact) -> None:
        collector.on_fact()
        recorder.record_fact(_fact)

    def record_update(update: UpdateEvent) -> None:
        collector.on_update(update)
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

    for tf, runners in engine.runners.items():
        for runner in runners:
            runner.maybe_save(now_ms(), autosave_seconds=0, autosave_updates=1)

    pending_tail = sum(len(runner.pending) for runners in engine.runners.values() for runner in runners)
    collector_errors = collector.errors
    counts = collector.summary()
    if counts["updates_total"] != counts["facts_total"]:
        collector_errors.append("facts_updates_mismatch")
    if counts["updates_total"] > counts["predictions_total"]:
        collector_errors.append("updates_exceed_predictions")
    if counts["predictions_total"] - counts["updates_total"] != pending_tail:
        collector_errors.append("pending_tail_mismatch")

    _check_lookahead(engine, candles_by_tf_full, samples, captured_features, collector_errors)

    _guard_direction_shares(collector.overall, "overall", collector_errors)
    _guard_overconfidence(collector.overall, "overall", collector_errors)
    for key, stats in collector.by_model.items():
        _guard_direction_shares(stats, key, collector_errors)
        _guard_overconfidence(stats, key, collector_errors)

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

    warnings: List[str] = []
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
            "overconf_gap": OVERCONF_GAP,
            "overconf_ece": OVERCONF_ECE,
        },
    }

    _write_report(run_dir, summary)

    recorder.close()
    pattern_store.close()
    state_store.close()

    if collector_errors:
        logging.warning(
            "Offline replay completed with %d failed checks. See report.md for details.",
            len(collector_errors),
        )


if __name__ == "__main__":
    main()
