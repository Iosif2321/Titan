from __future__ import annotations

import argparse
import json
import logging
from math import sqrt
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


def _pearson(n: int, sum_x: float, sum_y: float, sum_x2: float, sum_y2: float, sum_xy: float) -> float:
    if n < 2:
        return 0.0
    num = n * sum_xy - sum_x * sum_y
    denom_x = n * sum_x2 - sum_x * sum_x
    denom_y = n * sum_y2 - sum_y * sum_y
    if denom_x <= 0.0 or denom_y <= 0.0:
        return 0.0
    return float(num / sqrt(denom_x * denom_y))


def _bin_index(confidence: float, bins: int) -> int:
    if confidence <= 0.0:
        return 0
    if confidence >= 1.0:
        return bins - 1
    return min(int(confidence * bins), bins - 1)


def _calibration_stats(
    counts: List[int],
    correct: List[int],
    conf_sum: List[float],
    abs_ret_sum: List[float],
    bins: int,
) -> Tuple[float, float, List[dict[str, float]]]:
    total = sum(counts)
    if total == 0:
        return 0.0, 0.0, []

    ece = 0.0
    mce = 0.0
    bin_stats: List[dict[str, float]] = []
    for idx in range(bins):
        count = counts[idx]
        bin_low = idx / bins
        bin_high = (idx + 1) / bins
        if count > 0:
            acc = correct[idx] / count
            avg_conf = conf_sum[idx] / count
            avg_abs_ret = abs_ret_sum[idx] / count
            err = abs(acc - avg_conf)
            ece += (count / total) * err
            mce = max(mce, err)
        else:
            acc = 0.0
            avg_conf = 0.0
            avg_abs_ret = 0.0
        bin_stats.append(
            {
                "bin_low": bin_low,
                "bin_high": bin_high,
                "count": count,
                "accuracy": acc,
                "avg_conf": avg_conf,
                "avg_abs_ret_bps": avg_abs_ret,
            }
        )
    return ece, mce, bin_stats


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Skipping invalid JSON at %s:%d", path, line_num)


def load_facts(
    path: Path,
    interval_filter: Optional[str],
    start_ts: Optional[int],
    end_ts: Optional[int],
) -> Dict[Tuple[str, int], dict]:
    facts: Dict[Tuple[str, int], dict] = {}
    for record in iter_jsonl(path):
        if "target_ts" not in record:
            continue
        try:
            target_ts = int(record["target_ts"])
        except (TypeError, ValueError):
            continue
        interval = str(record.get("interval", ""))
        if interval_filter and interval != interval_filter:
            continue
        if start_ts is not None and target_ts < start_ts:
            continue
        if end_ts is not None and target_ts > end_ts:
            continue
        facts[(interval, target_ts)] = record
    return facts


def build_report(
    predictions_path: Path,
    facts_path: Path,
    bins: int,
    interval_filter: Optional[str],
    model_id_filter: Optional[str],
    start_ts: Optional[int],
    end_ts: Optional[int],
) -> dict:
    facts = load_facts(facts_path, interval_filter, start_ts, end_ts)
    total_preds = 0
    matched = 0
    correct = 0
    nonflat_total = 0
    nonflat_correct = 0
    flat_total = 0
    flat_correct = 0
    conf_sum = 0.0
    conf_sq_sum = 0.0
    abs_ret_sum = 0.0
    abs_ret_sq_sum = 0.0
    conf_abs_ret_sum = 0.0
    conf_count = 0
    conf_sum_nonflat = 0.0
    conf_sq_sum_nonflat = 0.0
    abs_ret_sum_nonflat = 0.0
    abs_ret_sq_sum_nonflat = 0.0
    conf_abs_ret_sum_nonflat = 0.0
    conf_count_nonflat = 0
    calib_counts = [0 for _ in range(bins)]
    calib_correct = [0 for _ in range(bins)]
    calib_conf_sum = [0.0 for _ in range(bins)]
    calib_abs_ret_sum = [0.0 for _ in range(bins)]
    min_ts = None
    max_ts = None

    for record in iter_jsonl(predictions_path):
        if "target_ts" not in record:
            continue
        interval = str(record.get("interval", ""))
        if interval_filter and interval != interval_filter:
            continue
        if model_id_filter and record.get("model_id") != model_id_filter:
            continue
        try:
            target_ts = int(record["target_ts"])
        except (TypeError, ValueError):
            continue
        if start_ts is not None and target_ts < start_ts:
            continue
        if end_ts is not None and target_ts > end_ts:
            continue
        total_preds += 1
        key = (interval, target_ts)
        fact = facts.get(key)
        if fact is None:
            continue

        matched += 1
        min_ts = target_ts if min_ts is None else min(min_ts, target_ts)
        max_ts = target_ts if max_ts is None else max(max_ts, target_ts)
        pred_dir = str(record.get("direction", ""))
        fact_dir = str(fact.get("direction", ""))
        try:
            confidence = float(record.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if "confidence" not in record:
            try:
                p_up = float(record.get("p_up", 0.0))
                p_down = float(record.get("p_down", 0.0))
                confidence = max(p_up, p_down)
            except (TypeError, ValueError):
                confidence = 0.0

        try:
            abs_ret_bps = abs(float(fact.get("ret_bps", 0.0)))
        except (TypeError, ValueError):
            abs_ret_bps = 0.0

        is_correct = pred_dir == fact_dir
        is_nonflat = pred_dir != "FLAT"
        is_flat_correct = pred_dir == "FLAT" and fact_dir == "FLAT"

        correct += int(is_correct)
        conf_sum += confidence
        conf_sq_sum += confidence * confidence
        abs_ret_sum += abs_ret_bps
        abs_ret_sq_sum += abs_ret_bps * abs_ret_bps
        conf_abs_ret_sum += confidence * abs_ret_bps
        conf_count += 1

        if is_nonflat:
            nonflat_total += 1
            nonflat_correct += int(is_correct)
            conf_sum_nonflat += confidence
            conf_sq_sum_nonflat += confidence * confidence
            abs_ret_sum_nonflat += abs_ret_bps
            abs_ret_sq_sum_nonflat += abs_ret_bps * abs_ret_bps
            conf_abs_ret_sum_nonflat += confidence * abs_ret_bps
            conf_count_nonflat += 1
            bin_idx = _bin_index(confidence, bins)
            calib_counts[bin_idx] += 1
            calib_correct[bin_idx] += int(is_correct)
            calib_conf_sum[bin_idx] += confidence
            calib_abs_ret_sum[bin_idx] += abs_ret_bps
        else:
            flat_total += 1
            flat_correct += int(is_flat_correct)

    ece, mce, calib_stats = _calibration_stats(
        calib_counts, calib_correct, calib_conf_sum, calib_abs_ret_sum, bins
    )
    corr_all = _pearson(conf_count, conf_sum, abs_ret_sum, conf_sq_sum, abs_ret_sq_sum, conf_abs_ret_sum)
    corr_nonflat = _pearson(
        conf_count_nonflat,
        conf_sum_nonflat,
        abs_ret_sum_nonflat,
        conf_sq_sum_nonflat,
        abs_ret_sq_sum_nonflat,
        conf_abs_ret_sum_nonflat,
    )

    return {
        "filters": {
            "interval": interval_filter,
            "model_id": model_id_filter,
            "start_ts": start_ts,
            "end_ts": end_ts,
        },
        "summary": {
            "predictions_total": total_preds,
            "predictions_matched": matched,
            "predictions_unmatched": total_preds - matched,
            "facts_total": len(facts),
            "accuracy": _safe_div(correct, matched),
            "accuracy_nonflat": _safe_div(nonflat_correct, nonflat_total),
            "flat_rate": _safe_div(flat_total, matched),
            "flat_accuracy": _safe_div(flat_correct, flat_total),
            "coverage_nonflat": _safe_div(nonflat_total, matched),
            "avg_confidence": _safe_div(conf_sum, conf_count),
            "avg_abs_ret_bps": _safe_div(abs_ret_sum, conf_count),
            "min_target_ts": min_ts,
            "max_target_ts": max_ts,
        },
        "calibration": {
            "bins": bins,
            "ece": ece,
            "mce": mce,
            "bin_stats": calib_stats,
        },
        "correlation": {
            "confidence_abs_ret_bps": corr_all,
            "confidence_abs_ret_bps_nonflat": corr_nonflat,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline evaluation for predictions.jsonl vs facts.jsonl"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--interval", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--start-ts", type=int, default=None)
    parser.add_argument("--end-ts", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    bins = max(2, int(args.bins))
    report = build_report(
        predictions_path=args.predictions,
        facts_path=args.facts,
        bins=bins,
        interval_filter=args.interval,
        model_id_filter=args.model_id,
        start_ts=args.start_ts,
        end_ts=args.end_ts,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
