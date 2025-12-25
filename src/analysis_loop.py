from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

from .analysis_cycle import AnalysisConfig, Stats, UpdateSample
from .recording import JsonlWriter
from .types import Direction
from .utils import now_ms, ts_iso


def _safe_int(value: object) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_dir(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).upper()
    if text in (Direction.UP.value, Direction.DOWN.value, Direction.FLAT.value):
        return text
    return None


def _parse_update(record: dict) -> Optional[Tuple[str, str, UpdateSample]]:
    tf = record.get("tf") or record.get("interval")
    model_id = record.get("model_id") or record.get("model_type") or record.get("model")
    ts = _safe_int(record.get("ts") or record.get("ts_eval"))
    pred_dir = _normalize_dir(record.get("pred_dir"))
    fact_dir = _normalize_dir(record.get("fact_dir"))
    pred_conf = record.get("pred_conf")
    ret_bps = record.get("ret_bps")
    reward = record.get("reward")

    pred = None
    fact = None
    if pred_dir is None or fact_dir is None:
        pred = record.get("pred") or {}
        fact = record.get("fact") or {}
        pred_dir = pred_dir or _normalize_dir(pred.get("direction") or pred.get("dir"))
        fact_dir = fact_dir or _normalize_dir(fact.get("direction") or fact.get("fact_dir"))
        if pred_conf is None:
            pred_conf = pred.get("confidence")
        if ret_bps is None:
            ret_bps = fact.get("ret_bps")

    if pred_conf is None and pred is not None:
        p_up = _safe_float(pred.get("p_up"))
        p_down = _safe_float(pred.get("p_down"))
        pred_conf = max(p_up, p_down)

    if pred_dir is None or fact_dir is None:
        return None
    if tf is None or model_id is None:
        return None
    if ts is None:
        ts = now_ms()

    sample = UpdateSample(
        ts=ts,
        pred_dir=pred_dir,
        fact_dir=fact_dir,
        reward=_safe_float(reward),
        pred_conf=_safe_float(pred_conf),
        ret_bps=_safe_float(ret_bps),
    )
    return str(tf), str(model_id), sample


def _build_records(
    events: Dict[str, Deque[UpdateSample]],
    totals: Dict[str, Stats],
    now_ts: int,
    window_seconds: int,
) -> list[dict]:
    window_ms = window_seconds * 1000
    window_start = now_ts - window_ms
    records: list[dict] = []
    for key, deque_samples in events.items():
        while deque_samples and deque_samples[0].ts < window_start:
            deque_samples.popleft()
        window_stats = Stats()
        for sample in deque_samples:
            window_stats.add(sample)
        total_stats = totals[key]
        tf, model_id = key.split(":", 1)
        records.append(
            {
                "ts": now_ts,
                "ts_iso": ts_iso(now_ts),
                "window_start_ts": window_start,
                "window_end_ts": now_ts,
                "window_seconds": window_seconds,
                "tf": tf,
                "model_id": model_id,
                "window": window_stats.summary(),
                "cumulative": total_stats.summary(),
            }
        )
    return records


def _emit_records(records: list[dict], writer: Optional[JsonlWriter]) -> None:
    for record in records:
        window = record.get("window", {})
        cumulative = record.get("cumulative", {})
        logging.info(
            "analysis window=%s..%s tf=%s model=%s win_total=%s win_acc=%.3f win_cov=%.3f "
            "win_reward=%.3f cum_total=%s cum_acc=%.3f",
            ts_iso(int(record.get("window_start_ts"))),
            ts_iso(int(record.get("window_end_ts"))),
            record.get("tf"),
            record.get("model_id"),
            window.get("total", 0),
            window.get("accuracy", 0.0),
            window.get("coverage_nonflat", 0.0),
            window.get("avg_reward", 0.0),
            cumulative.get("total", 0),
            cumulative.get("accuracy", 0.0),
        )
        if writer is not None:
            writer.write(record)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Periodic analysis loop for updates.jsonl"
    )
    parser.add_argument(
        "--updates",
        type=Path,
        default=Path("out/updates.jsonl"),
        help="Path to updates.jsonl",
    )
    parser.add_argument(
        "--analysis-out",
        type=Path,
        default=Path("out/analysis.jsonl"),
        help="Output analysis.jsonl path",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=AnalysisConfig().interval_seconds,
        help="Emit analysis every N seconds",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=AnalysisConfig().window_seconds,
        help="Window size for interim stats",
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=AnalysisConfig().duration_seconds,
        help="Total run duration before exit",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=1.0,
        help="Polling interval for new updates",
    )
    parser.add_argument(
        "--from-end",
        action="store_true",
        help="Start tailing from end of file (ignore existing history)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    config = AnalysisConfig(
        interval_seconds=max(1, int(args.interval_seconds)),
        duration_seconds=max(1, int(args.duration_seconds)),
        window_seconds=max(1, int(args.window_seconds)),
    )

    writer = None
    args.analysis_out.parent.mkdir(parents=True, exist_ok=True)
    writer = JsonlWriter(args.analysis_out)

    events: Dict[str, Deque[UpdateSample]] = defaultdict(deque)
    totals: Dict[str, Stats] = defaultdict(Stats)

    if args.updates.exists() and not args.from_end:
        with args.updates.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                parsed = _parse_update(record)
                if not parsed:
                    continue
                tf, model_id, sample = parsed
                key = f"{tf}:{model_id}"
                events[key].append(sample)
                totals[key].add(sample)

    stream = None
    if args.updates.exists():
        stream = args.updates.open("r", encoding="utf-8")
        stream.seek(0, 2)

    start_ts = now_ms()
    next_emit = start_ts + config.interval_seconds * 1000
    deadline = start_ts + config.duration_seconds * 1000

    try:
        while True:
            now = now_ms()
            if now >= deadline:
                break
            if now >= next_emit:
                if events:
                    records = _build_records(events, totals, now, config.window_seconds)
                    _emit_records(records, writer)
                else:
                    logging.info(
                        "analysis window=%s..%s (no updates yet)",
                        ts_iso(now - config.window_seconds * 1000),
                        ts_iso(now),
                    )
                next_emit = now + config.interval_seconds * 1000

            if stream is None:
                if args.updates.exists():
                    stream = args.updates.open("r", encoding="utf-8")
                    stream.seek(0, 2)
                else:
                    time.sleep(args.poll_seconds)
                    continue

            line = stream.readline()
            if line:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                parsed = _parse_update(record)
                if not parsed:
                    continue
                tf, model_id, sample = parsed
                key = f"{tf}:{model_id}"
                events[key].append(sample)
                totals[key].add(sample)
                continue

            try:
                current_size = args.updates.stat().st_size if args.updates.exists() else 0
                if stream.tell() > current_size:
                    stream.close()
                    stream = None
            except OSError:
                stream = None
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        logging.info("Analysis loop stopped")
    finally:
        if stream is not None:
            stream.close()
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
