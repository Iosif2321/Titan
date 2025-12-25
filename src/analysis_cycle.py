from __future__ import annotations

import asyncio
import logging
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from typing import Deque, Dict, Iterable, Optional

from .pattern_store import PatternStore
from .types import Direction, UpdateEvent
from .utils import now_ms, ts_iso


@dataclass(frozen=True)
class AnalysisConfig:
    interval_seconds: int = 1800
    duration_seconds: int = 12 * 3600
    window_seconds: int = 1800
    pattern_top_n: int = 5


@dataclass(frozen=True)
class UpdateSample:
    ts: int
    pred_dir: str
    fact_dir: str
    reward: float
    pred_conf: float
    ret_bps: float


class Stats:
    def __init__(self) -> None:
        self.total = 0
        self.correct = 0
        self.pred_nonflat = 0
        self.correct_nonflat = 0
        self.fact_nonflat = 0
        self.correct_fact_nonflat = 0
        self.sum_reward = 0.0
        self.sum_conf = 0.0
        self.sum_abs_ret = 0.0
        self.fact_counts = Counter()

    def add(self, sample: UpdateSample) -> None:
        self.total += 1
        if sample.pred_dir == sample.fact_dir:
            self.correct += 1
        if sample.pred_dir != Direction.FLAT.value:
            self.pred_nonflat += 1
            if sample.pred_dir == sample.fact_dir:
                self.correct_nonflat += 1
        if sample.fact_dir != Direction.FLAT.value:
            self.fact_nonflat += 1
            if sample.pred_dir == sample.fact_dir:
                self.correct_fact_nonflat += 1
        self.sum_reward += sample.reward
        self.sum_conf += sample.pred_conf
        self.sum_abs_ret += abs(sample.ret_bps)
        self.fact_counts[sample.fact_dir] += 1

    def summary(self) -> Dict[str, float]:
        def safe_div(n: float, d: float) -> float:
            return float(n / d) if d else 0.0

        return {
            "total": self.total,
            "accuracy": safe_div(self.correct, self.total),
            "accuracy_nonflat_pred": safe_div(self.correct_nonflat, self.pred_nonflat),
            "accuracy_nonflat_fact": safe_div(self.correct_fact_nonflat, self.fact_nonflat),
            "coverage_nonflat": safe_div(self.pred_nonflat, self.total),
            "avg_reward": safe_div(self.sum_reward, self.total),
            "avg_conf": safe_div(self.sum_conf, self.total),
            "avg_abs_ret_bps": safe_div(self.sum_abs_ret, self.total),
            "fact_up": self.fact_counts[Direction.UP.value],
            "fact_down": self.fact_counts[Direction.DOWN.value],
            "fact_flat": self.fact_counts[Direction.FLAT.value],
            "baseline_up": safe_div(self.fact_counts[Direction.UP.value], self.total),
            "baseline_down": safe_div(self.fact_counts[Direction.DOWN.value], self.total),
            "baseline_flat": safe_div(self.fact_counts[Direction.FLAT.value], self.total),
        }


class PeriodicAnalyzer:
    def __init__(
        self,
        config: AnalysisConfig,
        record_analysis,
        pattern_store: PatternStore | None = None,
    ) -> None:
        self.config = config
        self._record_analysis = record_analysis
        self._pattern_store = pattern_store
        self._events: Dict[str, Deque[UpdateSample]] = defaultdict(deque)
        self._totals: Dict[str, Stats] = defaultdict(Stats)
        self._lock = Lock()
        self._stop = False
        self._start_ts = now_ms()

    def stop(self) -> None:
        self._stop = True

    def observe_update(self, update: UpdateEvent) -> None:
        if self._stop:
            return
        sample = UpdateSample(
            ts=update.ts,
            pred_dir=update.pred_dir.value,
            fact_dir=update.fact_dir.value,
            reward=float(update.reward),
            pred_conf=float(update.pred_conf),
            ret_bps=float(update.ret_bps),
        )
        key = f"{update.tf}:{update.model_id}"
        with self._lock:
            self._events[key].append(sample)
            self._totals[key].add(sample)

    async def run(self) -> None:
        if self.config.interval_seconds <= 0:
            return
        deadline = self._start_ts + self.config.duration_seconds * 1000
        next_ts = self._start_ts + self.config.interval_seconds * 1000
        try:
            while not self._stop:
                now = now_ms()
                if now >= deadline:
                    self._stop = True
                    break
                sleep_s = (next_ts - now) / 1000.0
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
                now = now_ms()
                records = self._build_reports(now)
                if records:
                    for record in records:
                        self._emit(record)
                else:
                    logging.info(
                        "analysis window=%s..%s (no updates yet)",
                        ts_iso(now - self.config.window_seconds * 1000),
                        ts_iso(now),
                    )
                next_ts = now + self.config.interval_seconds * 1000
        except asyncio.CancelledError:
            return

    def _build_reports(self, now_ts: int) -> Iterable[Dict[str, object]]:
        window_ms = self.config.window_seconds * 1000
        window_start = now_ts - window_ms
        records: list[Dict[str, object]] = []
        with self._lock:
            for key, events in self._events.items():
                while events and events[0].ts < window_start:
                    events.popleft()
                window_stats = Stats()
                for sample in events:
                    window_stats.add(sample)
                total_stats = self._totals[key]
                tf, model_id = key.split(":", 1)
                record = {
                    "ts": now_ts,
                    "ts_iso": ts_iso(now_ts),
                    "window_start_ts": window_start,
                    "window_end_ts": now_ts,
                    "window_seconds": self.config.window_seconds,
                    "tf": tf,
                    "model_id": model_id,
                    "window": window_stats.summary(),
                    "cumulative": total_stats.summary(),
                }
                if self._pattern_store is not None:
                    try:
                        record["patterns"] = {
                            "summary": self._pattern_store.stats_summary(tf, model_id),
                            "usage": self._pattern_store.usage_summary(tf, model_id, window_start),
                            "top": self._pattern_store.top_patterns(
                                tf, model_id, limit=self.config.pattern_top_n
                            ),
                        }
                    except Exception:
                        record["patterns"] = {"error": "failed to load pattern stats"}
                records.append(record)
        return records

    def _emit(self, record: Dict[str, object]) -> None:
        window = record.get("window", {})
        cumulative = record.get("cumulative", {})
        window_start_ts = record.get("window_start_ts")
        window_end_ts = record.get("window_end_ts")
        window_start_iso = ts_iso(window_start_ts) if isinstance(window_start_ts, int) else "n/a"
        window_end_iso = ts_iso(window_end_ts) if isinstance(window_end_ts, int) else "n/a"
        logging.info(
            "analysis window=%s..%s tf=%s model=%s win_total=%s win_acc=%.3f win_cov=%.3f "
            "win_reward=%.3f cum_total=%s cum_acc=%.3f",
            window_start_iso,
            window_end_iso,
            record.get("tf"),
            record.get("model_id"),
            window.get("total", 0),
            window.get("accuracy", 0.0),
            window.get("coverage_nonflat", 0.0),
            window.get("avg_reward", 0.0),
            cumulative.get("total", 0),
            cumulative.get("accuracy", 0.0),
        )
        if self._record_analysis is not None:
            self._record_analysis(record)
