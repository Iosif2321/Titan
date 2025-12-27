from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List

from .types import Direction


@dataclass
class MetricSample:
    correct: bool
    reward: float
    flat: bool
    pred_dir: Direction
    fact_dir: Direction


class RollingMetrics:
    def __init__(self, window: int = 200) -> None:
        self.window = max(1, window)
        self.total = 0
        self.correct = 0
        self.flat_total = 0
        self.reward_sum = 0.0
        self.last_reward = 0.0
        self.pred_dir_counts = {Direction.UP: 0, Direction.DOWN: 0, Direction.FLAT: 0}
        self.fact_dir_counts = {Direction.UP: 0, Direction.DOWN: 0, Direction.FLAT: 0}
        self.pred_dir_correct = {Direction.UP: 0, Direction.DOWN: 0, Direction.FLAT: 0}
        self._window: Deque[MetricSample] = deque(maxlen=self.window)
        self._window_total = 0
        self._window_correct = 0
        self._window_flat_total = 0
        self._window_reward_sum = 0.0
        self._window_pred_dir_counts = {Direction.UP: 0, Direction.DOWN: 0, Direction.FLAT: 0}
        self._window_fact_dir_counts = {Direction.UP: 0, Direction.DOWN: 0, Direction.FLAT: 0}
        self._window_pred_dir_correct = {Direction.UP: 0, Direction.DOWN: 0, Direction.FLAT: 0}

    def update(self, pred_dir: Direction, fact_dir: Direction, reward: float) -> None:
        correct = pred_dir == fact_dir
        flat = pred_dir == Direction.FLAT
        self.total += 1
        self.correct += int(correct)
        self.flat_total += int(flat)
        self.reward_sum += reward
        self.last_reward = reward
        self.pred_dir_counts[pred_dir] += 1
        self.fact_dir_counts[fact_dir] += 1
        if correct:
            self.pred_dir_correct[pred_dir] += 1

        if len(self._window) == self._window.maxlen:
            old = self._window.popleft()
            self._window_total -= 1
            self._window_correct -= int(old.correct)
            self._window_flat_total -= int(old.flat)
            self._window_reward_sum -= old.reward
            self._window_pred_dir_counts[old.pred_dir] -= 1
            self._window_fact_dir_counts[old.fact_dir] -= 1
            if old.correct:
                self._window_pred_dir_correct[old.pred_dir] -= 1

        sample = MetricSample(
            correct=correct,
            reward=reward,
            flat=flat,
            pred_dir=pred_dir,
            fact_dir=fact_dir,
        )
        self._window.append(sample)
        self._window_total += 1
        self._window_correct += int(correct)
        self._window_flat_total += int(flat)
        self._window_reward_sum += reward
        self._window_pred_dir_counts[pred_dir] += 1
        self._window_fact_dir_counts[fact_dir] += 1
        if correct:
            self._window_pred_dir_correct[pred_dir] += 1

    def snapshot(self) -> Dict[str, float]:
        accuracy = self.correct / self.total if self.total else 0.0
        flat_rate = self.flat_total / self.total if self.total else 0.0
        avg_reward = self.reward_sum / self.total if self.total else 0.0
        window_accuracy = self._window_correct / self._window_total if self._window_total else 0.0
        window_flat_rate = (
            self._window_flat_total / self._window_total if self._window_total else 0.0
        )
        window_avg_reward = (
            self._window_reward_sum / self._window_total if self._window_total else 0.0
        )
        pred_dir = {k.value: v for k, v in self.pred_dir_counts.items()}
        fact_dir = {k.value: v for k, v in self.fact_dir_counts.items()}
        window_pred_dir = {k.value: v for k, v in self._window_pred_dir_counts.items()}
        window_fact_dir = {k.value: v for k, v in self._window_fact_dir_counts.items()}
        accuracy_by_pred_dir = {
            k.value: (self.pred_dir_correct[k] / self.pred_dir_counts[k])
            if self.pred_dir_counts[k]
            else 0.0
            for k in self.pred_dir_counts
        }
        window_accuracy_by_pred_dir = {
            k.value: (self._window_pred_dir_correct[k] / self._window_pred_dir_counts[k])
            if self._window_pred_dir_counts[k]
            else 0.0
            for k in self._window_pred_dir_counts
        }
        return {
            "total": float(self.total),
            "accuracy": accuracy,
            "flat_rate": flat_rate,
            "avg_reward": avg_reward,
            "window_total": float(self._window_total),
            "window_accuracy": window_accuracy,
            "window_flat_rate": window_flat_rate,
            "window_avg_reward": window_avg_reward,
            "last_reward": self.last_reward,
            "pred_dir": pred_dir,
            "fact_dir": fact_dir,
            "accuracy_by_pred_dir": accuracy_by_pred_dir,
            "window_pred_dir": window_pred_dir,
            "window_fact_dir": window_fact_dir,
            "window_accuracy_by_pred_dir": window_accuracy_by_pred_dir,
        }

    def to_state(self) -> Dict[str, object]:
        return {
            "window": self.window,
            "total": self.total,
            "correct": self.correct,
            "flat_total": self.flat_total,
            "reward_sum": self.reward_sum,
            "last_reward": self.last_reward,
            "pred_dir_counts": {k.value: v for k, v in self.pred_dir_counts.items()},
            "fact_dir_counts": {k.value: v for k, v in self.fact_dir_counts.items()},
            "pred_dir_correct": {k.value: v for k, v in self.pred_dir_correct.items()},
            "window_samples": [
                {
                    "correct": s.correct,
                    "reward": s.reward,
                    "flat": s.flat,
                    "pred_dir": s.pred_dir.value,
                    "fact_dir": s.fact_dir.value,
                }
                for s in self._window
            ],
        }

    def load_state(self, state: Dict[str, object]) -> None:
        self.window = max(1, int(state.get("window", self.window)))
        self.total = int(state.get("total", 0))
        self.correct = int(state.get("correct", 0))
        self.flat_total = int(state.get("flat_total", 0))
        self.reward_sum = float(state.get("reward_sum", 0.0))
        self.last_reward = float(state.get("last_reward", 0.0))
        pred_dir_counts = state.get("pred_dir_counts", {})
        fact_dir_counts = state.get("fact_dir_counts", {})
        pred_dir_correct = state.get("pred_dir_correct", {})
        self.pred_dir_counts = {
            Direction.UP: int(pred_dir_counts.get(Direction.UP.value, 0)),
            Direction.DOWN: int(pred_dir_counts.get(Direction.DOWN.value, 0)),
            Direction.FLAT: int(pred_dir_counts.get(Direction.FLAT.value, 0)),
        }
        self.fact_dir_counts = {
            Direction.UP: int(fact_dir_counts.get(Direction.UP.value, 0)),
            Direction.DOWN: int(fact_dir_counts.get(Direction.DOWN.value, 0)),
            Direction.FLAT: int(fact_dir_counts.get(Direction.FLAT.value, 0)),
        }
        self.pred_dir_correct = {
            Direction.UP: int(pred_dir_correct.get(Direction.UP.value, 0)),
            Direction.DOWN: int(pred_dir_correct.get(Direction.DOWN.value, 0)),
            Direction.FLAT: int(pred_dir_correct.get(Direction.FLAT.value, 0)),
        }
        self._window = deque(maxlen=self.window)
        self._window_total = 0
        self._window_correct = 0
        self._window_flat_total = 0
        self._window_reward_sum = 0.0
        self._window_pred_dir_counts = {Direction.UP: 0, Direction.DOWN: 0, Direction.FLAT: 0}
        self._window_fact_dir_counts = {Direction.UP: 0, Direction.DOWN: 0, Direction.FLAT: 0}
        self._window_pred_dir_correct = {Direction.UP: 0, Direction.DOWN: 0, Direction.FLAT: 0}
        samples = state.get("window_samples", [])
        if isinstance(samples, list):
            for sample in samples:
                pred_raw = str(sample.get("pred_dir", Direction.FLAT.value))
                fact_raw = str(sample.get("fact_dir", Direction.FLAT.value))
                try:
                    pred_dir = Direction(pred_raw)
                except ValueError:
                    pred_dir = Direction.FLAT
                try:
                    fact_dir = Direction(fact_raw)
                except ValueError:
                    fact_dir = Direction.FLAT
                s = MetricSample(
                    correct=bool(sample.get("correct", False)),
                    reward=float(sample.get("reward", 0.0)),
                    flat=bool(sample.get("flat", False)),
                    pred_dir=pred_dir,
                    fact_dir=fact_dir,
                )
                self._window.append(s)
                self._window_total += 1
                self._window_correct += int(s.correct)
                self._window_flat_total += int(s.flat)
                self._window_reward_sum += s.reward
                self._window_pred_dir_counts[s.pred_dir] += 1
                self._window_fact_dir_counts[s.fact_dir] += 1
                if s.correct:
                    self._window_pred_dir_correct[s.pred_dir] += 1


@dataclass
class CalibrationSample:
    confidence: float
    correct: bool
    abs_ret_bps: float
    bin_idx: int


class CalibrationMetrics:
    def __init__(self, bins: int = 10, window: int = 200) -> None:
        self.bins = max(2, bins)
        self.window = max(1, window)
        self.total = 0
        self.correct = 0
        self.brier_sum = 0.0
        self.conf_sum = 0.0
        self.abs_ret_sum = 0.0
        self.conf_sq_sum = 0.0
        self.abs_ret_sq_sum = 0.0
        self.conf_abs_ret_sum = 0.0
        self.counts = [0 for _ in range(self.bins)]
        self.correct_bins = [0 for _ in range(self.bins)]
        self.conf_sum_bins = [0.0 for _ in range(self.bins)]
        self.abs_ret_sum_bins = [0.0 for _ in range(self.bins)]
        self._window: Deque[CalibrationSample] = deque(maxlen=self.window)
        self._window_total = 0
        self._window_correct = 0
        self._window_brier_sum = 0.0
        self._window_conf_sum = 0.0
        self._window_abs_ret_sum = 0.0
        self._window_conf_sq_sum = 0.0
        self._window_abs_ret_sq_sum = 0.0
        self._window_conf_abs_ret_sum = 0.0
        self._window_counts = [0 for _ in range(self.bins)]
        self._window_correct_bins = [0 for _ in range(self.bins)]
        self._window_conf_sum_bins = [0.0 for _ in range(self.bins)]
        self._window_abs_ret_sum_bins = [0.0 for _ in range(self.bins)]

    def update(self, confidence: float, correct: bool, abs_ret_bps: float) -> None:
        confidence = max(0.0, min(1.0, float(confidence)))
        abs_ret_bps = float(abs_ret_bps)
        brier = (confidence - (1.0 if correct else 0.0)) ** 2
        bin_idx = _bin_index(confidence, self.bins)

        self.total += 1
        self.correct += int(correct)
        self.brier_sum += brier
        self.conf_sum += confidence
        self.abs_ret_sum += abs_ret_bps
        self.conf_sq_sum += confidence * confidence
        self.abs_ret_sq_sum += abs_ret_bps * abs_ret_bps
        self.conf_abs_ret_sum += confidence * abs_ret_bps
        self.counts[bin_idx] += 1
        self.correct_bins[bin_idx] += int(correct)
        self.conf_sum_bins[bin_idx] += confidence
        self.abs_ret_sum_bins[bin_idx] += abs_ret_bps

        if len(self._window) == self._window.maxlen:
            old = self._window.popleft()
            self._window_total -= 1
            self._window_correct -= int(old.correct)
            self._window_brier_sum -= (old.confidence - (1.0 if old.correct else 0.0)) ** 2
            self._window_conf_sum -= old.confidence
            self._window_abs_ret_sum -= old.abs_ret_bps
            self._window_conf_sq_sum -= old.confidence * old.confidence
            self._window_abs_ret_sq_sum -= old.abs_ret_bps * old.abs_ret_bps
            self._window_conf_abs_ret_sum -= old.confidence * old.abs_ret_bps
            self._window_counts[old.bin_idx] -= 1
            self._window_correct_bins[old.bin_idx] -= int(old.correct)
            self._window_conf_sum_bins[old.bin_idx] -= old.confidence
            self._window_abs_ret_sum_bins[old.bin_idx] -= old.abs_ret_bps

        sample = CalibrationSample(
            confidence=confidence,
            correct=correct,
            abs_ret_bps=abs_ret_bps,
            bin_idx=bin_idx,
        )
        self._window.append(sample)
        self._window_total += 1
        self._window_correct += int(correct)
        self._window_brier_sum += brier
        self._window_conf_sum += confidence
        self._window_abs_ret_sum += abs_ret_bps
        self._window_conf_sq_sum += confidence * confidence
        self._window_abs_ret_sq_sum += abs_ret_bps * abs_ret_bps
        self._window_conf_abs_ret_sum += confidence * abs_ret_bps
        self._window_counts[bin_idx] += 1
        self._window_correct_bins[bin_idx] += int(correct)
        self._window_conf_sum_bins[bin_idx] += confidence
        self._window_abs_ret_sum_bins[bin_idx] += abs_ret_bps

    def snapshot(self) -> Dict[str, object]:
        overall = _calibration_snapshot(
            self.total,
            self.correct,
            self.brier_sum,
            self.conf_sum,
            self.abs_ret_sum,
            self.conf_sq_sum,
            self.abs_ret_sq_sum,
            self.conf_abs_ret_sum,
            self.counts,
            self.correct_bins,
            self.conf_sum_bins,
            self.abs_ret_sum_bins,
            self.bins,
        )
        window = _calibration_snapshot(
            self._window_total,
            self._window_correct,
            self._window_brier_sum,
            self._window_conf_sum,
            self._window_abs_ret_sum,
            self._window_conf_sq_sum,
            self._window_abs_ret_sq_sum,
            self._window_conf_abs_ret_sum,
            self._window_counts,
            self._window_correct_bins,
            self._window_conf_sum_bins,
            self._window_abs_ret_sum_bins,
            self.bins,
        )
        overall["window"] = window
        return overall

    def to_state(self) -> Dict[str, object]:
        return {
            "bins": self.bins,
            "window": self.window,
            "total": self.total,
            "correct": self.correct,
            "brier_sum": self.brier_sum,
            "conf_sum": self.conf_sum,
            "abs_ret_sum": self.abs_ret_sum,
            "conf_sq_sum": self.conf_sq_sum,
            "abs_ret_sq_sum": self.abs_ret_sq_sum,
            "conf_abs_ret_sum": self.conf_abs_ret_sum,
            "counts": list(self.counts),
            "correct_bins": list(self.correct_bins),
            "conf_sum_bins": list(self.conf_sum_bins),
            "abs_ret_sum_bins": list(self.abs_ret_sum_bins),
            "window_samples": [
                {
                    "confidence": s.confidence,
                    "correct": s.correct,
                    "abs_ret_bps": s.abs_ret_bps,
                    "bin_idx": s.bin_idx,
                }
                for s in self._window
            ],
        }

    def load_state(self, state: Dict[str, object]) -> None:
        if not isinstance(state, dict):
            return
        self.bins = max(2, int(state.get("bins", self.bins)))
        self.window = max(1, int(state.get("window", self.window)))
        self.total = int(state.get("total", 0))
        self.correct = int(state.get("correct", 0))
        self.brier_sum = float(state.get("brier_sum", 0.0))
        self.conf_sum = float(state.get("conf_sum", 0.0))
        self.abs_ret_sum = float(state.get("abs_ret_sum", 0.0))
        self.conf_sq_sum = float(state.get("conf_sq_sum", 0.0))
        self.abs_ret_sq_sum = float(state.get("abs_ret_sq_sum", 0.0))
        self.conf_abs_ret_sum = float(state.get("conf_abs_ret_sum", 0.0))
        self.counts = _safe_list(state.get("counts"), self.bins, 0)
        self.correct_bins = _safe_list(state.get("correct_bins"), self.bins, 0)
        self.conf_sum_bins = _safe_list(state.get("conf_sum_bins"), self.bins, 0.0)
        self.abs_ret_sum_bins = _safe_list(state.get("abs_ret_sum_bins"), self.bins, 0.0)
        self._window = deque(maxlen=self.window)
        self._window_total = 0
        self._window_correct = 0
        self._window_brier_sum = 0.0
        self._window_conf_sum = 0.0
        self._window_abs_ret_sum = 0.0
        self._window_conf_sq_sum = 0.0
        self._window_abs_ret_sq_sum = 0.0
        self._window_conf_abs_ret_sum = 0.0
        self._window_counts = [0 for _ in range(self.bins)]
        self._window_correct_bins = [0 for _ in range(self.bins)]
        self._window_conf_sum_bins = [0.0 for _ in range(self.bins)]
        self._window_abs_ret_sum_bins = [0.0 for _ in range(self.bins)]
        samples = state.get("window_samples", [])
        if isinstance(samples, list):
            for sample in samples:
                try:
                    confidence = float(sample.get("confidence", 0.0))
                except (TypeError, ValueError):
                    confidence = 0.0
                correct = bool(sample.get("correct", False))
                try:
                    abs_ret_bps = float(sample.get("abs_ret_bps", 0.0))
                except (TypeError, ValueError):
                    abs_ret_bps = 0.0
                try:
                    bin_idx = int(sample.get("bin_idx", _bin_index(confidence, self.bins)))
                except (TypeError, ValueError):
                    bin_idx = _bin_index(confidence, self.bins)
                bin_idx = max(0, min(self.bins - 1, bin_idx))
                s = CalibrationSample(
                    confidence=confidence,
                    correct=correct,
                    abs_ret_bps=abs_ret_bps,
                    bin_idx=bin_idx,
                )
                self._window.append(s)
                self._window_total += 1
                self._window_correct += int(correct)
                self._window_brier_sum += (confidence - (1.0 if correct else 0.0)) ** 2
                self._window_conf_sum += confidence
                self._window_abs_ret_sum += abs_ret_bps
                self._window_conf_sq_sum += confidence * confidence
                self._window_abs_ret_sq_sum += abs_ret_bps * abs_ret_bps
                self._window_conf_abs_ret_sum += confidence * abs_ret_bps
                self._window_counts[bin_idx] += 1
                self._window_correct_bins[bin_idx] += int(correct)
                self._window_conf_sum_bins[bin_idx] += confidence
                self._window_abs_ret_sum_bins[bin_idx] += abs_ret_bps

    def get_recent_ece(self, window_size: int = 50) -> float | None:
        """
        Calculate ECE (Expected Calibration Error) over recent window.

        This is used for adaptive learning rate feedback. Returns ECE computed
        over the last window_size samples, or None if insufficient data.

        Args:
            window_size: Number of recent samples to use (default: 50)

        Returns:
            ECE over recent window, or None if < window_size samples available
        """
        if self._window_total < window_size:
            return None

        # Get recent samples from deque
        recent = list(self._window)[-window_size:]

        # Recalculate ECE for window
        bins = self.bins
        counts = [0] * bins
        correct_bins = [0] * bins
        conf_sum_bins = [0.0] * bins

        for sample in recent:
            counts[sample.bin_idx] += 1
            if sample.correct:
                correct_bins[sample.bin_idx] += 1
            conf_sum_bins[sample.bin_idx] += sample.confidence

        # Compute ECE
        ece = 0.0
        total = len(recent)
        for idx in range(bins):
            if counts[idx] > 0:
                acc = correct_bins[idx] / counts[idx]
                avg_conf = conf_sum_bins[idx] / counts[idx]
                err = abs(acc - avg_conf)
                ece += (counts[idx] / total) * err

        return ece


def _bin_index(confidence: float, bins: int) -> int:
    if confidence <= 0.0:
        return 0
    if confidence >= 1.0:
        return bins - 1
    return min(int(confidence * bins), bins - 1)


def _pearson(n: int, sum_x: float, sum_y: float, sum_x2: float, sum_y2: float, sum_xy: float) -> float:
    if n < 2:
        return 0.0
    num = n * sum_xy - sum_x * sum_y
    denom_x = n * sum_x2 - sum_x * sum_x
    denom_y = n * sum_y2 - sum_y * sum_y
    if denom_x <= 0.0 or denom_y <= 0.0:
        return 0.0
    return float(num / math.sqrt(denom_x * denom_y))


def _calibration_snapshot(
    total: int,
    correct: int,
    brier_sum: float,
    conf_sum: float,
    abs_ret_sum: float,
    conf_sq_sum: float,
    abs_ret_sq_sum: float,
    conf_abs_ret_sum: float,
    counts: List[int],
    correct_bins: List[int],
    conf_sum_bins: List[float],
    abs_ret_sum_bins: List[float],
    bins: int,
) -> Dict[str, object]:
    if total <= 0:
        return {
            "total": 0,
            "accuracy": 0.0,
            "brier": 0.0,
            "avg_conf": 0.0,
            "avg_abs_ret_bps": 0.0,
            "ece": 0.0,
            "mce": 0.0,
            "corr_conf_abs_ret": 0.0,
            "bin_stats": [],
        }
    ece = 0.0
    mce = 0.0
    bin_stats: List[Dict[str, float]] = []
    for idx in range(bins):
        count = counts[idx]
        bin_low = idx / bins
        bin_high = (idx + 1) / bins
        if count > 0:
            acc = correct_bins[idx] / count
            avg_conf = conf_sum_bins[idx] / count
            avg_abs_ret = abs_ret_sum_bins[idx] / count
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
    corr = _pearson(total, conf_sum, abs_ret_sum, conf_sq_sum, abs_ret_sq_sum, conf_abs_ret_sum)
    return {
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "brier": brier_sum / total if total else 0.0,
        "avg_conf": conf_sum / total if total else 0.0,
        "avg_abs_ret_bps": abs_ret_sum / total if total else 0.0,
        "ece": ece,
        "mce": mce,
        "corr_conf_abs_ret": corr,
        "bin_stats": bin_stats,
    }


def _safe_list(values: object, size: int, default: float | int) -> List[float | int]:
    if not isinstance(values, list):
        return [default for _ in range(size)]
    out: List[float | int] = []
    for idx in range(size):
        try:
            out.append(values[idx])
        except IndexError:
            out.append(default)
    return out
