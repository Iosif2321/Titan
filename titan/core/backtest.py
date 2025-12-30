import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from titan.core.adapters.pattern import PatternAdjuster, PatternModelAdjuster, PatternReader
from titan.core.analysis import PredictionAnalyzer, PredictionDetail, StatisticalValidator
from titan.core.calibration import OnlineCalibrator
from titan.core.config import ConfigStore
from titan.core.data.loader import CsvCandleReader
from titan.core.features.stream import FeatureStream, build_conditions
from titan.core.models.heuristic import Oscillator, TrendVIC, VolumeMetrix
from titan.core.models.ml import DirectionalClassifier, create_ml_classifier, HAS_LIGHTGBM
from titan.core.ensemble import Ensemble
from titan.core.monitor import PerformanceMonitor
from titan.core.patterns import PatternExperience, PatternStore
from titan.core.regime import RegimeDetector
from titan.core.state_store import StateStore
from titan.core.types import Decision, ModelOutput, Outcome, PatternContext, PredictionRecord
from titan.core.weights import AdaptiveWeightManager, WeightManager


@dataclass
class ModelStats:
    total: int = 0
    correct: int = 0
    up: int = 0
    down: int = 0
    flat: int = 0
    # Confusion matrix: predicted -> actual -> count
    confusion: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "UP": {"UP": 0, "DOWN": 0, "FLAT": 0},
        "DOWN": {"UP": 0, "DOWN": 0, "FLAT": 0},
        "FLAT": {"UP": 0, "DOWN": 0, "FLAT": 0},
    })

    def accuracy(self) -> float:
        return (self.correct / self.total) if self.total else 0.0

    def precision(self, direction: str) -> float:
        """Precision: Of all predicted as direction, how many were actually direction."""
        predicted = sum(self.confusion[direction].values())
        if predicted == 0:
            return 0.0
        return self.confusion[direction][direction] / predicted

    def recall(self, direction: str) -> float:
        """Recall: Of all actual direction, how many were predicted as direction."""
        actual = sum(self.confusion[d][direction] for d in ["UP", "DOWN", "FLAT"])
        if actual == 0:
            return 0.0
        return self.confusion[direction][direction] / actual

    def f1(self, direction: str) -> float:
        p = self.precision(direction)
        r = self.recall(direction)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


class BacktestStats:
    def __init__(self, model_names: List[str], interval_minutes: int = 1) -> None:
        self.total = 0
        self.correct = 0
        self.confidence_sum = 0.0
        self.decision_counts = {"UP": 0, "DOWN": 0, "FLAT": 0}
        self.actual_counts = {"UP": 0, "DOWN": 0, "FLAT": 0}
        self.models = {name: ModelStats() for name in model_names}
        self.interval_minutes = interval_minutes

        # Extended metrics
        self.confusion: Dict[str, Dict[str, int]] = {
            "UP": {"UP": 0, "DOWN": 0, "FLAT": 0},
            "DOWN": {"UP": 0, "DOWN": 0, "FLAT": 0},
            "FLAT": {"UP": 0, "DOWN": 0, "FLAT": 0},
        }
        self.confidence_buckets: Dict[str, List[int]] = {
            "0.50-0.55": [0, 0],  # [correct, total]
            "0.55-0.60": [0, 0],
            "0.60-0.65": [0, 0],
            "0.65-0.70": [0, 0],
            "0.70-0.80": [0, 0],
            "0.80-1.00": [0, 0],
        }
        self.returns: List[float] = []
        self.cumulative_return = 0.0
        self.max_drawdown = 0.0
        self.peak_return = 0.0
        self.win_streak = 0
        self.loss_streak = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0
        self.profitable_trades = 0
        self.losing_trades = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0

        # Extended calibration metrics
        self.calibration_data: List[Tuple[float, int]] = []  # (confidence, correct)
        self.confident_wrong_count = 0  # conf >= 0.7 but wrong
        self.confident_total = 0  # conf >= 0.7 total

        # Model agreement tracking
        self.agreement_counts = {"full": 0, "partial": 0, "none": 0}
        self.agreement_correct = {"full": 0, "partial": 0, "none": 0}

        # Rolling accuracy (for time-series analysis)
        self.rolling_window = 50
        self.rolling_correct: List[int] = []
        self.rolling_accuracies: List[float] = []

        # Price regime analysis
        self.volatility_buckets: Dict[str, List[int]] = {
            "low": [0, 0],      # [correct, total] for low volatility
            "medium": [0, 0],
            "high": [0, 0],
        }
        self.last_prices: List[float] = []

        # Regime-based tracking
        self.regime_counts: Dict[str, int] = {
            "trending_up": 0,
            "trending_down": 0,
            "ranging": 0,
            "volatile": 0,
        }
        self.regime_correct: Dict[str, int] = {
            "trending_up": 0,
            "trending_down": 0,
            "ranging": 0,
            "volatile": 0,
        }

        # Sprint 15: Filtered accuracy (high-confidence predictions only)
        self.filter_threshold = 0.55  # Will be set from config
        self.filtered_total = 0
        self.filtered_correct = 0
        self.filtered_confidence_sum = 0.0

    def _get_confidence_bucket(self, confidence: float) -> str:
        if confidence < 0.55:
            return "0.50-0.55"
        elif confidence < 0.60:
            return "0.55-0.60"
        elif confidence < 0.65:
            return "0.60-0.65"
        elif confidence < 0.70:
            return "0.65-0.70"
        elif confidence < 0.80:
            return "0.70-0.80"
        else:
            return "0.80-1.00"

    def update(
        self,
        prediction: PredictionRecord,
        outcome: Outcome,
        model_decisions: Dict[str, str],
        regime: Optional[str] = None,
    ) -> None:
        self.total += 1
        self.confidence_sum += prediction.decision.confidence
        pred_dir = prediction.decision.direction
        actual_dir = outcome.actual_direction

        self.decision_counts[pred_dir] += 1
        self.actual_counts[actual_dir] += 1
        self.confusion[pred_dir][actual_dir] += 1

        is_correct = pred_dir == actual_dir
        if is_correct:
            self.correct += 1

        # Sprint 15: Track filtered accuracy (high-confidence only)
        if prediction.decision.confidence >= self.filter_threshold:
            self.filtered_total += 1
            self.filtered_confidence_sum += prediction.decision.confidence
            if is_correct:
                self.filtered_correct += 1

        # Track regime statistics
        if regime and regime in self.regime_counts:
            self.regime_counts[regime] += 1
            if is_correct:
                self.regime_correct[regime] += 1

        # Confidence calibration
        bucket = self._get_confidence_bucket(prediction.decision.confidence)
        self.confidence_buckets[bucket][1] += 1
        if is_correct:
            self.confidence_buckets[bucket][0] += 1

        # Profit simulation (directional trading)
        trade_return = 0.0
        if pred_dir == "UP":
            trade_return = outcome.return_pct
        elif pred_dir == "DOWN":
            trade_return = -outcome.return_pct
        # FLAT = no trade

        if pred_dir != "FLAT":
            self.returns.append(trade_return)
            self.cumulative_return += trade_return
            self.peak_return = max(self.peak_return, self.cumulative_return)
            drawdown = self.peak_return - self.cumulative_return
            self.max_drawdown = max(self.max_drawdown, drawdown)

            if trade_return > 0:
                self.profitable_trades += 1
                self.gross_profit += trade_return
                self.win_streak += 1
                self.loss_streak = 0
                self.max_win_streak = max(self.max_win_streak, self.win_streak)
            elif trade_return < 0:
                self.losing_trades += 1
                self.gross_loss += abs(trade_return)
                self.loss_streak += 1
                self.win_streak = 0
                self.max_loss_streak = max(self.max_loss_streak, self.loss_streak)

        # Extended calibration tracking
        conf = prediction.decision.confidence
        self.calibration_data.append((conf, 1 if is_correct else 0))

        if conf >= 0.7:
            self.confident_total += 1
            if not is_correct:
                self.confident_wrong_count += 1

        # Rolling accuracy tracking
        self.rolling_correct.append(1 if is_correct else 0)
        if len(self.rolling_correct) >= self.rolling_window:
            roll_acc = sum(self.rolling_correct[-self.rolling_window:]) / self.rolling_window
            self.rolling_accuracies.append(roll_acc)

        # Per-model stats
        for name, decision in model_decisions.items():
            stats = self.models[name]
            stats.total += 1
            stats.confusion[decision][actual_dir] += 1
            if decision == actual_dir:
                stats.correct += 1
            if decision == "UP":
                stats.up += 1
            elif decision == "DOWN":
                stats.down += 1
            else:
                stats.flat += 1

        # Model agreement tracking
        directions = [d for d in model_decisions.values() if d != "FLAT"]
        if len(directions) >= 2:
            if len(set(directions)) == 1:
                self.agreement_counts["full"] += 1
                if is_correct:
                    self.agreement_correct["full"] += 1
            else:
                up_count = sum(1 for d in directions if d == "UP")
                down_count = sum(1 for d in directions if d == "DOWN")
                if up_count > 0 and down_count > 0:
                    self.agreement_counts["none"] += 1
                    if is_correct:
                        self.agreement_correct["none"] += 1
                else:
                    self.agreement_counts["partial"] += 1
                    if is_correct:
                        self.agreement_correct["partial"] += 1

    def precision(self, direction: str) -> float:
        predicted = sum(self.confusion[direction].values())
        if predicted == 0:
            return 0.0
        return self.confusion[direction][direction] / predicted

    def recall(self, direction: str) -> float:
        actual = sum(self.confusion[d][direction] for d in ["UP", "DOWN", "FLAT"])
        if actual == 0:
            return 0.0
        return self.confusion[direction][direction] / actual

    def f1(self, direction: str) -> float:
        p = self.precision(direction)
        r = self.recall(direction)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    def profit_factor(self) -> float:
        if self.gross_loss == 0:
            return float('inf') if self.gross_profit > 0 else 0.0
        return self.gross_profit / self.gross_loss

    def win_rate(self) -> float:
        total_trades = self.profitable_trades + self.losing_trades
        if total_trades == 0:
            return 0.0
        return self.profitable_trades / total_trades

    def ece(self, num_bins: int = 10) -> float:
        """Expected Calibration Error - measures calibration quality."""
        if not self.calibration_data:
            return 0.0

        bins: Dict[int, List[Tuple[float, int]]] = {i: [] for i in range(num_bins)}
        for conf, correct in self.calibration_data:
            bin_idx = min(int(conf * num_bins), num_bins - 1)
            bins[bin_idx].append((conf, correct))

        ece_sum = 0.0
        total = len(self.calibration_data)
        for bin_data in bins.values():
            if not bin_data:
                continue
            avg_conf = sum(c for c, _ in bin_data) / len(bin_data)
            avg_acc = sum(correct for _, correct in bin_data) / len(bin_data)
            ece_sum += (len(bin_data) / total) * abs(avg_conf - avg_acc)
        return ece_sum

    def mce(self, num_bins: int = 10) -> float:
        """Maximum Calibration Error - worst bin calibration."""
        if not self.calibration_data:
            return 0.0

        bins: Dict[int, List[Tuple[float, int]]] = {i: [] for i in range(num_bins)}
        for conf, correct in self.calibration_data:
            bin_idx = min(int(conf * num_bins), num_bins - 1)
            bins[bin_idx].append((conf, correct))

        max_error = 0.0
        for bin_data in bins.values():
            if not bin_data:
                continue
            avg_conf = sum(c for c, _ in bin_data) / len(bin_data)
            avg_acc = sum(correct for _, correct in bin_data) / len(bin_data)
            max_error = max(max_error, abs(avg_conf - avg_acc))
        return max_error

    def brier_score(self) -> float:
        """Brier score - measures probabilistic prediction quality."""
        if not self.calibration_data:
            return 0.0
        return sum((conf - correct) ** 2 for conf, correct in self.calibration_data) / len(self.calibration_data)

    def confident_wrong_rate(self) -> float:
        """Rate of high-confidence predictions that were wrong."""
        if self.confident_total == 0:
            return 0.0
        return self.confident_wrong_count / self.confident_total

    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio based on interval_minutes."""
        if len(self.returns) < 2:
            return 0.0
        mean_ret = sum(self.returns) / len(self.returns)
        var = sum((r - mean_ret) ** 2 for r in self.returns) / len(self.returns)
        std = math.sqrt(var) if var > 0 else 1e-10
        # Annualize: sqrt(periods_per_year), where periods_per_year = 525600 / interval_minutes
        periods_per_year = 525600 / self.interval_minutes
        annualization_factor = math.sqrt(periods_per_year)
        return (mean_ret / std) * annualization_factor

    def sortino_ratio(self) -> float:
        """Sortino ratio - Sharpe but only penalizes downside volatility."""
        if len(self.returns) < 2:
            return 0.0
        mean_ret = sum(self.returns) / len(self.returns)
        downside = [r for r in self.returns if r < 0]
        if not downside:
            return float('inf') if mean_ret > 0 else 0.0
        downside_var = sum(r ** 2 for r in downside) / len(downside)
        downside_std = math.sqrt(downside_var) if downside_var > 0 else 1e-10
        # Annualize: sqrt(periods_per_year), where periods_per_year = 525600 / interval_minutes
        periods_per_year = 525600 / self.interval_minutes
        annualization_factor = math.sqrt(periods_per_year)
        return (mean_ret / downside_std) * annualization_factor

    def direction_balance(self) -> Dict[str, float]:
        """Check for direction collapse (one direction dominating)."""
        total = self.decision_counts["UP"] + self.decision_counts["DOWN"]
        if total == 0:
            return {"up_ratio": 0.5, "down_ratio": 0.5, "balance_score": 1.0}
        up_ratio = self.decision_counts["UP"] / total
        down_ratio = self.decision_counts["DOWN"] / total
        # Balance score: 1.0 = perfect balance, 0.0 = complete collapse
        balance_score = 1.0 - abs(up_ratio - 0.5) * 2
        return {"up_ratio": up_ratio, "down_ratio": down_ratio, "balance_score": balance_score}

    def regime_accuracy(self) -> Dict[str, float]:
        """Accuracy broken down by market regime."""
        result = {}
        for regime in self.regime_counts:
            total = self.regime_counts[regime]
            correct = self.regime_correct[regime]
            result[regime] = correct / total if total > 0 else 0.0
        return result

    def rolling_accuracy_stats(self) -> Dict[str, float]:
        """Statistics on rolling accuracy over time."""
        if not self.rolling_accuracies:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "trend": 0.0}
        mean = sum(self.rolling_accuracies) / len(self.rolling_accuracies)
        var = sum((a - mean) ** 2 for a in self.rolling_accuracies) / len(self.rolling_accuracies)
        std = math.sqrt(var) if var > 0 else 0.0
        # Trend: positive = improving, negative = degrading
        if len(self.rolling_accuracies) >= 2:
            first_half = sum(self.rolling_accuracies[:len(self.rolling_accuracies)//2]) / (len(self.rolling_accuracies)//2)
            second_half = sum(self.rolling_accuracies[len(self.rolling_accuracies)//2:]) / (len(self.rolling_accuracies) - len(self.rolling_accuracies)//2)
            trend = second_half - first_half
        else:
            trend = 0.0
        return {
            "min": min(self.rolling_accuracies),
            "max": max(self.rolling_accuracies),
            "mean": mean,
            "std": std,
            "trend": trend,
        }

    def model_agreement_accuracy(self) -> Dict[str, float]:
        """Accuracy when models agree vs disagree."""
        result = {}
        for agreement_type in ["full", "partial", "none"]:
            total = self.agreement_counts[agreement_type]
            correct = self.agreement_correct[agreement_type]
            result[agreement_type] = correct / total if total > 0 else 0.0
        return result

    def summary(self) -> Dict[str, object]:
        accuracy = (self.correct / self.total) if self.total else 0.0
        avg_conf = (self.confidence_sum / self.total) if self.total else 0.0

        # Confidence calibration summary
        calibration = {}
        for bucket, (correct, total) in self.confidence_buckets.items():
            if total > 0:
                calibration[bucket] = {
                    "accuracy": correct / total,
                    "count": total,
                }

        # Direction balance check
        dir_balance = self.direction_balance()

        # Sharpe/Sortino with inf handling
        sharpe = self.sharpe_ratio()
        sortino = self.sortino_ratio()

        return {
            "total_predictions": self.total,
            "ensemble_accuracy": accuracy,
            "avg_confidence": avg_conf,
            "ensemble_counts": dict(self.decision_counts),
            "actual_counts": dict(self.actual_counts),
            "confusion_matrix": {k: dict(v) for k, v in self.confusion.items()},
            "precision": {d: self.precision(d) for d in ["UP", "DOWN", "FLAT"]},
            "recall": {d: self.recall(d) for d in ["UP", "DOWN", "FLAT"]},
            "f1_score": {d: self.f1(d) for d in ["UP", "DOWN", "FLAT"]},
            "confidence_calibration": calibration,
            # NEW: Extended calibration metrics
            "calibration_metrics": {
                "ece": self.ece(),
                "mce": self.mce(),
                "brier_score": self.brier_score(),
                "confident_wrong_rate": self.confident_wrong_rate(),
                "confident_wrong_count": self.confident_wrong_count,
                "confident_total": self.confident_total,
            },
            # Sprint 15: Filtered accuracy (high-confidence predictions only)
            "filtered_accuracy": {
                "threshold": self.filter_threshold,
                "total": self.filtered_total,
                "correct": self.filtered_correct,
                "accuracy": (self.filtered_correct / self.filtered_total) if self.filtered_total > 0 else 0.0,
                "avg_confidence": (self.filtered_confidence_sum / self.filtered_total) if self.filtered_total > 0 else 0.0,
                "coverage": (self.filtered_total / self.total) if self.total > 0 else 0.0,
            },
            # NEW: Direction balance analysis
            "direction_balance": {
                "up_ratio": dir_balance["up_ratio"],
                "down_ratio": dir_balance["down_ratio"],
                "balance_score": dir_balance["balance_score"],
                "collapse_warning": dir_balance["balance_score"] < 0.6,
            },
            # NEW: Model agreement analysis
            "model_agreement": {
                "counts": dict(self.agreement_counts),
                "accuracy": self.model_agreement_accuracy(),
            },
            # NEW: Rolling accuracy over time
            "rolling_accuracy": self.rolling_accuracy_stats(),
            # NEW: Regime-based accuracy
            "regime_analysis": {
                "counts": dict(self.regime_counts),
                "accuracy": self.regime_accuracy(),
            },
            "trading_metrics": {
                "total_trades": self.profitable_trades + self.losing_trades,
                "profitable_trades": self.profitable_trades,
                "losing_trades": self.losing_trades,
                "win_rate": self.win_rate(),
                "cumulative_return_pct": self.cumulative_return * 100,
                "max_drawdown_pct": self.max_drawdown * 100,
                "profit_factor": self.profit_factor(),
                "gross_profit_pct": self.gross_profit * 100,
                "gross_loss_pct": self.gross_loss * 100,
                "max_win_streak": self.max_win_streak,
                "max_loss_streak": self.max_loss_streak,
                # NEW: Risk-adjusted metrics
                "sharpe_ratio": sharpe if sharpe != float('inf') else None,
                "sortino_ratio": sortino if sortino != float('inf') else None,
            },
            "per_model": {
                name: {
                    "total": stats.total,
                    "accuracy": stats.accuracy(),
                    "counts": {"UP": stats.up, "DOWN": stats.down, "FLAT": stats.flat},
                    "balance_score": 1.0 - abs((stats.up / (stats.up + stats.down + 0.001)) - 0.5) * 2 if (stats.up + stats.down) > 0 else 1.0,
                    "precision": {d: stats.precision(d) for d in ["UP", "DOWN", "FLAT"]},
                    "recall": {d: stats.recall(d) for d in ["UP", "DOWN", "FLAT"]},
                    "f1_score": {d: stats.f1(d) for d in ["UP", "DOWN", "FLAT"]},
                    "confusion_matrix": {k: dict(v) for k, v in stats.confusion.items()},
                }
                for name, stats in self.models.items()
            },
        }


class OnlineStats:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min = None
        self.max = None

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        if self.min is None or value < self.min:
            self.min = value
        if self.max is None or value > self.max:
            self.max = value

    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    def std(self) -> float:
        return math.sqrt(self.variance())

    def summary(self) -> Dict[str, object]:
        return {
            "count": self.count,
            "mean": self.mean if self.count else 0.0,
            "std": self.std() if self.count else 0.0,
            "min": self.min,
            "max": self.max,
        }


class BinaryStats:
    def __init__(self) -> None:
        self.pos_count = 0
        self.pos_sum = 0.0
        self.neg_count = 0
        self.neg_sum = 0.0
        self.all_stats = OnlineStats()

    def update(self, value: float, label: int) -> None:
        self.all_stats.update(value)
        if label == 1:
            self.pos_count += 1
            self.pos_sum += value
        else:
            self.neg_count += 1
            self.neg_sum += value

    def mean_pos(self) -> float:
        return self.pos_sum / self.pos_count if self.pos_count else 0.0

    def mean_neg(self) -> float:
        return self.neg_sum / self.neg_count if self.neg_count else 0.0

    def effect_size(self) -> float:
        std = self.all_stats.std()
        if std == 0.0:
            return 0.0
        return (self.mean_pos() - self.mean_neg()) / std

    def correlation(self) -> float:
        total = self.pos_count + self.neg_count
        if total == 0:
            return 0.0
        std = self.all_stats.std()
        if std == 0.0:
            return 0.0
        p = self.pos_count / total
        q = self.neg_count / total
        return (self.mean_pos() - self.mean_neg()) / std * math.sqrt(p * q)

    def summary(self) -> Dict[str, object]:
        return {
            "count_total": self.pos_count + self.neg_count,
            "count_pos": self.pos_count,
            "count_neg": self.neg_count,
            "mean_pos": self.mean_pos(),
            "mean_neg": self.mean_neg(),
            "std": self.all_stats.std(),
            "correlation": self.correlation(),
            "effect_size": self.effect_size(),
        }


class FeatureAggregate:
    def __init__(self) -> None:
        self.total = 0
        self.by_direction = {
            "UP": OnlineStats(),
            "DOWN": OnlineStats(),
            "FLAT": OnlineStats(),
        }
        self.up_down = BinaryStats()

    def update(self, value: float, direction: str) -> None:
        self.total += 1
        self.by_direction[direction].update(value)
        if direction == "UP":
            self.up_down.update(value, 1)
        elif direction == "DOWN":
            self.up_down.update(value, 0)

    def summary(self) -> Dict[str, object]:
        return {
            "count_total": self.total,
            "by_direction": {k: v.summary() for k, v in self.by_direction.items()},
            "up_down": self.up_down.summary(),
        }


class ModelFeatureAggregate:
    def __init__(self) -> None:
        self.total = 0
        self.correct_stats = BinaryStats()

    def update(self, value: float, correct: bool) -> None:
        self.total += 1
        self.correct_stats.update(value, 1 if correct else 0)

    def summary(self) -> Dict[str, object]:
        total = self.correct_stats.pos_count + self.correct_stats.neg_count
        correct_rate = self.correct_stats.pos_count / total if total else 0.0
        summary = self.correct_stats.summary()
        summary["correct_rate"] = correct_rate
        return summary


class FeatureAnalyzer:
    def __init__(self, model_names: List[str]) -> None:
        self._overall: Dict[str, FeatureAggregate] = {}
        self._per_model: Dict[str, Dict[str, ModelFeatureAggregate]] = {
            name: {} for name in model_names
        }

    def update(
        self,
        features: Dict[str, float],
        actual_direction: str,
        model_decisions: Dict[str, str],
    ) -> None:
        for name, value in features.items():
            if not isinstance(value, (int, float)):
                continue
            aggregate = self._overall.setdefault(name, FeatureAggregate())
            aggregate.update(float(value), actual_direction)

            for model_name, decision in model_decisions.items():
                per_model = self._per_model[model_name]
                model_agg = per_model.setdefault(name, ModelFeatureAggregate())
                model_agg.update(float(value), decision == actual_direction)

    def _top_by_correlation(self, data: Dict[str, object], count: int) -> List[Dict[str, object]]:
        ranked = sorted(
            data.items(),
            key=lambda item: abs(item[1]["correlation"]),
            reverse=True,
        )
        top = []
        for name, stats in ranked[:count]:
            top.append(
                {
                    "feature": name,
                    "correlation": stats["correlation"],
                    "effect_size": stats["effect_size"],
                    "count": stats["count_total"],
                }
            )
        return top

    def summary(self, top_n: int = 5) -> Dict[str, object]:
        overall_features = {name: agg.summary() for name, agg in self._overall.items()}
        overall_rank = {name: agg.up_down.summary() for name, agg in self._overall.items()}

        per_model: Dict[str, object] = {}
        for model_name, features in self._per_model.items():
            model_features = {name: agg.summary() for name, agg in features.items()}
            model_rank = {name: agg.correct_stats.summary() for name, agg in features.items()}
            per_model[model_name] = {
                "features": model_features,
                "top_by_correlation": self._top_by_correlation(model_rank, top_n),
            }

        return {
            "overall": {
                "features": overall_features,
                "top_by_correlation": self._top_by_correlation(overall_rank, top_n),
            },
            "per_model": per_model,
        }


class DetailWriter:
    def __init__(self, path: str) -> None:
        self._handle = open(path, "w", encoding="utf-8")

    def write(
        self,
        prediction: PredictionRecord,
        outcome: Outcome,
        model_decisions: Dict[str, str],
    ) -> None:
        payload = {
            "ts": prediction.ts,
            "pattern_id": prediction.pattern_id,
            "features": prediction.features,
            "decision": {
                "direction": prediction.decision.direction,
                "confidence": prediction.decision.confidence,
                "prob_up": prediction.decision.prob_up,
                "prob_down": prediction.decision.prob_down,
            },
            "actual": {
                "direction": outcome.actual_direction,
                "price_delta": outcome.price_delta,
                "return_pct": outcome.return_pct,
            },
            "models": {
                output.model_name: {
                    "prob_up": output.prob_up,
                    "prob_down": output.prob_down,
                    "decision": model_decisions[output.model_name],
                    "state": output.state,
                    "metrics": output.metrics,
                }
                for output in prediction.outputs
            },
        }
        self._handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def close(self) -> None:
        self._handle.close()


def print_summary(stats: BacktestStats, run_meta: Optional[Dict[str, object]] = None) -> None:
    """Print comprehensive backtest summary to console."""

    def pct(val: float) -> str:
        return f"{val * 100:.2f}%"

    def num(val: float) -> str:
        return f"{val:.4f}"

    print("\n" + "=" * 70)
    print("                    BACKTEST SUMMARY")
    print("=" * 70)

    # Run metadata
    if run_meta:
        symbol = run_meta.get("symbol", "N/A")
        interval = run_meta.get("interval", "N/A")
        start_iso = run_meta.get("start_iso", "N/A")
        end_iso = run_meta.get("end_iso", "N/A")
        candle_count = run_meta.get("candles", "N/A")
        print(f"\n  Symbol: {symbol}  |  Interval: {interval}m")
        print(f"  Period: {start_iso} -> {end_iso}")
        print(f"  Candles: {candle_count}")

    # Ensemble accuracy
    accuracy = stats.correct / stats.total if stats.total else 0
    avg_conf = stats.confidence_sum / stats.total if stats.total else 0
    print("\n" + "-" * 70)
    print("  ENSEMBLE PERFORMANCE")
    print("-" * 70)
    print(f"  Total Predictions: {stats.total}")
    print(f"  Accuracy:          {pct(accuracy)}  ({stats.correct}/{stats.total})")
    print(f"  Avg Confidence:    {pct(avg_conf)}")

    # Direction distribution
    print("\n  Direction Distribution:")
    print(f"    Predicted:  UP={stats.decision_counts['UP']:4d}  DOWN={stats.decision_counts['DOWN']:4d}  FLAT={stats.decision_counts['FLAT']:4d}")
    print(f"    Actual:     UP={stats.actual_counts['UP']:4d}  DOWN={stats.actual_counts['DOWN']:4d}  FLAT={stats.actual_counts['FLAT']:4d}")

    # Confusion Matrix
    print("\n  Confusion Matrix (Predicted \\ Actual):")
    print("              UP    DOWN    FLAT")
    for pred in ["UP", "DOWN", "FLAT"]:
        row = stats.confusion[pred]
        print(f"    {pred:5s}  {row['UP']:5d}   {row['DOWN']:5d}   {row['FLAT']:5d}")

    # Precision / Recall / F1
    print("\n  Classification Metrics:")
    print("            Precision   Recall      F1")
    for d in ["UP", "DOWN", "FLAT"]:
        print(f"    {d:5s}   {pct(stats.precision(d)):>8s}   {pct(stats.recall(d)):>8s}   {pct(stats.f1(d)):>8s}")

    # Confidence Calibration
    print("\n  Confidence Calibration:")
    print("    Bucket       Accuracy    Count")
    for bucket, (correct, total) in stats.confidence_buckets.items():
        if total > 0:
            bucket_acc = correct / total
            print(f"    {bucket}    {pct(bucket_acc):>8s}     {total:4d}")

    # NEW: Extended Calibration Metrics
    print("\n" + "-" * 70)
    print("  CALIBRATION QUALITY")
    print("-" * 70)
    ece_val = stats.ece()
    mce_val = stats.mce()
    brier_val = stats.brier_score()
    cwr = stats.confident_wrong_rate()

    ece_status = "[OK]" if ece_val < 0.10 else ("[WARN]" if ece_val < 0.20 else "[POOR]")
    mce_status = "[OK]" if mce_val < 0.15 else ("[WARN]" if mce_val < 0.30 else "[POOR]")
    brier_status = "[OK]" if brier_val < 0.20 else ("[WARN]" if brier_val < 0.30 else "[POOR]")
    cwr_status = "[OK]" if cwr < 0.10 else ("[WARN]" if cwr < 0.30 else "[POOR]")

    print(f"  ECE (Expected Calibration Error):   {pct(ece_val):>8s}  {ece_status}")
    print(f"  MCE (Maximum Calibration Error):    {pct(mce_val):>8s}  {mce_status}")
    print(f"  Brier Score:                        {brier_val:.4f}    {brier_status}")
    print(f"  Confident Wrong Rate (conf>=70%):   {pct(cwr):>8s}  {cwr_status}  ({stats.confident_wrong_count}/{stats.confident_total})")

    # Sprint 15: Filtered Accuracy
    print("\n" + "-" * 70)
    print("  FILTERED ACCURACY (High-Confidence Only)")
    print("-" * 70)
    filt_acc = (stats.filtered_correct / stats.filtered_total) if stats.filtered_total > 0 else 0.0
    filt_conf = (stats.filtered_confidence_sum / stats.filtered_total) if stats.filtered_total > 0 else 0.0
    coverage = (stats.filtered_total / stats.total) if stats.total > 0 else 0.0
    filt_status = "[GREAT]" if filt_acc >= 0.65 else ("[OK]" if filt_acc >= 0.55 else "[NEEDS WORK]")
    print(f"  Threshold:         conf >= {stats.filter_threshold:.0%}")
    print(f"  Filtered Accuracy: {pct(filt_acc):>8s}  ({stats.filtered_correct}/{stats.filtered_total})  {filt_status}")
    print(f"  Avg Confidence:    {pct(filt_conf):>8s}")
    print(f"  Coverage:          {pct(coverage):>8s}  (of all predictions)")

    # NEW: Direction Balance Analysis
    print("\n" + "-" * 70)
    print("  DIRECTION BALANCE")
    print("-" * 70)
    dir_balance = stats.direction_balance()
    balance_status = "[BALANCED]" if dir_balance["balance_score"] >= 0.8 else (
        "[IMBALANCED]" if dir_balance["balance_score"] >= 0.6 else "[COLLAPSED]"
    )
    print(f"  UP Ratio:       {pct(dir_balance['up_ratio']):>8s}")
    print(f"  DOWN Ratio:     {pct(dir_balance['down_ratio']):>8s}")
    print(f"  Balance Score:  {dir_balance['balance_score']:.3f}   {balance_status}")
    if dir_balance["balance_score"] < 0.6:
        dominant = "UP" if dir_balance["up_ratio"] > 0.5 else "DOWN"
        print(f"  WARNING: Model is collapsing toward {dominant} predictions!")

    # NEW: Model Agreement Analysis
    print("\n" + "-" * 70)
    print("  MODEL AGREEMENT")
    print("-" * 70)
    agreement_acc = stats.model_agreement_accuracy()
    print(f"  Full Agreement:    {stats.agreement_counts['full']:4d} predictions  ->  Accuracy: {pct(agreement_acc['full']):>7s}")
    print(f"  Partial Agreement: {stats.agreement_counts['partial']:4d} predictions  ->  Accuracy: {pct(agreement_acc['partial']):>7s}")
    print(f"  No Agreement:      {stats.agreement_counts['none']:4d} predictions  ->  Accuracy: {pct(agreement_acc['none']):>7s}")

    # NEW: Rolling Accuracy Stats
    rolling_stats = stats.rolling_accuracy_stats()
    if rolling_stats["mean"] > 0:
        print("\n" + "-" * 70)
        print("  ACCURACY STABILITY (rolling window=50)")
        print("-" * 70)
        trend_indicator = "[IMPROVING]" if rolling_stats["trend"] > 0.02 else (
            "[DEGRADING]" if rolling_stats["trend"] < -0.02 else "[STABLE]"
        )
        print(f"  Min:   {pct(rolling_stats['min']):>8s}")
        print(f"  Max:   {pct(rolling_stats['max']):>8s}")
        print(f"  Mean:  {pct(rolling_stats['mean']):>8s}")
        print(f"  Std:   {pct(rolling_stats['std']):>8s}")
        print(f"  Trend: {rolling_stats['trend']:+.4f}  {trend_indicator}")

    # Trading Metrics
    print("\n" + "-" * 70)
    print("  TRADING SIMULATION (directional, no fees)")
    print("-" * 70)
    total_trades = stats.profitable_trades + stats.losing_trades
    print(f"  Total Trades:       {total_trades}")
    print(f"  Win/Loss:           {stats.profitable_trades} / {stats.losing_trades}")
    print(f"  Win Rate:           {pct(stats.win_rate())}")
    print(f"  Cumulative Return:  {stats.cumulative_return * 100:+.4f}%")
    print(f"  Max Drawdown:       {stats.max_drawdown * 100:.4f}%")
    pf = stats.profit_factor()
    pf_str = f"{pf:.2f}" if pf != float('inf') else "inf"
    print(f"  Profit Factor:      {pf_str}")
    print(f"  Gross Profit:       {stats.gross_profit * 100:+.4f}%")
    print(f"  Gross Loss:         {stats.gross_loss * 100:.4f}%")
    print(f"  Max Win Streak:     {stats.max_win_streak}")
    print(f"  Max Loss Streak:    {stats.max_loss_streak}")

    # NEW: Risk-adjusted metrics
    sharpe = stats.sharpe_ratio()
    sortino = stats.sortino_ratio()
    sharpe_str = f"{sharpe:.2f}" if sharpe != float('inf') else "inf"
    sortino_str = f"{sortino:.2f}" if sortino != float('inf') else "inf"
    sharpe_status = "[GOOD]" if sharpe > 1.5 else ("[OK]" if sharpe > 0.5 else "[POOR]")
    sortino_status = "[GOOD]" if sortino > 2.0 else ("[OK]" if sortino > 1.0 else "[POOR]")
    print(f"\n  Risk-Adjusted Metrics:")
    print(f"    Sharpe Ratio (annualized):  {sharpe_str:>8s}  {sharpe_status}")
    print(f"    Sortino Ratio (annualized): {sortino_str:>8s}  {sortino_status}")

    # Per-model performance
    print("\n" + "-" * 70)
    print("  PER-MODEL PERFORMANCE")
    print("-" * 70)
    for name, model_stats in stats.models.items():
        # Calculate balance score for this model
        model_total = model_stats.up + model_stats.down
        if model_total > 0:
            model_up_ratio = model_stats.up / model_total
            model_balance = 1.0 - abs(model_up_ratio - 0.5) * 2
        else:
            model_balance = 1.0
        balance_indicator = "+" if model_balance >= 0.8 else ("~" if model_balance >= 0.6 else "-")

        print(f"\n  [{name}]")
        print(f"    Accuracy: {pct(model_stats.accuracy()):>8s}  ({model_stats.correct}/{model_stats.total})")
        print(f"    Predictions: UP={model_stats.up}  DOWN={model_stats.down}  FLAT={model_stats.flat}  Balance: {model_balance:.2f} {balance_indicator}")
        print(f"    Precision:   UP={pct(model_stats.precision('UP')):>7s}  DOWN={pct(model_stats.precision('DOWN')):>7s}  FLAT={pct(model_stats.precision('FLAT')):>7s}")
        print(f"    Recall:      UP={pct(model_stats.recall('UP')):>7s}  DOWN={pct(model_stats.recall('DOWN')):>7s}  FLAT={pct(model_stats.recall('FLAT')):>7s}")
        print(f"    F1:          UP={pct(model_stats.f1('UP')):>7s}  DOWN={pct(model_stats.f1('DOWN')):>7s}  FLAT={pct(model_stats.f1('FLAT')):>7s}")
        # Mini confusion matrix
        print(f"    Confusion:   pred\\actual  UP    DOWN   FLAT")
        for pred_dir in ["UP", "DOWN", "FLAT"]:
            row = model_stats.confusion[pred_dir]
            print(f"                 {pred_dir:5s}       {row['UP']:4d}   {row['DOWN']:4d}   {row['FLAT']:4d}")

    # Statistical significance
    stat_result = StatisticalValidator.summary(stats.correct, stats.total)
    print("\n" + "-" * 70)
    print("  STATISTICAL SIGNIFICANCE")
    print("-" * 70)
    print(f"  Accuracy:       {pct(stat_result['accuracy'])} ({stat_result['correct']}/{stat_result['total']})")
    print(f"  95% CI:         {stat_result['ci_95_str']}")
    print(f"  p-value:        {stat_result['p_value']}")
    sig_status = "[SIGNIFICANT]" if stat_result['significant'] else "[NOT SIGNIFICANT]"
    print(f"  vs Random:      {sig_status}")
    print(f"  Interpretation: {stat_result['interpretation']}")

    # Final summary line
    print("\n" + "=" * 70)
    print("  OVERALL ASSESSMENT")
    print("=" * 70)
    overall_acc = stats.correct / stats.total if stats.total else 0
    acc_status = "[GOOD]" if overall_acc >= 0.55 else ("[MEDIOCRE]" if overall_acc >= 0.45 else "[POOR]")
    print(f"  Accuracy:    {pct(overall_acc):>8s}  {acc_status}")
    print(f"  Calibration: ECE={pct(ece_val):>7s}  {ece_status}")
    print(f"  Balance:     {dir_balance['balance_score']:.3f}     {balance_status}")
    if not stat_result['significant']:
        print("\n  WARNING: Results not statistically significant vs random!")
        print("    - Need more data or better model")
    if overall_acc < 0.45:
        print("\n  RECOMMENDATION: Model accuracy is below 45%. Consider:")
        print("    - Reviewing feature engineering")
        print("    - Checking for direction collapse")
        print("    - Increasing training data")
    elif ece_val > 0.20:
        print("\n  RECOMMENDATION: Model is poorly calibrated (ECE > 20%). Consider:")
        print("    - Adding calibration layer")
        print("    - Reducing overconfidence")
    print("\n" + "=" * 70)


def _model_decision(output: ModelOutput, config: ConfigStore) -> str:
    threshold = float(config.get("model.flat_threshold", 0.55))
    confidence = max(output.prob_up, output.prob_down)
    if confidence < threshold:
        return "FLAT"
    return "UP" if output.prob_up >= output.prob_down else "DOWN"


def _evaluate(prediction: PredictionRecord, next_close: float) -> Outcome:
    """Evaluate prediction outcome.

    Args:
        prediction: The prediction record
        next_close: The closing price of the next candle

    Returns:
        Outcome with actual direction, price delta, and return percentage.

    Note:
        Actual direction is ALWAYS UP or DOWN - the market always moves.
        FLAT is only valid for MODEL predictions (uncertainty), not for actual.
    """
    delta = next_close - prediction.price
    return_pct = (delta / prediction.price) if prediction.price else 0.0

    # Actual direction is always UP or DOWN (market always moves)
    if delta >= 0:
        direction = "UP"
    else:
        direction = "DOWN"

    return Outcome(
        actual_direction=direction,
        price_delta=delta,
        return_pct=return_pct,
    )


def generate_report_md(
    stats: BacktestStats,
    run_meta: Optional[Dict[str, object]] = None,
    feature_summary: Optional[Dict[str, object]] = None,
    advanced_analysis: Optional[Dict[str, object]] = None,
) -> str:
    """Generate a concise Markdown report for quick analysis."""

    def pct(val: float) -> str:
        return f"{val * 100:.1f}%"

    def status(val: float, good: float, warn: float, reverse: bool = False) -> str:
        if reverse:
            return "+" if val < good else ("~" if val < warn else "-")
        return "+" if val >= good else ("~" if val >= warn else "-")

    lines = ["# Backtest Report\n"]

    # Meta
    if run_meta:
        symbol = run_meta.get("symbol", "N/A")
        interval = run_meta.get("interval", "N/A")
        start_iso = run_meta.get("start_iso", "N/A")
        end_iso = run_meta.get("end_iso", "N/A")
        duration = run_meta.get("duration_hours", 0)
        lines.append(f"**Symbol:** {symbol} | **Interval:** {interval}m | **Duration:** {duration:.1f}h\n")
        lines.append(f"**Period:** {start_iso} -> {end_iso}\n")

    # Quick metrics
    acc = stats.correct / stats.total if stats.total else 0
    ece = stats.ece()
    cwr = stats.confident_wrong_rate()
    dir_balance = stats.direction_balance()
    sharpe = stats.sharpe_ratio()

    lines.append("\n## Quick Summary\n")
    lines.append("| Metric | Value | Status |")
    lines.append("|--------|-------|--------|")
    lines.append(f"| **Accuracy** | {pct(acc)} ({stats.correct}/{stats.total}) | {status(acc, 0.55, 0.45)} |")
    lines.append(f"| **ECE** | {pct(ece)} | {status(ece, 0.10, 0.20, reverse=True)} |")
    lines.append(f"| **Confident Wrong** | {pct(cwr)} | {status(cwr, 0.10, 0.30, reverse=True)} |")
    lines.append(f"| **Direction Balance** | {dir_balance['balance_score']:.2f} | {status(dir_balance['balance_score'], 0.8, 0.6)} |")
    lines.append(f"| **Win Rate** | {pct(stats.win_rate())} | {status(stats.win_rate(), 0.55, 0.45)} |")
    sharpe_str = f"{sharpe:.1f}" if sharpe != float('inf') else "inf"
    lines.append(f"| **Sharpe Ratio** | {sharpe_str} | {status(sharpe, 1.5, 0.5)} |")

    # Warnings
    warnings = []
    if acc < 0.45:
        warnings.append("**Low accuracy** - below 45%")
    if ece > 0.20:
        warnings.append("**Poor calibration** - ECE > 20%")
    if cwr > 0.30:
        warnings.append("**Overconfident** - too many confident wrong predictions")
    if dir_balance["balance_score"] < 0.6:
        dominant = "UP" if dir_balance["up_ratio"] > 0.5 else "DOWN"
        warnings.append(f"**Direction collapse** - model biased toward {dominant}")

    if warnings:
        lines.append("\n## Warnings\n")
        for w in warnings:
            lines.append(f"- {w}")

    # Direction distribution
    lines.append("\n## Direction Distribution\n")
    lines.append("| | UP | DOWN | FLAT |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Predicted | {stats.decision_counts['UP']} | {stats.decision_counts['DOWN']} | {stats.decision_counts['FLAT']} |")
    lines.append(f"| Actual | {stats.actual_counts['UP']} | {stats.actual_counts['DOWN']} | {stats.actual_counts['FLAT']} |")

    # Confusion matrix
    lines.append("\n## Confusion Matrix\n")
    lines.append("| pred \\ actual | UP | DOWN | FLAT |")
    lines.append("|---|---:|---:|---:|")
    for pred in ["UP", "DOWN", "FLAT"]:
        row = stats.confusion[pred]
        lines.append(f"| **{pred}** | {row['UP']} | {row['DOWN']} | {row['FLAT']} |")

    # Per-model summary
    lines.append("\n## Per-Model Performance\n")
    lines.append("| Model | Accuracy | UP | DOWN | FLAT | Balance |")
    lines.append("|-------|----------|---:|-----:|-----:|---------|")
    for name, model_stats in stats.models.items():
        model_total = model_stats.up + model_stats.down
        if model_total > 0:
            model_balance = 1.0 - abs((model_stats.up / model_total) - 0.5) * 2
        else:
            model_balance = 1.0
        bal_icon = status(model_balance, 0.8, 0.6)
        lines.append(f"| {name} | {pct(model_stats.accuracy())} | {model_stats.up} | {model_stats.down} | {model_stats.flat} | {model_balance:.2f} {bal_icon} |")

    # Model agreement
    agreement_acc = stats.model_agreement_accuracy()
    lines.append("\n## Model Agreement\n")
    lines.append("| Agreement | Count | Accuracy |")
    lines.append("|-----------|------:|----------|")
    lines.append(f"| Full | {stats.agreement_counts['full']} | {pct(agreement_acc['full'])} |")
    lines.append(f"| Partial | {stats.agreement_counts['partial']} | {pct(agreement_acc['partial'])} |")
    lines.append(f"| None | {stats.agreement_counts['none']} | {pct(agreement_acc['none'])} |")

    # Trading metrics
    lines.append("\n## Trading Simulation\n")
    pf = stats.profit_factor()
    pf_str = f"{pf:.2f}" if pf != float('inf') else "inf"
    sortino = stats.sortino_ratio()
    sortino_str = f"{sortino:.1f}" if sortino != float('inf') else "inf"

    lines.append(f"- **Trades:** {stats.profitable_trades + stats.losing_trades} (Win: {stats.profitable_trades}, Loss: {stats.losing_trades})")
    lines.append(f"- **Cumulative Return:** {stats.cumulative_return * 100:+.4f}%")
    lines.append(f"- **Max Drawdown:** {stats.max_drawdown * 100:.4f}%")
    lines.append(f"- **Profit Factor:** {pf_str}")
    lines.append(f"- **Sharpe:** {sharpe_str} | **Sortino:** {sortino_str}")

    # Rolling accuracy
    rolling = stats.rolling_accuracy_stats()
    if rolling["mean"] > 0:
        trend_icon = "^" if rolling["trend"] > 0.02 else ("v" if rolling["trend"] < -0.02 else "-")
        lines.append(f"\n## Accuracy Stability\n")
        lines.append(f"- **Range:** {pct(rolling['min'])} - {pct(rolling['max'])}")
        lines.append(f"- **Mean:** {pct(rolling['mean'])} ± {pct(rolling['std'])}")
        lines.append(f"- **Trend:** {rolling['trend']:+.3f} {trend_icon}")

    # Feature effectiveness
    if feature_summary:
        lines.append("\n## Feature Effectiveness (Top by correlation)\n")
        lines.append("| Scope | Feature | Correlation | Effect Size | Count |")
        lines.append("|-------|---------|------------:|------------:|------:|")

        overall = feature_summary.get("overall", {})
        for item in overall.get("top_by_correlation", []):
            lines.append(
                f"| Overall | {item['feature']} | {item['correlation']:.4f} | "
                f"{item['effect_size']:.4f} | {item['count']} |"
            )

        per_model = feature_summary.get("per_model", {})
        for model_name, model_data in per_model.items():
            for item in model_data.get("top_by_correlation", []):
                lines.append(
                    f"| {model_name} | {item['feature']} | {item['correlation']:.4f} | "
                    f"{item['effect_size']:.4f} | {item['count']} |"
                )

    # Advanced Analysis (NEW)
    if advanced_analysis:
        # Statistical significance
        stat = advanced_analysis.get("statistical", {})
        if stat:
            lines.append("\n## Statistical Significance\n")
            lines.append(f"- **Accuracy:** {pct(stat.get('accuracy', 0))} ({stat.get('correct', 0)}/{stat.get('total', 0)})")
            lines.append(f"- **95% CI:** {stat.get('ci_95_str', 'N/A')}")
            lines.append(f"- **p-value:** {stat.get('p_value', 1.0)}")
            sig_icon = "+" if stat.get("significant") else "-"
            lines.append(f"- **vs Random:** {sig_icon} {stat.get('interpretation', '')}")

        # Temporal analysis
        temporal = advanced_analysis.get("temporal", {})
        if temporal:
            by_session = temporal.get("by_session", {})
            if by_session:
                lines.append("\n## Accuracy by Session\n")
                lines.append("| Session | Accuracy | Count |")
                lines.append("|---------|----------|------:|")
                for session in ["asia", "europe", "us"]:
                    data = by_session.get(session, {})
                    if data.get("total", 0) > 0:
                        lines.append(f"| {session.upper()} | {pct(data['accuracy'])} | {data['total']} |")

            best = temporal.get("best_hour", {})
            worst = temporal.get("worst_hour", {})
            if best.get("total", 0) > 0:
                lines.append(f"\n- **Best hour:** {best['hour']}:00 UTC ({pct(best['accuracy'])})")
            if worst.get("total", 0) > 0:
                lines.append(f"- **Worst hour:** {worst['hour']}:00 UTC ({pct(worst['accuracy'])})")

        # Magnitude analysis
        magnitude = advanced_analysis.get("magnitude", {})
        if magnitude:
            by_mag = magnitude.get("by_magnitude", {})
            if by_mag:
                lines.append("\n## Accuracy by Movement Size\n")
                lines.append("| Size | Accuracy | Count | % of Total |")
                lines.append("|------|----------|------:|------------|")
                for bucket in ["tiny", "small", "medium", "large"]:
                    data = by_mag.get(bucket, {})
                    if data.get("total", 0) > 0:
                        lines.append(f"| {bucket} | {pct(data['accuracy'])} | {data['total']} | {pct(data['pct_of_total'])} |")

        # Error streaks
        streaks = advanced_analysis.get("streaks", {})
        if streaks:
            error_info = streaks.get("error_streaks", {})
            if error_info.get("count", 0) > 0:
                lines.append("\n## Error Streaks (3+)\n")
                lines.append(f"- **Count:** {error_info['count']}")
                lines.append(f"- **Max length:** {error_info['max_length']}")
                lines.append(f"- **Avg length:** {error_info['avg_length']:.1f}")
                regimes = error_info.get("common_regimes", {})
                if regimes:
                    top_regime = max(regimes.items(), key=lambda x: x[1])
                    lines.append(f"- **Common regime:** {top_regime[0]} ({top_regime[1]} streaks)")

        # Errors analysis
        errors = advanced_analysis.get("errors", {})
        if errors:
            conf_wrong = errors.get("confident_wrong", {})
            if conf_wrong.get("count", 0) > 0:
                lines.append("\n## Confident Wrong Predictions (conf >= 65%)\n")
                lines.append(f"- **Count:** {conf_wrong['count']}")
                lines.append(f"- **Avg confidence:** {pct(conf_wrong['avg_confidence'])}")

            by_regime = errors.get("by_regime", {})
            if by_regime:
                lines.append("\n## Error Rate by Regime\n")
                lines.append("| Regime | Errors | Total | Error Rate |")
                lines.append("|--------|-------:|------:|------------|")
                for regime, data in sorted(by_regime.items(), key=lambda x: -x[1].get("error_rate", 0)):
                    if data.get("total", 0) > 0:
                        lines.append(f"| {regime} | {data['errors']} | {data['total']} | {pct(data['error_rate'])} |")

    return "\n".join(lines)


def _tune_weights(stats: BacktestStats, config: ConfigStore) -> Dict[str, float]:
    min_weight = float(config.get("weights.min_weight", 0.1))
    weights = {}
    total = 0.0
    for name, model_stats in stats.models.items():
        weight = max(model_stats.accuracy(), min_weight)
        weights[name] = weight
        total += weight

    if total <= 0:
        total = 1.0

    return {name: weight / total for name, weight in weights.items()}


def run_backtest(
    csv_path: str,
    db_path: str,
    out_dir: str,
    limit: Optional[int] = None,
    tune_weights: bool = True,
    run_meta: Optional[Dict[str, object]] = None,
    target_start_ts: Optional[int] = None,
    target_end_ts: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Run backtest and return summary dict.

    Args:
        csv_path: Path to OHLCV CSV file
        db_path: SQLite database path
        out_dir: Output directory for results
        limit: Optional limit on rows to process
        tune_weights: Whether to tune model weights after backtest
        run_meta: Optional metadata to include in summary
        target_start_ts: Only record predictions at/after this ts (UTC seconds)
        target_end_ts: Only record predictions at/before this ts (UTC seconds)
        verbose: Print progress and summary to console

    Returns:
        Summary dictionary with all metrics
    """
    os.makedirs(out_dir, exist_ok=True)

    state_store = StateStore(db_path)
    config_store = ConfigStore(state_store)
    config_store.ensure_defaults()
    # Sprint 12: Pass config for config-driven pattern behavior
    pattern_store = PatternStore(db_path, config=config_store)

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

    # Create regime detector and performance monitor
    regime_detector = RegimeDetector(config_store)
    performance_monitor = PerformanceMonitor(window=100)

    # Use adaptive weight manager with regime support
    weight_manager = AdaptiveWeightManager(
        state_store,
        regime_detector=regime_detector,
        monitor=performance_monitor,
        performance_blend=0.3,  # 30% performance, 70% base regime weights
    )

    models = [
        TrendVIC(config_store),
        Oscillator(config_store),
        VolumeMetrix(config_store),
    ]

    # Sprint 14: Create ML classifier for directional prediction
    ml_classifier = create_ml_classifier()
    ml_enabled = ml_classifier is not None and HAS_LIGHTGBM
    ml_train_interval = int(config_store.get("ml.train_interval", 1000))  # Retrain every N samples
    ml_min_samples = int(config_store.get("ml.min_samples", 500))  # Min samples to start training
    ml_last_train_count = 0
    if verbose and ml_enabled:
        print(f"[Backtest] ML Classifier enabled (LightGBM)")
    elif verbose:
        print(f"[Backtest] ML Classifier disabled (LightGBM not available)")

    feature_stream = FeatureStream(config_store)
    ensemble = Ensemble(
        config_store,
        weight_manager,
        regime_detector=regime_detector,
        pattern_adjuster=pattern_adjuster,
    )
    calibrator = OnlineCalibrator(config_store)

    # Extract interval from run_meta for proper Sharpe/Sortino annualization
    interval_minutes = 1  # default
    if run_meta and "interval" in run_meta:
        try:
            interval_minutes = int(run_meta["interval"])
        except (ValueError, TypeError):
            interval_minutes = 1
    # Build model names list including ML classifier if enabled
    model_names = [model.name for model in models]
    if ml_enabled:
        model_names.append("ML_CLASSIFIER")
    stats = BacktestStats(model_names, interval_minutes=interval_minutes)
    # Sprint 15: Set filter threshold from config
    stats.filter_threshold = float(config_store.get("confidence_filter.threshold", 0.55))
    feature_analyzer = FeatureAnalyzer(model_names)
    prediction_analyzer = PredictionAnalyzer()  # NEW: Advanced analysis
    details_path = os.path.join(out_dir, "predictions.jsonl")
    detail_writer = DetailWriter(details_path)

    pending: Optional[PredictionRecord] = None
    candle_count = 0
    skipped_predictions = 0
    start_time = time.time()

    if verbose:
        print(f"\n[Backtest] Processing {csv_path}")
        print(f"[Backtest] Output: {out_dir}")

    def _in_target(ts: int) -> bool:
        if target_start_ts is not None and ts < target_start_ts:
            return False
        if target_end_ts is not None and ts > target_end_ts:
            return False
        return True

    for candle in CsvCandleReader(csv_path, limit=limit):
        candle_count += 1

        # Progress indicator every 100 candles
        if verbose and candle_count % 100 == 0:
            elapsed = time.time() - start_time
            rate = candle_count / elapsed if elapsed > 0 else 0
            acc = (stats.correct / stats.total * 100) if stats.total > 0 else 0
            sys.stdout.write(
                f"\r[Backtest] Candles: {candle_count:,}  |  "
                f"Predictions: {stats.total:,}  |  "
                f"Accuracy: {acc:.1f}%  |  "
                f"Rate: {rate:.0f}/s"
            )
            sys.stdout.flush()

        features = feature_stream.update(candle)
        if features is None:
            continue

        if pending is not None:
            outcome = _evaluate(pending, candle.close)
            raw_confidence = max(pending.decision.prob_up, pending.decision.prob_down)
            calibrator.update(
                raw_confidence,
                pending.decision.direction == outcome.actual_direction,
            )
            if _in_target(pending.ts):
                # Detect regime from pending prediction's features
                pending_regime = regime_detector.detect(pending.features)

                model_decisions: Dict[str, str] = {}

                for output in pending.outputs:
                    model_direction = _model_decision(output, config_store)
                    model_decisions[output.model_name] = model_direction

                    # Update performance monitor for adaptive weighting
                    performance_monitor.update(
                        model_name=output.model_name,
                        predicted=model_direction,
                        actual=outcome.actual_direction,
                        regime=pending_regime,
                        confidence=max(output.prob_up, output.prob_down),
                    )

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
                        "regime": pending_regime,
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
                    "regime": pending_regime,
                }
                pattern_store.record_usage(
                    pending.pattern_id,
                    "ENSEMBLE",
                    ensemble_event,
                    event_ts=pending.ts,
                    features_snapshot=pending.features,
                )

                stats.update(pending, outcome, model_decisions, regime=pending_regime)
                feature_analyzer.update(pending.features, outcome.actual_direction, model_decisions)
                detail_writer.write(pending, outcome, model_decisions)

                # Sprint 14: Add training sample to ML classifier
                if ml_enabled and ml_classifier is not None:
                    ml_classifier.add_training_sample(pending.features, outcome.actual_direction)
                    # Periodically retrain the model
                    samples_since_train = ml_classifier.training_samples - ml_last_train_count
                    if (ml_classifier.training_samples >= ml_min_samples and
                            samples_since_train >= ml_train_interval):
                        if ml_classifier.train():
                            ml_last_train_count = ml_classifier.training_samples
                            if verbose and ml_classifier.training_samples == ml_min_samples + samples_since_train:
                                print(f"\n[Backtest] ML Classifier trained on {ml_classifier.training_samples} samples")

                # NEW: Add to prediction analyzer for advanced analysis
                pred_detail = PredictionDetail(
                    ts=pending.ts,
                    predicted=pending.decision.direction,
                    actual=outcome.actual_direction,
                    confidence=pending.decision.confidence,
                    return_pct=outcome.return_pct,
                    regime=pending_regime,
                    features=pending.features,
                    model_outputs={
                        o.model_name: {"prob_up": o.prob_up, "prob_down": o.prob_down}
                        for o in pending.outputs
                    },
                )
                prediction_analyzer.add(pred_detail)
            else:
                skipped_predictions += 1

        # Get pattern_id first so we can pass it to ensemble.decide()
        # Sprint 12: Pass ts for extended conditions (hour, session, day_of_week)
        conditions = None
        pattern_id = -1
        if _in_target(candle.ts):
            conditions = build_conditions(features, config_store, ts=candle.ts)
            pattern_id = pattern_store.get_or_create(conditions, ts=candle.ts)

        pattern_contexts: Dict[str, Optional[PatternContext]] = {}
        if pattern_id > 0 and conditions is not None:
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

        # Sprint 14: Add ML classifier output if trained
        if ml_enabled and ml_classifier is not None and ml_classifier.is_trained:
            ml_output = ml_classifier.predict(features)
            outputs.append(ml_output)
        if pattern_id > 0 and model_adjuster_enabled:
            outputs = [
                pattern_model_adjuster.adjust_model_output(
                    output, pattern_id, conditions=conditions, ts=candle.ts
                )
                for output in outputs
            ]
        # Sprint 13: Pass pattern_id for experience-based adjustment
        decision = ensemble.decide(
            outputs,
            features,
            ts=candle.ts,
            pattern_id=pattern_id if pattern_id > 0 else None,
        )
        decision = Decision(
            direction=decision.direction,
            confidence=calibrator.calibrate(decision.confidence),
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

    detail_writer.close()

    elapsed = time.time() - start_time
    if verbose:
        print(f"\r[Backtest] Completed {candle_count:,} candles in {elapsed:.1f}s" + " " * 30)

    summary = stats.summary()
    summary["feature_analysis"] = feature_analyzer.summary()
    summary["online_calibration"] = calibrator.summary()
    summary["run_meta"] = {
        "input_csv": csv_path,
        "db_path": db_path,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "processing_time_sec": round(elapsed, 2),
        "candles_processed": candle_count,
        "skipped_predictions": skipped_predictions,
    }
    if run_meta:
        summary["run_meta"].update(run_meta)
    summary["weights"] = weight_manager.get_model_weights()
    summary["performance_monitor"] = performance_monitor.get_statistics()

    # Sprint 14: Add ML classifier info
    if ml_enabled and ml_classifier is not None:
        summary["ml_classifier"] = {
            "enabled": True,
            "is_trained": ml_classifier.is_trained,
            "training_samples": ml_classifier.training_samples,
            "feature_importance": ml_classifier.get_feature_importance() if ml_classifier.is_trained else {},
        }
    else:
        summary["ml_classifier"] = {"enabled": False}

    # NEW: Add advanced analysis
    summary["advanced_analysis"] = prediction_analyzer.full_summary()

    if tune_weights:
        tuned = _tune_weights(stats, config_store)
        weight_manager.set_model_weights(tuned)
        summary["tuned_weights"] = tuned

    # Sprint 12: Run pattern lifecycle maintenance
    lifecycle = pattern_store.run_lifecycle_maintenance()
    summary["pattern_lifecycle"] = lifecycle

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=True, indent=2))

    # Generate and save Markdown report
    report_md = generate_report_md(
        stats,
        run_meta,
        feature_summary=summary["feature_analysis"],
        advanced_analysis=summary["advanced_analysis"],
    )
    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report_md)

    # Print comprehensive summary
    if verbose:
        print_summary(stats, run_meta)
        print(f"\n  Output files:")
        print(f"    - {report_path}")
        print(f"    - {summary_path}")
        print(f"    - {details_path}")
        if tune_weights:
            print(f"\n  Weights tuned based on accuracy:")
            for name, weight in tuned.items():
                print(f"    {name}: {weight:.4f}")
        print()

    return summary
