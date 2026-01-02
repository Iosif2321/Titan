"""Advanced analysis tools for prediction system debugging and improvement."""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class PredictionDetail:
    """Detailed prediction record for analysis."""

    ts: int
    predicted: str  # "UP", "DOWN", "FLAT"
    actual: str  # "UP", "DOWN"
    confidence: float
    return_pct: float
    regime: Optional[str] = None
    features: Optional[Dict[str, float]] = None
    model_outputs: Optional[Dict[str, Dict[str, float]]] = None

    @property
    def is_correct(self) -> bool:
        return self.predicted == self.actual

    @property
    def hour_utc(self) -> int:
        return datetime.fromtimestamp(self.ts, tz=timezone.utc).hour

    @property
    def day_of_week(self) -> int:
        """0=Monday, 6=Sunday"""
        return datetime.fromtimestamp(self.ts, tz=timezone.utc).weekday()

    @property
    def session(self) -> str:
        """Trading session based on hour UTC."""
        hour = self.hour_utc
        if 0 <= hour < 8:
            return "asia"
        elif 8 <= hour < 14:
            return "europe"
        else:
            return "us"

    @property
    def magnitude_bucket(self) -> str:
        """Size of actual price movement."""
        abs_ret = abs(self.return_pct) * 100  # to percentage
        if abs_ret < 0.01:
            return "tiny"
        elif abs_ret < 0.05:
            return "small"
        elif abs_ret < 0.10:
            return "medium"
        else:
            return "large"


@dataclass
class StreakInfo:
    """Information about a streak of correct/incorrect predictions."""

    streak_type: str  # "correct" or "error"
    length: int
    start_ts: int
    end_ts: int
    predictions: List[PredictionDetail] = field(default_factory=list)

    @property
    def common_regime(self) -> Optional[str]:
        """Most common regime in this streak."""
        if not self.predictions:
            return None
        regimes = [p.regime for p in self.predictions if p.regime]
        if not regimes:
            return None
        from collections import Counter
        return Counter(regimes).most_common(1)[0][0]

    @property
    def avg_confidence(self) -> float:
        if not self.predictions:
            return 0.0
        return sum(p.confidence for p in self.predictions) / len(self.predictions)


class TemporalAnalyzer:
    """Analyze prediction accuracy by time segments."""

    def __init__(self) -> None:
        self._hourly: Dict[int, List[bool]] = defaultdict(list)
        self._daily: Dict[int, List[bool]] = defaultdict(list)
        self._session: Dict[str, List[bool]] = defaultdict(list)

    def add(self, pred: PredictionDetail) -> None:
        is_correct = pred.is_correct
        self._hourly[pred.hour_utc].append(is_correct)
        self._daily[pred.day_of_week].append(is_correct)
        self._session[pred.session].append(is_correct)

    def accuracy_by_hour(self) -> Dict[int, Dict[str, Any]]:
        """Accuracy for each hour (0-23 UTC)."""
        result = {}
        for hour in range(24):
            data = self._hourly.get(hour, [])
            if data:
                correct = sum(data)
                total = len(data)
                result[hour] = {
                    "accuracy": correct / total,
                    "correct": correct,
                    "total": total,
                }
            else:
                result[hour] = {"accuracy": 0.0, "correct": 0, "total": 0}
        return result

    def accuracy_by_day(self) -> Dict[str, Dict[str, Any]]:
        """Accuracy for each day of week."""
        day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        result = {}
        for day_idx, day_name in enumerate(day_names):
            data = self._daily.get(day_idx, [])
            if data:
                correct = sum(data)
                total = len(data)
                result[day_name] = {
                    "accuracy": correct / total,
                    "correct": correct,
                    "total": total,
                }
            else:
                result[day_name] = {"accuracy": 0.0, "correct": 0, "total": 0}
        return result

    def accuracy_by_session(self) -> Dict[str, Dict[str, Any]]:
        """Accuracy by trading session (Asia/Europe/US)."""
        result = {}
        for session in ["asia", "europe", "us"]:
            data = self._session.get(session, [])
            if data:
                correct = sum(data)
                total = len(data)
                result[session] = {
                    "accuracy": correct / total,
                    "correct": correct,
                    "total": total,
                }
            else:
                result[session] = {"accuracy": 0.0, "correct": 0, "total": 0}
        return result

    def summary(self) -> Dict[str, Any]:
        """Full temporal analysis summary."""
        hourly = self.accuracy_by_hour()
        by_session = self.accuracy_by_session()
        by_day = self.accuracy_by_day()

        # Find best/worst hours
        valid_hours = [(h, d) for h, d in hourly.items() if d["total"] > 0]
        if valid_hours:
            best_hour = max(valid_hours, key=lambda x: x[1]["accuracy"])
            worst_hour = min(valid_hours, key=lambda x: x[1]["accuracy"])
        else:
            best_hour = (0, {"accuracy": 0.0})
            worst_hour = (0, {"accuracy": 0.0})

        return {
            "by_hour": hourly,
            "by_session": by_session,
            "by_day": by_day,
            "best_hour": {"hour": best_hour[0], **best_hour[1]},
            "worst_hour": {"hour": worst_hour[0], **worst_hour[1]},
        }


class MagnitudeAnalyzer:
    """Analyze prediction accuracy by price movement magnitude."""

    def __init__(self) -> None:
        self._buckets: Dict[str, List[bool]] = defaultdict(list)
        self._returns: Dict[str, List[float]] = defaultdict(list)

    def add(self, pred: PredictionDetail) -> None:
        bucket = pred.magnitude_bucket
        self._buckets[bucket].append(pred.is_correct)
        self._returns[bucket].append(pred.return_pct)

    def accuracy_by_magnitude(self) -> Dict[str, Dict[str, Any]]:
        """Accuracy for each magnitude bucket."""
        result = {}
        for bucket in ["tiny", "small", "medium", "large"]:
            data = self._buckets.get(bucket, [])
            returns = self._returns.get(bucket, [])
            if data:
                correct = sum(data)
                total = len(data)
                avg_return = sum(abs(r) for r in returns) / len(returns) * 100
                result[bucket] = {
                    "accuracy": correct / total,
                    "correct": correct,
                    "total": total,
                    "avg_return_pct": avg_return,
                    "pct_of_total": 0.0,  # filled later
                }
            else:
                result[bucket] = {
                    "accuracy": 0.0,
                    "correct": 0,
                    "total": 0,
                    "avg_return_pct": 0.0,
                    "pct_of_total": 0.0,
                }

        # Calculate percentage of total
        grand_total = sum(d["total"] for d in result.values())
        if grand_total > 0:
            for bucket in result:
                result[bucket]["pct_of_total"] = result[bucket]["total"] / grand_total

        return result

    def summary(self) -> Dict[str, Any]:
        by_mag = self.accuracy_by_magnitude()

        # Find where we're best/worst
        valid = [(b, d) for b, d in by_mag.items() if d["total"] > 10]
        if valid:
            best = max(valid, key=lambda x: x[1]["accuracy"])
            worst = min(valid, key=lambda x: x[1]["accuracy"])
        else:
            best = ("unknown", {"accuracy": 0.0})
            worst = ("unknown", {"accuracy": 0.0})

        return {
            "by_magnitude": by_mag,
            "best_bucket": {"bucket": best[0], **best[1]},
            "worst_bucket": {"bucket": worst[0], **worst[1]},
        }


class StreakAnalyzer:
    """Analyze streaks of correct/incorrect predictions."""

    def __init__(self) -> None:
        self._predictions: List[PredictionDetail] = []
        self._error_streaks: List[StreakInfo] = []
        self._correct_streaks: List[StreakInfo] = []

    def add(self, pred: PredictionDetail) -> None:
        self._predictions.append(pred)

    def analyze(self) -> None:
        """Compute streaks from collected predictions."""
        if not self._predictions:
            return

        self._error_streaks = []
        self._correct_streaks = []

        current_streak: List[PredictionDetail] = []
        current_type: Optional[bool] = None

        for pred in self._predictions:
            is_correct = pred.is_correct

            if current_type is None:
                current_type = is_correct
                current_streak = [pred]
            elif is_correct == current_type:
                current_streak.append(pred)
            else:
                # Streak ended, save it
                self._save_streak(current_streak, current_type)
                current_type = is_correct
                current_streak = [pred]

        # Save last streak
        if current_streak:
            self._save_streak(current_streak, current_type)

    def _save_streak(self, preds: List[PredictionDetail], is_correct: bool) -> None:
        if len(preds) < 2:
            return  # Only track streaks of 2+

        streak = StreakInfo(
            streak_type="correct" if is_correct else "error",
            length=len(preds),
            start_ts=preds[0].ts,
            end_ts=preds[-1].ts,
            predictions=preds,
        )

        if is_correct:
            self._correct_streaks.append(streak)
        else:
            self._error_streaks.append(streak)

    def get_error_streaks(self, min_length: int = 3) -> List[StreakInfo]:
        """Get error streaks of at least min_length."""
        return [s for s in self._error_streaks if s.length >= min_length]

    def get_correct_streaks(self, min_length: int = 3) -> List[StreakInfo]:
        """Get correct streaks of at least min_length."""
        return [s for s in self._correct_streaks if s.length >= min_length]

    def summary(self) -> Dict[str, Any]:
        self.analyze()

        error_streaks = self.get_error_streaks(min_length=3)
        correct_streaks = self.get_correct_streaks(min_length=3)

        # Analyze what's common in error streaks
        error_regimes: Dict[str, int] = defaultdict(int)
        for streak in error_streaks:
            if streak.common_regime:
                error_regimes[streak.common_regime] += 1

        return {
            "error_streaks": {
                "count": len(error_streaks),
                "max_length": max((s.length for s in error_streaks), default=0),
                "avg_length": (
                    sum(s.length for s in error_streaks) / len(error_streaks)
                    if error_streaks
                    else 0
                ),
                "common_regimes": dict(error_regimes),
                "details": [
                    {
                        "length": s.length,
                        "start_ts": s.start_ts,
                        "regime": s.common_regime,
                        "avg_confidence": round(s.avg_confidence, 4),
                    }
                    for s in sorted(error_streaks, key=lambda x: -x.length)[:5]
                ],
            },
            "correct_streaks": {
                "count": len(correct_streaks),
                "max_length": max((s.length for s in correct_streaks), default=0),
                "avg_length": (
                    sum(s.length for s in correct_streaks) / len(correct_streaks)
                    if correct_streaks
                    else 0
                ),
            },
        }


class StatisticalValidator:
    """Statistical tests for prediction significance."""

    @staticmethod
    def binomial_test(correct: int, total: int, null_prob: float = 0.5) -> float:
        """
        Calculate p-value for accuracy vs null hypothesis.

        Uses normal approximation for large samples.
        Returns p-value (two-tailed).
        """
        if total < 10:
            return 1.0  # Not enough data

        observed_prob = correct / total
        expected = total * null_prob
        std = math.sqrt(total * null_prob * (1 - null_prob))

        if std == 0:
            return 1.0

        z = abs(correct - expected) / std

        # Two-tailed p-value using normal approximation
        # Using approximation: p ≈ 2 * (1 - Φ(z)) ≈ 2 * exp(-0.5 * z^2) / sqrt(2π) for large z
        # Simplified approximation for common cases
        if z < 0.5:
            p_value = 0.6
        elif z < 1.0:
            p_value = 0.3
        elif z < 1.5:
            p_value = 0.13
        elif z < 1.96:
            p_value = 0.05
        elif z < 2.5:
            p_value = 0.01
        elif z < 3.0:
            p_value = 0.003
        else:
            p_value = 0.001

        return round(p_value, 4)

    @staticmethod
    def confidence_interval(
        accuracy: float, n: int, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Wilson score interval for proportion.

        Returns (lower, upper) bounds.
        """
        if n == 0:
            return (0.0, 1.0)

        # Z-score for confidence level
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_scores.get(confidence, 1.96)

        # Wilson score interval
        denominator = 1 + z * z / n
        center = (accuracy + z * z / (2 * n)) / denominator
        spread = z * math.sqrt((accuracy * (1 - accuracy) + z * z / (4 * n)) / n) / denominator

        lower = max(0.0, center - spread)
        upper = min(1.0, center + spread)

        return (round(lower, 4), round(upper, 4))

    @staticmethod
    def is_significant(correct: int, total: int, threshold: float = 0.05) -> bool:
        """Check if accuracy is significantly different from random."""
        p_value = StatisticalValidator.binomial_test(correct, total)
        return p_value < threshold

    @staticmethod
    def summary(correct: int, total: int) -> Dict[str, Any]:
        if total == 0:
            return {
                "accuracy": 0.0,
                "ci_95": (0.0, 1.0),
                "p_value": 1.0,
                "significant": False,
                "interpretation": "No data",
            }

        accuracy = correct / total
        ci = StatisticalValidator.confidence_interval(accuracy, total)
        p_value = StatisticalValidator.binomial_test(correct, total)
        significant = p_value < 0.05

        if not significant:
            interpretation = "Not significantly different from random (50%)"
        elif accuracy > 0.5:
            interpretation = f"Significantly BETTER than random (p={p_value})"
        else:
            interpretation = f"Significantly WORSE than random (p={p_value})"

        return {
            "accuracy": round(accuracy, 4),
            "correct": correct,
            "total": total,
            "ci_95": ci,
            "ci_95_str": f"{ci[0]*100:.1f}% - {ci[1]*100:.1f}%",
            "p_value": p_value,
            "significant": significant,
            "interpretation": interpretation,
        }


class ErrorExplorer:
    """Tools for exploring and understanding prediction errors."""

    def __init__(self) -> None:
        self._all: List[PredictionDetail] = []
        self._errors: List[PredictionDetail] = []
        self._correct: List[PredictionDetail] = []

    def add(self, pred: PredictionDetail) -> None:
        self._all.append(pred)
        if pred.is_correct:
            self._correct.append(pred)
        else:
            self._errors.append(pred)

    def get_confident_wrong(self, min_confidence: float = 0.65) -> List[PredictionDetail]:
        """Get high-confidence predictions that were wrong."""
        return sorted(
            [e for e in self._errors if e.confidence >= min_confidence],
            key=lambda x: -x.confidence,
        )

    def get_worst_errors(self, n: int = 20) -> List[PredictionDetail]:
        """Get the most confident wrong predictions."""
        return sorted(self._errors, key=lambda x: -x.confidence)[:n]

    def filter_errors(
        self,
        regime: Optional[str] = None,
        confidence_min: Optional[float] = None,
        confidence_max: Optional[float] = None,
        magnitude: Optional[str] = None,
        session: Optional[str] = None,
    ) -> List[PredictionDetail]:
        """Filter errors by various criteria."""
        result = self._errors

        if regime:
            result = [e for e in result if e.regime == regime]
        if confidence_min is not None:
            result = [e for e in result if e.confidence >= confidence_min]
        if confidence_max is not None:
            result = [e for e in result if e.confidence <= confidence_max]
        if magnitude:
            result = [e for e in result if e.magnitude_bucket == magnitude]
        if session:
            result = [e for e in result if e.session == session]

        return result

    def feature_comparison(self) -> Dict[str, Dict[str, float]]:
        """Compare average feature values for correct vs wrong predictions."""
        if not self._correct or not self._errors:
            return {}

        # Get all feature names
        feature_names: set = set()
        for pred in self._all:
            if pred.features:
                feature_names.update(pred.features.keys())

        result = {}
        for feature in feature_names:
            correct_vals = [
                p.features[feature]
                for p in self._correct
                if p.features and feature in p.features
            ]
            error_vals = [
                p.features[feature]
                for p in self._errors
                if p.features and feature in p.features
            ]

            if correct_vals and error_vals:
                correct_mean = sum(correct_vals) / len(correct_vals)
                error_mean = sum(error_vals) / len(error_vals)
                result[feature] = {
                    "when_correct": round(correct_mean, 4),
                    "when_wrong": round(error_mean, 4),
                    "difference": round(error_mean - correct_mean, 4),
                }

        # Sort by absolute difference
        result = dict(
            sorted(result.items(), key=lambda x: abs(x[1]["difference"]), reverse=True)
        )
        return result

    def error_distribution_by_regime(self) -> Dict[str, Dict[str, Any]]:
        """Distribution of errors across regimes."""
        regime_errors: Dict[str, int] = defaultdict(int)
        regime_totals: Dict[str, int] = defaultdict(int)

        for pred in self._all:
            if pred.regime:
                regime_totals[pred.regime] += 1
                if not pred.is_correct:
                    regime_errors[pred.regime] += 1

        result = {}
        for regime in regime_totals:
            total = regime_totals[regime]
            errors = regime_errors[regime]
            result[regime] = {
                "total": total,
                "errors": errors,
                "error_rate": round(errors / total, 4) if total > 0 else 0.0,
            }

        return result

    def summary(self) -> Dict[str, Any]:
        confident_wrong = self.get_confident_wrong(0.65)
        worst = self.get_worst_errors(10)

        return {
            "total_predictions": len(self._all),
            "total_errors": len(self._errors),
            "error_rate": (
                round(len(self._errors) / len(self._all), 4) if self._all else 0.0
            ),
            "confident_wrong": {
                "count": len(confident_wrong),
                "avg_confidence": (
                    sum(e.confidence for e in confident_wrong) / len(confident_wrong)
                    if confident_wrong
                    else 0.0
                ),
            },
            "worst_errors": [
                {
                    "ts": e.ts,
                    "predicted": e.predicted,
                    "actual": e.actual,
                    "confidence": round(e.confidence, 4),
                    "regime": e.regime,
                }
                for e in worst[:5]
            ],
            "by_regime": self.error_distribution_by_regime(),
            "feature_comparison": dict(list(self.feature_comparison().items())[:5]),
        }


class PredictionAnalyzer:
    """Main analyzer that combines all analysis tools."""

    def __init__(self) -> None:
        self.temporal = TemporalAnalyzer()
        self.magnitude = MagnitudeAnalyzer()
        self.streaks = StreakAnalyzer()
        self.errors = ErrorExplorer()
        self._all_predictions: List[PredictionDetail] = []

    def add(self, pred: PredictionDetail) -> None:
        """Add a prediction to all analyzers."""
        self._all_predictions.append(pred)
        self.temporal.add(pred)
        self.magnitude.add(pred)
        self.streaks.add(pred)
        self.errors.add(pred)

    def full_summary(self) -> Dict[str, Any]:
        """Generate comprehensive analysis summary."""
        # Basic stats
        total = len(self._all_predictions)
        correct = sum(1 for p in self._all_predictions if p.is_correct)

        return {
            "statistical": StatisticalValidator.summary(correct, total),
            "temporal": self.temporal.summary(),
            "magnitude": self.magnitude.summary(),
            "streaks": self.streaks.summary(),
            "errors": self.errors.summary(),
        }
