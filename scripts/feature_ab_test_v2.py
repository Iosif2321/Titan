#!/usr/bin/env python3
"""
Feature A/B Testing Framework V2 - Uses REAL model logic

Key improvements over V1:
1. Uses actual model classes (TrendVIC, Oscillator, VolumeMetrix)
2. Integrates PatternContext for pattern-based adjustments
3. Uses real Ensemble logic with regime detection
4. Includes calibration (ConfidenceCompressor)
5. Accounts for feature relationships via pattern system

Usage:
    python scripts/feature_ab_test_v2.py --hours 168
    python scripts/feature_ab_test_v2.py --test-all --hours 168
"""

import argparse
import copy
import json
import sys
import time as time_module
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from titan.core.config import ConfigStore
from titan.core.data.bybit_rest import fetch_klines
from titan.core.features.stream import FeatureStream
from titan.core.state_store import StateStore
from titan.core.types import ModelOutput
from titan.core.models.heuristic import TrendVIC, Oscillator, VolumeMetrix
from titan.core.regime import RegimeDetector
from titan.core.calibration import ConfidenceCompressor
from titan.core.patterns import PatternStore


@dataclass
class TestResultV2:
    """Result of a single test run with detailed metrics."""
    name: str
    model_accuracies: Dict[str, float]
    ensemble_accuracy: float
    high_conf_accuracy: float  # conf >= 55%
    high_conf_count: int
    total_count: int
    flat_rate: float = 0.0
    ece: float = 0.0
    regime_accuracies: Dict[str, float] = field(default_factory=dict)

    def __str__(self):
        return (f"{self.name}: Ens={self.ensemble_accuracy:.2f}%, "
                f"TREND={self.model_accuracies.get('TRENDVIC', 0)*100:.1f}%, "
                f"OSC={self.model_accuracies.get('OSCILLATOR', 0)*100:.1f}%, "
                f"VOL={self.model_accuracies.get('VOLUMEMETRIX', 0)*100:.1f}%, "
                f"FLAT={self.flat_rate:.1f}%, ECE={self.ece:.2f}%")


class ModelWrapper:
    """Wrapper that allows modifying which features a model uses."""

    def __init__(self, model, feature_filter: Optional[List[str]] = None):
        self.model = model
        self.name = model.name
        self.feature_filter = feature_filter

    def predict(self, features: Dict[str, float], pattern_context=None) -> ModelOutput:
        # If we have a feature filter, zero out features not in the list
        if self.feature_filter:
            filtered = {}
            for key, value in features.items():
                if key in self.feature_filter:
                    filtered[key] = value
                else:
                    # Keep essential features that are always needed
                    if key in ('close', 'open', 'high', 'low', 'volume', 'timestamp'):
                        filtered[key] = value
                    else:
                        filtered[key] = 0.0  # Zero out non-selected features
            features = filtered

        return self.model.predict(features, pattern_context)


class FeatureABTesterV2:
    """
    A/B testing framework using REAL model logic.

    This version:
    - Uses actual TrendVIC, Oscillator, VolumeMetrix classes
    - Integrates PatternContext (can be disabled for cleaner comparison)
    - Uses real regime detection
    - Applies calibration
    """

    def __init__(
        self,
        hours: int = 168,
        symbol: str = "BTCUSDT",
        interval: int = 1,
        use_patterns: bool = False,  # Disable for cleaner A/B comparison
        use_calibration: bool = True,
    ):
        self.hours = hours
        self.symbol = symbol
        self.interval = interval
        self.use_patterns = use_patterns
        self.use_calibration = use_calibration

        self.state_store = StateStore(":memory:")
        self.config = ConfigStore(self.state_store)
        self.config.ensure_defaults()

        self.candles = None
        self.features_list = None
        self.actuals = None
        self.regimes = None
        self.timestamps = None
        self.results: List[TestResultV2] = []

        self.pattern_store = PatternStore(":memory:") if use_patterns else None
        self.regime_detector = RegimeDetector(self.config)
        self.calibrator = ConfidenceCompressor(self.config) if use_calibration else None

    def load_data(self):
        """Load and prepare data with regime info."""
        print(f"\n{'='*70}")
        print(f"  Loading {self.hours}h of {self.symbol} {self.interval}m data...")
        print(f"  Patterns: {'ON' if self.use_patterns else 'OFF'}")
        print(f"  Calibration: {'ON' if self.use_calibration else 'OFF'}")
        print(f"{'='*70}")

        stream = FeatureStream(self.config)

        end_ts = int(time_module.time())
        start_ts = end_ts - (self.hours * 3600) - (100 * self.interval * 60)
        self.candles = fetch_klines(self.symbol, str(self.interval), start_ts, end_ts)

        if not self.candles:
            raise ValueError("No candles fetched")

        print(f"  Fetched {len(self.candles)} candles")

        # Prepare data
        self.features_list = []
        self.actuals = []
        self.regimes = []
        self.timestamps = []

        warmup = 40
        for i, candle in enumerate(self.candles[:-1]):
            features = stream.update(candle)
            if features is None or i < warmup:
                continue

            next_candle = self.candles[i + 1]
            actual = "UP" if next_candle.close > candle.close else "DOWN"
            regime = self.regime_detector.detect(features)

            self.features_list.append(features)
            self.actuals.append(actual)
            self.regimes.append(regime)
            self.timestamps.append(candle.ts)

        print(f"  Prepared {len(self.features_list)} test samples")

        # Regime distribution
        regime_counts = {}
        for r in self.regimes:
            regime_counts[r] = regime_counts.get(r, 0) + 1
        print(f"  Regimes: {regime_counts}")

    def _create_models(self, feature_configs: Dict[str, List[str]]) -> Dict[str, ModelWrapper]:
        """Create model wrappers with feature filters."""
        base_models = {
            "TRENDVIC": TrendVIC(self.config),
            "OSCILLATOR": Oscillator(self.config),
            "VOLUMEMETRIX": VolumeMetrix(self.config),
        }

        wrapped = {}
        for name, model in base_models.items():
            feature_filter = feature_configs.get(name)
            wrapped[name] = ModelWrapper(model, feature_filter)

        return wrapped

    def _get_ensemble_decision(
        self,
        outputs: Dict[str, ModelOutput],
        regime: str,
    ) -> Tuple[str, float, float]:
        """
        Get ensemble decision using real logic.
        Returns (direction, confidence, prob_up)
        """
        # Regime-based weights
        regime_weights = {
            "trending_up":   {"TRENDVIC": 0.50, "OSCILLATOR": 0.20, "VOLUMEMETRIX": 0.30},
            "trending_down": {"TRENDVIC": 0.50, "OSCILLATOR": 0.20, "VOLUMEMETRIX": 0.30},
            "ranging":       {"TRENDVIC": 0.20, "OSCILLATOR": 0.50, "VOLUMEMETRIX": 0.30},
            "volatile":      {"TRENDVIC": 0.25, "OSCILLATOR": 0.25, "VOLUMEMETRIX": 0.50},
        }

        weights = regime_weights.get(regime, regime_weights["ranging"])

        # Weighted average
        prob_up = sum(outputs[m].prob_up * weights[m] for m in outputs)
        prob_down = sum(outputs[m].prob_down * weights[m] for m in outputs)

        # Agreement check
        directions = []
        for m, output in outputs.items():
            if output.prob_up > output.prob_down:
                directions.append("UP")
            elif output.prob_down > output.prob_up:
                directions.append("DOWN")
            else:
                directions.append("FLAT")

        up_votes = directions.count("UP")
        down_votes = directions.count("DOWN")

        # Agreement boost
        if up_votes == 3 or down_votes == 3:
            # Full agreement - boost confidence
            if prob_up > prob_down:
                prob_up = min(prob_up + 0.05, 0.65)
                prob_down = 1.0 - prob_up
            else:
                prob_down = min(prob_down + 0.05, 0.65)
                prob_up = 1.0 - prob_down
        elif up_votes >= 2 or down_votes >= 2:
            # Partial agreement - small boost
            if prob_up > prob_down:
                prob_up = min(prob_up + 0.02, 0.62)
                prob_down = 1.0 - prob_up
            else:
                prob_down = min(prob_down + 0.02, 0.62)
                prob_up = 1.0 - prob_down

        # Direction
        if prob_up > prob_down:
            direction = "UP"
            confidence = prob_up
        elif prob_down > prob_up:
            direction = "DOWN"
            confidence = prob_down
        else:
            direction = "FLAT"
            confidence = 0.5

        # Calibration
        if self.calibrator:
            confidence = self.calibrator.compress(confidence)

        return direction, confidence, prob_up

    def run_test(self, name: str, feature_configs: Dict[str, List[str]]) -> TestResultV2:
        """Run a single test with given feature configurations."""
        models = self._create_models(feature_configs)

        # Tracking
        model_correct = {m: 0 for m in models}
        model_total = {m: 0 for m in models}
        model_flat = {m: 0 for m in models}
        ensemble_correct = 0
        ensemble_flat = 0
        high_conf_correct = 0
        high_conf_total = 0

        # ECE calculation
        confidence_bins = {i: {"correct": 0, "total": 0} for i in range(10)}

        # Regime tracking
        regime_correct = {}
        regime_total = {}

        for features, actual, regime, ts in zip(
            self.features_list, self.actuals, self.regimes, self.timestamps
        ):
            # Get model outputs
            outputs = {}
            for model_name, model in models.items():
                output = model.predict(features, None)  # No pattern context for clean comparison
                outputs[model_name] = output

                # Model accuracy
                model_total[model_name] += 1
                if output.prob_up > output.prob_down:
                    pred = "UP"
                elif output.prob_down > output.prob_up:
                    pred = "DOWN"
                else:
                    pred = "FLAT"
                    model_flat[model_name] += 1

                if pred == actual:
                    model_correct[model_name] += 1

            # Ensemble decision
            direction, confidence, prob_up = self._get_ensemble_decision(outputs, regime)

            if direction == "FLAT":
                ensemble_flat += 1

            if direction == actual:
                ensemble_correct += 1

            # High confidence
            if confidence >= 0.55:
                high_conf_total += 1
                if direction == actual:
                    high_conf_correct += 1

            # ECE
            bin_idx = min(int(confidence * 10), 9)
            confidence_bins[bin_idx]["total"] += 1
            if direction == actual:
                confidence_bins[bin_idx]["correct"] += 1

            # Regime accuracy
            if regime not in regime_correct:
                regime_correct[regime] = 0
                regime_total[regime] = 0
            regime_total[regime] += 1
            if direction == actual:
                regime_correct[regime] += 1

        total = len(self.features_list)

        # Calculate ECE
        ece = 0.0
        for bin_idx, data in confidence_bins.items():
            if data["total"] > 0:
                bin_conf = (bin_idx + 0.5) / 10
                bin_acc = data["correct"] / data["total"]
                ece += abs(bin_acc - bin_conf) * data["total"]
        ece = (ece / total) * 100 if total > 0 else 0.0

        result = TestResultV2(
            name=name,
            model_accuracies={m: model_correct[m] / model_total[m] if model_total[m] > 0 else 0 for m in models},
            ensemble_accuracy=ensemble_correct / total * 100 if total > 0 else 0,
            high_conf_accuracy=high_conf_correct / high_conf_total * 100 if high_conf_total > 0 else 0,
            high_conf_count=high_conf_total,
            total_count=total,
            flat_rate=ensemble_flat / total * 100 if total > 0 else 0,
            ece=ece,
            regime_accuracies={r: regime_correct[r] / regime_total[r] * 100 for r in regime_total},
        )

        self.results.append(result)
        return result

    def run_baseline(self) -> TestResultV2:
        """Run baseline test with no feature filtering (all features)."""
        print("\n" + "=" * 70)
        print("  BASELINE TEST (all features, real model logic)")
        print("=" * 70)
        result = self.run_test("BASELINE", {})  # Empty = no filtering
        print(f"  {result}")
        return result

    def run_feature_ablation(self) -> Dict[str, TestResultV2]:
        """
        Run ablation study: remove one feature at a time from each model.
        This shows which features are actually important.
        """
        baseline = self.run_baseline()

        # Define features used by each model (from heuristic.py)
        model_features = {
            "TRENDVIC": ["ma_delta", "volatility", "close", "body_ratio",
                        "candle_direction", "price_momentum_3"],
            "OSCILLATOR": ["rsi", "rsi_momentum", "volatility_z"],
            "VOLUMEMETRIX": ["volume_z", "return_1", "volatility", "ma_delta",
                           "volume_trend", "upper_wick_ratio", "lower_wick_ratio"],
        }

        print("\n" + "=" * 70)
        print("  FEATURE ABLATION STUDY")
        print("  (Remove one feature at a time - if accuracy drops, feature is useful)")
        print("=" * 70)

        ablation_results = {}

        for model_name, features in model_features.items():
            print(f"\n  === {model_name} ===")

            for remove_feature in features:
                # Create config with this feature removed
                remaining = [f for f in features if f != remove_feature]
                test_name = f"{model_name}_without_{remove_feature}"

                result = self.run_test(test_name, {model_name: remaining})
                ablation_results[test_name] = result

                # Compare
                model_baseline = baseline.model_accuracies[model_name] * 100
                model_new = result.model_accuracies[model_name] * 100
                delta = model_new - model_baseline

                status = "NEEDED" if delta < -0.5 else "OPTIONAL" if delta < 0.5 else "HARMFUL"
                print(f"    Without {remove_feature:20s}: {model_new:.2f}% ({delta:+.2f}%) - {status}")

        return ablation_results

    def run_feature_addition(self) -> Dict[str, TestResultV2]:
        """
        Run addition study: add new features one at a time.
        Tests potential improvements.
        """
        baseline = self.run_baseline()

        # Additional features to try for each model
        additions = {
            "TRENDVIC": ["ema_10_spread_pct", "ema_20_spread_pct", "return_5",
                        "upper_wick_ratio", "lower_wick_ratio", "atr_pct"],
            "OSCILLATOR": ["volume_z", "return_1", "rsi_oversold", "rsi_overbought",
                          "body_ratio", "candle_direction"],
            "VOLUMEMETRIX": ["body_ratio", "atr_pct", "volume_change_pct",
                            "candle_direction", "rsi"],
        }

        # Current features
        current = {
            "TRENDVIC": ["ma_delta", "volatility", "close", "body_ratio",
                        "candle_direction", "price_momentum_3"],
            "OSCILLATOR": ["rsi", "rsi_momentum", "volatility_z"],
            "VOLUMEMETRIX": ["volume_z", "return_1", "volatility", "ma_delta",
                           "volume_trend", "upper_wick_ratio", "lower_wick_ratio"],
        }

        print("\n" + "=" * 70)
        print("  FEATURE ADDITION STUDY")
        print("  (Add one feature at a time - if accuracy rises, feature is useful)")
        print("=" * 70)

        addition_results = {}

        for model_name, new_features in additions.items():
            print(f"\n  === {model_name} ===")
            base_features = current[model_name]

            for add_feature in new_features:
                if add_feature in base_features:
                    continue  # Already have it

                # Create config with this feature added
                expanded = base_features + [add_feature]
                test_name = f"{model_name}_plus_{add_feature}"

                result = self.run_test(test_name, {model_name: expanded})
                addition_results[test_name] = result

                # Compare
                model_baseline = baseline.model_accuracies[model_name] * 100
                model_new = result.model_accuracies[model_name] * 100
                delta = model_new - model_baseline

                status = "USEFUL" if delta > 0.5 else "NEUTRAL" if delta > -0.5 else "HARMFUL"
                print(f"    Plus {add_feature:20s}: {model_new:.2f}% ({delta:+.2f}%) - {status}")

        return addition_results

    def run_all_tests(self) -> Dict[str, TestResultV2]:
        """Run complete analysis."""
        print("\n" + "=" * 70)
        print("  COMPLETE A/B FEATURE ANALYSIS V2")
        print("  (Using REAL model logic)")
        print("=" * 70)

        # 1. Baseline
        baseline = self.run_baseline()

        # 2. Ablation study
        ablation = self.run_feature_ablation()

        # 3. Addition study
        addition = self.run_feature_addition()

        # Summary
        print("\n" + "=" * 70)
        print("  SUMMARY")
        print("=" * 70)

        print(f"\n  BASELINE: {baseline.ensemble_accuracy:.2f}% ensemble")
        print(f"           ECE: {baseline.ece:.2f}%, FLAT: {baseline.flat_rate:.1f}%")

        # Find best improvements
        all_results = {r.name: r for r in self.results}

        improvements = []
        for name, result in all_results.items():
            if name != "BASELINE":
                delta = result.ensemble_accuracy - baseline.ensemble_accuracy
                improvements.append((name, delta, result))

        improvements.sort(key=lambda x: x[1], reverse=True)

        print("\n  TOP 10 IMPROVEMENTS:")
        for name, delta, result in improvements[:10]:
            print(f"    {name:40s}: {delta:+.2f}% (ens: {result.ensemble_accuracy:.2f}%)")

        print("\n  BOTTOM 5 (HARMFUL):")
        for name, delta, result in improvements[-5:]:
            print(f"    {name:40s}: {delta:+.2f}% (ens: {result.ensemble_accuracy:.2f}%)")

        return all_results

    def generate_recommendations(self) -> Dict[str, Any]:
        """Generate actionable recommendations based on results."""
        if not self.results:
            self.run_all_tests()

        baseline = next(r for r in self.results if r.name == "BASELINE")

        recommendations = {
            "TRENDVIC": {"keep": [], "remove": [], "add": []},
            "OSCILLATOR": {"keep": [], "remove": [], "add": []},
            "VOLUMEMETRIX": {"keep": [], "remove": [], "add": []},
        }

        for result in self.results:
            if result.name == "BASELINE":
                continue

            for model in recommendations:
                if result.name.startswith(model):
                    delta = result.model_accuracies[model] - baseline.model_accuracies[model]
                    delta_pct = delta * 100

                    if "_without_" in result.name:
                        feature = result.name.split("_without_")[1]
                        if delta_pct < -0.5:
                            recommendations[model]["keep"].append((feature, delta_pct))
                        elif delta_pct > 0.5:
                            recommendations[model]["remove"].append((feature, delta_pct))

                    elif "_plus_" in result.name:
                        feature = result.name.split("_plus_")[1]
                        if delta_pct > 0.5:
                            recommendations[model]["add"].append((feature, delta_pct))

        print("\n" + "=" * 70)
        print("  ACTIONABLE RECOMMENDATIONS")
        print("=" * 70)

        for model, recs in recommendations.items():
            print(f"\n  {model}:")

            if recs["keep"]:
                print("    KEEP (removing hurts):")
                for f, d in sorted(recs["keep"], key=lambda x: x[1]):
                    print(f"      - {f}: {d:+.2f}% when removed")

            if recs["remove"]:
                print("    REMOVE (removing helps):")
                for f, d in sorted(recs["remove"], key=lambda x: -x[1]):
                    print(f"      - {f}: {d:+.2f}% when removed")

            if recs["add"]:
                print("    ADD (adding helps):")
                for f, d in sorted(recs["add"], key=lambda x: -x[1]):
                    print(f"      + {f}: {d:+.2f}% when added")

        return recommendations

    def save_results(self, path: str = "ab_test_v2_results.json"):
        """Save results to JSON."""
        data = {
            "version": 2,
            "hours": self.hours,
            "symbol": self.symbol,
            "use_patterns": self.use_patterns,
            "use_calibration": self.use_calibration,
            "total_samples": len(self.features_list) if self.features_list else 0,
            "results": [
                {
                    "name": r.name,
                    "ensemble_accuracy": r.ensemble_accuracy,
                    "model_accuracies": r.model_accuracies,
                    "high_conf_accuracy": r.high_conf_accuracy,
                    "high_conf_count": r.high_conf_count,
                    "flat_rate": r.flat_rate,
                    "ece": r.ece,
                    "regime_accuracies": r.regime_accuracies,
                }
                for r in self.results
            ]
        }

        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"\nResults saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Feature A/B Testing V2 (Real Model Logic)")
    parser.add_argument("--hours", type=int, default=168, help="Hours of data")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol")
    parser.add_argument("--interval", type=int, default=1, help="Interval in minutes")
    parser.add_argument("--ablation", action="store_true", help="Run ablation only")
    parser.add_argument("--addition", action="store_true", help="Run addition only")
    parser.add_argument("--test-all", action="store_true", help="Run all tests")
    parser.add_argument("--with-patterns", action="store_true", help="Enable pattern context")
    parser.add_argument("--no-calibration", action="store_true", help="Disable calibration")

    args = parser.parse_args()

    tester = FeatureABTesterV2(
        hours=args.hours,
        symbol=args.symbol,
        interval=args.interval,
        use_patterns=args.with_patterns,
        use_calibration=not args.no_calibration,
    )
    tester.load_data()

    if args.ablation:
        tester.run_baseline()
        tester.run_feature_ablation()
    elif args.addition:
        tester.run_baseline()
        tester.run_feature_addition()
    elif args.test_all:
        tester.run_all_tests()
        tester.generate_recommendations()
    else:
        tester.run_baseline()

    tester.save_results()


if __name__ == "__main__":
    main()
