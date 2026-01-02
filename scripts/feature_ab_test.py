#!/usr/bin/env python3
"""
Feature A/B Testing Framework for Titan Models

Systematically tests feature combinations for each model:
1. Run baseline test
2. Make one change at a time
3. Compare results
4. Keep improvement or rollback

Usage:
    python scripts/feature_ab_test.py --hours 72
    python scripts/feature_ab_test.py --test-all --hours 72
"""

import argparse
import copy
import json
import sys
import time as time_module
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

from titan.core.config import ConfigStore
from titan.core.data.bybit_rest import fetch_klines
from titan.core.features.stream import FeatureStream
from titan.core.state_store import StateStore
from titan.core.types import ModelOutput


@dataclass
class TestResult:
    """Result of a single test run."""
    name: str
    model_accuracies: Dict[str, float]
    ensemble_accuracy: float
    high_conf_accuracy: float  # conf >= 55%
    high_conf_count: int
    total_count: int

    def __str__(self):
        return (f"{self.name}: Ensemble={self.ensemble_accuracy:.1f}%, "
                f"TREND={self.model_accuracies.get('TRENDVIC', 0)*100:.1f}%, "
                f"OSC={self.model_accuracies.get('OSCILLATOR', 0)*100:.1f}%, "
                f"VOL={self.model_accuracies.get('VOLUMEMETRIX', 0)*100:.1f}%, "
                f"HighConf={self.high_conf_accuracy:.1f}% ({self.high_conf_count})")


class ConfigurableModel:
    """Model with configurable feature set."""

    def __init__(self, name: str, features: List[str], logic: Callable):
        self.name = name
        self.features = features
        self.logic = logic

    def predict(self, all_features: Dict[str, float]) -> Tuple[float, float, Dict]:
        """Returns (prob_up, prob_down, state)."""
        # Extract only the features this model uses
        model_features = {f: all_features.get(f, 0.0) for f in self.features}
        return self.logic(model_features, all_features)


# ============== MODEL LOGIC FUNCTIONS ==============

def trendvic_logic(features: Dict[str, float], all_features: Dict[str, float]) -> Tuple[float, float, Dict]:
    """TrendVIC: Trend-following logic."""
    ma_delta = features.get("ma_delta", 0.0) or all_features.get("ma_delta", 0.0)
    ma_delta_pct = features.get("ma_delta_pct", 0.0) or all_features.get("ma_delta_pct", 0.0)
    volatility = features.get("volatility", 0.001) or all_features.get("volatility", 0.001)
    close = features.get("close", 1.0) or all_features.get("close", 1.0)
    body_ratio = features.get("body_ratio", 0.5)
    candle_direction = features.get("candle_direction", 0.0)
    price_momentum_3 = features.get("price_momentum_3", 0.0)
    upper_wick_ratio = features.get("upper_wick_ratio", 0.0)
    ema_10_spread = features.get("ema_10_spread_pct", 0.0)
    ema_20_spread = features.get("ema_20_spread_pct", 0.0)
    return_5 = features.get("return_5", 0.0)

    # Primary signal from ma_delta or ma_delta_pct
    if "ma_delta_pct" in features and ma_delta_pct != 0:
        signal = ma_delta_pct * 100  # Scale up
    elif ma_delta != 0:
        scale = max(volatility * close, 1e-12)
        signal = ma_delta / scale
    else:
        signal = 0.0

    # Base strength
    base_strength = min(abs(signal), 1.0) * 0.3

    # Confirmation factors
    confirmation = 1.0

    if "body_ratio" in features and body_ratio > 0.7:
        if (signal > 0 and candle_direction > 0) or (signal < 0 and candle_direction < 0):
            confirmation *= 1.2
        elif (signal > 0 and candle_direction < 0) or (signal < 0 and candle_direction > 0):
            confirmation *= 0.8

    if "price_momentum_3" in features:
        if (signal > 0 and price_momentum_3 > 0.001) or (signal < 0 and price_momentum_3 < -0.001):
            confirmation *= 1.1

    if "upper_wick_ratio" in features and upper_wick_ratio > 0.4:
        if signal > 0:
            confirmation *= 0.85  # Upper wick contradicts UP

    if "ema_10_spread_pct" in features:
        if (signal > 0 and ema_10_spread > 0) or (signal < 0 and ema_10_spread < 0):
            confirmation *= 1.05

    if "return_5" in features:
        if (signal > 0 and return_5 > 0) or (signal < 0 and return_5 < 0):
            confirmation *= 1.05

    strength = max(min(base_strength * confirmation, 0.5), 0.05)

    if signal >= 0:
        prob_up = 0.5 + strength
        prob_down = 0.5 - strength
    else:
        prob_up = 0.5 - strength
        prob_down = 0.5 + strength

    return prob_up, prob_down, {"signal": signal, "strength": strength}


def oscillator_logic(features: Dict[str, float], all_features: Dict[str, float]) -> Tuple[float, float, Dict]:
    """Oscillator: Mean reversion logic."""
    rsi = features.get("rsi", 50.0)
    rsi_momentum = features.get("rsi_momentum", 0.0)
    volatility_z = features.get("volatility_z", 0.0)
    volume_z = features.get("volume_z", 0.0)
    rsi_oversold = features.get("rsi_oversold", 0.0)
    rsi_overbought = features.get("rsi_overbought", 0.0)
    return_1 = features.get("return_1", 0.0)

    # Distance from equilibrium
    distance_from_50 = abs(rsi - 50.0)

    # Base strength
    if distance_from_50 < 10:
        base_strength = distance_from_50 / 100.0
    elif distance_from_50 < 20:
        base_strength = 0.10 + (distance_from_50 - 10) / 50.0
    else:
        base_strength = 0.30 + (distance_from_50 - 20) / 100.0

    # RSI momentum confirmation
    momentum_factor = 1.0
    if "rsi_momentum" in features:
        if rsi < 50:
            if rsi_momentum > 0.5:
                momentum_factor = 1.3
            elif rsi_momentum < -0.5:
                momentum_factor = 0.6
        else:
            if rsi_momentum < -0.5:
                momentum_factor = 1.3
            elif rsi_momentum > 0.5:
                momentum_factor = 0.6

    strength = base_strength * momentum_factor

    # Volume confirmation (NEW)
    if "volume_z" in features and volume_z > 1.0:
        strength *= 1.15  # High volume confirms reversal

    # Volatility penalty
    if "volatility_z" in features:
        if volatility_z > 1.5:
            strength *= 0.6
        elif volatility_z > 1.0:
            strength *= 0.8

    # RSI zone bonus
    if "rsi_oversold" in features and rsi_oversold > 0:
        strength *= 1.1
    if "rsi_overbought" in features and rsi_overbought > 0:
        strength *= 1.1

    strength = max(min(strength, 0.5), 0.05)

    # Mean reversion direction
    if rsi <= 50:
        prob_up = 0.5 + strength
        prob_down = 0.5 - strength
    else:
        prob_up = 0.5 - strength
        prob_down = 0.5 + strength

    return prob_up, prob_down, {"rsi": rsi, "strength": strength}


def volumemetrix_logic(features: Dict[str, float], all_features: Dict[str, float]) -> Tuple[float, float, Dict]:
    """VolumeMetrix: Volume-price relationship logic."""
    volume_z = features.get("volume_z", 0.0)
    return_1 = features.get("return_1", 0.0)
    volatility = features.get("volatility", 0.001) or all_features.get("volatility", 0.001)
    ma_delta = features.get("ma_delta", 0.0) or all_features.get("ma_delta", 0.0)
    volume_trend = features.get("volume_trend", 0.0)
    upper_wick = features.get("upper_wick_ratio", 0.0)
    lower_wick = features.get("lower_wick_ratio", 0.0)
    volume_change_pct = features.get("volume_change_pct", 0.0)
    body_ratio = features.get("body_ratio", 0.5)
    atr_pct = features.get("atr_pct", 0.0)

    # Relative move size
    ret_z = abs(return_1) / (volatility + 1e-10)

    # Determine pattern
    direction = "UP" if return_1 >= 0 else "DOWN"
    strength = 0.10

    if volume_z > 1.5:
        if ret_z > 1.0:
            # High volume + big move = continuation
            strength = min((volume_z + ret_z) / 8.0, 0.40)
        else:
            # High volume + small move = absorption (reversal)
            strength = min(volume_z / 6.0, 0.30)
            direction = "DOWN" if return_1 >= 0 else "UP"
    elif volume_z < -0.5:
        # Low volume - follow trend weakly
        strength = 0.05
        if ma_delta != 0:
            direction = "UP" if ma_delta > 0 else "DOWN"
    else:
        strength = min(ret_z / 6.0, 0.20)

    # Volume trend
    if "volume_trend" in features:
        if volume_trend > 0.2 and volume_z > 1.0:
            strength = min(strength * 1.15, 0.45)
        elif volume_trend < -0.2:
            strength *= 0.85

    # Volume change
    if "volume_change_pct" in features and abs(volume_change_pct) > 0.5:
        strength = min(strength * 1.1, 0.45)

    # Wick analysis
    if "upper_wick_ratio" in features and upper_wick > 0.4:
        if direction == "UP":
            strength *= 0.8
        else:
            strength = min(strength * 1.1, 0.45)

    if "lower_wick_ratio" in features and lower_wick > 0.4:
        if direction == "DOWN":
            strength *= 0.8
        else:
            strength = min(strength * 1.1, 0.45)

    # Body ratio
    if "body_ratio" in features and body_ratio > 0.7:
        strength = min(strength * 1.1, 0.45)

    strength = max(strength, 0.05)

    if direction == "UP":
        prob_up = 0.5 + strength
        prob_down = 0.5 - strength
    else:
        prob_up = 0.5 - strength
        prob_down = 0.5 + strength

    return prob_up, prob_down, {"volume_z": volume_z, "strength": strength}


# ============== FEATURE CONFIGURATIONS ==============

# Baseline configurations (current)
BASELINE_CONFIGS = {
    "TRENDVIC": ["ma_delta", "volatility", "close", "body_ratio", "candle_direction", "price_momentum_3"],
    "OSCILLATOR": ["rsi", "rsi_momentum", "volatility_z"],
    "VOLUMEMETRIX": ["volume_z", "return_1", "volatility", "ma_delta", "volume_trend", "upper_wick_ratio", "lower_wick_ratio"],
}

# Test configurations
TEST_CONFIGS = {
    # Oscillator tests
    "OSC_add_volume_z": {
        "OSCILLATOR": ["rsi", "rsi_momentum", "volatility_z", "volume_z"],
    },
    "OSC_add_volume_z_return1": {
        "OSCILLATOR": ["rsi", "rsi_momentum", "volatility_z", "volume_z", "return_1"],
    },
    "OSC_add_rsi_zones": {
        "OSCILLATOR": ["rsi", "rsi_momentum", "volatility_z", "rsi_oversold", "rsi_overbought"],
    },
    "OSC_full": {
        "OSCILLATOR": ["rsi", "rsi_momentum", "volatility_z", "volume_z", "rsi_oversold", "rsi_overbought", "return_1"],
    },

    # TrendVIC tests
    "TREND_remove_dead": {
        "TRENDVIC": ["ma_delta", "body_ratio", "candle_direction", "price_momentum_3"],  # removed close, volatility
    },
    "TREND_add_wick": {
        "TRENDVIC": ["ma_delta", "body_ratio", "candle_direction", "price_momentum_3", "upper_wick_ratio"],
    },
    "TREND_use_pct": {
        "TRENDVIC": ["ma_delta_pct", "body_ratio", "candle_direction", "price_momentum_3"],
    },
    "TREND_add_ema": {
        "TRENDVIC": ["ma_delta_pct", "body_ratio", "price_momentum_3", "ema_10_spread_pct", "ema_20_spread_pct"],
    },
    "TREND_add_return5": {
        "TRENDVIC": ["ma_delta_pct", "body_ratio", "price_momentum_3", "ema_10_spread_pct", "return_5"],
    },
    "TREND_full": {
        "TRENDVIC": ["ma_delta_pct", "ema_10_spread_pct", "ema_20_spread_pct", "price_momentum_3", "body_ratio", "upper_wick_ratio", "return_5"],
    },

    # VolumeMetrix tests
    "VOL_remove_dead": {
        "VOLUMEMETRIX": ["volume_z", "return_1", "lower_wick_ratio"],  # removed volatility, ma_delta, volume_trend, upper_wick
    },
    "VOL_add_change": {
        "VOLUMEMETRIX": ["volume_z", "return_1", "volume_change_pct", "lower_wick_ratio"],
    },
    "VOL_add_body": {
        "VOLUMEMETRIX": ["volume_z", "return_1", "volume_change_pct", "body_ratio", "lower_wick_ratio"],
    },
    "VOL_add_atr": {
        "VOLUMEMETRIX": ["volume_z", "return_1", "volume_change_pct", "body_ratio", "atr_pct"],
    },
    "VOL_full": {
        "VOLUMEMETRIX": ["volume_z", "volume_change_pct", "return_1", "body_ratio", "atr_pct", "lower_wick_ratio"],
    },

    # COMBINED TESTS - Best of each model
    "COMBO_trend_ema": {
        "TRENDVIC": ["ma_delta_pct", "body_ratio", "price_momentum_3", "ema_10_spread_pct", "ema_20_spread_pct"],
    },
    "COMBO_trend_vol": {
        "TRENDVIC": ["ma_delta_pct", "body_ratio", "price_momentum_3", "ema_10_spread_pct", "ema_20_spread_pct"],
        "VOLUMEMETRIX": ["volume_z", "return_1", "lower_wick_ratio"],
    },
    "COMBO_trend_osc": {
        "TRENDVIC": ["ma_delta_pct", "body_ratio", "price_momentum_3", "ema_10_spread_pct", "ema_20_spread_pct"],
        "OSCILLATOR": ["rsi", "rsi_momentum", "volatility_z", "rsi_oversold", "rsi_overbought"],
    },
    "COMBO_all_best": {
        "TRENDVIC": ["ma_delta_pct", "body_ratio", "price_momentum_3", "ema_10_spread_pct", "ema_20_spread_pct"],
        "OSCILLATOR": ["rsi", "rsi_momentum", "volatility_z", "rsi_oversold", "rsi_overbought"],
        "VOLUMEMETRIX": ["volume_z", "return_1", "lower_wick_ratio"],
    },
    "COMBO_all_full": {
        "TRENDVIC": ["ma_delta_pct", "ema_10_spread_pct", "ema_20_spread_pct", "price_momentum_3", "body_ratio", "upper_wick_ratio", "return_5"],
        "OSCILLATOR": ["rsi", "rsi_momentum", "volatility_z", "rsi_oversold", "rsi_overbought"],
        "VOLUMEMETRIX": ["volume_z", "volume_change_pct", "return_1", "body_ratio", "atr_pct", "lower_wick_ratio"],
    },
}


class FeatureABTester:
    """A/B testing framework for feature combinations."""

    def __init__(self, hours: int = 72, symbol: str = "BTCUSDT", interval: int = 1):
        self.hours = hours
        self.symbol = symbol
        self.interval = interval
        self.candles = None
        self.features_list = None
        self.actuals = None
        self.results: List[TestResult] = []

    def load_data(self):
        """Load and prepare data."""
        print(f"\nLoading {self.hours}h of {self.symbol} {self.interval}m data...")

        state_store = StateStore(":memory:")
        config = ConfigStore(state_store)
        config.ensure_defaults()

        stream = FeatureStream(config)

        end_ts = int(time_module.time())
        start_ts = end_ts - (self.hours * 3600) - (100 * self.interval * 60)
        self.candles = fetch_klines(self.symbol, str(self.interval), start_ts, end_ts)

        if not self.candles:
            raise ValueError("No candles fetched")

        print(f"Fetched {len(self.candles)} candles")

        # Prepare features and actuals
        self.features_list = []
        self.actuals = []

        warmup = 40
        for i, candle in enumerate(self.candles[:-1]):
            features = stream.update(candle)
            if features is None or i < warmup:
                continue

            next_candle = self.candles[i + 1]
            actual = "UP" if next_candle.close > candle.close else "DOWN"

            self.features_list.append(features)
            self.actuals.append(actual)

        print(f"Prepared {len(self.features_list)} test samples")

    def run_test(self, name: str, configs: Dict[str, List[str]]) -> TestResult:
        """Run a single test with given feature configurations."""
        # Merge with baseline for models not specified
        full_configs = copy.deepcopy(BASELINE_CONFIGS)
        full_configs.update(configs)

        # Create models
        models = {
            "TRENDVIC": ConfigurableModel("TRENDVIC", full_configs["TRENDVIC"], trendvic_logic),
            "OSCILLATOR": ConfigurableModel("OSCILLATOR", full_configs["OSCILLATOR"], oscillator_logic),
            "VOLUMEMETRIX": ConfigurableModel("VOLUMEMETRIX", full_configs["VOLUMEMETRIX"], volumemetrix_logic),
        }

        # Run predictions
        model_correct = {m: 0 for m in models}
        model_total = {m: 0 for m in models}
        ensemble_correct = 0
        high_conf_correct = 0
        high_conf_total = 0

        for features, actual in zip(self.features_list, self.actuals):
            # Get predictions
            predictions = {}
            for model_name, model in models.items():
                prob_up, prob_down, _ = model.predict(features)
                pred = "UP" if prob_up > prob_down else "DOWN"
                conf = max(prob_up, prob_down)
                predictions[model_name] = (pred, conf, prob_up)

                # Model accuracy
                model_total[model_name] += 1
                if pred == actual:
                    model_correct[model_name] += 1

            # Ensemble (weighted average)
            weights = {"TRENDVIC": 0.33, "OSCILLATOR": 0.34, "VOLUMEMETRIX": 0.33}
            ensemble_prob_up = sum(predictions[m][2] * weights[m] for m in models)
            ensemble_pred = "UP" if ensemble_prob_up > 0.5 else "DOWN"
            ensemble_conf = abs(ensemble_prob_up - 0.5) * 2 + 0.5

            if ensemble_pred == actual:
                ensemble_correct += 1

            # High confidence tracking
            if ensemble_conf >= 0.55:
                high_conf_total += 1
                if ensemble_pred == actual:
                    high_conf_correct += 1

        total = len(self.features_list)

        result = TestResult(
            name=name,
            model_accuracies={m: model_correct[m] / model_total[m] for m in models},
            ensemble_accuracy=ensemble_correct / total * 100,
            high_conf_accuracy=high_conf_correct / high_conf_total * 100 if high_conf_total > 0 else 0,
            high_conf_count=high_conf_total,
            total_count=total,
        )

        self.results.append(result)
        return result

    def run_baseline(self) -> TestResult:
        """Run baseline test."""
        print("\n" + "=" * 70)
        print("  BASELINE TEST")
        print("=" * 70)
        result = self.run_test("BASELINE", {})
        print(f"  {result}")
        return result

    def run_single_test(self, test_name: str) -> Tuple[TestResult, bool]:
        """Run a single test and compare to baseline."""
        if test_name not in TEST_CONFIGS:
            raise ValueError(f"Unknown test: {test_name}")

        baseline = next((r for r in self.results if r.name == "BASELINE"), None)
        if not baseline:
            baseline = self.run_baseline()

        print(f"\n  Testing: {test_name}")
        print(f"  Config: {TEST_CONFIGS[test_name]}")

        result = self.run_test(test_name, TEST_CONFIGS[test_name])

        # Compare to baseline
        improvement = result.ensemble_accuracy - baseline.ensemble_accuracy
        is_better = improvement > 0

        status = "BETTER" if is_better else "WORSE"
        print(f"  Result: {result}")
        print(f"  vs Baseline: {improvement:+.2f}% ({status})")

        return result, is_better

    def run_all_tests(self) -> Dict[str, TestResult]:
        """Run all tests and return results."""
        print("\n" + "=" * 70)
        print("  RUNNING ALL A/B TESTS")
        print("=" * 70)

        baseline = self.run_baseline()

        improvements = []

        for test_name in TEST_CONFIGS:
            result, is_better = self.run_single_test(test_name)
            improvement = result.ensemble_accuracy - baseline.ensemble_accuracy
            improvements.append((test_name, improvement, result))

        # Sort by improvement
        improvements.sort(key=lambda x: x[1], reverse=True)

        print("\n" + "=" * 70)
        print("  RESULTS SUMMARY (sorted by improvement)")
        print("=" * 70)
        print(f"\n  {'Test':<30} {'Ensemble':>10} {'Change':>10} {'HighConf':>12}")
        print("  " + "-" * 65)
        print(f"  {'BASELINE':<30} {baseline.ensemble_accuracy:>9.2f}% {'+0.00%':>10} {baseline.high_conf_accuracy:>6.1f}% ({baseline.high_conf_count})")

        for test_name, improvement, result in improvements:
            status = "+" if improvement > 0 else ""
            print(f"  {test_name:<30} {result.ensemble_accuracy:>9.2f}% {status}{improvement:>+.2f}% {result.high_conf_accuracy:>6.1f}% ({result.high_conf_count})")

        # Best per model
        print("\n  BEST CONFIGS PER MODEL:")
        for model in ["TRENDVIC", "OSCILLATOR", "VOLUMEMETRIX"]:
            model_prefix = model[:3].upper() if model != "VOLUMEMETRIX" else "VOL"
            model_tests = [(n, i, r) for n, i, r in improvements if n.startswith(model_prefix)]
            if model_tests:
                best = max(model_tests, key=lambda x: x[2].model_accuracies[model])
                print(f"    {model}: {best[0]} ({best[2].model_accuracies[model]*100:.1f}%)")

        return {r.name: r for r in self.results}

    def find_best_combination(self) -> Dict[str, List[str]]:
        """Find the best feature combination for each model."""
        if not self.results:
            self.run_all_tests()

        baseline = next(r for r in self.results if r.name == "BASELINE")

        best_configs = copy.deepcopy(BASELINE_CONFIGS)

        # Find best for each model
        for model in ["TRENDVIC", "OSCILLATOR", "VOLUMEMETRIX"]:
            model_prefix = model[:3].upper() if model != "VOLUMEMETRIX" else "VOL"
            model_results = [(r.name, r.model_accuracies[model])
                           for r in self.results
                           if r.name.startswith(model_prefix) or r.name == "BASELINE"]

            if model_results:
                best_name, best_acc = max(model_results, key=lambda x: x[1])
                if best_name != "BASELINE" and best_name in TEST_CONFIGS:
                    best_configs[model] = TEST_CONFIGS[best_name].get(model, best_configs[model])
                    print(f"  {model}: Using {best_name} config ({best_acc*100:.1f}%)")

        return best_configs

    def save_results(self, path: str = "ab_test_results.json"):
        """Save results to JSON."""
        data = {
            "hours": self.hours,
            "symbol": self.symbol,
            "total_samples": len(self.features_list) if self.features_list else 0,
            "results": [
                {
                    "name": r.name,
                    "ensemble_accuracy": r.ensemble_accuracy,
                    "model_accuracies": r.model_accuracies,
                    "high_conf_accuracy": r.high_conf_accuracy,
                    "high_conf_count": r.high_conf_count,
                }
                for r in self.results
            ]
        }

        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"\nResults saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Feature A/B Testing")
    parser.add_argument("--hours", type=int, default=72, help="Hours of data")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol")
    parser.add_argument("--interval", type=int, default=1, help="Interval")
    parser.add_argument("--test", type=str, help="Run specific test")
    parser.add_argument("--test-all", action="store_true", help="Run all tests")

    args = parser.parse_args()

    tester = FeatureABTester(hours=args.hours, symbol=args.symbol, interval=args.interval)
    tester.load_data()

    if args.test:
        tester.run_baseline()
        tester.run_single_test(args.test)
    elif args.test_all:
        tester.run_all_tests()
        tester.find_best_combination()
    else:
        tester.run_baseline()

    tester.save_results()


if __name__ == "__main__":
    main()
