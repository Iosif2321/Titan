#!/usr/bin/env python3
"""
Feature Analysis Script v2 for Titan Models

More accurate feature importance using:
- Point-biserial correlation (continuous feature vs binary outcome)
- Information gain (entropy reduction)
- Feature effectiveness per model

Usage:
    python scripts/feature_analysis_v2.py --hours 168
"""

import argparse
import json
import math
import sys
import time as time_module
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from titan.core.config import ConfigStore
from titan.core.data.bybit_rest import fetch_klines
from titan.core.features.stream import FeatureStream
from titan.core.models.heuristic import TrendVIC, Oscillator, VolumeMetrix
from titan.core.state_store import StateStore
from titan.core.types import ModelOutput


@dataclass
class FeatureRecord:
    """Single record of features + outcome."""
    features: Dict[str, float]
    actual_direction: str  # UP or DOWN
    actual_return: float   # actual price change %
    model_predictions: Dict[str, str]
    model_confidences: Dict[str, float]
    model_correct: Dict[str, bool]


def pearson_correlation(x: List[float], y: List[float]) -> float:
    """Calculate Pearson correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

    if denom_x == 0 or denom_y == 0:
        return 0.0

    return numerator / (denom_x * denom_y)


def point_biserial_correlation(continuous: List[float], binary: List[bool]) -> float:
    """Calculate point-biserial correlation (continuous vs binary variable)."""
    if len(continuous) != len(binary) or len(continuous) < 10:
        return 0.0

    group_1 = [c for c, b in zip(continuous, binary) if b]
    group_0 = [c for c, b in zip(continuous, binary) if not b]

    if not group_1 or not group_0:
        return 0.0

    n1 = len(group_1)
    n0 = len(group_0)
    n = n1 + n0

    mean_1 = sum(group_1) / n1
    mean_0 = sum(group_0) / n0

    # Overall std
    overall_mean = sum(continuous) / n
    overall_var = sum((x - overall_mean) ** 2 for x in continuous) / n
    overall_std = math.sqrt(overall_var) if overall_var > 0 else 1e-10

    # Point-biserial correlation
    rpb = (mean_1 - mean_0) / overall_std * math.sqrt(n1 * n0 / (n * n))

    return rpb


def calculate_entropy(values: List[bool]) -> float:
    """Calculate entropy of binary list."""
    if not values:
        return 0.0

    p = sum(values) / len(values)
    if p == 0 or p == 1:
        return 0.0

    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def information_gain(feature_values: List[float], outcomes: List[bool], n_bins: int = 5) -> float:
    """Calculate information gain when splitting by feature."""
    if len(feature_values) != len(outcomes) or len(feature_values) < 20:
        return 0.0

    # Calculate base entropy
    base_entropy = calculate_entropy(outcomes)

    # Sort and split into bins
    sorted_data = sorted(zip(feature_values, outcomes))
    bin_size = len(sorted_data) // n_bins

    weighted_entropy = 0.0
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_data)
        bin_outcomes = [o for _, o in sorted_data[start:end]]

        if bin_outcomes:
            bin_entropy = calculate_entropy(bin_outcomes)
            weight = len(bin_outcomes) / len(outcomes)
            weighted_entropy += weight * bin_entropy

    return base_entropy - weighted_entropy


class FeatureAnalyzerV2:
    """Enhanced feature analyzer with better metrics."""

    # Features used by each model
    MODEL_FEATURES = {
        "TRENDVIC": ["ma_delta", "volatility", "close", "body_ratio",
                     "candle_direction", "price_momentum_3"],
        "OSCILLATOR": ["rsi", "rsi_momentum", "volatility_z"],
        "VOLUMEMETRIX": ["volume_z", "return_1", "volatility", "ma_delta",
                         "volume_trend", "upper_wick_ratio", "lower_wick_ratio"],
    }

    # Scale-invariant features (good for ML)
    SCALE_INVARIANT = [
        "return_1", "log_return_1", "ma_delta_pct", "volatility_z", "vol_ratio",
        "volume_z", "volume_trend", "volume_change_pct",
        "rsi", "rsi_momentum", "rsi_oversold", "rsi_overbought", "rsi_neutral",
        "body_ratio", "body_pct", "upper_wick_ratio", "lower_wick_ratio",
        "candle_direction", "price_momentum_3",
        "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_4", "return_lag_5",
        "atr_pct", "high_low_range_pct",
        "ema_10_spread_pct", "ema_20_spread_pct",
        "return_5", "return_10",
    ]

    def __init__(self):
        self.records: List[FeatureRecord] = []

    def add_record(self, record: FeatureRecord):
        """Add a single record."""
        self.records.append(record)

    def get_direction_correlation(self) -> Dict[str, float]:
        """Get correlation between feature and actual direction (UP=1, DOWN=0)."""
        correlations = {}

        # Convert direction to binary
        directions = [1.0 if r.actual_direction == "UP" else 0.0 for r in self.records]

        for feat_name in self.SCALE_INVARIANT:
            values = [r.features.get(feat_name, 0.0) for r in self.records]
            corr = pearson_correlation(values, directions)
            correlations[feat_name] = corr

        return dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))

    def get_model_effectiveness(self, model: str) -> Dict[str, float]:
        """Get feature effectiveness when model is correct vs wrong.

        Uses point-biserial correlation: how different is feature value
        when model predicts correctly vs incorrectly.
        """
        effectiveness = {}

        correct = [r.model_correct.get(model, False) for r in self.records]

        for feat_name in self.SCALE_INVARIANT:
            values = [r.features.get(feat_name, 0.0) for r in self.records]
            rpb = point_biserial_correlation(values, correct)
            effectiveness[feat_name] = rpb

        return dict(sorted(effectiveness.items(), key=lambda x: abs(x[1]), reverse=True))

    def get_information_gain(self, model: str) -> Dict[str, float]:
        """Get information gain for predicting model correctness."""
        gains = {}

        correct = [r.model_correct.get(model, False) for r in self.records]

        for feat_name in self.SCALE_INVARIANT:
            values = [r.features.get(feat_name, 0.0) for r in self.records]
            ig = information_gain(values, correct)
            gains[feat_name] = ig

        return dict(sorted(gains.items(), key=lambda x: x[1], reverse=True))

    def get_quintile_accuracy(self, model: str, feature: str) -> Dict[str, Tuple[float, int]]:
        """Get model accuracy for each quintile of a feature."""
        data = [(r.features.get(feature, 0.0), r.model_correct.get(model, False))
                for r in self.records]

        if len(data) < 25:
            return {}

        data.sort(key=lambda x: x[0])
        quintile_size = len(data) // 5

        results = {}
        for i in range(5):
            start = i * quintile_size
            end = (i + 1) * quintile_size if i < 4 else len(data)
            subset = data[start:end]

            correct = sum(1 for _, c in subset if c)
            total = len(subset)
            accuracy = correct / total if total > 0 else 0.0

            label = ["Q1 (lowest)", "Q2", "Q3 (middle)", "Q4", "Q5 (highest)"][i]
            results[label] = (accuracy, total)

        return results

    def get_feature_ranges_when_correct(self, model: str) -> Dict[str, Dict[str, float]]:
        """Get feature statistics when model is correct."""
        results = {}

        correct_records = [r for r in self.records if r.model_correct.get(model, False)]
        wrong_records = [r for r in self.records if not r.model_correct.get(model, False)]

        for feat_name in self.SCALE_INVARIANT:
            correct_vals = [r.features.get(feat_name, 0.0) for r in correct_records]
            wrong_vals = [r.features.get(feat_name, 0.0) for r in wrong_records]

            if not correct_vals or not wrong_vals:
                continue

            results[feat_name] = {
                "correct_mean": sum(correct_vals) / len(correct_vals),
                "wrong_mean": sum(wrong_vals) / len(wrong_vals),
                "correct_std": math.sqrt(sum((x - sum(correct_vals)/len(correct_vals))**2
                                            for x in correct_vals) / len(correct_vals)),
                "diff": (sum(correct_vals)/len(correct_vals)) - (sum(wrong_vals)/len(wrong_vals)),
            }

        return results

    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        lines = []
        lines.append("=" * 80)
        lines.append("  TITAN FEATURE ANALYSIS REPORT v2")
        lines.append("=" * 80)
        lines.append(f"\nTotal records: {len(self.records)}")

        # Model accuracies
        lines.append("\n" + "-" * 80)
        lines.append("  1. MODEL ACCURACIES")
        lines.append("-" * 80)

        for model in ["TRENDVIC", "OSCILLATOR", "VOLUMEMETRIX"]:
            correct = sum(1 for r in self.records if r.model_correct.get(model, False))
            total = len(self.records)
            accuracy = correct / total * 100 if total > 0 else 0.0
            features = self.MODEL_FEATURES[model]
            lines.append(f"\n  {model}:")
            lines.append(f"    Accuracy: {accuracy:.1f}% ({correct}/{total})")
            lines.append(f"    Current features: {features}")

        # Direction correlation
        lines.append("\n" + "-" * 80)
        lines.append("  2. FEATURE-DIRECTION CORRELATION")
        lines.append("  (Positive = feature higher when price goes UP)")
        lines.append("-" * 80)

        dir_corr = self.get_direction_correlation()
        lines.append(f"\n  {'Feature':<25} {'Correlation':>12} {'Strength':<15}")
        lines.append("  " + "-" * 55)

        for feat, corr in list(dir_corr.items())[:20]:
            strength = "STRONG" if abs(corr) > 0.15 else "Moderate" if abs(corr) > 0.05 else "Weak"
            direction = "UP" if corr > 0 else "DOWN"
            bar = "|" + "=" * int(abs(corr) * 40)
            lines.append(f"  {feat:<25} {corr:>+.4f} {direction:>6} {strength:<8}")

        # Model-specific analysis
        for model in ["TRENDVIC", "OSCILLATOR", "VOLUMEMETRIX"]:
            lines.append("\n" + "-" * 80)
            lines.append(f"  3. {model} - FEATURE EFFECTIVENESS")
            lines.append(f"  (Point-biserial correlation with model correctness)")
            lines.append("-" * 80)

            effectiveness = self.get_model_effectiveness(model)
            info_gain = self.get_information_gain(model)

            lines.append(f"\n  {'Feature':<25} {'Effectiveness':>12} {'Info Gain':>10} {'Used':>6}")
            lines.append("  " + "-" * 60)

            current_features = set(self.MODEL_FEATURES[model])
            for i, (feat, eff) in enumerate(list(effectiveness.items())[:15]):
                ig = info_gain.get(feat, 0.0)
                used = "YES" if feat in current_features else ""
                lines.append(f"  {feat:<25} {eff:>+.4f} {ig:>10.4f} {used:>6}")

            # Best quintiles
            lines.append(f"\n  {model} - Best Feature Zones:")

            for feat in list(effectiveness.keys())[:5]:
                quintiles = self.get_quintile_accuracy(model, feat)
                if quintiles:
                    best_q = max(quintiles.items(), key=lambda x: x[1][0])
                    worst_q = min(quintiles.items(), key=lambda x: x[1][0])
                    lines.append(f"    {feat}:")
                    lines.append(f"      Best:  {best_q[0]} = {best_q[1][0]*100:.1f}%")
                    lines.append(f"      Worst: {worst_q[0]} = {worst_q[1][0]*100:.1f}%")

        # Recommendations
        lines.append("\n" + "-" * 80)
        lines.append("  4. FEATURE RECOMMENDATIONS BY MODEL")
        lines.append("-" * 80)

        for model in ["TRENDVIC", "OSCILLATOR", "VOLUMEMETRIX"]:
            lines.append(f"\n  {model}:")

            effectiveness = self.get_model_effectiveness(model)
            current = set(self.MODEL_FEATURES[model])

            # Top effective features not currently used
            to_add = [(f, e) for f, e in effectiveness.items()
                      if f not in current and abs(e) > 0.02][:5]

            # Currently used but ineffective
            to_remove = [f for f in current
                        if abs(effectiveness.get(f, 0)) < 0.01]

            lines.append(f"    Currently using: {list(current)}")

            if to_add:
                lines.append(f"    + CONSIDER ADDING:")
                for feat, eff in to_add:
                    lines.append(f"        {feat} (effectiveness: {eff:+.4f})")

            if to_remove:
                lines.append(f"    - CONSIDER REMOVING (low effectiveness):")
                for feat in to_remove:
                    eff = effectiveness.get(feat, 0)
                    lines.append(f"        {feat} (effectiveness: {eff:+.4f})")

        # Optimal feature sets
        lines.append("\n" + "-" * 80)
        lines.append("  5. OPTIMAL FEATURE SETS (by model philosophy)")
        lines.append("-" * 80)

        # TrendVIC - trend following
        lines.append("\n  TRENDVIC (Trend Following):")
        lines.append("    Philosophy: Follow the trend, use momentum confirmation")
        trend_features = ["ma_delta", "ma_delta_pct", "ema_10_spread_pct", "ema_20_spread_pct",
                        "price_momentum_3", "return_5", "return_10", "candle_direction"]
        lines.append(f"    Recommended: {trend_features}")

        # Oscillator - mean reversion
        lines.append("\n  OSCILLATOR (Mean Reversion):")
        lines.append("    Philosophy: Extreme RSI = reversal, confirm with momentum")
        osc_features = ["rsi", "rsi_momentum", "rsi_oversold", "rsi_overbought",
                       "volatility_z", "return_1"]
        lines.append(f"    Recommended: {osc_features}")

        # VolumeMetrix - volume-price
        lines.append("\n  VOLUMEMETRIX (Volume-Price Relationship):")
        lines.append("    Philosophy: Volume confirms moves, detect absorption/continuation")
        vol_features = ["volume_z", "volume_trend", "volume_change_pct", "return_1",
                       "body_ratio", "atr_pct"]
        lines.append(f"    Recommended: {vol_features}")

        # Missing features from Chronos
        lines.append("\n" + "-" * 80)
        lines.append("  6. MISSING FEATURES (from Chronos analysis)")
        lines.append("-" * 80)

        missing = {
            "vol_imbalance": "Volume imbalance UP vs DOWN - CRITICAL for VolumeMetrix",
            "realized_vol": "Log-return based volatility - better than simple std",
            "bb_position": "Position within Bollinger Bands - good for mean reversion",
            "bb_width_pct": "Bollinger Band width - volatility squeeze detection",
            "macd_pct": "MACD as % of price - trend strength",
            "parkinson_vol": "High-Low volatility estimator - more robust",
        }

        for feat, desc in missing.items():
            lines.append(f"\n    {feat}:")
            lines.append(f"      {desc}")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)


def run_analysis(symbol: str, interval: int, hours: int) -> FeatureAnalyzerV2:
    """Run feature analysis on historical data."""
    print(f"\nRunning feature analysis v2 for {symbol} {interval}m ({hours}h)...")

    # Initialize config
    state_store = StateStore(":memory:")
    config = ConfigStore(state_store)
    config.ensure_defaults()

    stream = FeatureStream(config)

    # Initialize models
    models = {
        "TRENDVIC": TrendVIC(config),
        "OSCILLATOR": Oscillator(config),
        "VOLUMEMETRIX": VolumeMetrix(config),
    }

    analyzer = FeatureAnalyzerV2()

    # Fetch candles
    end_ts = int(time_module.time())
    start_ts = end_ts - (hours * 3600) - (100 * interval * 60)
    candles = fetch_klines(symbol, str(interval), start_ts, end_ts)

    if not candles:
        print("ERROR: No candles fetched")
        return analyzer

    print(f"Fetched {len(candles)} candles")

    # Process candles
    warmup = 40
    processed = 0

    for i, candle in enumerate(candles[:-1]):
        features = stream.update(candle)

        if features is None:
            continue

        if i < warmup:
            continue

        # Get next candle for actual direction
        next_candle = candles[i + 1]
        actual_direction = "UP" if next_candle.close > candle.close else "DOWN"
        actual_return = (next_candle.close - candle.close) / candle.close

        # Get model predictions
        model_predictions = {}
        model_confidences = {}
        model_correct = {}

        for model_name, model in models.items():
            output = model.predict(features)
            pred_dir = "UP" if output.prob_up > output.prob_down else "DOWN"
            confidence = max(output.prob_up, output.prob_down)

            model_predictions[model_name] = pred_dir
            model_confidences[model_name] = confidence
            model_correct[model_name] = (pred_dir == actual_direction)

        # Create record
        record = FeatureRecord(
            features=features.copy(),
            actual_direction=actual_direction,
            actual_return=actual_return,
            model_predictions=model_predictions,
            model_confidences=model_confidences,
            model_correct=model_correct,
        )

        analyzer.add_record(record)
        processed += 1

        if processed % 1000 == 0:
            print(f"  Processed {processed} records...")

    print(f"Analysis complete: {processed} records")
    return analyzer


def main():
    parser = argparse.ArgumentParser(description="Feature analysis v2 for Titan models")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--interval", type=int, default=1, help="Interval in minutes")
    parser.add_argument("--hours", type=int, default=168, help="Hours of history")
    parser.add_argument("--output", default=None, help="Output file")

    args = parser.parse_args()

    analyzer = run_analysis(args.symbol, args.interval, args.hours)
    report = analyzer.generate_report()

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"\nReport saved to {args.output}")
    else:
        # Write to file anyway to avoid encoding issues
        Path("feature_analysis_v2_report.txt").write_text(report, encoding="utf-8")
        print(f"\nReport saved to feature_analysis_v2_report.txt")

    # Save detailed JSON
    results = {
        "records_count": len(analyzer.records),
        "direction_correlation": analyzer.get_direction_correlation(),
        "model_effectiveness": {
            model: analyzer.get_model_effectiveness(model)
            for model in ["TRENDVIC", "OSCILLATOR", "VOLUMEMETRIX"]
        },
        "information_gain": {
            model: analyzer.get_information_gain(model)
            for model in ["TRENDVIC", "OSCILLATOR", "VOLUMEMETRIX"]
        },
    }

    Path("feature_analysis_v2_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print("JSON results saved to feature_analysis_v2_results.json")


if __name__ == "__main__":
    main()
