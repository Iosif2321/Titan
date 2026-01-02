#!/usr/bin/env python3
"""
Feature Analysis Script for Titan Models

Analyzes feature importance and correlation for each model:
- TrendVIC: Trend-following model (MA crossover + candle confirmation)
- Oscillator: Mean reversion model (RSI-based)
- VolumeMetrix: Volume-Price relationship model

Output:
- Feature correlation with actual direction
- Feature importance when model is correct vs wrong
- Optimal feature set recommendations

Usage:
    python scripts/feature_analysis.py --hours 168
    python scripts/feature_analysis.py --hours 24 --symbol BTCUSDT
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
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
    model_predictions: Dict[str, str]  # model_name -> predicted direction
    model_confidences: Dict[str, float]  # model_name -> confidence
    model_correct: Dict[str, bool]  # model_name -> was correct


@dataclass
class FeatureStats:
    """Statistics for a single feature."""
    name: str
    values_when_up: List[float] = field(default_factory=list)
    values_when_down: List[float] = field(default_factory=list)
    values_when_correct: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    values_when_wrong: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    def mean_up(self) -> float:
        return sum(self.values_when_up) / len(self.values_when_up) if self.values_when_up else 0.0

    def mean_down(self) -> float:
        return sum(self.values_when_down) / len(self.values_when_down) if self.values_when_down else 0.0

    def direction_correlation(self) -> float:
        """Returns correlation: positive = predicts UP, negative = predicts DOWN."""
        mean_up = self.mean_up()
        mean_down = self.mean_down()
        if mean_up == 0 and mean_down == 0:
            return 0.0
        # Normalized difference
        total_range = abs(mean_up) + abs(mean_down) + 1e-10
        return (mean_up - mean_down) / total_range

    def model_importance(self, model: str) -> float:
        """Returns importance: how different is feature when model is correct vs wrong."""
        correct = self.values_when_correct.get(model, [])
        wrong = self.values_when_wrong.get(model, [])
        if not correct or not wrong:
            return 0.0
        mean_correct = sum(correct) / len(correct)
        mean_wrong = sum(wrong) / len(wrong)
        total_range = abs(mean_correct) + abs(mean_wrong) + 1e-10
        return abs(mean_correct - mean_wrong) / total_range


class FeatureAnalyzer:
    """Analyzes feature importance for models."""

    # Features used by each model (from heuristic.py analysis)
    MODEL_FEATURES = {
        "TRENDVIC": [
            "ma_delta", "volatility", "close", "body_ratio",
            "candle_direction", "price_momentum_3"
        ],
        "OSCILLATOR": [
            "rsi", "rsi_momentum", "volatility_z"
        ],
        "VOLUMEMETRIX": [
            "volume_z", "return_1", "volatility", "ma_delta",
            "volume_trend", "upper_wick_ratio", "lower_wick_ratio"
        ],
    }

    # All available features
    ALL_FEATURES = [
        "return_1", "log_return_1", "ma_delta", "ma_delta_pct",
        "volatility", "volatility_z", "vol_ratio",
        "volume_z", "volume_trend", "volume_change_pct",
        "rsi", "rsi_momentum", "rsi_oversold", "rsi_overbought", "rsi_neutral",
        "body_ratio", "body_pct", "upper_wick_ratio", "lower_wick_ratio",
        "candle_direction", "price_momentum_3",
        "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_4", "return_lag_5",
        "atr_pct", "high_low_range_pct",
        "ema_10_spread_pct", "ema_20_spread_pct",
        "return_5", "return_10",
    ]

    # Potential new features (from Chronos analysis)
    POTENTIAL_FEATURES = [
        "vol_imbalance_20", "vol_imbalance_60",  # Volume imbalance
        "realized_vol_20", "realized_vol_60",    # Realized volatility
        "bb_width_pct", "bb_position",           # Bollinger Bands
        "bb_dist_upper", "bb_dist_lower",
        "macd_pct", "macd_signal_pct",           # MACD
        "sma_dist_20", "sma_dist_50",            # SMA distance
    ]

    def __init__(self):
        self.records: List[FeatureRecord] = []
        self.feature_stats: Dict[str, FeatureStats] = {}

        # Initialize stats for all features
        for feat in self.ALL_FEATURES:
            self.feature_stats[feat] = FeatureStats(name=feat)

    def add_record(self, record: FeatureRecord):
        """Add a single record."""
        self.records.append(record)

        # Update feature stats
        for feat_name, feat_value in record.features.items():
            if feat_name not in self.feature_stats:
                self.feature_stats[feat_name] = FeatureStats(name=feat_name)

            stats = self.feature_stats[feat_name]

            # Direction correlation
            if record.actual_direction == "UP":
                stats.values_when_up.append(feat_value)
            else:
                stats.values_when_down.append(feat_value)

            # Model importance
            for model_name, correct in record.model_correct.items():
                if correct:
                    stats.values_when_correct[model_name].append(feat_value)
                else:
                    stats.values_when_wrong[model_name].append(feat_value)

    def analyze_direction_correlation(self) -> Dict[str, float]:
        """Get direction correlation for all features."""
        correlations = {}
        for feat_name, stats in self.feature_stats.items():
            correlations[feat_name] = stats.direction_correlation()
        return dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))

    def analyze_model_importance(self, model: str) -> Dict[str, float]:
        """Get feature importance for a specific model."""
        importance = {}
        for feat_name, stats in self.feature_stats.items():
            importance[feat_name] = stats.model_importance(model)
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def get_model_accuracy_by_feature_quintile(
        self, model: str, feature: str
    ) -> Dict[str, Tuple[float, int]]:
        """Get model accuracy for each quintile of a feature."""
        # Collect feature values and outcomes
        data = []
        for record in self.records:
            if feature in record.features:
                data.append((
                    record.features[feature],
                    record.model_correct.get(model, False)
                ))

        if len(data) < 10:
            return {}

        # Sort by feature value
        data.sort(key=lambda x: x[0])

        # Split into quintiles
        quintile_size = len(data) // 5
        results = {}

        for i in range(5):
            start = i * quintile_size
            end = (i + 1) * quintile_size if i < 4 else len(data)
            quintile_data = data[start:end]

            correct = sum(1 for _, c in quintile_data if c)
            total = len(quintile_data)
            accuracy = correct / total if total > 0 else 0.0

            label = ["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"][i]
            results[label] = (accuracy, total)

        return results

    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        lines = []
        lines.append("=" * 70)
        lines.append("  TITAN FEATURE ANALYSIS REPORT")
        lines.append("=" * 70)
        lines.append(f"\nTotal records: {len(self.records)}")

        # Model accuracies
        lines.append("\n" + "-" * 70)
        lines.append("  MODEL ACCURACIES")
        lines.append("-" * 70)

        for model in ["TRENDVIC", "OSCILLATOR", "VOLUMEMETRIX"]:
            correct = sum(1 for r in self.records if r.model_correct.get(model, False))
            total = len(self.records)
            accuracy = correct / total * 100 if total > 0 else 0.0
            lines.append(f"  {model:15} {accuracy:5.1f}% ({correct}/{total})")

        # Direction correlation
        lines.append("\n" + "-" * 70)
        lines.append("  FEATURE DIRECTION CORRELATION")
        lines.append("  (Positive = higher when UP, Negative = higher when DOWN)")
        lines.append("-" * 70)

        correlations = self.analyze_direction_correlation()
        lines.append(f"\n  {'Feature':<25} {'Correlation':>12} {'Interpretation':<25}")
        lines.append("  " + "-" * 62)

        for feat, corr in list(correlations.items())[:20]:
            if abs(corr) > 0.1:
                interp = "Strong UP predictor" if corr > 0.2 else \
                         "Weak UP predictor" if corr > 0 else \
                         "Strong DOWN predictor" if corr < -0.2 else \
                         "Weak DOWN predictor"
            else:
                interp = "Neutral"
            lines.append(f"  {feat:<25} {corr:>+12.3f} {interp:<25}")

        # Model-specific importance
        for model in ["TRENDVIC", "OSCILLATOR", "VOLUMEMETRIX"]:
            lines.append("\n" + "-" * 70)
            lines.append(f"  {model} FEATURE IMPORTANCE")
            lines.append(f"  Currently uses: {', '.join(self.MODEL_FEATURES[model])}")
            lines.append("-" * 70)

            importance = self.analyze_model_importance(model)
            lines.append(f"\n  {'Feature':<25} {'Importance':>12} {'Currently Used':>15}")
            lines.append("  " + "-" * 52)

            for feat, imp in list(importance.items())[:15]:
                used = "YES" if feat in self.MODEL_FEATURES[model] else ""
                bar = "█" * int(imp * 20)
                lines.append(f"  {feat:<25} {imp:>8.3f} {bar:<10} {used:>5}")

            # Recommendations
            lines.append(f"\n  RECOMMENDATIONS for {model}:")
            current_features = set(self.MODEL_FEATURES[model])

            # Features to add
            to_add = []
            for feat, imp in list(importance.items())[:10]:
                if feat not in current_features and imp > 0.1:
                    to_add.append(feat)

            if to_add:
                lines.append(f"    + Add: {', '.join(to_add[:3])}")

            # Features to remove (low importance)
            to_remove = []
            for feat in current_features:
                if importance.get(feat, 0) < 0.05:
                    to_remove.append(feat)

            if to_remove:
                lines.append(f"    - Consider removing: {', '.join(to_remove)}")

        # Feature quintile analysis for key features
        lines.append("\n" + "-" * 70)
        lines.append("  FEATURE QUINTILE ANALYSIS")
        lines.append("  (Accuracy at different feature value ranges)")
        lines.append("-" * 70)

        key_features = ["rsi", "volatility_z", "volume_z", "ma_delta_pct", "body_ratio"]
        for feat in key_features:
            lines.append(f"\n  {feat}:")
            for model in ["TRENDVIC", "OSCILLATOR", "VOLUMEMETRIX"]:
                quintiles = self.get_model_accuracy_by_feature_quintile(model, feat)
                if quintiles:
                    accs = [f"{q}: {a*100:.0f}%" for q, (a, _) in quintiles.items()]
                    lines.append(f"    {model:15} {' | '.join(accs)}")

        # Optimal feature sets
        lines.append("\n" + "-" * 70)
        lines.append("  RECOMMENDED OPTIMAL FEATURE SETS")
        lines.append("-" * 70)

        for model in ["TRENDVIC", "OSCILLATOR", "VOLUMEMETRIX"]:
            importance = self.analyze_model_importance(model)
            top_features = [f for f, i in list(importance.items())[:8] if i > 0.05]

            lines.append(f"\n  {model}:")
            lines.append(f"    Current: {self.MODEL_FEATURES[model]}")
            lines.append(f"    Optimal: {top_features}")

        # Missing features analysis
        lines.append("\n" + "-" * 70)
        lines.append("  POTENTIAL NEW FEATURES (from Chronos)")
        lines.append("-" * 70)
        lines.append("\n  Features that could improve prediction:")
        for feat in self.POTENTIAL_FEATURES:
            lines.append(f"    - {feat}")

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)


def run_analysis(symbol: str, interval: int, hours: int) -> FeatureAnalyzer:
    """Run feature analysis on historical data."""
    print(f"\nRunning feature analysis for {symbol} {interval}m ({hours}h)...")

    # Initialize config
    state_store = StateStore(":memory:")  # In-memory for analysis
    config = ConfigStore(state_store)
    config.ensure_defaults()

    stream = FeatureStream(config)

    # Initialize models
    models = {
        "TRENDVIC": TrendVIC(config),
        "OSCILLATOR": Oscillator(config),
        "VOLUMEMETRIX": VolumeMetrix(config),
    }

    analyzer = FeatureAnalyzer()

    # Fetch candles
    import time as time_module
    end_ts = int(time_module.time())
    start_ts = end_ts - (hours * 3600) - (100 * interval * 60)  # Extra for warmup
    candles = fetch_klines(symbol, str(interval), start_ts, end_ts)

    if not candles:
        print("ERROR: No candles fetched")
        return analyzer

    print(f"Fetched {len(candles)} candles")

    # Process candles
    prev_features = None
    warmup = 40
    processed = 0

    for i, candle in enumerate(candles[:-1]):  # Skip last (incomplete)
        features = stream.update(candle)

        if features is None:
            continue

        if i < warmup:
            prev_features = features
            continue

        # Get next candle for actual direction
        next_candle = candles[i + 1]
        actual_direction = "UP" if next_candle.close > candle.close else "DOWN"

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
            model_predictions=model_predictions,
            model_confidences=model_confidences,
            model_correct=model_correct,
        )

        analyzer.add_record(record)
        processed += 1

        if processed % 500 == 0:
            print(f"  Processed {processed} records...")

        prev_features = features

    print(f"Analysis complete: {processed} records")
    return analyzer


def main():
    parser = argparse.ArgumentParser(description="Analyze feature importance for Titan models")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--interval", type=int, default=1, help="Interval in minutes")
    parser.add_argument("--hours", type=int, default=168, help="Hours of history to analyze")
    parser.add_argument("--output", default=None, help="Output file (default: stdout)")

    args = parser.parse_args()

    analyzer = run_analysis(args.symbol, args.interval, args.hours)
    report = analyzer.generate_report()

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"\nReport saved to {args.output}")
    else:
        print(report)

    # Also save JSON results for further analysis
    results = {
        "direction_correlation": analyzer.analyze_direction_correlation(),
        "model_importance": {
            model: analyzer.analyze_model_importance(model)
            for model in ["TRENDVIC", "OSCILLATOR", "VOLUMEMETRIX"]
        },
        "records_count": len(analyzer.records),
    }

    json_path = Path("feature_analysis_results.json")
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nJSON results saved to {json_path}")


if __name__ == "__main__":
    main()
