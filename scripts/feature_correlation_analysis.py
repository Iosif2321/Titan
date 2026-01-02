#!/usr/bin/env python3
"""
Feature Correlation Analysis for Titan Models

Analyzes:
1. Correlation of each feature with price direction
2. Feature distributions and statistics
3. Potential new features to add
4. Optimal feature distribution across models

Usage:
    python scripts/feature_correlation_analysis.py --hours 168
"""

import argparse
import json
import math
import sys
import time as time_module
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from titan.core.config import ConfigStore
from titan.core.data.bybit_rest import fetch_klines
from titan.core.features.stream import FeatureStream
from titan.core.state_store import StateStore


@dataclass
class FeatureStats:
    """Statistics for a single feature."""
    name: str
    mean: float
    std: float
    min_val: float
    max_val: float
    correlation: float  # Point-biserial correlation with direction
    info_gain: float  # Information gain
    quintile_accuracy: List[float]  # Accuracy per quintile
    up_mean: float  # Mean when actual=UP
    down_mean: float  # Mean when actual=DOWN
    effect_size: float  # Cohen's d


def point_biserial_correlation(feature_values: List[float], binary_labels: List[int]) -> float:
    """Calculate point-biserial correlation between continuous and binary variable."""
    n = len(feature_values)
    if n < 2:
        return 0.0

    # Split by label
    group_0 = [f for f, b in zip(feature_values, binary_labels) if b == 0]
    group_1 = [f for f, b in zip(feature_values, binary_labels) if b == 1]

    if not group_0 or not group_1:
        return 0.0

    mean_0 = sum(group_0) / len(group_0)
    mean_1 = sum(group_1) / len(group_1)

    # Overall stats
    mean_all = sum(feature_values) / n
    var_all = sum((x - mean_all) ** 2 for x in feature_values) / n
    std_all = math.sqrt(var_all) if var_all > 0 else 1e-10

    p0 = len(group_0) / n
    p1 = len(group_1) / n

    rpb = (mean_1 - mean_0) / std_all * math.sqrt(p0 * p1)
    return rpb


def calculate_info_gain(feature_values: List[float], labels: List[int], n_bins: int = 5) -> float:
    """Calculate information gain using binned feature values."""
    n = len(feature_values)
    if n < 10:
        return 0.0

    # Base entropy
    p_up = sum(labels) / n
    p_down = 1 - p_up
    if p_up == 0 or p_up == 1:
        return 0.0
    base_entropy = -p_up * math.log2(p_up) - p_down * math.log2(p_down)

    # Bin features
    sorted_idx = sorted(range(n), key=lambda i: feature_values[i])
    bin_size = n // n_bins

    weighted_entropy = 0.0
    for b in range(n_bins):
        start = b * bin_size
        end = (b + 1) * bin_size if b < n_bins - 1 else n
        bin_labels = [labels[sorted_idx[i]] for i in range(start, end)]

        if not bin_labels:
            continue

        p_up_bin = sum(bin_labels) / len(bin_labels)
        p_down_bin = 1 - p_up_bin

        if p_up_bin == 0 or p_up_bin == 1:
            bin_entropy = 0
        else:
            bin_entropy = -p_up_bin * math.log2(p_up_bin) - p_down_bin * math.log2(p_down_bin)

        weighted_entropy += (len(bin_labels) / n) * bin_entropy

    return base_entropy - weighted_entropy


def calculate_quintile_accuracy(
    feature_values: List[float],
    labels: List[int],
    predictions: List[int]
) -> List[float]:
    """Calculate accuracy per quintile of feature values."""
    n = len(feature_values)
    if n < 10:
        return [0.5] * 5

    # Sort by feature value
    sorted_data = sorted(zip(feature_values, labels, predictions), key=lambda x: x[0])
    quintile_size = n // 5

    accuracies = []
    for q in range(5):
        start = q * quintile_size
        end = (q + 1) * quintile_size if q < 4 else n
        q_data = sorted_data[start:end]

        correct = sum(1 for _, label, pred in q_data if label == pred)
        acc = correct / len(q_data) if q_data else 0.5
        accuracies.append(acc)

    return accuracies


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Calculate Cohen's d effect size."""
    if not group1 or not group2:
        return 0.0

    mean1 = sum(group1) / len(group1)
    mean2 = sum(group2) / len(group2)

    var1 = sum((x - mean1) ** 2 for x in group1) / max(len(group1) - 1, 1)
    var2 = sum((x - mean2) ** 2 for x in group2) / max(len(group2) - 1, 1)

    pooled_std = math.sqrt((var1 + var2) / 2)
    if pooled_std == 0:
        return 0.0

    return (mean1 - mean2) / pooled_std


class FeatureAnalyzer:
    """Comprehensive feature analysis."""

    def __init__(self, hours: int = 168, symbol: str = "BTCUSDT", interval: int = 1):
        self.hours = hours
        self.symbol = symbol
        self.interval = interval

        self.state_store = StateStore(":memory:")
        self.config = ConfigStore(self.state_store)
        self.config.ensure_defaults()

        self.features_data: Dict[str, List[float]] = defaultdict(list)
        self.actuals: List[int] = []  # 1=UP, 0=DOWN
        self.predictions: List[int] = []  # From simple model
        self.feature_stats: Dict[str, FeatureStats] = {}

    def load_data(self):
        """Load and prepare data."""
        print(f"\n{'='*70}")
        print(f"  Loading {self.hours}h of {self.symbol} {self.interval}m data...")
        print(f"{'='*70}")

        stream = FeatureStream(self.config)

        end_ts = int(time_module.time())
        start_ts = end_ts - (self.hours * 3600) - (100 * self.interval * 60)
        candles = fetch_klines(self.symbol, str(self.interval), start_ts, end_ts)

        if not candles:
            raise ValueError("No candles fetched")

        print(f"  Fetched {len(candles)} candles")

        warmup = 40
        for i, candle in enumerate(candles[:-1]):
            features = stream.update(candle)
            if features is None or i < warmup:
                continue

            next_candle = candles[i + 1]
            actual = 1 if next_candle.close > candle.close else 0

            # Store all features
            for fname, fval in features.items():
                self.features_data[fname].append(fval)

            self.actuals.append(actual)

            # Simple prediction based on ma_delta
            pred = 1 if features.get("ma_delta", 0) >= 0 else 0
            self.predictions.append(pred)

        print(f"  Prepared {len(self.actuals)} samples")
        print(f"  Features: {len(self.features_data)}")

    def analyze_features(self):
        """Analyze all features."""
        print(f"\n{'='*70}")
        print(f"  Analyzing {len(self.features_data)} features...")
        print(f"{'='*70}")

        for fname, fvalues in self.features_data.items():
            if len(fvalues) < 100:
                continue

            # Basic stats
            mean_val = sum(fvalues) / len(fvalues)
            var_val = sum((x - mean_val) ** 2 for x in fvalues) / len(fvalues)
            std_val = math.sqrt(var_val)

            # Correlation
            corr = point_biserial_correlation(fvalues, self.actuals)

            # Info gain
            info_gain = calculate_info_gain(fvalues, self.actuals)

            # Quintile accuracy
            quintile_acc = calculate_quintile_accuracy(fvalues, self.actuals, self.predictions)

            # Split by direction
            up_values = [f for f, a in zip(fvalues, self.actuals) if a == 1]
            down_values = [f for f, a in zip(fvalues, self.actuals) if a == 0]

            up_mean = sum(up_values) / len(up_values) if up_values else 0
            down_mean = sum(down_values) / len(down_values) if down_values else 0

            effect_size = cohens_d(up_values, down_values)

            self.feature_stats[fname] = FeatureStats(
                name=fname,
                mean=mean_val,
                std=std_val,
                min_val=min(fvalues),
                max_val=max(fvalues),
                correlation=corr,
                info_gain=info_gain,
                quintile_accuracy=quintile_acc,
                up_mean=up_mean,
                down_mean=down_mean,
                effect_size=effect_size,
            )

    def print_correlation_ranking(self):
        """Print features ranked by correlation."""
        print(f"\n{'='*70}")
        print("  FEATURE CORRELATION RANKING (with price direction)")
        print(f"{'='*70}")

        sorted_features = sorted(
            self.feature_stats.values(),
            key=lambda x: abs(x.correlation),
            reverse=True
        )

        print(f"\n  {'Feature':<25} {'Corr':>8} {'InfoGain':>10} {'Effect':>8} {'Interpretation'}")
        print("  " + "-" * 75)

        for fs in sorted_features[:30]:
            interpretation = self._interpret_correlation(fs)
            print(f"  {fs.name:<25} {fs.correlation:>+8.4f} {fs.info_gain:>10.4f} {fs.effect_size:>+8.3f} {interpretation}")

    def _interpret_correlation(self, fs: FeatureStats) -> str:
        """Interpret what the correlation means."""
        corr = fs.correlation
        if abs(corr) < 0.01:
            return "No effect"
        elif corr > 0.05:
            return f"High {fs.name} -> UP"
        elif corr < -0.05:
            return f"High {fs.name} -> DOWN"
        elif corr > 0:
            return f"Weak: High -> UP"
        else:
            return f"Weak: High -> DOWN"

    def print_quintile_analysis(self):
        """Print quintile analysis for top features."""
        print(f"\n{'='*70}")
        print("  QUINTILE ACCURACY ANALYSIS (Top 15 features)")
        print(f"{'='*70}")

        sorted_features = sorted(
            self.feature_stats.values(),
            key=lambda x: abs(x.correlation),
            reverse=True
        )

        print(f"\n  {'Feature':<25} {'Q1':>7} {'Q2':>7} {'Q3':>7} {'Q4':>7} {'Q5':>7} {'Best Q'}")
        print("  " + "-" * 70)

        for fs in sorted_features[:15]:
            q_str = [f"{q*100:.1f}%" for q in fs.quintile_accuracy]
            best_q = fs.quintile_accuracy.index(max(fs.quintile_accuracy)) + 1
            print(f"  {fs.name:<25} {q_str[0]:>7} {q_str[1]:>7} {q_str[2]:>7} {q_str[3]:>7} {q_str[4]:>7} Q{best_q}")

    def analyze_missing_features(self):
        """Identify potentially useful features not yet implemented."""
        print(f"\n{'='*70}")
        print("  POTENTIAL NEW FEATURES TO ADD")
        print(f"{'='*70}")

        # Check what we have
        existing = set(self.features_data.keys())

        potential_features = [
            ("vol_imbalance_20", "Volume imbalance (UP vs DOWN volume)", "VOLUMEMETRIX"),
            ("bb_position", "Bollinger Band position (0-1)", "OSCILLATOR"),
            ("bb_width_pct", "Bollinger Band width %", "OSCILLATOR"),
            ("macd_pct", "MACD as % of price", "TRENDVIC"),
            ("macd_signal_pct", "MACD signal as % of price", "TRENDVIC"),
            ("obv_z", "On-Balance Volume z-score", "VOLUMEMETRIX"),
            ("vwap_spread_pct", "VWAP spread as % of price", "VOLUMEMETRIX"),
            ("stoch_k", "Stochastic %K", "OSCILLATOR"),
            ("stoch_d", "Stochastic %D", "OSCILLATOR"),
            ("adx", "Average Directional Index", "TRENDVIC"),
            ("di_plus", "DI+", "TRENDVIC"),
            ("di_minus", "DI-", "TRENDVIC"),
            ("mfi", "Money Flow Index", "VOLUMEMETRIX"),
            ("cmf", "Chaikin Money Flow", "VOLUMEMETRIX"),
            ("roc_5", "Rate of Change 5-period", "TRENDVIC"),
            ("roc_10", "Rate of Change 10-period", "TRENDVIC"),
            ("pivot_distance", "Distance from pivot point", "OSCILLATOR"),
            ("support_distance", "Distance from support", "OSCILLATOR"),
            ("resistance_distance", "Distance from resistance", "OSCILLATOR"),
        ]

        print(f"\n  {'Feature':<25} {'Description':<35} {'Best For'}")
        print("  " + "-" * 70)

        for fname, desc, model in potential_features:
            if fname not in existing:
                print(f"  {fname:<25} {desc:<35} {model}")

    def recommend_feature_distribution(self):
        """Recommend feature distribution across models."""
        print(f"\n{'='*70}")
        print("  RECOMMENDED FEATURE DISTRIBUTION")
        print(f"{'='*70}")

        # Current model feature usage
        current_usage = {
            "TRENDVIC": ["ma_delta", "volatility", "close", "body_ratio",
                        "candle_direction", "price_momentum_3"],
            "OSCILLATOR": ["rsi", "rsi_momentum", "volatility_z"],
            "VOLUMEMETRIX": ["volume_z", "return_1", "volatility", "ma_delta",
                           "volume_trend", "upper_wick_ratio", "lower_wick_ratio"],
        }

        # Analyze which features are best for each model's philosophy
        print("\n  === TRENDVIC (Trend Following) ===")
        print("  Philosophy: Follow the trend, use momentum indicators")
        trend_features = [
            fs for fs in self.feature_stats.values()
            if any(x in fs.name for x in ['ma', 'ema', 'momentum', 'trend', 'return_5', 'return_10'])
        ]
        trend_sorted = sorted(trend_features, key=lambda x: abs(x.correlation), reverse=True)
        print("  Best features by correlation:")
        for fs in trend_sorted[:8]:
            in_use = "USED" if fs.name in current_usage["TRENDVIC"] else "ADD?"
            print(f"    {fs.name:<25} corr={fs.correlation:>+.4f} [{in_use}]")

        print("\n  === OSCILLATOR (Mean Reversion) ===")
        print("  Philosophy: Overbought/oversold, expect reversion")
        osc_features = [
            fs for fs in self.feature_stats.values()
            if any(x in fs.name for x in ['rsi', 'overbought', 'oversold', 'neutral', 'stoch'])
        ]
        osc_sorted = sorted(osc_features, key=lambda x: abs(x.correlation), reverse=True)
        print("  Best features by correlation:")
        for fs in osc_sorted[:8]:
            in_use = "USED" if fs.name in current_usage["OSCILLATOR"] else "ADD?"
            print(f"    {fs.name:<25} corr={fs.correlation:>+.4f} [{in_use}]")

        print("\n  === VOLUMEMETRIX (Volume-Price) ===")
        print("  Philosophy: Volume confirms/denies price moves")
        vol_features = [
            fs for fs in self.feature_stats.values()
            if any(x in fs.name for x in ['volume', 'vol_', 'body', 'wick', 'candle'])
        ]
        vol_sorted = sorted(vol_features, key=lambda x: abs(x.correlation), reverse=True)
        print("  Best features by correlation:")
        for fs in vol_sorted[:8]:
            in_use = "USED" if fs.name in current_usage["VOLUMEMETRIX"] else "ADD?"
            print(f"    {fs.name:<25} corr={fs.correlation:>+.4f} [{in_use}]")

    def print_actionable_recommendations(self):
        """Print specific actionable recommendations."""
        print(f"\n{'='*70}")
        print("  ACTIONABLE RECOMMENDATIONS")
        print(f"{'='*70}")

        # Find features with highest absolute correlation that aren't heavily used
        all_sorted = sorted(
            self.feature_stats.values(),
            key=lambda x: abs(x.correlation),
            reverse=True
        )

        print("\n  TOP 10 MOST PREDICTIVE FEATURES:")
        for i, fs in enumerate(all_sorted[:10], 1):
            direction = "UP" if fs.correlation > 0 else "DOWN"
            print(f"  {i:2d}. {fs.name:<25} corr={fs.correlation:>+.4f} (high value -> {direction})")

        print("\n  SPECIFIC RECOMMENDATIONS:")

        # Check for underutilized features
        recs = []

        # RSI zones
        rsi_oversold = self.feature_stats.get("rsi_oversold")
        rsi_overbought = self.feature_stats.get("rsi_overbought")
        if rsi_oversold and rsi_overbought:
            if abs(rsi_oversold.correlation) > 0.02 or abs(rsi_overbought.correlation) > 0.02:
                recs.append(("OSCILLATOR", "rsi_oversold/rsi_overbought",
                           f"corr: {rsi_oversold.correlation:+.4f}/{rsi_overbought.correlation:+.4f}"))

        # Candle direction (mean reversion signal)
        candle_dir = self.feature_stats.get("candle_direction")
        if candle_dir and abs(candle_dir.correlation) > 0.03:
            recs.append(("ALL MODELS", "candle_direction",
                        f"corr: {candle_dir.correlation:+.4f} (mean reversion!)"))

        # Return lag features
        for i in range(1, 6):
            lag_name = f"return_lag_{i}"
            lag_fs = self.feature_stats.get(lag_name)
            if lag_fs and abs(lag_fs.correlation) > 0.02:
                recs.append(("OSCILLATOR/TREND", lag_name,
                           f"corr: {lag_fs.correlation:+.4f}"))

        # Volume features
        vol_change = self.feature_stats.get("volume_change_pct")
        if vol_change and abs(vol_change.correlation) > 0.02:
            recs.append(("VOLUMEMETRIX", "volume_change_pct",
                        f"corr: {vol_change.correlation:+.4f}"))

        # ATR
        atr = self.feature_stats.get("atr_pct")
        if atr and abs(atr.correlation) > 0.02:
            recs.append(("ALL MODELS", "atr_pct",
                        f"corr: {atr.correlation:+.4f}"))

        for model, feature, reason in recs:
            print(f"  - Add {feature} to {model}: {reason}")

        # New features to implement
        print("\n  NEW FEATURES TO IMPLEMENT (Priority Order):")
        print("  1. vol_imbalance_20: Volume UP vs DOWN ratio (VOLUMEMETRIX)")
        print("  2. bb_position: Bollinger Band position 0-1 (OSCILLATOR)")
        print("  3. adx: Average Directional Index (TRENDVIC)")
        print("  4. mfi: Money Flow Index (VOLUMEMETRIX)")

    def save_results(self, path: str = "feature_analysis_results.json"):
        """Save results to JSON."""
        data = {
            "hours": self.hours,
            "symbol": self.symbol,
            "total_samples": len(self.actuals),
            "features": {
                name: {
                    "correlation": fs.correlation,
                    "info_gain": fs.info_gain,
                    "effect_size": fs.effect_size,
                    "mean": fs.mean,
                    "std": fs.std,
                    "quintile_accuracy": fs.quintile_accuracy,
                }
                for name, fs in self.feature_stats.items()
            }
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"\nResults saved to {path}")

    def run(self):
        """Run complete analysis."""
        self.load_data()
        self.analyze_features()
        self.print_correlation_ranking()
        self.print_quintile_analysis()
        self.analyze_missing_features()
        self.recommend_feature_distribution()
        self.print_actionable_recommendations()
        self.save_results()


def main():
    parser = argparse.ArgumentParser(description="Feature Correlation Analysis")
    parser.add_argument("--hours", type=int, default=168, help="Hours of data")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol")
    parser.add_argument("--interval", type=int, default=1, help="Interval")

    args = parser.parse_args()

    analyzer = FeatureAnalyzer(hours=args.hours, symbol=args.symbol, interval=args.interval)
    analyzer.run()


if __name__ == "__main__":
    main()
