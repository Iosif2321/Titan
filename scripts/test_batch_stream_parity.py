"""Test batch (training) vs stream (live) feature parity.

This script compares features computed by:
1. TFTDataset (batch mode - used in training)
2. FeatureStream (streaming mode - used in live)

If there's a mismatch, the model trained on batch data will perform
poorly on live streaming data.

Usage:
    python scripts/test_batch_stream_parity.py --csv data/candles_BTCUSDT_1m_168h.csv
"""
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List

from titan.core.config import ConfigStore
from titan.core.state_store import StateStore
from titan.core.features.stream import FeatureStream
from titan.core.training import TFTDataset
from titan.core.models.tft import ALL_FEATURES, TREND_FEATURES, OSCILLATOR_FEATURES, VOLUME_FEATURES
from titan.core.data.schema import Candle


def load_candles_from_csv(csv_path: str) -> List[Candle]:
    """Load candles from CSV file."""
    df = pd.read_csv(csv_path)
    candles = []
    for _, row in df.iterrows():
        candle = Candle(
            ts=int(row["ts"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
        candles.append(candle)
    return candles


def compare_features(batch_features: Dict[str, float], stream_features: Dict[str, float]) -> Dict[str, float]:
    """Compare batch and stream features, return differences."""
    diffs = {}
    for key in batch_features:
        if key in stream_features:
            batch_val = batch_features[key]
            stream_val = stream_features[key]
            if batch_val != 0:
                diff_pct = abs(batch_val - stream_val) / abs(batch_val) * 100
            else:
                diff_pct = abs(stream_val) * 100 if stream_val != 0 else 0
            diffs[key] = diff_pct
    return diffs


def main():
    parser = argparse.ArgumentParser(description="Test batch/stream feature parity")
    parser.add_argument("--csv", required=True, help="Path to candles CSV file")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to test")
    args = parser.parse_args()

    print("=" * 70)
    print("BATCH vs STREAM FEATURE PARITY TEST")
    print("=" * 70)

    # Load candles
    candles = load_candles_from_csv(args.csv)
    print(f"Loaded {len(candles)} candles from {args.csv}")

    # Setup
    state_store = StateStore(":memory:")
    config = ConfigStore(state_store)
    config.ensure_defaults()

    # Create FeatureStream and warm it up
    # Need 100 (seq_len) + 50 (normalizer window) = 150 candles
    stream = FeatureStream(config)
    warmup_needed = 150

    print(f"\nWarming up FeatureStream with {warmup_needed} candles...")
    for i in range(warmup_needed):
        stream.update(candles[i])

    # Create TFTDataset
    dataset = TFTDataset(args.csv, seq_len=100, label_horizon=1)
    print(f"TFTDataset has {len(dataset)} samples")

    # Compare features for several samples
    # Start from warmup_needed, compare against dataset[warmup_needed - 99]
    test_start = warmup_needed
    test_end = min(test_start + args.samples, len(candles) - 1)
    print(f"Starting comparison from candle {test_start}, dataset[{test_start - 99}]")

    print(f"\nComparing features for samples {test_start} to {test_end}...")
    print("-" * 70)

    all_diffs = {feat: [] for feat in ALL_FEATURES}
    mismatch_count = 0

    for i in range(test_start, test_end):
        # Get stream features
        stream_features = stream.update(candles[i])
        if stream_features is None:
            continue

        # Get batch features (from dataset at corresponding index)
        # Dataset[idx] has sequence [idx : idx + 100], last candle at idx + 99
        # So for candle i: idx = i - 99
        dataset_idx = i - 99
        if dataset_idx < 0 or dataset_idx >= len(dataset):
            continue

        # Get the sequence and extract last row's features
        seq, trend_f, osc_f, vol_f, label, ret, ts = dataset[dataset_idx]

        # Build batch features dict from the last row of sequence
        batch_features = {}
        for j, feat in enumerate(dataset.feature_cols):
            if feat in ALL_FEATURES:
                batch_features[feat] = seq[-1, j].item()

        # Compare
        diffs = compare_features(batch_features, stream_features)

        has_mismatch = False
        for feat, diff_pct in diffs.items():
            all_diffs[feat].append(diff_pct)
            if diff_pct > 1.0:  # More than 1% difference
                has_mismatch = True

        if has_mismatch:
            mismatch_count += 1

    # Summary
    print("\nFEATURE PARITY SUMMARY:")
    print("-" * 70)
    print(f"{'Feature':<25} {'Mean Diff %':<15} {'Max Diff %':<15} {'Status'}")
    print("-" * 70)

    critical_mismatches = []
    for feat in ALL_FEATURES:
        if all_diffs[feat]:
            mean_diff = np.mean(all_diffs[feat])
            max_diff = np.max(all_diffs[feat])
            if max_diff > 5.0:
                status = "CRITICAL"
                critical_mismatches.append(feat)
            elif max_diff > 1.0:
                status = "WARNING"
            else:
                status = "OK"
            print(f"{feat:<25} {mean_diff:<15.4f} {max_diff:<15.4f} {status}")

    print("-" * 70)
    print(f"\nSamples with >1% mismatch: {mismatch_count}/{test_end - test_start}")

    if critical_mismatches:
        print(f"\nCRITICAL MISMATCHES (>5%): {critical_mismatches}")
        print("These features differ significantly between batch and stream mode!")
        print("The model may perform poorly in live mode.")
        return 1
    else:
        print("\nAll features within acceptable tolerance.")
        print("Batch and stream modes are aligned.")
        return 0


if __name__ == "__main__":
    exit(main())
