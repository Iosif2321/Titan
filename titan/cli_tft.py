"""CLI commands for TFT model training and inference.

Sprint 23: Two-Head MLP & SessionEmbeddingMLP support.

Usage:
    python -m titan.cli_tft train --model-class SessionEmbeddingMLP --dropout 0.4 --hours 168
    python -m titan.cli_tft test --checkpoint checkpoints/best.pt
"""
import argparse
import os
import sys
import time
from typing import Optional

import torch


def fetch_training_data(symbol: str, interval: str, hours: int) -> str:
    """Fetch historical data for training.

    Returns path to saved CSV file.
    """
    from titan.core.data.bybit_rest import fetch_klines

    print(f"Fetching {hours} hours of {symbol} {interval}m data...")

    # Calculate time range
    end_ts = int(time.time())
    start_ts = end_ts - (hours * 3600)

    # Fetch in batches (Bybit limit is 1000)
    candles = fetch_klines(symbol, interval, start_ts, end_ts)

    if not candles:
        raise RuntimeError("No candles fetched")

    print(f"Fetched {len(candles)} candles")

    # Save to CSV
    import pandas as pd

    data = []
    for c in candles:
        data.append({
            # Keep column name aligned with CsvCandleReader/TFTDataset ("timestamp")
            "timestamp": c.ts,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        })

    df = pd.DataFrame(data)

    # Save path
    os.makedirs("data", exist_ok=True)
    path = f"data/candles_{symbol}_{interval}m_{hours}h.csv"
    df.to_csv(path, index=False)
    print(f"Saved to {path}")

    return path


def train_model(
    model_class_name: str,
    data_path: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    seq_len: int = 100,
    label_horizon: int = 1,
    dropout: float = 0.2,
    weight_decay: float = 0.01,
    checkpoint_dir: str = "checkpoints",
) -> None:
    """Train selected model architecture on historical data."""
    from titan.core.config import ConfigStore
    from titan.core.models.tft import (
        ThreeHeadTFT, TwoHeadMLP, SessionEmbeddingMLP, SessionGatedMLP, ALL_FEATURES
    )
    from titan.core.state_store import StateStore
    from titan.core.training import TFTTrainer, TFTDataset, TrainingConfig

    print("=" * 60)
    print(f"{model_class_name} Training")
    print("=" * 60)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Config
    state_store = StateStore(":memory:")
    config = ConfigStore(state_store)
    config.ensure_defaults()

    # Override config
    config.set("tft.seq_len", seq_len)
    config.set("tft.dropout", dropout)

    # Load dataset
    print(f"\nLoading data from {data_path}...")
    full_dataset = TFTDataset(data_path, seq_len=seq_len, label_horizon=label_horizon)
    print(f"Total samples: {len(full_dataset)}")

    # Split train/val (80/20) with gap to prevent data leakage
    total_size = len(full_dataset)
    gap = label_horizon
    train_size = int(0.8 * total_size) - gap
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size + gap, total_size))
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model
    print(f"\nCreating model: {model_class_name}...")
    
    common_args = {
        "config": config,
        "input_dim": len(ALL_FEATURES),
        "hidden_dim": 64,
        "dropout": dropout,
    }

    if model_class_name == "SessionEmbeddingMLP":
        model = SessionEmbeddingMLP(
            **common_args,
            num_lstm_layers=1,
            session_embed_dim=8
        )
    elif model_class_name == "SessionGatedMLP":
        model = SessionGatedMLP(
            **common_args,
            num_lstm_layers=1
        )
    elif model_class_name == "TwoHeadMLP":
        model = TwoHeadMLP(
            **common_args,
            num_lstm_layers=1
        )
    else: # ThreeHeadTFT
        model = ThreeHeadTFT(
            **common_args,
            num_heads=4,
            num_lstm_layers=2
        )

    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Training config
    training_config = TrainingConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        seq_len=seq_len,
        gradient_clip=1.0,
        weight_decay=weight_decay,
        early_stopping_patience=20,
        checkpoint_dir=checkpoint_dir,
    )

    # Create trainer
    trainer = TFTTrainer(model, config, training_config, device)

    # Wrap Subset for DataLoader
    class SubsetDataset(torch.utils.data.Dataset):
        def __init__(self, subset):
            self.subset = subset
        def __len__(self): return len(self.subset)
        def __getitem__(self, idx): return self.subset[idx]

    # Train
    print("\nStarting training...")
    trainer.train_offline(
        SubsetDataset(train_dataset),
        SubsetDataset(val_dataset),
    )


def test_model(checkpoint_path: str, model_class_name: str = "TwoHeadMLP") -> None:
    """Test trained model."""
    from titan.core.config import ConfigStore
    from titan.core.models.tft import (
        ThreeHeadTFT, TwoHeadMLP, SessionEmbeddingMLP, SessionGatedMLP, ALL_FEATURES
    )
    from titan.core.state_store import StateStore

    print("=" * 60)
    print(f"{model_class_name} Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Config
    state_store = StateStore(":memory:")
    config = ConfigStore(state_store)
    config.ensure_defaults()

    # Recreate model structure
    common_args = {
        "config": config,
        "input_dim": len(ALL_FEATURES),
        "hidden_dim": 64,
        # Default architecture params matching training
        "dropout": 0.0  # Dropout not needed for inference structure
    }

    if model_class_name == "SessionEmbeddingMLP":
        model = SessionEmbeddingMLP(**common_args, num_lstm_layers=1, session_embed_dim=8)
    elif model_class_name == "SessionGatedMLP":
        model = SessionGatedMLP(**common_args, num_lstm_layers=1)
    elif model_class_name == "TwoHeadMLP":
        model = TwoHeadMLP(**common_args, num_lstm_layers=1)
    else:
        model = ThreeHeadTFT(**common_args, num_heads=4, num_lstm_layers=2)

    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
    print("\nModel is ready for inference.")


def main():
    parser = argparse.ArgumentParser(description="Titan Model Training CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--model-class", default="TwoHeadMLP", 
                             choices=["ThreeHeadTFT", "TwoHeadMLP", "SessionEmbeddingMLP", "SessionGatedMLP"])
    train_parser.add_argument("--symbol", default="BTCUSDT")
    train_parser.add_argument("--interval", default="5", help="Candle interval in minutes (default: 5)")
    train_parser.add_argument("--hours", type=int, default=168)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=0.001)
    train_parser.add_argument("--dropout", type=float, default=0.2)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--label-horizon", type=int, default=1)
    train_parser.add_argument("--data", help="Path to existing CSV")
    train_parser.add_argument("--checkpoint-dir", default="checkpoints")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test model")
    test_parser.add_argument("--checkpoint", required=True)
    test_parser.add_argument("--model-class", default="TwoHeadMLP")

    args = parser.parse_args()

    if args.command == "train":
        data_path = args.data if args.data else fetch_training_data(args.symbol, args.interval, args.hours)
        train_model(
            model_class_name=args.model_class,
            data_path=data_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            label_horizon=args.label_horizon,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            checkpoint_dir=args.checkpoint_dir,
        )
    elif args.command == "test":
        test_model(args.checkpoint, args.model_class)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
