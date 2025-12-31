"""Training module for Three-Head TFT Model.

Sprint 23: Implements offline pretraining and online fine-tuning for TFT.

Training modes:
    1. Offline: Train on historical candle data (CSV)
    2. Online: Fine-tune during live trading with reward feedback

Reward function (R2 with streak bonus):
    reward = return_pct * direction_match * streak_mult
    streak_mult = 1.0 + 0.1 * min(streak_length, 5)  # max 1.5x
"""
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from titan.core.config import ConfigStore
from titan.core.models.tft import (
    ThreeHeadTFT,
    ALL_FEATURES,
    TREND_FEATURES,
    OSCILLATOR_FEATURES,
    VOLUME_FEATURES,
)


@dataclass
class TrainingConfig:
    """Configuration for TFT training."""
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    seq_len: int = 100
    gradient_clip: float = 1.0
    weight_decay: float = 0.01
    warmup_steps: int = 100
    early_stopping_patience: int = 10
    early_stopping_delta: float = 0.001
    val_split: float = 0.2
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 100


class RewardCalculator:
    """Calculate rewards for RL-based training.

    Reward function (R2 with streak bonus):
        reward = return_pct * direction_match * streak_mult
        streak_mult = 1.0 + 0.1 * min(streak_length, 5)  # max 1.5x
    """

    def __init__(self, config: ConfigStore):
        self._config = config
        self._streak_length = 0
        self._last_hit = False

    def calculate(
        self,
        predicted_direction: str,
        actual_direction: str,
        return_pct: float,
        confidence: float,
    ) -> float:
        """Calculate reward for a prediction.

        Args:
            predicted_direction: "UP" or "DOWN"
            actual_direction: "UP" or "DOWN"
            return_pct: Actual return percentage
            confidence: Model confidence (0.5-1.0)

        Returns:
            Reward value (can be negative)
        """
        # Direction match: +1 if correct, -1 if wrong
        is_hit = predicted_direction == actual_direction
        direction_match = 1.0 if is_hit else -1.0

        # Update streak
        if is_hit:
            if self._last_hit:
                self._streak_length += 1
            else:
                self._streak_length = 1
        else:
            self._streak_length = 0
        self._last_hit = is_hit

        # Streak multiplier (max 1.5x for 5+ streak)
        streak_mult = 1.0 + 0.1 * min(self._streak_length, 5)

        # R2 reward: return-weighted with streak bonus
        reward = return_pct * direction_match * streak_mult

        # Optional: confidence bonus/penalty
        # Higher confidence on correct = more reward
        # Higher confidence on wrong = more penalty
        confidence_factor = float(self._config.get("training.confidence_factor", 0.0))
        if confidence_factor > 0:
            reward *= 1.0 + confidence_factor * (confidence - 0.5) * direction_match

        return reward

    def reset_streak(self) -> None:
        """Reset streak counter (e.g., at start of new session)."""
        self._streak_length = 0
        self._last_hit = False

    def get_streak(self) -> int:
        """Get current streak length."""
        return self._streak_length


class TFTDataset(Dataset):
    """Dataset for TFT training from historical candle data."""

    def __init__(
        self,
        candles_path: str,
        seq_len: int = 100,
        feature_cols: Optional[List[str]] = None,
    ):
        """Initialize dataset from candles CSV.

        Args:
            candles_path: Path to candles CSV file
            seq_len: Sequence length for model input
            feature_cols: Feature column names (default: ALL_FEATURES)
        """
        self.seq_len = seq_len
        self.feature_cols = feature_cols or ALL_FEATURES

        # Load and preprocess data
        self.data = self._load_data(candles_path)
        self.labels = self._compute_labels()

        # Validate
        if len(self.data) < seq_len + 1:
            raise ValueError(f"Not enough data: {len(self.data)} < {seq_len + 1}")

    def _load_data(self, path: str) -> torch.Tensor:
        """Load candles CSV and convert to tensor."""
        import pandas as pd

        df = pd.read_csv(path)

        # Compute features from OHLCV if needed
        df = self._compute_features(df)

        # Extract feature columns (fill missing with 0)
        feature_data = []
        for col in self.feature_cols:
            if col in df.columns:
                feature_data.append(df[col].fillna(0).values)
            else:
                feature_data.append(np.zeros(len(df)))

        # Stack and transpose to [n_samples, n_features]
        import numpy as np
        data = np.stack(feature_data, axis=1).astype(np.float32)

        return torch.from_numpy(data)

    def _compute_features(self, df) -> "pd.DataFrame":
        """Compute technical features from OHLCV."""
        import numpy as np

        # Basic features
        df["return_1"] = df["close"].pct_change().fillna(0)

        # Moving averages
        df["ma_fast"] = df["close"].rolling(5).mean()
        df["ma_slow"] = df["close"].rolling(20).mean()
        df["ma_delta"] = df["ma_fast"] - df["ma_slow"]
        df["ma_delta_pct"] = df["ma_delta"] / df["close"]

        # EMAs
        df["ema_10"] = df["close"].ewm(span=10).mean()
        df["ema_20"] = df["close"].ewm(span=20).mean()
        df["ema_10_spread_pct"] = (df["close"] - df["ema_10"]) / df["close"]
        df["ema_20_spread_pct"] = (df["close"] - df["ema_20"]) / df["close"]

        # Volatility
        df["volatility"] = df["return_1"].rolling(20).std()
        vol_mean = df["volatility"].rolling(100).mean()
        vol_std = df["volatility"].rolling(100).std()
        df["volatility_z"] = (df["volatility"] - vol_mean) / (vol_std + 1e-10)

        # Volume
        df["volume_ma"] = df["volume"].rolling(20).mean()
        df["volume_std"] = df["volume"].rolling(20).std()
        df["volume_z"] = (df["volume"] - df["volume_ma"]) / (df["volume_std"] + 1e-10)
        df["volume_trend"] = (df["volume"] > df["volume_ma"]).astype(float)
        df["volume_change_pct"] = df["volume"].pct_change().fillna(0)

        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi_momentum"] = df["rsi"].diff()
        df["rsi_oversold"] = (df["rsi"] < 30).astype(float)
        df["rsi_overbought"] = (df["rsi"] > 70).astype(float)

        # Price momentum
        df["price_momentum_3"] = df["close"].pct_change(3)
        df["return_5"] = df["close"].pct_change(5)
        df["return_10"] = df["close"].pct_change(10)

        # Candle features
        df["body"] = abs(df["close"] - df["open"])
        df["range"] = df["high"] - df["low"]
        df["body_ratio"] = df["body"] / (df["range"] + 1e-10)
        df["candle_direction"] = (df["close"] > df["open"]).astype(float)
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["upper_wick_ratio"] = df["upper_wick"] / (df["range"] + 1e-10)
        df["lower_wick_ratio"] = df["lower_wick"] / (df["range"] + 1e-10)
        df["body_pct"] = df["body"] / df["close"]
        df["high_low_range_pct"] = df["range"] / df["close"]

        # ATR
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1))
            )
        )
        df["atr"] = df["tr"].rolling(14).mean()
        df["atr_pct"] = df["atr"] / df["close"]

        # ADX (simplified)
        plus_dm = df["high"].diff()
        minus_dm = -df["low"].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        df["adx"] = (plus_dm.rolling(14).mean() + minus_dm.rolling(14).mean()) / 2

        # Bollinger Bands position
        bb_mid = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_position"] = (df["close"] - bb_mid) / (2 * bb_std + 1e-10)

        # Stochastic
        low_14 = df["low"].rolling(14).min()
        high_14 = df["high"].rolling(14).max()
        df["stochastic_k"] = (df["close"] - low_14) / (high_14 - low_14 + 1e-10) * 100
        df["stochastic_d"] = df["stochastic_k"].rolling(3).mean()

        # MFI (Money Flow Index)
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        money_flow = typical_price * df["volume"]
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        df["mfi"] = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-10)))

        # Volume ratio and imbalance
        df["vol_ratio"] = df["volume"] / (df["volume_ma"] + 1e-10)
        df["vol_imbalance_20"] = df["volume"].rolling(20).apply(
            lambda x: (x[x > x.mean()].sum() - x[x <= x.mean()].sum()) / (x.sum() + 1e-10)
        )

        # Fill NaN values
        df = df.fillna(0)

        return df

    def _compute_labels(self) -> torch.Tensor:
        """Compute direction labels (1 for UP, 0 for DOWN)."""
        # Label is next period's direction based on return
        # Use return_1 shifted by -1 (look-ahead)
        return_idx = self.feature_cols.index("return_1") if "return_1" in self.feature_cols else 0

        # Get returns and shift to get future direction
        returns = self.data[:, return_idx].numpy()
        import numpy as np
        labels = np.zeros(len(returns))
        labels[:-1] = (returns[1:] > 0).astype(float)

        return torch.from_numpy(labels).float()

    def __len__(self) -> int:
        """Number of samples (accounting for sequence length)."""
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Get a training sample.

        Returns:
            sequence: [seq_len, n_features]
            trend_features: [n_trend]
            oscillator_features: [n_osc]
            volume_features: [n_vol]
            label: scalar (0 or 1)
            return_pct: scalar (actual return for reward)
        """
        # Sequence
        sequence = self.data[idx:idx + self.seq_len]

        # Current features (last in sequence)
        current = self.data[idx + self.seq_len - 1]

        # Extract head-specific features
        trend_indices = [self.feature_cols.index(f) for f in TREND_FEATURES if f in self.feature_cols]
        osc_indices = [self.feature_cols.index(f) for f in OSCILLATOR_FEATURES if f in self.feature_cols]
        vol_indices = [self.feature_cols.index(f) for f in VOLUME_FEATURES if f in self.feature_cols]

        trend_features = current[trend_indices] if trend_indices else torch.zeros(len(TREND_FEATURES))
        osc_features = current[osc_indices] if osc_indices else torch.zeros(len(OSCILLATOR_FEATURES))
        vol_features = current[vol_indices] if vol_indices else torch.zeros(len(VOLUME_FEATURES))

        # Pad if needed
        if len(trend_features) < len(TREND_FEATURES):
            trend_features = torch.cat([trend_features, torch.zeros(len(TREND_FEATURES) - len(trend_features))])
        if len(osc_features) < len(OSCILLATOR_FEATURES):
            osc_features = torch.cat([osc_features, torch.zeros(len(OSCILLATOR_FEATURES) - len(osc_features))])
        if len(vol_features) < len(VOLUME_FEATURES):
            vol_features = torch.cat([vol_features, torch.zeros(len(VOLUME_FEATURES) - len(vol_features))])

        # Label and return
        label = self.labels[idx + self.seq_len - 1]

        # Get return_pct for reward calculation
        return_idx = self.feature_cols.index("return_1") if "return_1" in self.feature_cols else 0
        return_pct = self.data[idx + self.seq_len, return_idx] if idx + self.seq_len < len(self.data) else 0.0

        return sequence, trend_features, osc_features, vol_features, label, return_pct


class TFTTrainer:
    """Trainer for Three-Head TFT Model.

    Supports:
    - Offline pretraining on historical data
    - Online fine-tuning with reward feedback
    - Mixed precision training
    - Gradient clipping
    - Early stopping
    - Checkpointing
    """

    def __init__(
        self,
        model: ThreeHeadTFT,
        config: ConfigStore,
        training_config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self._config = config
        self.training_config = training_config or TrainingConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model = self.model.to(self.device)

        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
        )

        # Loss function (cross-entropy for direction prediction)
        self.criterion = nn.CrossEntropyLoss()

        # Reward calculator for online learning
        self.reward_calculator = RewardCalculator(config)

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Create checkpoint directory
        os.makedirs(self.training_config.checkpoint_dir, exist_ok=True)

    def train_offline(
        self,
        train_dataset: TFTDataset,
        val_dataset: Optional[TFTDataset] = None,
    ) -> Dict[str, List[float]]:
        """Train model on historical data.

        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset

        Returns:
            Training history with losses and metrics
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.training_config.batch_size,
                shuffle=False,
                num_workers=0,
            )

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        print(f"Starting offline training on {self.device}")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset) if val_dataset else 0}")

        for epoch in range(self.training_config.epochs):
            # Training
            train_loss, train_acc = self._train_epoch(train_loader)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # Validation
            if val_loader:
                val_loss, val_acc = self._validate(val_loader)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                # Early stopping check
                if val_loss < self.best_val_loss - self.training_config.early_stopping_delta:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint("best.pt")
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.training_config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

                print(f"Epoch {epoch + 1}/{self.training_config.epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{self.training_config.epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f"epoch_{epoch + 1}.pt")

        # Save final checkpoint
        self._save_checkpoint("final.pt")

        return history

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            sequence, trend_feat, osc_feat, vol_feat, labels, _ = batch

            # Move to device
            sequence = sequence.to(self.device)
            trend_feat = trend_feat.to(self.device)
            osc_feat = osc_feat.to(self.device)
            vol_feat = vol_feat.to(self.device)
            labels = labels.to(self.device).long()

            # Forward pass
            self.optimizer.zero_grad()
            trend_probs, osc_probs, vol_probs, _ = self.model(
                sequence, trend_feat, osc_feat, vol_feat
            )

            # Combined loss from all heads
            loss = (
                self.criterion(trend_probs, labels) +
                self.criterion(osc_probs, labels) +
                self.criterion(vol_probs, labels)
            ) / 3

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.training_config.gradient_clip,
            )

            self.optimizer.step()
            self.scheduler.step()

            # Statistics
            total_loss += loss.item() * len(labels)

            # Accuracy (use ensemble of all heads)
            ensemble_probs = (trend_probs + osc_probs + vol_probs) / 3
            predictions = ensemble_probs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += len(labels)

            self.global_step += 1

        return total_loss / total, correct / total

    def _validate(self, loader: DataLoader) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                sequence, trend_feat, osc_feat, vol_feat, labels, _ = batch

                # Move to device
                sequence = sequence.to(self.device)
                trend_feat = trend_feat.to(self.device)
                osc_feat = osc_feat.to(self.device)
                vol_feat = vol_feat.to(self.device)
                labels = labels.to(self.device).long()

                # Forward pass
                trend_probs, osc_probs, vol_probs, _ = self.model(
                    sequence, trend_feat, osc_feat, vol_feat
                )

                # Combined loss
                loss = (
                    self.criterion(trend_probs, labels) +
                    self.criterion(osc_probs, labels) +
                    self.criterion(vol_probs, labels)
                ) / 3

                total_loss += loss.item() * len(labels)

                # Accuracy
                ensemble_probs = (trend_probs + osc_probs + vol_probs) / 3
                predictions = ensemble_probs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += len(labels)

        return total_loss / total, correct / total

    def train_online_step(
        self,
        sequence: torch.Tensor,
        trend_feat: torch.Tensor,
        osc_feat: torch.Tensor,
        vol_feat: torch.Tensor,
        actual_direction: str,
        return_pct: float,
    ) -> Dict[str, float]:
        """Online training step with reward feedback.

        Args:
            sequence: [1, seq_len, n_features]
            trend_feat: [1, n_trend]
            osc_feat: [1, n_osc]
            vol_feat: [1, n_vol]
            actual_direction: "UP" or "DOWN"
            return_pct: Actual return percentage

        Returns:
            Dict with loss, reward, accuracy info
        """
        self.model.train()

        # Forward pass
        trend_probs, osc_probs, vol_probs, _ = self.model(
            sequence.to(self.device),
            trend_feat.to(self.device),
            osc_feat.to(self.device),
            vol_feat.to(self.device),
        )

        # Get predictions
        ensemble_probs = (trend_probs + osc_probs + vol_probs) / 3
        predicted_idx = ensemble_probs.argmax(dim=1).item()
        predicted_direction = "UP" if predicted_idx == 0 else "DOWN"
        confidence = ensemble_probs[0, predicted_idx].item()

        # Calculate reward
        reward = self.reward_calculator.calculate(
            predicted_direction, actual_direction, return_pct, confidence
        )

        # Convert actual direction to label
        label = torch.tensor([0 if actual_direction == "UP" else 1], device=self.device)

        # Compute loss weighted by reward
        # Positive reward = lower loss weight (correct predictions)
        # Negative reward = higher loss weight (wrong predictions)
        loss_weight = max(0.5, 1.0 - reward * 10)  # Clamp to reasonable range

        loss = loss_weight * (
            self.criterion(trend_probs, label) +
            self.criterion(osc_probs, label) +
            self.criterion(vol_probs, label)
        ) / 3

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.training_config.gradient_clip,
        )
        self.optimizer.step()

        self.global_step += 1

        return {
            "loss": loss.item(),
            "reward": reward,
            "predicted": predicted_direction,
            "actual": actual_direction,
            "confidence": confidence,
            "hit": predicted_direction == actual_direction,
            "streak": self.reward_calculator.get_streak(),
        }

    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        path = os.path.join(self.training_config.checkpoint_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        path = os.path.join(self.training_config.checkpoint_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        print(f"Checkpoint loaded: {path}")

    def get_model_summary(self) -> Dict[str, object]:
        """Get model summary statistics."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "device": str(self.device),
        }
