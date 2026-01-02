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
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from titan.core.config import ConfigStore
from titan.core.models.tft import (
    ThreeHeadTFT,
    TwoHeadMLP,
    SessionEmbeddingMLP,
    SessionGatedMLP,
    SESSION_TO_IDX,
    TREND_FEATURES,
    OSCILLATOR_FEATURES,
    VOLUME_FEATURES,
)
from titan.core.features.calculator import (
    ALL_FEATURES,
    compute_and_normalize_batch,
)


@dataclass
class TrainingConfig:
    """Configuration for TFT training."""
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    seq_len: int = 100
    gradient_clip: float = 1.0
    # Gemini recommendation: Aggressive regularization to reduce 90%/54% gap
    weight_decay: float = 0.05  # Was 0.01, increased for better generalization
    dropout: float = 0.4  # New: High dropout to prevent overfitting on noisy crypto data
    warmup_steps: int = 100
    early_stopping_patience: int = 10
    early_stopping_delta: float = 0.001
    val_split: float = 0.2
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 100
    # Session-aware loss weighting (P1): Higher weight = more focus on that session
    # Default: equal weights. Set higher for worst-performing session (e.g., ASIA: 1.5)
    # 4 sessions aligned with patterns.py
    session_weights: Dict[str, float] = field(default_factory=lambda: {
        "ASIA": 1.0,
        "EUROPE": 1.0,
        "OVERLAP": 1.0,
        "US": 1.0,
    })


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
        label_horizon: int = 1,
    ):
        """Initialize dataset from candles CSV.

        Args:
            candles_path: Path to candles CSV file
            seq_len: Sequence length for model input
            feature_cols: Feature column names (default: ALL_FEATURES)
            label_horizon: Number of future bars for label (1 = next-bar, 3 = smoothed-3)
                          For 1m timeframe use 1, for aggregated/smoothed use higher.
                          Default: 1 (next-bar, no smoothing)
        """
        self.seq_len = seq_len
        self.feature_cols = feature_cols or ALL_FEATURES
        self.label_horizon = label_horizon

        # Load and preprocess data (also stores timestamps)
        self.data, self.timestamps = self._load_data(candles_path)
        self.labels = self._compute_labels(smoothing_window=label_horizon)

        # Validate
        if len(self.data) < seq_len + 1:
            raise ValueError(f"Not enough data: {len(self.data)} < {seq_len + 1}")

    def _load_data(self, path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load candles CSV and convert to tensor.

        Returns:
            Tuple of (features tensor, timestamps tensor)
        """
        import pandas as pd
        import numpy as np

        df = pd.read_csv(path)

        # Store timestamps before feature computation
        timestamps = df["timestamp"].values.copy() if "timestamp" in df.columns else np.arange(len(df))

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
        data = np.stack(feature_data, axis=1).astype(np.float32)

        return torch.from_numpy(data), torch.from_numpy(timestamps.astype(np.int64))

    def _compute_features(self, df) -> "pd.DataFrame":
        """Compute technical features from OHLCV.

        Uses unified calculator module to ensure train/inference consistency.
        """
        return compute_and_normalize_batch(df)

    def _compute_labels(self, smoothing_window: int = 1) -> torch.Tensor:
        """Compute direction labels (1 for UP, 0 for DOWN).

        For 1m timeframe: use smoothing_window=1 (next-bar only, no smoothing).
        For aggregated horizons: use smoothing_window=N to sum next N returns.

        Args:
            smoothing_window: Number of future returns to sum.
                             1 = next-bar (default for 1m timeframe)
                             3 = smoothed-3 (for aggregated predictions)
        """
        import numpy as np

        return_idx = self.feature_cols.index("return_1") if "return_1" in self.feature_cols else 0
        returns = self.data[:, return_idx].numpy()

        labels = np.zeros(len(returns))

        # Smoothed target: sum of next N returns
        # IMPORTANT: Backtest/live treats delta==0 as UP (see backtest._evaluate: delta >= 0 -> UP).
        # To keep training/evaluation labels consistent, treat sum==0 as UP as well.
        for i in range(len(returns) - smoothing_window):
            future_returns = returns[i + 1:i + 1 + smoothing_window]
            labels[i] = 1.0 if np.sum(future_returns) >= 0 else 0.0

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
            timestamp: int64 (Unix timestamp for session detection)
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

        # Get timestamp for session detection
        timestamp = self.timestamps[idx + self.seq_len - 1]

        return sequence, trend_features, osc_features, vol_features, label, return_pct, timestamp

    @staticmethod
    def get_session(timestamp: int) -> str:
        """Get trading session from Unix timestamp.

        Sessions (UTC) - aligned with patterns.py:get_trading_session:
        - ASIA: 00:00 - 08:00 and 22:00 - 24:00 (Tokyo, Hong Kong, Singapore)
        - EUROPE: 08:00 - 13:00 (London, Frankfurt)
        - OVERLAP: 13:00 - 17:00 (London + New York overlap)
        - US: 17:00 - 22:00 (New York)
        """
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        hour = dt.hour
        if 0 <= hour < 8:
            return "ASIA"
        elif 8 <= hour < 13:
            return "EUROPE"
        elif 13 <= hour < 17:
            return "OVERLAP"
        elif 17 <= hour < 22:
            return "US"
        else:  # 22-23
            return "ASIA"


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
        """Train for one epoch with session-aware loss weighting (P1)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # Per-sample criterion for session weighting
        criterion_none = nn.CrossEntropyLoss(reduction='none')

        for batch in loader:
            sequence, trend_feat, osc_feat, vol_feat, labels, _, timestamps = batch

            # Move to device
            sequence = sequence.to(self.device)
            trend_feat = trend_feat.to(self.device)
            osc_feat = osc_feat.to(self.device)
            vol_feat = vol_feat.to(self.device)
            labels = labels.to(self.device).long()

            # Compute per-sample session info (for P1 weighting and P2 gating)
            session_weights = torch.ones(len(labels), device=self.device)
            session_idx = torch.zeros(len(labels), dtype=torch.long, device=self.device)
            for i, ts in enumerate(timestamps):
                session = TFTDataset.get_session(ts.item())
                session_weights[i] = self.training_config.session_weights.get(session, 1.0)
                session_idx[i] = SESSION_TO_IDX.get(session, 0)

            # Forward pass with logits for training
            self.optimizer.zero_grad()
            if isinstance(self.model, (SessionGatedMLP, SessionEmbeddingMLP)):
                # P2/P2.5: Pass session_idx to session-aware models
                trend_probs, osc_probs, vol_probs, aux = self.model(
                    sequence, trend_feat, osc_feat, vol_feat, session_idx, return_logits=True
                )
            else:
                trend_probs, osc_probs, vol_probs, aux = self.model(
                    sequence, trend_feat, osc_feat, vol_feat, return_logits=True
                )

            # Use LOGITS for CrossEntropyLoss (not probs!)
            trend_logits = aux["trend_logits"]
            osc_logits = aux["osc_logits"]
            vol_logits = aux["vol_logits"]

            # Combined loss from all heads using logits with session weighting
            # For TwoHeadMLP/SessionGatedMLP: osc_logits == vol_logits (same aux head)
            # Use 1/2 trend + 1/2 aux to avoid double-counting aux
            if isinstance(self.model, (TwoHeadMLP, SessionGatedMLP, SessionEmbeddingMLP)):
                per_sample_loss = (
                    criterion_none(trend_logits, labels) +
                    criterion_none(osc_logits, labels)
                ) / 2
            else:
                per_sample_loss = (
                    criterion_none(trend_logits, labels) +
                    criterion_none(osc_logits, labels) +
                    criterion_none(vol_logits, labels)
                ) / 3

            # Apply session weights and mean (weighted average)
            weighted_loss = per_sample_loss * session_weights
            loss = weighted_loss.mean()

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

            # Accuracy (TwoHeadMLP/SessionGatedMLP: 1/2 trend + 1/2 aux)
            if isinstance(self.model, (TwoHeadMLP, SessionGatedMLP, SessionEmbeddingMLP)):
                ensemble_probs = (trend_probs + osc_probs) / 2
            else:
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
        is_two_head = isinstance(self.model, (TwoHeadMLP, SessionGatedMLP, SessionEmbeddingMLP))
        is_session_gated = isinstance(self.model, (SessionGatedMLP, SessionEmbeddingMLP))

        with torch.no_grad():
            for batch in loader:
                sequence, trend_feat, osc_feat, vol_feat, labels, _, timestamps = batch

                # Move to device
                sequence = sequence.to(self.device)
                trend_feat = trend_feat.to(self.device)
                osc_feat = osc_feat.to(self.device)
                vol_feat = vol_feat.to(self.device)
                labels = labels.to(self.device).long()

                # Forward pass with logits for loss calculation
                if is_session_gated:
                    # Compute session_idx for P2 session gating
                    session_idx = torch.zeros(len(labels), dtype=torch.long, device=self.device)
                    for i, ts in enumerate(timestamps):
                        session = TFTDataset.get_session(ts.item())
                        session_idx[i] = SESSION_TO_IDX.get(session, 0)
                    trend_probs, osc_probs, vol_probs, aux = self.model(
                        sequence, trend_feat, osc_feat, vol_feat, session_idx, return_logits=True
                    )
                else:
                    trend_probs, osc_probs, vol_probs, aux = self.model(
                        sequence, trend_feat, osc_feat, vol_feat, return_logits=True
                    )

                # Use LOGITS for loss
                trend_logits = aux["trend_logits"]
                osc_logits = aux["osc_logits"]
                vol_logits = aux["vol_logits"]

                # Combined loss using logits (TwoHeadMLP/SessionGatedMLP: 1/2 + 1/2)
                if is_two_head:
                    loss = (
                        self.criterion(trend_logits, labels) +
                        self.criterion(osc_logits, labels)
                    ) / 2
                else:
                    loss = (
                        self.criterion(trend_logits, labels) +
                        self.criterion(osc_logits, labels) +
                        self.criterion(vol_logits, labels)
                    ) / 3

                total_loss += loss.item() * len(labels)

                # Accuracy (TwoHeadMLP/SessionGatedMLP: 1/2 trend + 1/2 aux)
                if is_two_head:
                    ensemble_probs = (trend_probs + osc_probs) / 2
                else:
                    ensemble_probs = (trend_probs + osc_probs + vol_probs) / 3
                predictions = ensemble_probs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += len(labels)

        return total_loss / total, correct / total

    def validate_with_session_breakdown(
        self, loader: DataLoader
    ) -> Dict[str, Dict[str, float]]:
        """Validate model with per-session accuracy breakdown.

        Returns:
            Dict with overall and per-session stats:
            {
                "overall": {"loss": 0.5, "accuracy": 0.54, "total": 1000},
                "ASIA": {"accuracy": 0.55, "total": 350, "correct": 192},
                "EUROPE": {"accuracy": 0.52, "total": 300, "correct": 156},
                "US": {"accuracy": 0.54, "total": 350, "correct": 189},
            }
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        is_two_head = isinstance(self.model, (TwoHeadMLP, SessionGatedMLP, SessionEmbeddingMLP))
        is_session_gated = isinstance(self.model, (SessionGatedMLP, SessionEmbeddingMLP))

        # Per-session tracking
        # 4 sessions aligned with patterns.py
        session_stats = {
            "ASIA": {"correct": 0, "total": 0},
            "EUROPE": {"correct": 0, "total": 0},
            "OVERLAP": {"correct": 0, "total": 0},
            "US": {"correct": 0, "total": 0},
        }

        with torch.no_grad():
            for batch in loader:
                sequence, trend_feat, osc_feat, vol_feat, labels, _, timestamps = batch

                # Move to device
                sequence = sequence.to(self.device)
                trend_feat = trend_feat.to(self.device)
                osc_feat = osc_feat.to(self.device)
                vol_feat = vol_feat.to(self.device)
                labels = labels.to(self.device).long()

                # Compute session info for all samples
                sessions = [TFTDataset.get_session(ts.item()) for ts in timestamps]
                session_idx = torch.tensor(
                    [SESSION_TO_IDX.get(s, 0) for s in sessions],
                    dtype=torch.long, device=self.device
                )

                # Forward pass with logits for loss calculation
                if is_session_gated:
                    trend_probs, osc_probs, vol_probs, aux = self.model(
                        sequence, trend_feat, osc_feat, vol_feat, session_idx, return_logits=True
                    )
                else:
                    trend_probs, osc_probs, vol_probs, aux = self.model(
                        sequence, trend_feat, osc_feat, vol_feat, return_logits=True
                    )

                # Use LOGITS for loss
                trend_logits = aux["trend_logits"]
                osc_logits = aux["osc_logits"]
                vol_logits = aux["vol_logits"]

                # Combined loss using logits (TwoHeadMLP/SessionGatedMLP: 1/2 + 1/2)
                if is_two_head:
                    loss = (
                        self.criterion(trend_logits, labels) +
                        self.criterion(osc_logits, labels)
                    ) / 2
                else:
                    loss = (
                        self.criterion(trend_logits, labels) +
                        self.criterion(osc_logits, labels) +
                        self.criterion(vol_logits, labels)
                    ) / 3

                total_loss += loss.item() * len(labels)

                # Accuracy (TwoHeadMLP/SessionGatedMLP: 1/2 trend + 1/2 aux)
                if is_two_head:
                    ensemble_probs = (trend_probs + osc_probs) / 2
                else:
                    ensemble_probs = (trend_probs + osc_probs + vol_probs) / 3
                predictions = ensemble_probs.argmax(dim=1)
                hits = (predictions == labels).cpu()
                correct += hits.sum().item()
                total += len(labels)

                # Per-session accuracy
                for i, session in enumerate(sessions):
                    session_stats[session]["total"] += 1
                    if hits[i].item():
                        session_stats[session]["correct"] += 1

        # Build result
        result = {
            "overall": {
                "loss": total_loss / total if total > 0 else 0,
                "accuracy": correct / total if total > 0 else 0,
                "total": total,
                "correct": correct,
            }
        }

        for session, stats in session_stats.items():
            if stats["total"] > 0:
                result[session] = {
                    "accuracy": stats["correct"] / stats["total"],
                    "total": stats["total"],
                    "correct": stats["correct"],
                }
            else:
                result[session] = {"accuracy": 0.0, "total": 0, "correct": 0}

        return result

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

        # Forward pass with logits for loss
        trend_probs, osc_probs, vol_probs, aux = self.model(
            sequence.to(self.device),
            trend_feat.to(self.device),
            osc_feat.to(self.device),
            vol_feat.to(self.device),
            return_logits=True,
        )

        # Get predictions - consistent with offline: idx 1 = UP (return > 0)
        ensemble_probs = (trend_probs + osc_probs + vol_probs) / 3
        predicted_idx = ensemble_probs.argmax(dim=1).item()
        predicted_direction = "UP" if predicted_idx == 1 else "DOWN"
        confidence = ensemble_probs[0, predicted_idx].item()

        # Calculate reward
        reward = self.reward_calculator.calculate(
            predicted_direction, actual_direction, return_pct, confidence
        )

        # Convert actual direction to label - CONSISTENT with offline: 1 = UP, 0 = DOWN
        label = torch.tensor([1 if actual_direction == "UP" else 0], device=self.device)

        # Get logits for loss
        trend_logits = aux["trend_logits"]
        osc_logits = aux["osc_logits"]
        vol_logits = aux["vol_logits"]

        # Compute loss weighted by reward using LOGITS
        # Positive reward = lower loss weight (correct predictions)
        # Negative reward = higher loss weight (wrong predictions)
        loss_weight = max(0.5, 1.0 - reward * 10)  # Clamp to reasonable range

        loss = loss_weight * (
            self.criterion(trend_logits, label) +
            self.criterion(osc_logits, label) +
            self.criterion(vol_logits, label)
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
