"""
Transformer-based ensemble fusion with overfitting prevention.

Combines model outputs using self-attention mechanism with learned weights.
Implements multiple regularization techniques to prevent overfitting.

Sprint 21: TransformerFusion
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Try to import PyTorch, fall back to numpy-only implementation
HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except (ImportError, RuntimeError, Exception):
    pass

from titan.core.config import ConfigStore
from titan.core.types import ModelOutput


class TransformerFusion:
    """Transformer-based ensemble fusion with overfitting prevention.

    Combines outputs from multiple models using learned attention weights.
    Falls back to simple averaging when PyTorch is not available.

    Implements comprehensive regularization:
    - Dropout (0.1-0.3)
    - L2 regularization
    - Early stopping
    - Learning rate scheduling with warmup
    - Gradient clipping
    - Train/validation split monitoring

    Features:
    - Online learning (update after each prediction)
    - Mini-batch training
    - Automatic validation tracking
    - Attention weight analysis
    """

    def __init__(self, config: ConfigStore, n_models: int = 4):
        """Initialize transformer fusion layer.

        Args:
            config: Configuration store
            n_models: Number of models to fuse (default: 4)
        """
        self.config = config
        self.n_models = n_models

        # Configuration
        self.enabled = bool(config.get("fusion.enabled", True)) and HAS_TORCH
        self._hidden_dim = int(config.get("fusion.hidden_dim", 32))
        self._num_heads = int(config.get("fusion.num_heads", 2))
        self._dropout = float(config.get("fusion.dropout", 0.2))
        self._learning_rate = float(config.get("fusion.learning_rate", 0.001))
        self._l2_lambda = float(config.get("fusion.l2_lambda", 0.01))
        self._warmup_steps = int(config.get("fusion.warmup_steps", 100))
        self._min_samples = int(config.get("fusion.min_samples", 200))
        self._val_split = float(config.get("fusion.val_split", 0.2))
        self._gradient_clip = float(config.get("fusion.gradient_clip", 1.0))
        self._early_stopping_patience = int(config.get("fusion.early_stopping_patience", 50))
        self._early_stopping_delta = float(config.get("fusion.early_stopping_delta", 0.001))

        # Training state
        self._training_buffer: List[Dict] = []
        self._validation_buffer: List[Dict] = []
        self._train_losses: List[float] = []
        self._val_losses: List[float] = []
        self._best_val_loss = float('inf')
        self._patience_counter = 0
        self._is_overfitting = False
        self._step = 0
        self._last_attention: Optional[np.ndarray] = None

        # Statistics
        self._stats = {
            "samples_trained": 0,
            "val_samples": 0,
            "best_val_loss": float('inf'),
            "current_lr": self._learning_rate,
            "is_overfitting": False,
            "avg_attention_entropy": 0.0,
        }

        # Build network if PyTorch is available
        if self.enabled:
            self._build_network()
        else:
            # Numpy-based simple weights
            self._model_weights = np.ones(n_models) / n_models

    def _build_network(self):
        """Build the transformer network (PyTorch only)."""
        if not HAS_TORCH:
            return

        # Input projection: (prob_up, prob_down) -> hidden_dim
        self._input_proj = nn.Linear(2, self._hidden_dim)

        # Multi-head attention
        self._attention = nn.MultiheadAttention(
            embed_dim=self._hidden_dim,
            num_heads=self._num_heads,
            dropout=self._dropout,
            batch_first=True,
        )

        # Feed-forward network
        self._ffn = nn.Sequential(
            nn.Linear(self._hidden_dim, self._hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(self._dropout),
            nn.Linear(self._hidden_dim * 4, self._hidden_dim),
            nn.Dropout(self._dropout),
        )

        # Layer normalization
        self._norm1 = nn.LayerNorm(self._hidden_dim)
        self._norm2 = nn.LayerNorm(self._hidden_dim)

        # Output projection: hidden_dim -> 2 (prob_up, prob_down)
        self._output_proj = nn.Sequential(
            nn.Linear(self._hidden_dim, self._hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self._dropout),
            nn.Linear(self._hidden_dim // 2, 2),
        )

        # Optimizer with L2 regularization (weight decay)
        all_params = (
            list(self._input_proj.parameters()) +
            list(self._attention.parameters()) +
            list(self._ffn.parameters()) +
            list(self._norm1.parameters()) +
            list(self._norm2.parameters()) +
            list(self._output_proj.parameters())
        )
        self._optimizer = torch.optim.AdamW(
            all_params,
            lr=self._learning_rate,
            weight_decay=self._l2_lambda,
        )

        # Learning rate scheduler with warmup
        def lr_lambda(step):
            if step < self._warmup_steps:
                return float(step) / float(max(1, self._warmup_steps))
            else:
                progress = float(step - self._warmup_steps) / float(max(1, 10000 - self._warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        self._scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._optimizer,
            lr_lambda=lr_lambda,
        )

    def forward(
        self,
        model_outputs: List[ModelOutput],
        features: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float]:
        """Forward pass to combine model outputs.

        Args:
            model_outputs: List of ModelOutput from individual models
            features: Optional market features (not used yet, for future cross-attention)

        Returns:
            (prob_up, prob_down): Combined probabilities
        """
        if not self.enabled or len(model_outputs) == 0:
            return self._simple_average(model_outputs)

        if len(model_outputs) != self.n_models:
            return self._simple_average(model_outputs)

        # Don't use fusion until properly trained (needs min_samples on validation set)
        min_samples_for_inference = int(self.config.get("fusion.min_samples", 200))
        if self._stats["samples_trained"] < min_samples_for_inference:
            return self._simple_average(model_outputs)

        if not HAS_TORCH:
            return self._numpy_forward(model_outputs)

        # Extract probabilities
        probs = torch.tensor(
            [[out.prob_up, out.prob_down] for out in model_outputs],
            dtype=torch.float32,
        ).unsqueeze(0)  # (1, n_models, 2)

        # Set to eval mode for inference
        self._set_eval_mode()

        with torch.no_grad():
            # Project to hidden dimension
            x = self._input_proj(probs)  # (1, n_models, hidden_dim)

            # Self-attention with residual
            attn_out, attn_weights = self._attention(x, x, x, need_weights=True)
            x = self._norm1(x + attn_out)

            # FFN with residual
            ffn_out = self._ffn(x)
            x = self._norm2(x + ffn_out)

            # Global pooling (average over models)
            x = x.mean(dim=1)  # (1, hidden_dim)

            # Output projection
            output = self._output_proj(x)  # (1, 2)

            # Apply softmax to get probabilities
            probs_out = F.softmax(output, dim=-1).squeeze(0)  # (2,)

            prob_up = float(probs_out[0])
            prob_down = float(probs_out[1])

            # Store attention weights for analysis (may fail if numpy unavailable)
            try:
                self._last_attention = attn_weights.squeeze(0).mean(dim=0).cpu().numpy()
            except (RuntimeError, Exception):
                # NumPy conversion may fail with version mismatch
                self._last_attention = [float(x) for x in attn_weights.squeeze(0).mean(dim=0).cpu().tolist()]

        # Set back to train mode
        self._set_train_mode()

        return prob_up, prob_down

    def _numpy_forward(self, model_outputs: List[ModelOutput]) -> Tuple[float, float]:
        """Numpy-based forward pass (fallback when PyTorch unavailable)."""
        probs_up = np.array([out.prob_up for out in model_outputs])
        probs_down = np.array([out.prob_down for out in model_outputs])

        # Weighted average
        prob_up = float(np.sum(probs_up * self._model_weights))
        prob_down = float(np.sum(probs_down * self._model_weights))

        # Normalize
        total = prob_up + prob_down
        if total > 0:
            prob_up /= total
            prob_down /= total

        return prob_up, prob_down

    def _simple_average(self, model_outputs: List[ModelOutput]) -> Tuple[float, float]:
        """Fallback to simple averaging."""
        if len(model_outputs) == 0:
            return 0.5, 0.5

        avg_up = sum(out.prob_up for out in model_outputs) / len(model_outputs)
        avg_down = sum(out.prob_down for out in model_outputs) / len(model_outputs)

        total = avg_up + avg_down
        if total > 0:
            avg_up /= total
            avg_down /= total
        else:
            avg_up = 0.5
            avg_down = 0.5

        return avg_up, avg_down

    def _set_eval_mode(self):
        """Set network to evaluation mode."""
        if HAS_TORCH:
            self._input_proj.eval()
            self._attention.eval()
            self._ffn.eval()
            self._norm1.eval()
            self._norm2.eval()
            self._output_proj.eval()

    def _set_train_mode(self):
        """Set network to training mode."""
        if HAS_TORCH:
            self._input_proj.train()
            self._attention.train()
            self._ffn.train()
            self._norm1.train()
            self._norm2.train()
            self._output_proj.train()

    def update(
        self,
        model_outputs: List[ModelOutput],
        features: Dict[str, float],
        actual_direction: str,
    ):
        """Update the fusion network with new prediction outcome.

        Args:
            model_outputs: List of ModelOutput from individual models
            features: Market features
            actual_direction: Actual outcome ("UP" or "DOWN")
        """
        if not self.enabled:
            return

        if len(model_outputs) != self.n_models:
            return

        # Store sample in buffer
        sample = {
            "outputs": model_outputs,
            "features": features,
            "target": 1.0 if actual_direction == "UP" else 0.0,
        }

        # Split into train/validation
        if np.random.random() < self._val_split:
            self._validation_buffer.append(sample)
        else:
            self._training_buffer.append(sample)

        # Only start training after min_samples
        if len(self._training_buffer) < self._min_samples:
            return

        # Train on batch
        self._train_step()

        # Evaluate on validation set
        if len(self._validation_buffer) >= 10:
            self._validate()

        # Check for overfitting
        self._check_overfitting()

        self._step += 1

    def _train_step(self):
        """Perform one training step."""
        if not HAS_TORCH or len(self._training_buffer) == 0:
            return

        # Sample mini-batch
        batch_size = min(32, len(self._training_buffer))
        indices = np.random.choice(len(self._training_buffer), batch_size, replace=False)
        batch = [self._training_buffer[i] for i in indices]

        # Prepare batch tensors
        probs_batch = []
        targets_batch = []

        for sample in batch:
            probs = [[out.prob_up, out.prob_down] for out in sample["outputs"]]
            probs_batch.append(probs)
            targets_batch.append(sample["target"])

        probs_tensor = torch.tensor(probs_batch, dtype=torch.float32)
        targets_tensor = torch.tensor(targets_batch, dtype=torch.float32)

        # Forward pass
        self._optimizer.zero_grad()

        x = self._input_proj(probs_tensor)
        attn_out, _ = self._attention(x, x, x)
        x = self._norm1(x + attn_out)
        ffn_out = self._ffn(x)
        x = self._norm2(x + ffn_out)
        x = x.mean(dim=1)
        output = self._output_proj(x)

        # Loss
        log_probs = F.log_softmax(output, dim=-1)
        targets_onehot = torch.zeros_like(output)
        targets_onehot[:, 0] = targets_tensor
        targets_onehot[:, 1] = 1.0 - targets_tensor
        loss = F.kl_div(log_probs, targets_onehot, reduction='batchmean')

        # Backward pass
        loss.backward()

        # Gradient clipping
        all_params = (
            list(self._input_proj.parameters()) +
            list(self._attention.parameters()) +
            list(self._ffn.parameters()) +
            list(self._norm1.parameters()) +
            list(self._norm2.parameters()) +
            list(self._output_proj.parameters())
        )
        torch.nn.utils.clip_grad_norm_(all_params, self._gradient_clip)

        self._optimizer.step()
        self._scheduler.step()

        # Track loss
        self._train_losses.append(float(loss))
        self._stats["samples_trained"] += batch_size
        self._stats["current_lr"] = self._scheduler.get_last_lr()[0]

        # Keep buffer manageable
        if len(self._training_buffer) > 1000:
            self._training_buffer = self._training_buffer[-1000:]

    def _validate(self):
        """Evaluate on validation set."""
        if not HAS_TORCH or len(self._validation_buffer) == 0:
            return

        probs_batch = []
        targets_batch = []

        for sample in self._validation_buffer:
            probs = [[out.prob_up, out.prob_down] for out in sample["outputs"]]
            probs_batch.append(probs)
            targets_batch.append(sample["target"])

        probs_tensor = torch.tensor(probs_batch, dtype=torch.float32)
        targets_tensor = torch.tensor(targets_batch, dtype=torch.float32)

        with torch.no_grad():
            x = self._input_proj(probs_tensor)
            attn_out, _ = self._attention(x, x, x)
            x = self._norm1(x + attn_out)
            ffn_out = self._ffn(x)
            x = self._norm2(x + ffn_out)
            x = x.mean(dim=1)
            output = self._output_proj(x)

            log_probs = F.log_softmax(output, dim=-1)
            targets_onehot = torch.zeros_like(output)
            targets_onehot[:, 0] = targets_tensor
            targets_onehot[:, 1] = 1.0 - targets_tensor
            loss = F.kl_div(log_probs, targets_onehot, reduction='batchmean')

        val_loss = float(loss)
        self._val_losses.append(val_loss)
        self._stats["val_samples"] = len(self._validation_buffer)

        # Track best validation loss
        if val_loss < self._best_val_loss - self._early_stopping_delta:
            self._best_val_loss = val_loss
            self._patience_counter = 0
            self._stats["best_val_loss"] = val_loss
        else:
            self._patience_counter += 1

        if len(self._validation_buffer) > 200:
            self._validation_buffer = self._validation_buffer[-200:]

    def _check_overfitting(self):
        """Check for overfitting."""
        if len(self._train_losses) < 10 or len(self._val_losses) < 10:
            return

        recent_train = np.mean(self._train_losses[-10:])
        recent_val = np.mean(self._val_losses[-10:])

        if recent_val > recent_train * 1.2 and self._patience_counter >= self._early_stopping_patience:
            self._is_overfitting = True
            self._stats["is_overfitting"] = True
        else:
            self._is_overfitting = False
            self._stats["is_overfitting"] = False

    def get_attention_weights(self) -> Dict[str, float]:
        """Get current attention weights for each model."""
        if self._last_attention is None:
            return {f"model_{i}": 1.0 / self.n_models for i in range(self.n_models)}

        attn = self._last_attention.mean(axis=0)
        entropy = -np.sum(attn * np.log(attn + 1e-9))
        self._stats["avg_attention_entropy"] = float(entropy)

        return {f"model_{i}": float(attn[i]) for i in range(self.n_models)}

    def get_training_stats(self) -> Dict:
        """Get current training statistics."""
        stats = self._stats.copy()

        if len(self._train_losses) > 0:
            stats["train_loss"] = float(np.mean(self._train_losses[-10:]))
        else:
            stats["train_loss"] = None

        if len(self._val_losses) > 0:
            stats["val_loss"] = float(np.mean(self._val_losses[-10:]))
        else:
            stats["val_loss"] = None

        stats["patience_counter"] = self._patience_counter
        stats["step"] = self._step
        stats["has_torch"] = HAS_TORCH

        return stats

    def save(self, path: str):
        """Save model state to disk."""
        if not HAS_TORCH:
            return

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save({
            "input_proj": self._input_proj.state_dict(),
            "attention": self._attention.state_dict(),
            "ffn": self._ffn.state_dict(),
            "norm1": self._norm1.state_dict(),
            "norm2": self._norm2.state_dict(),
            "output_proj": self._output_proj.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict(),
        }, save_path / "fusion_model.pt")

        state = {
            "step": self._step,
            "best_val_loss": self._best_val_loss,
            "patience_counter": self._patience_counter,
            "is_overfitting": self._is_overfitting,
            "stats": self._stats,
            "train_losses": self._train_losses[-100:],
            "val_losses": self._val_losses[-100:],
        }

        with open(save_path / "fusion_state.json", "w") as f:
            json.dump(state, f, indent=2)

    def load(self, path: str):
        """Load model state from disk."""
        if not HAS_TORCH:
            return

        load_path = Path(path)

        checkpoint = torch.load(load_path / "fusion_model.pt")
        self._input_proj.load_state_dict(checkpoint["input_proj"])
        self._attention.load_state_dict(checkpoint["attention"])
        self._ffn.load_state_dict(checkpoint["ffn"])
        self._norm1.load_state_dict(checkpoint["norm1"])
        self._norm2.load_state_dict(checkpoint["norm2"])
        self._output_proj.load_state_dict(checkpoint["output_proj"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])
        self._scheduler.load_state_dict(checkpoint["scheduler"])

        with open(load_path / "fusion_state.json", "r") as f:
            state = json.load(f)

        self._step = state["step"]
        self._best_val_loss = state["best_val_loss"]
        self._patience_counter = state["patience_counter"]
        self._is_overfitting = state["is_overfitting"]
        self._stats = state["stats"]
        self._train_losses = state["train_losses"]
        self._val_losses = state["val_losses"]


def create_fusion(config: ConfigStore, n_models: int = 4) -> TransformerFusion:
    """Factory function to create TransformerFusion."""
    return TransformerFusion(config=config, n_models=n_models)
