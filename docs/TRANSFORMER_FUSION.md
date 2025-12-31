# TransformerFusion - Ensemble Learning with Overfitting Prevention

## Overview

`TransformerFusion` is a PyTorch-based neural ensemble fusion system that combines outputs from multiple models using a self-attention mechanism. It implements comprehensive regularization techniques to prevent overfitting while learning optimal model combinations online.

**Sprint 21** - Created 2025-12-30

---

## Architecture

### Network Structure

```
Input: List[ModelOutput]
  │
  ├─> Extract (prob_up, prob_down) → (batch, n_models, 2)
  │
  ├─> Input Projection → (batch, n_models, hidden_dim)
  │
  ├─> TransformerFusionLayer
  │   ├─> Multi-head Self-Attention (num_heads=2)
  │   ├─> Layer Normalization + Residual
  │   ├─> Feed-Forward Network (hidden_dim → 4*hidden_dim → hidden_dim)
  │   └─> Layer Normalization + Residual
  │
  ├─> Global Pooling (average over models) → (batch, hidden_dim)
  │
  ├─> Output Projection → (batch, 2)
  │
  └─> Softmax → (prob_up, prob_down)
```

### Key Components

1. **Input Projection**: Linear layer (2 → hidden_dim)
   - Projects model probabilities to higher-dimensional space
   - Enables richer feature representation

2. **TransformerFusionLayer**: Self-attention + FFN
   - Multi-head attention learns relationships between models
   - FFN adds non-linear transformation
   - Residual connections prevent gradient degradation
   - Layer normalization for stable training

3. **Output Projection**: MLP (hidden_dim → hidden_dim/2 → 2)
   - ReLU activation for non-linearity
   - Dropout for regularization
   - Final layer outputs logits for UP/DOWN

---

## Overfitting Prevention Mechanisms

### 1. Dropout (0.1-0.3)
- Applied in attention layers, FFN, and output projection
- Randomly zeros elements during training
- Forces network to learn robust features
- **Config**: `fusion.dropout`

### 2. L2 Regularization (Weight Decay)
- Penalizes large weights via AdamW optimizer
- Prevents overfitting to training data
- **Config**: `fusion.l2_lambda`

### 3. Early Stopping
- Monitors validation loss divergence from training loss
- Stops training if validation loss doesn't improve for N steps
- **Config**: `fusion.early_stopping_patience`, `fusion.early_stopping_delta`

### 4. Learning Rate Scheduling
- **Warmup phase** (0 → warmup_steps): Linear increase
  - Prevents large gradient updates early in training
  - Stabilizes initial training
- **Decay phase** (after warmup): Cosine annealing
  - Gradually reduces LR for fine-tuning
- **Config**: `fusion.warmup_steps`, `fusion.learning_rate`

### 5. Gradient Clipping
- Limits gradient norm to prevent exploding gradients
- Essential for stable training with recurrent/attention networks
- **Config**: `fusion.gradient_clip`

### 6. Train/Validation Split
- Automatic 80/20 split of incoming samples
- Independent validation set for unbiased performance estimation
- **Config**: `fusion.val_split`

### 7. Overfitting Detection
- Monitors ratio: `val_loss / train_loss`
- Triggers warning if ratio > 1.2 for extended period
- Sets `is_overfitting` flag in stats

---

## Configuration Parameters

```python
# In titan/core/config.py DEFAULT_CONFIG

"fusion.enabled": True,                     # Enable/disable fusion
"fusion.hidden_dim": 32,                   # Hidden dimension size
"fusion.num_heads": 2,                     # Number of attention heads
"fusion.dropout": 0.2,                     # Dropout rate (0.0-0.5)
"fusion.learning_rate": 0.001,             # Initial learning rate
"fusion.l2_lambda": 0.01,                  # L2 regularization strength
"fusion.warmup_steps": 100,                # LR warmup duration
"fusion.min_samples": 200,                 # Min samples before training
"fusion.val_split": 0.2,                   # Validation split ratio
"fusion.gradient_clip": 1.0,               # Max gradient norm
"fusion.early_stopping_patience": 50,      # Early stopping patience
"fusion.early_stopping_delta": 0.001,      # Min improvement threshold
```

### Tuning Recommendations

**For Higher Accuracy (Risk: Overfitting)**
- Increase `hidden_dim`: 32 → 64 → 128
- Increase `num_heads`: 2 → 4
- Decrease `dropout`: 0.2 → 0.1
- Decrease `l2_lambda`: 0.01 → 0.005

**For Better Generalization (Risk: Underfitting)**
- Increase `dropout`: 0.2 → 0.3
- Increase `l2_lambda`: 0.01 → 0.02
- Decrease `hidden_dim`: 32 → 16
- Increase `early_stopping_patience`

**For Faster Training**
- Increase `learning_rate`: 0.001 → 0.003
- Decrease `warmup_steps`: 100 → 50
- Decrease `min_samples`: 200 → 100

**For More Stable Training**
- Increase `warmup_steps`: 100 → 200
- Decrease `learning_rate`: 0.001 → 0.0005
- Increase `gradient_clip`: 1.0 → 0.5

---

## Usage

### Basic Integration

```python
from titan.core.config import ConfigStore
from titan.core.fusion import TransformerFusion
from titan.core.state_store import StateStore

# Initialize
state = StateStore("titan.db")
config = ConfigStore(state)
config.ensure_defaults()

fusion = TransformerFusion(config, n_models=4)

# Forward pass (combine model outputs)
model_outputs = [trendvic_out, oscillator_out, volumemetrix_out, ml_out]
features = {"close": 50000.0, "rsi": 55.0, ...}

prob_up, prob_down = fusion.forward(model_outputs, features)

# Update after knowing actual outcome
fusion.update(model_outputs, features, actual_direction="UP")
```

### Ensemble Integration

```python
class Ensemble:
    def __init__(self, config):
        self.fusion = TransformerFusion(config, n_models=4)
        # ... other components

    def decide(self, outputs, features, ...):
        # Option 1: Use fusion directly
        if self.fusion.enabled:
            prob_up, prob_down = self.fusion.forward(outputs, features)
        else:
            # Fallback to traditional weighted average
            prob_up, prob_down = self._weighted_combine(outputs)

        # ... rest of decision logic

        return Decision(...)

    def record_outcome(self, outputs, features, actual):
        # Train fusion online
        self.fusion.update(outputs, features, actual)
```

### Monitoring

```python
# Get training statistics
stats = fusion.get_training_stats()

print(f"Samples trained: {stats['samples_trained']}")
print(f"Train loss: {stats['train_loss']}")
print(f"Val loss: {stats['val_loss']}")
print(f"Learning rate: {stats['current_lr']}")
print(f"Overfitting: {stats['is_overfitting']}")

# Get attention weights (model importance)
weights = fusion.get_attention_weights()
# {'model_0': 0.28, 'model_1': 0.24, 'model_2': 0.22, 'model_3': 0.26}

# Attention entropy (diversity measure)
print(f"Entropy: {stats['avg_attention_entropy']:.2f}")
# Low entropy (< 1.0): Focuses on few models
# High entropy (> 1.3): Uses all models equally
```

### Save/Load

```python
# Save trained model
fusion.save("models/fusion_checkpoint")

# Load later
fusion2 = TransformerFusion(config, n_models=4)
fusion2.load("models/fusion_checkpoint")
```

---

## Online Learning Strategy

### Buffer Management

1. **Incoming samples** are randomly split:
   - 80% → Training buffer
   - 20% → Validation buffer

2. **Training**:
   - Triggered when training buffer ≥ `min_samples`
   - Samples mini-batch (max 32) randomly
   - Performs gradient descent step
   - Updates learning rate via scheduler

3. **Validation**:
   - Evaluated every 10+ validation samples
   - Tracks validation loss
   - Updates early stopping counter

4. **Buffer limits**:
   - Training buffer: Max 1000 samples (FIFO)
   - Validation buffer: Max 200 samples (FIFO)

### Loss Function

Binary cross-entropy via KL divergence:

```
targets_onehot = [1.0, 0.0] for UP, [0.0, 1.0] for DOWN
output = model(inputs)  # Logits
log_probs = log_softmax(output)
loss = KL_div(log_probs, targets_onehot)
```

This is equivalent to cross-entropy but numerically stable.

---

## Attention Mechanism

### How It Works

The self-attention layer learns which models to trust:

```
Q, K, V = Linear(x), Linear(x), Linear(x)
Attention(Q, K, V) = softmax(QK^T / √d) × V
```

- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What information do I have?"
- **Value (V)**: "The actual information"

**Result**: Model learns to attend more to models that:
- Have historically better accuracy
- Complement each other (diversity)
- Are appropriate for current market conditions

### Interpreting Attention Weights

```python
weights = fusion.get_attention_weights()
# {'model_0': 0.35, 'model_1': 0.15, 'model_2': 0.30, 'model_3': 0.20}
```

- **High weight (> 0.30)**: Model is highly trusted
- **Low weight (< 0.20)**: Model is less influential
- **Balanced weights (0.25 ± 0.05)**: All models contribute equally

**Entropy** measures concentration:
- `H = -Σ p_i log(p_i)`
- **Uniform distribution**: H = log(4) ≈ 1.39
- **Concentrated**: H < 1.0 (focuses on 1-2 models)
- **Diverse**: H > 1.2 (uses all models)

---

## Expected Performance

### Advantages Over Simple Averaging

1. **Learned Weights**: Adapts to model performance over time
2. **Non-Linear Combinations**: Captures complex interactions
3. **Contextual**: Can weight models differently in different market regimes
4. **Continual Learning**: Improves as more data arrives

### Limitations

1. **Requires Data**: Needs 200+ samples before effective
2. **Computational Cost**: ~10x slower than weighted average
3. **Overfitting Risk**: Requires careful regularization
4. **Black Box**: Harder to interpret than linear weights

### When to Use

**Use TransformerFusion if:**
- You have sufficient training data (>500 samples)
- Models have complex interdependencies
- You want automatic adaptation
- You can afford the computational cost

**Use Simple Weighted Ensemble if:**
- Limited data (< 200 samples)
- Need interpretability
- Computational constraints
- Models are largely independent

---

## Troubleshooting

### Overfitting Detected

**Symptoms**: `is_overfitting=True`, `val_loss >> train_loss`

**Solutions**:
1. Increase `dropout`: 0.2 → 0.3
2. Increase `l2_lambda`: 0.01 → 0.02
3. Decrease `hidden_dim`: 32 → 16
4. Increase early stopping patience

### Training Not Starting

**Symptoms**: `samples_trained=0` after many predictions

**Solutions**:
1. Check `min_samples` is not too high
2. Verify `fusion.enabled=True`
3. Check training buffer is filling (80% of samples)

### Poor Performance

**Symptoms**: Fusion worse than simple average

**Solutions**:
1. Increase `min_samples` (need more data)
2. Check for overfitting (val_loss > train_loss)
3. Reduce model complexity (smaller `hidden_dim`)
4. Verify input models have reasonable quality

### Training Loss Not Decreasing

**Symptoms**: `train_loss` stays constant or increases

**Solutions**:
1. Increase `learning_rate`: 0.001 → 0.003
2. Decrease `warmup_steps`: 100 → 50
3. Check `gradient_clip` is not too aggressive
4. Verify samples have both UP and DOWN outcomes

---

## Testing

Run the comprehensive test suite:

```bash
cd C:\Projects\Titan
python test_fusion.py
```

**Tests**:
1. Forward pass and probability validation
2. Online learning and buffer management
3. Attention weight extraction
4. Overfitting detection mechanism
5. Save/load functionality
6. Regularization features (dropout, LR scheduling)
7. Disabled mode fallback

**Expected output**: All 7 tests should pass.

---

## Future Enhancements

1. **Cross-Attention with Features**
   - Attend to market features (volatility, volume, etc.)
   - Condition model weights on market regime

2. **Multi-Task Learning**
   - Predict both direction AND magnitude
   - Auxiliary tasks: regime classification, confidence calibration

3. **Meta-Learning**
   - Learn to learn: adapt faster to new patterns
   - Few-shot learning for rare market events

4. **Ensemble Diversity Reward**
   - Penalize models that provide redundant information
   - Encourage complementary predictions

5. **Adaptive Architecture**
   - Dynamically adjust `hidden_dim` based on performance
   - Neural architecture search for optimal structure

---

## References

- **Attention Mechanism**: "Attention Is All You Need" (Vaswani et al., 2017)
- **Dropout**: "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (Srivastava et al., 2014)
- **AdamW**: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
- **LR Warmup**: "Accurate, Large Minibatch SGD" (Goyal et al., 2017)

---

## Summary

TransformerFusion provides a sophisticated, production-ready ensemble learning system with:

- **Self-attention** for learned model combinations
- **Comprehensive regularization** to prevent overfitting
- **Online learning** for continuous adaptation
- **Monitoring tools** for performance tracking
- **Graceful fallback** when disabled or data-starved

Use it when you need adaptive, high-performance ensemble fusion with robust overfitting prevention.
