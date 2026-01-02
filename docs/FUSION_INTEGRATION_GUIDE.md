# TransformerFusion Integration Guide

Quick guide for integrating `TransformerFusion` into the Titan ensemble system.

---

## Step 1: Update Ensemble Class

**File**: `titan/core/ensemble.py`

### Add Import

```python
from titan.core.fusion import TransformerFusion
```

### Initialize in __init__

```python
class Ensemble:
    def __init__(
        self,
        config: ConfigStore,
        models: List[BaseModel],
        regime_detector: RegimeDetector,
        weight_manager: AdaptiveWeightManager,
    ):
        self._config = config
        self._models = models
        self._regime = regime_detector
        self._weights = weight_manager

        # NEW: Initialize TransformerFusion
        self._fusion = TransformerFusion(config, n_models=len(models))

        # ... rest of initialization
```

### Modify decide() Method

```python
def decide(
    self,
    features: Dict[str, float],
    timestamp: Optional[int] = None,
    pattern_id: Optional[int] = None,
) -> Decision:
    """Make ensemble decision with optional fusion."""

    # 1. Get model outputs (existing code)
    outputs = [model.predict(features, pattern_context) for model in self._models]

    # 2. Option A: Use TransformerFusion if enabled
    if self._config.get("fusion.enabled", False):
        # Check if fusion is ready (has enough training data)
        fusion_stats = self._fusion.get_training_stats()

        if fusion_stats['samples_trained'] >= self._config.get("fusion.min_samples", 200):
            # Use fusion for combining probabilities
            prob_up, prob_down = self._fusion.forward(outputs, features)
        else:
            # Fallback to weighted average while fusion is warming up
            prob_up, prob_down = self._weighted_combine(outputs, regime)
    else:
        # Option B: Traditional weighted combination (existing code)
        prob_up, prob_down = self._weighted_combine(outputs, regime)

    # 3. Rest of decision logic (existing code)
    # - Apply calibration
    # - Apply temporal adjustment
    # - Apply pattern adjustment
    # - Build Decision object

    return Decision(direction, confidence, prob_up, prob_down)
```

### Add record_outcome() Method

```python
def record_outcome(
    self,
    outputs: List[ModelOutput],
    features: Dict[str, float],
    actual_direction: str,
):
    """Record outcome for fusion online learning."""

    if self._config.get("fusion.enabled", False):
        self._fusion.update(outputs, features, actual_direction)
```

---

## Step 2: Update Backtest

**File**: `titan/core/backtest.py`

### Call record_outcome After Evaluation

```python
def run_backtest(...):
    # ... existing code ...

    for i in range(warmup_size, len(candles)):
        # ... prediction code ...

        # Get decision
        decision = ensemble.decide(features, timestamp, pattern_id)

        # Evaluate
        actual_direction = "UP" if actual_return > 0 else "DOWN"

        # NEW: Record outcome for fusion learning
        ensemble.record_outcome(outputs, features, actual_direction)

        # ... rest of backtest logic ...
```

### Add Fusion Stats to Report

```python
# At end of backtest, add fusion statistics
if config.get("fusion.enabled", False):
    fusion_stats = ensemble._fusion.get_training_stats()
    attention_weights = ensemble._fusion.get_attention_weights()

    report_sections.append({
        "name": "Transformer Fusion",
        "stats": {
            "samples_trained": fusion_stats['samples_trained'],
            "train_loss": fusion_stats.get('train_loss'),
            "val_loss": fusion_stats.get('val_loss'),
            "current_lr": fusion_stats['current_lr'],
            "is_overfitting": fusion_stats['is_overfitting'],
            "attention_weights": attention_weights,
            "attention_entropy": fusion_stats['avg_attention_entropy'],
        }
    })
```

---

## Step 3: Update Live Mode

**File**: `titan/core/live.py`

### Similar Changes as Backtest

```python
def run_live(...):
    # ... existing code ...

    while True:
        # Get prediction
        decision = ensemble.decide(features, timestamp, pattern_id)

        # Wait for next candle
        time.sleep(60)

        # Get actual outcome
        actual_direction = "UP" if next_candle.close > current_candle.close else "DOWN"

        # NEW: Update fusion
        ensemble.record_outcome(outputs, features, actual_direction)
```

---

## Step 4: Configuration

### Recommended Starting Configuration

For **24-hour backtests** (~1440 predictions):

```python
"fusion.enabled": False,  # Start disabled, enable after verifying baseline
"fusion.hidden_dim": 32,
"fusion.num_heads": 2,
"fusion.dropout": 0.2,
"fusion.learning_rate": 0.001,
"fusion.l2_lambda": 0.01,
"fusion.warmup_steps": 100,
"fusion.min_samples": 200,
"fusion.val_split": 0.2,
```

For **7-day backtests** (~10,000 predictions):

```python
"fusion.enabled": True,
"fusion.hidden_dim": 64,  # Can afford larger model
"fusion.num_heads": 4,
"fusion.dropout": 0.2,
"fusion.learning_rate": 0.001,
"fusion.l2_lambda": 0.01,
"fusion.warmup_steps": 200,
"fusion.min_samples": 500,  # Need more samples for larger model
"fusion.val_split": 0.2,
```

---

## Step 5: Testing Strategy

### Phase 1: Baseline (fusion.enabled=False)

```bash
python -m titan.cli history --symbol BTCUSDT --interval 1 --hours 24
```

Record baseline metrics:
- Ensemble accuracy
- Per-model accuracy
- ECE
- Sharpe ratio

### Phase 2: Enable Fusion (fusion.enabled=True)

```bash
python -m titan.cli history --symbol BTCUSDT --interval 1 --hours 24
```

Compare:
- Did ensemble accuracy improve?
- Check fusion stats: `train_loss`, `val_loss`, `is_overfitting`
- Check attention weights: Which models are trusted?

### Phase 3: Optimize

If fusion helps:
- Try larger `hidden_dim`: 32 → 64
- Try more heads: 2 → 4
- Run 7-day backtest for more data

If fusion hurts:
- Check for overfitting: `val_loss > train_loss`
- Increase regularization: `dropout=0.3`, `l2_lambda=0.02`
- Decrease complexity: `hidden_dim=16`

---

## Step 6: Monitoring

### In Reports

Add section to `report.md`:

```markdown
## Transformer Fusion

**Status**: Enabled
**Samples Trained**: 1152
**Train Loss**: 0.6107
**Val Loss**: 0.6009
**Overfitting**: No
**Learning Rate**: 0.000360

### Attention Weights
- TrendVIC: 0.28
- Oscillator: 0.24
- VolumeMetrix: 0.26
- ML_Classifier: 0.22

**Entropy**: 1.38 (balanced)
```

### Interpretation

**Good Signs**:
- `val_loss ≈ train_loss` (not overfitting)
- `attention_entropy > 1.2` (using all models)
- Ensemble accuracy > simple average

**Bad Signs**:
- `val_loss >> train_loss` (overfitting!)
- `attention_entropy < 0.8` (only using 1 model)
- Ensemble accuracy < simple average

---

## Step 7: Save/Load Trained Fusion

### Save After Good Backtest

```python
# At end of backtest
if config.get("fusion.enabled") and fusion_stats['samples_trained'] > 500:
    save_path = f"models/fusion_{symbol}_{interval}m"
    ensemble._fusion.save(save_path)
    print(f"Saved fusion model to {save_path}")
```

### Load Before Live Trading

```python
# At start of live mode
if config.get("fusion.enabled"):
    try:
        load_path = f"models/fusion_{symbol}_{interval}m"
        ensemble._fusion.load(load_path)
        print(f"Loaded fusion model from {load_path}")
    except FileNotFoundError:
        print("No saved fusion model, starting fresh")
```

---

## Expected Results

### Baseline (No Fusion)

```
Ensemble: 52.12% accuracy
ECE: 1.92%
Sharpe: 2.55
```

### With Fusion (Optimistic Target)

```
Ensemble: 53-55% accuracy  (+1-3%)
ECE: 1.5-2.0%  (similar or better calibration)
Sharpe: 2.7-3.0  (+0.15-0.45)
```

**Why the improvement?**
- Non-linear combinations capture model synergies
- Learned weights adapt to changing market conditions
- Attention mechanism focuses on best models per situation

---

## Rollback Plan

If fusion degrades performance:

1. **Immediate**: Set `fusion.enabled=False` in config
2. **Investigate**: Check fusion stats for overfitting
3. **Adjust**: Try lower complexity, higher regularization
4. **Re-test**: Run new backtest with adjusted params

The system gracefully falls back to weighted average when fusion is disabled.

---

## Summary

Integration steps:
1. Add `TransformerFusion` to `Ensemble.__init__()`
2. Use `fusion.forward()` in `Ensemble.decide()`
3. Call `ensemble.record_outcome()` in backtest/live
4. Add fusion stats to reports
5. Test baseline vs fusion
6. Save/load trained models

Configuration:
- Start with `enabled=False` to establish baseline
- Enable with conservative params (small `hidden_dim`, high `dropout`)
- Gradually increase complexity if validation loss stays low
- Monitor `train_loss` vs `val_loss` for overfitting

Expected gain: +1-3% accuracy with proper tuning.
