# TransformerFusion - Quick Start Guide

**5-minute setup for Transformer-based ensemble fusion**

---

## Step 1: Verify Installation (30 seconds)

```bash
cd C:\Projects\Titan
python test_fusion.py
```

**Expected output**: `[SUCCESS] ALL TESTS PASSED`

---

## Step 2: Review Configuration (1 minute)

**File**: `titan/core/config.py`

Configuration is already added with sensible defaults:

```python
"fusion.enabled": True,           # Toggle fusion on/off
"fusion.hidden_dim": 32,         # Model size (start small)
"fusion.num_heads": 2,           # Attention heads
"fusion.dropout": 0.2,           # Regularization
"fusion.learning_rate": 0.001,   # Training speed
"fusion.min_samples": 200,       # Min data before training
```

**Action**: No changes needed for initial testing.

---

## Step 3: Test Current Baseline (1 minute)

Run backtest with fusion **disabled** to establish baseline:

```bash
# Edit config.py: Set fusion.enabled = False
python -m titan.cli history --symbol BTCUSDT --interval 1 --hours 24
```

**Record these metrics**:
- Ensemble accuracy: _____%
- ECE: _____%
- Sharpe ratio: _____

---

## Step 4: Enable Fusion (30 seconds)

**File**: `titan/core/config.py`

```python
"fusion.enabled": True,  # Change from False to True
```

---

## Step 5: Run With Fusion (1 minute)

```bash
python -m titan.cli history --symbol BTCUSDT --interval 1 --hours 24
```

**Compare metrics**:
- Did accuracy improve?
- Is ECE similar or better?
- Check report for fusion statistics

---

## Step 6: Check Fusion Status (30 seconds)

Look for new section in `runs/history_*/report.md`:

```markdown
## Transformer Fusion

**Status**: Enabled
**Samples Trained**: 1152
**Train Loss**: 0.6107
**Val Loss**: 0.6009
**Overfitting**: No

### Attention Weights
- TrendVIC: 0.28
- Oscillator: 0.24
- VolumeMetrix: 0.26
- ML_Classifier: 0.22
```

**What to look for**:
- `Samples Trained > 200`: Fusion is active
- `Val Loss ≈ Train Loss`: Not overfitting
- `Attention Weights`: Which models are trusted

---

## Step 7: Interpret Results (1 minute)

### Success Indicators

✓ Accuracy improved by 1-3%
✓ `val_loss ≈ train_loss` (within 20%)
✓ Attention weights are balanced (no single model > 50%)
✓ ECE stayed low (< 2%)

**Action**: Keep fusion enabled!

### Failure Indicators

✗ Accuracy decreased
✗ `val_loss >> train_loss` (overfitting!)
✗ One model dominates attention (> 60%)
✗ ECE increased significantly

**Action**: See troubleshooting below

---

## Troubleshooting

### Problem: Overfitting (`val_loss > train_loss × 1.2`)

**Solution**: Increase regularization

```python
"fusion.dropout": 0.3,        # Increase from 0.2
"fusion.l2_lambda": 0.02,     # Increase from 0.01
"fusion.hidden_dim": 16,      # Decrease from 32
```

### Problem: Accuracy Worse Than Baseline

**Solution**: Fusion needs more data or is too complex

```python
"fusion.min_samples": 500,    # Increase from 200
"fusion.hidden_dim": 16,      # Decrease from 32
```

Or disable fusion:
```python
"fusion.enabled": False,
```

### Problem: One Model Dominates (attention > 60%)

**Solution**: This is actually okay! It means fusion learned that one model is significantly better. But if you want more diversity:

```python
"fusion.num_heads": 4,        # Increase from 2
"fusion.hidden_dim": 64,      # Increase from 32
```

---

## Next Steps

### If Fusion Helps (+1-3% accuracy)

1. **Save the model**:
   ```python
   fusion.save("models/fusion_btcusdt_1m")
   ```

2. **Test on longer period**:
   ```bash
   python -m titan.cli history --hours 168  # 7 days
   ```

3. **Optimize hyperparameters**:
   - Try `hidden_dim=64`
   - Try `num_heads=4`
   - Adjust `dropout` between 0.1-0.3

4. **Use in live trading**:
   ```python
   fusion.load("models/fusion_btcusdt_1m")
   python -m titan.cli live --symbol BTCUSDT --interval 1
   ```

### If Fusion Doesn't Help

1. **Disable for now**:
   ```python
   "fusion.enabled": False,
   ```

2. **Collect more data**: Fusion needs 500+ samples to be effective

3. **Check model quality**: If individual models are poor, fusion won't help

4. **Try simpler baseline**: Make sure weighted ensemble works well first

---

## Full Integration (Advanced)

For complete integration into ensemble:

**See**: `docs/FUSION_INTEGRATION_GUIDE.md`

Steps:
1. Add `fusion.forward()` to `Ensemble.decide()`
2. Add `ensemble.record_outcome()` calls in backtest
3. Add fusion stats to reports
4. Implement save/load for persistence

**Time required**: ~30 minutes

---

## Documentation

- **Technical details**: `docs/TRANSFORMER_FUSION.md`
- **Integration guide**: `docs/FUSION_INTEGRATION_GUIDE.md`
- **Architecture**: `docs/FUSION_ARCHITECTURE.txt`
- **Complete summary**: `FUSION_SUMMARY.md`

---

## Quick Commands

```bash
# Run tests
python test_fusion.py

# Baseline (fusion off)
# Edit config: fusion.enabled = False
python -m titan.cli history --hours 24

# With fusion (fusion on)
# Edit config: fusion.enabled = True
python -m titan.cli history --hours 24

# Compare results in runs/ directory
```

---

## Expected Timeline

| Task | Time | Status |
|------|------|--------|
| Run tests | 30s | [ ] |
| Establish baseline | 1m | [ ] |
| Enable fusion | 30s | [ ] |
| Test with fusion | 1m | [ ] |
| Compare results | 1m | [ ] |
| **Total** | **5m** | |

---

## Success Metrics

**Minimum viable**: Fusion doesn't hurt (accuracy same or better)

**Good**: +1% accuracy improvement, no overfitting

**Excellent**: +2-3% accuracy improvement, better calibration

---

## Support

**Questions?** Check:
1. `docs/TRANSFORMER_FUSION.md` - Troubleshooting section
2. `test_fusion.py` - Usage examples
3. `FUSION_SUMMARY.md` - Complete overview

**Still stuck?** The system has graceful fallback - just set `fusion.enabled = False`

---

**Ready to start? Run Step 1!** ⬆️
