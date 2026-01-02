# Hyperparameter Tuning with Optuna

This guide explains how to use Titan's hyperparameter optimization system to find the best configuration for your trading system.

## Overview

The hyperparameter tuner uses **Optuna** with Tree-structured Parzen Estimator (TPE) to efficiently search the parameter space. It includes:

- **Smart pruning** to stop unpromising trials early
- **Multi-objective optimization** (accuracy + Sharpe ratio)
- **Parallel trial execution** for faster optimization
- **Visualization** of parameter importance and optimization history
- **Resume capability** to continue previous optimization runs

## Installation

```bash
pip install optuna
pip install plotly kaleido  # For visualizations (optional)
```

## Quick Start

### Using the CLI

```bash
# Basic optimization (50 trials, accuracy objective)
python -m titan.cli tune --csv data/btc_24h.csv --trials 50

# Multi-objective optimization (accuracy + Sharpe)
python -m titan.cli tune --csv data/btc_24h.csv --trials 100 --objective multi

# Parallel execution (4 jobs)
python -m titan.cli tune --csv data/btc_24h.csv --trials 100 --jobs 4

# Custom timeout and pruner
python -m titan.cli tune --csv data/btc_24h.csv --trials 50 \
    --timeout 3600 --timeout-per-trial 300 --pruner hyperband
```

### Using Python API

```python
from titan.core.tuner import create_tuner

# Create tuner
tuner = create_tuner(
    db_path="titan.db",
    n_trials=50,
    objective="accuracy",
    pruner="median",
)

# Run optimization
best_params = tuner.optimize(candles_path="data/btc_24h.csv")

# Save results
tuner.save_results("results.json")
tuner.export_config("optimized_config.json")
tuner.visualize("plots/")

# Get summary
summary = tuner.get_summary()
print(f"Best accuracy: {summary['best_accuracy']:.4f}")

# Get parameter importance
importance = tuner.get_importance()
for param, score in sorted(importance.items(), key=lambda x: -x[1])[:5]:
    print(f"{param}: {score:.4f}")
```

## Parameters Being Tuned

The tuner optimizes these parameter groups:

### Model Thresholds
- `model.flat_threshold` (0.50 - 0.60): Threshold for FLAT predictions
- `model.rsi_oversold` (25 - 35): RSI oversold level
- `model.rsi_overbought` (65 - 75): RSI overbought level

### Ensemble Configuration
- `ensemble.flat_threshold` (0.50 - 0.60): Ensemble FLAT threshold
- `ensemble.min_margin` (0.02 - 0.10): Minimum probability margin

### Pattern System
- `pattern.boost_threshold` (0.52 - 0.60): Accuracy threshold for confidence boost
- `pattern.penalty_threshold` (0.40 - 0.48): Accuracy threshold for confidence penalty
- `pattern.max_boost` (0.01 - 0.05): Maximum confidence boost
- `pattern.max_penalty` (0.01 - 0.05): Maximum confidence penalty

### Calibration
- `confidence_compressor.max_confidence` (0.60 - 0.75): Maximum output confidence
- `confidence_filter.threshold` (0.52 - 0.60): Minimum actionable confidence

### Online Learning
- `online.learning_rate` (0.001 - 0.1, log scale): Learning rate for weight updates
- `online.min_weight` (0.05 - 0.20): Minimum model weight
- `online.max_weight` (0.40 - 0.60): Maximum model weight

### Feature Engineering
- `feature.fast_window` (3 - 10): Fast moving average window
- `feature.slow_window` (15 - 30): Slow moving average window
- `feature.rsi_window` (10 - 20): RSI calculation window

### ML Classifier (if enabled)
- `ml.learning_rate` (0.01 - 0.2, log scale): LightGBM learning rate
- `ml.max_depth` (4 - 10): Maximum tree depth
- `ml.num_leaves` (20 - 50): Number of leaves per tree
- `ml.n_estimators` (50 - 200): Number of boosting rounds

## Optimization Objectives

### Accuracy (default)
Maximizes prediction accuracy. Best for systems focused on directional correctness.

```bash
python -m titan.cli tune --csv data.csv --objective accuracy
```

### Sharpe Ratio
Maximizes risk-adjusted returns. Best for trading-focused systems.

```bash
python -m titan.cli tune --csv data.csv --objective sharpe
```

### Multi-objective
Optimizes both accuracy and Sharpe ratio simultaneously. Returns Pareto-optimal solutions.

```bash
python -m titan.cli tune --csv data.csv --objective multi
```

## Pruning Strategies

### Median Pruner (default)
Stops trials that are performing worse than the median of previous trials. Good balance of speed and thoroughness.

```python
tuner = create_tuner(pruner="median")
```

### Hyperband Pruner
More aggressive pruning using successive halving. Faster but may miss good solutions.

```python
tuner = create_tuner(pruner="hyperband")
```

### No Pruner
Runs all trials to completion. Slowest but most thorough.

```python
tuner = create_tuner(pruner="none")
```

## Advanced Usage

### Resuming Optimization

The tuner automatically saves progress to SQLite. To resume:

```python
# First run
tuner = create_tuner(db_path="titan.db", n_trials=50)
tuner.optimize("data.csv")  # Runs 50 trials

# Later - resume with more trials
tuner2 = create_tuner(db_path="titan.db", n_trials=100)
tuner2.optimize("data.csv")  # Runs 50 MORE trials (total 100)
```

### Custom Configuration

```python
from titan.core.tuner import AutoTuner, TunerConfig

config = TunerConfig(
    n_trials=200,
    timeout_per_trial=600,  # 10 minutes per trial
    study_name="titan_btc_optimization",
    pruner_type="median",
    n_jobs=4,  # Parallel trials
    objective_type="multi",
    ece_constraint=0.05,  # Reject trials with ECE > 5%
    min_predictions=500,  # Minimum predictions for valid trial
    accuracy_weight=0.7,  # For multi-objective
    sharpe_weight=0.3,
)

tuner = AutoTuner(db_path="titan.db", config=config)
best_params = tuner.optimize("data.csv")
```

### Analyzing Results

```python
# Get detailed summary
summary = tuner.get_summary()
print(f"Trials: {summary['total_trials']}")
print(f"Completed: {summary['completed_trials']}")
print(f"Pruned: {summary['pruned_trials']}")
print(f"Failed: {summary['failed_trials']}")
print(f"Best accuracy: {summary['best_accuracy']:.4f}")
print(f"Avg accuracy: {summary['avg_accuracy']:.4f}")
print(f"Avg ECE: {summary['avg_ece']:.4f}")

# Parameter importance (Fanova)
importance = tuner.get_importance()
for param, score in sorted(importance.items(), key=lambda x: -x[1]):
    print(f"{param}: {score:.4f}")

# Access Optuna study directly
study = tuner.study
print(f"Best trial: {study.best_trial.number}")
print(f"Best value: {study.best_value:.4f}")

# Get all completed trials
completed = [t for t in study.trials if t.state.name == "COMPLETE"]
for trial in completed[:5]:
    print(f"Trial {trial.number}: {trial.value:.4f}")
```

## Visualizations

The tuner generates interactive HTML visualizations:

### Optimization History
Shows objective value over trials. Useful for seeing convergence.

### Parameter Importance
Ranks parameters by their impact on the objective. Focus tuning efforts on important parameters.

### Parallel Coordinate Plot
Shows relationships between parameters and objective. Identify parameter interactions.

### Contour Plot
2D heatmap of parameter pairs. See how pairs of parameters interact.

### Slice Plot
Individual parameter effects. See how each parameter affects the objective.

## Best Practices

### 1. Use Representative Data
Optimize on data similar to what you'll use in production:
```bash
# Good: Recent 24h data for live trading
python -m titan.cli tune --csv recent_24h.csv

# Bad: Old data from different market regime
python -m titan.cli tune --csv data_from_2020.csv
```

### 2. Start with Fewer Trials
Test your setup with 10-20 trials before running 100+:
```bash
# Test run
python -m titan.cli tune --csv data.csv --trials 10

# Full run after confirming it works
python -m titan.cli tune --csv data.csv --trials 100
```

### 3. Use Parallel Execution
Speed up optimization with multiple jobs (if you have CPU cores):
```bash
python -m titan.cli tune --csv data.csv --trials 100 --jobs 4
```

### 4. Set Reasonable Timeouts
Prevent trials from running too long:
```bash
python -m titan.cli tune --csv data.csv --timeout-per-trial 300  # 5 min max
```

### 5. Check ECE Constraint
Ensure optimized models are well-calibrated:
```python
config = TunerConfig(
    ece_constraint=0.05,  # Reject if ECE > 5%
)
```

### 6. Validate on Hold-out Set
After optimization, test on separate data:
```python
# Optimize on training data
tuner.optimize("train_data.csv")

# Export config
tuner.export_config("optimized_config.json")

# Test on hold-out data
from titan.core.backtest import run_backtest
stats = run_backtest(
    csv_path="test_data.csv",
    db_path="titan_test.db",  # Use config with optimized params
    out_dir="validation/",
)
```

## Troubleshooting

### Issue: "Optuna is required for hyperparameter tuning"
**Solution**: Install Optuna
```bash
pip install optuna
```

### Issue: All trials getting pruned
**Solution**: Use less aggressive pruner or disable pruning
```bash
python -m titan.cli tune --csv data.csv --pruner none
```

### Issue: Trials taking too long
**Solution**: Reduce timeout or use smaller dataset
```bash
python -m titan.cli tune --csv data.csv --timeout-per-trial 120
```

### Issue: Visualizations failing
**Solution**: Install plotly
```bash
pip install plotly kaleido
```

### Issue: Out of memory with parallel jobs
**Solution**: Reduce number of jobs
```bash
python -m titan.cli tune --csv data.csv --jobs 2  # Instead of 4
```

## Example Workflow

Complete workflow from optimization to deployment:

```bash
# 1. Prepare data (24h recent data)
python -m titan.cli history --symbol BTCUSDT --hours 24 --store-candles

# 2. Run optimization (50 trials, multi-objective)
python -m titan.cli tune \
    --csv runs/history_latest/candles.csv \
    --trials 50 \
    --objective multi \
    --pruner median \
    --jobs 4 \
    --out runs/tuning

# 3. Review results
cat runs/tuning/tune_*/optimization_results.json

# 4. Copy optimized config
cp runs/tuning/tune_*/config_optimized.json config_prod.json

# 5. Validate on fresh data
python -m titan.cli history --symbol BTCUSDT --hours 24 --run-id validation

# 6. Apply optimized params and test
# (Load config_prod.json in your application)

# 7. Deploy to live if validation looks good
python -m titan.cli live --symbol BTCUSDT --interval 1
```

## Tips for Better Results

1. **More trials = better results**: 100+ trials recommended for production
2. **Use multi-objective**: Balances accuracy and risk-adjusted returns
3. **Optimize on recent data**: Market conditions change over time
4. **Check parameter importance**: Focus on what matters
5. **Cross-validate**: Test optimized params on different time periods
6. **Re-optimize periodically**: Markets evolve, so should your parameters
7. **Monitor ECE**: Well-calibrated confidence is crucial
8. **Use median pruner**: Good default for most cases
9. **Start with default ranges**: Expand if best params hit boundaries
10. **Document your runs**: Keep notes on what worked and what didn't

## Configuration Reference

All tuner-related config parameters:

```python
{
    "tuner.n_trials": 100,              # Number of optimization trials
    "tuner.timeout_per_trial": 300,      # Seconds per trial
    "tuner.study_name": "titan_optimization",  # Study name in DB
    "tuner.pruner": "median",            # Pruner algorithm
    "tuner.objective": "accuracy",       # Optimization objective
    "tuner.ece_constraint": 0.05,        # Max ECE to accept
    "tuner.min_predictions": 100,        # Min predictions per trial
}
```

## See Also

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [TPE Sampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)
- [Pruning Algorithms](https://optuna.readthedocs.io/en/stable/reference/pruners.html)
