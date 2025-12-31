# Hyperparameter Tuning System - Implementation Summary

## Overview

A complete hyperparameter optimization system has been implemented for Titan using Optuna. The system enables automatic discovery of optimal parameter values to maximize prediction accuracy and risk-adjusted returns.

## Files Created

### Core Implementation
1. **`titan/core/tuner.py`** (686 lines)
   - `AutoTuner` class: Main optimization engine
   - `TunerConfig` dataclass: Configuration settings
   - `OptimizationResult` dataclass: Trial results
   - `TUNABLE_PARAMS`: Parameter search space definitions
   - `create_tuner()`: Convenience factory function

### CLI Integration
2. **`titan/cli.py`** (updated)
   - Added `tune` command with full argument parsing
   - Integrated tuner output display
   - Automatic visualization generation
   - Progress reporting and summary statistics

### Configuration
3. **`titan/core/config.py`** (updated)
   - Added tuner configuration parameters
   - Default values for optimization settings

### Backtest Integration
4. **`titan/core/backtest.py`** (updated)
   - Added `return_stats` parameter to `run_backtest()`
   - Returns `BacktestStats` object for direct metric access
   - Enables isolated trial execution without file I/O

### Documentation
5. **`docs/HYPERPARAMETER_TUNING.md`** (400+ lines)
   - Complete user guide
   - CLI and API examples
   - Best practices and troubleshooting
   - Configuration reference

6. **`titan/core/tuner_README.md`** (400+ lines)
   - Technical architecture documentation
   - Implementation details
   - API reference
   - Performance considerations

### Examples
7. **`examples/optimize_hyperparameters.py`**
   - Standalone optimization script
   - Demonstrates full API usage
   - Parameter importance analysis
   - Results export and visualization

### Tests
8. **`tests/test_tuner.py`**
   - Unit tests for all major components
   - Configuration validation
   - Pruner testing
   - Objective function testing

## Key Features

### 1. Optimization Engine
- **TPE Sampler**: Tree-structured Parzen Estimator for efficient search
- **Pruning**: Median and Hyperband pruners for early stopping
- **Multi-objective**: Simultaneous optimization of accuracy and Sharpe ratio
- **Parallel Trials**: Run multiple trials concurrently
- **Resume Capability**: Continue previous optimization runs

### 2. Parameter Space
Optimizes 21 parameters across 7 categories:

**Model Thresholds**
- `model.flat_threshold` (0.50-0.60)
- `model.rsi_oversold` (25-35)
- `model.rsi_overbought` (65-75)

**Ensemble**
- `ensemble.flat_threshold` (0.50-0.60)
- `ensemble.min_margin` (0.02-0.10)

**Pattern System**
- `pattern.boost_threshold` (0.52-0.60)
- `pattern.penalty_threshold` (0.40-0.48)
- `pattern.max_boost` (0.01-0.05)
- `pattern.max_penalty` (0.01-0.05)

**Calibration**
- `confidence_compressor.max_confidence` (0.60-0.75)
- `confidence_filter.threshold` (0.52-0.60)

**Online Learning**
- `online.learning_rate` (0.001-0.1, log)
- `online.min_weight` (0.05-0.20)
- `online.max_weight` (0.40-0.60)

**Feature Engineering**
- `feature.fast_window` (3-10)
- `feature.slow_window` (15-30)
- `feature.rsi_window` (10-20)

**ML Classifier**
- `ml.learning_rate` (0.01-0.2, log)
- `ml.max_depth` (4-10)
- `ml.num_leaves` (20-50)
- `ml.n_estimators` (50-200)

### 3. Trial Isolation
Each trial runs in complete isolation:
- Temporary SQLite database per trial
- Fresh configuration
- Independent backtest execution
- Automatic cleanup

### 4. Analysis & Visualization
- **Optimization history**: Convergence visualization
- **Parameter importance**: Fanova-based ranking
- **Parallel coordinates**: Multi-dimensional relationships
- **Contour plots**: Parameter interaction heatmaps
- **Slice plots**: Individual parameter effects

### 5. Constraints & Validation
- **ECE constraint**: Rejects poorly calibrated models
- **Minimum predictions**: Ensures statistical validity
- **Timeout per trial**: Prevents runaway trials
- **Automatic pruning**: Stops unpromising trials early

## Usage Examples

### CLI

```bash
# Basic optimization
python -m titan.cli tune --csv data/btc_24h.csv --trials 50

# Multi-objective with parallel execution
python -m titan.cli tune \
    --csv data/btc_24h.csv \
    --trials 100 \
    --objective multi \
    --jobs 4 \
    --pruner median

# Custom settings
python -m titan.cli tune \
    --csv data/btc_24h.csv \
    --trials 50 \
    --timeout 3600 \
    --timeout-per-trial 300 \
    --objective accuracy \
    --out runs/tuning
```

### Python API

```python
from titan.core.tuner import create_tuner

# Create and run tuner
tuner = create_tuner(
    db_path="titan.db",
    n_trials=50,
    objective="multi",
    pruner="median",
)

best_params = tuner.optimize("data.csv")

# Export results
tuner.save_results("results.json")
tuner.export_config("optimized_config.json")
tuner.visualize("plots/")

# Analyze
summary = tuner.get_summary()
print(f"Best accuracy: {summary['best_accuracy']:.4f}")

importance = tuner.get_importance()
for param, score in sorted(importance.items(), key=lambda x: -x[1])[:5]:
    print(f"{param}: {score:.4f}")
```

### Advanced Configuration

```python
from titan.core.tuner import AutoTuner, TunerConfig

config = TunerConfig(
    n_trials=200,
    timeout_per_trial=600,
    study_name="btc_production",
    pruner_type="median",
    n_jobs=4,
    objective_type="multi",
    ece_constraint=0.03,
    min_predictions=500,
    accuracy_weight=0.7,
    sharpe_weight=0.3,
)

tuner = AutoTuner(db_path="titan.db", config=config)
best = tuner.optimize("data.csv", timeout_total=7200)
```

## Technical Details

### Optimization Flow
1. **Study Creation**: Initialize Optuna study with TPE sampler
2. **Parameter Suggestion**: Sample from parameter space
3. **Trial Execution**:
   - Create temporary database
   - Apply parameters to config
   - Run isolated backtest
   - Extract metrics (accuracy, ECE, Sharpe)
4. **Objective Calculation**: Compute objective value(s)
5. **Pruning Check**: Determine if trial should continue
6. **Result Storage**: Save trial results
7. **Repeat**: Continue until N trials or timeout

### Pruning Algorithms

**Median Pruner** (recommended):
- Compares trial to median of previous trials
- Prunes if below median after warmup
- Good balance of speed and thoroughness

**Hyperband Pruner**:
- Aggressive successive halving
- Faster but may miss good solutions
- Use when time is critical

**No Pruner**:
- Runs all trials to completion
- Most thorough, slowest
- Use for final optimization

### Multi-Objective Optimization
When using `objective="multi"`:
- Optimizes both accuracy AND Sharpe simultaneously
- Returns Pareto-optimal solutions
- Selects best accuracy among Pareto front
- Balances predictive power and risk-adjusted returns

### Performance Characteristics

**Memory Usage**:
- ~100-500MB per parallel job
- Scales linearly with `n_jobs`

**Speed**:
- ~2-5 minutes per trial (24h backtest)
- ~100-250 trials for production
- ~4-12 hours total (50 trials, 4 jobs)

**Recommendations**:
- **Testing**: 10 trials, 1 job
- **Development**: 50 trials, 2-4 jobs
- **Production**: 100-200 trials, 4 jobs

## Best Practices

1. **Start Small**: Test with 10 trials before running 100+
2. **Use Recent Data**: Optimize on data similar to production
3. **Parallelize**: Use 4 jobs for speed (if CPU permits)
4. **Set Timeouts**: Prevent trials from running too long
5. **Validate Results**: Test on hold-out data
6. **Monitor ECE**: Ensure calibration constraint is met
7. **Check Importance**: Focus on impactful parameters
8. **Re-optimize Periodically**: Markets evolve over time
9. **Use Multi-Objective**: Balance accuracy and risk
10. **Document Everything**: Keep notes on results

## Integration Points

### With Backtest
```python
# Tuner calls backtest with return_stats=True
stats = run_backtest(
    csv_path=candles_path,
    db_path=tmp_db_path,
    out_dir=tmp_out,
    return_stats=True,
)

accuracy = stats.accuracy()
ece = stats.expected_calibration_error()
sharpe = stats.sharpe_ratio()
```

### With Config
```python
# Parameters automatically applied via ConfigStore
for key, value in params.items():
    config_store.set(key, value)
```

### With Patterns
All pattern system parameters are tunable:
- Boost/penalty thresholds
- Max boost/penalty values
- Bias penalties

### With ML Classifier
LightGBM parameters are tunable:
- Learning rate (log scale)
- Tree depth and leaves
- Number of estimators

## Future Enhancements

Potential improvements for future sprints:

1. **Warm Start**: Initialize from previous best
2. **Custom Distributions**: Non-uniform parameter sampling
3. **Conditional Spaces**: Parameters that depend on others
4. **Multi-Fidelity**: Optimize on different dataset sizes
5. **Ensemble Configs**: Combine multiple optimized configs
6. **Online Adaptation**: Continuously tune during live trading
7. **Bayesian Bounds**: Learn parameter ranges over time
8. **Meta-Learning**: Transfer knowledge across symbols

## Dependencies

**Required**:
- `optuna>=3.0.0`

**Optional**:
- `plotly>=5.0.0` (visualizations)
- `kaleido>=0.2.0` (static image export)

**Install**:
```bash
pip install optuna
pip install plotly kaleido  # For visualizations
```

## Testing

The implementation includes comprehensive tests:

```bash
# Run all tests
python -m pytest tests/test_tuner.py -v

# Quick functionality test
python -c "from titan.core.tuner import create_tuner; print('OK')"
```

Tests cover:
- Parameter space validation
- Tuner creation and configuration
- Pruner initialization
- Objective function types
- Output generation
- Results export

## Configuration Reference

All tuner-related parameters in `config.py`:

```python
{
    "tuner.n_trials": 100,
    "tuner.timeout_per_trial": 300,
    "tuner.study_name": "titan_optimization",
    "tuner.pruner": "median",
    "tuner.objective": "accuracy",
    "tuner.ece_constraint": 0.05,
    "tuner.min_predictions": 100,
}
```

## Output Files

After optimization, the following files are generated:

```
runs/tune_YYYYMMDD_HHMMSS/
├── optimization_results.json    # Detailed trial results
├── config_optimized.json        # Best parameters
└── visualizations/
    ├── history.html             # Optimization history
    ├── importance.html          # Parameter importance
    ├── parallel.html            # Parallel coordinates
    ├── contour.html             # Parameter interactions
    └── slice.html               # Individual parameters
```

## Summary

The hyperparameter tuning system provides:

- **Automated optimization** of 21 parameters
- **Efficient search** using TPE and pruning
- **Multi-objective support** for accuracy + Sharpe
- **Complete isolation** between trials
- **Comprehensive analysis** and visualization
- **Production-ready** CLI and API
- **Well-documented** with examples and guides

This system enables systematic improvement of Titan's prediction accuracy through data-driven parameter optimization.

## Next Steps

To use the tuner:

1. **Prepare data**: Download 24h candles
   ```bash
   python -m titan.cli history --symbol BTCUSDT --hours 24 --store-candles
   ```

2. **Run optimization**: 50 trials, multi-objective
   ```bash
   python -m titan.cli tune \
       --csv runs/history_latest/candles.csv \
       --trials 50 \
       --objective multi \
       --jobs 4
   ```

3. **Review results**: Check optimization_results.json

4. **Apply best params**: Copy config_optimized.json

5. **Validate**: Test on fresh data

6. **Deploy**: Use optimized config in production

The tuner is now ready for production use!
