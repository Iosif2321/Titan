# AutoTuner Module

Hyperparameter optimization system using Optuna for Titan's prediction models.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AutoTuner                              │
│  - Study management (SQLite storage)                        │
│  - Parameter suggestion (TPE sampler)                       │
│  - Trial execution (isolated DB per trial)                  │
│  - Pruning (Median/Hyperband)                              │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Trial Execution                            │
│  1. Suggest parameters                                      │
│  2. Create temporary DB                                     │
│  3. Apply parameters to config                              │
│  4. Run backtest                                            │
│  5. Extract metrics (accuracy, ECE, Sharpe)                 │
│  6. Report to Optuna                                        │
│  7. Check if should prune                                   │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Results & Analysis                         │
│  - Best parameters                                          │
│  - Parameter importance (Fanova)                            │
│  - Optimization history                                     │
│  - Visualizations (plots)                                   │
│  - Config export                                            │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### AutoTuner
Main optimization class. Manages the entire optimization process.

**Methods:**
- `optimize(candles_path, timeout_total)`: Run optimization
- `get_best_params()`: Get best parameter set
- `get_importance()`: Get parameter importance scores
- `export_config(path)`: Export optimized config
- `visualize(dir)`: Generate visualization plots
- `get_summary()`: Get optimization statistics
- `save_results(path)`: Save detailed results to JSON

### TunerConfig
Configuration dataclass for tuner settings.

**Fields:**
- `n_trials`: Number of optimization trials
- `timeout_per_trial`: Max seconds per trial
- `study_name`: Name for Optuna study
- `pruner_type`: Pruning algorithm (median, hyperband, none)
- `n_jobs`: Number of parallel trials
- `objective_type`: Optimization objective (accuracy, sharpe, multi)
- `ece_constraint`: Maximum ECE to accept (trials with higher ECE are penalized)
- `min_predictions`: Minimum predictions required for valid trial
- `accuracy_weight`: Weight for accuracy in multi-objective (default 0.7)
- `sharpe_weight`: Weight for Sharpe in multi-objective (default 0.3)

### OptimizationResult
Results from a single trial.

**Fields:**
- `trial_number`: Trial ID
- `params`: Parameters tested
- `accuracy`: Achieved accuracy
- `ece`: Expected Calibration Error
- `sharpe`: Sharpe ratio
- `objective_value`: Final objective value
- `duration_seconds`: Trial duration
- `state`: Trial state (COMPLETE, PRUNED, FAIL)

## Parameter Search Space

### TUNABLE_PARAMS
Dictionary defining parameter ranges:

```python
{
    "param_name": (min, max, distribution_type),
}
```

**Distribution types:**
- `"int"`: Integer parameter (discrete)
- `"float"`: Float parameter (continuous)
- `"log"`: Float parameter (log scale, for learning rates)

**Example:**
```python
"online.learning_rate": (0.001, 0.1, "log"),  # Log scale
"model.rsi_oversold": (25, 35, "int"),        # Integer
"pattern.max_boost": (0.01, 0.05, "float"),   # Float
```

## Optimization Flow

### Single Objective
1. Suggest parameters using TPE sampler
2. Run backtest with parameters
3. Calculate objective (accuracy or Sharpe)
4. Report value to Optuna
5. Check if should prune
6. Repeat for N trials
7. Return best parameters

### Multi-Objective
1. Suggest parameters using TPE sampler
2. Run backtest with parameters
3. Calculate both accuracy AND Sharpe
4. Report tuple (accuracy, sharpe) to Optuna
5. Check if should prune
6. Repeat for N trials
7. Get Pareto front
8. Return parameters with best accuracy among Pareto-optimal

## Pruning

### How It Works
Optuna monitors intermediate values during trial execution. If a trial is performing poorly compared to others, it's stopped early to save time.

### Median Pruner
- Compares trial to median of previous trials
- Prunes if below median after warmup period
- Good balance of speed vs thoroughness
- **Recommended for most use cases**

**Parameters:**
- `n_startup_trials=5`: Don't prune first 5 trials
- `n_warmup_steps=10`: Don't prune before 10 predictions
- `interval_steps=5`: Check every 5 predictions

### Hyperband Pruner
- More aggressive, uses successive halving
- Prunes aggressively based on resource allocation
- Faster but may miss good solutions
- **Use when time is critical**

**Parameters:**
- `min_resource=10`: Minimum predictions before pruning
- `max_resource=1000`: Maximum predictions
- `reduction_factor=3`: How aggressively to prune

### No Pruner
- Runs all trials to completion
- Slowest but most thorough
- **Use for final optimization runs**

## Trial Isolation

Each trial runs in complete isolation:

1. **Temporary Database**: Creates new SQLite DB for trial
2. **Fresh Config**: Applies trial parameters to config
3. **Independent Backtest**: Runs backtest with no shared state
4. **Cleanup**: Deletes temporary DB after trial

This ensures trials don't interfere with each other.

## Performance Considerations

### Memory Usage
- Each parallel job uses ~100-500MB RAM
- Reduce `n_jobs` if running out of memory
- Use `limit` parameter in backtest for testing

### Speed Optimization
- **Parallel trials**: Set `n_jobs=4` (or number of CPU cores)
- **Pruning**: Use median or hyperband pruner
- **Timeout**: Set `timeout_per_trial` to prevent runaway trials
- **Smaller dataset**: Use representative subset for optimization

### Recommendations
- **Quick test**: 10 trials, 1 job, median pruner
- **Production**: 100+ trials, 4 jobs, median pruner
- **Final tuning**: 200 trials, 4 jobs, no pruner

## Error Handling

### Trial Failure
When a trial fails:
1. Error is logged
2. Trial marked as FAIL
3. Optimization continues with next trial
4. Failed trial doesn't affect best parameters

### Pruned Trials
When a trial is pruned:
1. Trial stopped early
2. Marked as PRUNED (not FAIL)
3. Resources freed for next trial
4. Counts toward total trials

### Common Issues

**ECE too high:**
- Trial penalized (accuracy *= 0.9)
- Trial continues (not pruned)
- Helps ensure calibrated models

**Too few predictions:**
- Trial pruned immediately
- Prevents invalid trials
- Check `min_predictions` config

**Timeout:**
- Trial marked as FAIL
- Check `timeout_per_trial` setting
- May need to reduce dataset size

## Visualization

### Generated Plots

**optimization_history.html**
- X-axis: Trial number
- Y-axis: Objective value
- Shows convergence over time

**param_importances.html**
- Bar chart of parameter importance
- Based on Fanova analysis
- Identifies key parameters

**parallel_coordinate.html**
- Multiple parameter dimensions
- Color-coded by objective value
- Shows parameter interactions

**contour.html**
- 2D heatmap of parameter pairs
- Shows optimal parameter combinations
- Interactive zoom/pan

**slice.html**
- Individual parameter effects
- One plot per parameter
- Shows marginal effect

## API Examples

### Basic Usage
```python
from titan.core.tuner import create_tuner

tuner = create_tuner(
    db_path="titan.db",
    n_trials=50,
    objective="accuracy",
)

best = tuner.optimize("data.csv")
tuner.export_config("config.json")
```

### Advanced Usage
```python
from titan.core.tuner import AutoTuner, TunerConfig

config = TunerConfig(
    n_trials=100,
    timeout_per_trial=600,
    study_name="btc_optimization",
    pruner_type="median",
    n_jobs=4,
    objective_type="multi",
    ece_constraint=0.03,
    min_predictions=500,
)

tuner = AutoTuner("titan.db", config)
best = tuner.optimize("data.csv", timeout_total=3600)

# Analysis
summary = tuner.get_summary()
importance = tuner.get_importance()
tuner.save_results("results.json")
tuner.visualize("plots/")
```

### Resume Previous Study
```python
# Study automatically saved to DB
tuner1 = create_tuner(db_path="titan.db", n_trials=50)
tuner1.optimize("data.csv")  # 50 trials

# Later, resume with more trials
tuner2 = create_tuner(db_path="titan.db", n_trials=100)
tuner2.optimize("data.csv")  # 50 MORE trials (total 100)
```

## Integration with Backtest

The tuner uses `run_backtest()` with `return_stats=True`:

```python
stats = run_backtest(
    csv_path=candles_path,
    db_path=tmp_db_path,
    out_dir=tmp_out,
    limit=None,
    tune_weights=False,  # Don't tune during optimization
    return_stats=True,   # Return BacktestStats object
)

accuracy = stats.accuracy()
ece = stats.expected_calibration_error()
sharpe = stats.sharpe_ratio()
```

This provides direct access to metrics without parsing JSON files.

## Study Storage

Optuna stores study data in SQLite database:

```
titan.db
├── optuna_studies          # Study metadata
├── optuna_trials           # Trial records
├── optuna_trial_params     # Parameter values
├── optuna_trial_values     # Objective values
├── optuna_trial_intermediate_values  # For pruning
└── ... (other Titan tables)
```

**Benefits:**
- Persistent across runs
- Resumable optimization
- Multiple concurrent studies
- SQL queryable

## Dependencies

**Required:**
- `optuna>=3.0.0`: Optimization framework

**Optional:**
- `plotly>=5.0.0`: Interactive visualizations
- `kaleido>=0.2.0`: Static image export

**Install:**
```bash
pip install optuna
pip install plotly kaleido  # For visualizations
```

## Best Practices

1. **Start small**: Test with 10 trials before running 100+
2. **Use representative data**: Optimize on recent, relevant data
3. **Validate results**: Test optimized params on hold-out set
4. **Monitor ECE**: Ensure calibration constraint is met
5. **Check importance**: Focus on impactful parameters
6. **Save everything**: Export config, results, and plots
7. **Document runs**: Note date, data, and results
8. **Re-optimize periodically**: Markets change over time
9. **Use multi-objective**: Balance accuracy and risk
10. **Parallelize**: Use multiple jobs for speed

## Future Enhancements

Potential improvements:
- [ ] Warm start from previous best
- [ ] Custom parameter distributions
- [ ] Conditional parameter spaces
- [ ] Multi-fidelity optimization (different dataset sizes)
- [ ] Ensemble of optimized configs
- [ ] Online parameter adaptation
- [ ] Bayesian hyperparameter bounds
- [ ] Meta-learning across symbols
