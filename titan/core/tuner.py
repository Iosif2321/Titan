"""
Hyperparameter optimization system using Optuna.

This module provides automatic hyperparameter tuning for Titan's prediction system.
Uses Tree-structured Parzen Estimator (TPE) with pruning for efficient optimization.

Sprint: Hyperparameter Optimization
"""

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import optuna
    from optuna.pruners import MedianPruner, HyperbandPruner
    from optuna.samplers import TPESampler
    from optuna.trial import Trial, TrialState
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    Trial = Any  # type: ignore

from titan.core.backtest import run_backtest
from titan.core.config import ConfigStore, DEFAULT_CONFIG
from titan.core.state_store import StateStore

logger = logging.getLogger(__name__)


# Parameter search space definitions
# Format: (min, max, distribution_type)
TUNABLE_PARAMS = {
    # Model thresholds
    "model.flat_threshold": (0.50, 0.60, "float"),
    "model.rsi_oversold": (25, 35, "int"),
    "model.rsi_overbought": (65, 75, "int"),

    # Ensemble
    "ensemble.flat_threshold": (0.50, 0.60, "float"),
    "ensemble.min_margin": (0.02, 0.10, "float"),

    # Pattern System
    "pattern.boost_threshold": (0.52, 0.60, "float"),
    "pattern.penalty_threshold": (0.40, 0.48, "float"),
    "pattern.max_boost": (0.01, 0.05, "float"),
    "pattern.max_penalty": (0.01, 0.05, "float"),

    # Calibration
    "confidence_compressor.max_confidence": (0.60, 0.75, "float"),
    "confidence_filter.threshold": (0.52, 0.60, "float"),

    # Online learning
    "online.learning_rate": (0.001, 0.1, "log"),
    "online.min_weight": (0.05, 0.20, "float"),
    "online.max_weight": (0.40, 0.60, "float"),

    # Feature engineering
    "feature.fast_window": (3, 10, "int"),
    "feature.slow_window": (15, 30, "int"),
    "feature.rsi_window": (10, 20, "int"),

    # ML Classifier (if enabled)
    "ml.learning_rate": (0.01, 0.2, "log"),
    "ml.max_depth": (4, 10, "int"),
    "ml.num_leaves": (20, 50, "int"),
    "ml.n_estimators": (50, 200, "int"),
}


@dataclass
class OptimizationResult:
    """Results from a single optimization trial."""
    trial_number: int
    params: Dict[str, Any]
    accuracy: float
    ece: float
    sharpe: float
    objective_value: float
    duration_seconds: float
    state: str  # COMPLETE, PRUNED, FAIL


@dataclass
class TunerConfig:
    """Configuration for AutoTuner."""
    n_trials: int = 100
    timeout_per_trial: int = 300  # seconds
    study_name: str = "titan_optimization"
    pruner_type: str = "median"  # median, hyperband, none
    n_jobs: int = 1  # parallel trials
    objective_type: str = "accuracy"  # accuracy, sharpe, multi
    ece_constraint: float = 0.05  # max ECE to consider valid
    min_predictions: int = 100  # minimum predictions for valid trial

    # Multi-objective weights (if objective_type == "multi")
    accuracy_weight: float = 0.7
    sharpe_weight: float = 0.3


class AutoTuner:
    """
    Automatic hyperparameter optimization for Titan.

    Uses Optuna with TPE sampler for efficient parameter search.
    Supports pruning, parallel trials, and multi-objective optimization.

    Example:
        tuner = AutoTuner(db_path="titan.db", n_trials=50)
        best_params = tuner.optimize(candles_path="data.csv")
        tuner.export_config("config_optimized.json")
    """

    def __init__(
        self,
        db_path: str,
        config: Optional[TunerConfig] = None,
    ) -> None:
        """
        Initialize AutoTuner.

        Args:
            db_path: Path to Titan database
            config: Tuner configuration (uses defaults if None)
        """
        if not HAS_OPTUNA:
            raise ImportError(
                "Optuna is required for hyperparameter tuning. "
                "Install with: pip install optuna"
            )

        self.db_path = db_path
        self.config = config or TunerConfig()
        self.study: Optional[optuna.Study] = None
        self.results: List[OptimizationResult] = []
        # On Windows, temporary sqlite DB files can't be deleted while connections are open.
        # Our unit tests create temp *.db files and expect to delete them in finally blocks.
        # When the DB lives in the OS temp dir, auto-dispose connections after cheap ops (ask).
        try:
            tmp_dir = Path(tempfile.gettempdir()).resolve()
            self._auto_dispose_temp_db = os.name == "nt" and Path(db_path).resolve().is_relative_to(tmp_dir)
        except Exception:
            self._auto_dispose_temp_db = False

        # Create/load study
        self._setup_study()

        logger.info(
            f"AutoTuner initialized: {self.config.n_trials} trials, "
            f"{self.config.objective_type} objective"
        )

    def _setup_study(self) -> None:
        """Create or load Optuna study."""
        # Create pruner
        if self.config.pruner_type == "median":
            pruner = MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=5,
            )
        elif self.config.pruner_type == "hyperband":
            pruner = HyperbandPruner(
                min_resource=10,
                max_resource=1000,
                reduction_factor=3,
            )
        else:
            pruner = optuna.pruners.NopPruner()

        # Create sampler
        sampler = TPESampler(
            n_startup_trials=10,
            multivariate=True,
            seed=42,
        )

        # Create or load study
        storage = f"sqlite:///{self.db_path}"

        if self.config.objective_type == "multi":
            # Multi-objective optimization
            self.study = optuna.create_study(
                study_name=self.config.study_name,
                storage=storage,
                load_if_exists=True,
                directions=["maximize", "maximize"],  # accuracy, sharpe
                pruner=pruner,
                sampler=sampler,
            )
        else:
            # Single objective
            self.study = optuna.create_study(
                study_name=self.config.study_name,
                storage=storage,
                load_if_exists=True,
                direction="maximize",
                pruner=pruner,
                sampler=sampler,
            )

        logger.info(
            f"Study '{self.config.study_name}' loaded with "
            f"{len(self.study.trials)} existing trials"
        )

        # Windows NOTE:
        # Optuna wraps RDBStorage inside CachedStorage and keeps DB connections open.
        # On Windows this prevents deleting the sqlite file (tests use temp .db files).
        # Disposing the underlying SQLAlchemy engine releases the file lock while
        # keeping the Study object usable (engine will reconnect lazily).
        self._dispose_storage_engine()
        self._maybe_wrap_study_for_temp_db()

    def _maybe_wrap_study_for_temp_db(self) -> None:
        """Wrap Study methods to dispose connections for temp sqlite DBs (Windows tests)."""
        if not getattr(self, "_auto_dispose_temp_db", False):
            return
        if self.study is None:
            return
        try:
            orig_ask = self.study.ask

            def ask(*args, **kwargs):  # type: ignore[no-untyped-def]
                trial = orig_ask(*args, **kwargs)
                self._dispose_storage_engine()
                return trial

            # Monkey-patch (best-effort) to avoid Windows file locks in tests
            self.study.ask = ask  # type: ignore[method-assign]
        except Exception:
            return

    def _dispose_storage_engine(self) -> None:
        """Dispose Optuna RDBStorage engine if present (releases sqlite file locks)."""
        if self.study is None:
            return
        try:
            storage = getattr(self.study, "_storage", None)
            backend = getattr(storage, "_backend", None)  # CachedStorage -> RDBStorage
            engine = getattr(backend, "engine", None)
            if engine is not None:
                engine.dispose()
        except Exception:
            # Best-effort cleanup; do not fail tuning if disposal isn't available.
            return

    def _suggest_params(self, trial: Trial) -> Dict[str, Any]:
        """Suggest parameters for a trial."""
        params = {}

        for param_name, config in TUNABLE_PARAMS.items():
            if len(config) == 3:
                min_val, max_val, dist_type = config
            else:
                min_val, max_val = config
                dist_type = "float"

            if dist_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    int(min_val),
                    int(max_val),
                )
            elif dist_type == "log":
                params[param_name] = trial.suggest_float(
                    param_name,
                    min_val,
                    max_val,
                    log=True,
                )
            else:  # float
                params[param_name] = trial.suggest_float(
                    param_name,
                    min_val,
                    max_val,
                )

        # For temp sqlite DBs on Windows (unit tests), parameter suggestion can open
        # new DB connections via Trial.suggest_* calls. Dispose them so the temp
        # file can be deleted in test finally blocks.
        if getattr(self, "_auto_dispose_temp_db", False):
            self._dispose_storage_engine()

        return params

    def _run_trial(
        self,
        trial: Trial,
        candles_path: str,
    ) -> Union[float, Tuple[float, float]]:
        """
        Run a single optimization trial.

        Returns:
            Objective value (or tuple for multi-objective)
        """
        start_time = time.time()

        try:
            # Get suggested parameters
            params = self._suggest_params(trial)

            # Create temporary database for this trial
            with tempfile.NamedTemporaryFile(
                suffix=".db",
                delete=False,
            ) as tmp_db:
                tmp_db_path = tmp_db.name

            try:
                # Apply parameters to temporary config
                state_store = StateStore(tmp_db_path)
                config_store = ConfigStore(state_store)
                config_store.ensure_defaults()

                for key, value in params.items():
                    config_store.set(key, value)

                # Run backtest with timeout
                with tempfile.TemporaryDirectory() as tmp_out:
                    stats = run_backtest(
                        csv_path=candles_path,
                        db_path=tmp_db_path,
                        out_dir=tmp_out,
                        limit=None,
                        tune_weights=False,  # Don't tune during optimization
                        return_stats=True,  # Return stats object
                    )

                # Extract metrics
                accuracy = stats.correct / stats.total if stats.total else 0.0
                ece = stats.ece()
                sharpe = stats.sharpe_ratio()
                n_predictions = stats.total

                # Validate trial
                if n_predictions < self.config.min_predictions:
                    logger.warning(
                        f"Trial {trial.number}: Too few predictions "
                        f"({n_predictions} < {self.config.min_predictions})"
                    )
                    raise optuna.TrialPruned()

                if ece > self.config.ece_constraint:
                    logger.warning(
                        f"Trial {trial.number}: ECE too high "
                        f"({ece:.4f} > {self.config.ece_constraint})"
                    )
                    # Don't prune, but penalize
                    accuracy *= 0.9

                # Report intermediate value for pruning
                trial.report(accuracy, step=n_predictions)

                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

                # Calculate objective
                if self.config.objective_type == "accuracy":
                    objective = accuracy
                elif self.config.objective_type == "sharpe":
                    objective = sharpe
                else:  # multi
                    # Return tuple for multi-objective
                    duration = time.time() - start_time

                    # Store result
                    self.results.append(OptimizationResult(
                        trial_number=trial.number,
                        params=params,
                        accuracy=accuracy,
                        ece=ece,
                        sharpe=sharpe,
                        objective_value=accuracy,  # primary
                        duration_seconds=duration,
                        state="COMPLETE",
                    ))

                    logger.info(
                        f"Trial {trial.number} complete: "
                        f"acc={accuracy:.4f}, sharpe={sharpe:.2f}, "
                        f"ece={ece:.4f} ({duration:.1f}s)"
                    )

                    return (accuracy, sharpe)

                # Store result (single objective)
                duration = time.time() - start_time
                self.results.append(OptimizationResult(
                    trial_number=trial.number,
                    params=params,
                    accuracy=accuracy,
                    ece=ece,
                    sharpe=sharpe,
                    objective_value=objective,
                    duration_seconds=duration,
                    state="COMPLETE",
                ))

                logger.info(
                    f"Trial {trial.number} complete: "
                    f"obj={objective:.4f}, acc={accuracy:.4f}, "
                    f"sharpe={sharpe:.2f}, ece={ece:.4f} ({duration:.1f}s)"
                )

                return objective

            finally:
                # Clean up temporary database (may fail on Windows due to file locking)
                if os.path.exists(tmp_db_path):
                    try:
                        os.unlink(tmp_db_path)
                    except (PermissionError, OSError):
                        pass  # Windows may keep file locked, ignore

        except optuna.TrialPruned:
            duration = time.time() - start_time
            self.results.append(OptimizationResult(
                trial_number=trial.number,
                params=trial.params,
                accuracy=0.0,
                ece=0.0,
                sharpe=0.0,
                objective_value=0.0,
                duration_seconds=duration,
                state="PRUNED",
            ))
            logger.info(f"Trial {trial.number} pruned ({duration:.1f}s)")
            raise

        except Exception as e:
            duration = time.time() - start_time
            self.results.append(OptimizationResult(
                trial_number=trial.number,
                params=trial.params if hasattr(trial, 'params') else {},
                accuracy=0.0,
                ece=0.0,
                sharpe=0.0,
                objective_value=0.0,
                duration_seconds=duration,
                state="FAIL",
            ))
            logger.error(f"Trial {trial.number} failed: {e}")
            raise

    def optimize(
        self,
        candles_path: str,
        timeout_total: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            candles_path: Path to CSV with candle data
            timeout_total: Total timeout in seconds (optional)

        Returns:
            Best parameters found
        """
        if self.study is None:
            raise RuntimeError("Study not initialized")

        logger.info(
            f"Starting optimization: {self.config.n_trials} trials, "
            f"timeout={self.config.timeout_per_trial}s/trial"
        )

        # Create objective function
        def objective(trial: Trial) -> Union[float, Tuple[float, float]]:
            return self._run_trial(trial, candles_path)

        # Run optimization
        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=timeout_total,
            n_jobs=self.config.n_jobs,
            show_progress_bar=True,
        )

        # Get best parameters
        best_params = self.get_best_params()

        logger.info(
            f"Optimization complete: {len(self.study.trials)} total trials, "
            f"{len([t for t in self.study.trials if t.state == TrialState.COMPLETE])} "
            f"completed"
        )

        return best_params

    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters from study."""
        if self.study is None:
            raise RuntimeError("No study available")

        if self.config.objective_type == "multi":
            # For multi-objective, use Pareto front
            # Return params with best accuracy among Pareto-optimal solutions
            pareto_trials = self.study.best_trials
            if not pareto_trials:
                return {}

            # Pick trial with highest accuracy
            best_trial = max(pareto_trials, key=lambda t: t.values[0])
            return best_trial.params
        else:
            # Single objective
            try:
                return self.study.best_trial.params
            except ValueError:
                # No completed trials yet
                return {}

    def get_importance(self) -> Dict[str, float]:
        """
        Get parameter importance scores.

        Returns:
            Dict mapping parameter name to importance score (0-1)
        """
        if self.study is None:
            raise RuntimeError("No study available")

        try:
            # Use Optuna's built-in importance evaluation
            importance = optuna.importance.get_param_importances(
                self.study,
                evaluator=optuna.importance.FanovaImportanceEvaluator(),
            )
            return importance
        except Exception as e:
            logger.warning(f"Failed to compute importance: {e}")
            return {}

    def export_config(self, output_path: str) -> None:
        """
        Export best parameters as config file.

        Args:
            output_path: Path to save config JSON
        """
        best_params = self.get_best_params()

        # Merge with default config
        config = DEFAULT_CONFIG.copy()
        config.update(best_params)

        # Save to file
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Config exported to {output_path}")

    def visualize(self, output_dir: str) -> None:
        """
        Generate optimization visualizations.

        Args:
            output_dir: Directory to save plots
        """
        if self.study is None:
            raise RuntimeError("No study available")

        try:
            import optuna.visualization as vis

            os.makedirs(output_dir, exist_ok=True)

            # Optimization history
            fig = vis.plot_optimization_history(self.study)
            fig.write_html(os.path.join(output_dir, "history.html"))

            # Parameter importances
            try:
                fig = vis.plot_param_importances(self.study)
                fig.write_html(os.path.join(output_dir, "importance.html"))
            except Exception as e:
                logger.warning(f"Failed to plot importance: {e}")

            # Parallel coordinate plot
            fig = vis.plot_parallel_coordinate(self.study)
            fig.write_html(os.path.join(output_dir, "parallel.html"))

            # Contour plot (first 2 params)
            params = list(TUNABLE_PARAMS.keys())
            if len(params) >= 2:
                fig = vis.plot_contour(
                    self.study,
                    params=[params[0], params[1]],
                )
                fig.write_html(os.path.join(output_dir, "contour.html"))

            # Slice plot
            fig = vis.plot_slice(self.study)
            fig.write_html(os.path.join(output_dir, "slice.html"))

            logger.info(f"Visualizations saved to {output_dir}")

        except ImportError:
            logger.warning(
                "Visualization requires plotly. "
                "Install with: pip install plotly kaleido"
            )

    def get_summary(self) -> Dict[str, Any]:
        """
        Get optimization summary.

        Returns:
            Dict with optimization statistics
        """
        if self.study is None:
            raise RuntimeError("No study available")

        completed = [
            t for t in self.study.trials
            if t.state == TrialState.COMPLETE
        ]
        pruned = [
            t for t in self.study.trials
            if t.state == TrialState.PRUNED
        ]
        failed = [
            t for t in self.study.trials
            if t.state == TrialState.FAIL
        ]

        summary = {
            "study_name": self.config.study_name,
            "objective_type": self.config.objective_type,
            "total_trials": len(self.study.trials),
            "completed_trials": len(completed),
            "pruned_trials": len(pruned),
            "failed_trials": len(failed),
            "best_params": self.get_best_params(),
        }

        if completed:
            if self.config.objective_type == "multi":
                best_trial = max(completed, key=lambda t: t.values[0])
                summary["best_accuracy"] = best_trial.values[0]
                summary["best_sharpe"] = best_trial.values[1]
            else:
                summary["best_value"] = self.study.best_value

            # Add statistics from results
            completed_results = [
                r for r in self.results
                if r.state == "COMPLETE"
            ]
            if completed_results:
                summary["avg_accuracy"] = sum(
                    r.accuracy for r in completed_results
                ) / len(completed_results)
                summary["avg_ece"] = sum(
                    r.ece for r in completed_results
                ) / len(completed_results)
                summary["avg_sharpe"] = sum(
                    r.sharpe for r in completed_results
                ) / len(completed_results)
                summary["avg_duration"] = sum(
                    r.duration_seconds for r in completed_results
                ) / len(completed_results)

        return summary

    def save_results(self, output_path: str) -> None:
        """
        Save detailed results to JSON.

        Args:
            output_path: Path to save results
        """
        results_data = {
            "summary": self.get_summary(),
            "importance": self.get_importance(),
            "trials": [
                {
                    "trial_number": r.trial_number,
                    "params": r.params,
                    "accuracy": r.accuracy,
                    "ece": r.ece,
                    "sharpe": r.sharpe,
                    "objective_value": r.objective_value,
                    "duration_seconds": r.duration_seconds,
                    "state": r.state,
                }
                for r in self.results
            ],
        }

        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Results saved to {output_path}")

    def close(self) -> None:
        """Close study and release database connections."""
        self._dispose_storage_engine()
        # Optuna handles connections internally, just clear reference
        self.study = None


def create_tuner(
    db_path: str,
    n_trials: int = 100,
    objective: str = "accuracy",
    pruner: str = "median",
) -> AutoTuner:
    """
    Convenience function to create AutoTuner.

    Args:
        db_path: Path to Titan database
        n_trials: Number of optimization trials
        objective: Objective type (accuracy, sharpe, multi)
        pruner: Pruner type (median, hyperband, none)

    Returns:
        Configured AutoTuner instance
    """
    config = TunerConfig(
        n_trials=n_trials,
        objective_type=objective,
        pruner_type=pruner,
    )

    return AutoTuner(db_path=db_path, config=config)
