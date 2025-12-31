"""
Tests for hyperparameter tuner.

These are integration tests that verify the tuner works correctly.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

try:
    from titan.core.tuner import (
        AutoTuner,
        TunerConfig,
        create_tuner,
        HAS_OPTUNA,
        TUNABLE_PARAMS,
    )
    SKIP_TESTS = not HAS_OPTUNA
    SKIP_REASON = "Optuna not installed"
except ImportError:
    SKIP_TESTS = True
    SKIP_REASON = "Could not import tuner module"


@unittest.skipIf(SKIP_TESTS, SKIP_REASON)
class TestTunerBasics(unittest.TestCase):
    """Test basic tuner functionality."""

    def test_tunable_params_defined(self):
        """Test that tunable parameters are properly defined."""
        self.assertGreater(len(TUNABLE_PARAMS), 0)

        # Check format of each parameter
        for param_name, config in TUNABLE_PARAMS.items():
            self.assertIsInstance(param_name, str)
            self.assertIn(len(config), [2, 3])  # (min, max) or (min, max, type)

            min_val, max_val = config[0], config[1]
            self.assertLess(min_val, max_val)

            if len(config) == 3:
                dist_type = config[2]
                self.assertIn(dist_type, ["int", "float", "log"])

    def test_create_tuner(self):
        """Test tuner creation."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            tuner = create_tuner(
                db_path=db_path,
                n_trials=5,
                objective="accuracy",
                pruner="median",
            )

            self.assertIsInstance(tuner, AutoTuner)
            self.assertEqual(tuner.config.n_trials, 5)
            self.assertEqual(tuner.config.objective_type, "accuracy")
            self.assertEqual(tuner.config.pruner_type, "median")

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_tuner_config(self):
        """Test TunerConfig dataclass."""
        config = TunerConfig(
            n_trials=50,
            timeout_per_trial=300,
            study_name="test_study",
            pruner_type="hyperband",
            n_jobs=2,
            objective_type="sharpe",
            ece_constraint=0.03,
            min_predictions=200,
        )

        self.assertEqual(config.n_trials, 50)
        self.assertEqual(config.timeout_per_trial, 300)
        self.assertEqual(config.study_name, "test_study")
        self.assertEqual(config.pruner_type, "hyperband")
        self.assertEqual(config.n_jobs, 2)
        self.assertEqual(config.objective_type, "sharpe")
        self.assertEqual(config.ece_constraint, 0.03)
        self.assertEqual(config.min_predictions, 200)

    def test_tuner_initialization(self):
        """Test AutoTuner initialization."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            config = TunerConfig(n_trials=10, objective_type="accuracy")
            tuner = AutoTuner(db_path=db_path, config=config)

            self.assertIsNotNone(tuner.study)
            self.assertEqual(len(tuner.results), 0)
            self.assertEqual(tuner.config.n_trials, 10)

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_suggest_params(self):
        """Test parameter suggestion."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            tuner = create_tuner(db_path=db_path, n_trials=1)

            # Create a trial to test parameter suggestion
            trial = tuner.study.ask()
            params = tuner._suggest_params(trial)

            # Check that all expected parameters are suggested
            self.assertIsInstance(params, dict)
            self.assertGreater(len(params), 0)

            # Check parameter values are in valid ranges
            for param_name, value in params.items():
                if param_name in TUNABLE_PARAMS:
                    config = TUNABLE_PARAMS[param_name]
                    min_val, max_val = config[0], config[1]
                    self.assertGreaterEqual(value, min_val)
                    self.assertLessEqual(value, max_val)

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


@unittest.skipIf(SKIP_TESTS, SKIP_REASON)
class TestTunerOutput(unittest.TestCase):
    """Test tuner output methods."""

    def setUp(self):
        """Create a tuner for testing."""
        self.tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.tmp_db.name
        self.tmp_db.close()  # Close file handle
        self.tuner = create_tuner(
            db_path=self.db_path,
            n_trials=5,
            objective="accuracy",
        )

    def tearDown(self):
        """Clean up temporary files."""
        # Close tuner to release DB connections
        if hasattr(self, 'tuner') and self.tuner:
            self.tuner.close()

        # Small delay for Windows file system
        import time
        time.sleep(0.1)

        if os.path.exists(self.db_path):
            try:
                os.unlink(self.db_path)
            except PermissionError:
                pass  # File still locked, ignore

    def test_get_summary_empty(self):
        """Test get_summary with no trials."""
        summary = self.tuner.get_summary()

        self.assertIsInstance(summary, dict)
        self.assertIn("total_trials", summary)
        self.assertIn("completed_trials", summary)
        self.assertEqual(summary["total_trials"], 0)
        self.assertEqual(summary["completed_trials"], 0)

    def test_export_config(self):
        """Test config export."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "config.json")
            self.tuner.export_config(config_path)

            self.assertTrue(os.path.exists(config_path))

            with open(config_path) as f:
                config = json.load(f)

            self.assertIsInstance(config, dict)
            # Should contain default config values
            self.assertGreater(len(config), 0)

    def test_save_results(self):
        """Test results saving."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_path = os.path.join(tmp_dir, "results.json")
            self.tuner.save_results(results_path)

            self.assertTrue(os.path.exists(results_path))

            with open(results_path) as f:
                results = json.load(f)

            self.assertIsInstance(results, dict)
            self.assertIn("summary", results)
            self.assertIn("trials", results)


@unittest.skipIf(SKIP_TESTS, SKIP_REASON)
class TestTunerPruners(unittest.TestCase):
    """Test different pruner types."""

    def test_median_pruner(self):
        """Test median pruner creation."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            tuner = create_tuner(db_path=db_path, pruner="median")
            self.assertEqual(tuner.config.pruner_type, "median")
            self.assertIsNotNone(tuner.study)

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_hyperband_pruner(self):
        """Test hyperband pruner creation."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            tuner = create_tuner(db_path=db_path, pruner="hyperband")
            self.assertEqual(tuner.config.pruner_type, "hyperband")
            self.assertIsNotNone(tuner.study)

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_no_pruner(self):
        """Test no pruner."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            tuner = create_tuner(db_path=db_path, pruner="none")
            self.assertEqual(tuner.config.pruner_type, "none")
            self.assertIsNotNone(tuner.study)

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


@unittest.skipIf(SKIP_TESTS, SKIP_REASON)
class TestTunerObjectives(unittest.TestCase):
    """Test different objective types."""

    def test_accuracy_objective(self):
        """Test accuracy objective."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            tuner = create_tuner(db_path=db_path, objective="accuracy")
            self.assertEqual(tuner.config.objective_type, "accuracy")

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_sharpe_objective(self):
        """Test Sharpe objective."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            tuner = create_tuner(db_path=db_path, objective="sharpe")
            self.assertEqual(tuner.config.objective_type, "sharpe")

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_multi_objective(self):
        """Test multi-objective optimization."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            tuner = create_tuner(db_path=db_path, objective="multi")
            self.assertEqual(tuner.config.objective_type, "multi")

            # Multi-objective should use different study direction
            self.assertIsNotNone(tuner.study)
            self.assertEqual(len(tuner.study.directions), 2)

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


if __name__ == "__main__":
    unittest.main()
