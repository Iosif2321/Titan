import unittest
from collections import deque

from src.adaptive_calibration import AdaptiveCalibController
from src.calibration import AffineLogitCalibState, AffineLogitCalibConfig
from src.config import PerModelCalibConfig, TrainingConfig


class AdaptiveCalibrationTests(unittest.TestCase):
    def test_create_with_defaults(self) -> None:
        """Test factory creation with default TrainingConfig."""
        training = TrainingConfig()
        controller = AdaptiveCalibController.create(
            model_type="TEST", training_config=training, per_model_config=None
        )

        self.assertEqual(controller.model_type, "TEST")
        self.assertEqual(controller.state.a, 1.0)
        self.assertEqual(controller.state.b, 0.0)
        self.assertEqual(controller.lr_current, training.calib_lr)
        self.assertEqual(controller.lr_base, training.calib_lr)
        self.assertEqual(controller.ece_target, training.calib_ece_target)

    def test_create_with_per_model_override(self) -> None:
        """Test factory creation with per-model config override."""
        training = TrainingConfig()
        per_model = PerModelCalibConfig(lr=0.01, a_min=0.25, lr_max=0.03, ece_target=0.08)

        controller = AdaptiveCalibController.create(
            model_type="OSCILLATOR", training_config=training, per_model_config=per_model
        )

        self.assertEqual(controller.model_type, "OSCILLATOR")
        self.assertEqual(controller.lr_current, 0.01)
        self.assertEqual(controller.lr_base, 0.01)
        self.assertEqual(controller.config.a_min, 0.25)
        self.assertEqual(controller.lr_max, 0.03)
        self.assertEqual(controller.ece_target, 0.08)

    def test_update_without_ece(self) -> None:
        """Test update works when ECE is None (insufficient data)."""
        training = TrainingConfig()
        controller = AdaptiveCalibController.create(
            model_type="TEST", training_config=training, per_model_config=None
        )

        initial_a = controller.state.a
        initial_lr = controller.lr_current

        controller.update(
            logit_up=1.0, logit_down=0.0, y_up=1.0, weight=1.0, flip=False, current_ece=None
        )

        # State should update via SGD
        self.assertNotEqual(controller.state.a, initial_a)
        # But lr should not adapt (no ECE)
        self.assertEqual(controller.lr_current, initial_lr)
        self.assertEqual(len(controller.ece_window), 0)

    def test_update_with_ece(self) -> None:
        """Test update tracks ECE in window."""
        training = TrainingConfig()
        controller = AdaptiveCalibController.create(
            model_type="TEST", training_config=training, per_model_config=None
        )

        controller.update(
            logit_up=1.0, logit_down=0.0, y_up=1.0, weight=1.0, flip=False, current_ece=0.15
        )

        self.assertEqual(len(controller.ece_window), 1)
        self.assertEqual(controller.ece_window[0], 0.15)

    def test_lr_adaptation_high_ece(self) -> None:
        """Test lr increases when ECE is high."""
        training = TrainingConfig()
        controller = AdaptiveCalibController.create(
            model_type="TEST", training_config=training, per_model_config=None
        )

        initial_lr = controller.lr_current

        # Fill window with enough samples (min_samples_for_adaptation=30) and high ECE
        for i in range(controller.min_samples_for_adaptation + controller.adaptation_interval):
            controller.update(
                logit_up=1.0,
                logit_down=0.0,
                y_up=0.0,  # Wrong direction to increase ECE
                weight=1.0,
                flip=False,
                current_ece=0.40,  # High ECE (> bad_threshold=0.25)
            )

        # lr should have increased
        self.assertGreater(controller.lr_current, initial_lr)
        self.assertGreater(controller.ece_worsening_streak, 0)
        self.assertEqual(controller.ece_improving_streak, 0)

    def test_lr_adaptation_low_ece(self) -> None:
        """Test lr decreases when ECE is low."""
        training = TrainingConfig()
        controller = AdaptiveCalibController.create(
            model_type="TEST", training_config=training, per_model_config=None
        )

        # Start with higher lr
        controller.lr_current = 0.01
        initial_lr = controller.lr_current

        # Fill window with enough samples and low ECE
        for i in range(controller.min_samples_for_adaptation + controller.adaptation_interval):
            controller.update(
                logit_up=1.0,
                logit_down=0.0,
                y_up=1.0,
                weight=1.0,
                flip=False,
                current_ece=0.05,  # Low ECE (< good_threshold=0.10)
            )

        # lr should have decreased
        self.assertLess(controller.lr_current, initial_lr)
        self.assertEqual(controller.ece_worsening_streak, 0)
        self.assertGreater(controller.ece_improving_streak, 0)

    def test_lr_clamping(self) -> None:
        """Test lr is clamped to min/max bounds during adaptation."""
        training = TrainingConfig()
        controller = AdaptiveCalibController.create(
            model_type="TEST", training_config=training, per_model_config=None
        )

        # Set lr close to max and trigger increase with high ECE
        controller.lr_current = controller.lr_max * 0.8
        for i in range(controller.min_samples_for_adaptation + controller.adaptation_interval):
            controller.update(
                logit_up=1.0, logit_down=0.0, y_up=0.0, weight=1.0, flip=False, current_ece=0.45
            )

        # lr should be clamped at max
        self.assertLessEqual(controller.lr_current, controller.lr_max)

        # Set lr close to min and trigger decrease with low ECE
        controller.lr_current = controller.lr_min * 1.5
        controller.update_count = 0
        controller.last_adaptation_step = 0
        controller.ece_window.clear()
        for i in range(controller.min_samples_for_adaptation + controller.adaptation_interval):
            controller.update(
                logit_up=1.0, logit_down=0.0, y_up=1.0, weight=1.0, flip=False, current_ece=0.02
            )

        # lr should be clamped at min
        self.assertGreaterEqual(controller.lr_current, controller.lr_min)

    def test_adaptation_interval(self) -> None:
        """Test adaptation only occurs at specified intervals."""
        training = TrainingConfig(calib_adaptation_interval=10, calib_min_samples_for_adaptation=15)
        controller = AdaptiveCalibController.create(
            model_type="TEST", training_config=training, per_model_config=None
        )

        initial_lr = controller.lr_current

        # Fill with enough ECE samples first
        for i in range(15):
            controller.update(
                logit_up=1.0, logit_down=0.0, y_up=0.0, weight=1.0, flip=False, current_ece=0.40
            )

        # Reset to check interval behavior
        controller.last_adaptation_step = controller.update_count
        lr_after_first_adapt = controller.lr_current

        # Fill with high ECE but not enough updates to next interval
        for i in range(9):  # One less than interval
            controller.update(
                logit_up=1.0, logit_down=0.0, y_up=0.0, weight=1.0, flip=False, current_ece=0.40
            )

        # lr should not have adapted yet
        self.assertEqual(controller.lr_current, lr_after_first_adapt)

        # One more update should trigger adaptation
        controller.update(
            logit_up=1.0, logit_down=0.0, y_up=0.0, weight=1.0, flip=False, current_ece=0.40
        )

        # Now lr should have increased again
        self.assertGreater(controller.lr_current, lr_after_first_adapt)

    def test_min_samples_for_adaptation(self) -> None:
        """Test adaptation requires minimum ECE samples."""
        training = TrainingConfig(
            calib_min_samples_for_adaptation=50, calib_adaptation_interval=10
        )
        controller = AdaptiveCalibController.create(
            model_type="TEST", training_config=training, per_model_config=None
        )

        initial_lr = controller.lr_current

        # Reach adaptation interval but not enough ECE samples
        for i in range(10):
            controller.update(
                logit_up=1.0,
                logit_down=0.0,
                y_up=0.0,
                weight=1.0,
                flip=False,
                current_ece=0.40 if i < 10 else None,  # Only 10 ECE samples
            )

        # lr should not have adapted (need 50 samples)
        self.assertEqual(controller.lr_current, initial_lr)

    def test_state_serialization(self) -> None:
        """Test to_dict/load_state roundtrip."""
        training = TrainingConfig()
        controller = AdaptiveCalibController.create(
            model_type="TEST", training_config=training, per_model_config=None
        )

        # Modify state
        controller.state.a = 0.75
        controller.state.b = 0.15
        controller.state.n = 100
        controller.lr_current = 0.008
        controller.update_count = 100
        controller.ece_improving_streak = 3
        controller.ece_window.append(0.12)
        controller.ece_window.append(0.10)

        # Serialize
        data = controller.to_dict()

        # Create new controller and load state
        controller2 = AdaptiveCalibController.create(
            model_type="TEST2", training_config=training, per_model_config=None
        )
        controller2.load_state(data)

        # Verify state restored
        self.assertEqual(controller2.state.a, 0.75)
        self.assertEqual(controller2.state.b, 0.15)
        self.assertEqual(controller2.state.n, 100)
        self.assertEqual(controller2.lr_current, 0.008)
        self.assertEqual(controller2.update_count, 100)
        self.assertEqual(controller2.ece_improving_streak, 3)
        self.assertEqual(len(controller2.ece_window), 2)
        self.assertEqual(controller2.ece_window[0], 0.12)
        self.assertEqual(controller2.ece_window[1], 0.10)

    def test_legacy_state_loading(self) -> None:
        """Test backward compatibility with old calibration_affine format."""
        training = TrainingConfig()
        controller = AdaptiveCalibController.create(
            model_type="TEST", training_config=training, per_model_config=None
        )

        # Legacy format (old calibration_affine)
        legacy_data = {"a": 0.65, "b": 0.20, "n": 50}

        controller.load_legacy_state(legacy_data)

        # Verify state loaded
        self.assertEqual(controller.state.a, 0.65)
        self.assertEqual(controller.state.b, 0.20)
        self.assertEqual(controller.state.n, 50)

        # Other fields should remain at defaults
        self.assertEqual(controller.lr_current, training.calib_lr)


if __name__ == "__main__":
    unittest.main()
