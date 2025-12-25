import unittest

from src.calibration import (
    AffineLogitCalibConfig,
    AffineLogitCalibState,
    _clamp,
    update_affine_calib,
)


class CalibrationTests(unittest.TestCase):
    def test_clamp_bounds(self) -> None:
        self.assertEqual(_clamp(-1.0, 0.0, 1.0), 0.0)
        self.assertEqual(_clamp(2.0, 0.0, 1.0), 1.0)
        self.assertEqual(_clamp(0.4, 0.0, 1.0), 0.4)

    def test_update_moves_upward(self) -> None:
        state = AffineLogitCalibState(a=0.5, b=0.0, n=0)
        cfg = AffineLogitCalibConfig(
            lr=0.1,
            a_min=0.05,
            a_max=2.0,
            b_min=-1.0,
            b_max=1.0,
            l2_a=0.0,
            l2_b=0.0,
            a_anchor=0.5,
            b_anchor=0.0,
        )
        updated = update_affine_calib(
            state,
            logit_up=2.0,
            logit_down=0.0,
            y_up=1.0,
            weight=1.0,
            cfg=cfg,
        )
        self.assertGreater(updated.a, state.a)
        self.assertGreater(updated.b, state.b)

    def test_update_moves_downward(self) -> None:
        state = AffineLogitCalibState(a=0.5, b=0.1, n=0)
        cfg = AffineLogitCalibConfig(
            lr=0.1,
            a_min=0.05,
            a_max=2.0,
            b_min=-1.0,
            b_max=1.0,
            l2_a=0.0,
            l2_b=0.0,
            a_anchor=0.5,
            b_anchor=0.0,
        )
        updated = update_affine_calib(
            state,
            logit_up=2.0,
            logit_down=0.0,
            y_up=0.0,
            weight=1.0,
            cfg=cfg,
        )
        self.assertLess(updated.a, state.a)
        self.assertLess(updated.b, state.b)


if __name__ == "__main__":
    unittest.main()
