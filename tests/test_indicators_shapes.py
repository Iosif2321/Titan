import unittest

import numpy as np

from src import indicators


class IndicatorShapeTests(unittest.TestCase):
    def test_ema_series_shape(self) -> None:
        values = np.arange(10, dtype=np.float32)
        out = indicators.ema_series(values, period=5)
        self.assertEqual(out.shape, values.shape)

    def test_obv_shape(self) -> None:
        closes = np.array([1, 2, 3, 2, 2], dtype=np.float32)
        volumes = np.array([10, 12, 9, 11, 8], dtype=np.float32)
        out = indicators.obv(closes, volumes)
        self.assertEqual(out.shape, closes.shape)

    def test_returns_bps_shape(self) -> None:
        closes = np.array([1, 2, 4, 3], dtype=np.float32)
        out = indicators.returns_bps(closes)
        self.assertEqual(out.shape[0], closes.shape[0] - 1)

    def test_stochastic_outputs(self) -> None:
        highs = np.linspace(10, 20, 20, dtype=np.float32)
        lows = highs - 1.0
        closes = highs - 0.5
        k, d = indicators.stochastic(highs, lows, closes, 14, 3, 3)
        self.assertTrue(np.isfinite(k))
        self.assertTrue(np.isfinite(d))


if __name__ == "__main__":
    unittest.main()
