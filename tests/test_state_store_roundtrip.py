import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.optimizer import AdamState
from src.state_store import ModelStateStore


class StateStoreRoundTripTests(unittest.TestCase):
    def test_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.db"
            store = ModelStateStore(str(db_path))

            params = {"w": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), "b": np.array([0.1, -0.1], dtype=np.float32)}
            ema = {"w": params["w"] * 0.9, "b": params["b"] * 0.9}
            anchor = {"w": params["w"] * 0.8, "b": params["b"] * 0.8}
            opt_state = AdamState(
                m={"w": np.zeros_like(params["w"]), "b": np.zeros_like(params["b"])},
                v={"w": np.zeros_like(params["w"]), "b": np.zeros_like(params["b"])},
                t=5,
            )
            metrics = {"total": 3, "accuracy": 0.66}

            store.save_state(
                model_id="TRENDVIC",
                tf="1",
                saved_at=123,
                params=params,
                ema_params=ema,
                anchor_params=anchor,
                opt_state=opt_state,
                metrics=metrics,
            )

            loaded = store.load_latest("TRENDVIC", "1")
            self.assertIsNotNone(loaded)
            np.testing.assert_allclose(loaded.params["w"], params["w"])
            np.testing.assert_allclose(loaded.ema_params["b"], ema["b"])
            np.testing.assert_allclose(loaded.anchor_params["w"], anchor["w"])
            self.assertEqual(loaded.opt_state.t, opt_state.t)
            self.assertEqual(loaded.metrics.get("total"), 3)
            store.close()


if __name__ == "__main__":
    unittest.main()
