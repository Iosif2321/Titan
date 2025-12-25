import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.config import DecisionConfig, ModelInitConfig, PatternConfig, TrainingConfig
from src.engine import build_context_key, build_decision_key
from src.features import FeatureBundle, FeatureSpec
from src.pattern_store import PatternStore
from src.state_store import ModelStateStore
from src.types import Candle, Direction
from src.engine import ModelRunner


class DummyFeatureBuilder:
    def __init__(self) -> None:
        self.spec = FeatureSpec(
            input_size=1,
            feature_names=["x"],
            lookback=1,
            required_lookback=1,
            schema_version="test",
            model_type="TRENDVIC",
        )

    def build(self, candles):
        return FeatureBundle(
            values=np.array([0.0], dtype=np.float32),
            context_coarse={"a": "1"},
            context_fine={"a": "1"},
        )


class AntiPatternAbstainTests(unittest.TestCase):
    def test_anti_pattern_abstain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pattern_db = Path(tmp) / "patterns.db"
            state_db = Path(tmp) / "state.db"
            config = PatternConfig(db_path=pattern_db)
            store = PatternStore(str(pattern_db), config)

            now = 1_700_000_000_000
            context_key = build_context_key("1", "TRENDVIC", "fine", {"a": "1"})
            decision_key = build_decision_key(context_key, Direction.UP)
            store.conn.execute(
                """
                INSERT INTO pattern_stats (
                  pattern_key, tf, model_id, kind, count, ema_win, ema_reward,
                  ema_p_up, ema_p_down, ema_p_flat, streak_bad, last_seen_ts
                ) VALUES (?, ?, ?, 'decision', ?, ?, ?, NULL, NULL, NULL, ?, ?)
                """,
                (decision_key, "1", "TRENDVIC", 20, 0.3, -0.5, 5, now),
            )
            store.conn.commit()

            state_store = ModelStateStore(str(state_db))
            runner = ModelRunner(
                tf="1",
                model_type="TRENDVIC",
                feature_builder=DummyFeatureBuilder(),
                model_init=ModelInitConfig(),
                training=TrainingConfig(),
                decision=DecisionConfig(),
                lr_base=0.001,
                pattern_store=store,
                state_store=state_store,
            )
            runner.model.predict = lambda features, use_ema=True: (0.0, 0.0, 0.53, 0.47)

            candle = Candle(
                start_ts=now,
                end_ts=now + 60_000 - 1,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=10.0,
                confirmed=True,
                tf="1",
            )
            prediction = runner.predict(candle, [candle], now_ts=now)
            self.assertIsNotNone(prediction)
            self.assertEqual(prediction.direction, Direction.FLAT)
            self.assertIn("anti_pattern_abstain", prediction.notes)
            store.close()
            state_store.close()


if __name__ == "__main__":
    unittest.main()
