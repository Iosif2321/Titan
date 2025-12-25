import tempfile
import unittest
from pathlib import Path

from src.config import DecisionConfig, FeatureConfig, ModelInitConfig, PatternConfig, TrainingConfig
from src.engine import CandleBuffer, ModelRunner
from src.features import FeatureBuilder, MODEL_TREND
from src.pattern_store import PatternStore
from src.state_store import ModelStateStore
from src.types import Candle


class PendingAlignmentTests(unittest.TestCase):
    def test_pred_fact_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pattern_db = Path(tmp) / "patterns.db"
            state_db = Path(tmp) / "state.db"
            feature_config = FeatureConfig(lookback=5)
            feature_config.ma_periods = [2]
            feature_config.macd_fast = 2
            feature_config.macd_slow = 3
            feature_config.macd_signal = 2
            feature_config.vol_z_window = 3

            builder = FeatureBuilder(feature_config, MODEL_TREND)
            pattern_store = PatternStore(str(pattern_db), config=PatternConfig(db_path=pattern_db))
            state_store = ModelStateStore(str(state_db))
            runner = ModelRunner(
                tf="1",
                model_type=MODEL_TREND,
                feature_builder=builder,
                model_init=ModelInitConfig(),
                training=TrainingConfig(flat_bps=0.0),
                decision=DecisionConfig(),
                lr_base=0.001,
                pattern_store=pattern_store,
                state_store=state_store,
            )

            buffer = CandleBuffer(builder.spec.required_lookback)
            candles = []
            for i in range(builder.spec.required_lookback):
                ts = i * 60_000
                candles.append(
                    Candle(
                        start_ts=ts,
                        end_ts=ts + 60_000 - 1,
                        open=100 + i,
                        high=101 + i,
                        low=99 + i,
                        close=100 + i,
                        volume=10 + i,
                        confirmed=True,
                        tf="1",
                    )
                )
            buffer.seed(candles)
            last_candle = candles[-1]
            prediction = runner.predict(last_candle, buffer.values(), now_ts=last_candle.start_ts)
            self.assertIsNotNone(prediction)
            target_ts = last_candle.start_ts + 60_000
            self.assertIn(target_ts, runner.pending)

            next_candle = Candle(
                start_ts=target_ts,
                end_ts=target_ts + 60_000 - 1,
                open=110,
                high=111,
                low=109,
                close=112,
                volume=20,
                confirmed=True,
                tf="1",
            )
            result = runner.on_fact(next_candle, now_ts=target_ts)
            self.assertIsNotNone(result)
            fact, _ = result
            self.assertEqual(fact.prev_ts, last_candle.start_ts)
            self.assertEqual(fact.curr_ts, target_ts)
            self.assertEqual(fact.close_prev, last_candle.close)
            self.assertEqual(fact.close_curr, next_candle.close)
            pattern_store.close()
            state_store.close()


if __name__ == "__main__":
    unittest.main()
