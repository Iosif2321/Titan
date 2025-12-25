import tempfile
import unittest
from pathlib import Path

from src.config import PatternConfig
from src.pattern_store import PatternStore
from src.types import Direction
from src.utils import now_ms


class PatternStoreRoundTripTests(unittest.TestCase):
    def test_roundtrip_and_maintenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "patterns.db"
            cfg = PatternConfig(db_path=db_path, event_ttl_days=1, max_events=5, maintenance_seconds=1)
            store = PatternStore(str(db_path), cfg)
            cur = store.conn.execute("PRAGMA journal_mode")
            mode = cur.fetchone()[0]
            self.assertEqual(str(mode).lower(), "wal")

            now = now_ms()
            context_key = "1:TRENDVIC:coarse:a=1"
            decision_key = f"{context_key}|PRED=UP"
            store.update_patterns(
                tf="1",
                model_id="TRENDVIC",
                context_key=context_key,
                decision_key=decision_key,
                pred_dir=Direction.UP,
                fact_dir=Direction.UP,
                reward=1.0,
                ts=now,
            )
            stat = store.get_stat(decision_key)
            self.assertIsNotNone(stat)
            self.assertGreaterEqual(stat.count, 1)

            old_ts = now - 2 * 86400 * 1000
            for idx in range(7):
                store.record_event(
                    kind="decision",
                    pattern_key=decision_key,
                    ts=old_ts + idx,
                    tf="1",
                    model_id="TRENDVIC",
                    candle_ts=old_ts + idx,
                    target_ts=old_ts + idx + 60_000,
                    close_prev=100.0,
                    close_curr=101.0,
                    pred_dir=Direction.UP,
                    pred_conf=0.7,
                    fact_dir=Direction.UP,
                    reward=1.0,
                    ret_bps=10.0,
                    lr_eff=0.001,
                    anchor_lambda_eff=0.0001,
                )

            store.maintenance(now_ts=now)
            cur = store.conn.execute("SELECT COUNT(*) FROM pattern_events")
            remaining = int(cur.fetchone()[0])
            self.assertLess(remaining, 7)
            store.close()


if __name__ == "__main__":
    unittest.main()
