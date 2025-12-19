"""Smoke test for LMDB-backed PatternStore."""

from datetime import datetime

import numpy as np

from btc_oracle.core.config import PatternsConfig, PatternStorageConfig
from btc_oracle.core.types import Features, Label
from btc_oracle.patterns.encoder import PatternEncoder
from btc_oracle.patterns.store import PatternStore


def test_pattern_store_roundtrip(tmp_path):
    storage = PatternStorageConfig(
        lmdb_path=str(tmp_path / "patterns.lmdb"),
        log_dir=str(tmp_path / "logs"),
        disk_cap_gb=0.1,
    )
    patterns_cfg = PatternsConfig(storage=storage)

    encoder = PatternEncoder(patterns_cfg)
    store = PatternStore(patterns_cfg)

    features = Features(
        ts=datetime.utcnow(),
        timeframe="1m",
        vector=np.random.randn(10).astype(np.float32),
        meta={"atr": 1.0, "close": 100.0},
    )

    key = encoder.encode(features, horizon=1, timeframe="1m", context={"p_up": 0.6, "p_down": 0.4})

    store.update(key, Label.UP, magnitude=0.05)
    store.update(key, Label.UP, magnitude=0.04)

    opinion = store.get_opinion(key, min_samples=1)
    assert opinion is not None
    assert opinion.p_up_mem >= 0.5

    store.close()
