"""Pattern encoder with dual SimHash (market + context)."""

from __future__ import annotations

import hashlib

import numpy as np

from btc_oracle.core.config import PatternsConfig
from btc_oracle.core.types import Features, PatternKey


_CONTEXT_KEYS: tuple[str, ...] = (
    "p_up",
    "p_down",
    "p_flat",
    "u_dir",
    "u_mag",
    "consensus",
    "disagreement",
)


class PatternEncoder:
    """Encode features and model context into PatternKey."""

    def __init__(self, config: PatternsConfig):
        self.bits = config.hash.bits
        self.seed = config.hash.seed
        self._proj_cache: dict[tuple[int, int], np.ndarray] = {}

    def encode(
        self,
        features: Features,
        horizon: int,
        timeframe: str = "1m",
        context: object | dict | None = None,
    ) -> PatternKey:
        market_vec = self._normalize(features.vector)
        context_vec = self._build_context_vector(context)

        market_hash = self._simhash(market_vec, seed_offset=0)
        if context_vec.size:
            combined = np.concatenate([market_vec, context_vec])
        else:
            combined = market_vec
        context_hash = self._simhash(combined, seed_offset=101)

        regime_key = self._regime_key(features, horizon)
        pattern_id = self._pattern_id(market_hash, context_hash)

        return PatternKey(
            timeframe=timeframe,
            horizon=horizon,
            pattern_id=pattern_id,
            market_hash=market_hash,
            context_hash=context_hash,
            regime_key=regime_key,
        )

    def _pattern_id(self, market_hash: bytes, context_hash: bytes) -> int:
        digest = hashlib.sha256(market_hash + context_hash).digest()[:16]
        return int.from_bytes(digest, byteorder="big", signed=False)

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        v = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        norm = np.linalg.norm(v)
        if norm <= 1e-12:
            return v
        return v / norm

    def _build_context_vector(self, context: object | dict | None) -> np.ndarray:
        if context is None:
            return np.zeros(0, dtype=np.float32)

        if isinstance(context, dict):
            values = [float(context.get(k, 0.0)) for k in _CONTEXT_KEYS]
            return self._quantize(values)

        values = [float(getattr(context, k, 0.0)) for k in _CONTEXT_KEYS]
        return self._quantize(values)

    def _quantize(self, values: list[float]) -> np.ndarray:
        """Снизить чувствительность ключа к шуму вероятностей."""
        clipped = [max(0.0, min(1.0, v)) for v in values]
        rounded = [round(v, 2) for v in clipped]
        return np.array(rounded, dtype=np.float32)

    def _regime_key(self, features: Features, horizon: int) -> int:
        atr = float(features.meta.get("atr", 0.0))
        close = float(features.meta.get("close", 1.0))
        atr_rel = atr / close if close > 0 else 0.0

        if atr_rel < 0.01:
            vol_regime = 0
        elif atr_rel < 0.03:
            vol_regime = 1
        else:
            vol_regime = 2

        hour = features.ts.hour
        if 0 <= hour < 8:
            session = 0
        elif 8 <= hour < 16:
            session = 1
        else:
            session = 2

        if horizon <= 5:
            horizon_bucket = 0
        elif horizon <= 15:
            horizon_bucket = 1
        elif horizon <= 60:
            horizon_bucket = 2
        else:
            horizon_bucket = 3

        return (vol_regime << 4) | (session << 2) | horizon_bucket

    def _simhash(self, vector: np.ndarray, seed_offset: int) -> bytes:
        proj = self._get_projection(vector.size, seed_offset)
        dots = proj @ vector
        bits = dots >= 0
        packed = np.packbits(bits.astype(np.uint8), bitorder="big")
        return packed.tobytes()

    def _get_projection(self, dim: int, seed_offset: int) -> np.ndarray:
        key = (dim, seed_offset)
        if key in self._proj_cache:
            return self._proj_cache[key]

        rng = np.random.default_rng(self.seed + seed_offset + dim * 31)
        proj = rng.standard_normal((self.bits, dim)).astype(np.float32)
        self._proj_cache[key] = proj
        return proj
