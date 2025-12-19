"""Pattern store with LMDB + binary logs (ring buffer, extremes)."""

from __future__ import annotations

import os
import pickle
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import lmdb
import numpy as np

from btc_oracle.core.config import PatternsConfig
from btc_oracle.core.log import get_logger
from btc_oracle.core.types import Features, Label, MemoryOpinion, PatternKey
from btc_oracle.patterns.stats import PatternStats

logger = get_logger(__name__)


_RECORD_STRUCT = struct.Struct("<QHHBBBBeeeee")
_RECORD_SIZE = _RECORD_STRUCT.size
_EXTREME_HEADER = struct.Struct("<QIQI")
_FLOAT16_MAX = float(np.finfo(np.float16).max)

_FLAG_MISDIRECTION = 1 << 0
_FLAG_CONFLICT = 1 << 1
_FLAG_UNCERTAIN = 1 << 2
_FLAG_FLAT = 1 << 3


@dataclass
class DecisionRecord:
    ts_ms: int
    tf_id: int
    model_id: int
    head_id: int
    pred_class: int
    actual_class: int
    flags: int
    p_up: float
    p_down: float
    confidence: float
    reward: float
    outcome_margin: float

    def pack(self) -> bytes:
        p_up = _clamp_float16(self.p_up)
        p_down = _clamp_float16(self.p_down)
        confidence = _clamp_float16(self.confidence)
        reward = _clamp_float16(self.reward)
        outcome_margin = _clamp_float16(self.outcome_margin)
        return _RECORD_STRUCT.pack(
            self.ts_ms,
            self.tf_id,
            self.model_id,
            self.head_id,
            self.pred_class,
            self.actual_class,
            self.flags,
            p_up,
            p_down,
            confidence,
            reward,
            outcome_margin,
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "DecisionRecord":
        values = _RECORD_STRUCT.unpack(payload)
        return cls(
            ts_ms=values[0],
            tf_id=values[1],
            model_id=values[2],
            head_id=values[3],
            pred_class=values[4],
            actual_class=values[5],
            flags=values[6],
            p_up=values[7],
            p_down=values[8],
            confidence=values[9],
            reward=values[10],
            outcome_margin=values[11],
        )


@dataclass
class PatternMeta:
    pattern_id: int
    timeframe: str
    horizon: int
    regime_key: int
    market_hash: bytes
    context_hash: bytes
    status: str = "active"
    contrarian: bool = False
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    last_seen_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    stats: PatternStats | None = None
    ring_offset: Optional[int] = None
    ring_head: int = 0
    ring_count: int = 0
    ring_capacity: int = 0
    agg_count: int = 0
    sum_reward: float = 0.0
    sum_conf: float = 0.0
    sum_conf2: float = 0.0
    sum_margin: float = 0.0
    correct_count: int = 0
    wrong_count: int = 0
    misdirection_count: int = 0
    confident_wrong_count: int = 0
    mean_vector: Optional[bytes] = None
    mean_dim: int = 0
    mean_count: int = 0
    extremes_best: list[tuple[float, int, int]] = field(default_factory=list)
    extremes_worst: list[tuple[float, int, int]] = field(default_factory=list)

    def to_bytes(self) -> bytes:
        return pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_bytes(cls, payload: bytes) -> "PatternMeta":
        return pickle.loads(payload)


class PatternStore:
    """Pattern store with LMDB and binary logs."""

    def __init__(self, config: PatternsConfig):
        self.config = config
        self._lock = threading.Lock()

        lmdb_path = Path(config.storage.lmdb_path)
        lmdb_path.parent.mkdir(parents=True, exist_ok=True)
        map_size = int(config.storage.disk_cap_gb * 1024**3 * 1.2)
        self._env = lmdb.open(
            str(lmdb_path),
            map_size=max(map_size, 256 * 1024 * 1024),
            max_dbs=4,
            lock=True,
            subdir=True,
        )
        self._meta_db = self._env.open_db(b"patterns_meta")
        self._lsh_market_db = self._env.open_db(b"lsh_market")
        self._lsh_context_db = self._env.open_db(b"lsh_context")
        self._free_segments_db = self._env.open_db(b"free_segments")

        self._log_dir = Path(config.storage.log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._ring_path = self._log_dir / "ring_buffer.bin"
        self._extreme_path = self._log_dir / "extremes.bin"
        self._blob_path = self._log_dir / "feature_blobs.bin"

        for path in (self._ring_path, self._extreme_path, self._blob_path):
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"")

        self._ring_file = open(self._ring_path, "r+b")
        self._extreme_file = open(self._extreme_path, "r+b")
        self._blob_file = open(self._blob_path, "r+b")

        self._disk_cap_bytes = int(config.storage.disk_cap_gb * 1024**3)

        if config.hash.band_bits * config.hash.bands != config.hash.bits:
            logger.warning(
                "LSH band config mismatch",
                bands=config.hash.bands,
                band_bits=config.hash.band_bits,
                hash_bits=config.hash.bits,
            )

    def close(self) -> None:
        self._ring_file.close()
        self._extreme_file.close()
        self._blob_file.close()
        self._env.close()

    def get_opinion(
        self,
        pattern_key: PatternKey,
        min_samples: int = 3,
        features: Optional[Features] = None,
    ) -> Optional[MemoryOpinion]:
        meta = self._find_best(pattern_key, features)
        if meta is None:
            return None

        stats = meta.stats
        if stats is None or stats.is_in_cooldown or stats.n < min_samples:
            return None

        p_up = stats.p_up_mem
        p_down = stats.p_down_mem
        if meta.contrarian:
            p_up, p_down = p_down, p_up

        return MemoryOpinion(
            p_up_mem=p_up,
            p_down_mem=p_down,
            p_flat_mem=stats.p_flat_mem,
            credibility=stats.credibility,
            n=stats.n,
            pattern_id=meta.pattern_id,
        )

    def get_top_patterns(self, limit: int = 10, sort_by: str = "n") -> list[dict]:
        patterns: list[dict] = []
        with self._env.begin(write=False) as txn:
            cursor = txn.cursor(db=self._meta_db)
            for _, payload in cursor:
                meta = PatternMeta.from_bytes(payload)
                stats = meta.stats
                if stats is None:
                    continue

                n = int(stats.n)
                correct = int(meta.correct_count)
                wrong = int(meta.wrong_count)
                n_dir = correct + wrong
                dir_acc = correct / n_dir if n_dir > 0 else None
                misdirection = meta.misdirection_count / n_dir if n_dir > 0 else 0.0

                patterns.append(
                    {
                        "pattern_id": format(meta.pattern_id, "032x"),
                        "timeframe": meta.timeframe,
                        "horizon": int(meta.horizon),
                        "status": meta.status,
                        "contrarian": bool(meta.contrarian),
                        "n": n,
                        "credibility": float(stats.credibility),
                        "p_up": float(stats.p_up_mem),
                        "p_down": float(stats.p_down_mem),
                        "p_flat": float(stats.p_flat_mem),
                        "dir_accuracy": dir_acc,
                        "misdirection_rate": float(misdirection),
                    }
                )

        if sort_by == "credibility":
            key_fn = lambda item: item["credibility"]
        elif sort_by == "dir_accuracy":
            key_fn = lambda item: item["dir_accuracy"] or 0.0
        else:
            key_fn = lambda item: item["n"]

        patterns.sort(key=key_fn, reverse=True)
        return patterns[: max(1, int(limit))]

    def get_patterns_summary(self) -> dict:
        summary = {
            "total": 0,
            "active": 0,
            "archived": 0,
            "probation": 0,
            "purged": 0,
            "contrarian": 0,
            "total_samples": 0,
        }
        with self._env.begin(write=False) as txn:
            cursor = txn.cursor(db=self._meta_db)
            for _, payload in cursor:
                meta = PatternMeta.from_bytes(payload)
                summary["total"] += 1
                if meta.status in summary:
                    summary[meta.status] += 1
                if meta.contrarian:
                    summary["contrarian"] += 1
                if meta.stats is not None:
                    summary["total_samples"] += int(meta.stats.n)
        return summary

    def list_patterns(
        self,
        *,
        offset: int = 0,
        limit: int = 200,
        sort_by: str = "n",
        order: str = "desc",
        status: Optional[str] = None,
        timeframe: Optional[str] = None,
        horizon: Optional[int] = None,
        contrarian: Optional[bool] = None,
        search: Optional[str] = None,
    ) -> dict:
        patterns: list[dict] = []
        total = 0
        search_norm = search.lower() if search else None
        with self._env.begin(write=False) as txn:
            cursor = txn.cursor(db=self._meta_db)
            for _, payload in cursor:
                meta = PatternMeta.from_bytes(payload)
                if status and meta.status != status:
                    continue
                if timeframe and meta.timeframe != timeframe:
                    continue
                if horizon is not None and meta.horizon != int(horizon):
                    continue
                if contrarian is not None and bool(meta.contrarian) != bool(contrarian):
                    continue

                pid_hex = format(meta.pattern_id, "032x")
                if search_norm and search_norm not in pid_hex:
                    continue

                total += 1
                stats = meta.stats
                n = int(stats.n) if stats else 0
                correct = int(meta.correct_count)
                wrong = int(meta.wrong_count)
                n_dir = correct + wrong
                dir_acc = correct / n_dir if n_dir > 0 else None
                misdirection = meta.misdirection_count / n_dir if n_dir > 0 else 0.0

                patterns.append(
                    {
                        "pattern_id": pid_hex,
                        "timeframe": meta.timeframe,
                        "horizon": int(meta.horizon),
                        "status": meta.status,
                        "contrarian": bool(meta.contrarian),
                        "n": n,
                        "credibility": float(stats.credibility) if stats else 0.0,
                        "p_up": float(stats.p_up_mem) if stats else 0.0,
                        "p_down": float(stats.p_down_mem) if stats else 0.0,
                        "p_flat": float(stats.p_flat_mem) if stats else 0.0,
                        "dir_accuracy": dir_acc,
                        "misdirection_rate": float(misdirection),
                        "last_seen_ms": int(meta.last_seen_ms),
                        "created_at_ms": int(meta.created_at_ms),
                        "ring_count": int(meta.ring_count),
                        "ring_capacity": int(meta.ring_capacity),
                        "agg_count": int(meta.agg_count),
                        "extremes_best": len(meta.extremes_best),
                        "extremes_worst": len(meta.extremes_worst),
                    }
                )

        reverse = order.lower() != "asc"
        sort_key = {
            "n": lambda item: item["n"],
            "credibility": lambda item: item["credibility"],
            "dir_accuracy": lambda item: item["dir_accuracy"] or 0.0,
            "misdirection": lambda item: item["misdirection_rate"],
            "last_seen": lambda item: item["last_seen_ms"],
            "created_at": lambda item: item["created_at_ms"],
            "ring_count": lambda item: item["ring_count"],
        }.get(sort_by, lambda item: item["n"])
        patterns.sort(key=sort_key, reverse=reverse)
        start = max(0, int(offset))
        end = start + max(1, int(limit))
        return {
            "total": total,
            "offset": start,
            "limit": end - start,
            "patterns": patterns[start:end],
        }

    def get_pattern_detail(
        self,
        pattern_id: int,
        *,
        ring_limit: int = 200,
        extremes_limit: int = 20,
        include_vectors: bool = False,
    ) -> Optional[dict]:
        with self._lock:
            meta = self._get_meta(pattern_id)
            if meta is None:
                return None

            stats = meta.stats
            stats_payload = None
            if stats is not None:
                stats_payload = stats.to_dict()
                stats_payload.update(
                    {
                        "p_up_mem": float(stats.p_up_mem),
                        "p_down_mem": float(stats.p_down_mem),
                        "p_flat_mem": float(stats.p_flat_mem),
                        "credibility": float(stats.credibility),
                        "is_in_cooldown": bool(stats.is_in_cooldown),
                    }
                )

            ring_records = []
            ring_capacity = meta.ring_capacity or self.config.limits.ring_buffer
            if meta.ring_offset is not None and meta.ring_count > 0 and ring_capacity > 0:
                count = min(int(meta.ring_count), int(ring_limit))
                for i in range(count):
                    idx = (meta.ring_head - 1 - i) % ring_capacity
                    pos = meta.ring_offset + idx * _RECORD_SIZE
                    record = self._read_record(pos)
                    if record:
                        ring_records.append(self._record_to_dict(record))

            extremes_best = self._load_extremes(meta.extremes_best, extremes_limit, include_vectors)
            extremes_worst = self._load_extremes(meta.extremes_worst, extremes_limit, include_vectors)

            mean_vector = None
            if include_vectors and meta.mean_vector is not None and meta.mean_dim > 0:
                mean_vector = np.frombuffer(meta.mean_vector, dtype=np.float32).tolist()

            return {
                "pattern_id": format(meta.pattern_id, "032x"),
                "meta": {
                    "timeframe": meta.timeframe,
                    "horizon": int(meta.horizon),
                    "status": meta.status,
                    "contrarian": bool(meta.contrarian),
                    "created_at_ms": int(meta.created_at_ms),
                    "last_seen_ms": int(meta.last_seen_ms),
                    "regime_key": int(meta.regime_key),
                    "market_hash": meta.market_hash.hex(),
                    "context_hash": meta.context_hash.hex(),
                    "ring_count": int(meta.ring_count),
                    "ring_capacity": int(ring_capacity),
                    "agg_count": int(meta.agg_count),
                    "sum_reward": float(meta.sum_reward),
                    "sum_conf": float(meta.sum_conf),
                    "sum_conf2": float(meta.sum_conf2),
                    "sum_margin": float(meta.sum_margin),
                    "correct_count": int(meta.correct_count),
                    "wrong_count": int(meta.wrong_count),
                    "misdirection_count": int(meta.misdirection_count),
                    "confident_wrong_count": int(meta.confident_wrong_count),
                    "mean_dim": int(meta.mean_dim),
                    "mean_count": int(meta.mean_count),
                    "mean_vector": mean_vector,
                },
                "stats": stats_payload,
                "ring_records": ring_records,
                "extremes_best": extremes_best,
                "extremes_worst": extremes_worst,
            }

    def _record_to_dict(self, record: DecisionRecord) -> dict:
        pred = _int_to_label(record.pred_class)
        actual = _int_to_label(record.actual_class)
        return {
            "ts_ms": int(record.ts_ms),
            "tf_id": int(record.tf_id),
            "model_id": int(record.model_id),
            "head_id": int(record.head_id),
            "pred_label": pred.value if pred else None,
            "actual_label": actual.value if actual else None,
            "flags": int(record.flags),
            "p_up": float(record.p_up),
            "p_down": float(record.p_down),
            "confidence": float(record.confidence),
            "reward": float(record.reward),
            "outcome_margin": float(record.outcome_margin),
        }

    def _load_extremes(
        self,
        items: list[tuple[float, int, int]],
        limit: int,
        include_vectors: bool,
    ) -> list[dict]:
        results: list[dict] = []
        for reward, offset, length in items[: max(0, int(limit))]:
            payload = self._read_extreme(offset, length)
            if payload is None:
                continue
            feature_offset, feature_len, candle_offset, candle_len, record = payload
            entry = {
                "reward": float(reward),
                "offset": int(offset),
                "length": int(length),
                "record": self._record_to_dict(record),
            }
            if include_vectors:
                if feature_len > 0:
                    blob = self._read_blob(feature_offset, feature_len)
                    entry["feature_vector"] = (
                        np.frombuffer(blob, dtype=np.float32).tolist() if blob else None
                    )
                if candle_len > 0:
                    blob = self._read_blob(candle_offset, candle_len)
                    entry["candles"] = self._decode_candle_blob(blob) if blob else None
            results.append(entry)
        return results

    def _read_extreme(self, offset: int, length: int) -> Optional[tuple[int, int, int, int, DecisionRecord]]:
        if offset < 0 or length <= 0:
            return None
        self._extreme_file.seek(offset)
        payload = self._extreme_file.read(length)
        min_len = _EXTREME_HEADER.size + _RECORD_SIZE
        if len(payload) < min_len:
            return None
        header = payload[: _EXTREME_HEADER.size]
        feature_offset, feature_len, candle_offset, candle_len = _EXTREME_HEADER.unpack(header)
        record_payload = payload[_EXTREME_HEADER.size : _EXTREME_HEADER.size + _RECORD_SIZE]
        record = DecisionRecord.unpack(record_payload)
        return int(feature_offset), int(feature_len), int(candle_offset), int(candle_len), record

    def _read_blob(self, offset: int, length: int) -> Optional[bytes]:
        if offset < 0 or length <= 0:
            return None
        self._blob_file.seek(offset)
        blob = self._blob_file.read(length)
        return blob if blob else None

    def _decode_candle_blob(self, blob: bytes) -> Optional[dict]:
        if not blob:
            return None
        data = np.frombuffer(blob, dtype=np.float32)
        if data.size < 10:
            return None
        forecast = {
            "open": float(data[0]),
            "high": float(data[1]),
            "low": float(data[2]),
            "close": float(data[3]),
            "volume": float(data[4]),
        }
        truth = {
            "open": float(data[5]),
            "high": float(data[6]),
            "low": float(data[7]),
            "close": float(data[8]),
            "volume": float(data[9]),
        }
        return {"forecast": forecast, "truth": truth}

    def record_decision(
        self,
        pattern_key: PatternKey,
        record: DecisionRecord,
        *,
        features: Optional[Features] = None,
        candle_blob: Optional[bytes] = None,
    ) -> None:
        with self._lock:
            meta = self._get_or_create_meta(pattern_key)
            meta.last_seen_ms = int(time.time() * 1000)

            if meta.stats is None:
                meta.stats = PatternStats(
                    pattern_id=meta.pattern_id,
                    timeframe=meta.timeframe,
                    horizon=meta.horizon,
                    alpha_up=self.config.beta_prior_alpha,
                    beta_down=self.config.beta_prior_beta,
                    alpha_flat=self.config.beta_prior_alpha_flat,
                    beta_not_flat=self.config.beta_prior_beta_flat,
                )

            self._update_stats(meta, record)
            self._update_ring(meta, record)
            self._update_mean_vector(meta, features)
            self._update_extremes(meta, record, features, candle_blob)
            self._maybe_cull(meta)
            self._save_meta(meta)

    def update(self, pattern_key: PatternKey, truth: Label, magnitude: float) -> None:
        record = DecisionRecord(
            ts_ms=int(time.time() * 1000),
            tf_id=pattern_key.horizon,
            model_id=0,
            head_id=0,
            pred_class=_label_to_int(truth),
            actual_class=_label_to_int(truth),
            flags=0,
            p_up=0.0,
            p_down=0.0,
            confidence=0.0,
            reward=0.0,
            outcome_margin=magnitude,
        )
        self.record_decision(pattern_key, record, features=None)

    def record_error(self, pattern_key: PatternKey) -> None:
        with self._lock:
            meta = self._get_or_create_meta(pattern_key)
            if meta.stats is None:
                meta.stats = PatternStats(
                    pattern_id=meta.pattern_id,
                    timeframe=meta.timeframe,
                    horizon=meta.horizon,
                )
            meta.stats.record_error(self.config.cooldown_duration_hours)
            self._save_meta(meta)

    def _get_or_create_meta(self, pattern_key: PatternKey) -> PatternMeta:
        meta = self._get_meta(pattern_key.pattern_id)
        if meta:
            return meta

        ring_offset = self._allocate_ring_segment()
        stats = PatternStats(
            pattern_id=pattern_key.pattern_id,
            timeframe=pattern_key.timeframe,
            horizon=pattern_key.horizon,
            alpha_up=self.config.beta_prior_alpha,
            beta_down=self.config.beta_prior_beta,
            alpha_flat=self.config.beta_prior_alpha_flat,
            beta_not_flat=self.config.beta_prior_beta_flat,
        )

        meta = PatternMeta(
            pattern_id=pattern_key.pattern_id,
            timeframe=pattern_key.timeframe,
            horizon=pattern_key.horizon,
            regime_key=pattern_key.regime_key,
            market_hash=pattern_key.market_hash,
            context_hash=pattern_key.context_hash,
            stats=stats,
            ring_offset=ring_offset,
            ring_head=0,
            ring_count=0,
            ring_capacity=self.config.limits.ring_buffer,
        )
        self._save_meta(meta, add_to_lsh=True)
        return meta

    def _get_meta(self, pattern_id: int) -> Optional[PatternMeta]:
        key = _pattern_id_bytes(pattern_id)
        with self._env.begin(write=False) as txn:
            payload = txn.get(key, db=self._meta_db)
            if payload is None:
                return None
            return PatternMeta.from_bytes(payload)

    def _save_meta(self, meta: PatternMeta, *, add_to_lsh: bool = False) -> None:
        key = _pattern_id_bytes(meta.pattern_id)
        payload = meta.to_bytes()
        with self._env.begin(write=True) as txn:
            txn.put(key, payload, db=self._meta_db)
            if add_to_lsh:
                self._lsh_add(txn, self._lsh_market_db, meta.market_hash, key)
                self._lsh_add(txn, self._lsh_context_db, meta.context_hash, key)

    def _update_stats(self, meta: PatternMeta, record: DecisionRecord) -> None:
        actual = _int_to_label(record.actual_class)
        pred = _int_to_label(record.pred_class)

        if meta.stats is not None and actual in (Label.UP, Label.DOWN, Label.FLAT):
            meta.stats.update(actual, record.outcome_margin, self.config.decay_half_life_hours)

        if actual in (Label.UP, Label.DOWN) and pred in (Label.UP, Label.DOWN):
            if actual == pred:
                meta.correct_count += 1
            else:
                meta.wrong_count += 1
                meta.misdirection_count += 1
                if record.confidence >= 0.6:
                    meta.confident_wrong_count += 1

    def _update_ring(self, meta: PatternMeta, record: DecisionRecord) -> None:
        if meta.ring_offset is None:
            return

        if meta.ring_capacity <= 0:
            meta.ring_capacity = self.config.limits.ring_buffer

        slot = meta.ring_head
        pos = meta.ring_offset + slot * _RECORD_SIZE

        if meta.ring_count >= meta.ring_capacity:
            old = self._read_record(pos)
            if old:
                self._aggregate_record(meta, old)

        self._write_record(pos, record)

        meta.ring_head = (meta.ring_head + 1) % meta.ring_capacity
        meta.ring_count = min(meta.ring_capacity, meta.ring_count + 1)

    def _update_mean_vector(self, meta: PatternMeta, features: Optional[Features]) -> None:
        if features is None:
            return

        vec = np.nan_to_num(features.vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if meta.mean_vector is None:
            meta.mean_vector = vec.tobytes()
            meta.mean_dim = int(vec.size)
            meta.mean_count = 1
            return

        mean = np.frombuffer(meta.mean_vector, dtype=np.float32)
        if mean.size != vec.size:
            meta.mean_vector = vec.tobytes()
            meta.mean_dim = int(vec.size)
            meta.mean_count = 1
            return

        n = meta.mean_count
        mean = (mean * n + vec) / float(n + 1)
        meta.mean_vector = mean.astype(np.float32).tobytes()
        meta.mean_dim = int(mean.size)
        meta.mean_count = n + 1

    def _update_extremes(
        self,
        meta: PatternMeta,
        record: DecisionRecord,
        features: Optional[Features],
        candle_blob: Optional[bytes],
    ) -> None:
        if features is None:
            return

        reward = float(record.reward)
        best = meta.extremes_best
        worst = meta.extremes_worst
        extremes_max = self.config.limits.extremes_max

        should_add_best = len(best) < extremes_max or reward > best[-1][0]
        should_add_worst = len(worst) < extremes_max or reward < worst[-1][0]

        if not should_add_best and not should_add_worst:
            return

        offset, length = self._write_extreme(record, features, candle_blob)

        if should_add_best:
            best.append((reward, offset, length))
            best.sort(key=lambda x: x[0], reverse=True)
            if len(best) > extremes_max:
                best.pop()

        if should_add_worst:
            worst.append((reward, offset, length))
            worst.sort(key=lambda x: x[0])
            if len(worst) > extremes_max:
                worst.pop()

    def _write_extreme(
        self,
        record: DecisionRecord,
        features: Features,
        candle_blob: Optional[bytes],
    ) -> tuple[int, int]:
        feature_offset, feature_len = self._write_blob(features.vector.astype(np.float32).tobytes())
        candle_offset, candle_len = 0, 0
        if candle_blob:
            candle_offset, candle_len = self._write_blob(candle_blob)

        header = _EXTREME_HEADER.pack(feature_offset, feature_len, candle_offset, candle_len)
        payload = header + record.pack()
        offset = self._append(self._extreme_file, payload)
        return offset, len(payload)

    def _write_blob(self, blob: bytes) -> tuple[int, int]:
        return self._append(self._blob_file, blob), len(blob)

    def _aggregate_record(self, meta: PatternMeta, record: DecisionRecord) -> None:
        if meta.agg_count >= self.config.limits.aggregates_max:
            factor = self.config.limits.aggregates_max / (self.config.limits.aggregates_max + 1)
            meta.sum_reward *= factor
            meta.sum_conf *= factor
            meta.sum_conf2 *= factor
            meta.sum_margin *= factor
            meta.agg_count = self.config.limits.aggregates_max
        else:
            meta.agg_count += 1

        meta.sum_reward += record.reward
        meta.sum_conf += record.confidence
        meta.sum_conf2 += record.confidence * record.confidence
        meta.sum_margin += record.outcome_margin

    def _maybe_cull(self, meta: PatternMeta) -> None:
        if self._disk_usage_bytes() <= self._disk_cap_bytes:
            return

        logger.warning("Pattern storage exceeds disk cap, initiating culling")
        self._cull_patterns()

    def _cull_patterns(self) -> None:
        metas = []
        with self._env.begin(write=False) as txn:
            cursor = txn.cursor(db=self._meta_db)
            for key, payload in cursor:
                meta = PatternMeta.from_bytes(payload)
                metas.append(meta)

        weak = []
        for meta in metas:
            if meta.status == "purged":
                continue
            score = self._pattern_score(meta)
            weak.append((score, meta))

        weak.sort(key=lambda x: x[0])

        for _, meta in weak:
            if self._disk_usage_bytes() <= self._disk_cap_bytes:
                break
            if meta.status == "active":
                meta.status = "probation"
            elif meta.status == "probation":
                self._archive_pattern(meta)
            elif meta.status == "archived":
                self._purge_pattern(meta)

    def _archive_pattern(self, meta: PatternMeta) -> None:
        meta.status = "archived"
        if meta.ring_offset is not None:
            self._release_ring_segment(meta.ring_offset)
            meta.ring_offset = None
            meta.ring_count = 0
            meta.ring_head = 0
        self._remove_from_lsh(meta)
        self._save_meta(meta)

    def _purge_pattern(self, meta: PatternMeta) -> None:
        key = _pattern_id_bytes(meta.pattern_id)
        self._remove_from_lsh(meta)
        with self._env.begin(write=True) as txn:
            txn.delete(key, db=self._meta_db)

    def _pattern_score(self, meta: PatternMeta) -> float:
        n_dir = meta.correct_count + meta.wrong_count
        if n_dir <= 0:
            return 0.0
        if n_dir < self.config.culling.n_min:
            return 0.5

        p = meta.correct_count / n_dir
        lb = _lower_bound(p, n_dir)
        p0 = 0.5
        delta = self.config.culling.delta

        if (
            n_dir >= self.config.culling.n_min
            and lb < p0 - delta
            and meta.misdirection_count / max(1, n_dir) > 0.6
        ):
            meta.contrarian = True
        return lb

    def _find_best(self, pattern_key: PatternKey, features: Optional[Features]) -> Optional[PatternMeta]:
        candidates = self._collect_candidates(pattern_key, features)
        if not candidates:
            return None

        best = max(candidates, key=lambda item: item[0])[1]
        return best

    def _collect_candidates(
        self,
        pattern_key: PatternKey,
        features: Optional[Features],
    ) -> list[tuple[float, PatternMeta]]:
        market_ids = self._lsh_lookup(self._lsh_market_db, pattern_key.market_hash)
        context_ids = self._lsh_lookup(self._lsh_context_db, pattern_key.context_hash)
        candidate_ids = market_ids & context_ids

        if not candidate_ids:
            return []

        similarity_min = self.config.search.similarity_min
        candidates: list[tuple[float, PatternMeta]] = []
        for pid in candidate_ids:
            meta = self._get_meta(pid)
            if meta is None or meta.status != "active":
                continue
            if self.config.search.regime_filter and meta.regime_key != pattern_key.regime_key:
                continue

            sim_market = _hash_similarity(meta.market_hash, pattern_key.market_hash, self.config.hash.bits)
            sim_context = _hash_similarity(meta.context_hash, pattern_key.context_hash, self.config.hash.bits)
            sim_hash = 0.5 * (sim_market + sim_context)

            full_score = None
            if features is not None:
                full_score = _full_vector_score(
                    features.vector,
                    meta.mean_vector,
                    meta.mean_dim,
                    self.config.search.full_vector_metric,
                )
                if full_score is not None and full_score < similarity_min:
                    continue

            if full_score is None:
                score = sim_hash
            else:
                score = 0.5 * (sim_hash + full_score)

            candidates.append((score, meta))

        if not candidates:
            return []

        threshold = similarity_min
        if len(candidates) > self.config.search.target_candidates_max:
            threshold = min(0.98, similarity_min + self.config.search.adaptive_step)
        elif len(candidates) < self.config.search.target_candidates_min:
            threshold = max(0.70, similarity_min - self.config.search.adaptive_step)

        filtered = [(score, meta) for score, meta in candidates if score >= threshold]
        if not filtered:
            filtered = candidates

        return filtered

    def _lsh_add(self, txn: lmdb.Transaction, db: lmdb._Database, hash_bytes: bytes, pid_key: bytes) -> None:
        for key in _lsh_keys(hash_bytes, self.config.hash.bands, self.config.hash.band_bits):
            existing = txn.get(key, db=db)
            if existing:
                if pid_key in _split_keys(existing):
                    continue
                new_val = existing + pid_key
            else:
                new_val = pid_key
            txn.put(key, new_val, db=db)

    def _lsh_lookup(self, db: lmdb._Database, hash_bytes: bytes) -> set[int]:
        ids: set[int] = set()
        with self._env.begin(write=False) as txn:
            for key in _lsh_keys(hash_bytes, self.config.hash.bands, self.config.hash.band_bits):
                payload = txn.get(key, db=db)
                if not payload:
                    continue
                for pid_bytes in _split_keys(payload):
                    ids.add(int.from_bytes(pid_bytes, byteorder="big", signed=False))
        return ids

    def _remove_from_lsh(self, meta: PatternMeta) -> None:
        key = _pattern_id_bytes(meta.pattern_id)
        with self._env.begin(write=True) as txn:
            self._lsh_remove(txn, self._lsh_market_db, meta.market_hash, key)
            self._lsh_remove(txn, self._lsh_context_db, meta.context_hash, key)

    def _lsh_remove(self, txn: lmdb.Transaction, db: lmdb._Database, hash_bytes: bytes, pid_key: bytes) -> None:
        for key in _lsh_keys(hash_bytes, self.config.hash.bands, self.config.hash.band_bits):
            payload = txn.get(key, db=db)
            if not payload:
                continue
            remaining = b"".join(k for k in _split_keys(payload) if k != pid_key)
            if remaining:
                txn.put(key, remaining, db=db)
            else:
                txn.delete(key, db=db)

    def _allocate_ring_segment(self) -> Optional[int]:
        segment_bytes = self.config.limits.ring_buffer * _RECORD_SIZE
        with self._env.begin(write=True) as txn:
            free = txn.get(b"free", db=self._free_segments_db)
            if free:
                offsets = _split_offsets(free)
                offset = offsets.pop()
                txn.put(b"free", _join_offsets(offsets), db=self._free_segments_db)
                return offset

        current_size = self._ring_file_size()
        if current_size + segment_bytes > self._disk_cap_bytes:
            self._cull_patterns()
            current_size = self._ring_file_size()
            if current_size + segment_bytes > self._disk_cap_bytes:
                logger.warning("Ring buffer segment allocation blocked by disk cap")
                return None

        self._ring_file.seek(0, os.SEEK_END)
        self._ring_file.write(b"\x00" * segment_bytes)
        self._ring_file.flush()
        return current_size

    def _release_ring_segment(self, offset: int) -> None:
        with self._env.begin(write=True) as txn:
            payload = txn.get(b"free", db=self._free_segments_db)
            offsets = _split_offsets(payload) if payload else []
            offsets.append(offset)
            txn.put(b"free", _join_offsets(offsets), db=self._free_segments_db)

    def _read_record(self, pos: int) -> Optional[DecisionRecord]:
        self._ring_file.seek(pos)
        data = self._ring_file.read(_RECORD_SIZE)
        if len(data) != _RECORD_SIZE:
            return None
        return DecisionRecord.unpack(data)

    def _write_record(self, pos: int, record: DecisionRecord) -> None:
        self._ring_file.seek(pos)
        self._ring_file.write(record.pack())
        self._ring_file.flush()

    def _append(self, file_obj, payload: bytes) -> int:
        file_obj.seek(0, os.SEEK_END)
        offset = file_obj.tell()
        file_obj.write(payload)
        file_obj.flush()
        return offset

    def _ring_file_size(self) -> int:
        self._ring_file.seek(0, os.SEEK_END)
        return self._ring_file.tell()

    def _disk_usage_bytes(self) -> int:
        total = 0
        for path in (self._ring_path, self._extreme_path, self._blob_path):
            if path.exists():
                total += path.stat().st_size
        env_path = self._env.path()
        if isinstance(env_path, bytes):
            env_path = env_path.decode()
        total += _dir_size(Path(env_path))
        return total


def _pattern_id_bytes(pattern_id: int) -> bytes:
    return int(pattern_id).to_bytes(16, byteorder="big", signed=False)


def _split_keys(payload: bytes) -> list[bytes]:
    if not payload:
        return []
    return [payload[i : i + 16] for i in range(0, len(payload), 16)]


def _lsh_keys(hash_bytes: bytes, bands: int, band_bits: int) -> Iterable[bytes]:
    band_bytes = band_bits // 8
    for i in range(bands):
        start = i * band_bytes
        end = start + band_bytes
        band = hash_bytes[start:end]
        yield bytes([i]) + band


def _hash_similarity(a: bytes, b: bytes, bits: int) -> float:
    ai = int.from_bytes(a, byteorder="big", signed=False)
    bi = int.from_bytes(b, byteorder="big", signed=False)
    diff = (ai ^ bi).bit_count()
    return 1.0 - diff / float(bits)


def _label_to_int(label: Label) -> int:
    if label == Label.UP:
        return 1
    if label == Label.DOWN:
        return 2
    if label == Label.FLAT:
        return 3
    return 0


def _int_to_label(value: int) -> Optional[Label]:
    if value == 1:
        return Label.UP
    if value == 2:
        return Label.DOWN
    if value == 3:
        return Label.FLAT
    if value == 4:
        return Label.UNCERTAIN
    return None


def _lower_bound(p: float, n: int) -> float:
    if n <= 0:
        return 0.0
    z = 1.645  # approx 5% quantile
    var = p * (1.0 - p) / float(n + 1)
    return max(0.0, p - z * (var ** 0.5))


def _split_offsets(payload: bytes | None) -> list[int]:
    if not payload:
        return []
    return [int.from_bytes(payload[i : i + 8], "big", signed=False) for i in range(0, len(payload), 8)]


def _join_offsets(offsets: list[int]) -> bytes:
    return b"".join(int(o).to_bytes(8, "big", signed=False) for o in offsets)


def _dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += (Path(root) / name).stat().st_size
    return total


def _clamp_float16(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if value > _FLOAT16_MAX:
        return _FLOAT16_MAX
    if value < -_FLOAT16_MAX:
        return -_FLOAT16_MAX
    return float(value)


def _full_vector_score(
    vector: np.ndarray,
    mean_bytes: Optional[bytes],
    mean_dim: int,
    metric: str,
) -> Optional[float]:
    if mean_bytes is None or mean_dim <= 0:
        return None

    mean = np.frombuffer(mean_bytes, dtype=np.float32)
    if mean.size != mean_dim or mean.size != vector.size:
        return None

    v = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if metric == "cosine":
        denom = float(np.linalg.norm(v) * np.linalg.norm(mean))
        if denom <= 1e-12:
            return None
        cos_sim = float(np.dot(v, mean) / denom)
        return max(0.0, min(1.0, (cos_sim + 1.0) * 0.5))

    if metric in ("l2", "euclidean"):
        dist = float(np.linalg.norm(v - mean))
        return 1.0 / (1.0 + dist)

    return None
