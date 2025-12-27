from __future__ import annotations

import json
import sqlite3
import zlib
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Optional

import numpy as np

from .optimizer import AdamState


@dataclass
class ModelState:
    model_id: str
    tf: str
    version: int
    saved_at: int
    params: Dict[str, np.ndarray]
    ema_params: Dict[str, np.ndarray]
    anchor_params: Dict[str, np.ndarray]
    opt_state: AdamState
    metrics: Dict[str, object]


class ModelStateStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_state (
              model_id TEXT,
              tf TEXT,
              version INTEGER,
              saved_at INTEGER,
              params_blob BLOB,
              ema_blob BLOB,
              anchor_blob BLOB,
              opt_blob BLOB,
              metrics_json TEXT,
              PRIMARY KEY(model_id, tf, version)
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_model_state_latest ON model_state(model_id, tf, version)"
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS flat_state (
              key TEXT PRIMARY KEY,
              json TEXT,
              updated_at INTEGER
            )
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def save_state(
        self,
        model_id: str,
        tf: str,
        saved_at: int,
        params: Dict[str, np.ndarray],
        ema_params: Dict[str, np.ndarray],
        anchor_params: Dict[str, np.ndarray],
        opt_state: AdamState,
        metrics: Dict[str, object],
    ) -> int:
        cur = self.conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM model_state WHERE model_id = ? AND tf = ?",
            (model_id, tf),
        )
        next_version = int(cur.fetchone()[0]) + 1
        payload_params = _pack_arrays(params)
        payload_ema = _pack_arrays(ema_params)
        payload_anchor = _pack_arrays(anchor_params)
        payload_opt = _pack_opt_state(opt_state)
        metrics_json = json.dumps(metrics, separators=(",", ":"))

        self.conn.execute("BEGIN")
        self.conn.execute(
            """
            INSERT INTO model_state (
              model_id, tf, version, saved_at,
              params_blob, ema_blob, anchor_blob, opt_blob, metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id,
                tf,
                next_version,
                saved_at,
                payload_params,
                payload_ema,
                payload_anchor,
                payload_opt,
                metrics_json,
            ),
        )
        self.conn.execute("COMMIT")
        return next_version

    def load_latest(self, model_id: str, tf: str) -> Optional[ModelState]:
        cur = self.conn.execute(
            """
            SELECT * FROM model_state
            WHERE model_id = ? AND tf = ?
            ORDER BY version DESC
            LIMIT 1
            """,
            (model_id, tf),
        )
        row = cur.fetchone()
        if row is None:
            return None
        params = _unpack_arrays(row["params_blob"])
        ema_params = _unpack_arrays(row["ema_blob"])
        anchor_params = _unpack_arrays(row["anchor_blob"])
        opt_state = _unpack_opt_state(row["opt_blob"])
        metrics_json = row["metrics_json"] or "{}"
        metrics = json.loads(metrics_json)
        return ModelState(
            model_id=row["model_id"],
            tf=row["tf"],
            version=int(row["version"]),
            saved_at=int(row["saved_at"]),
            params=params,
            ema_params=ema_params,
            anchor_params=anchor_params,
            opt_state=opt_state,
            metrics=metrics,
        )

    def save_flat_state(self, key: str, payload: Dict[str, object], updated_at: int) -> None:
        payload_json = json.dumps(payload, separators=(",", ":"))
        self.conn.execute("BEGIN")
        self.conn.execute(
            """
            INSERT INTO flat_state (key, json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
              json=excluded.json,
              updated_at=excluded.updated_at
            """,
            (key, payload_json, updated_at),
        )
        self.conn.execute("COMMIT")

    def load_flat_state(self, key: str) -> Optional[Dict[str, object]]:
        cur = self.conn.execute("SELECT json FROM flat_state WHERE key = ?", (key,))
        row = cur.fetchone()
        if row is None:
            return None
        raw = row["json"] or "{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None


def _pack_arrays(values: Dict[str, np.ndarray]) -> bytes:
    bio = BytesIO()
    np.savez(bio, **{k: np.asarray(v) for k, v in values.items()})
    return zlib.compress(bio.getvalue())


def _unpack_arrays(blob: bytes) -> Dict[str, np.ndarray]:
    raw = zlib.decompress(blob)
    data = np.load(BytesIO(raw), allow_pickle=False)
    return {k: data[k] for k in data.files}


def _pack_opt_state(opt_state: AdamState) -> bytes:
    payload: Dict[str, np.ndarray] = {"__t__": np.asarray(opt_state.t, dtype=np.int64)}
    for key, value in opt_state.m.items():
        payload[f"m_{key}"] = np.asarray(value)
    for key, value in opt_state.v.items():
        payload[f"v_{key}"] = np.asarray(value)
    return _pack_arrays(payload)


def _unpack_opt_state(blob: bytes) -> AdamState:
    payload = _unpack_arrays(blob)
    t = int(payload.get("__t__", np.asarray(0)).item())
    m: Dict[str, np.ndarray] = {}
    v: Dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if key.startswith("m_"):
            m[key[2:]] = value
        elif key.startswith("v_"):
            v[key[2:]] = value
    return AdamState(m=m, v=v, t=t)
