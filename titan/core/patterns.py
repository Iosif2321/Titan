import hashlib
import json
import time
from typing import Dict

from titan.core.storage import connect, encode_payload


class PatternStore:
    def __init__(self, db_path: str) -> None:
        self._conn = connect(db_path)
        self._setup()

    def _setup(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY,
                conditions_hash TEXT NOT NULL UNIQUE,
                conditions_blob BLOB NOT NULL,
                created_at INTEGER NOT NULL,
                usage_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pattern_events (
                id INTEGER PRIMARY KEY,
                pattern_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                event_ts INTEGER NOT NULL,
                event_blob BLOB NOT NULL,
                FOREIGN KEY(pattern_id) REFERENCES patterns(id)
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pattern_events_pattern ON pattern_events(pattern_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pattern_events_model ON pattern_events(model_name)"
        )
        self._conn.commit()

    def _hash_conditions(self, conditions: Dict[str, str]) -> str:
        raw = json.dumps(conditions, separators=(",", ":"), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_or_create(self, conditions: Dict[str, str]) -> int:
        cond_hash = self._hash_conditions(conditions)
        row = self._conn.execute(
            "SELECT id FROM patterns WHERE conditions_hash = ?", (cond_hash,)
        ).fetchone()
        if row is not None:
            return int(row["id"])

        blob = encode_payload(conditions)
        ts = int(time.time())
        cursor = self._conn.execute(
            """
            INSERT INTO patterns (conditions_hash, conditions_blob, created_at)
            VALUES (?, ?, ?)
            """,
            (cond_hash, blob, ts),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def record_usage(
        self,
        pattern_id: int,
        model_name: str,
        event: Dict[str, object],
        event_ts: int,
    ) -> None:
        blob = encode_payload(event)
        self._conn.execute(
            """
            INSERT INTO pattern_events (pattern_id, model_name, event_ts, event_blob)
            VALUES (?, ?, ?, ?)
            """,
            (pattern_id, model_name, event_ts, blob),
        )
        self._conn.execute(
            "UPDATE patterns SET usage_count = usage_count + 1 WHERE id = ?",
            (pattern_id,),
        )
        self._conn.commit()
