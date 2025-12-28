import time
from typing import Any, Dict, Iterable, Optional

from titan.core.storage import connect, decode_payload, encode_payload


class StateStore:
    def __init__(self, db_path: str) -> None:
        self._conn = connect(db_path)
        self._setup()

    def _setup(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kv (
                key TEXT PRIMARY KEY,
                value_blob BLOB NOT NULL,
                updated_at INTEGER NOT NULL
            )
            """
        )
        self._conn.commit()

    def get(self, key: str, default: Any = None) -> Any:
        row = self._conn.execute("SELECT value_blob FROM kv WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        return decode_payload(row["value_blob"])

    def set(self, key: str, value: Any) -> None:
        blob = encode_payload(value)
        ts = int(time.time())
        self._conn.execute(
            """
            INSERT INTO kv (key, value_blob, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value_blob = excluded.value_blob,
                updated_at = excluded.updated_at
            """,
            (key, blob, ts),
        )
        self._conn.commit()

    def set_if_missing(self, key: str, value: Any) -> None:
        blob = encode_payload(value)
        ts = int(time.time())
        self._conn.execute(
            "INSERT OR IGNORE INTO kv (key, value_blob, updated_at) VALUES (?, ?, ?)",
            (key, blob, ts),
        )
        self._conn.commit()

    def get_many(self, keys: Iterable[str]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key in keys:
            result[key] = self.get(key)
        return result

    def close(self) -> None:
        self._conn.close()
