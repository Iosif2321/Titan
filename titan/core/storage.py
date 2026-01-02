import json
import sqlite3
import zlib
from typing import Any


def encode_payload(payload: Any) -> bytes:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return zlib.compress(raw.encode("utf-8"), level=6)


def decode_payload(blob: bytes) -> Any:
    raw = zlib.decompress(blob).decode("utf-8")
    return json.loads(raw)


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn
