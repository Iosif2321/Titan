from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, List


def interval_to_ms(interval: str) -> int:
    try:
        minutes = int(interval)
    except (TypeError, ValueError):
        minutes = 1
    return minutes * 60_000


def ts_iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()


def now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def parse_tfs(raw: str | Iterable[str]) -> List[str]:
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",")]
    else:
        parts = [str(p).strip() for p in raw]
    seen = set()
    out: List[str] = []
    for part in parts:
        if not part:
            continue
        if part in seen:
            continue
        seen.add(part)
        out.append(part)
    return out
