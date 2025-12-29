import hashlib
import json
import math
import time
from typing import Dict, Optional, Tuple

from titan.core.storage import connect, encode_payload, decode_payload


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

    def get_events(self, pattern_id: int, limit: int = 1000) -> list:
        """Get events for a pattern, most recent first."""
        rows = self._conn.execute(
            """
            SELECT event_ts, event_blob FROM pattern_events
            WHERE pattern_id = ?
            ORDER BY event_ts DESC
            LIMIT ?
            """,
            (pattern_id, limit),
        ).fetchall()
        return [
            {"event_ts": row["event_ts"], **decode_payload(row["event_blob"])}
            for row in rows
        ]

    def get_usage_count(self, pattern_id: int) -> int:
        """Get total usage count for a pattern."""
        row = self._conn.execute(
            "SELECT usage_count FROM patterns WHERE id = ?",
            (pattern_id,),
        ).fetchone()
        return int(row["usage_count"]) if row else 0


class PatternExperience:
    """Analyzes historical pattern performance for prediction improvement.

    Sprint 13: Uses accumulated pattern history to adjust confidence
    based on how well each pattern has performed historically.
    """

    def __init__(self, pattern_store: PatternStore, decay_hours: int = 168) -> None:
        """Initialize pattern experience analyzer.

        Args:
            pattern_store: The pattern store with historical events
            decay_hours: Half-life for exponential decay (default 168 = 1 week)
        """
        self._store = pattern_store
        self._decay_hours = decay_hours
        self._cache: Dict[int, Dict] = {}  # pattern_id -> stats

    def get_pattern_stats(
        self, pattern_id: int, use_decay: bool = True
    ) -> Dict[str, float]:
        """Get performance statistics for a pattern.

        Returns:
            Dict with keys: total_uses, accuracy, up_accuracy, down_accuracy,
                           confidence (in the estimate)
        """
        # Check cache first
        cache_key = (pattern_id, use_decay)
        if pattern_id in self._cache:
            return self._cache[pattern_id]

        events = self._store.get_events(pattern_id)

        if not events:
            return {
                "total_uses": 0,
                "accuracy": 0.50,
                "up_accuracy": 0.50,
                "down_accuracy": 0.50,
                "confidence": 0.0,
            }

        now = time.time()
        weighted_correct = 0.0
        weighted_total = 0.0
        weighted_up_correct = 0.0
        weighted_up_total = 0.0
        weighted_down_correct = 0.0
        weighted_down_total = 0.0

        for event in events:
            # Calculate time-based weight (exponential decay)
            if use_decay:
                age_hours = (now - event["event_ts"]) / 3600
                weight = math.exp(-age_hours / self._decay_hours)
            else:
                weight = 1.0

            # Extract outcome info from event
            outcome = event.get("outcome", {})
            forecast = event.get("forecast", {})
            hit = outcome.get("hit", False)
            direction = forecast.get("direction", "UP")

            weighted_total += weight
            if hit:
                weighted_correct += weight

            if direction == "UP":
                weighted_up_total += weight
                if hit:
                    weighted_up_correct += weight
            else:
                weighted_down_total += weight
                if hit:
                    weighted_down_correct += weight

        # Calculate statistics
        accuracy = weighted_correct / weighted_total if weighted_total > 0 else 0.5
        up_acc = weighted_up_correct / weighted_up_total if weighted_up_total > 0 else 0.5
        down_acc = weighted_down_correct / weighted_down_total if weighted_down_total > 0 else 0.5

        # Confidence in our estimate (based on sample size)
        # More events = higher confidence, caps at 1.0 at 50+ events
        confidence = min(weighted_total / 50.0, 1.0)

        stats = {
            "total_uses": len(events),
            "accuracy": accuracy,
            "up_accuracy": up_acc,
            "down_accuracy": down_acc,
            "confidence": confidence,
        }

        # Cache the result
        self._cache[pattern_id] = stats
        return stats

    def get_pattern_bias(self, pattern_id: int) -> Optional[str]:
        """Determine if pattern has a directional bias.

        Returns:
            'UP' if pattern works better for UP predictions
            'DOWN' if pattern works better for DOWN predictions
            None if no clear bias (difference < 5%)
        """
        stats = self.get_pattern_stats(pattern_id)

        # Need sufficient confidence to determine bias
        if stats["confidence"] < 0.4:
            return None

        up_acc = stats["up_accuracy"]
        down_acc = stats["down_accuracy"]

        if up_acc > down_acc + 0.05:
            return "UP"
        elif down_acc > up_acc + 0.05:
            return "DOWN"
        return None

    def should_trust_pattern(self, pattern_id: int, min_uses: int = 20) -> bool:
        """Check if we have enough data to trust pattern statistics."""
        stats = self.get_pattern_stats(pattern_id)
        return stats["total_uses"] >= min_uses

    def clear_cache(self) -> None:
        """Clear the statistics cache."""
        self._cache.clear()
