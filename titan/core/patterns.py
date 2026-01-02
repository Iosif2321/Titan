import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from titan.core.storage import connect, encode_payload, decode_payload

if TYPE_CHECKING:
    from titan.core.config import ConfigStore

# Default constants for pattern system (can be overridden by config)
DEFAULT_MAX_DECISIONS = 50000  # Maximum decisions per pattern
DEFAULT_TOP_DECISIONS_COUNT = 1000  # Always keep top N brightest decisions
DEFAULT_INACTIVE_AFTER_DAYS = 30  # Move to inactive after N days unused
DEFAULT_DELETE_AFTER_DAYS = 90  # Delete after N days in inactive
DEFAULT_HIGH_CONF_THRESHOLD = 0.65  # Threshold for "high confidence"

FEATURE_BUCKETS = {
    "rsi": [(-math.inf, 30.0, "<=30"), (30.0, 70.0, "30-70"), (70.0, math.inf, ">=70")],
    "volatility_z": [
        (-math.inf, -1.0, "<=-1"),
        (-1.0, 0.0, "-1..0"),
        (0.0, 1.0, "0..1"),
        (1.0, 2.0, "1..2"),
        (2.0, math.inf, ">=2"),
    ],
    "volume_z": [
        (-math.inf, -1.0, "<=-1"),
        (-1.0, 0.0, "-1..0"),
        (0.0, 1.0, "0..1"),
        (1.0, 2.0, "1..2"),
        (2.0, math.inf, ">=2"),
    ],
    "rsi_momentum": [
        (-math.inf, -0.5, "<=-0.5"),
        (-0.5, 0.5, "-0.5..0.5"),
        (0.5, math.inf, ">=0.5"),
    ],
    "price_momentum_3": [
        (-math.inf, -0.001, "<=-0.001"),
        (-0.001, 0.001, "-0.001..0.001"),
        (0.001, math.inf, ">=0.001"),
    ],
    "volume_trend": [
        (-math.inf, -0.2, "<=-0.2"),
        (-0.2, 0.2, "-0.2..0.2"),
        (0.2, math.inf, ">=0.2"),
    ],
    "body_ratio": [
        (-math.inf, 0.3, "<=0.3"),
        (0.3, 0.7, "0.3..0.7"),
        (0.7, math.inf, ">=0.7"),
    ],
}


def get_feature_bucket(feature: str, value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    buckets = FEATURE_BUCKETS.get(feature)
    if not buckets:
        return None
    for low, high, label in buckets:
        if value <= high and value > low:
            return label
    return None


@dataclass
class PatternAggregates:
    """Pre-computed pattern aggregates (stored, not calculated)."""
    pattern_id: int
    total_uses: int
    last_used_ts: int
    accuracy: float
    up_accuracy: float
    down_accuracy: float
    avg_return: float
    cumulative_return: float
    win_rate: float
    conf_wrong_rate: float
    hit_count: int
    up_hit_count: int
    up_total_count: int
    down_hit_count: int
    down_total_count: int
    high_conf_count: int
    high_conf_wrong_count: int
    decision_count: int
    top_decisions_count: int
    status: str  # 'active', 'inactive', 'deleted'


@dataclass
class PatternModelAggregates:
    """Pre-computed per-model aggregates for a pattern."""
    pattern_id: int
    model_name: str
    total_uses: int
    last_used_ts: int
    accuracy: float
    up_accuracy: float
    down_accuracy: float
    avg_return: float
    cumulative_return: float
    win_rate: float
    conf_wrong_rate: float
    hit_count: int
    up_hit_count: int
    up_total_count: int
    down_hit_count: int
    down_total_count: int
    high_conf_count: int
    high_conf_wrong_count: int


@dataclass
class PatternMatch:
    """Result of fuzzy pattern search."""
    pattern_id: int
    pattern_key: str
    match_count: int  # Number of matching conditions
    match_ratio: float  # match_count / total_conditions
    status: str
    aggregates: Optional[PatternAggregates] = None


@dataclass
class ExtendedConditions:
    """Extended condition dimensions for v2 patterns."""
    trend: str  # 'UP', 'DOWN', 'FLAT'
    volatility: str  # 'HIGH', 'MID', 'LOW'
    volume: str  # 'HIGH', 'MID', 'LOW'
    hour: int  # 0-23
    session: str  # 'ASIA', 'EUROPE', 'US', 'OVERLAP'
    day_of_week: Optional[int] = None  # 0=Mon, 6=Sun


def get_trading_session(hour: int) -> str:
    """Determine trading session based on UTC hour.

    Sessions (approximate UTC times):
    - ASIA: 00:00 - 08:00 (Tokyo, Hong Kong, Singapore)
    - EUROPE: 08:00 - 13:00 (London, Frankfurt)
    - OVERLAP: 13:00 - 17:00 (London + New York overlap)
    - US: 17:00 - 22:00 (New York)
    - ASIA (again): 22:00 - 00:00 (early Asia)
    """
    if 0 <= hour < 8:
        return "ASIA"
    elif 8 <= hour < 13:
        return "EUROPE"
    elif 13 <= hour < 17:
        return "OVERLAP"
    elif 17 <= hour < 22:
        return "US"
    else:  # 22-23
        return "ASIA"


def build_pattern_key(
    trend: str,
    volatility: str,
    volume: str,
    hour: int,
    session: str,
) -> str:
    """Build human-readable pattern key.

    Format: {TREND}_{VOLATILITY}_{VOLUME}_{HOUR:02d}_{SESSION}
    Example: UP_HIGH_MID_12_EUROPE
    """
    return f"{trend.upper()}_{volatility.upper()}_{volume.upper()}_{hour:02d}_{session.upper()}"


def parse_pattern_key(key: str) -> Optional[ExtendedConditions]:
    """Parse pattern key back to conditions.

    Args:
        key: Pattern key like "UP_HIGH_MID_12_EUROPE"

    Returns:
        ExtendedConditions or None if invalid format
    """
    try:
        parts = key.split("_")
        if len(parts) != 5:
            return None
        trend, volatility, volume, hour_str, session = parts
        return ExtendedConditions(
            trend=trend,
            volatility=volatility,
            volume=volume,
            hour=int(hour_str),
            session=session,
        )
    except (ValueError, IndexError):
        return None


def build_extended_conditions(
    conditions: Dict[str, str],
    ts: int,
) -> ExtendedConditions:
    """Build ExtendedConditions from base conditions and timestamp."""
    from datetime import datetime

    dt = datetime.utcfromtimestamp(ts)
    hour = dt.hour
    if "hour" in conditions:
        try:
            hour = int(conditions["hour"])
        except (TypeError, ValueError):
            hour = dt.hour
    session = str(conditions.get("session") or get_trading_session(hour))
    if "day_of_week" in conditions:
        try:
            day_of_week = int(conditions["day_of_week"])
        except (TypeError, ValueError):
            day_of_week = dt.weekday()
    else:
        day_of_week = dt.weekday()
    return ExtendedConditions(
        trend=conditions.get("trend", "FLAT").upper(),
        volatility=conditions.get("volatility", "MID").upper(),
        volume=conditions.get("volume", "MID").upper(),
        hour=hour,
        session=session,
        day_of_week=day_of_week,
    )


class PatternStore:
    def __init__(
        self, db_path: str, config: Optional["ConfigStore"] = None
    ) -> None:
        self._conn = connect(db_path)
        self._config = config

        # Load config values or use defaults
        if config:
            self._max_decisions = int(config.get("pattern.max_decisions", DEFAULT_MAX_DECISIONS))
            self._top_decisions_count = int(config.get("pattern.top_decisions_count", DEFAULT_TOP_DECISIONS_COUNT))
            self._inactive_after_days = int(config.get("pattern.inactive_after_days", DEFAULT_INACTIVE_AFTER_DAYS))
            self._delete_after_days = int(config.get("pattern.delete_after_days", DEFAULT_DELETE_AFTER_DAYS))
            self._high_conf_threshold = float(config.get("pattern.high_conf_threshold", DEFAULT_HIGH_CONF_THRESHOLD))
            self._conditions_version = int(config.get("pattern.conditions_version", 1))
            self._snapshot_rate = float(config.get("pattern.snapshot_rate", 1.0))
            self._include_ensemble_global = bool(
                config.get("pattern.include_ensemble_in_global", False)
            )
        else:
            self._max_decisions = DEFAULT_MAX_DECISIONS
            self._top_decisions_count = DEFAULT_TOP_DECISIONS_COUNT
            self._inactive_after_days = DEFAULT_INACTIVE_AFTER_DAYS
            self._delete_after_days = DEFAULT_DELETE_AFTER_DAYS
            self._high_conf_threshold = DEFAULT_HIGH_CONF_THRESHOLD
            self._conditions_version = 1
            self._snapshot_rate = 1.0
            self._include_ensemble_global = False

        self._setup()
        self._migrate_v2()

    def _setup(self) -> None:
        """Create base tables (v1 schema)."""
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

    def _migrate_v2(self) -> None:
        """Migrate to v2 schema with extended columns and tables."""
        # Add new columns to patterns table
        migrations = [
            # Human-readable pattern key
            "ALTER TABLE patterns ADD COLUMN pattern_key TEXT",
            # Lifecycle status
            "ALTER TABLE patterns ADD COLUMN status TEXT DEFAULT 'active'",
            "ALTER TABLE patterns ADD COLUMN inactive_since INTEGER",
            # Decision counts
            "ALTER TABLE patterns ADD COLUMN decision_count INTEGER DEFAULT 0",
            "ALTER TABLE patterns ADD COLUMN top_decisions_count INTEGER DEFAULT 0",
            # Stored aggregates
            "ALTER TABLE patterns ADD COLUMN accuracy REAL DEFAULT 0.5",
            "ALTER TABLE patterns ADD COLUMN up_accuracy REAL DEFAULT 0.5",
            "ALTER TABLE patterns ADD COLUMN down_accuracy REAL DEFAULT 0.5",
            "ALTER TABLE patterns ADD COLUMN avg_return REAL DEFAULT 0.0",
            "ALTER TABLE patterns ADD COLUMN cumulative_return REAL DEFAULT 0.0",
            "ALTER TABLE patterns ADD COLUMN win_rate REAL DEFAULT 0.5",
            "ALTER TABLE patterns ADD COLUMN conf_wrong_rate REAL DEFAULT 0.0",
            "ALTER TABLE patterns ADD COLUMN last_used_ts INTEGER",
            # Counters for incremental updates
            "ALTER TABLE patterns ADD COLUMN hit_count INTEGER DEFAULT 0",
            "ALTER TABLE patterns ADD COLUMN up_hit_count INTEGER DEFAULT 0",
            "ALTER TABLE patterns ADD COLUMN up_total_count INTEGER DEFAULT 0",
            "ALTER TABLE patterns ADD COLUMN down_hit_count INTEGER DEFAULT 0",
            "ALTER TABLE patterns ADD COLUMN down_total_count INTEGER DEFAULT 0",
            "ALTER TABLE patterns ADD COLUMN high_conf_count INTEGER DEFAULT 0",
            "ALTER TABLE patterns ADD COLUMN high_conf_wrong_count INTEGER DEFAULT 0",
            # Per-model stats
            "ALTER TABLE patterns ADD COLUMN model_stats_blob BLOB",
            # Conditions version
            "ALTER TABLE patterns ADD COLUMN conditions_version INTEGER DEFAULT 1",
        ]

        for sql in migrations:
            try:
                self._conn.execute(sql)
            except Exception:
                pass  # Column may already exist

        # Add new columns to pattern_events table
        event_migrations = [
            "ALTER TABLE pattern_events ADD COLUMN decision TEXT",
            "ALTER TABLE pattern_events ADD COLUMN confidence REAL",
            "ALTER TABLE pattern_events ADD COLUMN hit INTEGER",
            "ALTER TABLE pattern_events ADD COLUMN return_pct REAL",
            "ALTER TABLE pattern_events ADD COLUMN price REAL",
            "ALTER TABLE pattern_events ADD COLUMN regime TEXT",
            "ALTER TABLE pattern_events ADD COLUMN significance REAL DEFAULT 0.0",
            "ALTER TABLE pattern_events ADD COLUMN is_top_decision INTEGER DEFAULT 0",
        ]

        for sql in event_migrations:
            try:
                self._conn.execute(sql)
            except Exception:
                pass

        # Create new tables
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pattern_conditions_v2 (
                pattern_id INTEGER PRIMARY KEY,
                trend TEXT NOT NULL,
                volatility TEXT NOT NULL,
                volume TEXT NOT NULL,
                hour INTEGER,
                day_of_week INTEGER,
                session TEXT,
                momentum TEXT,
                rsi_zone TEXT,
                FOREIGN KEY(pattern_id) REFERENCES patterns(id)
            )
            """
        )

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pattern_event_snapshots (
                event_id INTEGER PRIMARY KEY,
                features_blob BLOB NOT NULL,
                model_state_blob BLOB,
                FOREIGN KEY(event_id) REFERENCES pattern_events(id)
            )
            """
        )

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pattern_search_index (
                pattern_id INTEGER PRIMARY KEY,
                trend TEXT NOT NULL,
                volatility TEXT NOT NULL,
                volume TEXT NOT NULL,
                hour INTEGER NOT NULL,
                session TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                FOREIGN KEY(pattern_id) REFERENCES patterns(id)
            )
            """
        )

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pattern_model_stats (
                pattern_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                usage_count INTEGER NOT NULL DEFAULT 0,
                last_used_ts INTEGER,
                hit_count INTEGER NOT NULL DEFAULT 0,
                up_hit_count INTEGER NOT NULL DEFAULT 0,
                up_total_count INTEGER NOT NULL DEFAULT 0,
                down_hit_count INTEGER NOT NULL DEFAULT 0,
                down_total_count INTEGER NOT NULL DEFAULT 0,
                high_conf_count INTEGER NOT NULL DEFAULT 0,
                high_conf_wrong_count INTEGER NOT NULL DEFAULT 0,
                accuracy REAL DEFAULT 0.5,
                up_accuracy REAL DEFAULT 0.5,
                down_accuracy REAL DEFAULT 0.5,
                avg_return REAL DEFAULT 0.0,
                cumulative_return REAL DEFAULT 0.0,
                win_rate REAL DEFAULT 0.5,
                conf_wrong_rate REAL DEFAULT 0.0,
                PRIMARY KEY (pattern_id, model_name),
                FOREIGN KEY(pattern_id) REFERENCES patterns(id)
            )
            """
        )

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pattern_feature_buckets (
                pattern_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                feature TEXT NOT NULL,
                bucket TEXT NOT NULL,
                total_count INTEGER NOT NULL DEFAULT 0,
                hit_count INTEGER NOT NULL DEFAULT 0,
                cumulative_return REAL DEFAULT 0.0,
                PRIMARY KEY (pattern_id, model_name, feature, bucket),
                FOREIGN KEY(pattern_id) REFERENCES patterns(id)
            )
            """
        )

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pattern_temporal_stats (
                pattern_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                hour INTEGER NOT NULL,
                session TEXT NOT NULL,
                day_of_week INTEGER NOT NULL,
                total_count INTEGER NOT NULL DEFAULT 0,
                hit_count INTEGER NOT NULL DEFAULT 0,
                cumulative_return REAL DEFAULT 0.0,
                PRIMARY KEY (pattern_id, model_name, hour, session, day_of_week),
                FOREIGN KEY(pattern_id) REFERENCES patterns(id)
            )
            """
        )

        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_patterns_status ON patterns(status)",
            "CREATE INDEX IF NOT EXISTS idx_patterns_last_used ON patterns(last_used_ts DESC)",
            "CREATE INDEX IF NOT EXISTS idx_patterns_pattern_key ON patterns(pattern_key)",
            "CREATE INDEX IF NOT EXISTS idx_pe_significance ON pattern_events(pattern_id, significance DESC)",
            "CREATE INDEX IF NOT EXISTS idx_pe_hit ON pattern_events(pattern_id, hit)",
            "CREATE INDEX IF NOT EXISTS idx_pes_event ON pattern_event_snapshots(event_id)",
            "CREATE INDEX IF NOT EXISTS idx_psi_trend ON pattern_search_index(trend, status)",
            "CREATE INDEX IF NOT EXISTS idx_psi_vol ON pattern_search_index(volatility, status)",
            "CREATE INDEX IF NOT EXISTS idx_psi_volume ON pattern_search_index(volume, status)",
            "CREATE INDEX IF NOT EXISTS idx_psi_hour ON pattern_search_index(hour, status)",
            "CREATE INDEX IF NOT EXISTS idx_psi_session ON pattern_search_index(session, status)",
            "CREATE INDEX IF NOT EXISTS idx_psi_status ON pattern_search_index(status)",
            "CREATE INDEX IF NOT EXISTS idx_pms_pattern ON pattern_model_stats(pattern_id)",
            "CREATE INDEX IF NOT EXISTS idx_pms_model ON pattern_model_stats(model_name)",
            "CREATE INDEX IF NOT EXISTS idx_pfb_pattern ON pattern_feature_buckets(pattern_id)",
            "CREATE INDEX IF NOT EXISTS idx_pfb_model ON pattern_feature_buckets(model_name)",
            "CREATE INDEX IF NOT EXISTS idx_pts_pattern ON pattern_temporal_stats(pattern_id)",
            "CREATE INDEX IF NOT EXISTS idx_pts_model ON pattern_temporal_stats(model_name)",
        ]

        for sql in indexes:
            try:
                self._conn.execute(sql)
            except Exception:
                pass

        self._conn.commit()
        self._backfill_extended_data()
        self._backfill_model_stats()

    def _hash_conditions(self, conditions: Dict[str, str]) -> str:
        raw = json.dumps(conditions, separators=(",", ":"), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_or_create(
        self,
        conditions: Dict[str, str],
        extended: Optional[ExtendedConditions] = None,
        ts: Optional[int] = None,
    ) -> int:
        """Get or create a pattern for given conditions.

        Args:
            conditions: Base conditions dict (trend, volatility, volume)
            extended: Optional extended conditions (hour, session, etc.)
            ts: Optional timestamp to extract hour/session if extended not provided

        Returns:
            Pattern ID
        """
        # If extended conditions not provided but ts is, extract from timestamp
        if extended is None and ts is not None:
            extended = build_extended_conditions(conditions, ts)

        # Build pattern key if we have extended conditions
        pattern_key = None
        if extended is not None:
            pattern_key = build_pattern_key(
                extended.trend,
                extended.volatility,
                extended.volume,
                extended.hour,
                extended.session,
            )

        # First try to find by pattern_key (unique identifier including hour/session)
        # This avoids duplicates caused by day_of_week being in conditions hash
        if pattern_key:
            row = self._conn.execute(
                "SELECT id FROM patterns WHERE pattern_key = ?", (pattern_key,)
            ).fetchone()
            if row is not None:
                pattern_id = int(row["id"])
                if extended is not None:
                    self._insert_search_index(pattern_id, extended)
                    self._upsert_conditions_v2(pattern_id, extended)
                self._conn.commit()
                return pattern_id

        # Fall back to hash for legacy patterns without pattern_key
        cond_hash = self._hash_conditions(conditions)
        row = self._conn.execute(
            "SELECT id FROM patterns WHERE conditions_hash = ?", (cond_hash,)
        ).fetchone()
        if row is not None:
            pattern_id = int(row["id"])
            # Update pattern_key if not set and we have one
            if pattern_key:
                self._conn.execute(
                    "UPDATE patterns SET pattern_key = ? WHERE id = ? AND pattern_key IS NULL",
                    (pattern_key, pattern_id),
                )
            if extended is not None:
                self._insert_search_index(pattern_id, extended)
                self._upsert_conditions_v2(pattern_id, extended)
            self._conn.commit()
            return pattern_id

        blob = encode_payload(conditions)
        now = int(time.time())
        cursor = self._conn.execute(
            """
            INSERT INTO patterns (
                conditions_hash,
                conditions_blob,
                created_at,
                pattern_key,
                status,
                last_used_ts,
                conditions_version
            )
            VALUES (?, ?, ?, ?, 'active', ?, ?)
            """,
            (cond_hash, blob, now, pattern_key, now, self._conditions_version),
        )
        pattern_id = int(cursor.lastrowid)

        # Populate search index if we have extended conditions
        if extended is not None:
            self._insert_search_index(pattern_id, extended)
            self._upsert_conditions_v2(pattern_id, extended)

        self._conn.commit()
        return pattern_id

    def _insert_search_index(self, pattern_id: int, ext: ExtendedConditions) -> None:
        """Insert pattern into search index for fuzzy matching."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO pattern_search_index
            (pattern_id, trend, volatility, volume, hour, session, status)
            VALUES (?, ?, ?, ?, ?, ?, 'active')
            """,
            (
                pattern_id,
                ext.trend.upper(),
                ext.volatility.upper(),
                ext.volume.upper(),
                ext.hour,
                ext.session.upper(),
            ),
        )

    def _upsert_conditions_v2(self, pattern_id: int, ext: ExtendedConditions) -> None:
        """Upsert extended conditions into v2 table."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO pattern_conditions_v2
            (pattern_id, trend, volatility, volume, hour, day_of_week, session)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pattern_id,
                ext.trend.upper(),
                ext.volatility.upper(),
                ext.volume.upper(),
                ext.hour,
                ext.day_of_week,
                ext.session.upper(),
            ),
        )

    def _backfill_extended_data(self) -> None:
        """Backfill search index, conditions_v2, and pattern_key where possible."""
        rows = self._conn.execute(
            """
            SELECT
                p.id,
                p.pattern_key,
                p.conditions_blob,
                psi.pattern_id AS idx_id,
                pc.pattern_id AS cond_id
            FROM patterns p
            LEFT JOIN pattern_search_index psi ON psi.pattern_id = p.id
            LEFT JOIN pattern_conditions_v2 pc ON pc.pattern_id = p.id
            """
        ).fetchall()

        for row in rows:
            if not row["conditions_blob"]:
                continue
            conditions = decode_payload(row["conditions_blob"])
            if not isinstance(conditions, dict):
                continue

            hour = conditions.get("hour")
            session = conditions.get("session")
            if hour is None or session is None:
                continue

            try:
                ext = ExtendedConditions(
                    trend=str(conditions.get("trend", "FLAT")).upper(),
                    volatility=str(conditions.get("volatility", "MID")).upper(),
                    volume=str(conditions.get("volume", "MID")).upper(),
                    hour=int(hour),
                    session=str(session).upper(),
                    day_of_week=int(conditions.get("day_of_week")) if "day_of_week" in conditions else None,
                )
            except (TypeError, ValueError):
                continue

            if row["idx_id"] is None:
                self._insert_search_index(row["id"], ext)

            if row["cond_id"] is None:
                self._upsert_conditions_v2(row["id"], ext)

            if row["pattern_key"] is None:
                pattern_key = build_pattern_key(
                    ext.trend,
                    ext.volatility,
                    ext.volume,
                    ext.hour,
                    ext.session,
                )
                self._conn.execute(
                    "UPDATE patterns SET pattern_key = ? WHERE id = ? AND pattern_key IS NULL",
                    (pattern_key, row["id"]),
                )

        self._conn.commit()

    def _backfill_model_stats(self) -> None:
        """Backfill per-model aggregates from existing events if missing."""
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM pattern_model_stats"
        ).fetchone()
        if row and row["cnt"]:
            return

        rows = self._conn.execute(
            """
            SELECT
                pattern_id,
                model_name,
                COUNT(*) as usage_count,
                SUM(COALESCE(hit, 0)) as hit_count,
                SUM(CASE WHEN decision = 'UP' THEN 1 ELSE 0 END) as up_total_count,
                SUM(CASE WHEN decision = 'UP' AND hit = 1 THEN 1 ELSE 0 END) as up_hit_count,
                SUM(CASE WHEN decision = 'DOWN' THEN 1 ELSE 0 END) as down_total_count,
                SUM(CASE WHEN decision = 'DOWN' AND hit = 1 THEN 1 ELSE 0 END) as down_hit_count,
                SUM(CASE WHEN confidence >= ? THEN 1 ELSE 0 END) as high_conf_count,
                SUM(CASE WHEN confidence >= ? AND hit = 0 THEN 1 ELSE 0 END) as high_conf_wrong_count,
                SUM(COALESCE(return_pct, 0.0)) as cumulative_return,
                MAX(event_ts) as last_used_ts
            FROM pattern_events
            GROUP BY pattern_id, model_name
            """,
            (self._high_conf_threshold, self._high_conf_threshold),
        ).fetchall()

        if not rows:
            return

        inserts = []
        for r in rows:
            usage_count = r["usage_count"] or 0
            hit_count = r["hit_count"] or 0
            up_total = r["up_total_count"] or 0
            up_hit = r["up_hit_count"] or 0
            down_total = r["down_total_count"] or 0
            down_hit = r["down_hit_count"] or 0
            high_conf_count = r["high_conf_count"] or 0
            high_conf_wrong = r["high_conf_wrong_count"] or 0
            cumulative_return = r["cumulative_return"] or 0.0

            accuracy = (hit_count / usage_count) if usage_count else 0.5
            up_accuracy = (up_hit / up_total) if up_total else 0.5
            down_accuracy = (down_hit / down_total) if down_total else 0.5
            avg_return = (cumulative_return / usage_count) if usage_count else 0.0
            win_rate = accuracy
            conf_wrong_rate = (
                (high_conf_wrong / high_conf_count) if high_conf_count else 0.0
            )

            inserts.append(
                (
                    r["pattern_id"],
                    r["model_name"],
                    usage_count,
                    r["last_used_ts"] or 0,
                    hit_count,
                    up_hit,
                    up_total,
                    down_hit,
                    down_total,
                    high_conf_count,
                    high_conf_wrong,
                    accuracy,
                    up_accuracy,
                    down_accuracy,
                    avg_return,
                    cumulative_return,
                    win_rate,
                    conf_wrong_rate,
                )
            )

        self._conn.executemany(
            """
            INSERT INTO pattern_model_stats (
                pattern_id,
                model_name,
                usage_count,
                last_used_ts,
                hit_count,
                up_hit_count,
                up_total_count,
                down_hit_count,
                down_total_count,
                high_conf_count,
                high_conf_wrong_count,
                accuracy,
                up_accuracy,
                down_accuracy,
                avg_return,
                cumulative_return,
                win_rate,
                conf_wrong_rate
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            inserts,
        )
        self._conn.commit()

    def _update_search_index_status(self, pattern_id: int, status: str) -> None:
        """Update pattern status in search index."""
        self._conn.execute(
            "UPDATE pattern_search_index SET status = ? WHERE pattern_id = ?",
            (status, pattern_id),
        )

    def find_similar_patterns(
        self,
        conditions: ExtendedConditions,
        min_match: float = 0.8,
        include_inactive: bool = True,
        limit: int = 10,
    ) -> List[PatternMatch]:
        """Find patterns with fuzzy matching on conditions.

        Args:
            conditions: Target conditions to match against
            min_match: Minimum match ratio (0.0-1.0), default 0.8 = 80%
            include_inactive: Include inactive patterns in search
            limit: Maximum number of results

        Returns:
            List of PatternMatch sorted by match_count DESC, active first
        """
        total_conditions = 5  # trend, volatility, volume, hour, session
        min_matches = int(min_match * total_conditions)

        status_filter = "('active', 'inactive')" if include_inactive else "('active')"

        # Query all patterns and calculate match count using subquery
        rows = self._conn.execute(
            f"""
            SELECT * FROM (
                SELECT
                    psi.pattern_id,
                    p.pattern_key,
                    psi.status,
                    (CASE WHEN psi.trend = ? THEN 1 ELSE 0 END +
                     CASE WHEN psi.volatility = ? THEN 1 ELSE 0 END +
                     CASE WHEN psi.volume = ? THEN 1 ELSE 0 END +
                     CASE WHEN psi.hour = ? THEN 1 ELSE 0 END +
                     CASE WHEN psi.session = ? THEN 1 ELSE 0 END) as match_count,
                    p.last_used_ts
                FROM pattern_search_index psi
                JOIN patterns p ON p.id = psi.pattern_id
                WHERE psi.status IN {status_filter}
            ) sub
            WHERE match_count >= ?
            ORDER BY match_count DESC,
                     CASE WHEN status = 'active' THEN 0 ELSE 1 END,
                     last_used_ts DESC
            LIMIT ?
            """,
            (
                conditions.trend.upper(),
                conditions.volatility.upper(),
                conditions.volume.upper(),
                conditions.hour,
                conditions.session.upper(),
                min_matches,
                limit,
            ),
        ).fetchall()

        results = []
        for row in rows:
            results.append(
                PatternMatch(
                    pattern_id=row["pattern_id"],
                    pattern_key=row["pattern_key"] or "",
                    match_count=row["match_count"],
                    match_ratio=row["match_count"] / total_conditions,
                    status=row["status"],
                )
            )

        return results

    def find_exact_or_similar(
        self,
        conditions: Dict[str, str],
        ts: Optional[int] = None,
        min_match: float = 0.8,
    ) -> Tuple[Optional[int], List[PatternMatch]]:
        """Find exact pattern match or similar patterns.

        Args:
            conditions: Base conditions dict
            ts: Timestamp for extended conditions
            min_match: Minimum match ratio for fuzzy search

        Returns:
            (exact_pattern_id, similar_patterns)
            exact_pattern_id is None if no exact match found
        """
        # Try exact match first
        cond_hash = self._hash_conditions(conditions)
        row = self._conn.execute(
            "SELECT id FROM patterns WHERE conditions_hash = ?", (cond_hash,)
        ).fetchone()

        exact_id = int(row["id"]) if row else None

        # Build extended conditions for fuzzy search
        if ts is not None:
            extended = build_extended_conditions(conditions, ts)
            similar = self.find_similar_patterns(extended, min_match=min_match)
        else:
            similar = []

        return exact_id, similar

    def reactivate_pattern(self, pattern_id: int) -> None:
        """Reactivate an inactive pattern.

        Called when a model finds a match with an inactive pattern.
        """
        now = int(time.time())
        self._conn.execute(
            """
            UPDATE patterns
            SET status = 'active', inactive_since = NULL, last_used_ts = ?
            WHERE id = ? AND status = 'inactive'
            """,
            (now, pattern_id),
        )
        self._update_search_index_status(pattern_id, "active")
        self._conn.commit()

    def deactivate_stale_patterns(self, inactive_after_days: Optional[int] = None) -> int:
        """Move stale active patterns to inactive status.

        Args:
            inactive_after_days: Deactivate patterns not used for this many days
                                (uses config value if not specified)

        Returns:
            Number of patterns deactivated
        """
        if inactive_after_days is None:
            inactive_after_days = self._inactive_after_days

        now = int(time.time())
        cutoff_ts = now - (inactive_after_days * 24 * 3600)

        # Find patterns to deactivate
        rows = self._conn.execute(
            """
            SELECT id FROM patterns
            WHERE status = 'active'
              AND (last_used_ts IS NULL OR last_used_ts < ?)
            """,
            (cutoff_ts,),
        ).fetchall()

        if not rows:
            return 0

        pattern_ids = [row["id"] for row in rows]

        # Update patterns
        self._conn.execute(
            f"""
            UPDATE patterns
            SET status = 'inactive', inactive_since = ?
            WHERE id IN ({','.join('?' * len(pattern_ids))})
            """,
            [now] + pattern_ids,
        )

        # Update search index
        self._conn.execute(
            f"""
            UPDATE pattern_search_index
            SET status = 'inactive'
            WHERE pattern_id IN ({','.join('?' * len(pattern_ids))})
            """,
            pattern_ids,
        )

        self._conn.commit()
        return len(pattern_ids)

    def delete_old_inactive(self, delete_after_days: Optional[int] = None) -> int:
        """Delete patterns that have been inactive for too long.

        Args:
            delete_after_days: Delete patterns inactive for this many days
                              (uses config value if not specified)

        Returns:
            Number of patterns deleted
        """
        if delete_after_days is None:
            delete_after_days = self._delete_after_days

        now = int(time.time())
        cutoff_ts = now - (delete_after_days * 24 * 3600)

        # Find patterns to delete
        rows = self._conn.execute(
            """
            SELECT id FROM patterns
            WHERE status = 'inactive'
              AND inactive_since IS NOT NULL
              AND inactive_since < ?
            """,
            (cutoff_ts,),
        ).fetchall()

        if not rows:
            return 0

        pattern_ids = [row["id"] for row in rows]

        # Delete in correct order (foreign key constraints)
        # First delete snapshots
        self._conn.execute(
            f"""
            DELETE FROM pattern_event_snapshots
            WHERE event_id IN (
                SELECT id FROM pattern_events
                WHERE pattern_id IN ({','.join('?' * len(pattern_ids))})
            )
            """,
            pattern_ids,
        )

        # Delete events
        self._conn.execute(
            f"""
            DELETE FROM pattern_events
            WHERE pattern_id IN ({','.join('?' * len(pattern_ids))})
            """,
            pattern_ids,
        )

        # Delete from search index
        self._conn.execute(
            f"""
            DELETE FROM pattern_search_index
            WHERE pattern_id IN ({','.join('?' * len(pattern_ids))})
            """,
            pattern_ids,
        )

        # Delete from feature buckets
        self._conn.execute(
            f"""
            DELETE FROM pattern_feature_buckets
            WHERE pattern_id IN ({','.join('?' * len(pattern_ids))})
            """,
            pattern_ids,
        )

        # Delete from temporal stats
        self._conn.execute(
            f"""
            DELETE FROM pattern_temporal_stats
            WHERE pattern_id IN ({','.join('?' * len(pattern_ids))})
            """,
            pattern_ids,
        )

        # Delete from conditions v2
        self._conn.execute(
            f"""
            DELETE FROM pattern_conditions_v2
            WHERE pattern_id IN ({','.join('?' * len(pattern_ids))})
            """,
            pattern_ids,
        )

        # Finally delete patterns
        self._conn.execute(
            f"""
            DELETE FROM patterns
            WHERE id IN ({','.join('?' * len(pattern_ids))})
            """,
            pattern_ids,
        )

        self._conn.commit()
        return len(pattern_ids)

    def run_lifecycle_maintenance(self) -> Dict[str, int]:
        """Run all lifecycle maintenance tasks.

        Returns:
            Dict with counts: {'deactivated': N, 'deleted': M}
        """
        deactivated = self.deactivate_stale_patterns()
        deleted = self.delete_old_inactive()
        return {
            "deactivated": deactivated,
            "deleted": deleted,
        }

    def get_lifecycle_stats(self) -> Dict[str, int]:
        """Get pattern lifecycle statistics.

        Returns:
            Dict with counts by status
        """
        rows = self._conn.execute(
            """
            SELECT
                COALESCE(status, 'active') as status,
                COUNT(*) as count
            FROM patterns
            GROUP BY status
            """
        ).fetchall()

        stats = {"active": 0, "inactive": 0, "deleted": 0}
        for row in rows:
            status = row["status"] or "active"
            stats[status] = row["count"]

        return stats

    def get_aggregates(self, pattern_id: int) -> Optional[PatternAggregates]:
        """Get pre-computed aggregates for a pattern.

        This is a fast read of stored aggregates - no calculation needed.

        Args:
            pattern_id: Pattern to query

        Returns:
            PatternAggregates or None if pattern not found
        """
        row = self._conn.execute(
            """
            SELECT
                id, usage_count, last_used_ts, accuracy, up_accuracy, down_accuracy,
                avg_return, cumulative_return, win_rate, conf_wrong_rate,
                hit_count, up_hit_count, up_total_count, down_hit_count, down_total_count,
                high_conf_count, high_conf_wrong_count, decision_count, top_decisions_count,
                COALESCE(status, 'active') as status
            FROM patterns
            WHERE id = ?
            """,
            (pattern_id,),
        ).fetchone()

        if not row:
            return None

        return PatternAggregates(
            pattern_id=row["id"],
            total_uses=row["usage_count"] or 0,
            last_used_ts=row["last_used_ts"] or 0,
            accuracy=row["accuracy"] or 0.5,
            up_accuracy=row["up_accuracy"] or 0.5,
            down_accuracy=row["down_accuracy"] or 0.5,
            avg_return=row["avg_return"] or 0.0,
            cumulative_return=row["cumulative_return"] or 0.0,
            win_rate=row["win_rate"] or 0.5,
            conf_wrong_rate=row["conf_wrong_rate"] or 0.0,
            hit_count=row["hit_count"] or 0,
            up_hit_count=row["up_hit_count"] or 0,
            up_total_count=row["up_total_count"] or 0,
            down_hit_count=row["down_hit_count"] or 0,
            down_total_count=row["down_total_count"] or 0,
            high_conf_count=row["high_conf_count"] or 0,
            high_conf_wrong_count=row["high_conf_wrong_count"] or 0,
            decision_count=row["decision_count"] or 0,
            top_decisions_count=row["top_decisions_count"] or 0,
            status=row["status"],
        )

    def get_aggregates_batch(self, pattern_ids: List[int]) -> Dict[int, PatternAggregates]:
        """Get aggregates for multiple patterns in one query.

        Args:
            pattern_ids: List of pattern IDs to query

        Returns:
            Dict mapping pattern_id to PatternAggregates
        """
        if not pattern_ids:
            return {}

        placeholders = ",".join("?" * len(pattern_ids))
        rows = self._conn.execute(
            f"""
            SELECT
                id, usage_count, last_used_ts, accuracy, up_accuracy, down_accuracy,
                avg_return, cumulative_return, win_rate, conf_wrong_rate,
                hit_count, up_hit_count, up_total_count, down_hit_count, down_total_count,
                high_conf_count, high_conf_wrong_count, decision_count, top_decisions_count,
                COALESCE(status, 'active') as status
            FROM patterns
            WHERE id IN ({placeholders})
            """,
            pattern_ids,
        ).fetchall()

        result = {}
        for row in rows:
            result[row["id"]] = PatternAggregates(
                pattern_id=row["id"],
                total_uses=row["usage_count"] or 0,
                last_used_ts=row["last_used_ts"] or 0,
                accuracy=row["accuracy"] or 0.5,
                up_accuracy=row["up_accuracy"] or 0.5,
                down_accuracy=row["down_accuracy"] or 0.5,
                avg_return=row["avg_return"] or 0.0,
                cumulative_return=row["cumulative_return"] or 0.0,
                win_rate=row["win_rate"] or 0.5,
                conf_wrong_rate=row["conf_wrong_rate"] or 0.0,
                hit_count=row["hit_count"] or 0,
                up_hit_count=row["up_hit_count"] or 0,
                up_total_count=row["up_total_count"] or 0,
                down_hit_count=row["down_hit_count"] or 0,
                down_total_count=row["down_total_count"] or 0,
                high_conf_count=row["high_conf_count"] or 0,
                high_conf_wrong_count=row["high_conf_wrong_count"] or 0,
                decision_count=row["decision_count"] or 0,
                top_decisions_count=row["top_decisions_count"] or 0,
                status=row["status"],
            )

        return result

    def get_model_aggregates(
        self, pattern_id: int, model_name: str
    ) -> Optional[PatternModelAggregates]:
        """Get per-model aggregates for a pattern."""
        row = self._conn.execute(
            """
            SELECT
                pattern_id, model_name, usage_count, last_used_ts,
                accuracy, up_accuracy, down_accuracy,
                avg_return, cumulative_return, win_rate, conf_wrong_rate,
                hit_count, up_hit_count, up_total_count, down_hit_count, down_total_count,
                high_conf_count, high_conf_wrong_count
            FROM pattern_model_stats
            WHERE pattern_id = ? AND model_name = ?
            """,
            (pattern_id, model_name),
        ).fetchone()

        if not row:
            return None

        return PatternModelAggregates(
            pattern_id=row["pattern_id"],
            model_name=row["model_name"],
            total_uses=row["usage_count"] or 0,
            last_used_ts=row["last_used_ts"] or 0,
            accuracy=row["accuracy"] or 0.5,
            up_accuracy=row["up_accuracy"] or 0.5,
            down_accuracy=row["down_accuracy"] or 0.5,
            avg_return=row["avg_return"] or 0.0,
            cumulative_return=row["cumulative_return"] or 0.0,
            win_rate=row["win_rate"] or 0.5,
            conf_wrong_rate=row["conf_wrong_rate"] or 0.0,
            hit_count=row["hit_count"] or 0,
            up_hit_count=row["up_hit_count"] or 0,
            up_total_count=row["up_total_count"] or 0,
            down_hit_count=row["down_hit_count"] or 0,
            down_total_count=row["down_total_count"] or 0,
            high_conf_count=row["high_conf_count"] or 0,
            high_conf_wrong_count=row["high_conf_wrong_count"] or 0,
        )

    def get_model_aggregates_batch(
        self, pattern_ids: List[int], model_name: str
    ) -> Dict[int, PatternModelAggregates]:
        """Get per-model aggregates for multiple patterns."""
        if not pattern_ids:
            return {}

        placeholders = ",".join("?" * len(pattern_ids))
        rows = self._conn.execute(
            f"""
            SELECT
                pattern_id, model_name, usage_count, last_used_ts,
                accuracy, up_accuracy, down_accuracy,
                avg_return, cumulative_return, win_rate, conf_wrong_rate,
                hit_count, up_hit_count, up_total_count, down_hit_count, down_total_count,
                high_conf_count, high_conf_wrong_count
            FROM pattern_model_stats
            WHERE model_name = ? AND pattern_id IN ({placeholders})
            """,
            [model_name] + pattern_ids,
        ).fetchall()

        result: Dict[int, PatternModelAggregates] = {}
        for row in rows:
            result[row["pattern_id"]] = PatternModelAggregates(
                pattern_id=row["pattern_id"],
                model_name=row["model_name"],
                total_uses=row["usage_count"] or 0,
                last_used_ts=row["last_used_ts"] or 0,
                accuracy=row["accuracy"] or 0.5,
                up_accuracy=row["up_accuracy"] or 0.5,
                down_accuracy=row["down_accuracy"] or 0.5,
                avg_return=row["avg_return"] or 0.0,
                cumulative_return=row["cumulative_return"] or 0.0,
                win_rate=row["win_rate"] or 0.5,
                conf_wrong_rate=row["conf_wrong_rate"] or 0.0,
                hit_count=row["hit_count"] or 0,
                up_hit_count=row["up_hit_count"] or 0,
                up_total_count=row["up_total_count"] or 0,
                down_hit_count=row["down_hit_count"] or 0,
                down_total_count=row["down_total_count"] or 0,
                high_conf_count=row["high_conf_count"] or 0,
                high_conf_wrong_count=row["high_conf_wrong_count"] or 0,
            )

        return result

    def get_feature_bucket_stats(
        self, pattern_id: int, model_name: str, feature: str
    ) -> Dict[str, Dict[str, float]]:
        """Get bucket stats for a feature within a pattern/model."""
        rows = self._conn.execute(
            """
            SELECT bucket, total_count, hit_count, cumulative_return
            FROM pattern_feature_buckets
            WHERE pattern_id = ? AND model_name = ? AND feature = ?
            """,
            (pattern_id, model_name, feature),
        ).fetchall()

        stats: Dict[str, Dict[str, float]] = {}
        for row in rows:
            total = row["total_count"] or 0
            hit = row["hit_count"] or 0
            cumulative_return = row["cumulative_return"] or 0.0
            accuracy = (hit / total) if total else 0.5
            avg_return = (cumulative_return / total) if total else 0.0
            stats[row["bucket"]] = {
                "count": float(total),
                "accuracy": float(accuracy),
                "avg_return": float(avg_return),
            }
        return stats

    def get_feature_bucket_stats_all(
        self, pattern_id: int, model_name: str
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Get bucket stats for all features within a pattern/model."""
        rows = self._conn.execute(
            """
            SELECT feature, bucket, total_count, hit_count, cumulative_return
            FROM pattern_feature_buckets
            WHERE pattern_id = ? AND model_name = ?
            """,
            (pattern_id, model_name),
        ).fetchall()

        stats: Dict[str, Dict[str, Dict[str, float]]] = {}
        for row in rows:
            total = row["total_count"] or 0
            hit = row["hit_count"] or 0
            cumulative_return = row["cumulative_return"] or 0.0
            accuracy = (hit / total) if total else 0.5
            avg_return = (cumulative_return / total) if total else 0.0
            feature = row["feature"]
            bucket = row["bucket"]
            stats.setdefault(feature, {})[bucket] = {
                "count": float(total),
                "accuracy": float(accuracy),
                "avg_return": float(avg_return),
            }
        return stats

    def get_temporal_stats(
        self,
        pattern_id: int,
        model_name: str,
        hour: int,
        session: str,
        day_of_week: int,
    ) -> Optional[Dict[str, float]]:
        """Get temporal stats for a pattern/model at a specific time slice."""
        row = self._conn.execute(
            """
            SELECT total_count, hit_count, cumulative_return
            FROM pattern_temporal_stats
            WHERE pattern_id = ? AND model_name = ? AND hour = ? AND session = ? AND day_of_week = ?
            """,
            (pattern_id, model_name, hour, session, day_of_week),
        ).fetchone()
        if not row:
            return None
        total = row["total_count"] or 0
        hit = row["hit_count"] or 0
        cumulative_return = row["cumulative_return"] or 0.0
        accuracy = (hit / total) if total else 0.5
        avg_return = (cumulative_return / total) if total else 0.0
        return {
            "count": float(total),
            "accuracy": float(accuracy),
            "avg_return": float(avg_return),
        }

    def get_temporal_stats_by_session(
        self, pattern_id: int, model_name: str, session: str
    ) -> Optional[Dict[str, float]]:
        row = self._conn.execute(
            """
            SELECT SUM(total_count) as total_count,
                   SUM(hit_count) as hit_count,
                   SUM(cumulative_return) as cumulative_return
            FROM pattern_temporal_stats
            WHERE pattern_id = ? AND model_name = ? AND session = ?
            """,
            (pattern_id, model_name, session),
        ).fetchone()
        if not row or row["total_count"] is None:
            return None
        total = row["total_count"] or 0
        hit = row["hit_count"] or 0
        cumulative_return = row["cumulative_return"] or 0.0
        accuracy = (hit / total) if total else 0.5
        avg_return = (cumulative_return / total) if total else 0.0
        return {
            "count": float(total),
            "accuracy": float(accuracy),
            "avg_return": float(avg_return),
        }

    def get_temporal_stats_by_hour(
        self, pattern_id: int, model_name: str, hour: int
    ) -> Optional[Dict[str, float]]:
        row = self._conn.execute(
            """
            SELECT SUM(total_count) as total_count,
                   SUM(hit_count) as hit_count,
                   SUM(cumulative_return) as cumulative_return
            FROM pattern_temporal_stats
            WHERE pattern_id = ? AND model_name = ? AND hour = ?
            """,
            (pattern_id, model_name, hour),
        ).fetchone()
        if not row or row["total_count"] is None:
            return None
        total = row["total_count"] or 0
        hit = row["hit_count"] or 0
        cumulative_return = row["cumulative_return"] or 0.0
        accuracy = (hit / total) if total else 0.5
        avg_return = (cumulative_return / total) if total else 0.0
        return {
            "count": float(total),
            "accuracy": float(accuracy),
            "avg_return": float(avg_return),
        }

    def get_temporal_stats_by_day(
        self, pattern_id: int, model_name: str, day_of_week: int
    ) -> Optional[Dict[str, float]]:
        row = self._conn.execute(
            """
            SELECT SUM(total_count) as total_count,
                   SUM(hit_count) as hit_count,
                   SUM(cumulative_return) as cumulative_return
            FROM pattern_temporal_stats
            WHERE pattern_id = ? AND model_name = ? AND day_of_week = ?
            """,
            (pattern_id, model_name, day_of_week),
        ).fetchone()
        if not row or row["total_count"] is None:
            return None
        total = row["total_count"] or 0
        hit = row["hit_count"] or 0
        cumulative_return = row["cumulative_return"] or 0.0
        accuracy = (hit / total) if total else 0.5
        avg_return = (cumulative_return / total) if total else 0.0
        return {
            "count": float(total),
            "accuracy": float(accuracy),
            "avg_return": float(avg_return),
        }

    def get_pattern_info(self, pattern_id: int) -> Optional[Dict[str, Any]]:
        """Get full pattern information including key and conditions.

        Args:
            pattern_id: Pattern to query

        Returns:
            Dict with pattern_key, conditions, aggregates, or None
        """
        row = self._conn.execute(
            """
            SELECT pattern_key, conditions_blob, status, conditions_version
            FROM patterns
            WHERE id = ?
            """,
            (pattern_id,),
        ).fetchone()

        if not row:
            return None

        aggregates = self.get_aggregates(pattern_id)
        conditions = decode_payload(row["conditions_blob"]) if row["conditions_blob"] else {}

        return {
            "pattern_id": pattern_id,
            "pattern_key": row["pattern_key"],
            "conditions": conditions,
            "status": row["status"] or "active",
            "conditions_version": row["conditions_version"] or 1,
            "aggregates": aggregates,
        }

    @staticmethod
    def calc_significance(
        hit: bool,
        confidence: float,
        return_pct: float,
        high_conf_threshold: float = 0.65,
    ) -> float:
        """Calculate significance score for a decision.

        Higher score = more important to keep in history.

        Factors:
        - Absolute return (larger moves are more informative)
        - Correct high-confidence predictions (validates model)
        - Edge cases (unusual confidence + return combinations)

        Args:
            hit: Whether prediction was correct
            confidence: Model confidence (0-1)
            return_pct: Actual return percentage
            high_conf_threshold: Threshold for "high confidence"

        Returns:
            Significance score (higher = more important)
        """
        # Base: absolute return scaled
        base = abs(return_pct) * 10000  # Scale so 0.01% return = 1.0

        # Bonus for correct predictions
        if hit:
            base += confidence * 50  # High confidence + correct = very valuable

        # Bonus for high-confidence correct predictions
        if hit and confidence >= high_conf_threshold:
            base += 30  # Extra bonus for validated high confidence

        # Penalty for high-confidence wrong predictions (but keep some for learning)
        if not hit and confidence >= high_conf_threshold:
            base += 20  # Still valuable for learning what NOT to do

        # Edge case bonus: extreme returns regardless of hit
        if abs(return_pct) > 0.005:  # More than 0.5% move
            base += 25

        return base

    def _enforce_decision_limit(self, pattern_id: int) -> int:
        """Enforce maximum decision count per pattern.

        Keeps top TOP_DECISIONS_COUNT decisions and removes lowest significance
        decisions when count exceeds max_decisions config.

        Returns:
            Number of decisions deleted
        """
        # Get current decision count
        row = self._conn.execute(
            "SELECT decision_count FROM patterns WHERE id = ?",
            (pattern_id,),
        ).fetchone()

        if not row:
            return 0

        current_count = row["decision_count"] or 0

        if current_count <= self._max_decisions:
            return 0

        # How many to delete
        delete_count = current_count - self._max_decisions

        # FIXED: First collect IDs of events to delete
        events_to_delete = self._conn.execute(
            """
            SELECT id FROM pattern_events
            WHERE pattern_id = ? AND is_top_decision = 0
            ORDER BY significance ASC
            LIMIT ?
            """,
            (pattern_id, delete_count),
        ).fetchall()

        if not events_to_delete:
            return 0

        event_ids = [row["id"] for row in events_to_delete]
        placeholders = ",".join("?" * len(event_ids))

        # FIXED: Delete snapshots ONLY for events being removed
        self._conn.execute(
            f"DELETE FROM pattern_event_snapshots WHERE event_id IN ({placeholders})",
            event_ids,
        )

        # Delete the events themselves
        self._conn.execute(
            f"DELETE FROM pattern_events WHERE id IN ({placeholders})",
            event_ids,
        )

        # Update decision count
        new_count = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM pattern_events WHERE pattern_id = ?",
            (pattern_id,),
        ).fetchone()["cnt"]

        self._conn.execute(
            "UPDATE patterns SET decision_count = ? WHERE id = ?",
            (new_count, pattern_id),
        )

        return len(event_ids)

    def _update_top_decisions(self, pattern_id: int) -> None:
        """Update the pool of top decisions for a pattern.

        Marks the top N decisions by significance as is_top_decision = 1,
        and all others as 0 (N = config pattern.top_decisions_count).
        """
        # First, reset all to 0
        self._conn.execute(
            "UPDATE pattern_events SET is_top_decision = 0 WHERE pattern_id = ?",
            (pattern_id,),
        )

        # Then mark top N as 1
        self._conn.execute(
            """
            UPDATE pattern_events
            SET is_top_decision = 1
            WHERE id IN (
                SELECT id FROM pattern_events
                WHERE pattern_id = ?
                ORDER BY significance DESC
                LIMIT ?
            )
            """,
            (pattern_id, self._top_decisions_count),
        )

        # Update top_decisions_count
        top_count = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM pattern_events WHERE pattern_id = ? AND is_top_decision = 1",
            (pattern_id,),
        ).fetchone()["cnt"]

        self._conn.execute(
            "UPDATE patterns SET top_decisions_count = ? WHERE id = ?",
            (top_count, pattern_id),
        )

    def get_top_decisions(self, pattern_id: int, limit: int = 100) -> List[Dict]:
        """Get top decisions for a pattern.

        Args:
            pattern_id: Pattern to query
            limit: Maximum number of decisions to return

        Returns:
            List of top decision events with their data
        """
        rows = self._conn.execute(
            """
            SELECT pe.*, pes.features_blob, pes.model_state_blob
            FROM pattern_events pe
            LEFT JOIN pattern_event_snapshots pes ON pes.event_id = pe.id
            WHERE pe.pattern_id = ? AND pe.is_top_decision = 1
            ORDER BY pe.significance DESC
            LIMIT ?
            """,
            (pattern_id, limit),
        ).fetchall()

        results = []
        for row in rows:
            event = {
                "id": row["id"],
                "model_name": row["model_name"],
                "event_ts": row["event_ts"],
                "decision": row["decision"],
                "confidence": row["confidence"],
                "hit": row["hit"],
                "return_pct": row["return_pct"],
                "price": row["price"],
                "significance": row["significance"],
            }

            # Add event blob data
            if row["event_blob"]:
                event.update(decode_payload(row["event_blob"]))

            # Add features snapshot if available
            if row["features_blob"]:
                event["features_snapshot"] = decode_payload(row["features_blob"])

            # Add model state if available
            if row["model_state_blob"]:
                event["model_state"] = decode_payload(row["model_state_blob"])

            results.append(event)

        return results

    def record_usage(
        self,
        pattern_id: int,
        model_name: str,
        event: Dict[str, object],
        event_ts: int,
        features_snapshot: Optional[Dict[str, float]] = None,
    ) -> int:
        """Record a model usage event and update pattern aggregates.

        Args:
            pattern_id: Target pattern
            model_name: Name of the model making prediction
            event: Event data containing forecast, outcome, etc.
            event_ts: Event timestamp
            features_snapshot: Optional full features snapshot (stored for 100%)

        Returns:
            Event ID
        """
        # Extract key fields from event for denormalized storage
        forecast = event.get("forecast", {})
        outcome = event.get("outcome", {})
        model_state = event.get("model_state", {})

        decision = forecast.get("direction", "FLAT")
        confidence = float(forecast.get("prob_up", 0.5))
        if forecast.get("direction") == "DOWN":
            confidence = float(forecast.get("prob_down", 0.5))

        hit = 1 if outcome.get("hit", False) else 0
        return_pct = float(outcome.get("return_pct", 0.0))
        price = float(event.get("price", 0.0)) if "price" in event else None
        regime = event.get("regime")

        # Calculate significance score (using config threshold)
        significance = self.calc_significance(
            hit=bool(hit),
            confidence=confidence,
            return_pct=return_pct,
            high_conf_threshold=self._high_conf_threshold,
        )

        # Store event with denormalized fields
        blob = encode_payload(event)
        cursor = self._conn.execute(
            """
            INSERT INTO pattern_events
            (pattern_id, model_name, event_ts, event_blob, decision, confidence, hit, return_pct, price, regime, significance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (pattern_id, model_name, event_ts, blob, decision, confidence, hit, return_pct, price, regime, significance),
        )
        event_id = cursor.lastrowid

        # Store features snapshot (100% of events)
        store_snapshot = features_snapshot is not None and (
            self._snapshot_rate >= 1.0 or random.random() < self._snapshot_rate
        )
        if store_snapshot:
            features_blob = encode_payload(features_snapshot)
            model_state_blob = encode_payload(model_state) if model_state else None
            self._conn.execute(
                """
                INSERT INTO pattern_event_snapshots (event_id, features_blob, model_state_blob)
                VALUES (?, ?, ?)
                """,
                (event_id, features_blob, model_state_blob),
            )

        # Update pattern aggregates atomically
        # FLAT is model uncertainty - don't count in directional accuracy
        is_directional = decision in ("UP", "DOWN")

        if is_directional:
            if model_name != "ENSEMBLE" or self._include_ensemble_global:
                self._update_pattern_aggregates(
                    pattern_id, decision, hit, return_pct, confidence
                )
            self._update_model_aggregates(
                pattern_id, model_name, decision, hit, return_pct, confidence
            )

            if features_snapshot is not None:
                self._update_feature_buckets(
                    pattern_id, model_name, features_snapshot, hit, return_pct
                )

            self._update_temporal_stats(
                pattern_id, model_name, event_ts, hit, return_pct
            )

        self._conn.commit()

        # Periodic maintenance: enforce limits and update top decisions
        # Only check every 100 events to avoid performance hit
        decision_count = self._conn.execute(
            "SELECT decision_count FROM patterns WHERE id = ?",
            (pattern_id,),
        ).fetchone()["decision_count"] or 0

        if decision_count > 0 and decision_count % 100 == 0:
            self._enforce_decision_limit(pattern_id)
            self._update_top_decisions(pattern_id)
            self._conn.commit()

        return event_id

    def _update_pattern_aggregates(
        self,
        pattern_id: int,
        decision: str,
        hit: int,
        return_pct: float,
        confidence: float,
    ) -> None:
        """Update pattern aggregates atomically.

        Uses SQL to compute running averages efficiently.
        """
        now = int(time.time())

        # Determine direction flags
        is_up = 1 if decision == "UP" else 0
        is_down = 1 if decision == "DOWN" else 0
        is_high_conf = 1 if confidence >= self._high_conf_threshold else 0
        is_high_conf_wrong = 1 if is_high_conf and not hit else 0

        self._conn.execute(
            """
            UPDATE patterns SET
                usage_count = usage_count + 1,
                decision_count = decision_count + 1,
                last_used_ts = ?,
                hit_count = hit_count + ?,
                up_total_count = up_total_count + ?,
                up_hit_count = up_hit_count + CASE WHEN ? = 1 AND ? = 1 THEN 1 ELSE 0 END,
                down_total_count = down_total_count + ?,
                down_hit_count = down_hit_count + CASE WHEN ? = 1 AND ? = 1 THEN 1 ELSE 0 END,
                high_conf_count = high_conf_count + ?,
                high_conf_wrong_count = high_conf_wrong_count + ?,
                cumulative_return = cumulative_return + ?,
                accuracy = CASE WHEN (usage_count + 1) > 0
                    THEN CAST(hit_count + ? AS REAL) / (usage_count + 1)
                    ELSE 0.5 END,
                up_accuracy = CASE WHEN (up_total_count + ?) > 0
                    THEN CAST(up_hit_count + CASE WHEN ? = 1 AND ? = 1 THEN 1 ELSE 0 END AS REAL) / (up_total_count + ?)
                    ELSE 0.5 END,
                down_accuracy = CASE WHEN (down_total_count + ?) > 0
                    THEN CAST(down_hit_count + CASE WHEN ? = 1 AND ? = 1 THEN 1 ELSE 0 END AS REAL) / (down_total_count + ?)
                    ELSE 0.5 END,
                avg_return = CASE WHEN (usage_count + 1) > 0
                    THEN (cumulative_return + ?) / (usage_count + 1)
                    ELSE 0.0 END,
                win_rate = CASE WHEN (usage_count + 1) > 0
                    THEN CAST(hit_count + ? AS REAL) / (usage_count + 1)
                    ELSE 0.5 END,
                conf_wrong_rate = CASE WHEN (high_conf_count + ?) > 0
                    THEN CAST(high_conf_wrong_count + ? AS REAL) / (high_conf_count + ?)
                    ELSE 0.0 END
            WHERE id = ?
            """,
            (
                now,
                hit,
                is_up,
                is_up, hit,
                is_down,
                is_down, hit,
                is_high_conf,
                is_high_conf_wrong,
                return_pct,
                hit,
                is_up, is_up, hit, is_up,
                is_down, is_down, hit, is_down,
                return_pct,
                hit,
                is_high_conf, is_high_conf_wrong, is_high_conf,
                pattern_id,
            ),
        )

    def _update_model_aggregates(
        self,
        pattern_id: int,
        model_name: str,
        decision: str,
        hit: int,
        return_pct: float,
        confidence: float,
    ) -> None:
        """Update per-model aggregates for a pattern."""
        now = int(time.time())
        is_up = 1 if decision == "UP" else 0
        is_down = 1 if decision == "DOWN" else 0
        is_high_conf = 1 if confidence >= self._high_conf_threshold else 0
        is_high_conf_wrong = 1 if is_high_conf and not hit else 0
        up_hit = hit if is_up else 0
        down_hit = hit if is_down else 0
        accuracy = float(hit)
        up_accuracy = float(hit) if is_up else 0.5
        down_accuracy = float(hit) if is_down else 0.5
        conf_wrong_rate = float(is_high_conf_wrong) if is_high_conf else 0.0

        self._conn.execute(
            """
            INSERT INTO pattern_model_stats (
                pattern_id,
                model_name,
                usage_count,
                last_used_ts,
                hit_count,
                up_hit_count,
                up_total_count,
                down_hit_count,
                down_total_count,
                high_conf_count,
                high_conf_wrong_count,
                accuracy,
                up_accuracy,
                down_accuracy,
                avg_return,
                cumulative_return,
                win_rate,
                conf_wrong_rate
            )
            VALUES (
                ?, ?, 1, ?,
                ?, ?,
                ?, ?,
                ?, ?,
                ?, ?,
                ?, ?,
                ?, ?,
                ?, ?
            )
            ON CONFLICT(pattern_id, model_name) DO UPDATE SET
                usage_count = usage_count + 1,
                last_used_ts = ?,
                hit_count = hit_count + ?,
                up_total_count = up_total_count + ?,
                up_hit_count = up_hit_count + CASE WHEN ? = 1 AND ? = 1 THEN 1 ELSE 0 END,
                down_total_count = down_total_count + ?,
                down_hit_count = down_hit_count + CASE WHEN ? = 1 AND ? = 1 THEN 1 ELSE 0 END,
                high_conf_count = high_conf_count + ?,
                high_conf_wrong_count = high_conf_wrong_count + ?,
                cumulative_return = cumulative_return + ?,
                accuracy = CASE WHEN (usage_count + 1) > 0
                    THEN CAST(hit_count + ? AS REAL) / (usage_count + 1)
                    ELSE 0.5 END,
                up_accuracy = CASE WHEN (up_total_count + ?) > 0
                    THEN CAST(up_hit_count + CASE WHEN ? = 1 AND ? = 1 THEN 1 ELSE 0 END AS REAL) / (up_total_count + ?)
                    ELSE 0.5 END,
                down_accuracy = CASE WHEN (down_total_count + ?) > 0
                    THEN CAST(down_hit_count + CASE WHEN ? = 1 AND ? = 1 THEN 1 ELSE 0 END AS REAL) / (down_total_count + ?)
                    ELSE 0.5 END,
                avg_return = CASE WHEN (usage_count + 1) > 0
                    THEN (cumulative_return + ?) / (usage_count + 1)
                    ELSE 0.0 END,
                win_rate = CASE WHEN (usage_count + 1) > 0
                    THEN CAST(hit_count + ? AS REAL) / (usage_count + 1)
                    ELSE 0.5 END,
                conf_wrong_rate = CASE WHEN (high_conf_count + ?) > 0
                    THEN CAST(high_conf_wrong_count + ? AS REAL) / (high_conf_count + ?)
                    ELSE 0.0 END
            """,
            (
                pattern_id,
                model_name,
                now,
                hit,
                up_hit,
                is_up,
                down_hit,
                is_down,
                is_high_conf,
                is_high_conf_wrong,
                accuracy,
                up_accuracy,
                down_accuracy,
                return_pct,
                return_pct,
                accuracy,
                conf_wrong_rate,
                now,
                hit,
                is_up,
                is_up, hit,
                is_down,
                is_down, hit,
                is_high_conf,
                is_high_conf_wrong,
                return_pct,
                hit,
                is_up, is_up, hit, is_up,
                is_down, is_down, hit, is_down,
                return_pct,
                hit,
                is_high_conf, is_high_conf_wrong, is_high_conf,
            ),
        )

    def _update_feature_buckets(
        self,
        pattern_id: int,
        model_name: str,
        features: Dict[str, float],
        hit: int,
        return_pct: float,
    ) -> None:
        rows = []
        for feature in FEATURE_BUCKETS.keys():
            value = features.get(feature)
            bucket = get_feature_bucket(feature, value)
            if bucket is None:
                continue
            rows.append((pattern_id, model_name, feature, bucket, hit, return_pct))

        if not rows:
            return

        self._conn.executemany(
            """
            INSERT INTO pattern_feature_buckets (
                pattern_id,
                model_name,
                feature,
                bucket,
                total_count,
                hit_count,
                cumulative_return
            )
            VALUES (?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(pattern_id, model_name, feature, bucket) DO UPDATE SET
                total_count = total_count + 1,
                hit_count = hit_count + excluded.hit_count,
                cumulative_return = cumulative_return + excluded.cumulative_return
            """,
            rows,
        )

    def _update_temporal_stats(
        self,
        pattern_id: int,
        model_name: str,
        event_ts: int,
        hit: int,
        return_pct: float,
    ) -> None:
        from datetime import datetime

        dt = datetime.utcfromtimestamp(event_ts)
        hour = dt.hour
        session = get_trading_session(hour)
        day_of_week = dt.weekday()

        self._conn.execute(
            """
            INSERT INTO pattern_temporal_stats (
                pattern_id,
                model_name,
                hour,
                session,
                day_of_week,
                total_count,
                hit_count,
                cumulative_return
            )
            VALUES (?, ?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(pattern_id, model_name, hour, session, day_of_week) DO UPDATE SET
                total_count = total_count + 1,
                hit_count = hit_count + excluded.hit_count,
                cumulative_return = cumulative_return + excluded.cumulative_return
            """,
            (
                pattern_id,
                model_name,
                hour,
                session,
                day_of_week,
                hit,
                return_pct,
            ),
        )

    def get_events(
        self,
        pattern_id: int,
        limit: int = 1000,
        max_ts: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> list:
        """Get events for a pattern, most recent first.

        Args:
            pattern_id: Pattern to query
            limit: Maximum events to return
            max_ts: Optional timestamp ceiling to prevent data leakage in backtest

        Returns:
            List of event dicts
        """
        if max_ts is not None:
            # Time-bounded query for backtest (prevents data leakage)
            if model_name:
                rows = self._conn.execute(
                    """
                    SELECT event_ts, event_blob FROM pattern_events
                    WHERE pattern_id = ? AND event_ts <= ? AND model_name = ?
                    ORDER BY event_ts DESC
                    LIMIT ?
                    """,
                    (pattern_id, max_ts, model_name, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT event_ts, event_blob FROM pattern_events
                    WHERE pattern_id = ? AND event_ts <= ?
                    ORDER BY event_ts DESC
                    LIMIT ?
                    """,
                    (pattern_id, max_ts, limit),
                ).fetchall()
        else:
            # Unbounded query for live mode
            if model_name:
                rows = self._conn.execute(
                    """
                    SELECT event_ts, event_blob FROM pattern_events
                    WHERE pattern_id = ? AND model_name = ?
                    ORDER BY event_ts DESC
                    LIMIT ?
                    """,
                    (pattern_id, model_name, limit),
                ).fetchall()
            else:
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

    def get_model_usage_count(self, pattern_id: int, model_name: str) -> int:
        """Get total usage count for a pattern/model pair."""
        row = self._conn.execute(
            """
            SELECT usage_count FROM pattern_model_stats
            WHERE pattern_id = ? AND model_name = ?
            """,
            (pattern_id, model_name),
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
        self._cache: Dict[Tuple[int, Optional[str], bool], Dict[str, object]] = {}

    def get_pattern_stats(
        self,
        pattern_id: int,
        use_decay: bool = True,
        max_ts: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """Get performance statistics for a pattern.

        Args:
            pattern_id: Pattern to analyze
            use_decay: Apply time decay weighting
            max_ts: Optional timestamp ceiling to prevent data leakage in backtest.
                    When provided, only events with event_ts <= max_ts are used.

        Returns:
            Dict with keys: total_uses, accuracy, up_accuracy, down_accuracy,
                           confidence (in the estimate)
        """
        # Check cache first (only use cache if no time bound specified)
        cache_key = (pattern_id, model_name, use_decay)
        if max_ts is None and cache_key in self._cache:
            cached = self._cache[cache_key]
            cached_count = int(cached["usage_count"])
            if model_name:
                current_count = self._store.get_model_usage_count(pattern_id, model_name)
            else:
                current_count = self._store.get_usage_count(pattern_id)
            if current_count == cached_count:
                return cached["stats"]  # type: ignore[return-value]

        # Fast path: use stored aggregates when no decay/time bound
        if max_ts is None and not use_decay:
            if model_name:
                agg = self._store.get_model_aggregates(pattern_id, model_name)
            else:
                agg = self._store.get_aggregates(pattern_id)
            if agg:
                stats = {
                    "total_uses": agg.total_uses,
                    "accuracy": agg.accuracy,
                    "up_accuracy": agg.up_accuracy,
                    "down_accuracy": agg.down_accuracy,
                    "confidence": min(agg.total_uses / 50.0, 1.0),
                }
                self._cache[cache_key] = {
                    "usage_count": agg.total_uses,
                    "stats": stats,
                }
                return stats

        # FIXED: Pass max_ts to prevent data leakage in backtest
        events = self._store.get_events(pattern_id, max_ts=max_ts, model_name=model_name)

        if not events:
            return {
                "total_uses": 0,
                "accuracy": 0.50,
                "up_accuracy": 0.50,
                "down_accuracy": 0.50,
                "confidence": 0.0,
            }

        # Use max_ts as "now" for decay calculation if provided (backtest mode)
        now = max_ts if max_ts is not None else time.time()
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

        # Cache the result (only if not time-bounded)
        if max_ts is None:
            if model_name:
                current_count = self._store.get_model_usage_count(pattern_id, model_name)
            else:
                current_count = self._store.get_usage_count(pattern_id)
            self._cache[cache_key] = {
                "usage_count": current_count,
                "stats": stats,
            }
        return stats

    def get_pattern_bias(
        self,
        pattern_id: int,
        max_ts: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> Optional[str]:
        """Determine if pattern has a directional bias.

        Args:
            pattern_id: Pattern to analyze
            max_ts: Optional timestamp ceiling for backtest

        Returns:
            'UP' if pattern works better for UP predictions
            'DOWN' if pattern works better for DOWN predictions
            None if no clear bias (difference < 5%)
        """
        stats = self.get_pattern_stats(pattern_id, max_ts=max_ts, model_name=model_name)

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

    def should_trust_pattern(
        self,
        pattern_id: int,
        min_uses: int = 20,
        max_ts: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> bool:
        """Check if we have enough data to trust pattern statistics.

        Args:
            pattern_id: Pattern to check
            min_uses: Minimum uses required for trust
            max_ts: Optional timestamp ceiling for backtest
        """
        stats = self.get_pattern_stats(pattern_id, max_ts=max_ts, model_name=model_name)
        return stats["total_uses"] >= min_uses

    def clear_cache(self) -> None:
        """Clear the statistics cache."""
        self._cache.clear()
