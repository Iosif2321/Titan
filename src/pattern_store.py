from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .config import PatternConfig
from .types import Direction
from .utils import now_ms


@dataclass
class PatternStats:
    pattern_key: str
    tf: str
    model_id: str
    kind: str
    count: int
    ema_win: Optional[float]
    ema_reward: float
    ema_p_up: Optional[float]
    ema_p_down: Optional[float]
    ema_p_flat: Optional[float]
    streak_bad: int
    last_seen_ts: int


class PatternStore:
    def __init__(self, path: str, config: PatternConfig) -> None:
        self.path = path
        self.config = config
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema()
        self._last_maintenance_ts = 0

    def _ensure_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pattern_stats (
              pattern_key TEXT PRIMARY KEY,
              tf TEXT,
              model_id TEXT,
              kind TEXT,
              count INTEGER,
              ema_win REAL,
              ema_reward REAL,
              ema_p_up REAL,
              ema_p_down REAL,
              ema_p_flat REAL,
              streak_bad INTEGER,
              last_seen_ts INTEGER
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pattern_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts INTEGER,
              tf TEXT,
              model_id TEXT,
              kind TEXT,
              pattern_key TEXT,
              candle_ts INTEGER,
              target_ts INTEGER,
              close_prev REAL,
              close_curr REAL,
              pred_dir TEXT,
              pred_conf REAL,
              fact_dir TEXT,
              reward REAL,
              ret_bps REAL,
              lr_eff REAL,
              anchor_lambda_eff REAL
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_pattern_stats_tf_model ON pattern_stats(tf, model_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_pattern_stats_last_seen ON pattern_stats(last_seen_ts)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_pattern_events_ts ON pattern_events(ts)"
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def _row_to_stat(self, row: sqlite3.Row) -> PatternStats:
        return PatternStats(
            pattern_key=row["pattern_key"],
            tf=row["tf"],
            model_id=row["model_id"],
            kind=row["kind"],
            count=int(row["count"]),
            ema_win=None if row["ema_win"] is None else float(row["ema_win"]),
            ema_reward=float(row["ema_reward"]),
            ema_p_up=None if row["ema_p_up"] is None else float(row["ema_p_up"]),
            ema_p_down=None if row["ema_p_down"] is None else float(row["ema_p_down"]),
            ema_p_flat=None if row["ema_p_flat"] is None else float(row["ema_p_flat"]),
            streak_bad=int(row["streak_bad"]),
            last_seen_ts=int(row["last_seen_ts"]),
        )

    def get_stat(self, pattern_key: str) -> Optional[PatternStats]:
        cur = self.conn.execute(
            "SELECT * FROM pattern_stats WHERE pattern_key = ?", (pattern_key,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_stat(row)

    def _update_context(
        self,
        pattern_key: str,
        tf: str,
        model_id: str,
        fact_dir: Direction,
        reward: float,
        ts: int,
    ) -> PatternStats:
        stat = self.get_stat(pattern_key)
        decay = self.config.ema_decay
        up_val = 1.0 if fact_dir == Direction.UP else 0.0
        down_val = 1.0 if fact_dir == Direction.DOWN else 0.0
        flat_val = 1.0 if fact_dir == Direction.FLAT else 0.0

        if stat is None:
            ema_up = up_val
            ema_down = down_val
            ema_flat = flat_val
            ema_reward = reward
            count = 1
            streak_bad = 0
        else:
            ema_up = decay * (stat.ema_p_up or 0.0) + (1.0 - decay) * up_val
            ema_down = decay * (stat.ema_p_down or 0.0) + (1.0 - decay) * down_val
            ema_flat = decay * (stat.ema_p_flat or 0.0) + (1.0 - decay) * flat_val
            ema_reward = decay * stat.ema_reward + (1.0 - decay) * reward
            count = stat.count + 1
            streak_bad = stat.streak_bad

        self.conn.execute(
            """
            INSERT INTO pattern_stats (
              pattern_key, tf, model_id, kind, count, ema_win, ema_reward,
              ema_p_up, ema_p_down, ema_p_flat, streak_bad, last_seen_ts
            ) VALUES (?, ?, ?, 'context', ?, NULL, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pattern_key) DO UPDATE SET
              count=excluded.count,
              ema_reward=excluded.ema_reward,
              ema_p_up=excluded.ema_p_up,
              ema_p_down=excluded.ema_p_down,
              ema_p_flat=excluded.ema_p_flat,
              streak_bad=excluded.streak_bad,
              last_seen_ts=excluded.last_seen_ts
            """,
            (
                pattern_key,
                tf,
                model_id,
                count,
                ema_reward,
                ema_up,
                ema_down,
                ema_flat,
                streak_bad,
                ts,
            ),
        )
        self.conn.commit()
        return PatternStats(
            pattern_key=pattern_key,
            tf=tf,
            model_id=model_id,
            kind="context",
            count=count,
            ema_win=None,
            ema_reward=ema_reward,
            ema_p_up=ema_up,
            ema_p_down=ema_down,
            ema_p_flat=ema_flat,
            streak_bad=streak_bad,
            last_seen_ts=ts,
        )

    def _update_decision(
        self,
        pattern_key: str,
        tf: str,
        model_id: str,
        pred_dir: Direction,
        fact_dir: Direction,
        reward: float,
        ts: int,
    ) -> PatternStats:
        stat = self.get_stat(pattern_key)
        decay = self.config.ema_decay
        win_val = 1.0 if pred_dir == fact_dir else 0.0

        if stat is None:
            ema_win = win_val
            ema_reward = reward
            count = 1
            streak_bad = 0 if win_val > 0 else 1
        else:
            ema_win = decay * (stat.ema_win or 0.0) + (1.0 - decay) * win_val
            ema_reward = decay * stat.ema_reward + (1.0 - decay) * reward
            count = stat.count + 1
            streak_bad = 0 if win_val > 0 else stat.streak_bad + 1

        self.conn.execute(
            """
            INSERT INTO pattern_stats (
              pattern_key, tf, model_id, kind, count, ema_win, ema_reward,
              ema_p_up, ema_p_down, ema_p_flat, streak_bad, last_seen_ts
            ) VALUES (?, ?, ?, 'decision', ?, ?, ?, NULL, NULL, NULL, ?, ?)
            ON CONFLICT(pattern_key) DO UPDATE SET
              count=excluded.count,
              ema_win=excluded.ema_win,
              ema_reward=excluded.ema_reward,
              streak_bad=excluded.streak_bad,
              last_seen_ts=excluded.last_seen_ts
            """,
            (
                pattern_key,
                tf,
                model_id,
                count,
                ema_win,
                ema_reward,
                streak_bad,
                ts,
            ),
        )
        self.conn.commit()
        return PatternStats(
            pattern_key=pattern_key,
            tf=tf,
            model_id=model_id,
            kind="decision",
            count=count,
            ema_win=ema_win,
            ema_reward=ema_reward,
            ema_p_up=None,
            ema_p_down=None,
            ema_p_flat=None,
            streak_bad=streak_bad,
            last_seen_ts=ts,
        )

    def update_patterns(
        self,
        tf: str,
        model_id: str,
        context_key: str,
        decision_key: str,
        pred_dir: Direction,
        fact_dir: Direction,
        reward: float,
        ts: int,
    ) -> tuple[PatternStats, PatternStats]:
        ctx = self._update_context(context_key, tf, model_id, fact_dir, reward, ts)
        dec = self._update_decision(decision_key, tf, model_id, pred_dir, fact_dir, reward, ts)
        return ctx, dec

    def record_event(
        self,
        kind: str,
        pattern_key: str,
        ts: int,
        tf: str,
        model_id: str,
        candle_ts: int,
        target_ts: int,
        close_prev: float,
        close_curr: float,
        pred_dir: Direction,
        pred_conf: float,
        fact_dir: Direction,
        reward: float,
        ret_bps: float,
        lr_eff: float,
        anchor_lambda_eff: float,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO pattern_events (
              ts, tf, model_id, kind, pattern_key, candle_ts, target_ts,
              close_prev, close_curr, pred_dir, pred_conf, fact_dir,
              reward, ret_bps, lr_eff, anchor_lambda_eff
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                tf,
                model_id,
                kind,
                pattern_key,
                candle_ts,
                target_ts,
                close_prev,
                close_curr,
                pred_dir.value,
                pred_conf,
                fact_dir.value,
                reward,
                ret_bps,
                lr_eff,
                anchor_lambda_eff,
            ),
        )
        self.conn.commit()

    def trust(self, stat: Optional[PatternStats], now_ts: int) -> float:
        if stat is None:
            return 0.0
        support_factor = min(1.0, stat.count / max(1, self.config.support_k))
        tau_ms = self.config.recency_tau_hours * 3600.0 * 1000.0
        recency_factor = math.exp(-(now_ts - stat.last_seen_ts) / tau_ms) if tau_ms > 0 else 1.0
        if stat.kind == "decision":
            quality = stat.ema_win or 0.0
        else:
            quality = max(stat.ema_p_up or 0.0, stat.ema_p_down or 0.0)
        return support_factor * recency_factor * quality

    def is_anti_pattern(self, stat: Optional[PatternStats], now_ts: int) -> bool:
        if stat is None:
            return False
        if stat.kind != "decision":
            return False
        trust_val = self.trust(stat, now_ts)
        if stat.count < self.config.anti_min_support:
            return False
        if (stat.ema_win or 0.0) > self.config.anti_win_threshold:
            return False
        return trust_val >= self.config.anti_trust_threshold

    def maintenance(self, now_ts: Optional[int] = None) -> None:
        now_ts = now_ts or now_ms()
        if self.config.maintenance_seconds <= 0:
            return
        if now_ts - self._last_maintenance_ts < self.config.maintenance_seconds * 1000:
            return

        ttl_ms = self.config.event_ttl_days * 86400 * 1000
        if ttl_ms > 0:
            cutoff = now_ts - ttl_ms
            self.conn.execute("DELETE FROM pattern_events WHERE ts < ?", (cutoff,))

        if self.config.max_events > 0:
            cur = self.conn.execute("SELECT COUNT(*) FROM pattern_events")
            total_events = int(cur.fetchone()[0])
            if total_events > self.config.max_events:
                delete_count = max(int(total_events * 0.2), total_events - self.config.max_events)
                self.conn.execute(
                    """
                    DELETE FROM pattern_events
                    WHERE id IN (
                      SELECT id FROM pattern_events ORDER BY ts ASC LIMIT ?
                    )
                    """,
                    (delete_count,),
                )

        stale_7d = now_ts - 7 * 86400 * 1000
        stale_30d = now_ts - 30 * 86400 * 1000
        self.conn.execute(
            "DELETE FROM pattern_stats WHERE count <= 5 AND last_seen_ts < ?",
            (stale_7d,),
        )
        self.conn.execute(
            "DELETE FROM pattern_stats WHERE count <= 50 AND last_seen_ts < ?",
            (stale_30d,),
        )

        if self.config.max_patterns > 0:
            cur = self.conn.execute("SELECT COUNT(*) FROM pattern_stats")
            total_stats = int(cur.fetchone()[0])
            if total_stats > self.config.max_patterns:
                cur = self.conn.execute(
                    "SELECT pattern_key, count, ema_reward, last_seen_ts FROM pattern_stats"
                )
                candidates = cur.fetchall()
                tau_ms = self.config.recency_tau_hours * 3600.0 * 1000.0
                scored: List[tuple[float, str]] = []
                for row in candidates:
                    count = int(row["count"])
                    support_factor = min(1.0, count / max(1, self.config.support_k))
                    recency_factor = (
                        math.exp(-(now_ts - int(row["last_seen_ts"])) / tau_ms)
                        if tau_ms > 0
                        else 1.0
                    )
                    ema_reward = float(row["ema_reward"])
                    score = support_factor * recency_factor * (0.5 + abs(ema_reward))
                    scored.append((score, row["pattern_key"]))
                scored.sort(key=lambda x: x[0])
                to_remove = total_stats - self.config.max_patterns
                keys = [k for _, k in scored[:to_remove]]
                self.conn.executemany(
                    "DELETE FROM pattern_stats WHERE pattern_key = ?", [(k,) for k in keys]
                )

        self.conn.commit()
        self._last_maintenance_ts = now_ts

    def top_patterns(self, tf: str, model_id: str, limit: int = 10) -> Dict[str, List[Dict[str, object]]]:
        cur = self.conn.execute(
            """
            SELECT * FROM pattern_stats
            WHERE tf = ? AND model_id = ? AND kind = 'decision'
            ORDER BY ema_win DESC, count DESC
            LIMIT ?
            """,
            (tf, model_id, limit),
        )
        good = [dict(row) for row in cur.fetchall()]
        cur = self.conn.execute(
            """
            SELECT * FROM pattern_stats
            WHERE tf = ? AND model_id = ? AND kind = 'decision'
            ORDER BY ema_win ASC, count DESC
            LIMIT ?
            """,
            (tf, model_id, limit * 2),
        )
        now_ts = now_ms()
        anti: List[Dict[str, object]] = []
        for row in cur.fetchall():
            stat = self._row_to_stat(row)
            if self.is_anti_pattern(stat, now_ts):
                anti.append(dict(row))
                if len(anti) >= limit:
                    break
        return {"good": good, "anti": anti}

    def stats_overview(self, tf: str, model_id: str) -> Dict[str, object]:
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM pattern_stats WHERE tf = ? AND model_id = ?",
            (tf, model_id),
        )
        total = int(cur.fetchone()[0])
        return {"total": total}

    def stats_summary(self, tf: str, model_id: str) -> Dict[str, object]:
        cur = self.conn.execute(
            """
            SELECT kind, COUNT(*) as count, AVG(ema_reward) as avg_reward, AVG(ema_win) as avg_win,
                   MAX(last_seen_ts) as last_seen_ts
            FROM pattern_stats
            WHERE tf = ? AND model_id = ?
            GROUP BY kind
            """,
            (tf, model_id),
        )
        summary = {"total": 0, "context": 0, "decision": 0}
        for row in cur.fetchall():
            kind = str(row["kind"])
            count = int(row["count"])
            summary["total"] += count
            summary[kind] = count
            if kind == "decision":
                summary["avg_decision_win"] = float(row["avg_win"] or 0.0)
                summary["avg_decision_reward"] = float(row["avg_reward"] or 0.0)
                summary["last_seen_decision"] = int(row["last_seen_ts"] or 0)
            if kind == "context":
                summary["avg_context_reward"] = float(row["avg_reward"] or 0.0)
                summary["last_seen_context"] = int(row["last_seen_ts"] or 0)
        return summary

    def usage_summary(self, tf: str, model_id: str, since_ts: int) -> Dict[str, object]:
        cur = self.conn.execute(
            """
            SELECT kind, pred_dir, fact_dir, reward, ret_bps, pred_conf, pattern_key
            FROM pattern_events
            WHERE tf = ? AND model_id = ? AND ts >= ?
            """,
            (tf, model_id, since_ts),
        )
        counts = {"total": 0}
        by_kind: Dict[str, Dict[str, float]] = {}
        distinct = set()
        for row in cur.fetchall():
            counts["total"] += 1
            kind = str(row["kind"])
            distinct.add(str(row["pattern_key"]))
            stats = by_kind.setdefault(
                kind,
                {
                    "count": 0,
                    "correct": 0,
                    "avg_reward_sum": 0.0,
                    "avg_ret_sum": 0.0,
                    "avg_conf_sum": 0.0,
                },
            )
            stats["count"] += 1
            if row["pred_dir"] == row["fact_dir"]:
                stats["correct"] += 1
            stats["avg_reward_sum"] += float(row["reward"] or 0.0)
            stats["avg_ret_sum"] += abs(float(row["ret_bps"] or 0.0))
            stats["avg_conf_sum"] += float(row["pred_conf"] or 0.0)
        out: Dict[str, object] = {"total_events": counts["total"], "distinct_patterns": len(distinct)}
        for kind, stats in by_kind.items():
            count = stats["count"]
            out[kind] = {
                "count": count,
                "accuracy": (stats["correct"] / count) if count else 0.0,
                "avg_reward": (stats["avg_reward_sum"] / count) if count else 0.0,
                "avg_abs_ret_bps": (stats["avg_ret_sum"] / count) if count else 0.0,
                "avg_conf": (stats["avg_conf_sum"] / count) if count else 0.0,
            }
        return out

    def dump_patterns(self, keys: Iterable[str]) -> Dict[str, PatternStats]:
        if not keys:
            return {}
        placeholders = ",".join("?" for _ in keys)
        cur = self.conn.execute(
            f"SELECT * FROM pattern_stats WHERE pattern_key IN ({placeholders})",
            tuple(keys),
        )
        out: Dict[str, PatternStats] = {}
        for row in cur.fetchall():
            stat = self._row_to_stat(row)
            out[stat.pattern_key] = stat
        return out
