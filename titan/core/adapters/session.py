"""Session-based adaptation module.

Sprint 17: Per-session (ASIA/EUROPE/US) model configuration using
Thompson Sampling (Contextual Bandits) for parameter selection.

Key components:
- SessionMemory: SQLite storage for per-session statistics
- SessionAdapter: Adapts weights, parameters, and calibration per session
- Thompson Sampling: Bayesian exploration for discrete parameters
- Decay mechanism: 168h half-life for forgetting old data
- Trust blocks: Minimum samples before trusting session stats
"""

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from titan.core.config import ConfigStore


# Trading sessions (UTC hours)
SESSIONS = {
    "ASIA": (0, 8),      # 00:00 - 08:00 UTC
    "EUROPE": (8, 16),   # 08:00 - 16:00 UTC
    "US": (16, 24),      # 16:00 - 24:00 UTC
}

# Parameter options for Thompson Sampling
PARAM_OPTIONS: Dict[str, List[float]] = {
    "model.rsi_oversold": [25.0, 28.0, 30.0, 32.0, 35.0],
    "model.rsi_overbought": [65.0, 68.0, 70.0, 72.0, 75.0],
    "pattern.vol_z_high": [0.8, 1.0, 1.2, 1.5],
    "pattern.vol_z_low": [-1.5, -1.2, -1.0, -0.8],
    "ensemble.flat_threshold": [0.52, 0.55, 0.58],
}

# Confidence bins for calibration
CONF_BINS = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70)]

# Model names
MODEL_NAMES = ["TRENDVIC", "OSCILLATOR", "VOLUMEMETRIX", "ML_CLASSIFIER"]


@dataclass
class SessionStats:
    """Statistics for a model in a session."""
    total: int = 0
    correct: int = 0
    conf_sum: float = 0.0
    return_sum: float = 0.0
    last_update: int = 0
    decay_total: float = 0.0
    decay_correct: float = 0.0

    @property
    def accuracy(self) -> float:
        if self.decay_total < 1.0:
            return 0.5
        return self.decay_correct / self.decay_total

    @property
    def raw_accuracy(self) -> float:
        if self.total == 0:
            return 0.5
        return self.correct / self.total


@dataclass
class ParamStats:
    """Statistics for a parameter value."""
    total: int = 0
    correct: int = 0
    return_sum: float = 0.0
    last_tested: int = 0

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.5
        return self.correct / self.total


@dataclass
class CalibrationBin:
    """Statistics for a confidence bin."""
    total: int = 0
    correct: int = 0
    conf_sum: float = 0.0

    @property
    def empirical_accuracy(self) -> float:
        if self.total == 0:
            return 0.5
        return self.correct / self.total

    @property
    def avg_confidence(self) -> float:
        if self.total == 0:
            return 0.525  # Middle of first bin
        return self.conf_sum / self.total


class SessionMemory:
    """SQLite-backed storage for session statistics."""

    SCHEMA = """
    -- Per-session model statistics
    CREATE TABLE IF NOT EXISTS session_stats (
        session TEXT NOT NULL,
        model TEXT NOT NULL,
        regime TEXT,
        total INTEGER DEFAULT 0,
        correct INTEGER DEFAULT 0,
        conf_sum REAL DEFAULT 0,
        return_sum REAL DEFAULT 0,
        last_update INTEGER DEFAULT 0,
        decay_total REAL DEFAULT 0,
        decay_correct REAL DEFAULT 0,
        PRIMARY KEY (session, model, regime)
    );

    -- Thompson Sampling for discrete parameters
    CREATE TABLE IF NOT EXISTS session_params (
        session TEXT NOT NULL,
        param_key TEXT NOT NULL,
        param_value REAL NOT NULL,
        total INTEGER DEFAULT 0,
        correct INTEGER DEFAULT 0,
        return_sum REAL DEFAULT 0,
        last_tested INTEGER DEFAULT 0,
        PRIMARY KEY (session, param_key, param_value)
    );

    -- Per-session confidence calibration
    CREATE TABLE IF NOT EXISTS session_calibration (
        session TEXT NOT NULL,
        bin_idx INTEGER NOT NULL,
        total INTEGER DEFAULT 0,
        correct INTEGER DEFAULT 0,
        conf_sum REAL DEFAULT 0,
        PRIMARY KEY (session, bin_idx)
    );

    -- Global statistics (for shrinkage)
    CREATE TABLE IF NOT EXISTS global_stats (
        model TEXT NOT NULL,
        regime TEXT,
        total INTEGER DEFAULT 0,
        correct INTEGER DEFAULT 0,
        conf_sum REAL DEFAULT 0,
        return_sum REAL DEFAULT 0,
        PRIMARY KEY (model, regime)
    );

    -- Global calibration
    CREATE TABLE IF NOT EXISTS global_calibration (
        bin_idx INTEGER PRIMARY KEY,
        total INTEGER DEFAULT 0,
        correct INTEGER DEFAULT 0,
        conf_sum REAL DEFAULT 0
    );

    -- Prediction counter per session
    CREATE TABLE IF NOT EXISTS session_counters (
        session TEXT PRIMARY KEY,
        prediction_count INTEGER DEFAULT 0
    );
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database with schema."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # Session stats methods
    def get_session_stats(
        self, session: str, model: str, regime: Optional[str] = None
    ) -> SessionStats:
        """Get statistics for a model in a session."""
        cursor = self._conn.execute(
            """
            SELECT total, correct, conf_sum, return_sum, last_update,
                   decay_total, decay_correct
            FROM session_stats
            WHERE session = ? AND model = ? AND (regime = ? OR (regime IS NULL AND ? IS NULL))
            """,
            (session, model, regime, regime),
        )
        row = cursor.fetchone()
        if row:
            return SessionStats(
                total=row["total"],
                correct=row["correct"],
                conf_sum=row["conf_sum"],
                return_sum=row["return_sum"],
                last_update=row["last_update"],
                decay_total=row["decay_total"],
                decay_correct=row["decay_correct"],
            )
        return SessionStats()

    def get_session_stats_all_regimes(
        self, session: str, model: str
    ) -> SessionStats:
        """Get aggregated statistics for a model across all regimes."""
        cursor = self._conn.execute(
            """
            SELECT SUM(total) as total, SUM(correct) as correct,
                   SUM(conf_sum) as conf_sum, SUM(return_sum) as return_sum,
                   MAX(last_update) as last_update,
                   SUM(decay_total) as decay_total, SUM(decay_correct) as decay_correct
            FROM session_stats
            WHERE session = ? AND model = ?
            """,
            (session, model),
        )
        row = cursor.fetchone()
        if row and row["total"]:
            return SessionStats(
                total=row["total"],
                correct=row["correct"],
                conf_sum=row["conf_sum"],
                return_sum=row["return_sum"],
                last_update=row["last_update"] or 0,
                decay_total=row["decay_total"] or 0.0,
                decay_correct=row["decay_correct"] or 0.0,
            )
        return SessionStats()

    def update_session_stats(
        self,
        session: str,
        model: str,
        regime: Optional[str],
        hit: bool,
        conf: float,
        return_pct: float,
        ts: int,
        half_life_hours: float = 168.0,
    ) -> None:
        """Update session statistics with decay."""
        stats = self.get_session_stats(session, model, regime)

        # Apply decay
        if stats.last_update > 0 and stats.decay_total > 0:
            hours_since = (ts - stats.last_update) / 3600.0
            decay = 0.5 ** (hours_since / half_life_hours)
            stats.decay_total *= decay
            stats.decay_correct *= decay
        else:
            stats.decay_total = 0.0
            stats.decay_correct = 0.0

        # Update stats
        stats.total += 1
        stats.correct += 1 if hit else 0
        stats.conf_sum += conf
        stats.return_sum += return_pct
        stats.decay_total += 1.0
        stats.decay_correct += 1.0 if hit else 0.0
        stats.last_update = ts

        self._conn.execute(
            """
            INSERT INTO session_stats
                (session, model, regime, total, correct, conf_sum, return_sum,
                 last_update, decay_total, decay_correct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session, model, regime) DO UPDATE SET
                total = excluded.total,
                correct = excluded.correct,
                conf_sum = excluded.conf_sum,
                return_sum = excluded.return_sum,
                last_update = excluded.last_update,
                decay_total = excluded.decay_total,
                decay_correct = excluded.decay_correct
            """,
            (
                session, model, regime, stats.total, stats.correct,
                stats.conf_sum, stats.return_sum, stats.last_update,
                stats.decay_total, stats.decay_correct,
            ),
        )
        self._conn.commit()

    # Global stats methods
    def get_global_stats(self, model: str, regime: Optional[str] = None) -> SessionStats:
        """Get global statistics for a model."""
        cursor = self._conn.execute(
            """
            SELECT total, correct, conf_sum, return_sum
            FROM global_stats
            WHERE model = ? AND (regime = ? OR (regime IS NULL AND ? IS NULL))
            """,
            (model, regime, regime),
        )
        row = cursor.fetchone()
        if row:
            stats = SessionStats(
                total=row["total"],
                correct=row["correct"],
                conf_sum=row["conf_sum"],
                return_sum=row["return_sum"],
            )
            # For global stats, use raw accuracy (no decay)
            stats.decay_total = float(stats.total)
            stats.decay_correct = float(stats.correct)
            return stats
        return SessionStats()

    def get_global_stats_all_regimes(self, model: str) -> SessionStats:
        """Get aggregated global statistics for a model across all regimes."""
        cursor = self._conn.execute(
            """
            SELECT SUM(total) as total, SUM(correct) as correct,
                   SUM(conf_sum) as conf_sum, SUM(return_sum) as return_sum
            FROM global_stats
            WHERE model = ?
            """,
            (model,),
        )
        row = cursor.fetchone()
        if row and row["total"]:
            stats = SessionStats(
                total=row["total"],
                correct=row["correct"],
                conf_sum=row["conf_sum"],
                return_sum=row["return_sum"],
            )
            stats.decay_total = float(stats.total)
            stats.decay_correct = float(stats.correct)
            return stats
        return SessionStats()

    def update_global_stats(
        self,
        model: str,
        regime: Optional[str],
        hit: bool,
        conf: float,
        return_pct: float,
    ) -> None:
        """Update global statistics."""
        self._conn.execute(
            """
            INSERT INTO global_stats (model, regime, total, correct, conf_sum, return_sum)
            VALUES (?, ?, 1, ?, ?, ?)
            ON CONFLICT(model, regime) DO UPDATE SET
                total = total + 1,
                correct = correct + ?,
                conf_sum = conf_sum + ?,
                return_sum = return_sum + ?
            """,
            (model, regime, 1 if hit else 0, conf, return_pct,
             1 if hit else 0, conf, return_pct),
        )
        self._conn.commit()

    # Parameter stats methods
    def get_param_stats(
        self, session: str, param_key: str, param_value: float
    ) -> ParamStats:
        """Get statistics for a parameter value."""
        cursor = self._conn.execute(
            """
            SELECT total, correct, return_sum, last_tested
            FROM session_params
            WHERE session = ? AND param_key = ? AND param_value = ?
            """,
            (session, param_key, param_value),
        )
        row = cursor.fetchone()
        if row:
            return ParamStats(
                total=row["total"],
                correct=row["correct"],
                return_sum=row["return_sum"],
                last_tested=row["last_tested"],
            )
        return ParamStats()

    def update_param_stats(
        self,
        session: str,
        param_key: str,
        param_value: float,
        hit: bool,
        return_pct: float,
        ts: int,
    ) -> None:
        """Update parameter statistics."""
        self._conn.execute(
            """
            INSERT INTO session_params
                (session, param_key, param_value, total, correct, return_sum, last_tested)
            VALUES (?, ?, ?, 1, ?, ?, ?)
            ON CONFLICT(session, param_key, param_value) DO UPDATE SET
                total = total + 1,
                correct = correct + ?,
                return_sum = return_sum + ?,
                last_tested = ?
            """,
            (session, param_key, param_value, 1 if hit else 0, return_pct, ts,
             1 if hit else 0, return_pct, ts),
        )
        self._conn.commit()

    # Calibration methods
    def get_calibration_bin(self, session: str, bin_idx: int) -> CalibrationBin:
        """Get calibration statistics for a confidence bin."""
        cursor = self._conn.execute(
            """
            SELECT total, correct, conf_sum
            FROM session_calibration
            WHERE session = ? AND bin_idx = ?
            """,
            (session, bin_idx),
        )
        row = cursor.fetchone()
        if row:
            return CalibrationBin(
                total=row["total"],
                correct=row["correct"],
                conf_sum=row["conf_sum"],
            )
        return CalibrationBin()

    def get_global_calibration_bin(self, bin_idx: int) -> CalibrationBin:
        """Get global calibration statistics for a confidence bin."""
        cursor = self._conn.execute(
            """
            SELECT total, correct, conf_sum
            FROM global_calibration
            WHERE bin_idx = ?
            """,
            (bin_idx,),
        )
        row = cursor.fetchone()
        if row:
            return CalibrationBin(
                total=row["total"],
                correct=row["correct"],
                conf_sum=row["conf_sum"],
            )
        return CalibrationBin()

    def update_calibration(
        self, session: str, bin_idx: int, hit: bool, conf: float
    ) -> None:
        """Update calibration statistics."""
        # Update session calibration
        self._conn.execute(
            """
            INSERT INTO session_calibration (session, bin_idx, total, correct, conf_sum)
            VALUES (?, ?, 1, ?, ?)
            ON CONFLICT(session, bin_idx) DO UPDATE SET
                total = total + 1,
                correct = correct + ?,
                conf_sum = conf_sum + ?
            """,
            (session, bin_idx, 1 if hit else 0, conf, 1 if hit else 0, conf),
        )
        # Update global calibration
        self._conn.execute(
            """
            INSERT INTO global_calibration (bin_idx, total, correct, conf_sum)
            VALUES (?, 1, ?, ?)
            ON CONFLICT(bin_idx) DO UPDATE SET
                total = total + 1,
                correct = correct + ?,
                conf_sum = conf_sum + ?
            """,
            (bin_idx, 1 if hit else 0, conf, 1 if hit else 0, conf),
        )
        self._conn.commit()

    # Counter methods
    def get_prediction_count(self, session: str) -> int:
        """Get total prediction count for a session."""
        cursor = self._conn.execute(
            "SELECT prediction_count FROM session_counters WHERE session = ?",
            (session,),
        )
        row = cursor.fetchone()
        return row["prediction_count"] if row else 0

    def increment_prediction_count(self, session: str) -> int:
        """Increment prediction counter and return new count."""
        self._conn.execute(
            """
            INSERT INTO session_counters (session, prediction_count)
            VALUES (?, 1)
            ON CONFLICT(session) DO UPDATE SET
                prediction_count = prediction_count + 1
            """,
            (session,),
        )
        self._conn.commit()
        return self.get_prediction_count(session)


class SessionAdapter:
    """Per-session model configuration adapter.

    Uses Thompson Sampling (Contextual Bandits) for parameter selection
    and EMA with shrinkage for weight adaptation.
    """

    # Update frequencies (in predictions)
    WEIGHT_UPDATE_FREQ = 50
    PARAM_UPDATE_FREQ = 500
    CALIBRATION_UPDATE_FREQ = 100

    # Trust thresholds
    MIN_SAMPLES = 50
    MAX_CI_WIDTH = 0.10

    # Decay
    HALF_LIFE_HOURS = 168.0  # 1 week

    # Shrinkage prior strength
    PRIOR_STRENGTH = 1000

    # Weight bounds
    MIN_WEIGHT = 0.10
    MAX_WEIGHT = 0.50

    def __init__(
        self,
        db_path: str,
        config: ConfigStore,
        enabled: bool = True,
    ) -> None:
        """Initialize session adapter.

        Args:
            db_path: Path to SQLite database for session memory
            config: Global configuration store
            enabled: Whether session adaptation is enabled
        """
        self._memory = SessionMemory(db_path)
        self._config = config
        self._enabled = enabled

        # Cache for current session config
        self._session_weights: Dict[str, Dict[str, float]] = {}
        self._session_params: Dict[str, Dict[str, float]] = {}
        self._active_params: Dict[str, Dict[str, float]] = {}  # Currently selected params

    def close(self) -> None:
        """Close database connection."""
        self._memory.close()

    def is_enabled(self) -> bool:
        """Check if session adaptation is enabled."""
        return self._enabled

    def get_session(self, ts: int) -> str:
        """Determine trading session from timestamp.

        Args:
            ts: Unix timestamp

        Returns:
            Session name: 'ASIA', 'EUROPE', or 'US'
        """
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour

        for session, (start, end) in SESSIONS.items():
            if start <= hour < end:
                return session

        return "US"  # Default fallback

    def get_weights(
        self,
        session: str,
        regime: Optional[str] = None,
    ) -> Dict[str, float]:
        """Get model weights for session with shrinkage to global.

        Args:
            session: Trading session ('ASIA', 'EUROPE', 'US')
            regime: Market regime (optional)

        Returns:
            Dict of model name -> weight
        """
        if not self._enabled:
            return self._get_default_weights()

        weights = {}
        for model in MODEL_NAMES:
            session_stats = self._memory.get_session_stats(session, model, regime)
            global_stats = self._memory.get_global_stats(model, regime)

            # Shrinkage formula
            k = self.PRIOR_STRENGTH
            n = session_stats.total

            global_acc = global_stats.accuracy if global_stats.total > 0 else 0.5
            session_acc = session_stats.accuracy if n > 0 else global_acc

            # Effective accuracy with shrinkage
            if k + n > 0:
                effective_acc = (global_acc * k + session_acc * n) / (k + n)
            else:
                effective_acc = 0.5

            weights[model] = effective_acc

        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        # Apply bounds
        weights = {
            k: max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, v))
            for k, v in weights.items()
        }

        # Renormalize after bounds
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def select_param(self, session: str, param_key: str) -> float:
        """Select parameter value using Thompson Sampling.

        Args:
            session: Trading session
            param_key: Parameter key (e.g., 'model.rsi_oversold')

        Returns:
            Selected parameter value
        """
        if param_key not in PARAM_OPTIONS:
            return float(self._config.get(param_key, 0.0))

        if not self._enabled:
            return float(self._config.get(param_key, PARAM_OPTIONS[param_key][len(PARAM_OPTIONS[param_key]) // 2]))

        options = PARAM_OPTIONS[param_key]
        posteriors = []

        for opt in options:
            stats = self._memory.get_param_stats(session, param_key, opt)

            # Beta posterior: α = successes + 1, β = failures + 1
            alpha = stats.correct + 1
            beta = stats.total - stats.correct + 1

            # Sample from posterior
            sample = np.random.beta(alpha, beta)
            posteriors.append(sample)

        # Select best option
        best_idx = int(np.argmax(posteriors))
        selected = options[best_idx]

        # Cache the selection
        if session not in self._active_params:
            self._active_params[session] = {}
        self._active_params[session][param_key] = selected

        return selected

    def get_all_params(self, session: str) -> Dict[str, float]:
        """Get all session-specific parameters.

        Args:
            session: Trading session

        Returns:
            Dict of param_key -> selected value
        """
        params = {}
        for param_key in PARAM_OPTIONS.keys():
            params[param_key] = self.select_param(session, param_key)
        return params

    def calibrate_confidence(self, session: str, raw_conf: float) -> float:
        """Apply per-session temperature scaling to confidence.

        Args:
            session: Trading session
            raw_conf: Raw confidence value

        Returns:
            Calibrated confidence
        """
        if not self._enabled:
            return raw_conf

        bin_idx = self._get_conf_bin(raw_conf)
        bin_stats = self._memory.get_calibration_bin(session, bin_idx)

        # Blend with global if insufficient data
        if bin_stats.total < 30:
            global_stats = self._memory.get_global_calibration_bin(bin_idx)
            bin_stats = self._blend_calibration(bin_stats, global_stats)

        if bin_stats.total == 0:
            return raw_conf

        empirical_acc = bin_stats.empirical_accuracy

        # Temperature scaling
        # If raw_conf = 0.55 and empirical = 0.65, we should boost
        # If raw_conf = 0.60 and empirical = 0.50, we should reduce
        if abs(raw_conf - 0.5) < 1e-6:
            return raw_conf

        calibrated = 0.5 + (raw_conf - 0.5) * (empirical_acc - 0.5) / (raw_conf - 0.5)
        return max(0.50, min(0.70, calibrated))

    def record_outcome(
        self,
        session: str,
        model: str,
        regime: Optional[str],
        hit: bool,
        conf: float,
        return_pct: float,
        ts: int,
    ) -> None:
        """Record prediction outcome for learning.

        Args:
            session: Trading session
            model: Model name
            regime: Market regime
            hit: Whether prediction was correct
            conf: Confidence value
            return_pct: Return percentage
            ts: Unix timestamp
        """
        if not self._enabled:
            return

        # Update session stats (with decay)
        self._memory.update_session_stats(
            session, model, regime, hit, conf, return_pct, ts,
            half_life_hours=self.HALF_LIFE_HOURS,
        )

        # Update global stats
        self._memory.update_global_stats(model, regime, hit, conf, return_pct)

        # Update calibration
        bin_idx = self._get_conf_bin(conf)
        self._memory.update_calibration(session, bin_idx, hit, conf)

        # Update parameter stats for active params
        if session in self._active_params:
            for param_key, param_value in self._active_params[session].items():
                self._memory.update_param_stats(
                    session, param_key, param_value, hit, return_pct, ts
                )

        # Increment counter
        count = self._memory.increment_prediction_count(session)

        # Check if updates needed (logged for debugging)
        if count % self.WEIGHT_UPDATE_FREQ == 0:
            self._log_update("weights", session, count)
        if count % self.PARAM_UPDATE_FREQ == 0:
            self._log_update("params", session, count)
        if count % self.CALIBRATION_UPDATE_FREQ == 0:
            self._log_update("calibration", session, count)

    def can_trust_session(self, session: str, model: str) -> bool:
        """Check if we have enough data to trust session stats.

        Args:
            session: Trading session
            model: Model name

        Returns:
            True if session stats are trustworthy
        """
        stats = self._memory.get_session_stats_all_regimes(session, model)

        if stats.total < self.MIN_SAMPLES:
            return False

        # Wilson score confidence interval
        if stats.total == 0:
            return False

        acc = stats.raw_accuracy
        n = stats.total
        z = 1.96  # 95% CI

        # Wilson score interval width
        denominator = 1 + z * z / n
        center = (acc + z * z / (2 * n)) / denominator
        margin = z * math.sqrt((acc * (1 - acc) + z * z / (4 * n)) / n) / denominator
        ci_width = 2 * margin

        return ci_width <= self.MAX_CI_WIDTH

    def get_session_summary(self, session: str) -> Dict[str, Any]:
        """Get summary of session performance.

        Args:
            session: Trading session

        Returns:
            Dict with session statistics
        """
        summary = {
            "session": session,
            "prediction_count": self._memory.get_prediction_count(session),
            "models": {},
            "params": {},
            "calibration": {},
        }

        # Model stats (aggregated across all regimes)
        for model in MODEL_NAMES:
            stats = self._memory.get_session_stats_all_regimes(session, model)
            global_stats = self._memory.get_global_stats_all_regimes(model)
            summary["models"][model] = {
                "total": stats.total,
                "accuracy": stats.accuracy,
                "raw_accuracy": stats.raw_accuracy,
                "global_accuracy": global_stats.accuracy,
                "trusted": self.can_trust_session(session, model),
            }

        # Param stats
        if session in self._active_params:
            for param_key, param_value in self._active_params[session].items():
                stats = self._memory.get_param_stats(session, param_key, param_value)
                summary["params"][param_key] = {
                    "value": param_value,
                    "total": stats.total,
                    "accuracy": stats.accuracy,
                }

        # Calibration stats
        for i, (low, high) in enumerate(CONF_BINS):
            bin_stats = self._memory.get_calibration_bin(session, i)
            summary["calibration"][f"{low:.0%}-{high:.0%}"] = {
                "total": bin_stats.total,
                "empirical_accuracy": bin_stats.empirical_accuracy,
                "avg_confidence": bin_stats.avg_confidence,
            }

        return summary

    def _get_conf_bin(self, conf: float) -> int:
        """Get confidence bin index."""
        for i, (low, high) in enumerate(CONF_BINS):
            if low <= conf < high:
                return i
        return len(CONF_BINS) - 1  # Last bin for high confidence

    def _blend_calibration(
        self, session_bin: CalibrationBin, global_bin: CalibrationBin
    ) -> CalibrationBin:
        """Blend session and global calibration stats."""
        total = session_bin.total + global_bin.total
        correct = session_bin.correct + global_bin.correct
        conf_sum = session_bin.conf_sum + global_bin.conf_sum
        return CalibrationBin(total=total, correct=correct, conf_sum=conf_sum)

    def _get_default_weights(self) -> Dict[str, float]:
        """Get default model weights."""
        return {
            "TRENDVIC": 0.30,
            "OSCILLATOR": 0.25,
            "VOLUMEMETRIX": 0.25,
            "ML_CLASSIFIER": 0.20,
        }

    def _log_update(self, update_type: str, session: str, count: int) -> None:
        """Log update trigger (can be extended for actual logging)."""
        # Placeholder for logging
        pass
