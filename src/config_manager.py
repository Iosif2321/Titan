from __future__ import annotations

import asyncio
from typing import Any, Dict

from .config import DecisionConfig, FeatureConfig, ModelLRConfig, PatternConfig, TrainingConfig


class ConfigManager:
    def __init__(
        self,
        feature_config: FeatureConfig,
        decision_config: DecisionConfig,
        training_config: TrainingConfig,
        lr_config: ModelLRConfig,
        pattern_config: PatternConfig,
    ) -> None:
        self.feature_config = feature_config
        self.decision_config = decision_config
        self.training_config = training_config
        self.lr_config = lr_config
        self.pattern_config = pattern_config
        self.pending_restart: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def apply_updates(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with self._lock:
            applied: Dict[str, Any] = {}
            rejected: Dict[str, str] = {}
            pending: Dict[str, Any] = {}
            allowed_keys = {
                "flat_max_prob",
                "flat_max_delta",
                "flat_bps",
                "flat_update_weight",
                "reward_dir_correct",
                "reward_dir_wrong",
                "reward_flat_correct",
                "reward_flat_wrong",
                "flat_penalty",
                "class_balance_strength",
                "class_balance_ema",
                "class_balance_min",
                "class_balance_max",
                "class_balance_floor",
                "calib_lr",
                "calib_a_min",
                "calib_a_max",
                "calib_b_min",
                "calib_b_max",
                "calib_l2_a",
                "calib_l2_b",
                "calib_flat_bps",
                "calib_flat_weight",
                "perf_lr_gain",
                "perf_lr_min_mult",
                "perf_lr_max_mult",
                "perf_lr_baseline",
                "perf_lr_min_samples",
                "lr_trend",
                "lr_osc",
                "lr_vol",
                "lr_gain",
                "anchor_gain",
                "anchor_lambda_base",
                "pattern_ema_decay",
                "event_ttl_days",
                "max_events",
                "max_patterns",
                "support_k",
                "recency_tau_hours",
                "anti_min_support",
                "anti_win_threshold",
                "anti_trust_threshold",
                "maintenance_seconds",
                "lookback",
            }

            def reject(key: str, message: str) -> None:
                rejected[key] = message

            if "flat_max_prob" in payload:
                value = payload["flat_max_prob"]
                if isinstance(value, (int, float)) and 0 < value < 1:
                    self.decision_config.flat_max_prob = float(value)
                    applied["flat_max_prob"] = self.decision_config.flat_max_prob
                else:
                    reject("flat_max_prob", "must be in (0, 1)")

            if "flat_max_delta" in payload:
                value = payload["flat_max_delta"]
                if isinstance(value, (int, float)) and 0 <= value < 1:
                    self.decision_config.flat_max_delta = float(value)
                    applied["flat_max_delta"] = self.decision_config.flat_max_delta
                else:
                    reject("flat_max_delta", "must be in [0, 1)")

            if "flat_bps" in payload:
                value = payload["flat_bps"]
                if isinstance(value, (int, float)) and value >= 0:
                    self.training_config.flat_bps = float(value)
                    applied["flat_bps"] = self.training_config.flat_bps
                else:
                    reject("flat_bps", "must be >= 0")

            if "flat_update_weight" in payload:
                value = payload["flat_update_weight"]
                if isinstance(value, (int, float)) and 0 <= value <= 1:
                    self.training_config.flat_update_weight = float(value)
                    applied["flat_update_weight"] = self.training_config.flat_update_weight
                else:
                    reject("flat_update_weight", "must be in [0, 1]")

            if "reward_dir_correct" in payload:
                value = payload["reward_dir_correct"]
                if isinstance(value, (int, float)):
                    self.training_config.reward_dir_correct = float(value)
                    applied["reward_dir_correct"] = self.training_config.reward_dir_correct
                else:
                    reject("reward_dir_correct", "must be numeric")

            if "reward_dir_wrong" in payload:
                value = payload["reward_dir_wrong"]
                if isinstance(value, (int, float)):
                    self.training_config.reward_dir_wrong = float(value)
                    applied["reward_dir_wrong"] = self.training_config.reward_dir_wrong
                else:
                    reject("reward_dir_wrong", "must be numeric")

            if "reward_flat_correct" in payload:
                value = payload["reward_flat_correct"]
                if isinstance(value, (int, float)):
                    self.training_config.reward_flat_correct = float(value)
                    applied["reward_flat_correct"] = self.training_config.reward_flat_correct
                else:
                    reject("reward_flat_correct", "must be numeric")

            if "reward_flat_wrong" in payload:
                value = payload["reward_flat_wrong"]
                if isinstance(value, (int, float)):
                    self.training_config.reward_flat_wrong = float(value)
                    applied["reward_flat_wrong"] = self.training_config.reward_flat_wrong
                else:
                    reject("reward_flat_wrong", "must be numeric")

            if "flat_penalty" in payload:
                value = payload["flat_penalty"]
                if isinstance(value, (int, float)):
                    self.training_config.flat_penalty = float(value)
                    applied["flat_penalty"] = self.training_config.flat_penalty
                else:
                    reject("flat_penalty", "must be numeric")

            if "class_balance_strength" in payload:
                value = payload["class_balance_strength"]
                if isinstance(value, (int, float)) and value >= 0:
                    self.training_config.class_balance_strength = float(value)
                    applied["class_balance_strength"] = self.training_config.class_balance_strength
                else:
                    reject("class_balance_strength", "must be >= 0")

            if "class_balance_ema" in payload:
                value = payload["class_balance_ema"]
                if isinstance(value, (int, float)) and 0 < value < 1:
                    self.training_config.class_balance_ema = float(value)
                    applied["class_balance_ema"] = self.training_config.class_balance_ema
                else:
                    reject("class_balance_ema", "must be in (0, 1)")

            if "class_balance_min" in payload:
                value = payload["class_balance_min"]
                if isinstance(value, (int, float)) and value > 0:
                    self.training_config.class_balance_min = float(value)
                    applied["class_balance_min"] = self.training_config.class_balance_min
                else:
                    reject("class_balance_min", "must be > 0")

            if "class_balance_max" in payload:
                value = payload["class_balance_max"]
                if isinstance(value, (int, float)) and value > 0:
                    self.training_config.class_balance_max = float(value)
                    applied["class_balance_max"] = self.training_config.class_balance_max
                else:
                    reject("class_balance_max", "must be > 0")

            if "class_balance_floor" in payload:
                value = payload["class_balance_floor"]
                if isinstance(value, (int, float)) and value > 0:
                    self.training_config.class_balance_floor = float(value)
                    applied["class_balance_floor"] = self.training_config.class_balance_floor
                else:
                    reject("class_balance_floor", "must be > 0")

            if "calib_lr" in payload:
                value = payload["calib_lr"]
                if isinstance(value, (int, float)) and value >= 0:
                    self.training_config.calib_lr = float(value)
                    applied["calib_lr"] = self.training_config.calib_lr
                else:
                    reject("calib_lr", "must be >= 0")

            if "calib_a_min" in payload:
                value = payload["calib_a_min"]
                if isinstance(value, (int, float)) and value > 0:
                    self.training_config.calib_a_min = float(value)
                    applied["calib_a_min"] = self.training_config.calib_a_min
                else:
                    reject("calib_a_min", "must be > 0")

            if "calib_a_max" in payload:
                value = payload["calib_a_max"]
                if isinstance(value, (int, float)) and value > 0:
                    self.training_config.calib_a_max = float(value)
                    applied["calib_a_max"] = self.training_config.calib_a_max
                else:
                    reject("calib_a_max", "must be > 0")

            if "calib_b_min" in payload:
                value = payload["calib_b_min"]
                if isinstance(value, (int, float)):
                    self.training_config.calib_b_min = float(value)
                    applied["calib_b_min"] = self.training_config.calib_b_min
                else:
                    reject("calib_b_min", "must be numeric")

            if "calib_b_max" in payload:
                value = payload["calib_b_max"]
                if isinstance(value, (int, float)):
                    self.training_config.calib_b_max = float(value)
                    applied["calib_b_max"] = self.training_config.calib_b_max
                else:
                    reject("calib_b_max", "must be numeric")

            if "calib_l2_a" in payload:
                value = payload["calib_l2_a"]
                if isinstance(value, (int, float)) and value >= 0:
                    self.training_config.calib_l2_a = float(value)
                    applied["calib_l2_a"] = self.training_config.calib_l2_a
                else:
                    reject("calib_l2_a", "must be >= 0")

            if "calib_l2_b" in payload:
                value = payload["calib_l2_b"]
                if isinstance(value, (int, float)) and value >= 0:
                    self.training_config.calib_l2_b = float(value)
                    applied["calib_l2_b"] = self.training_config.calib_l2_b
                else:
                    reject("calib_l2_b", "must be >= 0")

            if "calib_flat_bps" in payload:
                value = payload["calib_flat_bps"]
                if isinstance(value, (int, float)) and value >= 0:
                    self.training_config.calib_flat_bps = float(value)
                    applied["calib_flat_bps"] = self.training_config.calib_flat_bps
                else:
                    reject("calib_flat_bps", "must be >= 0")

            if "calib_flat_weight" in payload:
                value = payload["calib_flat_weight"]
                if isinstance(value, (int, float)) and 0 <= value <= 1:
                    self.training_config.calib_flat_weight = float(value)
                    applied["calib_flat_weight"] = self.training_config.calib_flat_weight
                else:
                    reject("calib_flat_weight", "must be in [0, 1]")

            if "perf_lr_gain" in payload:
                value = payload["perf_lr_gain"]
                if isinstance(value, (int, float)) and value >= 0:
                    self.training_config.perf_lr_gain = float(value)
                    applied["perf_lr_gain"] = self.training_config.perf_lr_gain
                else:
                    reject("perf_lr_gain", "must be >= 0")

            if "perf_lr_min_mult" in payload:
                value = payload["perf_lr_min_mult"]
                if isinstance(value, (int, float)) and value > 0:
                    self.training_config.perf_lr_min_mult = float(value)
                    applied["perf_lr_min_mult"] = self.training_config.perf_lr_min_mult
                else:
                    reject("perf_lr_min_mult", "must be > 0")

            if "perf_lr_max_mult" in payload:
                value = payload["perf_lr_max_mult"]
                if isinstance(value, (int, float)) and value > 0:
                    self.training_config.perf_lr_max_mult = float(value)
                    applied["perf_lr_max_mult"] = self.training_config.perf_lr_max_mult
                else:
                    reject("perf_lr_max_mult", "must be > 0")

            if "perf_lr_baseline" in payload:
                value = payload["perf_lr_baseline"]
                if isinstance(value, (int, float)) and 0 < value < 1:
                    self.training_config.perf_lr_baseline = float(value)
                    applied["perf_lr_baseline"] = self.training_config.perf_lr_baseline
                else:
                    reject("perf_lr_baseline", "must be in (0, 1)")

            if "perf_lr_min_samples" in payload:
                value = payload["perf_lr_min_samples"]
                if isinstance(value, int) and value >= 0:
                    self.training_config.perf_lr_min_samples = int(value)
                    applied["perf_lr_min_samples"] = self.training_config.perf_lr_min_samples
                else:
                    reject("perf_lr_min_samples", "must be >= 0")

            if "lr_trend" in payload:
                value = payload["lr_trend"]
                if isinstance(value, (int, float)) and value > 0:
                    self.lr_config.lr_trend = float(value)
                    applied["lr_trend"] = self.lr_config.lr_trend
                else:
                    reject("lr_trend", "must be > 0")

            if "lr_osc" in payload:
                value = payload["lr_osc"]
                if isinstance(value, (int, float)) and value > 0:
                    self.lr_config.lr_osc = float(value)
                    applied["lr_osc"] = self.lr_config.lr_osc
                else:
                    reject("lr_osc", "must be > 0")

            if "lr_vol" in payload:
                value = payload["lr_vol"]
                if isinstance(value, (int, float)) and value > 0:
                    self.lr_config.lr_vol = float(value)
                    applied["lr_vol"] = self.lr_config.lr_vol
                else:
                    reject("lr_vol", "must be > 0")

            if "lr_gain" in payload:
                value = payload["lr_gain"]
                if isinstance(value, (int, float)) and value >= 0:
                    self.training_config.lr_gain = float(value)
                    applied["lr_gain"] = self.training_config.lr_gain
                else:
                    reject("lr_gain", "must be >= 0")

            if "anchor_gain" in payload:
                value = payload["anchor_gain"]
                if isinstance(value, (int, float)) and value >= 0:
                    self.training_config.anchor_gain = float(value)
                    applied["anchor_gain"] = self.training_config.anchor_gain
                else:
                    reject("anchor_gain", "must be >= 0")

            if "anchor_lambda_base" in payload:
                value = payload["anchor_lambda_base"]
                if isinstance(value, (int, float)) and value >= 0:
                    self.training_config.anchor_lambda_base = float(value)
                    applied["anchor_lambda_base"] = self.training_config.anchor_lambda_base
                else:
                    reject("anchor_lambda_base", "must be >= 0")

            if "pattern_ema_decay" in payload:
                value = payload["pattern_ema_decay"]
                if isinstance(value, (int, float)) and 0 < value < 1:
                    self.pattern_config.ema_decay = float(value)
                    applied["pattern_ema_decay"] = self.pattern_config.ema_decay
                else:
                    reject("pattern_ema_decay", "must be in (0, 1)")

            if "event_ttl_days" in payload:
                value = payload["event_ttl_days"]
                if isinstance(value, int) and value >= 0:
                    self.pattern_config.event_ttl_days = int(value)
                    applied["event_ttl_days"] = self.pattern_config.event_ttl_days
                else:
                    reject("event_ttl_days", "must be >= 0")

            if "max_events" in payload:
                value = payload["max_events"]
                if isinstance(value, int) and value >= 0:
                    self.pattern_config.max_events = int(value)
                    applied["max_events"] = self.pattern_config.max_events
                else:
                    reject("max_events", "must be >= 0")

            if "max_patterns" in payload:
                value = payload["max_patterns"]
                if isinstance(value, int) and value >= 0:
                    self.pattern_config.max_patterns = int(value)
                    applied["max_patterns"] = self.pattern_config.max_patterns
                else:
                    reject("max_patterns", "must be >= 0")

            if "support_k" in payload:
                value = payload["support_k"]
                if isinstance(value, int) and value > 0:
                    self.pattern_config.support_k = int(value)
                    applied["support_k"] = self.pattern_config.support_k
                else:
                    reject("support_k", "must be > 0")

            if "recency_tau_hours" in payload:
                value = payload["recency_tau_hours"]
                if isinstance(value, (int, float)) and value > 0:
                    self.pattern_config.recency_tau_hours = float(value)
                    applied["recency_tau_hours"] = self.pattern_config.recency_tau_hours
                else:
                    reject("recency_tau_hours", "must be > 0")

            if "anti_min_support" in payload:
                value = payload["anti_min_support"]
                if isinstance(value, int) and value >= 1:
                    self.pattern_config.anti_min_support = int(value)
                    applied["anti_min_support"] = self.pattern_config.anti_min_support
                else:
                    reject("anti_min_support", "must be >= 1")

            if "anti_win_threshold" in payload:
                value = payload["anti_win_threshold"]
                if isinstance(value, (int, float)) and 0 <= value <= 1:
                    self.pattern_config.anti_win_threshold = float(value)
                    applied["anti_win_threshold"] = self.pattern_config.anti_win_threshold
                else:
                    reject("anti_win_threshold", "must be in [0, 1]")

            if "anti_trust_threshold" in payload:
                value = payload["anti_trust_threshold"]
                if isinstance(value, (int, float)) and 0 <= value <= 1:
                    self.pattern_config.anti_trust_threshold = float(value)
                    applied["anti_trust_threshold"] = self.pattern_config.anti_trust_threshold
                else:
                    reject("anti_trust_threshold", "must be in [0, 1]")

            if "maintenance_seconds" in payload:
                value = payload["maintenance_seconds"]
                if isinstance(value, int) and value >= 0:
                    self.pattern_config.maintenance_seconds = int(value)
                    applied["maintenance_seconds"] = self.pattern_config.maintenance_seconds
                else:
                    reject("maintenance_seconds", "must be >= 0")

            if "lookback" in payload:
                value = payload["lookback"]
                if isinstance(value, int) and value > 1:
                    pending["lookback"] = int(value)
                    self.pending_restart["lookback"] = int(value)
                else:
                    reject("lookback", "must be integer > 1 (requires restart)")

            for key in payload:
                if key not in allowed_keys and key not in rejected:
                    rejected[key] = "unsupported"

            return {
                "applied": applied,
                "pending_restart": pending,
                "rejected": rejected,
                "current": self.snapshot(),
            }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "flat_max_prob": self.decision_config.flat_max_prob,
            "flat_max_delta": self.decision_config.flat_max_delta,
            "flat_bps": self.training_config.flat_bps,
            "calib_lr": self.training_config.calib_lr,
            "calib_a_min": self.training_config.calib_a_min,
            "calib_a_max": self.training_config.calib_a_max,
            "calib_b_min": self.training_config.calib_b_min,
            "calib_b_max": self.training_config.calib_b_max,
            "calib_l2_a": self.training_config.calib_l2_a,
            "calib_l2_b": self.training_config.calib_l2_b,
            "calib_flat_bps": self.training_config.calib_flat_bps,
            "calib_flat_weight": self.training_config.calib_flat_weight,
            "lr_trend": self.lr_config.lr_trend,
            "lr_osc": self.lr_config.lr_osc,
            "lr_vol": self.lr_config.lr_vol,
            "lr_gain": self.training_config.lr_gain,
            "anchor_gain": self.training_config.anchor_gain,
            "anchor_lambda_base": self.training_config.anchor_lambda_base,
            "pattern_ema_decay": self.pattern_config.ema_decay,
            "event_ttl_days": self.pattern_config.event_ttl_days,
            "max_events": self.pattern_config.max_events,
            "max_patterns": self.pattern_config.max_patterns,
            "support_k": self.pattern_config.support_k,
            "recency_tau_hours": self.pattern_config.recency_tau_hours,
            "anti_min_support": self.pattern_config.anti_min_support,
            "anti_win_threshold": self.pattern_config.anti_win_threshold,
            "anti_trust_threshold": self.pattern_config.anti_trust_threshold,
            "maintenance_seconds": self.pattern_config.maintenance_seconds,
            "lookback": self.feature_config.lookback,
            "pending_restart": self.pending_restart,
        }
