from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional

import numpy as np

from .config import DecisionConfig, FactConfig
from .types import Direction
from .utils import clamp


class AdaptiveFactFlatController:
    def __init__(self, config: FactConfig) -> None:
        self.window = int(config.fact_flat_window)
        self.update_every = int(config.fact_flat_update_every)
        self.min_samples = int(config.fact_flat_min_samples)
        self.target_lo = float(config.fact_flat_target_lo)
        self.target_hi = float(config.fact_flat_target_hi)
        self.p_step = float(config.fact_flat_p_step)
        self.p_min = float(config.fact_flat_p_min)
        self.p_max = float(config.fact_flat_p_max)
        self.bps_min = float(config.fact_flat_bps_min)
        self.bps_max = float(config.fact_flat_bps_max)
        self.beta = float(config.fact_flat_smooth_beta)
        self.buf: Deque[float] = deque(maxlen=self.window)
        self.p = clamp(float(config.fact_flat_p_start), self.p_min, self.p_max)
        if config.fact_flat_mode == "adaptive":
            self.T = clamp(float(config.fact_flat_bps_min), self.bps_min, self.bps_max)
        else:
            self.T = clamp(float(config.fact_flat_bps), self.bps_min, self.bps_max)
        self.step = 0
        self.last_ts: Optional[int] = None

    def current_T(self) -> float:
        return self.T

    def observe(self, abs_ret_bps: float, ts: Optional[int] = None) -> None:
        if ts is not None and self.last_ts == ts:
            return
        if ts is not None:
            self.last_ts = ts
        self.step += 1
        self.buf.append(float(abs_ret_bps))
        if len(self.buf) < self.min_samples:
            return
        if len(self.buf) == self.min_samples and self.step == self.min_samples:
            values = np.array(self.buf, dtype=np.float32)
            if values.size:
                t_raw = float(np.quantile(values, self.p))
                t_raw = clamp(t_raw, self.bps_min, self.bps_max)
                self.T = t_raw
        if self.update_every > 0 and self.step % self.update_every == 0:
            flat_rate = sum(1 for v in self.buf if v < self.T) / len(self.buf)
            if flat_rate > self.target_hi:
                self.p = max(self.p_min, self.p - self.p_step)
            elif flat_rate < self.target_lo:
                self.p = min(self.p_max, self.p + self.p_step)
        values = np.array(self.buf, dtype=np.float32)
        if values.size == 0:
            return
        t_raw = float(np.quantile(values, self.p))
        t_raw = clamp(t_raw, self.bps_min, self.bps_max)
        self.T = clamp((1.0 - self.beta) * self.T + self.beta * t_raw, self.bps_min, self.bps_max)

    def to_dict(self) -> Dict[str, object]:
        return {
            "p": self.p,
            "T": self.T,
            "buf": list(self.buf),
        }

    @classmethod
    def from_dict(cls, config: FactConfig, payload: Optional[Dict[str, object]]) -> "AdaptiveFactFlatController":
        controller = cls(config)
        if not payload:
            return controller
        if "p" in payload:
            try:
                controller.p = clamp(float(payload["p"]), controller.p_min, controller.p_max)
            except (TypeError, ValueError):
                pass
        if "T" in payload:
            try:
                controller.T = clamp(float(payload["T"]), controller.bps_min, controller.bps_max)
            except (TypeError, ValueError):
                pass
        buf = payload.get("buf")
        if isinstance(buf, list):
            controller.buf = deque((float(v) for v in buf), maxlen=controller.window)
        return controller


class AdaptivePredFlatController:
    def __init__(self, config: DecisionConfig) -> None:
        self.delta = float(config.flat_max_delta)
        self.target_lo = float(config.pred_flat_target_lo)
        self.target_hi = float(config.pred_flat_target_hi)
        self.delta_min = float(config.pred_flat_delta_min)
        self.delta_max = float(config.pred_flat_delta_max)
        self.adjust_rate = float(config.pred_flat_adjust_rate)
        self.decay = float(config.pred_flat_ema_decay)
        self.min_action_acc = float(config.pred_flat_min_action_acc)
        self.ema_pred_flat = 0.0
        self.ema_action = 0.0
        self.ema_action_correct = 0.0

    def update(self, pred_dir: Direction, fact_dir: Direction, allow_adjust: bool = True) -> Dict[str, float]:
        pred_flat = 1.0 if pred_dir == Direction.FLAT else 0.0
        action = 1.0 if pred_dir != Direction.FLAT else 0.0
        action_correct = 1.0 if pred_dir != Direction.FLAT and pred_dir == fact_dir else 0.0
        decay = self.decay
        self.ema_pred_flat = (1.0 - decay) * self.ema_pred_flat + decay * pred_flat
        self.ema_action = (1.0 - decay) * self.ema_action + decay * action
        self.ema_action_correct = (1.0 - decay) * self.ema_action_correct + decay * action_correct
        total = self.ema_pred_flat + self.ema_action
        pred_flat_rate = self.ema_pred_flat / max(total, 1e-9)
        action_acc = self.ema_action_correct / max(self.ema_action, 1e-9)
        if allow_adjust:
            if pred_flat_rate > self.target_hi:
                self.delta *= 1.0 - self.adjust_rate
            elif pred_flat_rate < self.target_lo and action_acc < self.min_action_acc:
                self.delta *= 1.0 + self.adjust_rate
            self.delta = clamp(self.delta, self.delta_min, self.delta_max)
        return {"pred_flat_rate": pred_flat_rate, "action_acc": action_acc}

    def to_dict(self) -> Dict[str, object]:
        return {
            "delta": self.delta,
            "ema_pred_flat": self.ema_pred_flat,
            "ema_action": self.ema_action,
            "ema_action_correct": self.ema_action_correct,
        }

    @classmethod
    def from_dict(
        cls, config: DecisionConfig, payload: Optional[Dict[str, object]]
    ) -> "AdaptivePredFlatController":
        controller = cls(config)
        if not payload:
            return controller
        for key in ("delta", "ema_pred_flat", "ema_action", "ema_action_correct"):
            if key not in payload:
                continue
            try:
                setattr(controller, key, float(payload[key]))
            except (TypeError, ValueError):
                pass
        controller.delta = clamp(controller.delta, controller.delta_min, controller.delta_max)
        return controller


class MicroShareTracker:
    def __init__(self, decay: float) -> None:
        self.decay = float(decay)
        self.ema_micro = 0.0
        self.ema_actions = 0.0

    def update(self, micro: float, action: float) -> float:
        decay = self.decay
        self.ema_micro = (1.0 - decay) * self.ema_micro + decay * micro
        self.ema_actions = (1.0 - decay) * self.ema_actions + decay * action
        return self.share()

    def share(self) -> float:
        return self.ema_micro / max(self.ema_actions, 1e-9)

    def to_dict(self) -> Dict[str, object]:
        return {"ema_micro": self.ema_micro, "ema_actions": self.ema_actions}

    @classmethod
    def from_dict(cls, decay: float, payload: Optional[Dict[str, object]]) -> "MicroShareTracker":
        tracker = cls(decay)
        if not payload:
            return tracker
        for key in ("ema_micro", "ema_actions"):
            if key not in payload:
                continue
            try:
                setattr(tracker, key, float(payload[key]))
            except (TypeError, ValueError):
                pass
        return tracker
