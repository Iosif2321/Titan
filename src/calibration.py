from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .types import Direction
from .utils import clamp


def _softmax2(logit_up: float, logit_down: float) -> tuple[float, float]:
    max_logit = logit_up if logit_up > logit_down else logit_down
    exp_up = math.exp(logit_up - max_logit)
    exp_down = math.exp(logit_down - max_logit)
    denom = exp_up + exp_down
    if denom <= 0.0:
        return 0.5, 0.5
    return exp_up / denom, exp_down / denom


@dataclass
class TemperatureState:
    log_temp: float
    lr: float
    min_temp: float
    max_temp: float
    count: int
    last_grad: float


class TemperatureScaler:
    def __init__(
        self,
        init_temp: float = 1.0,
        min_temp: float = 0.5,
        max_temp: float = 5.0,
        lr: float = 0.01,
    ) -> None:
        init_temp = max(init_temp, 1e-6)
        self.log_temp = math.log(init_temp)
        self.lr = lr
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.count = 0
        self.last_grad = 0.0

    @property
    def temp(self) -> float:
        return clamp(math.exp(self.log_temp), self.min_temp, self.max_temp)

    def scale_logits(self, logit_up: float, logit_down: float) -> tuple[float, float]:
        temp = self.temp
        if temp <= 0.0:
            return logit_up, logit_down
        return logit_up / temp, logit_down / temp

    def probs(self, logit_up: float, logit_down: float) -> tuple[float, float]:
        scaled_up, scaled_down = self.scale_logits(logit_up, logit_down)
        return _softmax2(scaled_up, scaled_down)

    def update(self, logit_up: float, logit_down: float, fact_dir: Direction) -> Optional[float]:
        if fact_dir not in (Direction.UP, Direction.DOWN):
            return None
        temp = self.temp
        p_up, p_down = self.probs(logit_up, logit_down)
        y_up = 1.0 if fact_dir == Direction.UP else 0.0
        y_down = 1.0 - y_up
        grad = -(1.0 / temp) * ((p_up - y_up) * logit_up + (p_down - y_down) * logit_down)
        self.log_temp -= self.lr * grad
        self.log_temp = math.log(clamp(math.exp(self.log_temp), self.min_temp, self.max_temp))
        self.count += 1
        self.last_grad = float(grad)
        return float(grad)

    def snapshot(self) -> dict[str, float | int]:
        return {
            "temp": self.temp,
            "lr": self.lr,
            "min_temp": self.min_temp,
            "max_temp": self.max_temp,
            "count": self.count,
            "last_grad": self.last_grad,
        }

    def to_state(self) -> TemperatureState:
        return TemperatureState(
            log_temp=self.log_temp,
            lr=self.lr,
            min_temp=self.min_temp,
            max_temp=self.max_temp,
            count=self.count,
            last_grad=self.last_grad,
        )

    def load_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        if "log_temp" in state:
            try:
                self.log_temp = float(state["log_temp"])
            except (TypeError, ValueError):
                pass
        if "lr" in state:
            try:
                self.lr = float(state["lr"])
            except (TypeError, ValueError):
                pass
        if "min_temp" in state:
            try:
                self.min_temp = float(state["min_temp"])
            except (TypeError, ValueError):
                pass
        if "max_temp" in state:
            try:
                self.max_temp = float(state["max_temp"])
            except (TypeError, ValueError):
                pass
        if "count" in state:
            try:
                self.count = int(state["count"])
            except (TypeError, ValueError):
                pass
        if "last_grad" in state:
            try:
                self.last_grad = float(state["last_grad"])
            except (TypeError, ValueError):
                pass
