from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Direction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    FLAT = "FLAT"


@dataclass(frozen=True)
class Candle:
    start_ts: int
    end_ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    confirmed: bool
    tf: str

    @property
    def interval(self) -> str:
        return self.tf

    @property
    def o(self) -> float:
        return self.open

    @property
    def h(self) -> float:
        return self.high

    @property
    def l(self) -> float:
        return self.low

    @property
    def c(self) -> float:
        return self.close


@dataclass(frozen=True)
class Prediction:
    ts: int
    tf: str
    model_id: str
    model_type: str
    candle_ts: int
    target_ts: int
    logits_up: float
    logits_down: float
    p_up: float
    p_down: float
    direction: Direction
    confidence: float
    used_ema: bool
    context_key_used: str
    decision_key_used: str
    trust_ctx: float
    trust_dec: float
    prior_ctx: Dict[str, float]
    prior_win_dec: float
    flat_thresholds: Dict[str, float]
    notes: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def dir(self) -> Direction:
        return self.direction


@dataclass(frozen=True)
class Fact:
    tf: str
    prev_ts: int
    curr_ts: int
    close_prev: float
    close_curr: float
    ret_bps: float
    fact_flat_bps: float
    direction: Direction

    @property
    def ts(self) -> int:
        return self.curr_ts

    @property
    def dir(self) -> Direction:
        return self.direction


@dataclass(frozen=True)
class UpdateEvent:
    ts: int
    tf: str
    model_id: str
    model_type: str
    target_ts: int
    candle_ts: int
    pred_dir: Direction
    pred_conf: float
    fact_dir: Direction
    ret_bps: float
    reward: float
    loss_task: float
    loss_total: float
    lr_eff: float
    anchor_lambda_eff: float
    weight_norms: Dict[str, float]
    anchor_update_applied: bool
    calib_a: Optional[float] = None
    calib_b: Optional[float] = None
    calib_n: Optional[int] = None
    p_up_raw: Optional[float] = None
    p_down_raw: Optional[float] = None
    p_up_cal: Optional[float] = None
    p_down_cal: Optional[float] = None
    margin_raw: Optional[float] = None
    margin_cal: Optional[float] = None
    close_prev: Optional[float] = None
    close_curr: Optional[float] = None
    delta: Optional[float] = None
    features: Optional[List[float]] = None
    notes: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
