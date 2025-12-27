from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class AffineLogitCalibState:
    a: float
    b: float
    n: int = 0


@dataclass
class AffineLogitCalibConfig:
    lr: float = 0.005
    a_min: float = 0.30
    a_max: float = 2.0
    b_min: float = -1.0
    b_max: float = 1.0
    l2_a: float = 0.01
    l2_b: float = 0.001
    a_anchor: float = 1.0
    b_anchor: float = 0.0


def calibrate_from_logits(
    logit_up: float,
    logit_down: float,
    st: AffineLogitCalibState,
    flip: bool = False,
) -> Tuple[float, float]:
    m = logit_up - logit_down
    if flip:
        m = -m
    m_cal = st.a * m + st.b
    p_up = _sigmoid(m_cal)
    return p_up, 1.0 - p_up


def update_affine_calib(
    st: AffineLogitCalibState,
    logit_up: float,
    logit_down: float,
    y_up: float,
    weight: float,
    cfg: AffineLogitCalibConfig,
    flip: bool = False,
) -> AffineLogitCalibState:
    m = logit_up - logit_down
    if flip:
        m = -m
    m_cal = st.a * m + st.b
    p_up = _sigmoid(m_cal)

    grad_scale = (p_up - y_up) * weight
    grad_a = grad_scale * m + cfg.l2_a * (st.a - cfg.a_anchor)
    grad_b = grad_scale + cfg.l2_b * (st.b - cfg.b_anchor)

    new_a = _clamp(st.a - cfg.lr * grad_a, cfg.a_min, cfg.a_max)
    new_b = _clamp(st.b - cfg.lr * grad_b, cfg.b_min, cfg.b_max)
    return AffineLogitCalibState(a=new_a, b=new_b, n=st.n + 1)
