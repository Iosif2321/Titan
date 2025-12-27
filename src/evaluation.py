from __future__ import annotations

from typing import Tuple

from .config import UpdateConfig
from .types import Candle, Direction, Fact, Prediction


def _ret_bps(close_prev: float, close_curr: float) -> float:
    if close_prev <= 0:
        return 0.0
    return ((close_curr - close_prev) / close_prev) * 10_000.0


def fact_from_candles(prev: Candle, curr: Candle, fact_flat_bps: float) -> Fact:
    ret_bps = _ret_bps(prev.close, curr.close)
    if abs(ret_bps) <= fact_flat_bps:
        direction = Direction.FLAT
    else:
        direction = Direction.UP if curr.close > prev.close else Direction.DOWN
    return Fact(
        tf=curr.interval,
        prev_ts=prev.start_ts,
        curr_ts=curr.start_ts,
        close_prev=prev.close,
        close_curr=curr.close,
        ret_bps=ret_bps,
        fact_flat_bps=fact_flat_bps,
        direction=direction,
    )


def reward_from_prediction(pred: Prediction, fact: Fact, cfg: UpdateConfig) -> float:
    if pred.direction == Direction.FLAT:
        reward = (
            cfg.reward_flat_correct
            if fact.direction == Direction.FLAT
            else -cfg.penalty_flat_wrong
        )
    else:
        reward = (
            cfg.reward_dir_correct
            if pred.direction == fact.direction
            else -cfg.penalty_dir_wrong
        )

    if cfg.reward_confidence_divisor > 0:
        scale = min(cfg.reward_confidence_cap, pred.confidence / cfg.reward_confidence_divisor)
        reward *= scale
    return reward


def action_weight_from_prediction(pred: Prediction, fact: Fact, cfg: UpdateConfig) -> float:
    if pred.direction == fact.direction:
        return 1.0
    if pred.direction == Direction.FLAT or fact.direction == Direction.FLAT:
        return 1.0 + cfg.flat_miss_penalty
    return 1.0 + cfg.action_miss_penalty


def targets_from_fact(fact: Fact) -> Tuple[int, int]:
    if fact.direction == Direction.UP:
        return 1, 0
    if fact.direction == Direction.DOWN:
        return 0, 1
    return 0, 0
