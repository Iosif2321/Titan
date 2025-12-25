from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import jax
import jax.numpy as jnp

Params = Dict[str, jnp.ndarray]


@dataclass
class AdamState:
    m: Params
    v: Params
    t: int


def init_adam(params: Params) -> AdamState:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return AdamState(m=zeros, v=zeros, t=0)


def adam_update(
    params: Params,
    grads: Params,
    state: AdamState,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[Params, AdamState]:
    t = state.t + 1
    m = jax.tree_util.tree_map(lambda m_v, g: beta1 * m_v + (1 - beta1) * g, state.m, grads)
    v = jax.tree_util.tree_map(lambda v_v, g: beta2 * v_v + (1 - beta2) * (g * g), state.v, grads)
    m_hat = jax.tree_util.tree_map(lambda m_v: m_v / (1 - beta1**t), m)
    v_hat = jax.tree_util.tree_map(lambda v_v: v_v / (1 - beta2**t), v)
    params = jax.tree_util.tree_map(
        lambda p, m_v, v_v: p - lr * m_v / (jnp.sqrt(v_v) + eps), params, m_hat, v_hat
    )
    return params, AdamState(m=m, v=v, t=t)
