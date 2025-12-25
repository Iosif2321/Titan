from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import ModelInitConfig

Params = Dict[str, jnp.ndarray]


def _heuristic_weights(input_size: int, logit_scale: float) -> jnp.ndarray:
    if input_size <= 0:
        return jnp.zeros((0,), dtype=jnp.float32)
    weights = np.linspace(0.5, 1.5, input_size, dtype=np.float32)
    norm = np.linalg.norm(weights) + 1e-8
    weights = weights / norm
    weights = weights * logit_scale
    return jnp.asarray(weights, dtype=jnp.float32)


def init_params(input_size: int, config: ModelInitConfig) -> Params:
    if input_size <= 0:
        raise ValueError("Model input_size must be positive")

    if config.init_mode == "heuristic":
        w_up = _heuristic_weights(input_size, config.logit_scale)
        w_down = -w_up
        w = jnp.stack([w_up, w_down], axis=1)
        b = jnp.zeros((2,), dtype=jnp.float32)
        return {"w": w, "b": b}

    key = jax.random.PRNGKey(config.seed)
    w = jax.random.normal(key, (input_size, 2), dtype=jnp.float32) * 0.1
    b = jnp.zeros((2,), dtype=jnp.float32)
    return {"w": w, "b": b}


def logits(params: Params, features: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    logits_vec = jnp.dot(features, params["w"]) + params["b"]
    return logits_vec[0], logits_vec[1]


def predict_proba(params: Params, features: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    logits_vec = jnp.dot(features, params["w"]) + params["b"]
    probs = jax.nn.softmax(logits_vec)
    return probs[0], probs[1]


def weight_norms(params: Params) -> Dict[str, float]:
    w_norm = float(jnp.linalg.norm(params["w"]))
    b_norm = float(jnp.linalg.norm(params["b"]))
    return {"w": w_norm, "b": b_norm}
