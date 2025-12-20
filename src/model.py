from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .features import FeatureSpec


@dataclass(frozen=True)
class ModelConfig:
    input_size: int
    init_mode: str = "heuristic"
    seed: int = 42
    logit_scale: float = 3.0


Params = Dict[str, jnp.ndarray]


def _heuristic_weights(spec: FeatureSpec, logit_scale: float) -> jnp.ndarray:
    if spec.n_returns > 0:
        returns_w = np.linspace(0.5, 1.5, spec.n_returns, dtype=np.float32)
    else:
        returns_w = np.array([], dtype=np.float32)

    extras_w = np.array([0.0, 0.8, 0.1], dtype=np.float32)
    weights = np.concatenate([returns_w, extras_w])
    norm = np.linalg.norm(weights) + 1e-8
    weights = weights / norm
    weights = weights * logit_scale
    return jnp.asarray(weights, dtype=jnp.float32)


def init_params(config: ModelConfig, spec: FeatureSpec) -> Params:
    if config.input_size != spec.input_size:
        raise ValueError("Model input_size does not match feature size")

    if config.init_mode == "heuristic":
        w_up = _heuristic_weights(spec, config.logit_scale)
        w_down = -w_up
        b_up = jnp.array(0.0, dtype=jnp.float32)
        b_down = jnp.array(0.0, dtype=jnp.float32)
        return {"w_up": w_up, "b_up": b_up, "w_down": w_down, "b_down": b_down}

    key = jax.random.PRNGKey(config.seed)
    k1, k2 = jax.random.split(key)
    w_up = jax.random.normal(k1, (config.input_size,), dtype=jnp.float32) * 0.1
    w_down = jax.random.normal(k2, (config.input_size,), dtype=jnp.float32) * 0.1
    b_up = jnp.array(0.0, dtype=jnp.float32)
    b_down = jnp.array(0.0, dtype=jnp.float32)
    return {"w_up": w_up, "b_up": b_up, "w_down": w_down, "b_down": b_down}


def _predict(params: Params, features: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    logit_up = jnp.dot(features, params["w_up"]) + params["b_up"]
    logit_down = jnp.dot(features, params["w_down"]) + params["b_down"]
    return jax.nn.sigmoid(logit_up), jax.nn.sigmoid(logit_down)


class TwoHeadLinearModel:
    def __init__(self, config: ModelConfig, spec: FeatureSpec) -> None:
        self.config = config
        self.spec = spec
        self.params = init_params(config, spec)
        self._predict = jax.jit(_predict)

    def predict(self, features: jnp.ndarray) -> Tuple[float, float]:
        p_up, p_down = self._predict(self.params, features)
        return float(p_up), float(p_down)
