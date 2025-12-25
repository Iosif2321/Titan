from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import UpdateConfig
from .model import Params, TwoHeadLinearModel, weight_norms
from .optimizer import AdamState, adam_update, init_adam


def _bce_with_logits(logit: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.softplus(logit) - target * logit


def _logits(params: Params, features: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    logit_up = jnp.dot(features, params["w_up"]) + params["b_up"]
    logit_down = jnp.dot(features, params["w_down"]) + params["b_down"]
    return logit_up, logit_down


def _loss_fn(
    params: Params,
    features: jnp.ndarray,
    y_up: jnp.ndarray,
    y_down: jnp.ndarray,
    l2_reg: float,
    sample_weight: jnp.ndarray,
) -> jnp.ndarray:
    logit_up, logit_down = _logits(params, features)
    loss_up = _bce_with_logits(logit_up, y_up)
    loss_down = _bce_with_logits(logit_down, y_down)
    l2 = l2_reg * (jnp.sum(params["w_up"] ** 2) + jnp.sum(params["w_down"] ** 2))
    return (loss_up + loss_down) * sample_weight + l2


@dataclass
class UpdateResult:
    loss: float | None
    lr_eff: float
    weight_norms: dict[str, float]
    skipped: bool


class OnlineTrainer:
    def __init__(
        self,
        model: TwoHeadLinearModel,
        update_config: UpdateConfig,
        opt_state: AdamState | None = None,
    ) -> None:
        self.model = model
        self.update_config = update_config
        self.opt_state = opt_state or init_adam(model.params)
        self._loss_and_grad = jax.jit(jax.value_and_grad(_loss_fn))

    def update(
        self,
        features: np.ndarray,
        y_up: int,
        y_down: int,
        reward: float,
        sample_weight: float = 1.0,
    ) -> UpdateResult:
        if self.update_config.no_update_on_flat and y_up == 0 and y_down == 0:
            return UpdateResult(
                loss=None,
                lr_eff=0.0,
                weight_norms=weight_norms(self.model.params),
                skipped=True,
            )

        lr_eff = self.update_config.learning_rate * (1 + self.update_config.reward_alpha * abs(reward))
        features_jnp = jnp.asarray(features, dtype=jnp.float32)
        weight = jnp.asarray(sample_weight, dtype=jnp.float32)
        loss, grads = self._loss_and_grad(
            self.model.params,
            features_jnp,
            jnp.asarray(y_up, dtype=jnp.float32),
            jnp.asarray(y_down, dtype=jnp.float32),
            self.model.config.l2_reg,
            weight,
        )
        params, opt_state = adam_update(self.model.params, grads, self.opt_state, lr_eff)
        if self.model.config.weight_clip > 0:
            params = {
                **params,
                "w_up": jnp.clip(params["w_up"], -self.model.config.weight_clip, self.model.config.weight_clip),
                "w_down": jnp.clip(
                    params["w_down"], -self.model.config.weight_clip, self.model.config.weight_clip
                ),
            }
        self.model.set_params(params)
        self.opt_state = opt_state
        return UpdateResult(
            loss=float(loss),
            lr_eff=lr_eff,
            weight_norms=weight_norms(self.model.params),
            skipped=False,
        )
