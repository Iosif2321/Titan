from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import ModelInitConfig, TrainingConfig
from .model import Params, init_params, logits, predict_proba, weight_norms
from .optimizer import AdamState, adam_update, init_adam


@dataclass
class UpdateResult:
    loss_task: float
    loss_total: float
    lr_eff: float
    weight_norms: dict[str, float]


class OnlineModel:
    def __init__(
        self,
        input_size: int,
        init_config: ModelInitConfig,
        training_config: TrainingConfig,
        lr_base: float,
        opt_state: AdamState | None = None,
    ) -> None:
        self.input_size = input_size
        self.init_config = init_config
        self.training_config = training_config
        self.lr_base = lr_base
        self.params = init_params(input_size, init_config)
        self.ema_params = {k: v.copy() for k, v in self.params.items()}
        self.anchor_params = {k: v.copy() for k, v in self.params.items()}
        self.opt_state = opt_state or init_adam(self.params)
        self._loss_and_grad = jax.jit(jax.value_and_grad(self._loss_fn, has_aux=True))
        self._predict = jax.jit(predict_proba)
        self._logits = jax.jit(logits)

    def _loss_fn(
        self,
        params: Params,
        features: jnp.ndarray,
        target: jnp.ndarray,
        anchor_params: Params,
        anchor_lambda: float,
        sample_weight: float,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        logit_up, logit_down = self._logits(params, features)
        logits_vec = jnp.stack([logit_up, logit_down])
        log_probs = jax.nn.log_softmax(logits_vec)
        loss_task = -jnp.sum(target * log_probs) * sample_weight
        anchor_loss = jnp.sum((params["w"] - anchor_params["w"]) ** 2) + jnp.sum(
            (params["b"] - anchor_params["b"]) ** 2
        )
        loss_total = loss_task + anchor_lambda * anchor_loss
        return loss_total, (loss_task, anchor_loss)

    def predict(self, features: np.ndarray, use_ema: bool = True) -> Tuple[float, float, float, float]:
        features_jnp = jnp.asarray(features, dtype=jnp.float32)
        params = self.ema_params if use_ema else self.params
        p_up, p_down = self._predict(params, features_jnp)
        logit_up, logit_down = self._logits(params, features_jnp)
        return float(logit_up), float(logit_down), float(p_up), float(p_down)

    def update(
        self,
        features: np.ndarray,
        target: np.ndarray,
        lr_eff: float,
        anchor_lambda: float,
        sample_weight: float,
        allow_anchor_update: bool,
    ) -> UpdateResult:
        features_jnp = jnp.asarray(features, dtype=jnp.float32)
        target_jnp = jnp.asarray(target, dtype=jnp.float32)
        (loss_total, (loss_task, _)), grads = self._loss_and_grad(
            self.params,
            features_jnp,
            target_jnp,
            self.anchor_params,
            anchor_lambda,
            sample_weight,
        )
        params, opt_state = adam_update(self.params, grads, self.opt_state, lr_eff)
        if self.init_config.weight_clip > 0:
            params = {
                **params,
                "w": jnp.clip(
                    params["w"],
                    -self.init_config.weight_clip,
                    self.init_config.weight_clip,
                ),
                "b": jnp.clip(
                    params["b"],
                    -self.init_config.weight_clip,
                    self.init_config.weight_clip,
                ),
            }
        self.params = params
        self.opt_state = opt_state

        ema_decay = self.training_config.ema_decay
        self.ema_params = jax.tree_util.tree_map(
            lambda e, p: ema_decay * e + (1.0 - ema_decay) * p,
            self.ema_params,
            self.params,
        )
        if allow_anchor_update:
            anchor_decay = self.training_config.anchor_decay
            self.anchor_params = jax.tree_util.tree_map(
                lambda a, p: anchor_decay * a + (1.0 - anchor_decay) * p,
                self.anchor_params,
                self.params,
            )
        return UpdateResult(
            loss_task=float(loss_task),
            loss_total=float(loss_total),
            lr_eff=lr_eff,
            weight_norms=weight_norms(self.params),
        )
