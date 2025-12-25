from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .calibration import (
    AffineLogitCalibConfig,
    AffineLogitCalibState,
    calibrate_from_logits,
    update_affine_calib,
)
from .config import DecisionConfig, FeatureConfig, ModelInitConfig, ModelLRConfig, TrainingConfig
from .features import FeatureBuilder, FeatureBundle, MODEL_OSC, MODEL_TREND, MODEL_VOL
from .metrics import CalibrationMetrics, RollingMetrics
from .model import weight_norms
from .online_model import OnlineModel, UpdateResult
from .optimizer import AdamState
from .pattern_store import PatternStore
from .state_store import ModelStateStore
from .types import Candle, Direction, Fact, Prediction, UpdateEvent
from .utils import clamp, interval_to_ms, now_ms

_CALIB_INIT = {
    MODEL_TREND: (0.50, 0.00),
    MODEL_OSC: (0.20, 0.00),
    MODEL_VOL: (0.20, 0.10),
}


@dataclass
class PendingPrediction:
    candle_ts: int
    target_ts: int
    close_prev: float
    features: np.ndarray
    logits_up: float
    logits_down: float
    p_up_raw: float
    p_down_raw: float
    p_up: float
    p_down: float
    conf_raw: float
    conf_cal: float
    pred_dir: Direction
    pred_conf: float
    context_key_used: str
    decision_key_used: str
    trust_ctx: float
    trust_dec: float
    prior_ctx: Dict[str, float]
    prior_win_dec: float
    flat_margin: float
    lr_eff: float
    anchor_lambda_eff: float
    anti_pattern: bool
    calib_a: float
    calib_b: float
    calib_n: int


@dataclass
class RunnerConfig:
    use_ema: bool = True
    use_patterns: bool = True
    update_patterns: bool = True
    enable_training: bool = True
    enable_anchor: bool = True
    enable_calibration_update: bool = True


class CandleBuffer:
    def __init__(self, maxlen: int) -> None:
        self.maxlen = maxlen
        self._candles = deque(maxlen=maxlen)
        self.last_ts: Optional[int] = None

    def seed(self, candles: List[Candle]) -> None:
        self._candles.clear()
        for candle in candles[-self.maxlen :]:
            self._candles.append(candle)
        self.last_ts = self._candles[-1].start_ts if self._candles else None

    def add(self, candle: Candle) -> bool:
        if self.last_ts is None or candle.start_ts > self.last_ts:
            self._candles.append(candle)
            self.last_ts = candle.start_ts
            return True
        if candle.start_ts == self.last_ts:
            if self._candles:
                self._candles[-1] = candle
            else:
                self._candles.append(candle)
            return False
        return False

    def values(self) -> List[Candle]:
        return list(self._candles)


class ModelRunner:
    def __init__(
        self,
        tf: str,
        model_type: str,
        feature_builder: FeatureBuilder,
        model_init: ModelInitConfig,
        training: TrainingConfig,
        decision: DecisionConfig,
        lr_base: float,
        pattern_store: PatternStore,
        state_store: ModelStateStore,
        metrics_window: int = 200,
        runner_config: RunnerConfig | None = None,
    ) -> None:
        config = runner_config or RunnerConfig()
        self.tf = tf
        self.model_type = model_type
        self.model_id = model_type
        self.feature_builder = feature_builder
        self.training = training
        self.decision = decision
        self.lr_base = lr_base
        self.pattern_store = pattern_store
        self.state_store = state_store
        self.use_ema = config.use_ema
        self.use_patterns = config.use_patterns
        self.update_patterns = config.update_patterns
        self.enable_training = config.enable_training
        self.enable_anchor = config.enable_anchor
        self.enable_calibration_update = config.enable_calibration_update
        self.pending: Dict[int, PendingPrediction] = {}
        self.metrics = RollingMetrics(metrics_window)
        self.calibration = CalibrationMetrics(training.calibration_bins, metrics_window)
        init_a, init_b = _CALIB_INIT.get(model_type, (1.0, 0.0))
        self.calib_state = AffineLogitCalibState(a=init_a, b=init_b, n=0)
        self.calib_config = AffineLogitCalibConfig(
            lr=training.calib_lr,
            a_min=training.calib_a_min,
            a_max=training.calib_a_max,
            b_min=training.calib_b_min,
            b_max=training.calib_b_max,
            l2_a=training.calib_l2_a,
            l2_b=training.calib_l2_b,
            a_anchor=init_a,
            b_anchor=init_b,
        )
        self.fact_ema = {
            Direction.UP: 1.0 / 3.0,
            Direction.DOWN: 1.0 / 3.0,
            Direction.FLAT: 1.0 / 3.0,
        }
        self.last_class_weight = 1.0
        self.last_perf_mult = 1.0
        self.update_count = 0
        self.last_save_ts = 0
        self.model = OnlineModel(
            input_size=feature_builder.spec.input_size,
            init_config=model_init,
            training_config=training,
            lr_base=lr_base,
        )

    def load_state(self, state: Optional[Dict[str, object]]) -> None:
        if not state:
            return
        params = state.get("params")
        ema_params = state.get("ema_params")
        anchor_params = state.get("anchor_params")
        opt_state = state.get("opt_state")
        metrics_state = state.get("metrics_state")
        if params and ema_params and anchor_params and opt_state:
            self.model.params = _to_jax(params)
            self.model.ema_params = _to_jax(ema_params)
            self.model.anchor_params = _to_jax(anchor_params)
            if isinstance(opt_state, AdamState):
                self.model.opt_state = AdamState(
                    m=_to_jax(opt_state.m),
                    v=_to_jax(opt_state.v),
                    t=opt_state.t,
                )
            else:
                self.model.opt_state = opt_state
        if isinstance(metrics_state, dict):
            if "rolling" in metrics_state:
                self.metrics.load_state(metrics_state.get("rolling", {}))
                self.calibration.load_state(metrics_state.get("calibration", {}))
                affine_state = metrics_state.get("calibration_affine", {})
                if isinstance(affine_state, dict):
                    try:
                        self.calib_state = AffineLogitCalibState(
                            a=float(affine_state.get("a", self.calib_state.a)),
                            b=float(affine_state.get("b", self.calib_state.b)),
                            n=int(affine_state.get("n", self.calib_state.n)),
                        )
                    except (TypeError, ValueError):
                        pass
                fact_ema = metrics_state.get("fact_ema")
                if isinstance(fact_ema, dict):
                    for key, direction in (
                        ("UP", Direction.UP),
                        ("DOWN", Direction.DOWN),
                        ("FLAT", Direction.FLAT),
                    ):
                        try:
                            self.fact_ema[direction] = float(fact_ema.get(key, self.fact_ema[direction]))
                        except (TypeError, ValueError):
                            continue
            else:
                self.metrics.load_state(metrics_state)

    def set_runtime_flags(
        self,
        *,
        enable_training: Optional[bool] = None,
        update_patterns: Optional[bool] = None,
        enable_calibration_update: Optional[bool] = None,
        enable_anchor: Optional[bool] = None,
        use_ema: Optional[bool] = None,
        use_patterns: Optional[bool] = None,
    ) -> None:
        if enable_training is not None:
            self.enable_training = enable_training
        if update_patterns is not None:
            self.update_patterns = update_patterns
        if enable_calibration_update is not None:
            self.enable_calibration_update = enable_calibration_update
        if enable_anchor is not None:
            self.enable_anchor = enable_anchor
        if use_ema is not None:
            self.use_ema = use_ema
        if use_patterns is not None:
            self.use_patterns = use_patterns

    def predict(self, candle: Candle, candles: List[Candle], now_ts: int) -> Optional[Prediction]:
        bundle = self.feature_builder.build(candles)
        if bundle is None:
            return None
        if self.use_patterns:
            context_key_used, trust_ctx, prior_ctx = self._context_stats(bundle, now_ts)
        else:
            context_key_used = f"{self.tf}:{self.model_id}:disabled"
            trust_ctx = 0.0
            prior_ctx = {
                "p_up": 1.0 / 3.0,
                "p_down": 1.0 / 3.0,
                "p_flat": 1.0 / 3.0,
            }
        logit_up, logit_down, raw_p_up, raw_p_down = self.model.predict(
            bundle.values, use_ema=self.use_ema
        )
        self._sync_calibration_config()
        p_up, p_down = calibrate_from_logits(logit_up, logit_down, self.calib_state)
        base_dir = decide_direction(p_up, p_down, self.decision.flat_max_delta)

        if self.use_patterns:
            decision_key_used, trust_dec, prior_win_dec, anti_pattern = self._decision_stats(
                bundle,
                base_dir,
                now_ts,
            )
            flat_margin = self._apply_context_adjustments(
                base_dir,
                trust_ctx,
                prior_ctx,
                self.decision.flat_max_delta,
            )
        else:
            decision_key_used = f"{context_key_used}|PRED={base_dir.value}"
            trust_dec = 0.0
            prior_win_dec = 0.5
            anti_pattern = False
            flat_margin = self.decision.flat_max_delta
        notes: List[str] = []
        final_dir = base_dir
        if anti_pattern and base_dir != Direction.FLAT and self.use_patterns:
            flat_margin = clamp(
                flat_margin + self.decision.anti_flat_delta_boost + self.decision.anti_flat_prob_boost,
                0.0,
                0.5,
            )
            final_dir = decide_direction(p_up, p_down, flat_margin)
            if final_dir == Direction.FLAT:
                notes.append("anti_pattern_abstain")

        if final_dir != base_dir and self.use_patterns:
            decision_key_used, trust_dec, prior_win_dec, _ = self._decision_stats(
                bundle,
                final_dir,
                now_ts,
            )

        lr_eff, anchor_lambda_eff = self._adaptation(trust_dec, prior_win_dec)
        conf_raw = max(raw_p_up, raw_p_down)
        conf_cal = max(p_up, p_down)
        calib_a = float(self.calib_state.a)
        calib_b = float(self.calib_state.b)
        calib_n = int(self.calib_state.n)
        target_ts = candle.start_ts + interval_to_ms(self.tf)
        prediction = Prediction(
            ts=now_ts,
            tf=self.tf,
            model_id=self.model_id,
            model_type=self.model_type,
            candle_ts=candle.start_ts,
            target_ts=target_ts,
            logits_up=logit_up,
            logits_down=logit_down,
            p_up=p_up,
            p_down=p_down,
            direction=final_dir,
            confidence=conf_cal,
            used_ema=True,
            context_key_used=context_key_used,
            decision_key_used=decision_key_used,
            trust_ctx=trust_ctx,
            trust_dec=trust_dec,
            prior_ctx=prior_ctx,
            prior_win_dec=prior_win_dec,
            flat_thresholds={
                "flat_margin": flat_margin,
                "base_flat_margin": self.decision.flat_max_delta,
            },
            notes=",".join(notes),
            meta={
                "feature_schema": self.feature_builder.spec.schema_version,
                "p_up_raw": raw_p_up,
                "p_down_raw": raw_p_down,
                "p_up_cal": p_up,
                "p_down_cal": p_down,
                "conf_raw": conf_raw,
                "conf_cal": conf_cal,
                "calib_a": calib_a,
                "calib_b": calib_b,
                "calib_n": calib_n,
            },
        )

        self.pending[target_ts] = PendingPrediction(
            candle_ts=candle.start_ts,
            target_ts=target_ts,
            close_prev=candle.close,
            features=bundle.values,
            logits_up=logit_up,
            logits_down=logit_down,
            p_up_raw=raw_p_up,
            p_down_raw=raw_p_down,
            p_up=p_up,
            p_down=p_down,
            conf_raw=conf_raw,
            conf_cal=conf_cal,
            pred_dir=final_dir,
            pred_conf=conf_cal,
            context_key_used=context_key_used,
            decision_key_used=decision_key_used,
            trust_ctx=trust_ctx,
            trust_dec=trust_dec,
            prior_ctx=prior_ctx,
            prior_win_dec=prior_win_dec,
            flat_margin=flat_margin,
            lr_eff=lr_eff,
            anchor_lambda_eff=anchor_lambda_eff,
            anti_pattern=anti_pattern,
            calib_a=calib_a,
            calib_b=calib_b,
            calib_n=calib_n,
        )
        return prediction

    def on_fact(self, candle: Candle, now_ts: int) -> Optional[tuple[Fact, UpdateEvent]]:
        pending = self.pending.pop(candle.start_ts, None)
        if pending is None:
            return None
        self._sync_calibration_config()
        fact = fact_from_prices(
            tf=self.tf,
            prev_ts=pending.candle_ts,
            curr_ts=candle.start_ts,
            close_prev=pending.close_prev,
            close_curr=candle.close,
            flat_bps=self.training.flat_bps,
        )
        reward, reward_reason, reward_base, flat_penalty = reward_from_dirs(
            pending.pred_dir, fact.direction, self.training
        )
        class_weight = self._class_weight(fact.direction)
        sample_weight = self.training.flat_update_weight if fact.direction == Direction.FLAT else 1.0
        sample_weight *= class_weight
        self._update_fact_ema(fact.direction)
        target = target_vector(fact.direction)

        anchor_lambda_eff = pending.anchor_lambda_eff
        if not self.enable_anchor or not self.enable_training:
            anchor_lambda_eff = 0.0

        allow_anchor_update = reward > 0
        if self.use_patterns:
            allow_anchor_update = allow_anchor_update or (
                pending.trust_dec >= self.pattern_store.config.fine_trust_threshold
            )
            if reward == 0 and pending.anti_pattern:
                allow_anchor_update = False
        if not self.enable_anchor or not self.enable_training:
            allow_anchor_update = False

        if not self.enable_training:
            update_result = UpdateResult(
                loss_task=0.0,
                loss_total=0.0,
                lr_eff=0.0,
                weight_norms=weight_norms(self.model.params),
            )
        else:
            update_result = self.model.update(
                pending.features,
                target,
                lr_eff=pending.lr_eff,
                anchor_lambda=anchor_lambda_eff,
                sample_weight=sample_weight,
                allow_anchor_update=allow_anchor_update,
            )
        self.metrics.update(pending.pred_dir, fact.direction, reward)
        if pending.pred_dir != Direction.FLAT:
            self.calibration.update(
                pending.pred_conf,
                pending.pred_dir == fact.direction,
                abs(fact.ret_bps),
            )
        calib_weight = 1.0
        if fact.direction == Direction.UP:
            y_up = 1.0
        elif fact.direction == Direction.DOWN:
            y_up = 0.0
        else:
            y_up = 0.5
        if fact.direction == Direction.FLAT or abs(fact.ret_bps) <= self.training.calib_flat_bps:
            y_up = 0.5
            calib_weight = self.training.calib_flat_weight
        if self.enable_calibration_update and self.enable_training:
            self.calib_state = update_affine_calib(
                self.calib_state,
                pending.logits_up,
                pending.logits_down,
                y_up=y_up,
                weight=calib_weight,
                cfg=self.calib_config,
            )
        self.last_class_weight = class_weight
        self.update_count += 1

        update_event = UpdateEvent(
            ts=now_ts,
            tf=self.tf,
            model_id=self.model_id,
            model_type=self.model_type,
            target_ts=pending.target_ts,
            candle_ts=pending.candle_ts,
            pred_dir=pending.pred_dir,
            pred_conf=pending.pred_conf,
            fact_dir=fact.direction,
            ret_bps=fact.ret_bps,
            reward=reward,
            loss_task=update_result.loss_task,
            loss_total=update_result.loss_total,
            lr_eff=update_result.lr_eff,
            anchor_lambda_eff=anchor_lambda_eff,
            weight_norms={
                "params": weight_norms(self.model.params),
                "ema": weight_norms(self.model.ema_params),
                "anchor": weight_norms(self.model.anchor_params),
            },
            anchor_update_applied=allow_anchor_update,
            notes="",
            meta={
                "reward_reason": reward_reason,
                "reward_base": reward_base,
                "flat_penalty": flat_penalty,
                "sample_weight": sample_weight,
                "class_weight": class_weight,
                "p_up_raw": pending.p_up_raw,
                "p_down_raw": pending.p_down_raw,
                "p_up_cal": pending.p_up,
                "p_down_cal": pending.p_down,
                "conf_raw": pending.conf_raw,
                "conf_cal": pending.conf_cal,
                "calib_a": pending.calib_a,
                "calib_b": pending.calib_b,
                "calib_n": pending.calib_n,
                "calib_weight": calib_weight,
            },
        )

        if self.update_patterns and self.use_patterns:
            self.pattern_store.update_patterns(
                tf=self.tf,
                model_id=self.model_id,
                context_key=pending.context_key_used,
                decision_key=pending.decision_key_used,
                pred_dir=pending.pred_dir,
                fact_dir=fact.direction,
                reward=reward,
                ts=now_ts,
            )
            self.pattern_store.record_event(
                kind="context",
                pattern_key=pending.context_key_used,
                ts=now_ts,
                tf=self.tf,
                model_id=self.model_id,
                candle_ts=pending.candle_ts,
                target_ts=pending.target_ts,
                close_prev=pending.close_prev,
                close_curr=candle.close,
                pred_dir=pending.pred_dir,
                pred_conf=pending.pred_conf,
                fact_dir=fact.direction,
                reward=reward,
                ret_bps=fact.ret_bps,
                lr_eff=update_result.lr_eff,
                anchor_lambda_eff=anchor_lambda_eff,
            )
            self.pattern_store.record_event(
                kind="decision",
                pattern_key=pending.decision_key_used,
                ts=now_ts,
                tf=self.tf,
                model_id=self.model_id,
                candle_ts=pending.candle_ts,
                target_ts=pending.target_ts,
                close_prev=pending.close_prev,
                close_curr=candle.close,
                pred_dir=pending.pred_dir,
                pred_conf=pending.pred_conf,
                fact_dir=fact.direction,
                reward=reward,
                ret_bps=fact.ret_bps,
                lr_eff=update_result.lr_eff,
                anchor_lambda_eff=anchor_lambda_eff,
            )
        return fact, update_event

    def maybe_save(self, now_ts: int, autosave_seconds: int, autosave_updates: int) -> bool:
        should_save = False
        if autosave_updates > 0 and self.update_count % autosave_updates == 0:
            should_save = True
        if autosave_seconds > 0 and now_ts - self.last_save_ts >= autosave_seconds * 1000:
            should_save = True
        if not should_save:
            return False
        self.state_store.save_state(
            model_id=self.model_id,
            tf=self.tf,
            saved_at=now_ts,
            params=_to_numpy(self.model.params),
            ema_params=_to_numpy(self.model.ema_params),
            anchor_params=_to_numpy(self.model.anchor_params),
            opt_state=self.model.opt_state,
            metrics={
                "rolling": self.metrics.to_state(),
                "calibration": self.calibration.to_state(),
                "calibration_affine": {
                    "a": float(self.calib_state.a),
                    "b": float(self.calib_state.b),
                    "n": int(self.calib_state.n),
                },
                "fact_ema": {
                    "UP": self.fact_ema.get(Direction.UP, 0.0),
                    "DOWN": self.fact_ema.get(Direction.DOWN, 0.0),
                    "FLAT": self.fact_ema.get(Direction.FLAT, 0.0),
                },
            },
        )
        self.last_save_ts = now_ts
        return True

    def metrics_snapshot(self) -> Dict[str, object]:
        snapshot = self.metrics.snapshot()
        snapshot["calibration"] = self.calibration.snapshot()
        snapshot["calibration_affine"] = {
            "a": float(self.calib_state.a),
            "b": float(self.calib_state.b),
            "n": int(self.calib_state.n),
            "lr": float(self.calib_config.lr),
            "a_min": float(self.calib_config.a_min),
            "a_max": float(self.calib_config.a_max),
            "b_min": float(self.calib_config.b_min),
            "b_max": float(self.calib_config.b_max),
            "a_anchor": float(self.calib_config.a_anchor),
            "b_anchor": float(self.calib_config.b_anchor),
        }
        snapshot["fact_ema"] = {
            "UP": self.fact_ema.get(Direction.UP, 0.0),
            "DOWN": self.fact_ema.get(Direction.DOWN, 0.0),
            "FLAT": self.fact_ema.get(Direction.FLAT, 0.0),
        }
        snapshot["last_class_weight"] = self.last_class_weight
        snapshot["perf_lr_mult"] = self.last_perf_mult
        return snapshot

    def _sync_calibration_config(self) -> None:
        self.calib_config.lr = self.training.calib_lr
        self.calib_config.a_min = self.training.calib_a_min
        self.calib_config.a_max = self.training.calib_a_max
        self.calib_config.b_min = self.training.calib_b_min
        self.calib_config.b_max = self.training.calib_b_max
        self.calib_config.l2_a = self.training.calib_l2_a
        self.calib_config.l2_b = self.training.calib_l2_b
        self.calib_state = AffineLogitCalibState(
            a=clamp(self.calib_state.a, self.calib_config.a_min, self.calib_config.a_max),
            b=clamp(self.calib_state.b, self.calib_config.b_min, self.calib_config.b_max),
            n=self.calib_state.n,
        )

    def _class_weight(self, fact_dir: Direction) -> float:
        if fact_dir == Direction.FLAT:
            return 1.0
        if self.training.class_balance_strength <= 0:
            return 1.0
        share = self.fact_ema.get(fact_dir, 1.0 / 3.0)
        share = max(share, self.training.class_balance_floor)
        target = 0.5
        inv = target / share
        weight = 1.0 + self.training.class_balance_strength * (inv - 1.0)
        return clamp(weight, self.training.class_balance_min, self.training.class_balance_max)

    def _update_fact_ema(self, fact_dir: Direction) -> None:
        decay = self.training.class_balance_ema
        for direction in (Direction.UP, Direction.DOWN, Direction.FLAT):
            val = 1.0 if fact_dir == direction else 0.0
            self.fact_ema[direction] = decay * self.fact_ema[direction] + (1.0 - decay) * val

    def _context_stats(
        self, bundle: FeatureBundle, now_ts: int
    ) -> tuple[str, float, Dict[str, float]]:
        key_fine = build_context_key(self.tf, self.model_id, "fine", bundle.context_fine)
        key_coarse = build_context_key(self.tf, self.model_id, "coarse", bundle.context_coarse)
        stat_fine = self.pattern_store.get_stat(key_fine)
        stat_coarse = self.pattern_store.get_stat(key_coarse)
        trust_fine = self.pattern_store.trust(stat_fine, now_ts)
        trust_coarse = self.pattern_store.trust(stat_coarse, now_ts)
        if trust_fine >= self.pattern_store.config.fine_trust_threshold:
            stat = stat_fine
            key_used = key_fine
            trust = trust_fine
        else:
            stat = stat_coarse
            key_used = key_coarse
            trust = trust_coarse
        prior = {
            "p_up": stat.ema_p_up if stat and stat.ema_p_up is not None else 1.0 / 3.0,
            "p_down": stat.ema_p_down if stat and stat.ema_p_down is not None else 1.0 / 3.0,
            "p_flat": stat.ema_p_flat if stat and stat.ema_p_flat is not None else 1.0 / 3.0,
        }
        return key_used, trust, prior

    def _decision_stats(
        self, bundle: FeatureBundle, pred_dir: Direction, now_ts: int
    ) -> tuple[str, float, float, bool]:
        key_fine = build_decision_key(
            build_context_key(self.tf, self.model_id, "fine", bundle.context_fine),
            pred_dir,
        )
        key_coarse = build_decision_key(
            build_context_key(self.tf, self.model_id, "coarse", bundle.context_coarse),
            pred_dir,
        )
        stat_fine = self.pattern_store.get_stat(key_fine)
        stat_coarse = self.pattern_store.get_stat(key_coarse)
        trust_fine = self.pattern_store.trust(stat_fine, now_ts)
        trust_coarse = self.pattern_store.trust(stat_coarse, now_ts)
        if trust_fine >= self.pattern_store.config.fine_trust_threshold:
            stat = stat_fine
            key_used = key_fine
            trust = trust_fine
        else:
            stat = stat_coarse
            key_used = key_coarse
            trust = trust_coarse
        prior_win = stat.ema_win if stat and stat.ema_win is not None else 0.5
        anti_pattern = self.pattern_store.is_anti_pattern(stat, now_ts)
        return key_used, trust, prior_win, anti_pattern

    def _apply_context_adjustments(
        self,
        pred_dir: Direction,
        trust_ctx: float,
        prior_ctx: Dict[str, float],
        flat_margin: float,
    ) -> float:
        if not self.decision.use_context_priors:
            return flat_margin
        if trust_ctx < self.decision.context_trust_min:
            return flat_margin
        flat_boost = self.decision.context_flat_gain * trust_ctx * prior_ctx.get("p_flat", 0.0)
        flat_margin = clamp(flat_margin + flat_boost, 0.0, 0.5)
        bias = prior_ctx.get("p_up", 0.0) - prior_ctx.get("p_down", 0.0)
        if pred_dir == Direction.UP and bias > 0:
            reduce = self.decision.context_dir_gain * trust_ctx * min(1.0, abs(bias))
            flat_margin = clamp(flat_margin - reduce, 0.0, 0.5)
        if pred_dir == Direction.DOWN and bias < 0:
            reduce = self.decision.context_dir_gain * trust_ctx * min(1.0, abs(bias))
            flat_margin = clamp(flat_margin - reduce, 0.0, 0.5)
        return flat_margin

    def _adaptation(self, trust_dec: float, prior_win: float) -> tuple[float, float]:
        lr_mult = clamp(
            1.0 + self.training.lr_gain * trust_dec * (0.5 - prior_win),
            self.training.lr_min_mult,
            self.training.lr_max_mult,
        )
        window = self.metrics.snapshot()
        window_total = int(window.get("window_total", 0))
        window_acc = float(window.get("window_accuracy", 0.0))
        perf_mult = 1.0
        if window_total >= self.training.perf_lr_min_samples:
            perf_mult = clamp(
                1.0 + self.training.perf_lr_gain * (window_acc - self.training.perf_lr_baseline),
                self.training.perf_lr_min_mult,
                self.training.perf_lr_max_mult,
            )
        anchor_mult = clamp(
            1.0 + self.training.anchor_gain * trust_dec * (prior_win - 0.5),
            self.training.anchor_min_mult,
            self.training.anchor_max_mult,
        )
        lr_eff = self.lr_base * lr_mult * perf_mult
        anchor_lambda_eff = self.training.anchor_lambda_base * anchor_mult
        self.last_perf_mult = perf_mult
        return lr_eff, anchor_lambda_eff


class MultiTimeframeEngine:
    def __init__(
        self,
        tfs: List[str],
        feature_config: FeatureConfig,
        model_init: ModelInitConfig,
        training: TrainingConfig,
        decision: DecisionConfig,
        lrs: ModelLRConfig,
        pattern_store: PatternStore,
        state_store: ModelStateStore,
        runner_config: RunnerConfig | None = None,
    ) -> None:
        self.tfs = tfs
        self.pattern_store = pattern_store
        self.state_store = state_store
        self.runners: Dict[str, List[ModelRunner]] = {}
        self.buffers: Dict[str, CandleBuffer] = {}
        self.last_processed_ts: Dict[str, int] = {}
        self.feature_config = feature_config

        for tf in tfs:
            builders = {
                MODEL_TREND: FeatureBuilder(feature_config, MODEL_TREND),
                MODEL_OSC: FeatureBuilder(feature_config, MODEL_OSC),
                MODEL_VOL: FeatureBuilder(feature_config, MODEL_VOL),
            }
            max_required = max(b.spec.required_lookback for b in builders.values())
            self.buffers[tf] = CandleBuffer(max_required)
            self.runners[tf] = [
                ModelRunner(
                    tf=tf,
                    model_type=MODEL_TREND,
                    feature_builder=builders[MODEL_TREND],
                    model_init=model_init,
                    training=training,
                    decision=decision,
                    lr_base=lrs.lr_trend,
                    pattern_store=pattern_store,
                    state_store=state_store,
                    runner_config=runner_config,
                ),
                ModelRunner(
                    tf=tf,
                    model_type=MODEL_OSC,
                    feature_builder=builders[MODEL_OSC],
                    model_init=model_init,
                    training=training,
                    decision=decision,
                    lr_base=lrs.lr_osc,
                    pattern_store=pattern_store,
                    state_store=state_store,
                    runner_config=runner_config,
                ),
                ModelRunner(
                    tf=tf,
                    model_type=MODEL_VOL,
                    feature_builder=builders[MODEL_VOL],
                    model_init=model_init,
                    training=training,
                    decision=decision,
                    lr_base=lrs.lr_vol,
                    pattern_store=pattern_store,
                    state_store=state_store,
                    runner_config=runner_config,
                ),
            ]

    def warm_start(self, tf: str, candles: List[Candle]) -> None:
        buffer = self.buffers[tf]
        buffer.seed(candles)
        if candles:
            self.last_processed_ts[tf] = candles[-1].start_ts

    def bootstrap_predictions(self, tf: str, record_prediction, runtime_state) -> None:
        buffer = self.buffers.get(tf)
        if buffer is None:
            return
        candles = buffer.values()
        if not candles:
            return
        candle = candles[-1]
        for runner in self.runners.get(tf, []):
            prediction = runner.predict(candle, candles, now_ms())
            if prediction is None:
                continue
            if runtime_state is not None:
                runtime_state.update_prediction(prediction)
            if record_prediction is not None:
                record_prediction(prediction)

    def process_candle(
        self,
        candle: Candle,
        record_prediction,
        record_fact,
        record_update,
        runtime_state,
        autosave_seconds: int,
        autosave_updates: int,
    ) -> bool:
        tf = candle.tf
        buffer = self.buffers.get(tf)
        if buffer is None:
            return False
        buffer.add(candle)

        last_ts = self.last_processed_ts.get(tf)
        if last_ts is not None and candle.start_ts <= last_ts:
            return False
        self.last_processed_ts[tf] = candle.start_ts

        candles = buffer.values()
        for runner in self.runners.get(tf, []):
            result = runner.on_fact(candle, now_ms())
            if result is not None:
                fact, update = result
                if runtime_state is not None:
                    runtime_state.update_fact(fact)
                    runtime_state.update_update(update)
                    model_key = f"{tf}:{runner.model_id}"
                    metrics_payload = runner.metrics_snapshot()
                    runtime_state.update_metrics(model_key, metrics_payload)
                    runtime_state.update_model_stats(
                        model_key,
                        {
                            "tf": tf,
                            "model_id": runner.model_id,
                            "model_type": runner.model_type,
                            "weight_norms": update.weight_norms,
                            "lr_base": runner.lr_base,
                            "updates": runner.update_count,
                            "metrics": metrics_payload,
                            "calibration_affine": {
                                "a": float(runner.calib_state.a),
                                "b": float(runner.calib_state.b),
                                "n": int(runner.calib_state.n),
                            },
                            "perf_lr_mult": runner.last_perf_mult,
                            "last_class_weight": runner.last_class_weight,
                        },
                    )
                if record_fact is not None:
                    record_fact(fact)
                if record_update is not None:
                    record_update(update)
                runner.maybe_save(now_ms(), autosave_seconds, autosave_updates)

        for runner in self.runners.get(tf, []):
            prediction = runner.predict(candle, candles, now_ms())
            if prediction is None:
                continue
            if runtime_state is not None:
                runtime_state.update_prediction(prediction)
            if record_prediction is not None:
                record_prediction(prediction)

        self.pattern_store.maintenance(now_ms())
        return True

    def load_states(self) -> None:
        for tf, runners in self.runners.items():
            for runner in runners:
                state = self.state_store.load_latest(runner.model_id, tf)
                if state is None:
                    continue
                runner.load_state(
                    {
                        "params": state.params,
                        "ema_params": state.ema_params,
                        "anchor_params": state.anchor_params,
                        "opt_state": state.opt_state,
                        "metrics_state": state.metrics,
                    }
                )

    def set_runtime_flags(
        self,
        *,
        enable_training: Optional[bool] = None,
        update_patterns: Optional[bool] = None,
        enable_calibration_update: Optional[bool] = None,
        enable_anchor: Optional[bool] = None,
        use_ema: Optional[bool] = None,
        use_patterns: Optional[bool] = None,
    ) -> None:
        for runners in self.runners.values():
            for runner in runners:
                runner.set_runtime_flags(
                    enable_training=enable_training,
                    update_patterns=update_patterns,
                    enable_calibration_update=enable_calibration_update,
                    enable_anchor=enable_anchor,
                    use_ema=use_ema,
                    use_patterns=use_patterns,
                )


def decide_direction(p_up: float, p_down: float, flat_margin: float) -> Direction:
    if abs(p_up - 0.5) < flat_margin:
        return Direction.FLAT
    return Direction.UP if p_up > p_down else Direction.DOWN


def build_context_key(tf: str, model_id: str, level: str, bins: Dict[str, str]) -> str:
    parts = [f"{k}={bins[k]}" for k in sorted(bins.keys())]
    return f"{tf}:{model_id}:{level}:" + "|".join(parts)


def build_decision_key(context_key: str, pred_dir: Direction) -> str:
    return f"{context_key}|PRED={pred_dir.value}"


def fact_from_prices(
    tf: str,
    prev_ts: int,
    curr_ts: int,
    close_prev: float,
    close_curr: float,
    flat_bps: float,
) -> Fact:
    if close_prev <= 0:
        ret_bps = 0.0
    else:
        ret_bps = ((close_curr - close_prev) / close_prev) * 10_000.0
    if abs(ret_bps) <= flat_bps:
        direction = Direction.FLAT
    else:
        direction = Direction.UP if close_curr > close_prev else Direction.DOWN
    return Fact(
        tf=tf,
        prev_ts=prev_ts,
        curr_ts=curr_ts,
        close_prev=close_prev,
        close_curr=close_curr,
        ret_bps=ret_bps,
        direction=direction,
    )


def reward_from_dirs(
    pred_dir: Direction, fact_dir: Direction, training: TrainingConfig
) -> tuple[float, str, float, float]:
    if pred_dir == Direction.FLAT:
        if fact_dir == Direction.FLAT:
            base = training.reward_flat_correct
            reason = "flat_correct"
        else:
            base = training.reward_flat_wrong
            reason = "flat_wrong"
        reward = base + training.flat_penalty
        return reward, reason, base, training.flat_penalty
    if pred_dir == fact_dir:
        base = training.reward_dir_correct
        reason = "dir_correct"
    else:
        base = training.reward_dir_wrong
        reason = "dir_wrong"
    return base, reason, base, 0.0


def target_vector(direction: Direction) -> np.ndarray:
    if direction == Direction.UP:
        return np.array([1.0, 0.0], dtype=np.float32)
    if direction == Direction.DOWN:
        return np.array([0.0, 1.0], dtype=np.float32)
    return np.array([0.5, 0.5], dtype=np.float32)


def _to_numpy(params: Dict[str, jnp.ndarray]) -> Dict[str, np.ndarray]:
    return {k: np.asarray(jax.device_get(v)) for k, v in params.items()}


def _to_jax(params: Dict[str, np.ndarray]) -> Dict[str, jnp.ndarray]:
    return {k: jnp.asarray(v, dtype=jnp.float32) for k, v in params.items()}
