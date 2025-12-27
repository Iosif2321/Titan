from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Deque, Dict, Optional

from .calibration import (
    AffineLogitCalibConfig,
    AffineLogitCalibState,
    update_affine_calib,
)

if TYPE_CHECKING:
    from .config import PerModelCalibConfig, TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveCalibController:
    """
    Adaptive calibration controller that combines affine logit calibration
    with ECE-driven learning rate adaptation.

    The controller tracks recent ECE (Expected Calibration Error) and
    adaptively adjusts the learning rate to improve calibration quality:
    - High ECE (>0.25): Increase lr by 1.5x for stronger corrections
    - Low ECE (<0.10): Decrease lr by 0.9x for stability
    - Moderate ECE: Check trend and adjust gently
    """

    # Identity
    model_type: str

    # Core calibration state and config
    state: AffineLogitCalibState
    config: AffineLogitCalibConfig

    # Adaptive learning rate
    lr_current: float
    lr_base: float
    lr_min: float = 0.0001
    lr_max: float = 0.02

    # ECE tracking
    ece_window: Deque[float] = field(default_factory=lambda: deque(maxlen=50))

    # Adaptation parameters
    ece_target: float = 0.05
    ece_good_threshold: float = 0.10
    ece_bad_threshold: float = 0.25
    lr_increase_factor: float = 1.5
    lr_decrease_factor: float = 0.9
    adaptation_interval: int = 20
    min_samples_for_adaptation: int = 30

    # Tracking
    update_count: int = 0
    last_adaptation_step: int = 0
    ece_improving_streak: int = 0
    ece_worsening_streak: int = 0

    @classmethod
    def create(
        cls,
        model_type: str,
        training_config: TrainingConfig,
        per_model_config: Optional[PerModelCalibConfig] = None,
    ) -> AdaptiveCalibController:
        """
        Factory method to create AdaptiveCalibController with config merging.

        Args:
            model_type: Model identifier (e.g., "TRENDVIC", "OSCILLATOR")
            training_config: Global training configuration
            per_model_config: Optional per-model overrides

        Returns:
            Configured AdaptiveCalibController instance
        """
        # Start with defaults from training_config
        lr = training_config.calib_lr
        a_min = training_config.calib_a_min
        a_max = training_config.calib_a_max
        b_min = training_config.calib_b_min
        b_max = training_config.calib_b_max
        l2_a = training_config.calib_l2_a
        l2_b = training_config.calib_l2_b

        lr_min = training_config.calib_lr_min
        lr_max = training_config.calib_lr_max
        ece_target = training_config.calib_ece_target
        ece_good_threshold = training_config.calib_ece_good_threshold
        ece_bad_threshold = training_config.calib_ece_bad_threshold
        lr_increase_factor = training_config.calib_lr_increase_factor
        lr_decrease_factor = training_config.calib_lr_decrease_factor
        adaptation_interval = training_config.calib_adaptation_interval
        min_samples = training_config.calib_min_samples_for_adaptation
        ece_window_size = training_config.calib_ece_window_size

        init_a = 1.0
        init_b = 0.0

        # Apply per-model overrides if provided
        if per_model_config is not None:
            if per_model_config.lr is not None:
                lr = per_model_config.lr
            if per_model_config.a_min is not None:
                a_min = per_model_config.a_min
            if per_model_config.a_max is not None:
                a_max = per_model_config.a_max
            if per_model_config.b_min is not None:
                b_min = per_model_config.b_min
            if per_model_config.b_max is not None:
                b_max = per_model_config.b_max
            if per_model_config.l2_a is not None:
                l2_a = per_model_config.l2_a
            if per_model_config.l2_b is not None:
                l2_b = per_model_config.l2_b
            if per_model_config.lr_min is not None:
                lr_min = per_model_config.lr_min
            if per_model_config.lr_max is not None:
                lr_max = per_model_config.lr_max
            if per_model_config.ece_target is not None:
                ece_target = per_model_config.ece_target
            if per_model_config.ece_good_threshold is not None:
                ece_good_threshold = per_model_config.ece_good_threshold
            if per_model_config.ece_bad_threshold is not None:
                ece_bad_threshold = per_model_config.ece_bad_threshold
            if per_model_config.lr_increase_factor is not None:
                lr_increase_factor = per_model_config.lr_increase_factor
            if per_model_config.lr_decrease_factor is not None:
                lr_decrease_factor = per_model_config.lr_decrease_factor
            if per_model_config.adaptation_interval is not None:
                adaptation_interval = per_model_config.adaptation_interval
            if per_model_config.init_a is not None:
                init_a = per_model_config.init_a
            if per_model_config.init_b is not None:
                init_b = per_model_config.init_b

        # Create state and config
        state = AffineLogitCalibState(a=init_a, b=init_b, n=0)
        config = AffineLogitCalibConfig(
            lr=lr,
            a_min=a_min,
            a_max=a_max,
            b_min=b_min,
            b_max=b_max,
            l2_a=l2_a,
            l2_b=l2_b,
            a_anchor=init_a,
            b_anchor=init_b,
        )

        return cls(
            model_type=model_type,
            state=state,
            config=config,
            lr_current=lr,
            lr_base=lr,
            lr_min=lr_min,
            lr_max=lr_max,
            ece_window=deque(maxlen=ece_window_size),
            ece_target=ece_target,
            ece_good_threshold=ece_good_threshold,
            ece_bad_threshold=ece_bad_threshold,
            lr_increase_factor=lr_increase_factor,
            lr_decrease_factor=lr_decrease_factor,
            adaptation_interval=adaptation_interval,
            min_samples_for_adaptation=min_samples,
        )

    def update(
        self,
        logit_up: float,
        logit_down: float,
        y_up: float,
        weight: float,
        flip: bool,
        current_ece: Optional[float],
    ) -> None:
        """
        Update calibration parameters via SGD and adapt learning rate.

        Args:
            logit_up: Logit for UP direction
            logit_down: Logit for DOWN direction
            y_up: Target for UP (1.0=UP, 0.0=DOWN, 0.5=FLAT)
            weight: Sample weight
            flip: Whether to flip directions
            current_ece: Recent ECE from sliding window (None if insufficient data)
        """
        # Track ECE for adaptation
        if current_ece is not None:
            self.ece_window.append(current_ece)

        # Update config lr before SGD step
        self.config.lr = self.lr_current

        # Perform SGD update
        self.state = update_affine_calib(
            st=self.state,
            logit_up=logit_up,
            logit_down=logit_down,
            y_up=y_up,
            weight=weight,
            cfg=self.config,
            flip=flip,
        )

        self.update_count += 1

        # Maybe adapt learning rate
        self._maybe_adapt_lr()

    def _maybe_adapt_lr(self) -> None:
        """
        Adapt learning rate based on recent ECE.

        Algorithm:
        - If ECE > bad_threshold: increase lr (stronger corrections needed)
        - If ECE < good_threshold: decrease lr (calibration good, stabilize)
        - If moderate: check trend and adjust gently
        - Use streak tracking to avoid over-reaction
        """
        # Check if adaptation is due
        if self.update_count - self.last_adaptation_step < self.adaptation_interval:
            return

        # Ensure sufficient samples
        if len(self.ece_window) < self.min_samples_for_adaptation:
            return

        # Calculate recent ECE statistics
        ece_values = list(self.ece_window)
        current_ece = sum(ece_values) / len(ece_values)

        # Calculate trend (recent vs older samples)
        ece_trend = 0.0
        if len(ece_values) >= 20:
            recent_half = ece_values[len(ece_values) // 2 :]
            older_half = ece_values[: len(ece_values) // 2]
            ece_trend = sum(recent_half) / len(recent_half) - sum(older_half) / len(older_half)

        # Determine adaptation action
        action = "hold"
        factor = 1.0

        if current_ece > self.ece_bad_threshold:
            # HIGH ECE: Increase lr for stronger corrections
            action = "increase"
            factor = self.lr_increase_factor
            self.ece_worsening_streak += 1
            self.ece_improving_streak = 0

        elif current_ece < self.ece_good_threshold:
            # LOW ECE: Decrease lr for stability
            action = "decrease"
            factor = self.lr_decrease_factor
            self.ece_improving_streak += 1
            self.ece_worsening_streak = 0

        else:
            # MODERATE ECE: Check trend
            if ece_trend > 0.05:  # ECE worsening
                action = "increase"
                factor = self.lr_increase_factor**0.5  # Gentler
            elif ece_trend < -0.05:  # ECE improving
                action = "decrease"
                factor = self.lr_decrease_factor**0.5  # Gentler

        # Apply adaptation with streak-based modulation
        if action != "hold":
            streak_bonus = 1.0
            if self.ece_worsening_streak > 3:
                # Persistent high ECE: boost lr increase
                streak_bonus = 1.2
            elif self.ece_improving_streak > 5:
                # Sustained improvement: more aggressive lr decrease
                streak_bonus = 1.1

            old_lr = self.lr_current
            self.lr_current *= factor**streak_bonus
            self.lr_current = max(self.lr_min, min(self.lr_max, self.lr_current))

            logger.debug(
                "ADAPT_LR model=%s action=%s ece=%.4f trend=%.4f "
                "streak_w=%d streak_i=%d lr: %.6f -> %.6f",
                self.model_type,
                action,
                current_ece,
                ece_trend,
                self.ece_worsening_streak,
                self.ece_improving_streak,
                old_lr,
                self.lr_current,
            )

            # Warn if lr stuck at bounds
            if self.lr_current >= self.lr_max - 1e-6:
                logger.warning(
                    "LR_AT_MAX model=%s lr=%.6f ece=%.4f (may need feature/architecture review)",
                    self.model_type,
                    self.lr_current,
                    current_ece,
                )
            elif self.lr_current <= self.lr_min + 1e-6:
                logger.warning(
                    "LR_AT_MIN model=%s lr=%.6f ece=%.4f",
                    self.model_type,
                    self.lr_current,
                    current_ece,
                )

        # Update tracking
        self.last_adaptation_step = self.update_count

    def to_dict(self) -> Dict[str, object]:
        """Serialize controller state for persistence."""
        return {
            "model_type": self.model_type,
            "state": {
                "a": float(self.state.a),
                "b": float(self.state.b),
                "n": int(self.state.n),
            },
            "config": {
                "lr": float(self.config.lr),
                "a_min": float(self.config.a_min),
                "a_max": float(self.config.a_max),
                "b_min": float(self.config.b_min),
                "b_max": float(self.config.b_max),
                "l2_a": float(self.config.l2_a),
                "l2_b": float(self.config.l2_b),
                "a_anchor": float(self.config.a_anchor),
                "b_anchor": float(self.config.b_anchor),
            },
            "adaptive": {
                "lr_current": float(self.lr_current),
                "lr_base": float(self.lr_base),
                "lr_min": float(self.lr_min),
                "lr_max": float(self.lr_max),
                "ece_window": [float(x) for x in self.ece_window],
                "ece_target": float(self.ece_target),
                "ece_good_threshold": float(self.ece_good_threshold),
                "ece_bad_threshold": float(self.ece_bad_threshold),
                "lr_increase_factor": float(self.lr_increase_factor),
                "lr_decrease_factor": float(self.lr_decrease_factor),
                "adaptation_interval": int(self.adaptation_interval),
                "min_samples_for_adaptation": int(self.min_samples_for_adaptation),
                "update_count": int(self.update_count),
                "last_adaptation_step": int(self.last_adaptation_step),
                "ece_improving_streak": int(self.ece_improving_streak),
                "ece_worsening_streak": int(self.ece_worsening_streak),
            },
        }

    def load_state(self, data: Dict[str, object]) -> None:
        """Load controller state from serialized dict."""
        if "state" in data and isinstance(data["state"], dict):
            state_dict = data["state"]
            self.state = AffineLogitCalibState(
                a=float(state_dict.get("a", 1.0)),
                b=float(state_dict.get("b", 0.0)),
                n=int(state_dict.get("n", 0)),
            )

        if "config" in data and isinstance(data["config"], dict):
            cfg_dict = data["config"]
            self.config = AffineLogitCalibConfig(
                lr=float(cfg_dict.get("lr", 0.005)),
                a_min=float(cfg_dict.get("a_min", 0.30)),
                a_max=float(cfg_dict.get("a_max", 2.0)),
                b_min=float(cfg_dict.get("b_min", -1.0)),
                b_max=float(cfg_dict.get("b_max", 1.0)),
                l2_a=float(cfg_dict.get("l2_a", 0.01)),
                l2_b=float(cfg_dict.get("l2_b", 0.001)),
                a_anchor=float(cfg_dict.get("a_anchor", 1.0)),
                b_anchor=float(cfg_dict.get("b_anchor", 0.0)),
            )

        if "adaptive" in data and isinstance(data["adaptive"], dict):
            adp_dict = data["adaptive"]
            self.lr_current = float(adp_dict.get("lr_current", self.lr_base))
            self.lr_base = float(adp_dict.get("lr_base", self.lr_current))
            self.lr_min = float(adp_dict.get("lr_min", 0.0001))
            self.lr_max = float(adp_dict.get("lr_max", 0.02))

            if "ece_window" in adp_dict and isinstance(adp_dict["ece_window"], list):
                self.ece_window = deque(
                    [float(x) for x in adp_dict["ece_window"]], maxlen=self.ece_window.maxlen
                )

            self.ece_target = float(adp_dict.get("ece_target", 0.05))
            self.ece_good_threshold = float(adp_dict.get("ece_good_threshold", 0.10))
            self.ece_bad_threshold = float(adp_dict.get("ece_bad_threshold", 0.25))
            self.lr_increase_factor = float(adp_dict.get("lr_increase_factor", 1.5))
            self.lr_decrease_factor = float(adp_dict.get("lr_decrease_factor", 0.9))
            self.adaptation_interval = int(adp_dict.get("adaptation_interval", 20))
            self.min_samples_for_adaptation = int(adp_dict.get("min_samples_for_adaptation", 30))
            self.update_count = int(adp_dict.get("update_count", 0))
            self.last_adaptation_step = int(adp_dict.get("last_adaptation_step", 0))
            self.ece_improving_streak = int(adp_dict.get("ece_improving_streak", 0))
            self.ece_worsening_streak = int(adp_dict.get("ece_worsening_streak", 0))

    def load_legacy_state(self, legacy_data: Dict[str, object]) -> None:
        """
        Load state from legacy calibration_affine format for backward compatibility.

        Args:
            legacy_data: Dict with keys 'a', 'b', 'n' from old format
        """
        self.state = AffineLogitCalibState(
            a=float(legacy_data.get("a", 1.0)),
            b=float(legacy_data.get("b", 0.0)),
            n=int(legacy_data.get("n", 0)),
        )
        # Keep current config and adaptive params unchanged
        logger.info(
            "LEGACY_CALIB_LOAD model=%s a=%.4f b=%.4f n=%d",
            self.model_type,
            self.state.a,
            self.state.b,
            self.state.n,
        )
