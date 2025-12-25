from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .config import AppConfig, PersistenceConfig, config_to_dict
from .metrics import MetricsTracker
from .optimizer import AdamState
from .model import Params


def config_hash(config: AppConfig) -> str:
    payload = json.dumps(config_to_dict(config), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _flatten_params(prefix: str, params: Params) -> Dict[str, np.ndarray]:
    return {f"{prefix}{k}": np.asarray(v) for k, v in params.items()}


def _unflatten_params(prefix: str, data: dict[str, np.ndarray]) -> Params:
    out: Params = {}
    for key, value in data.items():
        if key.startswith(prefix):
            out[key[len(prefix) :]] = value
    return out


@dataclass
class CheckpointData:
    params: Params
    opt_state: AdamState
    metrics_state: dict[str, object]
    config_hash: str
    last_ts: Optional[int]
    update_step: int


def save_checkpoint(
    state_dir: Path,
    params: Params,
    opt_state: AdamState,
    metrics: MetricsTracker,
    config: AppConfig,
    last_ts: Optional[int],
    update_step: int,
    snapshot_path: Optional[Path] = None,
) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = state_dir / ".checkpoint_tmp.npz"
    final_path = state_dir / "checkpoint_latest.npz"

    meta = {
        "config_hash": config_hash(config),
        "last_ts": last_ts,
        "update_step": update_step,
        "metrics_state": metrics.state_dict(),
        "opt_t": opt_state.t,
    }

    payload: Dict[str, np.ndarray] = {}
    payload.update(_flatten_params("params_", params))
    payload.update(_flatten_params("opt_m_", opt_state.m))
    payload.update(_flatten_params("opt_v_", opt_state.v))
    payload["meta"] = np.asarray(json.dumps(meta))

    np.savez(tmp_path, **payload)
    os.replace(tmp_path, final_path)

    if snapshot_path is not None:
        shutil.copy2(final_path, snapshot_path)


def load_checkpoint(state_dir: Path) -> Optional[CheckpointData]:
    path = state_dir / "checkpoint_latest.npz"
    if not path.exists():
        return None

    data = np.load(path, allow_pickle=False)
    meta_raw = data["meta"].item()
    meta = json.loads(meta_raw)

    params = _unflatten_params("params_", data)
    opt_m = _unflatten_params("opt_m_", data)
    opt_v = _unflatten_params("opt_v_", data)
    opt_state = AdamState(m=opt_m, v=opt_v, t=int(meta.get("opt_t", 0)))
    return CheckpointData(
        params=params,
        opt_state=opt_state,
        metrics_state=meta.get("metrics_state", {}),
        config_hash=meta.get("config_hash", ""),
        last_ts=meta.get("last_ts", None),
        update_step=int(meta.get("update_step", 0)),
    )


class CheckpointManager:
    def __init__(self, config: AppConfig, persistence: PersistenceConfig) -> None:
        self.config = config
        self.persistence = persistence
        self.last_save_ts = 0
        self.last_snapshot_ts = 0

    def maybe_save(
        self,
        params: Params,
        opt_state: AdamState,
        metrics: MetricsTracker,
        last_ts: Optional[int],
        update_step: int,
        now_ts: int,
    ) -> None:
        should_save = False
        if self.persistence.autosave_steps > 0 and update_step % self.persistence.autosave_steps == 0:
            should_save = True
        if self.persistence.autosave_seconds > 0 and now_ts - self.last_save_ts >= self.persistence.autosave_seconds * 1000:
            should_save = True

        if not should_save:
            return

        snapshot_path = None
        if self.persistence.snapshot_steps > 0 and update_step % self.persistence.snapshot_steps == 0:
            snapshot_path = self._snapshot_path(now_ts)
        if (
            self.persistence.snapshot_seconds > 0
            and now_ts - self.last_snapshot_ts >= self.persistence.snapshot_seconds * 1000
        ):
            snapshot_path = self._snapshot_path(now_ts)

        save_checkpoint(
            self.persistence.state_dir,
            params,
            opt_state,
            metrics,
            self.config,
            last_ts,
            update_step,
            snapshot_path=snapshot_path,
        )
        self.last_save_ts = now_ts
        if snapshot_path is not None:
            self.last_snapshot_ts = now_ts

    def _snapshot_path(self, now_ts: int) -> Path:
        return self.persistence.state_dir / f"checkpoint_{now_ts}.npz"
