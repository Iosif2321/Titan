from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import List, Tuple


def ensure_jax_backend() -> Tuple[str, List[str]]:
    try:
        import jax

        backend = jax.default_backend()
        devices = [str(device) for device in jax.devices()]
        return backend, devices
    except Exception as exc:
        msg = str(exc)
        if "jaxlib version" in msg and "incompatible" in msg:
            logging.error("JAX/JAXLIB mismatch: %s", msg)
            raise SystemExit(
                "JAX/JAXLIB mismatch. Fix with:\n"
                "  python -m pip uninstall -y jax jaxlib\n"
                "  python -m pip install \"jax==0.4.21\" \"jaxlib==0.4.21\"\n"
                "On Windows, GPU is not supported; use WSL2/Docker for CUDA."
            )
        forced = (os.environ.get("JAX_PLATFORMS") or os.environ.get("JAX_PLATFORM_NAME") or "").lower()
        if os.environ.get("TITAN_JAX_FALLBACK") == "1" or "cpu" in forced or not forced:
            raise
        logging.warning("JAX backend init failed (%s). Falling back to CPU.", exc)
        env = os.environ.copy()
        env["JAX_PLATFORMS"] = "cpu"
        env["TITAN_JAX_FALLBACK"] = "1"
        entrypoint = env.get("TITAN_ENTRYPOINT")
        if entrypoint:
            args = [sys.executable, "-m", entrypoint] + sys.argv[1:]
        else:
            args = [sys.executable] + sys.argv
        if sys.platform == "win32":
            code = subprocess.call(args, env=env)
            sys.exit(code)
        os.execve(sys.executable, args, env)
        raise
