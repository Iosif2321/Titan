from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict

from aiohttp import web

from ..config_manager import ConfigManager
from ..pattern_store import PatternStore
from ..runtime_state import RuntimeState
from ..utils import now_ms


def create_app(state: RuntimeState, config_manager: ConfigManager, pattern_store: PatternStore) -> web.Application:
    app = web.Application()
    app["state"] = state
    app["config_manager"] = config_manager
    app["pattern_store"] = pattern_store

    async def health(_: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def get_state(request: web.Request) -> web.Response:
        return web.json_response(request.app["state"].snapshot())

    async def get_models(request: web.Request) -> web.Response:
        return web.json_response(request.app["state"].models_snapshot())

    async def get_patterns(request: web.Request) -> web.Response:
        tf = request.query.get("tf")
        model_id = request.query.get("model_id")
        limit = int(request.query.get("limit", "10"))
        store: PatternStore = request.app["pattern_store"]
        if tf and model_id:
            payload = store.top_patterns(tf, model_id, limit=limit)
            return web.json_response({"tf": tf, "model_id": model_id, **payload})

        results: Dict[str, Any] = {}
        for model in request.app["state"].models_snapshot().get("models", []):
            tf_val = model.get("tf")
            model_val = model.get("model_id")
            if not tf_val or not model_val:
                continue
            key = f"{tf_val}:{model_val}"
            results[key] = store.top_patterns(tf_val, model_val, limit=limit)
        return web.json_response(results)

    async def get_patterns_summary(request: web.Request) -> web.Response:
        tf = request.query.get("tf")
        model_id = request.query.get("model_id")
        limit = int(request.query.get("limit", "5"))
        window_seconds = int(request.query.get("window_seconds", "3600"))
        store: PatternStore = request.app["pattern_store"]
        now_ts = now_ms()
        since_ts = now_ts - window_seconds * 1000
        if tf and model_id:
            payload = {
                "tf": tf,
                "model_id": model_id,
                "summary": store.stats_summary(tf, model_id),
                "usage": store.usage_summary(tf, model_id, since_ts),
                "top": store.top_patterns(tf, model_id, limit=limit),
            }
            return web.json_response(payload)

        results: Dict[str, Any] = {}
        for model in request.app["state"].models_snapshot().get("models", []):
            tf_val = model.get("tf")
            model_val = model.get("model_id")
            if not tf_val or not model_val:
                continue
            key = f"{tf_val}:{model_val}"
            results[key] = {
                "summary": store.stats_summary(tf_val, model_val),
                "usage": store.usage_summary(tf_val, model_val, since_ts),
                "top": store.top_patterns(tf_val, model_val, limit=limit),
            }
        return web.json_response(
            {"window_seconds": window_seconds, "ts": now_ts, "models": results}
        )

    async def post_config(request: web.Request) -> web.Response:
        payload = await request.json()
        result = await request.app["config_manager"].apply_updates(payload)
        return web.json_response(result)

    async def stream(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await resp.prepare(request)
        state = request.app["state"]

        try:
            while True:
                try:
                    event = await asyncio.wait_for(state.next_event(), timeout=10.0)
                    payload = json.dumps(event, separators=(",", ":"))
                    await resp.write(f"data: {payload}\n\n".encode("utf-8"))
                except asyncio.TimeoutError:
                    await resp.write(b": keepalive\n\n")
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        return resp

    async def index(_: web.Request) -> web.Response:
        root = Path(__file__).resolve().parent
        return web.FileResponse(root / "static" / "index.html")

    async def debug(_: web.Request) -> web.Response:
        root = Path(__file__).resolve().parent
        return web.FileResponse(root / "static" / "debug_index.html")

    async def debug_ui(_: web.Request) -> web.Response:
        root = Path(__file__).resolve().parent
        return web.FileResponse(root / "static" / "debug.html")

    app.router.add_get("/", index)
    app.router.add_get("/debug", debug)
    app.router.add_get("/debug/", debug)
    app.router.add_get("/debug/ui", debug_ui)
    app.router.add_get("/api/health", health)
    app.router.add_get("/api/state", get_state)
    app.router.add_get("/api/models", get_models)
    app.router.add_get("/api/patterns/top", get_patterns)
    app.router.add_get("/api/patterns/summary", get_patterns_summary)
    app.router.add_post("/api/config", post_config)
    app.router.add_get("/api/stream", stream)
    app.router.add_static("/static/", Path(__file__).resolve().parent / "static")
    return app


async def start_dashboard(
    state: RuntimeState,
    config_manager: ConfigManager,
    pattern_store: PatternStore,
    host: str,
    port: int,
) -> web.AppRunner:
    app = create_app(state, config_manager, pattern_store)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()
    return runner
