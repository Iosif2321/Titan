#!/usr/bin/env python3
"""
One-shot Bybit V5 Spot WS probe.
Connects once -> subscribes -> prints raw JSON messages -> exits.

Default topics:
  - tickers.BTCUSDT
  - publicTrade.BTCUSDT
  - orderbook.1.BTCUSDT
"""

import asyncio
import json
import os
import sys
import time
from typing import Any, Set, List

import websockets

WS_URL_DEFAULT = "wss://stream.bybit.com/v5/public/spot"


def now_ms() -> int:
    return int(time.time() * 1000)


def pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


async def run_once(symbol: str = "BTCUSDT", timeout_sec: float = 15.0) -> None:
    ws_url = os.getenv("BYBIT_WS_URL", WS_URL_DEFAULT)

    topics: List[str] = [
        f"tickers.{symbol}",
        f"publicTrade.{symbol}",
        f"orderbook.1.{symbol}",
    ]

    wanted = set(topics)
    got: Set[str] = set()

    print(f"WS URL: {ws_url}")
    print(f"Subscribing topics: {topics}")
    print("Tip: you can change WS url via env BYBIT_WS_URL (e.g. testnet).")

    # Bybit recommends custom ping packets примерно раз в 20 секунд,
    # но этот скрипт живёт недолго. Поэтому ping отправим 1 раз, чтобы увидеть формат pong.
    async with websockets.connect(
        ws_url,
        ping_interval=None,   # отключаем встроенный ws ping, используем Bybit-формат {"op":"ping"}
        close_timeout=2,
    ) as ws:
        req_id = f"once-{now_ms()}"
        sub_msg = {"req_id": req_id, "op": "subscribe", "args": topics}
        await ws.send(json.dumps(sub_msg))

        # Однократный ping — чисто чтобы ты увидел pong-ответ
        await ws.send(json.dumps({"req_id": f"ping-{now_ms()}", "op": "ping"}))

        deadline = time.time() + timeout_sec
        msg_count = 0

        while time.time() < deadline and got != wanted:
            remaining = max(0.1, deadline - time.time())
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                break

            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", errors="replace")

            msg_count += 1
            print(f"\n================= message #{msg_count} (raw) =================")
            print(raw)

            # Для удобства — ещё и распарсим/красиво выведем, если это JSON
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            print("----------------- parsed -----------------")
            print(pretty(msg))

            topic = msg.get("topic")
            if topic in wanted and "data" in msg:
                got.add(topic)
                print(f"✅ got data for {topic} ({len(got)}/{len(wanted)})")

        print("\n================= done =================")
        if got:
            print("Received at least one data message for:")
            for t in sorted(got):
                print(f" - {t}")
        else:
            print("No data messages received (timeout/network).")


def main() -> None:
    symbol = "BTCUSDT"
    timeout_sec = 15.0

    if len(sys.argv) >= 2:
        symbol = sys.argv[1].upper()
    if len(sys.argv) >= 3:
        timeout_sec = float(sys.argv[2])

    asyncio.run(run_once(symbol=symbol, timeout_sec=timeout_sec))


if __name__ == "__main__":
    main()
