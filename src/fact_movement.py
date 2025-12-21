# src/fact_movement.py
import argparse
import asyncio
import json
import logging
import time
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, List, Optional

from .bybit_ws import stream_candles
from .config import DataConfig
from .recording import JsonlWriter
from .types import Candle, Direction

BYBIT_REST_KLINE_URL = "https://api.bybit.com/v5/market/kline"


@dataclass(frozen=True)
class FactMove:
    prev_start_ts: int
    curr_start_ts: int
    close_prev: float
    close_curr: float
    delta: float
    ret_bps: float
    direction: Direction


def _ts_iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()


def _end_ts_from_interval(start_ts: int, interval: str) -> int:
    # Для нашего use-case TF=1m (и любых числовых минутных интервалов)
    try:
        minutes = int(interval)
    except ValueError:
        minutes = 1
    return start_ts + minutes * 60_000 - 1


def _direction_from_close(
    close_prev: float, close_curr: float, flat_bps: float
) -> Direction:
    if close_prev <= 0:
        return Direction.FLAT
    ret_bps = ((close_curr - close_prev) / close_prev) * 10_000.0
    if abs(ret_bps) <= flat_bps:
        return Direction.FLAT
    return Direction.UP if close_curr > close_prev else Direction.DOWN


def _make_fact(prev: Candle, curr: Candle, flat_bps: float) -> FactMove:
    delta = curr.close - prev.close
    ret_bps = (delta / prev.close) * 10_000.0 if prev.close > 0 else 0.0
    direction = _direction_from_close(prev.close, curr.close, flat_bps)
    return FactMove(
        prev_start_ts=prev.start_ts,
        curr_start_ts=curr.start_ts,
        close_prev=prev.close,
        close_curr=curr.close,
        delta=delta,
        ret_bps=ret_bps,
        direction=direction,
    )


def fetch_recent_candles_rest(
    symbol: str, interval: str, limit: int, timeout_s: float = 10.0
) -> List[Candle]:
    """
    Public REST Bybit: GET /v5/market/kline?category=spot&symbol=...&interval=...&limit=...
    result.list: массив строковых массивов, отсортирован в обратном порядке по startTime. :contentReference[oaicite:2]{index=2}
    """
    qs = urllib.parse.urlencode(
        {"category": "spot", "symbol": symbol, "interval": interval, "limit": str(limit)}
    )
    url = f"{BYBIT_REST_KLINE_URL}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": "Titan-FactMovement/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    if payload.get("retCode") != 0:
        raise RuntimeError(
            f"Bybit REST error: retCode={payload.get('retCode')} retMsg={payload.get('retMsg')}"
        )

    lst = payload.get("result", {}).get("list", [])
    candles: List[Candle] = []

    # Переворачиваем, чтобы получить возрастающий порядок по времени
    for item in reversed(lst):
        if not isinstance(item, list) or len(item) < 6:
            continue
        start_ts = int(item[0])
        candles.append(
            Candle(
                start_ts=start_ts,
                end_ts=_end_ts_from_interval(start_ts, interval),
                open=float(item[1]),
                high=float(item[2]),
                low=float(item[3]),
                close=float(item[4]),
                volume=float(item[5]),
                confirmed=True,
            )
        )

    # Dedup по start_ts на всякий случай
    dedup: List[Candle] = []
    last_ts: Optional[int] = None
    for c in candles:
        if last_ts is not None and c.start_ts == last_ts:
            dedup[-1] = c
        else:
            dedup.append(c)
            last_ts = c.start_ts

    # Убираем текущую минуту (open candle), чтобы не ловить "колебания"
    try:
        interval_ms = int(interval) * 60_000
    except ValueError:
        interval_ms = 60_000
    now_ms = int(time.time() * 1000)
    current_start = (now_ms // interval_ms) * interval_ms

    return [c for c in dedup if c.start_ts < current_start]


def _print_summary(facts: Deque[FactMove], window: int) -> None:
    last = list(facts)[-window:]
    pattern = " ".join(f.direction.value for f in last)
    up = sum(1 for f in last if f.direction == Direction.UP)
    down = sum(1 for f in last if f.direction == Direction.DOWN)
    flat = sum(1 for f in last if f.direction == Direction.FLAT)
    net_bps = sum(f.ret_bps for f in last)
    logging.info(
        "FACT asof=%s last_%d=[%s] up=%d down=%d flat=%d net_bps=%.2f",
        _ts_iso(last[-1].curr_start_ts),
        window,
        pattern,
        up,
        down,
        flat,
        net_bps,
    )


def _print_fact_line(f: FactMove) -> None:
    # ✅ тут добавили цены (close_prev/close_curr)
    logging.info(
        "  %s vs %s dir=%s close_prev=%.2f close_curr=%.2f delta=%.2f ret_bps=%.2f",
        _ts_iso(f.prev_start_ts),
        _ts_iso(f.curr_start_ts),
        f.direction.value,
        f.close_prev,
        f.close_curr,
        f.delta,
        f.ret_bps,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    d = DataConfig()
    p = argparse.ArgumentParser(
        description="FACT movement (close-to-close), Bybit Spot kline 1m."
    )
    p.add_argument("--symbol", default=d.symbol)
    p.add_argument("--interval", default=d.interval)
    p.add_argument("--ws-url", default=d.ws_url)
    p.add_argument("--reconnect-delay", type=float, default=d.reconnect_delay)
    p.add_argument("--ping-interval", type=float, default=d.ping_interval)
    p.add_argument("--log-level", default="INFO")

    p.add_argument("--window", type=int, default=5)
    p.add_argument("--flat-bps", type=float, default=0.0)

    p.add_argument("--history-limit", type=int, default=50, help="REST warmup candles (startup).")
    p.add_argument("--no-history", action="store_true", help="Disable REST warmup (will need time to fill).")

    p.add_argument("--print-details", action="store_true")
    p.add_argument(
        "--details-full-window",
        action="store_true",
        help="Print all last-window details every update (default: only NEW line after initial print).",
    )

    p.add_argument("--save-jsonl-dir", default=None, help="Write facts.jsonl into this dir.")
    return p


async def run_live(args: argparse.Namespace) -> None:
    data_cfg = DataConfig(
        symbol=args.symbol,
        interval=args.interval,
        ws_url=args.ws_url,
        use_confirmed_only=True,  # WS: только закрытые свечи (confirm=true) :contentReference[oaicite:3]{index=3}
        reconnect_delay=args.reconnect_delay,
        ping_interval=args.ping_interval,
    )

    window = max(1, args.window)
    facts: Deque[FactMove] = deque(maxlen=window)
    last_candle: Optional[Candle] = None

    writer: Optional[JsonlWriter] = None
    if args.save_jsonl_dir:
        out_dir = Path(args.save_jsonl_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        writer = JsonlWriter(out_dir / "facts.jsonl")

    printed_initial = False

    # 1) REST warmup: чтобы сразу иметь последние 5 направлений
    if not args.no_history:
        try:
            hist = fetch_recent_candles_rest(
                args.symbol, args.interval, limit=max(args.history_limit, window + 2)
            )
            if len(hist) >= window + 1:
                warm = hist[-(window + 1):]
                last_candle = warm[-1]
                for prev, curr in zip(warm[:-1], warm[1:]):
                    facts.append(_make_fact(prev, curr, args.flat_bps))

                if len(facts) >= window:
                    _print_summary(facts, window)
                    if args.print_details:
                        for f in list(facts)[-window:]:
                            _print_fact_line(f)
                        printed_initial = True
        except Exception as exc:
            logging.warning("History warmup failed: %s (continue WS only)", exc)

    # 2) Live WS: обновляем раз в минуту и печатаем ТОЛЬКО новое
    async for candle in stream_candles(data_cfg):
        # пропускаем старые/дубли (WS snapshot может пересечься с REST warmup)
        if last_candle is not None and candle.start_ts <= last_candle.start_ts:
            if candle.start_ts == last_candle.start_ts:
                last_candle = candle  # на всякий случай обновим
            continue

        if last_candle is None:
            last_candle = candle
            continue

        fact = _make_fact(last_candle, candle, args.flat_bps)
        facts.append(fact)

        if writer is not None:
            writer.write(
                {
                    "prev_ts": fact.prev_start_ts,
                    "candle_ts": fact.curr_start_ts,
                    "close_prev": fact.close_prev,
                    "close_curr": fact.close_curr,
                    "delta": fact.delta,
                    "ret_bps": fact.ret_bps,
                    "direction": fact.direction.value,
                }
            )

        last_candle = candle

        if len(facts) < window:
            continue

        _print_summary(facts, window)

        if args.print_details:
            if args.details_full_window or not printed_initial:
                # первый раз (или если явно попросили) печатаем окно целиком
                for f in list(facts)[-window:]:
                    _print_fact_line(f)
                printed_initial = True
            else:
                # ✅ дальше печатаем только НОВУЮ строку (без повторов старых)
                _print_fact_line(fact)

    if writer is not None:
        writer.close()


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(run_live(args))
    except KeyboardInterrupt:
        logging.info("Stopped")


if __name__ == "__main__":
    main()
