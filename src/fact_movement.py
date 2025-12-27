# src/fact_movement.py
import argparse
import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional

from .bybit_rest import fetch_klines
from .bybit_ws import stream_candles
from .config import DataConfig, FactConfig, RestConfig
from .recording import JsonlWriter
from .types import Candle, Direction
from .utils import ts_iso


@dataclass(frozen=True)
class FactMove:
    prev_start_ts: int
    curr_start_ts: int
    close_prev: float
    close_curr: float
    delta: float
    ret_bps: float
    direction: Direction


def _direction_from_close(
    close_prev: float, close_curr: float, fact_flat_bps: float
) -> Direction:
    if close_prev <= 0:
        return Direction.FLAT
    ret_bps = ((close_curr - close_prev) / close_prev) * 10_000.0
    if abs(ret_bps) <= fact_flat_bps:
        return Direction.FLAT
    return Direction.UP if close_curr > close_prev else Direction.DOWN


def _make_fact(prev: Candle, curr: Candle, fact_flat_bps: float) -> FactMove:
    delta = curr.close - prev.close
    ret_bps = (delta / prev.close) * 10_000.0 if prev.close > 0 else 0.0
    direction = _direction_from_close(prev.close, curr.close, fact_flat_bps)
    return FactMove(
        prev_start_ts=prev.start_ts,
        curr_start_ts=curr.start_ts,
        close_prev=prev.close,
        close_curr=curr.close,
        delta=delta,
        ret_bps=ret_bps,
        direction=direction,
    )


def _print_summary(facts: Deque[FactMove], window: int) -> None:
    last = list(facts)[-window:]
    pattern = " ".join(f.direction.value for f in last)
    up = sum(1 for f in last if f.direction == Direction.UP)
    down = sum(1 for f in last if f.direction == Direction.DOWN)
    flat = sum(1 for f in last if f.direction == Direction.FLAT)
    net_bps = sum(f.ret_bps for f in last)
    logging.info(
        "FACT asof=%s last_%d=[%s] up=%d down=%d flat=%d net_bps=%.2f",
        ts_iso(last[-1].curr_start_ts),
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
        ts_iso(f.prev_start_ts),
        ts_iso(f.curr_start_ts),
        f.direction.value,
        f.close_prev,
        f.close_curr,
        f.delta,
        f.ret_bps,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    d = DataConfig()
    r = RestConfig()
    p = argparse.ArgumentParser(
        description="FACT movement (close-to-close), Bybit Spot kline 1m."
    )
    p.add_argument("--symbol", default=d.symbol)
    p.add_argument("--tf", default=d.tfs[0])
    p.add_argument("--ws-url", default=d.ws_url)
    p.add_argument("--reconnect-delay", type=float, default=d.reconnect_delay)
    p.add_argument("--ping-interval", type=float, default=d.ping_interval)
    p.add_argument("--log-level", default="INFO")

    p.add_argument("--window", type=int, default=5)
    p.add_argument("--fact-flat-bps", type=float, default=FactConfig().fact_flat_bps)

    p.add_argument(
        "--history-limit",
        type=int,
        default=r.history_limit,
        help="REST warmup candles (startup).",
    )
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
        tfs=[args.tf],
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
            hist = fetch_klines(
                args.symbol,
                args.tf,
                limit=max(args.history_limit, window + 2),
                rest_config=RestConfig(),
            )
            if len(hist) >= window + 1:
                warm = hist[-(window + 1):]
                last_candle = warm[-1]
                for prev, curr in zip(warm[:-1], warm[1:]):
                    facts.append(_make_fact(prev, curr, args.fact_flat_bps))

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

        fact = _make_fact(last_candle, candle, args.fact_flat_bps)
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
