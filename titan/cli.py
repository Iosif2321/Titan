import argparse
import os
import time
from typing import Dict, List

from titan.core.backtest import run_backtest
from titan.core.history import resolve_range, run_history_backtest
from titan.core.live import run_live


def _default_run_id() -> str:
    return time.strftime("backtest_%Y%m%d_%H%M%S")


def _default_live_id() -> str:
    return time.strftime("live_%Y%m%d_%H%M%S")


def _default_history_id() -> str:
    return time.strftime("history_%Y%m%d_%H%M%S")


def _parse_overrides(values: List[str]) -> Dict[str, object]:
    overrides: Dict[str, object] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid override format: {item}")
        key, raw = item.split("=", 1)
        value: object
        lower = raw.lower()
        if lower in {"true", "false"}:
            value = lower == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        overrides[key] = value
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(prog="titan")
    sub = parser.add_subparsers(dest="command", required=True)

    backtest = sub.add_parser("backtest", help="Run offline backtest on CSV data")
    backtest.add_argument("--csv", required=True, help="Path to 1m OHLCV CSV")
    backtest.add_argument("--db", default="titan.db", help="SQLite DB path")
    backtest.add_argument("--out", default="runs", help="Output directory")
    backtest.add_argument("--run-id", default=None, help="Custom run id")
    backtest.add_argument("--limit", type=int, default=None, help="Limit rows for debug")
    backtest.add_argument(
        "--no-tune-weights",
        action="store_true",
        help="Disable automatic weight tuning",
    )

    live = sub.add_parser("live", help="Run live websocket test")
    live.add_argument("--symbol", default="BTCUSDT", help="Market symbol")
    live.add_argument("--interval", default="1", help="Kline interval in minutes")
    live.add_argument("--db", default="titan.db", help="SQLite DB path")
    live.add_argument("--out", default="runs", help="Output directory")
    live.add_argument("--run-id", default=None, help="Custom run id")
    live.add_argument(
        "--max-predictions",
        type=int,
        default=None,
        help="Stop after N evaluated predictions",
    )
    live.add_argument(
        "--no-tune-weights",
        action="store_true",
        help="Disable automatic weight tuning",
    )
    live.add_argument(
        "--no-store-candles",
        action="store_true",
        help="Disable candle storage in SQLite",
    )
    live.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config value, format: key=value",
    )

    history = sub.add_parser("history", help="Download historical data and backtest")
    history.add_argument("--symbol", default="BTCUSDT", help="Market symbol")
    history.add_argument("--interval", default="1", help="Kline interval in minutes")
    history.add_argument("--start", default=None, help="Start time (ISO8601 or epoch)")
    history.add_argument("--end", default=None, help="End time (ISO8601 or epoch)")
    history.add_argument("--hours", type=float, default=None, help="Lookback window in hours")
    history.add_argument("--db", default="titan.db", help="SQLite DB path")
    history.add_argument("--out", default="runs", help="Output directory")
    history.add_argument("--run-id", default=None, help="Custom run id")
    history.add_argument(
        "--prefill-minutes",
        type=float,
        default=None,
        help="Extra minutes before start to warm up features (default: auto)",
    )
    history.add_argument(
        "--no-eval-buffer",
        action="store_true",
        help="Disable fetching one extra interval for evaluation",
    )
    history.add_argument(
        "--no-tune-weights",
        action="store_true",
        help="Disable automatic weight tuning",
    )
    history.add_argument(
        "--store-candles",
        action="store_true",
        help="Store downloaded candles in SQLite",
    )

    args = parser.parse_args()

    if args.command == "backtest":
        run_id = args.run_id or _default_run_id()
        out_dir = os.path.join(args.out, run_id)
        run_backtest(
            csv_path=args.csv,
            db_path=args.db,
            out_dir=out_dir,
            limit=args.limit,
            tune_weights=not args.no_tune_weights,
        )
    elif args.command == "live":
        run_id = args.run_id or _default_live_id()
        out_dir = os.path.join(args.out, run_id)
        overrides = _parse_overrides(args.set)
        run_live(
            symbol=args.symbol,
            interval=args.interval,
            db_path=args.db,
            out_dir=out_dir,
            max_predictions=args.max_predictions,
            tune_weights=not args.no_tune_weights,
            store_candles=not args.no_store_candles,
            overrides=overrides or None,
        )
    elif args.command == "history":
        run_id = args.run_id or _default_history_id()
        out_dir = os.path.join(args.out, run_id)
        try:
            start_ts, end_ts = resolve_range(args.start, args.end, args.hours)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        run_history_backtest(
            symbol=args.symbol,
            interval=args.interval,
            start_ts=start_ts,
            end_ts=end_ts,
            db_path=args.db,
            out_dir=out_dir,
            tune_weights=not args.no_tune_weights,
            store_candles=args.store_candles,
            prefill_minutes=args.prefill_minutes,
            eval_buffer=not args.no_eval_buffer,
        )


if __name__ == "__main__":
    main()
