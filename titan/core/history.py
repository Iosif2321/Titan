import csv
import datetime as dt
import math
import os
import sys
import time
from typing import Dict, Optional, Tuple

from titan.core.backtest import run_backtest
from titan.core.config import ConfigStore
from titan.core.data.bybit_rest import fetch_klines, interval_to_ms, validate_candle_continuity
from titan.core.data.store import CandleStore
from titan.core.state_store import StateStore


def parse_time(value: str) -> int:
    raw = value.strip()
    try:
        if raw.replace(".", "", 1).isdigit():
            ts = int(float(raw))
            if ts > 10_000_000_000:
                ts //= 1000
            return ts
    except ValueError:
        pass

    if raw.endswith("Z"):
        raw = raw[:-1]
    raw = raw.replace(" ", "T")
    parsed = dt.datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return int(parsed.timestamp())


def resolve_range(
    start: Optional[str],
    end: Optional[str],
    hours: Optional[float],
) -> Tuple[int, int]:
    now_ts = int(time.time())
    if hours is not None:
        if start or end:
            raise ValueError("Use either --hours or --start/--end, not both.")
        start_ts = now_ts - int(hours * 3600)
        end_ts = now_ts
        return start_ts, end_ts

    if start is None:
        raise ValueError("--start is required when --hours is not set.")

    start_ts = parse_time(start)
    end_ts = parse_time(end) if end else now_ts

    if start_ts >= end_ts:
        raise ValueError("Start time must be earlier than end time.")

    return start_ts, end_ts


def run_history_backtest(
    symbol: str,
    interval: str,
    start_ts: int,
    end_ts: int,
    db_path: str,
    out_dir: str,
    tune_weights: bool,
    store_candles: bool,
    prefill_minutes: Optional[float] = None,
    eval_buffer: bool = True,
    verbose: bool = True,
    use_two_head: bool = False,
    two_head_checkpoint: Optional[str] = None,
    two_head_model_class: str = "TwoHeadMLP",
) -> Dict[str, object]:
    """Download historical candles and run backtest.

    Args:
        symbol: Market symbol (e.g., BTCUSDT)
        interval: Kline interval in minutes
        start_ts: Start timestamp (Unix seconds)
        end_ts: End timestamp (Unix seconds)
        db_path: SQLite database path
        out_dir: Output directory for results
        tune_weights: Whether to tune model weights after backtest
        store_candles: Whether to store candles in database
        prefill_minutes: Extra minutes before start to warm up features
        eval_buffer: Fetch one extra interval for evaluation
        verbose: Print progress to console

    Returns:
        Summary dictionary with all metrics
    """
    state_store = StateStore(db_path)
    config_store = ConfigStore(state_store)
    config_store.ensure_defaults()

    interval_sec = int(interval_to_ms(interval) / 1000)
    target_start_ts = start_ts
    target_end_ts = end_ts

    prefill_source = "auto"
    if prefill_minutes is None:
        slow = int(config_store.get("feature.slow_window", 20))
        volume = int(config_store.get("feature.volume_window", 20))
        vol = int(config_store.get("feature.vol_window", 20))
        rsi = int(config_store.get("feature.rsi_window", 14))
        # volatility_z requires: vol_window for returns.std() + vol_window for volatility.std()
        # So we need 2 * vol_window bars for proper warmup
        prefill_bars = max(slow, volume, 2 * vol, rsi + 1)
    else:
        if prefill_minutes <= 0:
            prefill_source = "disabled"
            prefill_bars = 0
        else:
            prefill_source = "custom"
            prefill_sec = prefill_minutes * 60
            prefill_bars = int(math.ceil(prefill_sec / interval_sec))

    prefill_sec = prefill_bars * interval_sec
    fetch_start_ts = max(target_start_ts - prefill_sec, 0)
    eval_buffer_sec = interval_sec if eval_buffer else 0
    fetch_end_ts = target_end_ts + eval_buffer_sec

    start_iso = dt.datetime.utcfromtimestamp(target_start_ts).isoformat() + "Z"
    end_iso = dt.datetime.utcfromtimestamp(target_end_ts).isoformat() + "Z"
    fetch_start_iso = dt.datetime.utcfromtimestamp(fetch_start_ts).isoformat() + "Z"
    fetch_end_iso = dt.datetime.utcfromtimestamp(fetch_end_ts).isoformat() + "Z"
    duration_hours = (end_ts - start_ts) / 3600

    if verbose:
        print("\n" + "=" * 70)
        print("                  HISTORY BACKTEST")
        print("=" * 70)
        print(f"\n  Symbol:   {symbol}")
        print(f"  Interval: {interval}m")
        print(f"  Period:   {start_iso}")
        print(f"         -> {end_iso}")
        print(f"  Prefill:  {prefill_bars} bars ({prefill_sec // 60} min, {prefill_source})")
        print(f"  Fetch:    {fetch_start_iso} -> {fetch_end_iso}")
        print(f"  Duration: {duration_hours:.1f} hours")
        print(f"\n[Download] Fetching candles from Bybit...")

    download_start = time.time()
    candles = fetch_klines(symbol, interval, fetch_start_ts, fetch_end_ts)
    download_time = time.time() - download_start

    if not candles:
        raise RuntimeError("No candles returned for the requested period.")

    if verbose:
        print(f"[Download] Received {len(candles):,} candles in {download_time:.1f}s")

    # Check for gaps in candle data
    gaps, total_missing = validate_candle_continuity(candles, interval_sec)
    if gaps:
        if verbose:
            print(f"[Download] WARNING: Found {len(gaps)} gaps ({total_missing} missing candles)")
            for gap in gaps[:5]:  # Show first 5 gaps
                expected_iso = dt.datetime.utcfromtimestamp(gap.expected_ts).isoformat() + "Z"
                actual_iso = dt.datetime.utcfromtimestamp(gap.actual_ts).isoformat() + "Z"
                print(f"           Gap: expected {expected_iso}, got {actual_iso} ({gap.missing_count} missing)")
            if len(gaps) > 5:
                print(f"           ... and {len(gaps) - 5} more gaps")

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "candles.csv")

    if verbose:
        print(f"[Download] Saving to {csv_path}")

    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["timestamp", "open", "high", "low", "close", "volume"]
        )
        writer.writeheader()
        writer.writerows(
            {
                "timestamp": candle.ts,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
            }
            for candle in candles
        )

    if store_candles:
        if verbose:
            print(f"[Download] Storing candles in database...")
        CandleStore(db_path).insert_many(symbol, candles)

    # Price range info
    prices = [c.close for c in candles]
    price_min = min(prices)
    price_max = max(prices)
    price_start = candles[0].close
    price_end = candles[-1].close
    price_change_pct = ((price_end - price_start) / price_start) * 100 if price_start else 0

    if verbose:
        print(f"\n  Price Range: {price_min:.2f} - {price_max:.2f}")
        print(f"  Price Change: {price_start:.2f} -> {price_end:.2f} ({price_change_pct:+.2f}%)")

    target_candles = [c for c in candles if target_start_ts <= c.ts <= target_end_ts]
    run_meta: Dict[str, object] = {
        "source": "bybit_rest",
        "symbol": symbol,
        "interval": interval,
        "start_ts": target_start_ts,
        "end_ts": target_end_ts,
        "candles": len(candles),
        "target_candles": len(target_candles),
        "start_iso": start_iso,
        "end_iso": end_iso,
        "fetch_start_ts": fetch_start_ts,
        "fetch_end_ts": fetch_end_ts,
        "fetch_start_iso": fetch_start_iso,
        "fetch_end_iso": fetch_end_iso,
        "prefill_bars": prefill_bars,
        "prefill_minutes": round(prefill_sec / 60, 2),
        "prefill_source": prefill_source,
        "eval_buffer_sec": eval_buffer_sec,
        "duration_hours": round(duration_hours, 2),
        "download_time_sec": round(download_time, 2),
        "price_range": {"min": price_min, "max": price_max},
        "price_change_pct": round(price_change_pct, 4),
        "gaps_count": len(gaps),
        "missing_candles": total_missing,
    }

    return run_backtest(
        csv_path=csv_path,
        db_path=db_path,
        out_dir=out_dir,
        tune_weights=tune_weights,
        run_meta=run_meta,
        target_start_ts=target_start_ts,
        target_end_ts=target_end_ts,
        verbose=verbose,
        use_two_head=use_two_head,
        two_head_checkpoint=two_head_checkpoint,
        two_head_model_class=two_head_model_class,
    )
