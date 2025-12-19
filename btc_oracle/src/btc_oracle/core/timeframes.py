"""Timeframe helpers for alignment and conversions."""

from datetime import datetime, timedelta


def timeframe_to_minutes(timeframe: str) -> int:
    """Convert timeframe string (e.g. '1m', '5m', '1h') into minutes."""
    if not timeframe:
        raise ValueError("timeframe is empty")
    tf = timeframe.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 1440
    if tf.endswith("w"):
        return int(tf[:-1]) * 10080
    if tf.isdigit():
        return int(tf)
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def align_to_minute(ts: datetime) -> datetime:
    """Drop seconds and microseconds without changing timezone."""
    return ts.replace(second=0, microsecond=0)


def candle_close_time(open_ts: datetime, timeframe_min: int) -> datetime:
    """Compute candle close timestamp from an open timestamp."""
    open_aligned = align_to_minute(open_ts)
    return open_aligned + timedelta(minutes=int(timeframe_min))
