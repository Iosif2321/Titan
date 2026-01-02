# TITAN

Offline core skeleton for 1-minute BTCUSDT forecasting.

## Requirements
Live websocket mode needs `websockets`:

```
pip install -r requirements.txt
```

## CSV format
The backtest expects a CSV with headers:

```
timestamp,open,high,low,close,volume
```

`timestamp` can be seconds or milliseconds since epoch.

## Run offline backtest

```
python -m titan.cli backtest --csv path/to/data.csv --db titan.db --out runs
```

Outputs:
- `runs/<run_id>/summary.json`
- `runs/<run_id>/predictions.jsonl`

SQLite (`titan.db`) stores model state (config/weights) and pattern history.

## Run historical backtest (Bybit REST)

Look back 2 hours from now:

```
python -m titan.cli history --symbol BTCUSDT --interval 1 --hours 2 --db titan.db --out runs
```

Prefill extra minutes before the start (warm-up for features):

```
python -m titan.cli history --symbol BTCUSDT --interval 1 --hours 2 --prefill-minutes 30
```

Disable evaluation buffer (no extra candle after end):

```
python -m titan.cli history --symbol BTCUSDT --interval 1 --hours 2 --no-eval-buffer
```

Or set an explicit range:

```
python -m titan.cli history --symbol BTCUSDT --interval 1 --start 2025-01-01T00:00:00Z --end 2025-01-01T02:00:00Z
```

## Run live websocket test

```
python -m titan.cli live --symbol BTCUSDT --interval 1 --db titan.db --out runs --max-predictions 5
```

Override config values (useful to reduce warm-up for a quick test):

```
python -m titan.cli live --symbol BTCUSDT --set feature.slow_window=5 --set feature.vol_window=5 --set feature.rsi_window=5
```
