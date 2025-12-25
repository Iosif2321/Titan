# BTCUSDT Predictor (JAX)

Multi-timeframe BTCUSDT spot predictor using Bybit public WebSocket + REST warmup. No trading. Three independent models per TF (TRENDVIC, OSCILLATOR, VOLUMEMETRIX) with EMA/anchor memory, pattern-aware adaptation, JSONL artifacts, and a minimal dashboard.

## WSL2 setup (GPU)

1. Verify GPU is visible:
   ```bash
   nvidia-smi
   ```
2. Create venv and install deps:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-base.txt
   pip install --upgrade "jax[cuda12]"
   ```
3. Run:
   ```bash
   python -m src.main \
     --tfs "1" \
     --save-jsonl-dir ./out \
     --state-db ./state/state.db \
     --pattern-db ./state/patterns.db \
     --dashboard \
     --dashboard-port 8000
   ```

## WSL2 setup (CPU)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.txt
python -m src.main \
  --tfs "1" \
  --save-jsonl-dir ./out \
  --state-db ./state/state.db \
  --pattern-db ./state/patterns.db
```

## Docker (GPU)

```bash
docker build -t btc-predict .
docker run --gpus all --rm \
  -v "$(pwd)/out:/app/out" \
  -v "$(pwd)/state:/app/state" \
  -p 8000:8000 \
  btc-predict \
  --save-jsonl-dir /app/out \
  --state-db /app/state/state.db \
  --pattern-db /app/state/patterns.db \
  --dashboard \
  --dashboard-port 8000
```

## Docker Compose (GPU)

```bash
docker compose up --build
```

## Diagnostics

```bash
python -m src.main --dry-run
```

This prints `jax.default_backend()` and `jax.devices()` then exits.

## Dashboard

Start with `--dashboard --dashboard-port 8000` and open:

```
http://localhost:8000
```

## Artifacts & State

- `out/`: JSONL artifacts (`candles.jsonl`, `predictions.jsonl`, `facts.jsonl`, `updates.jsonl`, `analysis.jsonl`)
- `state/state.db`: model params/EMA/anchor + optimizer + metrics
- `state/patterns.db`: pattern memory + events

## Calibration

Each model applies online affine logit calibration on the UP/DOWN margin:

```
m = logit_up - logit_down
m_cal = a * m + b
p_up = sigmoid(m_cal)
p_down = 1 - p_up
```

Per-model `(a, b)` are updated after each closed candle and persisted in `state.db`.
JSONL includes raw vs calibrated probabilities (`p_up_raw`, `p_up_cal`, `conf_raw`, `conf_cal`).

To verify: watch `avg_conf` and `ece` in metrics/analysis; expect avg confidence to drop and ECE to improve over time.

## Separate analysis loop

Runs interim analysis every 30 minutes for up to 12 hours (or until stopped).

```bash
python -m src.analysis_loop \
  --updates ./out/updates.jsonl \
  --analysis-out ./out/analysis.jsonl
```

## Offline replay

Run a fast offline replay on historical Bybit Spot candles. It uses the same
closed-candle handler as live mode, but writes reports to a run directory and
keeps state isolated from production.

Smoke test (quick sanity):

```bash
python -m src.offline_replay --symbol BTCUSDT --tf 1 --minutes 120 --mode smoke
```

Train replay:

```bash
python -m src.offline_replay \
  --symbol BTCUSDT --tf 1 \
  --start "2025-12-25T09:00:00Z" --end "2025-12-25T13:00:00Z" \
  --mode train --run-dir ./runs/offline_20251225_0900_1300
```

Walk-forward:

```bash
python -m src.offline_replay \
  --symbol BTCUSDT --tf 1 \
  --start "2025-12-25T00:00:00Z" --end "2025-12-26T00:00:00Z" \
  --mode walkforward --train-min 60 --eval-min 20 \
  --run-dir ./runs/wf_20251225
```

Artifacts:

- `runs/<name>/out/*.jsonl`
- `runs/<name>/summary.json` + `runs/<name>/report.md`

If checks fail, the replay exits with a non-zero status and the errors are listed
in the report.

## Docs

- `docs/ARCHITECTURE.md`
- `docs/PATTERNS.md`
- `docs/OPERATIONS.md`
