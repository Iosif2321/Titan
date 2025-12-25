# Operations

## WSL2 (GPU)

1. Verify GPU visibility:
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

## WSL2 (CPU)

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

## Persistence mounts

- `out/` stores JSONL artifacts.
- `state/state.db` stores model params/EMA/anchor/optimizer + metrics.
- `state/patterns.db` stores pattern memory + events.

## Diagnostics

```bash
python -m src.main --dry-run
```

This prints `jax.default_backend()` and `jax.devices()` then exits.

## Separate analysis loop

```bash
python -m src.analysis_loop \
  --updates ./out/updates.jsonl \
  --analysis-out ./out/analysis.jsonl
```

## Logging

- Use `--log-level INFO|DEBUG|WARNING`.
- Monitor WS reconnects and DB growth.
- Pattern maintenance runs periodically (see `--maintenance-seconds`).
