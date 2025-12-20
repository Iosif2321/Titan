# BTCUSDT Predictor (JAX)

Modular BTCUSDT spot 1m predictor using Bybit public WebSocket data. No trading.

## WSL2 setup (GPU)

1. Verify GPU is visible:
   ```bash
   nvidia-smi
   ```
2. Create venv and install base deps:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-base.txt
   pip install --upgrade "jax[cuda12]"
   ```
3. Run:
   ```bash
   python -m src.main --save-jsonl-dir ./out
   ```

## WSL2 setup (CPU)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.txt
python -m src.main --save-jsonl-dir ./out
```

## Docker (GPU)

```bash
docker build -t btc-predict .
docker run --gpus all --rm -v "$(pwd)/out:/app/out" btc-predict --save-jsonl-dir /app/out
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
