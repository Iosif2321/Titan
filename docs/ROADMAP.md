# Roadmap

## Stage 1 (MVP, current)
Scope: single TF (1m), single model, online updates, artifacts, dashboard.

Definition of Done:
- Runs on WSL2 and Docker.
- `--dry-run` prints JAX backend/devices and exits.
- Live mode consumes confirmed candles and writes JSONL artifacts.
- pred(t) is evaluated against fact(t + TF) with correct time alignment.
- Checkpoints restore weights, optimizer state, and metrics.
- Dashboard works without Streamlit and supports live config updates.

## Stage 2 (Multi-TF + ensemble)
- Add 5m/15m/60m pipelines with shared contracts.
- Implement ensemble strategy (weighted or gating) with explicit model IDs.
- Add per-TF performance metrics and model status reporting.

Definition of Done:
- Multiple TF streams run concurrently without lookahead.
- Ensemble outputs a single prediction with provenance.
- Dashboard shows per-TF models and aggregate metrics.

## Stage 3 (Evaluation tooling)
- Dataset builder and backfill evaluation (pred vs fact).
- Offline metrics and sanity checks for data integrity/drift.
- Feature schema versioning and compatibility checks.

Definition of Done:
- Offline evaluator can compute accuracy/flat rates from JSONL.
- Feature schema mismatches are detected and reported.
- Backfill can regenerate features without lookahead.

## Stage 4 (Adaptive training)
- Adaptive reward policies and learning-rate scheduling.
- Model quality monitoring and rollback/decay strategies.
- Confidence calibration and abstain optimization.

Definition of Done:
- Adaptive policy knobs are configurable and logged.
- Automatic rollback triggers on quality regression.
- Calibrated confidence metrics are tracked.

## Stage 5 (Operational hardening)
- Metrics export (Prometheus or JSON endpoint), alerting hooks.
- Improved dashboard UX and configuration workflows.
- Long-term retention and cleanup for checkpoints/artifacts.

Definition of Done:
- Ops endpoints are stable and documented.
- Retention policy is configurable and enforced.
- Dashboards and logs support production monitoring.
