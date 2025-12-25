# Architecture

This project is a prediction-only system for BTCUSDT Bybit Spot. It produces forecasts, facts, online updates, pattern memory, and persistence. No trading, orders, or positions.

## Layered modules

- Data: `src/bybit_ws.py` (WS klines), `src/bybit_rest.py` (REST warmup).
- Indicators: `src/indicators.py` (MA/MACD/RSI/etc).
- Features: `src/features.py` (TRENDVIC/OSCILLATOR/VOLUMEMETRIX feature builders).
- Online model: `src/model.py`, `src/online_model.py`, `src/optimizer.py`.
- Engine: `src/engine.py` (MultiTimeframeEngine, ModelRunner, pending pred->fact alignment).
- Patterns: `src/pattern_store.py` (SQLite memory + maintenance).
- Persistence: `src/state_store.py` (SQLite model state), `src/recording.py` (JSONL artifacts).
- Dashboard: `src/dashboard/server.py` + `src/dashboard/static/index.html`.
- Runtime state: `src/runtime_state.py`.
- Config: `src/config.py`, `src/config_manager.py`.

## Core contracts (data shapes)

- Candle: `start_ts`, `end_ts`, `open/high/low/close`, `volume`, `confirmed`, `tf`.
- Prediction: `ts`, `tf`, `model_id`, `model_type`, `candle_ts`, `target_ts`, `logits_up/down`, `p_up/p_down`, `direction`, `confidence`, `context_key_used`, `decision_key_used`, `flat_thresholds`.
- Fact: `tf`, `prev_ts`, `curr_ts`, `close_prev`, `close_curr`, `ret_bps`, `direction`.
- UpdateEvent: `ts`, `tf`, `model_id`, `model_type`, `target_ts`, `candle_ts`, `pred_dir`, `fact_dir`, `reward`, `loss_task`, `loss_total`, `lr_eff`, `anchor_lambda_eff`, `weight_norms`.

## Time alignment rule (no lookahead)

For each TF:

- Prediction created on the close of candle with `start_ts = t` has `target_ts = t + TF`.
- The pending record stores `close_prev` and `features` from time `t`.
- When candle `target_ts` closes, fact is computed from `close_curr` vs stored `close_prev`.
- Updates occur only after `target_ts` arrives, ensuring `pred(t) -> fact(t + TF)`.

## Three models per TF

Each timeframe has three independent models:

- TRENDVIC: OHLCV + trend indicators (MA/MACD/Parabolic SAR).
- OSCILLATOR: RSI/Stochastic/CCI only (no raw OHLCV features).
- VOLUMEMETRIX: OHLCV + volume analysis (OBV/MFI/volume z-score).

Each model outputs UP/DOWN logits with FLAT as abstain via thresholds.

## EMA + anchor memory

- `params`: trainable weights.
- `ema_params`: EMA smoothing for inference.
- `anchor_params`: slow memory (very slow EMA) for stability.
- Loss: `loss_total = loss_task + anchor_lambda * ||params - anchor_params||^2`.
- `ema_params` update after each step; `anchor_params` update under policy.

## Patterns and adaptation

- Context patterns track market condition priors.
- Decision patterns track outcomes for (context + predicted direction).
- Pattern trust controls learning rate and anchor strength.
- Anti-patterns push predictions to FLAT (abstain) rather than flip direction.

## Persistence

- `state/state.db`: model params/EMA/anchor/optimizer/metrics per TF/model.
- `state/patterns.db`: pattern stats + pattern events.
- JSONL artifacts in `out/`.

## Multi-timeframe engine

`MultiTimeframeEngine` manages:

- Candle buffers per TF.
- 3 `ModelRunner`s per TF.
- Pending predictions per model (target_ts -> features + close_prev).
- Online updates and autosave.
- Pattern maintenance (periodic).
