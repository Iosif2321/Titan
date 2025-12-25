# Patterns

Patterns are a lightweight SQLite memory that track market contexts and model decisions.

## Keys

Two levels are used for each model + TF:

- `fine` key: detailed bins from indicators.
- `coarse` key: reduced bins for fallback.

Keys are deterministic and stable:

```
context_key = "{tf}:{model_id}:{level}:k1=v1|k2=v2|..."
decision_key = context_key + "|PRED=UP/DOWN/FLAT"
```

## Tables

`pattern_stats` (long-lived):

- `pattern_key` (PK), `tf`, `model_id`, `kind` (context/decision)
- `count`, `ema_win`, `ema_reward`
- `ema_p_up`, `ema_p_down`, `ema_p_flat`
- `streak_bad`, `last_seen_ts`

`pattern_events` (bounded history):

- `ts`, `tf`, `model_id`, `kind`, `pattern_key`
- `candle_ts`, `target_ts`, `close_prev`, `close_curr`
- `pred_dir`, `pred_conf`, `fact_dir`, `reward`, `ret_bps`
- `lr_eff`, `anchor_lambda_eff`

## Updates

After each fact:

- Context stats update `ema_p_up/down/flat` and `ema_reward`.
- Decision stats update `ema_win`, `ema_reward`, `streak_bad`.

EMA decay is configurable via `--pattern-ema-decay`.

## Trust

```
support_factor = min(1, count / support_k)
recency_factor = exp(-(now - last_seen) / tau_ms)
quality = ema_win (decision) OR max(ema_p_up, ema_p_down) (context)
trust = support_factor * recency_factor * quality
```

Trust chooses fine vs coarse keys and modulates learning.

## Adaptation

Learning rate and anchor strength are adapted per decision key:

```
lr_mult = clamp(1 + lr_gain * trust_dec * (0.5 - prior_win))
anchor_mult = clamp(1 + anchor_gain * trust_dec * (prior_win - 0.5))
```

## Anti-patterns

A decision pattern becomes anti-pattern when:

- `count >= anti_min_support`
- `ema_win <= anti_win_threshold`
- `trust >= anti_trust_threshold`

Anti-patterns do not flip direction; they raise FLAT thresholds to abstain.

## Maintenance

Periodic cleanup (default 600s):

- Delete old `pattern_events` (TTL + max rows).
- Remove stale patterns by age + low support.
- Evict lowest-scoring patterns if `max_patterns` exceeded.
