import json
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

STATE_URL = "http://localhost:8000/api/state"
MODELS_URL = "http://localhost:8000/api/models"
LOG_PATH = Path("out_run/control_checks_8h_30m.log")

PRED_PATH = Path("out_run/predictions.jsonl")
FACT_PATH = Path("out_run/facts.jsonl")
CANDLE_PATH = Path("out_run/candles.jsonl")
UPDATE_PATH = Path("out_run/updates.jsonl")


def fetch_json(url: str) -> dict:
    with urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _safe_div(n: int, d: int) -> float:
    return float(n / d) if d else 0.0


def _read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _last_hour_stats(now_ms: int) -> dict:
    start_ts = now_ms - 3600_000
    facts = {}
    fact_dir = {"UP": 0, "DOWN": 0, "FLAT": 0}
    fact_abs_sum = 0.0
    fact_count = 0

    total_facts = 0
    last_hour_facts = 0
    for rec in _read_jsonl(FACT_PATH):
        total_facts += 1
        ts = rec.get("target_ts")
        if ts is None:
            ts = rec.get("curr_ts") or rec.get("ts")
        try:
            ts = int(ts)
        except Exception:
            continue
        if ts < start_ts:
            continue
        last_hour_facts += 1
        key = (str(rec.get("tf") or rec.get("interval") or ""), ts)
        facts[key] = rec
        direction = str(rec.get("direction") or rec.get("fact_dir") or "")
        if direction in fact_dir:
            fact_dir[direction] += 1
        try:
            fact_abs_sum += abs(float(rec.get("ret_bps", 0.0)))
        except Exception:
            pass
        fact_count += 1

    matched = 0
    correct = 0
    nonflat_total = 0
    nonflat_correct = 0
    flat_total = 0
    flat_correct = 0
    pred_dir = {"UP": 0, "DOWN": 0, "FLAT": 0}
    conf_sum = 0.0
    conf_count = 0
    total_preds = 0
    last_hour_preds = 0

    for rec in _read_jsonl(PRED_PATH):
        total_preds += 1
        ts = rec.get("target_ts")
        if ts is None:
            ts = rec.get("curr_ts") or rec.get("ts")
        try:
            ts = int(ts)
        except Exception:
            continue
        if ts < start_ts:
            continue
        last_hour_preds += 1
        direction = str(rec.get("direction") or rec.get("dir") or "")
        if direction in pred_dir:
            pred_dir[direction] += 1
        try:
            conf_sum += float(rec.get("confidence", 0.0))
            conf_count += 1
        except Exception:
            pass
        key = (str(rec.get("tf") or rec.get("interval") or ""), ts)
        fact = facts.get(key)
        if not fact:
            continue
        matched += 1
        fact_dir_val = str(fact.get("direction") or fact.get("fact_dir") or "")
        is_correct = direction == fact_dir_val
        if is_correct:
            correct += 1
        if direction != "FLAT":
            nonflat_total += 1
            if is_correct:
                nonflat_correct += 1
        else:
            flat_total += 1
            if fact_dir_val == "FLAT":
                flat_correct += 1

    total_candles = 0
    last_hour_candles = 0
    for rec in _read_jsonl(CANDLE_PATH):
        total_candles += 1
        ts = rec.get("ts_end") or rec.get("ts_start") or rec.get("curr_ts") or rec.get("ts")
        try:
            ts = int(ts)
        except Exception:
            continue
        if ts >= start_ts:
            last_hour_candles += 1

    total_updates = 0
    last_hour_updates = 0
    for rec in _read_jsonl(UPDATE_PATH):
        total_updates += 1
        ts = rec.get("ts_eval") or rec.get("ts")
        try:
            ts = int(ts)
        except Exception:
            continue
        if ts >= start_ts:
            last_hour_updates += 1

    return {
        "last_hour": {
            "matched": matched,
            "accuracy": _safe_div(correct, matched),
            "accuracy_nonflat": _safe_div(nonflat_correct, nonflat_total),
            "flat_rate": _safe_div(flat_total, matched),
            "flat_accuracy": _safe_div(flat_correct, flat_total),
            "avg_confidence": _safe_div(conf_sum, conf_count),
            "avg_abs_ret_bps": _safe_div(fact_abs_sum, fact_count),
            "pred_dir": pred_dir,
            "fact_dir": fact_dir,
        },
        "counts": {
            "total_preds": total_preds,
            "last_hour_preds": last_hour_preds,
            "total_facts": total_facts,
            "last_hour_facts": last_hour_facts,
            "total_candles": total_candles,
            "last_hour_candles": last_hour_candles,
            "total_updates": total_updates,
            "last_hour_updates": last_hour_updates,
        },
    }


def run_checks(cycles: int = 16, sleep_seconds: int = 1800) -> None:
    for cycle in range(1, cycles + 1):
        time.sleep(sleep_seconds)
        now = datetime.now().astimezone()
        now_ms = int(now.timestamp() * 1000)
        record = {
            "cycle": cycle,
            "ts": now.isoformat(),
            "state_ok": False,
        }
        try:
            state = fetch_json(STATE_URL)
            metrics = state.get("metrics", {}) if isinstance(state, dict) else {}
            ws = state.get("ws", {}) if isinstance(state, dict) else {}
            record.update(
                {
                    "state_ok": True,
                    "ws_connected": ws.get("connected"),
                    "ws_last_error": ws.get("last_error"),
                    "latest_candle": state.get("latest_candle"),
                    "latest_prediction": state.get("latest_prediction"),
                    "latest_fact": state.get("latest_fact"),
                    "latest_update": state.get("latest_update"),
                    "metrics": metrics,
                }
            )
        except Exception as exc:
            record["error"] = str(exc)

        try:
            models = fetch_json(MODELS_URL)
            record["models"] = models.get("models") if isinstance(models, dict) else models
        except Exception as exc:
            record["models_error"] = str(exc)

        record.update(_last_hour_stats(now_ms))

        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_checks()
