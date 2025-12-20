import argparse
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

import jax

from .bybit_ws import stream_candles
from .config import DataConfig, FeatureConfig, PredictorConfig
from .features import FeatureBuilder
from .model import ModelConfig, TwoHeadLinearModel
from .predictor import Predictor
from .recording import JsonlRecorder


def build_arg_parser() -> argparse.ArgumentParser:
    data_defaults = DataConfig()
    feature_defaults = FeatureConfig()
    predictor_defaults = PredictorConfig()

    parser = argparse.ArgumentParser(
        description="BTCUSDT spot kline predictor (UP/DOWN/FLAT)"
    )
    parser.add_argument("--symbol", default=data_defaults.symbol)
    parser.add_argument("--interval", default=data_defaults.interval)
    parser.add_argument("--ws-url", default=data_defaults.ws_url)
    parser.add_argument(
        "--include-unconfirmed",
        action="store_true",
        help="Use interim candles before close",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=data_defaults.reconnect_delay,
        help="Base delay in seconds for exponential reconnect backoff",
    )
    parser.add_argument(
        "--ping-interval",
        type=float,
        default=data_defaults.ping_interval,
        help="Seconds between Bybit heartbeat pings (0 disables)",
    )

    parser.add_argument("--lookback", type=int, default=feature_defaults.lookback)
    parser.add_argument(
        "--return-scale", type=float, default=feature_defaults.return_scale
    )
    parser.add_argument("--clip-return", type=float, default=feature_defaults.clip_return)

    parser.add_argument(
        "--flat-max-prob", type=float, default=predictor_defaults.flat_max_prob
    )
    parser.add_argument(
        "--flat-max-delta", type=float, default=predictor_defaults.flat_max_delta
    )

    parser.add_argument("--init-mode", default="heuristic", choices=["heuristic", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logit-scale", type=float, default=3.0)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--save-jsonl-dir",
        default=None,
        help="Directory for JSONL output (candles.jsonl, predictions.jsonl)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print JAX backend/devices and exit",
    )

    return parser


async def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info("JAX backend=%s devices=%s", jax.default_backend(), jax.devices())
    if args.dry_run:
        return

    data_config = DataConfig(
        symbol=args.symbol,
        interval=args.interval,
        ws_url=args.ws_url,
        use_confirmed_only=not args.include_unconfirmed,
        reconnect_delay=args.reconnect_delay,
        ping_interval=args.ping_interval,
    )
    feature_config = FeatureConfig(
        lookback=args.lookback,
        return_scale=args.return_scale,
        clip_return=args.clip_return,
    )
    predictor_config = PredictorConfig(
        flat_max_prob=args.flat_max_prob,
        flat_max_delta=args.flat_max_delta,
    )

    feature_builder = FeatureBuilder(feature_config)
    model_config = ModelConfig(
        input_size=feature_builder.spec.input_size,
        init_mode=args.init_mode,
        seed=args.seed,
        logit_scale=args.logit_scale,
    )
    model = TwoHeadLinearModel(model_config, feature_builder.spec)
    predictor = Predictor(feature_builder, model, predictor_config)

    logging.info(
        "Starting stream: %s %sm, lookback=%s",
        data_config.symbol,
        data_config.interval,
        feature_config.lookback,
    )

    recorder = JsonlRecorder(Path(args.save_jsonl_dir)) if args.save_jsonl_dir else None
    try:
        async for candle in stream_candles(data_config):
            if recorder is not None:
                recorder.record_candle(candle)

            prediction = predictor.ingest(candle)
            if prediction is None:
                continue

            if recorder is not None:
                recorder.record_prediction(prediction)

            ts = datetime.fromtimestamp(prediction.candle_ts / 1000, tz=timezone.utc)
            logging.info(
                "ts=%s p_up=%.4f p_down=%.4f dir=%s",
                ts.isoformat(),
                prediction.p_up,
                prediction.p_down,
                prediction.direction.value,
            )
    finally:
        if recorder is not None:
            recorder.close()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        logging.info("Stopped")


if __name__ == "__main__":
    main()
