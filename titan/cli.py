import argparse
import os
import time
from typing import Dict, List

from titan.core.backtest import run_backtest
from titan.core.history import resolve_range, run_history_backtest
from titan.core.live import run_live
from titan.core.tuner import create_tuner, HAS_OPTUNA


def _default_run_id() -> str:
    return time.strftime("backtest_%Y%m%d_%H%M%S")


def _default_live_id() -> str:
    return time.strftime("live_%Y%m%d_%H%M%S")


def _default_history_id() -> str:
    return time.strftime("history_%Y%m%d_%H%M%S")


def _parse_overrides(values: List[str]) -> Dict[str, object]:
    overrides: Dict[str, object] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid override format: {item}")
        key, raw = item.split("=", 1)
        value: object
        lower = raw.lower()
        if lower in {"true", "false"}:
            value = lower == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        overrides[key] = value
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(prog="titan")
    sub = parser.add_subparsers(dest="command", required=True)

    backtest = sub.add_parser("backtest", help="Run offline backtest on CSV data")
    backtest.add_argument("--csv", required=True, help="Path to 1m OHLCV CSV")
    backtest.add_argument("--db", default="titan.db", help="SQLite DB path")
    backtest.add_argument("--out", default="runs", help="Output directory")
    backtest.add_argument("--run-id", default=None, help="Custom run id")
    backtest.add_argument("--limit", type=int, default=None, help="Limit rows for debug")
    backtest.add_argument(
        "--no-tune-weights",
        action="store_true",
        help="Disable automatic weight tuning",
    )
    backtest.add_argument(
        "--use-two-head",
        action="store_true",
        help="Use TwoHeadMLP model instead of heuristic models",
    )
    backtest.add_argument(
        "--two-head-checkpoint",
        default=None,
        help="Path to TwoHeadMLP checkpoint file",
    )
    backtest.add_argument(
        "--two-head-model-class",
        default="TwoHeadMLP",
        choices=["TwoHeadMLP", "SessionEmbeddingMLP", "SessionGatedMLP"],
        help="TwoHeadMLP model class variant",
    )

    live = sub.add_parser("live", help="Run live websocket test")
    live.add_argument("--symbol", default="BTCUSDT", help="Market symbol")
    live.add_argument("--interval", default="1", help="Kline interval in minutes")
    live.add_argument("--db", default="titan.db", help="SQLite DB path")
    live.add_argument("--out", default="runs", help="Output directory")
    live.add_argument("--run-id", default=None, help="Custom run id")
    live.add_argument(
        "--max-predictions",
        type=int,
        default=None,
        help="Stop after N evaluated predictions",
    )
    live.add_argument(
        "--no-tune-weights",
        action="store_true",
        help="Disable automatic weight tuning",
    )
    live.add_argument(
        "--no-store-candles",
        action="store_true",
        help="Disable candle storage in SQLite",
    )
    live.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config value, format: key=value",
    )
    live.add_argument(
        "--use-two-head",
        action="store_true",
        help="Use TwoHeadMLP model instead of heuristic models",
    )
    live.add_argument(
        "--two-head-checkpoint",
        default=None,
        help="Path to TwoHeadMLP checkpoint file",
    )
    live.add_argument(
        "--two-head-model-class",
        default="TwoHeadMLP",
        choices=["TwoHeadMLP", "SessionEmbeddingMLP", "SessionGatedMLP"],
        help="TwoHeadMLP model class variant",
    )

    history = sub.add_parser("history", help="Download historical data and backtest")
    history.add_argument("--symbol", default="BTCUSDT", help="Market symbol")
    history.add_argument("--interval", default="1", help="Kline interval in minutes")
    history.add_argument("--start", default=None, help="Start time (ISO8601 or epoch)")
    history.add_argument("--end", default=None, help="End time (ISO8601 or epoch)")
    history.add_argument("--hours", type=float, default=None, help="Lookback window in hours")
    history.add_argument("--db", default="titan.db", help="SQLite DB path")
    history.add_argument("--out", default="runs", help="Output directory")
    history.add_argument("--run-id", default=None, help="Custom run id")
    history.add_argument(
        "--prefill-minutes",
        type=float,
        default=None,
        help="Extra minutes before start to warm up features (default: auto)",
    )
    history.add_argument(
        "--no-eval-buffer",
        action="store_true",
        help="Disable fetching one extra interval for evaluation",
    )
    history.add_argument(
        "--no-tune-weights",
        action="store_true",
        help="Disable automatic weight tuning",
    )
    history.add_argument(
        "--store-candles",
        action="store_true",
        help="Store downloaded candles in SQLite",
    )
    history.add_argument(
        "--use-two-head",
        action="store_true",
        help="Use TwoHeadMLP model instead of heuristic models",
    )
    history.add_argument(
        "--two-head-checkpoint",
        default=None,
        help="Path to TwoHeadMLP checkpoint file",
    )
    history.add_argument(
        "--two-head-model-class",
        default="TwoHeadMLP",
        choices=["TwoHeadMLP", "SessionEmbeddingMLP", "SessionGatedMLP"],
        help="TwoHeadMLP model class variant",
    )

    tune = sub.add_parser("tune", help="Hyperparameter optimization with Optuna")
    tune.add_argument("--csv", required=True, help="Path to 1m OHLCV CSV for optimization")
    tune.add_argument("--db", default="titan.db", help="SQLite DB path")
    tune.add_argument("--out", default="runs", help="Output directory")
    tune.add_argument("--trials", type=int, default=100, help="Number of optimization trials")
    tune.add_argument("--timeout", type=int, default=None, help="Total timeout in seconds")
    tune.add_argument("--timeout-per-trial", type=int, default=300, help="Timeout per trial in seconds")
    tune.add_argument(
        "--objective",
        choices=["accuracy", "sharpe", "multi"],
        default="accuracy",
        help="Optimization objective",
    )
    tune.add_argument(
        "--pruner",
        choices=["median", "hyperband", "none"],
        default="median",
        help="Pruner algorithm for early stopping",
    )
    tune.add_argument("--jobs", type=int, default=1, help="Number of parallel jobs")
    tune.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip generating visualizations",
    )

    args = parser.parse_args()

    if args.command == "backtest":
        run_id = args.run_id or _default_run_id()
        out_dir = os.path.join(args.out, run_id)
        run_backtest(
            csv_path=args.csv,
            db_path=args.db,
            out_dir=out_dir,
            limit=args.limit,
            tune_weights=not args.no_tune_weights,
            use_two_head=args.use_two_head,
            two_head_checkpoint=args.two_head_checkpoint,
            two_head_model_class=args.two_head_model_class,
        )
    elif args.command == "live":
        run_id = args.run_id or _default_live_id()
        out_dir = os.path.join(args.out, run_id)
        overrides = _parse_overrides(args.set)
        run_live(
            symbol=args.symbol,
            interval=args.interval,
            db_path=args.db,
            out_dir=out_dir,
            max_predictions=args.max_predictions,
            tune_weights=not args.no_tune_weights,
            store_candles=not args.no_store_candles,
            overrides=overrides or None,
            use_two_head=args.use_two_head,
            two_head_checkpoint=args.two_head_checkpoint,
            two_head_model_class=args.two_head_model_class,
        )
    elif args.command == "history":
        run_id = args.run_id or _default_history_id()
        out_dir = os.path.join(args.out, run_id)
        try:
            start_ts, end_ts = resolve_range(args.start, args.end, args.hours)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        run_history_backtest(
            symbol=args.symbol,
            interval=args.interval,
            start_ts=start_ts,
            end_ts=end_ts,
            db_path=args.db,
            out_dir=out_dir,
            tune_weights=not args.no_tune_weights,
            store_candles=args.store_candles,
            prefill_minutes=args.prefill_minutes,
            eval_buffer=not args.no_eval_buffer,
            use_two_head=args.use_two_head,
            two_head_checkpoint=args.two_head_checkpoint,
            two_head_model_class=args.two_head_model_class,
        )
    elif args.command == "tune":
        if not HAS_OPTUNA:
            raise SystemExit(
                "Optuna is required for hyperparameter tuning.\n"
                "Install with: pip install optuna"
            )

        run_id = time.strftime("tune_%Y%m%d_%H%M%S")
        out_dir = os.path.join(args.out, run_id)
        os.makedirs(out_dir, exist_ok=True)

        print(f"Starting hyperparameter optimization...")
        print(f"  Trials: {args.trials}")
        print(f"  Objective: {args.objective}")
        print(f"  Pruner: {args.pruner}")
        print(f"  Output: {out_dir}")
        print()

        # Create tuner
        from titan.core.tuner import TunerConfig, AutoTuner

        config = TunerConfig(
            n_trials=args.trials,
            timeout_per_trial=args.timeout_per_trial,
            objective_type=args.objective,
            pruner_type=args.pruner,
            n_jobs=args.jobs,
        )

        tuner = AutoTuner(db_path=args.db, config=config)

        # Run optimization
        best_params = tuner.optimize(
            candles_path=args.csv,
            timeout_total=args.timeout,
        )

        # Save results
        summary = tuner.get_summary()
        results_path = os.path.join(out_dir, "optimization_results.json")
        tuner.save_results(results_path)

        # Export best config
        config_path = os.path.join(out_dir, "config_optimized.json")
        tuner.export_config(config_path)

        # Generate visualizations
        if not args.no_visualize:
            viz_dir = os.path.join(out_dir, "visualizations")
            try:
                tuner.visualize(viz_dir)
                print(f"\n  Visualizations saved to: {viz_dir}")
            except Exception as e:
                print(f"\n  Warning: Failed to generate visualizations: {e}")

        # Print summary
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Total trials: {summary['total_trials']}")
        print(f"  Completed: {summary['completed_trials']}")
        print(f"  Pruned: {summary['pruned_trials']}")
        print(f"  Failed: {summary['failed_trials']}")

        if summary.get('best_accuracy'):
            print(f"\nBest Results:")
            print(f"  Accuracy: {summary['best_accuracy']:.4f}")
            if 'best_sharpe' in summary:
                print(f"  Sharpe: {summary['best_sharpe']:.2f}")

        if summary.get('avg_accuracy'):
            print(f"\nAverage (completed trials):")
            print(f"  Accuracy: {summary['avg_accuracy']:.4f}")
            print(f"  ECE: {summary['avg_ece']:.4f}")
            print(f"  Sharpe: {summary['avg_sharpe']:.2f}")
            print(f"  Duration: {summary['avg_duration']:.1f}s")

        print(f"\nBest Parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")

        print(f"\nResults saved to:")
        print(f"  - {results_path}")
        print(f"  - {config_path}")
        print()


if __name__ == "__main__":
    main()
