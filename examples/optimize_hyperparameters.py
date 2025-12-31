"""
Example: Hyperparameter optimization with Optuna.

This script demonstrates how to use the AutoTuner to optimize
Titan's hyperparameters for maximum accuracy.

Usage:
    python examples/optimize_hyperparameters.py --csv data/btc_1m.csv --trials 50
"""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from titan.core.tuner import create_tuner, HAS_OPTUNA


def main():
    if not HAS_OPTUNA:
        print("ERROR: Optuna is not installed.")
        print("Install with: pip install optuna")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Optimize Titan hyperparameters")
    parser.add_argument("--csv", required=True, help="Path to candle CSV data")
    parser.add_argument("--db", default="titan.db", help="Database path")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials")
    parser.add_argument(
        "--objective",
        choices=["accuracy", "sharpe", "multi"],
        default="accuracy",
        help="Optimization objective",
    )
    parser.add_argument(
        "--pruner",
        choices=["median", "hyperband", "none"],
        default="median",
        help="Pruner algorithm",
    )
    parser.add_argument("--output", default="runs/tuner", help="Output directory")

    args = parser.parse_args()

    print("=" * 60)
    print("TITAN HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print(f"Data: {args.csv}")
    print(f"Trials: {args.trials}")
    print(f"Objective: {args.objective}")
    print(f"Pruner: {args.pruner}")
    print()

    # Create tuner
    tuner = create_tuner(
        db_path=args.db,
        n_trials=args.trials,
        objective=args.objective,
        pruner=args.pruner,
    )

    # Run optimization
    print("Starting optimization...")
    best_params = tuner.optimize(candles_path=args.csv)

    # Save results
    os.makedirs(args.output, exist_ok=True)

    results_path = os.path.join(args.output, "results.json")
    tuner.save_results(results_path)
    print(f"\nResults saved to: {results_path}")

    config_path = os.path.join(args.output, "optimized_config.json")
    tuner.export_config(config_path)
    print(f"Config saved to: {config_path}")

    # Generate visualizations
    viz_dir = os.path.join(args.output, "plots")
    try:
        tuner.visualize(viz_dir)
        print(f"Visualizations saved to: {viz_dir}")
    except Exception as e:
        print(f"Warning: Could not generate visualizations: {e}")

    # Print summary
    summary = tuner.get_summary()
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"Completed trials: {summary['completed_trials']}/{summary['total_trials']}")

    if summary.get('best_accuracy'):
        print(f"\nBest accuracy: {summary['best_accuracy']:.4f}")
        if 'best_sharpe' in summary:
            print(f"Best Sharpe: {summary['best_sharpe']:.2f}")

    if summary.get('avg_accuracy'):
        print(f"\nAverage metrics (completed trials):")
        print(f"  Accuracy: {summary['avg_accuracy']:.4f}")
        print(f"  ECE: {summary['avg_ece']:.4f}")
        print(f"  Sharpe: {summary['avg_sharpe']:.2f}")

    print("\nBest parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # Get parameter importance
    importance = tuner.get_importance()
    if importance:
        print("\nParameter importance:")
        sorted_importance = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for param, score in sorted_importance[:10]:  # Top 10
            print(f"  {param}: {score:.4f}")

    print("\nTo use optimized parameters:")
    print(f"  1. Copy {config_path} to your project")
    print(f"  2. Load config in your code")
    print()


if __name__ == "__main__":
    main()
