"""CLI entry point for the continuous learning loop.

Usage:
    python scripts/run_loop.py --max-periods 52 --algorithm ppo
    python scripts/run_loop.py --max-periods 8 --config config/default.yaml
"""

import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Run continuous RL pricing loop")
    parser.add_argument("--max-periods", type=int, default=52)
    parser.add_argument("--algorithm", choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from src.pipeline.continuous_loop import ContinuousLoop

    loop = ContinuousLoop(config=config, algorithm=args.algorithm)

    def on_period(period, info):
        status = []
        if info.get("retrained"):
            status.append("RETRAINED")
        if info.get("promoted"):
            status.append("PROMOTED")
        alerts = info.get("alerts", {})
        if alerts.get("any_alert"):
            status.append("ALERT")

        metrics = info.get("eval_metrics", {})
        mean_r = metrics.get("mean_reward", 0.0)
        status_str = " | ".join(status) if status else "OK"
        print(f"Period {period:3d} | Mean reward: {mean_r:+.4f} | {status_str}")

    print(f"Starting continuous loop: {args.algorithm}, max {args.max_periods} periods")
    print(f"Retrain interval: {loop.retrain_interval}, warm start: {loop.warm_start}")
    print("-" * 70)

    history = loop.run(max_periods=args.max_periods, callback=on_period)

    # Summary
    retrains = sum(1 for h in history if h.get("retrained"))
    promotions = sum(1 for h in history if h.get("promoted"))
    alerts = sum(1 for h in history if h.get("alerts", {}).get("any_alert"))

    print("-" * 70)
    print(f"Completed {len(history)} periods")
    print(f"  Retrains: {retrains}")
    print(f"  Promotions: {promotions}")
    print(f"  Alert periods: {alerts}")

    champion = loop.registry.get_champion()
    if champion:
        print(f"  Champion: {champion.version_id} ({champion.model_path})")


if __name__ == "__main__":
    main()
