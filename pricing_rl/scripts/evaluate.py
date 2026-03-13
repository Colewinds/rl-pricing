"""Evaluation script for comparing pricing agents with item-level metrics.

Usage:
    python scripts/evaluate.py --agents ppo,heuristic --episodes 100
    python scripts/evaluate.py --ab-test --treatment ppo --control heuristic --simulations 100
    python scripts/evaluate.py --generate-report --output results/
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from src.agent.heuristic_baseline import HeuristicBaseline
from src.agent.rl_agent import RLAgent
from src.environment.customer import CustomerState, CustomerItemState
from src.environment.item import CATEGORIES
from src.environment.pricing_env import DynamicPricingEnv
from src.evaluation.ab_test_simulator import ABTestSimulator
from src.evaluation.metrics import (
    compute_action_entropy,
    compute_churn_rate_by_css,
    compute_css_migration,
    compute_portfolio_margin,
)
from src.monitoring.drift_detector import DriftDetector
from src.reward.reward_functions import CLVOptimizer


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_best_model(agent_name: str, model_dir: str = "results/models") -> str | None:
    """Find the most recent trained model for an agent type."""
    model_path = Path(model_dir)
    if not model_path.exists():
        return None

    candidates = sorted(
        [d for d in model_path.iterdir() if d.is_dir() and d.name.startswith(agent_name + "_")],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

    for d in candidates:
        best = d / "best_model.zip"
        if best.exists():
            return str(best.with_suffix(""))
        final = d / "final_model.zip"
        if final.exists():
            return str(final.with_suffix(""))
    return None


def get_agent(agent_name: str, env, model_path: str | None = None):
    """Instantiate an agent by name."""
    if agent_name == "heuristic":
        return HeuristicBaseline()
    elif agent_name in ("ppo", "dqn"):
        resolved_path = model_path or find_best_model(agent_name)
        if resolved_path:
            print(f"  Loading {agent_name} model from: {resolved_path}")
            return RLAgent(algorithm=agent_name, model_path=resolved_path, env=env)
        else:
            print(f"  WARNING: No trained model found for {agent_name}, using untrained agent")
            return RLAgent(algorithm=agent_name, env=env)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


def run_evaluation(agent, env, episodes: int, legacy_mode: bool = False) -> dict:
    """Run agent through multiple episodes and collect metrics.

    Collects both customer-level and item-level metrics.
    """
    all_rewards = []
    all_actions = []
    all_margins = []
    css_scores = []
    churned = []
    initial_css = []
    final_css = []

    # Item-level metrics
    category_margins = defaultdict(list)
    category_actions = defaultdict(list)
    concept_margins = defaultdict(list)

    for ep in range(episodes):
        obs, info = env.reset()
        ep_rewards = []
        ep_actions = []
        ep_margin = 0.0
        ep_initial_css = None
        ep_final_css = None
        ep_churned = False

        done = False
        while not done:
            if isinstance(agent, HeuristicBaseline):
                if legacy_mode:
                    cs = CustomerState.from_observation(obs)
                else:
                    cs = CustomerItemState.from_observation(obs).to_legacy_customer_state()
                if ep_initial_css is None:
                    ep_initial_css = cs.css_score
                action = agent.predict(cs)
            else:
                if ep_initial_css is None:
                    if legacy_mode:
                        cs = CustomerState.from_observation(obs)
                    else:
                        cs = CustomerItemState.from_observation(obs)
                        if hasattr(cs, 'to_legacy_customer_state'):
                            ep_initial_css = cs.css_score
                        else:
                            ep_initial_css = cs.css_score
                    ep_initial_css = cs.css_score
                action = agent.predict(obs)

            obs, reward, terminated, truncated, step_info = env.step(action)
            ep_rewards.append(reward)
            ep_actions.append(action)
            ep_margin += step_info.get("margin_dollars", 0.0)

            # Collect item-level info
            item_info = step_info.get("item_state", {})
            if item_info:
                cat_id = item_info.get("category", 0)
                cat_name = CATEGORIES.get(cat_id, "misc")
                category_margins[cat_name].append(item_info.get("item_margin_rate", 0.0))
                category_actions[cat_name].append(action)

            if terminated:
                ep_churned = step_info.get("churned", False)

            done = terminated or truncated

        if legacy_mode:
            cs_final = CustomerState.from_observation(obs)
        else:
            cs_final = CustomerItemState.from_observation(obs)
        ep_final_css = cs_final.css_score

        all_rewards.extend(ep_rewards)
        all_actions.extend(ep_actions)
        all_margins.append(ep_margin)
        css_scores.append(ep_initial_css or 3)
        churned.append(ep_churned)
        initial_css.append(ep_initial_css or 3)
        final_css.append(ep_final_css)

    # Compute metrics
    up, down, same = compute_css_migration(initial_css, final_css)

    result = {
        "mean_reward": float(np.mean(all_rewards)),
        "total_reward": float(np.sum(all_rewards)),
        "mean_episode_margin": float(np.mean(all_margins)),
        "action_entropy": compute_action_entropy(all_actions),
        "churn_rate_by_css": compute_churn_rate_by_css(css_scores, churned),
        "css_migration": {"up": up, "down": down, "same": same},
        "n_episodes": episodes,
        "n_steps": len(all_rewards),
    }

    # Add category-level metrics
    if category_margins:
        result["category_metrics"] = {}
        for cat_name in category_margins:
            margins = category_margins[cat_name]
            actions = category_actions[cat_name]
            result["category_metrics"][cat_name] = {
                "mean_margin": float(np.mean(margins)) if margins else 0.0,
                "n_actions": len(actions),
                "action_entropy": compute_action_entropy(actions) if actions else 0.0,
                "discount_share": sum(1 for a in actions if a in (3, 4, 5, 6)) / max(len(actions), 1),
            }

    return result


def compare_agents(args, config):
    """Compare multiple agents side by side."""
    agent_names = args.agents.split(",")
    reward_cfg = config.get("reward", {}).get("clv_optimizer")
    reward_fn = CLVOptimizer(reward_cfg)
    legacy_mode = getattr(args, "legacy", False)

    print(f"Evaluating agents: {agent_names} over {args.episodes} episodes each")
    print(f"Mode: {'legacy 17-dim' if legacy_mode else 'item-level 33-dim'}\n")

    results = {}
    for name in agent_names:
        name = name.strip()
        env = DynamicPricingEnv(config=config, reward_fn=reward_fn, legacy_mode=legacy_mode)
        agent = get_agent(name, env, args.model_path)
        metrics = run_evaluation(agent, env, args.episodes, legacy_mode=legacy_mode)
        results[name] = metrics

        print(f"--- {name} ---")
        print(f"  Mean reward:         {metrics['mean_reward']:.4f}")
        print(f"  Total reward:        {metrics['total_reward']:.2f}")
        print(f"  Mean episode margin: ${metrics['mean_episode_margin']:.2f}")
        print(f"  Action entropy:      {metrics['action_entropy']:.3f}")
        print(f"  Churn rate (overall):{sum(metrics['churn_rate_by_css'].values()) / max(len(metrics['churn_rate_by_css']), 1):.3f}")
        print(f"  CSS migration:       +{metrics['css_migration']['up']} "
              f"-{metrics['css_migration']['down']} "
              f"={metrics['css_migration']['same']}")

        # Print category metrics if available
        if "category_metrics" in metrics:
            print("  Category metrics:")
            for cat, cm in sorted(metrics["category_metrics"].items()):
                print(f"    {cat}: margin={cm['mean_margin']:.1%}, "
                      f"discount_share={cm['discount_share']:.1%}, "
                      f"actions={cm['n_actions']}")
        print()

    return results


def run_ab_test(args, config):
    """Run A/B test between treatment and control agents."""
    legacy_mode = getattr(args, "legacy", False)
    reward_fn = CLVOptimizer(config.get("reward", {}).get("clv_optimizer"))
    env = DynamicPricingEnv(config=config, reward_fn=reward_fn, legacy_mode=legacy_mode)

    treatment = get_agent(args.treatment, env)
    control = get_agent(args.control, env)

    sim = ABTestSimulator(
        treatment_agent=treatment,
        control_agent=control,
        env_config=config,
        n_simulations=args.simulations,
    )

    print(f"Running A/B test: {args.treatment} vs {args.control} "
          f"({args.simulations} simulations)...")
    result = sim.run()

    print(f"\n--- A/B Test Results ---")
    print(f"  Mean delta margin: ${result.mean_delta_margin:+.2f}")
    print(f"  95% CI:            (${result.ci_95[0]:+.2f}, ${result.ci_95[1]:+.2f})")
    print(f"  p-value:           {result.p_value:.4f}")
    sig = "YES" if result.p_value < 0.05 else "NO"
    print(f"  Significant (p<0.05): {sig}")

    return result


def generate_report(args, config):
    """Generate a comprehensive evaluation report with markdown and JSON."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_mode = getattr(args, "legacy", False)

    reward_fn = CLVOptimizer(config.get("reward", {}).get("clv_optimizer"))

    agent_names = ["heuristic", "ppo", "dqn"]
    results = {}
    for name in agent_names:
        env = DynamicPricingEnv(config=config, reward_fn=reward_fn, legacy_mode=legacy_mode)
        agent = get_agent(name, env)
        print(f"Evaluating {name}...")
        results[name] = run_evaluation(agent, env, episodes=100, legacy_mode=legacy_mode)

    # Write JSON report
    report_path = output_dir / f"evaluation_report_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Write markdown report
    md_path = output_dir / "results.md"
    with open(md_path, "w") as f:
        f.write("# RL Dynamic Pricing - Evaluation Results\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")

        f.write("## Agent Comparison (100 episodes each)\n\n")
        f.write("| Metric | " + " | ".join(agent_names) + " |\n")
        f.write("|--------|" + "|".join(["------" for _ in agent_names]) + "|\n")

        metrics_to_show = [
            ("Mean Reward", "mean_reward", ".4f"),
            ("Total Reward", "total_reward", ",.0f"),
            ("Mean Episode Margin ($)", "mean_episode_margin", ",.2f"),
            ("Action Entropy", "action_entropy", ".3f"),
            ("Episodes", "n_episodes", "d"),
            ("Total Steps", "n_steps", ",d"),
        ]

        for label, key, fmt in metrics_to_show:
            values = []
            for name in agent_names:
                if name in results and key in results[name]:
                    val = results[name][key]
                    if fmt == "d" or fmt == ",d":
                        values.append(f"{int(val):{fmt}}")
                    else:
                        values.append(f"{val:{fmt}}")
                else:
                    values.append("N/A")
            f.write(f"| {label} | " + " | ".join(values) + " |\n")

        f.write("\n## CSS Migration\n\n")
        f.write("| Agent | Upgrades | Downgrades | Same |\n")
        f.write("|-------|----------|------------|------|\n")
        for name in agent_names:
            if name in results:
                m = results[name]["css_migration"]
                f.write(f"| {name} | {m['up']} | {m['down']} | {m['same']} |\n")

        f.write("\n## Churn Rate by CSS Tier\n\n")
        f.write("| CSS Tier | " + " | ".join(agent_names) + " |\n")
        f.write("|----------|" + "|".join(["------" for _ in agent_names]) + "|\n")
        for css in range(1, 6):
            values = []
            for name in agent_names:
                if name in results:
                    rate = results[name]["churn_rate_by_css"].get(str(css), results[name]["churn_rate_by_css"].get(css, 0.0))
                    values.append(f"{rate:.3f}")
                else:
                    values.append("N/A")
            f.write(f"| CSS {css} | " + " | ".join(values) + " |\n")

        # Category metrics
        f.write("\n## Category-Level Metrics\n\n")
        for name in agent_names:
            if name in results and "category_metrics" in results[name]:
                f.write(f"\n### {name}\n\n")
                f.write("| Category | Mean Margin | Discount Share | Actions | Entropy |\n")
                f.write("|----------|------------|----------------|---------|--------|\n")
                for cat, cm in sorted(results[name]["category_metrics"].items()):
                    f.write(f"| {cat} | {cm['mean_margin']:.1%} | {cm['discount_share']:.1%} "
                            f"| {cm['n_actions']} | {cm['action_entropy']:.3f} |\n")

        f.write("\n## Recommendations\n\n")
        best_agent = max(
            [(name, results[name]["mean_episode_margin"]) for name in agent_names if name in results],
            key=lambda x: x[1],
        )
        heuristic_margin = results.get("heuristic", {}).get("mean_episode_margin", 0)
        if best_agent[0] != "heuristic" and heuristic_margin > 0:
            pct_improvement = ((best_agent[1] - heuristic_margin) / abs(heuristic_margin)) * 100
            f.write(f"- **{best_agent[0].upper()}** achieves the highest mean episode margin "
                    f"(${best_agent[1]:,.2f}), a {pct_improvement:+.1f}% difference vs heuristic baseline.\n")
        f.write(f"- Heuristic baseline mean episode margin: ${heuristic_margin:,.2f}\n")

    print(f"JSON report saved to {report_path}")
    print(f"Markdown report saved to {md_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL pricing agents")
    parser.add_argument("--agents", type=str, help="Comma-separated agent names")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--ab-test", action="store_true")
    parser.add_argument("--treatment", type=str, default="ppo")
    parser.add_argument("--control", type=str, default="heuristic")
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--generate-report", action="store_true")
    parser.add_argument("--output", type=str, default="results/")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--legacy", action="store_true", help="Use legacy 17-dim mode")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.ab_test:
        run_ab_test(args, config)
    elif args.generate_report:
        generate_report(args, config)
    elif args.agents:
        compare_agents(args, config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
