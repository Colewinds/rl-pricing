"""Evaluation script for comparing pricing agents.

Usage:
    python scripts/evaluate.py --agents ppo,heuristic --episodes 100
    python scripts/evaluate.py --ab-test --treatment ppo --control heuristic --simulations 100
    python scripts/evaluate.py --generate-report --output results/
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from src.agent.heuristic_baseline import HeuristicBaseline
from src.agent.rl_agent import RLAgent
from src.environment.customer import CustomerState
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


def get_agent(agent_name: str, env, model_path: str | None = None):
    """Instantiate an agent by name."""
    if agent_name == "heuristic":
        return HeuristicBaseline()
    elif agent_name in ("ppo", "dqn"):
        if model_path:
            return RLAgent(algorithm=agent_name, model_path=model_path, env=env)
        else:
            # Untrained agent for testing
            return RLAgent(algorithm=agent_name, env=env)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


def run_evaluation(agent, env, episodes: int) -> dict:
    """Run agent through multiple episodes and collect metrics."""
    all_rewards = []
    all_actions = []
    all_margins = []
    css_scores = []
    churned = []
    initial_css = []
    final_css = []

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
            # Get action
            if isinstance(agent, HeuristicBaseline):
                cs = CustomerState.from_observation(obs)
                if ep_initial_css is None:
                    ep_initial_css = cs.css_score
                action = agent.predict(cs)
            else:
                if ep_initial_css is None:
                    cs = CustomerState.from_observation(obs)
                    ep_initial_css = cs.css_score
                action = agent.predict(obs)

            obs, reward, terminated, truncated, step_info = env.step(action)
            ep_rewards.append(reward)
            ep_actions.append(action)
            ep_margin += step_info.get("margin_dollars", 0.0)

            if terminated:
                ep_churned = step_info.get("churned", False)

            done = terminated or truncated

        cs_final = CustomerState.from_observation(obs)
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

    return {
        "mean_reward": float(np.mean(all_rewards)),
        "total_reward": float(np.sum(all_rewards)),
        "mean_episode_margin": float(np.mean(all_margins)),
        "action_entropy": compute_action_entropy(all_actions),
        "churn_rate_by_css": compute_churn_rate_by_css(css_scores, churned),
        "css_migration": {"up": up, "down": down, "same": same},
        "n_episodes": episodes,
        "n_steps": len(all_rewards),
    }


def compare_agents(args, config):
    """Compare multiple agents side by side."""
    agent_names = args.agents.split(",")
    reward_fn = CLVOptimizer(config.get("reward", {}).get("clv_optimizer"))

    print(f"Evaluating agents: {agent_names} over {args.episodes} episodes each\n")

    results = {}
    for name in agent_names:
        env = DynamicPricingEnv(config=config, reward_fn=reward_fn)
        agent = get_agent(name.strip(), env, args.model_path)
        metrics = run_evaluation(agent, env, args.episodes)
        results[name.strip()] = metrics

        print(f"--- {name.strip()} ---")
        print(f"  Mean reward:         {metrics['mean_reward']:.4f}")
        print(f"  Mean episode margin: ${metrics['mean_episode_margin']:.2f}")
        print(f"  Action entropy:      {metrics['action_entropy']:.3f}")
        print(f"  CSS migration:       +{metrics['css_migration']['up']} "
              f"-{metrics['css_migration']['down']} "
              f"={metrics['css_migration']['same']}")
        print()

    return results


def run_ab_test(args, config):
    """Run A/B test between treatment and control agents."""
    reward_fn = CLVOptimizer(config.get("reward", {}).get("clv_optimizer"))
    env = DynamicPricingEnv(config=config, reward_fn=reward_fn)

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
    """Generate a comprehensive evaluation report."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    reward_fn = CLVOptimizer(config.get("reward", {}).get("clv_optimizer"))

    # Run all agents
    agents = ["heuristic"]
    results = {}
    for name in agents:
        env = DynamicPricingEnv(config=config, reward_fn=reward_fn)
        agent = get_agent(name, env)
        results[name] = run_evaluation(agent, env, episodes=50)

    # Write report
    report_path = output_dir / f"evaluation_report_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Report saved to {report_path}")
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
