"""Training entry point for RL pricing agents.

Usage:
    python scripts/train.py --agent ppo --reward clv_optimizer --timesteps 500000
    python scripts/train.py --agent dqn --reward clv_optimizer --timesteps 500000
    python scripts/train.py --agent heuristic --reward clv_optimizer --timesteps 52000
    python scripts/train.py --agent multi --reward portfolio_optimizer --timesteps 500000
"""

import argparse
from datetime import datetime
from pathlib import Path

import yaml
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.agent.heuristic_baseline import HeuristicBaseline
from src.agent.rl_agent import RLAgent
from src.environment.customer import CustomerState
from src.environment.pricing_env import DynamicPricingEnv
from src.reward.reward_functions import CLVOptimizer, MarginMaximizer, PortfolioOptimizer

REWARD_FUNCTIONS = {
    "margin_maximizer": MarginMaximizer,
    "clv_optimizer": CLVOptimizer,
    "portfolio_optimizer": PortfolioOptimizer,
}


def load_config(config_path: str, scenario: str | None = None) -> dict:
    """Load config with optional scenario overrides."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if scenario:
        scenario_path = Path(config_path).parent / "scenarios" / f"{scenario}.yaml"
        if scenario_path.exists():
            with open(scenario_path) as f:
                overrides = yaml.safe_load(f) or {}
            _deep_merge(config, overrides)
    return config


def _deep_merge(base: dict, overrides: dict):
    """Recursively merge overrides into base dict."""
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def train_single_agent(args, config):
    """Train a single RL agent or run heuristic baseline."""
    run_id = f"{args.agent}_{args.reward}_{datetime.now():%Y%m%d_%H%M%S}"
    log_dir = Path(config["training"]["log_dir"]) / run_id
    model_dir = Path(config["training"]["model_dir"]) / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    reward_cfg = config.get("reward", {}).get(args.reward, {})
    reward_fn = REWARD_FUNCTIONS[args.reward](reward_cfg)

    env = Monitor(DynamicPricingEnv(config=config, reward_fn=reward_fn))
    eval_env = Monitor(DynamicPricingEnv(config=config, reward_fn=reward_fn))

    if args.agent == "heuristic":
        agent = HeuristicBaseline(config.get("multi_agent", {}).get("heuristic"))
        print(f"Running heuristic baseline for {args.timesteps} steps...")
        obs, _ = eval_env.reset()
        total_reward = 0.0
        for step in range(args.timesteps):
            cs = CustomerState.from_observation(obs)
            action = agent.predict(cs)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            if terminated or truncated:
                obs, _ = eval_env.reset()
        print(f"Heuristic total reward: {total_reward:.2f}")
        return

    algo_config = config.get("training", {}).get(args.agent, {})
    agent = RLAgent(
        algorithm=args.agent,
        env=env,
        config={"training": {args.agent: algo_config}},
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=config["training"]["eval_freq"],
        deterministic=True,
        render=False,
    )

    print(f"Training {args.agent} with {args.reward} for {args.timesteps} timesteps...")
    agent.train(
        total_timesteps=args.timesteps,
        callback=eval_callback,
    )
    agent.save(str(model_dir / "final_model"))
    print(f"Model saved to {model_dir / 'final_model'}")


def main():
    parser = argparse.ArgumentParser(description="Train RL pricing agents")
    parser.add_argument(
        "--agent",
        choices=["ppo", "dqn", "heuristic", "multi"],
        required=True,
    )
    parser.add_argument(
        "--reward",
        choices=list(REWARD_FUNCTIONS.keys()),
        default="clv_optimizer",
    )
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()

    config = load_config(args.config, args.scenario)

    if args.agent == "multi":
        print("Multi-agent training: train individual agents first, then orchestrator.")
        print("Use: --agent ppo for Scout, --agent ppo for Guardian with restricted config.")
        return

    train_single_agent(args, config)


if __name__ == "__main__":
    main()
