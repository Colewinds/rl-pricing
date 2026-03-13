"""Training entry point for RL pricing agents.

Usage:
    python scripts/train.py --agent ppo --reward clv_optimizer --timesteps 500000
    python scripts/train.py --agent dqn --reward clv_optimizer --timesteps 500000
    python scripts/train.py --agent heuristic --reward clv_optimizer --timesteps 52000
    python scripts/train.py --agent multi --reward portfolio_optimizer --timesteps 500000
    python scripts/train.py --agent ppo --reward clv_optimizer --legacy  # 17-dim mode
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


def train_agent(
    algorithm: str,
    config: dict,
    reward_fn,
    timesteps: int,
    model_dir: str,
    legacy_mode: bool = False,
) -> RLAgent:
    """Train an RL agent and save to model_dir. Importable for use in pipeline.

    Args:
        algorithm: "ppo" or "dqn".
        config: Full config dict.
        reward_fn: Reward function instance.
        timesteps: Training timesteps.
        model_dir: Directory to save the model.
        legacy_mode: If True, use 17-dim customer-level env.

    Returns:
        Trained RLAgent.
    """
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    env = Monitor(DynamicPricingEnv(config=config, reward_fn=reward_fn, legacy_mode=legacy_mode))
    eval_env = Monitor(DynamicPricingEnv(config=config, reward_fn=reward_fn, legacy_mode=legacy_mode))

    algo_config = config.get("training", {}).get(algorithm, {})
    agent = RLAgent(
        algorithm=algorithm,
        env=env,
        config={"training": {algorithm: algo_config}},
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir_path),
        log_path=str(model_dir_path),
        eval_freq=config["training"].get("eval_freq", 10000),
        deterministic=True,
        render=False,
    )

    agent.train(total_timesteps=timesteps, callback=eval_callback)
    agent.save(str(model_dir_path / "final_model"))
    return agent


def train_single_agent(args, config):
    """Train a single RL agent or run heuristic baseline."""
    run_id = f"{args.agent}_{args.reward}_{datetime.now():%Y%m%d_%H%M%S}"
    log_dir = Path(config["training"]["log_dir"]) / run_id
    model_dir = Path(config["training"]["model_dir"]) / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    reward_cfg = config.get("reward", {}).get(args.reward, {})
    reward_fn = REWARD_FUNCTIONS[args.reward](reward_cfg)

    legacy_mode = getattr(args, "legacy", False)

    if args.agent == "heuristic":
        env = Monitor(DynamicPricingEnv(config=config, reward_fn=reward_fn, legacy_mode=legacy_mode))
        agent = HeuristicBaseline(config.get("multi_agent", {}).get("heuristic"))
        print(f"Running heuristic baseline for {args.timesteps} steps...")
        obs, _ = env.reset()
        total_reward = 0.0
        for step in range(args.timesteps):
            cs = CustomerState.from_observation(obs)
            action = agent.predict(cs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                obs, _ = env.reset()
        print(f"Heuristic total reward: {total_reward:.2f}")
        return

    print(f"Training {args.agent} with {args.reward} for {args.timesteps} timesteps...")
    mode_str = "legacy 17-dim" if legacy_mode else "item-level 33-dim"
    print(f"  Mode: {mode_str}")
    agent = train_agent(
        algorithm=args.agent,
        config=config,
        reward_fn=reward_fn,
        timesteps=args.timesteps,
        model_dir=str(model_dir),
        legacy_mode=legacy_mode,
    )
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
    parser.add_argument("--legacy", action="store_true", help="Use legacy 17-dim customer-level env")
    args = parser.parse_args()

    config = load_config(args.config, args.scenario)

    if args.agent == "multi":
        print("Multi-agent training: train individual agents first, then orchestrator.")
        print("Use: --agent ppo for Scout, --agent ppo for Guardian with restricted config.")
        return

    train_single_agent(args, config)


if __name__ == "__main__":
    main()
