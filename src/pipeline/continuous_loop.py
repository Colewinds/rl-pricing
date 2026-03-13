"""Continuous learning pipeline: champion/challenger with drift-triggered retraining."""

from datetime import datetime
from pathlib import Path

import numpy as np

from src.environment.pricing_env import DynamicPricingEnv
from src.evaluation.ab_test_simulator import ABTestSimulator
from src.monitoring.drift_detector import DriftDetector
from src.pipeline.model_registry import ModelRegistry
from src.reward.reward_functions import CLVOptimizer


def _train_agent(algorithm, config, reward_fn, timesteps, model_dir, warm_start_path=None):
    """Train an RL agent. Imports here to avoid circular deps."""
    from src.agent.rl_agent import RLAgent
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(DynamicPricingEnv(config=config, reward_fn=reward_fn))
    if warm_start_path:
        agent = RLAgent(algorithm=algorithm, model_path=warm_start_path, env=env)
    else:
        algo_config = config.get("training", {}).get(algorithm, {})
        agent = RLAgent(algorithm=algorithm, env=env, config={"training": {algorithm: algo_config}})

    agent.train(total_timesteps=timesteps)

    save_path = str(Path(model_dir) / f"model_{datetime.now():%Y%m%d_%H%M%S}")
    agent.save(save_path)
    return agent, save_path


class ContinuousLoop:
    """Hybrid scheduled + event-driven continuous learning pipeline.

    Each period:
    1. Evaluate champion model over N episodes
    2. Feed results to DriftDetector
    3. If drift alert OR scheduled retrain interval reached:
       a. Train challenger (optionally warm-started from champion)
       b. A/B test challenger vs champion
       c. If challenger wins (p < threshold, positive delta): promote
       d. Log to registry
    """

    def __init__(self, config: dict, algorithm: str = "ppo"):
        self.config = config
        self.algorithm = algorithm

        loop_cfg = config.get("continuous_loop", {})
        self.retrain_interval = loop_cfg.get("retrain_interval", 4)
        self.retrain_timesteps = loop_cfg.get("retrain_timesteps", 200000)
        self.eval_episodes = loop_cfg.get("eval_episodes_per_period", 50)
        self.ab_simulations = loop_cfg.get("ab_test_simulations", 100)
        self.p_threshold = loop_cfg.get("promotion_p_threshold", 0.05)
        self.warm_start = loop_cfg.get("warm_start", True)

        self.registry = ModelRegistry(
            registry_dir=config.get("training", {}).get("model_dir", "results/models")
        )
        self.drift_detector = DriftDetector(config.get("monitoring", {}))

        reward_cfg = config.get("reward", {}).get("clv_optimizer", {})
        self.reward_fn = CLVOptimizer(reward_cfg)

        self._period = 0
        self._history: list[dict] = []

    def run(self, max_periods: int = 52, callback=None):
        """Run the continuous loop for up to max_periods.

        Args:
            max_periods: Maximum number of evaluation periods.
            callback: Optional function called each period with (period, info).

        Returns:
            List of per-period info dicts.
        """
        for period in range(max_periods):
            self._period = period
            info = self._run_period()
            self._history.append(info)

            if callback:
                callback(period, info)

        return self._history

    def _run_period(self) -> dict:
        """Execute one period of the continuous loop."""
        info = {"period": self._period, "retrained": False, "promoted": False}

        # 1. Evaluate champion
        champion = self.registry.get_champion()
        eval_metrics = self._evaluate_model(champion)
        info["eval_metrics"] = eval_metrics

        # 2. Feed to drift detector
        for reward in eval_metrics.get("episode_rewards", []):
            self.drift_detector.update(reward=reward, action=0)
        self.drift_detector.end_period()
        alerts = self.drift_detector.check_alerts()
        info["alerts"] = alerts

        # 3. Check if retrain needed
        should_retrain = (
            alerts.get("any_alert", False)
            or (self._period > 0 and self._period % self.retrain_interval == 0)
        )

        if should_retrain:
            info["retrained"] = True
            model_dir = self.config.get("training", {}).get("model_dir", "results/models")

            # Train challenger
            warm_path = champion.model_path if (champion and self.warm_start) else None
            _agent, challenger_path = _train_agent(
                algorithm=self.algorithm,
                config=self.config,
                reward_fn=self.reward_fn,
                timesteps=self.retrain_timesteps,
                model_dir=model_dir,
                warm_start_path=warm_path,
            )

            # Register challenger
            challenger_metrics = self._evaluate_model_from_path(challenger_path)
            challenger_version = self.registry.register_model(
                model_path=challenger_path,
                algorithm=self.algorithm,
                training_timesteps=self.retrain_timesteps,
                eval_metrics=challenger_metrics,
            )

            # A/B test
            if champion:
                ab_result = self._ab_test(champion.model_path, challenger_path)
                info["ab_result"] = {
                    "mean_delta": ab_result.mean_delta_margin,
                    "p_value": ab_result.p_value,
                    "ci_95": ab_result.ci_95,
                }

                # Promote if significant improvement
                if ab_result.p_value < self.p_threshold and ab_result.mean_delta_margin > 0:
                    self.registry.promote_to_champion(challenger_version.version_id)
                    info["promoted"] = True
            else:
                # No champion yet, auto-promote
                self.registry.promote_to_champion(challenger_version.version_id)
                info["promoted"] = True

        return info

    def _evaluate_model(self, model_version) -> dict:
        """Evaluate a model version over eval_episodes."""
        if model_version is None:
            return {"mean_reward": 0.0, "episode_rewards": [0.0] * self.eval_episodes}
        return self._evaluate_model_from_path(model_version.model_path)

    def _evaluate_model_from_path(self, model_path: str) -> dict:
        """Evaluate a model from a file path."""
        from src.agent.rl_agent import RLAgent

        env = DynamicPricingEnv(config=self.config, reward_fn=self.reward_fn)
        try:
            agent = RLAgent(algorithm=self.algorithm, model_path=model_path, env=env)
        except Exception:
            agent = RLAgent(algorithm=self.algorithm, env=env)

        episode_rewards = []
        for _ in range(self.eval_episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action = agent.predict(obs)
                obs, reward, terminated, truncated, _info = env.step(action)
                total_reward += reward
                done = terminated or truncated
            episode_rewards.append(total_reward)

        return {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "episode_rewards": episode_rewards,
        }

    def _ab_test(self, champion_path: str, challenger_path: str):
        """Run A/B test between champion and challenger."""
        from src.agent.rl_agent import RLAgent

        env = DynamicPricingEnv(config=self.config, reward_fn=self.reward_fn)
        champion_agent = RLAgent(algorithm=self.algorithm, model_path=champion_path, env=env)
        challenger_agent = RLAgent(algorithm=self.algorithm, model_path=challenger_path, env=env)

        sim = ABTestSimulator(
            treatment_agent=challenger_agent,
            control_agent=champion_agent,
            env_config=self.config,
            n_simulations=self.ab_simulations,
        )
        return sim.run()
