"""RL agent wrapper for SB3 MaskablePPO and DQN."""

from pathlib import Path

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN


class RLAgent:
    """Unified wrapper for RL algorithms with action masking support.

    Supports MaskablePPO (with action masks) and DQN (without masks).
    Provides predict(), train(), save(), and load() interface.
    """

    ALGORITHMS = {
        "ppo": MaskablePPO,
        "dqn": DQN,
    }

    def __init__(
        self,
        algorithm: str = "ppo",
        env: gym.Env | None = None,
        config: dict | None = None,
        model_path: str | None = None,
    ):
        self.algorithm = algorithm
        self.config = config or {}
        algo_config = self.config.get("training", {}).get(algorithm, {})

        if model_path:
            cls = self.ALGORITHMS[algorithm]
            self.model = cls.load(model_path, env=env)
        elif env is not None:
            cls = self.ALGORITHMS[algorithm]
            self.model = cls(
                "MlpPolicy",
                env,
                verbose=0,
                **algo_config,
            )
        else:
            self.model = None

    def predict(
        self, obs: np.ndarray, action_masks: np.ndarray | None = None
    ) -> int:
        """Select an action given an observation.

        Args:
            obs: Observation vector.
            action_masks: Optional binary mask for valid actions (PPO only).

        Returns:
            Selected action index.
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Provide env or model_path.")

        if self.algorithm == "ppo" and action_masks is not None:
            action, _ = self.model.predict(obs, action_masks=action_masks)
        else:
            action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def train(self, total_timesteps: int, **kwargs):
        """Train the model for the specified number of timesteps."""
        if self.model is None:
            raise RuntimeError("No model to train. Provide env in constructor.")
        self.model.learn(total_timesteps=total_timesteps, **kwargs)

    def save(self, path: str):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    def load(self, path: str, env: gym.Env | None = None):
        """Load model from disk."""
        cls = self.ALGORITHMS[self.algorithm]
        self.model = cls.load(path, env=env)
