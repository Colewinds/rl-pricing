import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from src.environment.pricing_env import DynamicPricingEnv


def test_env_creation():
    env = DynamicPricingEnv()
    assert env is not None


def test_gymnasium_compliance():
    env = DynamicPricingEnv()
    check_env(env, skip_render_check=True)


def test_observation_space_shape():
    env = DynamicPricingEnv()
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)


def test_action_space():
    env = DynamicPricingEnv()
    assert env.action_space.n == 7


def test_episode_length():
    env = DynamicPricingEnv()
    obs, _ = env.reset()
    steps = 0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    assert steps <= 52  # max episode length


def test_action_masking():
    env = DynamicPricingEnv()
    env.reset()
    # Take action 6 (15% cut - a deep cut)
    env.step(6)
    mask = env.action_masks()
    # no_consecutive_deep_cuts: actions 5 and 6 should be masked
    assert mask[5] == 0 or mask[6] == 0  # at least deep cuts masked


def test_observation_lag():
    env = DynamicPricingEnv(config={"environment": {"observation_lag": 2}})
    obs0, _ = env.reset()
    obs1, _, _, _, info = env.step(3)  # price down 2%
    # With lag=2, env should still run without error
    assert obs1.shape == obs0.shape


def test_observation_lag_zero():
    env = DynamicPricingEnv(config={"environment": {"observation_lag": 0}})
    obs0, _ = env.reset()
    obs1, _, _, _, _ = env.step(3)
    assert obs1.shape == obs0.shape


def test_reward_is_float():
    env = DynamicPricingEnv()
    env.reset()
    _, reward, _, _, _ = env.step(0)
    assert isinstance(reward, float)


def test_info_contains_customer_state():
    env = DynamicPricingEnv()
    env.reset()
    _, _, _, _, info = env.step(0)
    assert "customer_state" in info
    assert "margin_dollars" in info


def test_reset_produces_valid_obs():
    env = DynamicPricingEnv()
    for _ in range(5):
        obs, info = env.reset()
        assert env.observation_space.contains(obs), f"Obs out of bounds: {obs}"


def test_multiple_episodes():
    env = DynamicPricingEnv()
    for _ in range(3):
        obs, _ = env.reset()
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            done = term or trunc
