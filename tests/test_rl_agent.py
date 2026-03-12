from src.agent.rl_agent import RLAgent
from src.environment.pricing_env import DynamicPricingEnv


def test_rl_agent_creation_ppo():
    agent = RLAgent(algorithm="ppo")
    assert agent.algorithm == "ppo"


def test_rl_agent_creation_dqn():
    agent = RLAgent(algorithm="dqn")
    assert agent.algorithm == "dqn"


def test_rl_agent_predict_shape():
    env = DynamicPricingEnv()
    agent = RLAgent(algorithm="ppo", env=env)
    obs, _ = env.reset()
    action = agent.predict(obs)
    assert 0 <= action < 7


def test_rl_agent_predict_dqn():
    env = DynamicPricingEnv()
    agent = RLAgent(algorithm="dqn", env=env)
    obs, _ = env.reset()
    action = agent.predict(obs)
    assert 0 <= action < 7


def test_rl_agent_train_short(tmp_path):
    env = DynamicPricingEnv()
    agent = RLAgent(algorithm="ppo", env=env)
    agent.train(total_timesteps=256)
    # Verify model exists after training
    assert agent.model is not None


def test_rl_agent_save_load(tmp_path):
    env = DynamicPricingEnv()
    agent = RLAgent(algorithm="ppo", env=env)
    agent.train(total_timesteps=256)

    save_path = str(tmp_path / "test_model")
    agent.save(save_path)

    # Load into new agent
    agent2 = RLAgent(algorithm="ppo", model_path=save_path, env=env)
    obs, _ = env.reset()
    action = agent2.predict(obs)
    assert 0 <= action < 7
