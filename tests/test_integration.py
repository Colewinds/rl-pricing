"""End-to-end integration test: generate data, create env, train briefly, evaluate."""

import numpy as np


def test_full_pipeline():
    """Integration test covering the complete pipeline."""
    from src.data.synthetic_generator import generate_customer_population
    from src.environment.pricing_env import DynamicPricingEnv
    from src.reward.reward_functions import CLVOptimizer
    from src.agent.rl_agent import RLAgent
    from src.evaluation.metrics import compute_portfolio_margin, compute_action_entropy

    # Step 1: Generate small dataset
    customers = generate_customer_population(n=100, seed=42)
    assert len(customers) == 100
    assert "css_score" in customers.columns

    # Step 2: Create environment with CLV reward
    reward_fn = CLVOptimizer()
    env = DynamicPricingEnv(reward_fn=reward_fn)

    # Step 3: Quick train (just verify it runs)
    agent = RLAgent(algorithm="ppo", env=env)
    agent.train(total_timesteps=1000)

    # Step 4: Evaluate over one episode
    obs, _ = env.reset()
    rewards = []
    actions = []
    margins = []
    for _ in range(52):
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        actions.append(action)
        margins.append(info["margin_dollars"])
        if terminated or truncated:
            break

    assert len(rewards) > 0
    assert all(isinstance(r, float) for r in rewards)

    # Step 5: Compute metrics
    entropy = compute_action_entropy(actions)
    assert 0.0 <= entropy <= 1.0

    total_margin = compute_portfolio_margin([np.mean(margins)])
    assert total_margin > 0


def test_heuristic_vs_env():
    """Test that heuristic baseline can run through the environment."""
    from src.agent.heuristic_baseline import HeuristicBaseline
    from src.environment.customer import CustomerState
    from src.environment.pricing_env import DynamicPricingEnv
    from src.reward.reward_functions import CLVOptimizer

    reward_fn = CLVOptimizer()
    env = DynamicPricingEnv(reward_fn=reward_fn)
    agent = HeuristicBaseline()

    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0

    done = False
    while not done:
        cs = CustomerState.from_observation(obs)
        action = agent.predict(cs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    assert steps > 0
    assert steps <= 52


def test_multi_agent_pipeline():
    """Test multi-agent orchestration through the environment."""
    from src.agent.heuristic_baseline import HeuristicBaseline
    from src.environment.customer import CustomerState
    from src.environment.pricing_env import DynamicPricingEnv
    from src.orchestrator.multi_agent import PortfolioManager, PriceScout, MarginGuardian
    from src.reward.reward_functions import CLVOptimizer

    reward_fn = CLVOptimizer()
    env = DynamicPricingEnv(reward_fn=reward_fn)
    pm = PortfolioManager()
    scout_agent = HeuristicBaseline()  # Stand-in
    guardian_agent = HeuristicBaseline()

    obs, _ = env.reset()
    for step in range(52):
        cs = CustomerState.from_observation(obs)
        agent_id = pm.assign(cs)

        if agent_id == "scout":
            action = scout_agent.predict(cs)
        else:
            action = guardian_agent.predict(cs)

        obs, reward, terminated, truncated, info = env.step(action)
        pm.log_result(agent_id, cs, action, reward)

        if pm.should_reallocate(step):
            pm.update_allocations()

        if terminated or truncated:
            break

    summary = pm.get_performance_summary()
    assert "scout" in summary or "guardian" in summary


def test_ab_test_integration():
    """Test A/B simulator with heuristic agents."""
    from src.agent.heuristic_baseline import HeuristicBaseline
    from src.evaluation.ab_test_simulator import ABTestSimulator

    sim = ABTestSimulator(
        treatment_agent=HeuristicBaseline(),
        control_agent=HeuristicBaseline(),
        env_config={},
        n_simulations=3,
    )
    result = sim.run()
    assert result.p_value >= 0
    assert len(result.treatment_margins) == 3


def test_drift_detector_integration():
    """Test drift detector with real env data."""
    from src.agent.heuristic_baseline import HeuristicBaseline
    from src.environment.customer import CustomerState
    from src.environment.pricing_env import DynamicPricingEnv
    from src.monitoring.drift_detector import DriftDetector
    from src.reward.reward_functions import CLVOptimizer

    reward_fn = CLVOptimizer()
    env = DynamicPricingEnv(reward_fn=reward_fn)
    agent = HeuristicBaseline()
    detector = DriftDetector()

    obs, _ = env.reset()
    done = False
    while not done:
        cs = CustomerState.from_observation(obs)
        action = agent.predict(cs)
        obs, reward, terminated, truncated, info = env.step(action)
        detector.update(reward=reward, action=action)
        done = terminated or truncated

    detector.end_period()
    report = detector.generate_report()
    assert report["period_count"] == 1
