import numpy as np
from src.evaluation.metrics import (
    compute_portfolio_margin,
    compute_churn_rate_by_css,
    compute_css_migration,
    compute_action_entropy,
    compute_regret_vs_oracle,
)


def test_portfolio_margin():
    margins = [100.0, 200.0, 300.0]
    assert compute_portfolio_margin(margins, periods=52) == sum(margins) * 52


def test_portfolio_margin_empty():
    assert compute_portfolio_margin([], periods=52) == 0.0


def test_churn_rate_by_css():
    css_scores = [1, 1, 2, 3, 3, 3, 4, 5]
    churned = [True, False, True, False, False, True, False, False]
    rates = compute_churn_rate_by_css(css_scores, churned)
    assert rates[1] == 0.5  # 1 of 2 churned
    assert rates[2] == 1.0  # 1 of 1 churned
    assert rates[5] == 0.0


def test_css_migration():
    initial = [1, 2, 3, 4, 5]
    final = [2, 2, 4, 4, 5]
    up, down, same = compute_css_migration(initial, final)
    assert up == 2  # css 1->2, css 3->4
    assert down == 0
    assert same == 3


def test_css_migration_downgrade():
    initial = [3, 4, 5]
    final = [2, 3, 4]
    up, down, same = compute_css_migration(initial, final)
    assert up == 0
    assert down == 3
    assert same == 0


def test_action_entropy():
    # Uniform distribution -> high entropy
    actions_uniform = list(range(7)) * 100
    entropy_uniform = compute_action_entropy(actions_uniform, n_actions=7)

    # Collapsed to single action -> zero entropy
    actions_collapsed = [0] * 700
    entropy_collapsed = compute_action_entropy(actions_collapsed, n_actions=7)

    assert entropy_uniform > entropy_collapsed
    assert entropy_collapsed < 0.01
    assert entropy_uniform > 0.99  # Should be very close to 1.0


def test_action_entropy_empty():
    assert compute_action_entropy([], n_actions=7) == 0.0


def test_regret_vs_oracle():
    agent_rewards = [1.0, 2.0, 3.0]
    oracle_rewards = [2.0, 3.0, 4.0]
    regret = compute_regret_vs_oracle(agent_rewards, oracle_rewards)
    assert regret == 3.0  # (2+3+4) - (1+2+3) = 3


def test_regret_vs_oracle_perfect():
    rewards = [1.0, 2.0, 3.0]
    regret = compute_regret_vs_oracle(rewards, rewards)
    assert regret == 0.0
