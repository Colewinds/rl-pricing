"""Evaluation metrics for portfolio, churn, migration, and action analysis."""

from collections import defaultdict

import numpy as np


def compute_portfolio_margin(
    weekly_margins: list[float], periods: int = 52
) -> float:
    """Compute total annualized portfolio margin.

    Args:
        weekly_margins: List of weekly margin dollar values per customer.
        periods: Number of periods to annualize over.

    Returns:
        Total portfolio margin (sum of weekly margins * periods).
    """
    return sum(weekly_margins) * periods


def compute_churn_rate_by_css(
    css_scores: list[int], churned: list[bool]
) -> dict[int, float]:
    """Compute churn rate broken down by CSS tier.

    Args:
        css_scores: CSS score for each customer.
        churned: Whether each customer churned.

    Returns:
        Dict mapping CSS score -> churn rate.
    """
    counts: dict[int, int] = defaultdict(int)
    churned_counts: dict[int, int] = defaultdict(int)

    for css, did_churn in zip(css_scores, churned):
        counts[css] += 1
        if did_churn:
            churned_counts[css] += 1

    return {
        css: churned_counts[css] / counts[css] if counts[css] > 0 else 0.0
        for css in sorted(counts.keys())
    }


def compute_css_migration(
    initial_css: list[int], final_css: list[int]
) -> tuple[int, int, int]:
    """Count CSS score migrations.

    Args:
        initial_css: Starting CSS scores.
        final_css: Ending CSS scores.

    Returns:
        Tuple of (upgrades, downgrades, same).
    """
    up = 0
    down = 0
    same = 0
    for i, f in zip(initial_css, final_css):
        if f > i:
            up += 1
        elif f < i:
            down += 1
        else:
            same += 1
    return up, down, same


def compute_action_entropy(
    actions: list[int], n_actions: int = 7
) -> float:
    """Compute Shannon entropy of the action distribution.

    High entropy = diverse actions. Low entropy = collapsed to few actions.

    Args:
        actions: List of action indices.
        n_actions: Total number of possible actions.

    Returns:
        Normalized entropy in [0, 1]. 1 = uniform, 0 = single action.
    """
    if not actions:
        return 0.0

    counts = np.zeros(n_actions)
    for a in actions:
        if 0 <= a < n_actions:
            counts[a] += 1

    probs = counts / counts.sum()
    # Filter out zeros to avoid log(0)
    probs = probs[probs > 0]

    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(n_actions)

    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def compute_regret_vs_oracle(
    agent_rewards: list[float],
    oracle_rewards: list[float],
) -> float:
    """Compute cumulative regret vs an oracle (best hindsight) policy.

    Args:
        agent_rewards: Rewards obtained by the agent.
        oracle_rewards: Rewards from the oracle/best policy.

    Returns:
        Cumulative regret (oracle_total - agent_total).
    """
    return sum(oracle_rewards) - sum(agent_rewards)
