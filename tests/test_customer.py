import numpy as np
from src.environment.customer import CustomerState


def test_customer_state_creation():
    cs = CustomerState(
        css_score=3,
        performance_percentile=0.65,
        potential_tier=1,
        current_margin_rate=0.24,
        current_margin_dollars=665.0,
        weekly_cases=15.0,
        weekly_sales=750.0,
        deliveries_per_week=2.5,
        elasticity_estimate=-1.5,
        price_change_history=[0, 0, 0, 0],
        periods_since_last_change=0,
        syw_flag=True,
        perks_flag=False,
        churn_probability=0.10,
    )
    assert cs.css_score == 3
    assert cs.syw_flag is True


def test_customer_to_observation():
    cs = CustomerState(
        css_score=3,
        performance_percentile=0.65,
        potential_tier=1,
        current_margin_rate=0.24,
        current_margin_dollars=665.0,
        weekly_cases=15.0,
        weekly_sales=750.0,
        deliveries_per_week=2.5,
        elasticity_estimate=-1.5,
        price_change_history=[0, 0, 0, 0],
        periods_since_last_change=0,
        syw_flag=True,
        perks_flag=False,
        churn_probability=0.10,
    )
    obs = cs.to_observation()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (17,)
    assert obs.dtype == np.float32
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)


def test_customer_from_observation_roundtrip():
    cs = CustomerState(
        css_score=3,
        performance_percentile=0.65,
        potential_tier=1,
        current_margin_rate=0.24,
        current_margin_dollars=665.0,
        weekly_cases=15.0,
        weekly_sales=750.0,
        deliveries_per_week=2.5,
        elasticity_estimate=-1.5,
        price_change_history=[0, 0, 0, 0],
        periods_since_last_change=0,
        syw_flag=True,
        perks_flag=False,
        churn_probability=0.10,
    )
    obs = cs.to_observation()
    cs2 = CustomerState.from_observation(obs)
    assert cs2.css_score == cs.css_score
    assert cs2.syw_flag == cs.syw_flag


def test_observation_all_zeros():
    """Edge case: minimal customer state."""
    cs = CustomerState(
        css_score=1,
        performance_percentile=0.0,
        potential_tier=0,
        current_margin_rate=0.0,
        current_margin_dollars=0.0,
        weekly_cases=0.0,
        weekly_sales=0.0,
        deliveries_per_week=0.0,
        elasticity_estimate=0.0,
        price_change_history=[0, 0, 0, 0],
        periods_since_last_change=0,
        syw_flag=False,
        perks_flag=False,
        churn_probability=0.0,
    )
    obs = cs.to_observation()
    assert obs.shape == (17,)
    assert np.all(obs >= 0.0)


def test_observation_max_values():
    """Edge case: high-value customer at normalization limits."""
    cs = CustomerState(
        css_score=5,
        performance_percentile=1.0,
        potential_tier=2,
        current_margin_rate=0.60,
        current_margin_dollars=5000.0,
        weekly_cases=200.0,
        weekly_sales=15000.0,
        deliveries_per_week=7.0,
        elasticity_estimate=-4.0,
        price_change_history=[6, 6, 6, 6],
        periods_since_last_change=52,
        syw_flag=True,
        perks_flag=True,
        churn_probability=1.0,
    )
    obs = cs.to_observation()
    assert obs.shape == (17,)
    assert np.all(obs <= 1.0)
