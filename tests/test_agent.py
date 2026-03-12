import numpy as np
from src.agent.heuristic_baseline import HeuristicBaseline
from src.environment.customer import CustomerState


def _make_state(**overrides) -> CustomerState:
    defaults = dict(
        css_score=3, performance_percentile=0.5, potential_tier=1,
        current_margin_rate=0.24, current_margin_dollars=665.0,
        weekly_cases=15.0, weekly_sales=750.0, deliveries_per_week=2.5,
        elasticity_estimate=-1.5, price_change_history=[0, 0, 0, 0],
        periods_since_last_change=4, syw_flag=False, perks_flag=False,
        churn_probability=0.10,
    )
    defaults.update(overrides)
    return CustomerState(**defaults)


def test_heuristic_css1_discounts():
    agent = HeuristicBaseline()
    state = _make_state(css_score=1)
    action = agent.predict(state)
    assert action == 4  # price down 5%


def test_heuristic_css2_discounts():
    agent = HeuristicBaseline()
    state = _make_state(css_score=2)
    action = agent.predict(state)
    assert action == 4  # price down 5%


def test_heuristic_css3_hold():
    agent = HeuristicBaseline()
    state = _make_state(css_score=3, current_margin_rate=0.24)
    action = agent.predict(state)
    assert action == 0  # hold


def test_heuristic_css3_low_margin_prices_up():
    agent = HeuristicBaseline()
    state = _make_state(css_score=3, current_margin_rate=0.18)
    action = agent.predict(state)
    assert action == 1  # price up 2%


def test_heuristic_css5_hold():
    agent = HeuristicBaseline()
    # baseline for CSS 5 is 25.0, so 25.0 is above the 22.5 threshold
    state = _make_state(css_score=5, weekly_cases=25.0)
    action = agent.predict(state)
    assert action == 0  # hold (volume stable)


def test_heuristic_css5_volume_drop():
    agent = HeuristicBaseline()
    # Simulate volume drop >10% from baseline by setting low volume
    state = _make_state(css_score=5, weekly_cases=10.0,
                        performance_percentile=0.9)
    action = agent.predict(state)
    assert action == 3  # price down 2%


def test_heuristic_css4_hold():
    agent = HeuristicBaseline()
    state = _make_state(css_score=4, weekly_cases=20.0)
    action = agent.predict(state)
    assert action == 0  # hold
