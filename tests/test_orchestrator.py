import numpy as np
from src.orchestrator.multi_agent import PortfolioManager, PriceScout, MarginGuardian
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


def test_portfolio_manager_routes_css1_to_scout():
    pm = PortfolioManager()
    state = _make_state(css_score=1)
    agent_id = pm.assign(state)
    assert agent_id == "scout"


def test_portfolio_manager_routes_css2_to_scout():
    pm = PortfolioManager()
    state = _make_state(css_score=2)
    agent_id = pm.assign(state)
    assert agent_id == "scout"


def test_portfolio_manager_routes_css5_to_guardian():
    pm = PortfolioManager()
    state = _make_state(css_score=5)
    agent_id = pm.assign(state)
    assert agent_id == "guardian"


def test_portfolio_manager_routes_css4_to_guardian():
    pm = PortfolioManager()
    state = _make_state(css_score=4)
    agent_id = pm.assign(state)
    assert agent_id == "guardian"


def test_portfolio_manager_routes_css3():
    pm = PortfolioManager()
    state = _make_state(css_score=3)
    agent_id = pm.assign(state)
    assert agent_id in ("scout", "guardian")


def test_portfolio_manager_reallocation():
    pm = PortfolioManager()
    # Log results showing scout doing well on CSS 3
    for _ in range(10):
        pm.log_result("scout", _make_state(css_score=3), action=4, reward=5.0)
        pm.log_result("guardian", _make_state(css_score=3), action=0, reward=1.0)
    pm.update_allocations()
    state = _make_state(css_score=3)
    agent_id = pm.assign(state)
    assert agent_id == "scout"  # scout is performing better on CSS 3


def test_portfolio_manager_should_reallocate():
    pm = PortfolioManager()
    assert pm.should_reallocate(0) is False
    assert pm.should_reallocate(4) is True
    assert pm.should_reallocate(8) is True
    assert pm.should_reallocate(3) is False


def test_price_scout_action_space():
    scout = PriceScout()
    assert scout.allowed_actions == list(range(7))


def test_price_scout_mask():
    scout = PriceScout()
    mask = scout.get_action_mask()
    assert mask == [1, 1, 1, 1, 1, 1, 1]


def test_margin_guardian_restricted_actions():
    guardian = MarginGuardian()
    assert guardian.allowed_actions == [0, 1, 2, 3]


def test_margin_guardian_mask():
    guardian = MarginGuardian()
    mask = guardian.get_action_mask()
    assert mask == [1, 1, 1, 1, 0, 0, 0]


def test_performance_summary():
    pm = PortfolioManager()
    pm.log_result("scout", _make_state(css_score=1), action=4, reward=3.0)
    pm.log_result("guardian", _make_state(css_score=5), action=0, reward=2.0)
    summary = pm.get_performance_summary()
    assert summary["scout"]["n_decisions"] == 1
    assert summary["guardian"]["mean_reward"] == 2.0
