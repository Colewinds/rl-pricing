from src.reward.reward_functions import MarginMaximizer, CLVOptimizer, PortfolioOptimizer
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


# --- Margin Maximizer ---

def test_margin_maximizer_positive():
    rf = MarginMaximizer()
    state = _make_state(current_margin_dollars=665.0)
    next_state = _make_state(current_margin_dollars=700.0)
    r = rf.compute(state, 1, next_state)
    assert r > 0  # margin went up


def test_margin_maximizer_negative():
    rf = MarginMaximizer()
    state = _make_state(current_margin_dollars=665.0)
    next_state = _make_state(current_margin_dollars=600.0)
    r = rf.compute(state, 5, next_state)
    assert r < 0  # margin went down


def test_margin_maximizer_zero():
    rf = MarginMaximizer()
    state = _make_state(current_margin_dollars=665.0)
    next_state = _make_state(current_margin_dollars=665.0)
    r = rf.compute(state, 0, next_state)
    assert r == 0.0


# --- CLV Optimizer ---

def test_clv_optimizer_churn_penalty():
    rf = CLVOptimizer()
    state = _make_state(css_score=3, churn_probability=0.10)
    next_state = _make_state(css_score=3, churn_probability=0.30,
                              current_margin_dollars=665.0)
    r = rf.compute(state, 5, next_state)
    r_no_churn = rf.compute(state, 0, _make_state(css_score=3, churn_probability=0.10,
                                                     current_margin_dollars=665.0))
    assert r < r_no_churn


def test_clv_optimizer_volatility_penalty():
    rf = CLVOptimizer()
    # 3 changes in last 4 periods -> volatility penalty
    state = _make_state(price_change_history=[3, 1, 4, 0])
    next_state = _make_state(price_change_history=[3, 1, 4, 0],
                              current_margin_dollars=665.0)
    r_volatile = rf.compute(state, 3, next_state)

    # 0 changes in last 4 periods -> no volatility penalty
    state_stable = _make_state(price_change_history=[0, 0, 0, 0])
    next_stable = _make_state(price_change_history=[0, 0, 0, 0],
                               current_margin_dollars=665.0)
    r_stable = rf.compute(state_stable, 0, next_stable)
    assert r_volatile < r_stable


def test_clv_optimizer_css5_protects_margin():
    rf = CLVOptimizer()
    state = _make_state(css_score=5, current_margin_dollars=1000.0)
    next_state = _make_state(css_score=5, current_margin_dollars=1050.0)
    r5 = rf.compute(state, 1, next_state)

    state1 = _make_state(css_score=1, current_margin_dollars=1000.0)
    next1 = _make_state(css_score=1, current_margin_dollars=1050.0)
    r1 = rf.compute(state1, 1, next1)

    # Same margin gain but CSS 5 should value it more (higher alpha)
    assert r5 > r1


# --- Explain ---

def test_explain_returns_string():
    rf = CLVOptimizer()
    state = _make_state()
    next_state = _make_state(current_margin_dollars=700.0)
    explanation = rf.explain(state, 1, next_state)
    assert isinstance(explanation, str)
    assert "Margin" in explanation or "margin" in explanation


def test_margin_maximizer_explain():
    rf = MarginMaximizer()
    state = _make_state(current_margin_dollars=665.0)
    next_state = _make_state(current_margin_dollars=700.0)
    explanation = rf.explain(state, 1, next_state)
    assert isinstance(explanation, str)


# --- Portfolio Optimizer ---

def test_portfolio_optimizer_extends_clv():
    rf = PortfolioOptimizer()
    state = _make_state()
    next_state = _make_state(current_margin_dollars=700.0)
    r = rf.compute(state, 1, next_state)
    assert isinstance(r, float)


def test_portfolio_optimizer_migration_bonus():
    rf = PortfolioOptimizer()
    state = _make_state(css_score=3)
    # CSS improved from 3 to 4
    next_state = _make_state(css_score=4, current_margin_dollars=665.0)
    r_migrated = rf.compute(state, 0, next_state)

    # CSS stayed at 3
    next_same = _make_state(css_score=3, current_margin_dollars=665.0)
    r_same = rf.compute(state, 0, next_same)

    assert r_migrated > r_same
