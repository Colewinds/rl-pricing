import numpy as np
from src.environment.market_simulator import MarketSimulator


def test_volume_response_basic():
    sim = MarketSimulator(seed=42)
    # 5% price cut with elasticity -1.5 should increase volume
    delta_vol = sim.compute_volume_response(
        price_change=-0.05, elasticity=-1.5, base_volume=100.0, css_score=3
    )
    assert delta_vol > 0  # price down -> volume up


def test_volume_response_price_up():
    sim = MarketSimulator(seed=42)
    delta_vol = sim.compute_volume_response(
        price_change=0.05, elasticity=-1.5, base_volume=100.0, css_score=3
    )
    assert delta_vol < 0  # price up -> volume down


def test_churn_probability():
    sim = MarketSimulator(seed=42)
    # Margin well above threshold -> low churn
    low_churn = sim.compute_churn_probability(
        margin_rate=0.30, css_score=3, syw=True,
        periods_stable=10, threshold=0.25
    )
    # Margin below threshold -> high churn
    high_churn = sim.compute_churn_probability(
        margin_rate=0.15, css_score=3, syw=False,
        periods_stable=2, threshold=0.25
    )
    assert high_churn > low_churn


def test_syw_reduces_churn():
    sim = MarketSimulator(seed=42)
    churn_no_syw = sim.compute_churn_probability(
        margin_rate=0.20, css_score=3, syw=False,
        periods_stable=5, threshold=0.25
    )
    churn_syw = sim.compute_churn_probability(
        margin_rate=0.20, css_score=3, syw=True,
        periods_stable=5, threshold=0.25
    )
    assert churn_syw < churn_no_syw


def test_stickiness_reduces_sensitivity():
    """Average over many trials to overcome noise."""
    n_trials = 500
    short_responses = []
    long_responses = []
    for i in range(n_trials):
        sim = MarketSimulator(seed=i)
        short_responses.append(abs(sim.compute_volume_response(
            price_change=0.05, elasticity=-1.5, base_volume=100.0,
            css_score=3, periods_stable=2
        )))
        sim2 = MarketSimulator(seed=i)
        long_responses.append(abs(sim2.compute_volume_response(
            price_change=0.05, elasticity=-1.5, base_volume=100.0,
            css_score=3, periods_stable=12
        )))
    assert np.mean(long_responses) < np.mean(short_responses)


def test_check_churn_deterministic():
    sim = MarketSimulator(seed=42)
    # With probability 0 -> never churn
    assert sim.check_churn(0.0) is False
    # With probability 1 -> always churn
    assert sim.check_churn(1.0) is True


def test_apply_seasonality():
    sim = MarketSimulator(seed=42)
    config = {
        "q1_modifier": 0.85,
        "q2_modifier": 1.0,
        "q3_modifier": 1.05,
        "q4_modifier": 1.15,
    }
    base = 100.0
    q1_val = sim.apply_seasonality(base, period=5, config=config)
    q4_val = sim.apply_seasonality(base, period=45, config=config)
    assert q1_val < base  # Q1 is below baseline
    assert q4_val > base  # Q4 is above baseline
    assert q4_val > q1_val


def test_volume_response_hold():
    """No price change should yield near-zero volume response (just noise)."""
    sim = MarketSimulator(seed=42)
    responses = []
    for _ in range(100):
        r = sim.compute_volume_response(
            price_change=0.0, elasticity=-1.5, base_volume=100.0, css_score=3
        )
        responses.append(r)
    # Mean should be close to zero
    assert abs(np.mean(responses)) < 5.0
