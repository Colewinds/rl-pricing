"""Integration tests for item-level pricing system.

Tests the full pipeline with item-level observations, rewards, routing,
and action masking across DynamicPricingEnv, CustomerItemState,
synthetic generators, CLVOptimizer, and PortfolioManager.
"""

import numpy as np
import pytest


def test_full_episode_item_level():
    """Full episode through DynamicPricingEnv in item mode produces 33-dim obs."""
    from src.environment.pricing_env import DynamicPricingEnv

    env = DynamicPricingEnv(legacy_mode=False)
    obs, info_reset = env.reset(seed=42)

    assert obs.shape == (33,), f"Expected 33-dim obs, got {obs.shape}"

    rewards = []
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (33,)
        assert np.isfinite(reward), f"Reward not finite at step {i}: {reward}"
        assert isinstance(reward, float)
        assert "customer_state" in info
        assert "item_state" in info
        assert info["step"] == i + 1

        rewards.append(reward)
        if terminated or truncated:
            break

    assert len(rewards) > 0


def test_legacy_mode_produces_17_dim():
    """Legacy mode env produces 17-dim obs with customer_state but no item_state."""
    from src.environment.pricing_env import DynamicPricingEnv

    env = DynamicPricingEnv(legacy_mode=True)
    obs, _ = env.reset(seed=99)

    assert obs.shape == (17,), f"Expected 17-dim obs, got {obs.shape}"

    obs, reward, terminated, truncated, info = env.step(0)

    assert obs.shape == (17,)
    assert "customer_state" in info
    assert "item_state" not in info


def test_customer_item_state_serialization_roundtrip():
    """CustomerItemState to_observation/from_observation roundtrip."""
    from src.environment.customer import CustomerItemState
    from src.environment.item import ItemState

    item = ItemState(
        category=2,
        subcategory=1,
        unit_cost=6.0,
        unit_price=9.0,
        item_margin_rate=0.22,
        weekly_units=25.0,
        weekly_revenue=225.0,
        perishability=0.3,
        substitutability=0.6,
        competitive_index=0.55,
        seasonal_index=1.1,
        price_change_history=[0, 3, 0, 1],
        periods_since_last_change=4,
    )

    ci = CustomerItemState(
        css_score=3,
        performance_percentile=0.65,
        potential_tier=1,
        customer_margin_rate=0.24,
        weekly_cases=15.0,
        weekly_sales=800.0,
        deliveries_per_week=2.5,
        concept=1,
        syw_flag=True,
        perks_flag=False,
        churn_probability=0.08,
        current_period=20,
        customer_elasticity=-1.4,
        item=item,
        item_share_of_wallet=0.05,
        category_margin_rate=0.22,
        customer_item_elasticity=-1.8,
        n_items_in_category=12,
        is_loss_leader=False,
        price_change_history=[0, 3, 0, 1],
        periods_since_last_change=4,
    )

    obs = ci.to_observation()
    assert obs.shape == (33,), f"Expected shape (33,), got {obs.shape}"
    assert np.all(obs >= 0.0), "Observation has values below 0"
    assert np.all(obs <= 1.0), "Observation has values above 1"

    reconstructed = CustomerItemState.from_observation(obs)
    assert reconstructed.css_score == ci.css_score
    assert reconstructed.potential_tier == ci.potential_tier
    assert reconstructed.item.category == ci.item.category


def test_item_catalog_generation():
    """generate_item_catalog returns valid catalog with expected schema."""
    from src.data.synthetic_generator import generate_item_catalog

    catalog = generate_item_catalog(seed=42)

    # Row count from config default (500)
    assert len(catalog) == 500, f"Expected 500 rows, got {len(catalog)}"

    required_columns = [
        "item_id",
        "category",
        "category_id",
        "unit_cost",
        "unit_price",
        "item_margin_rate",
        "perishability",
        "substitutability",
        "seasonal_index",
    ]
    for col in required_columns:
        assert col in catalog.columns, f"Missing column: {col}"

    # All margins positive
    assert (catalog["item_margin_rate"] > 0).all(), "Found non-positive margins"


def test_customer_item_generation():
    """generate_customer_items produces valid customer-item pairs."""
    from src.data.synthetic_generator import (
        generate_customer_population,
        generate_item_catalog,
        generate_customer_items,
    )

    customers = generate_customer_population(n=50, seed=42)
    catalog = generate_item_catalog(seed=42)
    ci_df = generate_customer_items(customers, catalog, seed=42)

    assert len(ci_df) > 0
    assert "customer_id" in ci_df.columns
    assert "item_id" in ci_df.columns
    assert "category" in ci_df.columns


def test_clv_optimizer_compute_item():
    """CLVOptimizer.compute_item returns finite float with valid inputs."""
    from src.environment.customer import CustomerItemState
    from src.environment.item import ItemState
    from src.reward.reward_functions import CLVOptimizer

    item = ItemState(
        category=0,
        subcategory=0,
        unit_cost=5.0,
        unit_price=8.0,
        item_margin_rate=0.24,
        weekly_units=10.0,
        weekly_revenue=80.0,
        perishability=0.5,
        substitutability=0.5,
        competitive_index=0.5,
        seasonal_index=1.0,
        price_change_history=[0, 0, 0, 0],
        periods_since_last_change=3,
    )

    ci = CustomerItemState(
        css_score=3,
        performance_percentile=0.5,
        potential_tier=1,
        customer_margin_rate=0.24,
        weekly_cases=15.0,
        weekly_sales=750.0,
        deliveries_per_week=2.0,
        concept=1,
        syw_flag=False,
        perks_flag=False,
        churn_probability=0.06,
        current_period=10,
        customer_elasticity=-1.5,
        item=item,
        item_share_of_wallet=0.05,
        category_margin_rate=0.22,
        customer_item_elasticity=-1.5,
        n_items_in_category=10,
        is_loss_leader=False,
        price_change_history=[0, 0, 0, 0],
        periods_since_last_change=3,
    )

    clv = CLVOptimizer()
    reward = clv.compute_item(
        prev_item_margin=0.22,
        prev_item_units=9.5,
        prev_customer_margin=0.23,
        prev_churn=0.05,
        action=3,
        state=ci,
    )

    assert isinstance(reward, float)
    assert np.isfinite(reward), f"Reward not finite: {reward}"


def test_multi_agent_item_routing():
    """PortfolioManager routes by substitutability and margin floor."""
    from src.environment.customer import CustomerItemState
    from src.environment.item import ItemState
    from src.orchestrator.multi_agent import PortfolioManager

    business_rules = {
        "category_margin_floors": {"protein": 0.12, "paper": 0.18},
    }
    pm = PortfolioManager(
        config={"css_routing": {"scout": [1, 2], "guardian": [4, 5], "contested": [3]}},
        business_rules=business_rules,
    )

    def _make_ci(category, substitutability, item_margin_rate, css_score):
        item = ItemState(
            category=category,
            subcategory=0,
            unit_cost=5.0,
            unit_price=8.0,
            item_margin_rate=item_margin_rate,
            weekly_units=10.0,
            weekly_revenue=80.0,
            perishability=0.5,
            substitutability=substitutability,
            competitive_index=0.5,
            seasonal_index=1.0,
        )
        return CustomerItemState(
            css_score=css_score,
            performance_percentile=0.5,
            potential_tier=1,
            customer_margin_rate=0.24,
            weekly_cases=15.0,
            weekly_sales=750.0,
            deliveries_per_week=2.0,
            concept=1,
            syw_flag=False,
            perks_flag=False,
            churn_probability=0.05,
            current_period=10,
            customer_elasticity=-1.5,
            item=item,
        )

    # High-substitutability item (>0.8) should route to scout
    high_sub = _make_ci(category=0, substitutability=0.9, item_margin_rate=0.30, css_score=3)
    assert pm.assign(high_sub) == "scout"

    # Item near protein margin floor (0.12 + 0.03 = 0.15) should route to guardian
    near_floor = _make_ci(category=0, substitutability=0.5, item_margin_rate=0.14, css_score=3)
    assert pm.assign(near_floor) == "guardian"


def test_action_masking_initial():
    """Fresh env returns action_masks() with shape (7,) and all 1s initially."""
    from src.environment.pricing_env import DynamicPricingEnv

    env = DynamicPricingEnv(legacy_mode=False)
    env.reset(seed=42)

    masks = env.action_masks()
    assert masks.shape == (7,), f"Expected shape (7,), got {masks.shape}"
    assert np.all(masks == 1), f"Expected all 1s initially, got {masks}"
