"""Gymnasium environment for dynamic pricing at customer-item level."""

from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml

from src.environment.customer import CustomerState, CustomerItemState
from src.environment.item import ItemState, CATEGORIES, CONCEPT_IDS
from src.environment.market_simulator import MarketSimulator


def _load_default_config() -> dict:
    config_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# Action index -> fractional price change
_DEFAULT_ACTIONS = {
    0: 0.0,     # hold
    1: 0.02,    # price up 2%
    2: 0.05,    # price up 5%
    3: -0.02,   # price down 2%
    4: -0.05,   # price down 5%
    5: -0.10,   # price down 10%
    6: -0.15,   # price down 15%
}

# Deep cut actions (10%+ discount)
_DEEP_CUT_ACTIONS = {5, 6}
_DISCOUNT_ACTIONS = {3, 4, 5, 6}

LEGACY_OBS_DIM = 17
ITEM_OBS_DIM = 33


class DynamicPricingEnv(gym.Env):
    """Customer-item dynamic pricing environment.

    Observation: 33-float normalized vector from CustomerItemState (or 17 in legacy mode).
    Action: Discrete(7) price adjustment actions applied to a specific item.
    Episode: Up to 52 weekly steps, terminates early on churn.

    Supports action masking via action_masks() for MaskablePPO.
    Configurable observation lag via ring buffer.

    Legacy mode (legacy_mode=True): Operates at customer level only with 17-dim obs,
    fully backward compatible with existing trained models.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict | None = None, reward_fn=None, legacy_mode: bool = False):
        super().__init__()

        # Merge provided config with defaults
        default_config = _load_default_config()
        if config:
            self._deep_merge(default_config, config)
        self.config = default_config
        self.legacy_mode = legacy_mode

        env_cfg = self.config.get("environment", {})
        self.episode_length = env_cfg.get("episode_length", 52)
        self.obs_lag = env_cfg.get("observation_lag", 2)
        self.stickiness_threshold = env_cfg.get("stickiness_threshold_periods", 8)

        # Action configuration
        actions_cfg = env_cfg.get("actions", {})
        self.action_map = {}
        for k, v in (actions_cfg or _DEFAULT_ACTIONS).items():
            idx = int(k)
            if isinstance(v, dict):
                self.action_map[idx] = v["pct_change"]
            else:
                self.action_map[idx] = float(v)
        if not self.action_map:
            self.action_map = dict(_DEFAULT_ACTIONS)

        n_actions = env_cfg.get("action_space_size", len(self.action_map))

        # Masking config
        masking_cfg = env_cfg.get("masking", {})
        self.no_consecutive_deep_cuts = masking_cfg.get("no_consecutive_deep_cuts", True)
        self.no_oscillation_periods = masking_cfg.get("no_updown_oscillation_periods", 2)

        # Business rules config
        self.business_rules = self.config.get("business_rules", {})

        # Spaces
        obs_dim = LEGACY_OBS_DIM if legacy_mode else ITEM_OBS_DIM
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(n_actions)

        # Reward function (default: simple margin delta)
        self.reward_fn = reward_fn

        # Internal state (set in reset)
        self._customer: CustomerState | None = None
        self._customer_item: CustomerItemState | None = None
        self._step_count = 0
        self._action_history: list[int] = []
        self._obs_buffer: deque | None = None
        self._market_sim: MarketSimulator | None = None
        self._base_volume: float = 0.0
        self._base_sales: float = 0.0
        self._consecutive_discounts: int = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Create market simulator with new seed
        sim_seed = self.np_random.integers(0, 2**31)
        self._market_sim = MarketSimulator(
            seed=int(sim_seed),
            config=self.config.get("environment", {}),
        )

        if self.legacy_mode:
            self._customer = self._random_customer()
            obs = self._customer.to_observation()
            self._base_volume = self._customer.weekly_cases
            self._base_sales = self._customer.weekly_sales
        else:
            self._customer_item = self._random_customer_item()
            obs = self._customer_item.to_observation()
            self._base_volume = self._customer_item.item.weekly_units
            self._base_sales = self._customer_item.item.weekly_revenue

        self._step_count = 0
        self._action_history = []
        self._consecutive_discounts = 0

        # Initialize observation lag buffer
        self._obs_buffer = deque(maxlen=max(self.obs_lag + 1, 1))
        for _ in range(self._obs_buffer.maxlen):
            self._obs_buffer.append(obs.copy())

        return self._get_lagged_obs(), {}

    def step(self, action: int):
        action = int(action)

        if self.legacy_mode:
            return self._step_legacy(action)
        return self._step_item(action)

    def _step_item(self, action: int):
        """Item-level step with seasonality and cross-item effects."""
        ci = self._customer_item
        assert ci is not None, "Must call reset() before step()"

        # Snapshot previous state
        prev_item_margin = ci.item.item_margin_rate
        prev_item_units = ci.item.weekly_units
        prev_item_revenue = ci.item.weekly_revenue
        prev_customer_margin = ci.customer_margin_rate
        prev_churn = ci.churn_probability

        price_change = self.action_map.get(action, 0.0)

        # Apply seasonality to base volume
        seasonality_cfg = self.config.get("synthetic_data", {}).get("seasonality", {})
        seasonal_volume = self._market_sim.apply_seasonality(
            self._base_volume, ci.current_period, seasonality_cfg
        )

        # Compute item-specific elasticity with seasonal modulation
        item_elasticity = self._market_sim.compute_item_elasticity(
            base_elasticity=ci.customer_elasticity,
            category=ci.item.category,
            concept=ci.concept,
            substitutability=ci.item.substitutability,
            perishability=ci.item.perishability,
            config=self.config.get("items", {}),
        )
        item_elasticity = self._market_sim.apply_seasonal_elasticity(
            item_elasticity, ci.item.seasonal_index, ci.current_period, seasonality_cfg
        )

        # Volume response at item level
        delta_vol = self._market_sim.compute_volume_response(
            price_change=price_change,
            elasticity=item_elasticity,
            base_volume=seasonal_volume,
            css_score=ci.css_score,
            periods_stable=ci.item.periods_since_last_change,
            seasonal_index=ci.item.seasonal_index,
            perishability=ci.item.perishability,
        )

        # Update item state
        ci.item.weekly_units = max(0.1, ci.item.weekly_units + delta_vol)
        ci.item.item_margin_rate = float(np.clip(
            ci.item.item_margin_rate * (1 + price_change), 0.01, 0.60
        ))
        ci.item.unit_price = ci.item.unit_cost / max(1 - ci.item.item_margin_rate, 0.01)
        ci.item.weekly_revenue = ci.item.weekly_units * ci.item.unit_price

        # Update item action history
        ci.item.price_change_history = [action] + ci.item.price_change_history[:3]
        if action == 0:
            ci.item.periods_since_last_change += 1
        else:
            ci.item.periods_since_last_change = 0

        # Sync legacy fields
        ci.price_change_history = ci.item.price_change_history
        ci.periods_since_last_change = ci.item.periods_since_last_change

        # Update customer-level aggregates
        # Approximate: adjust customer margin based on item's weight
        share = ci.item_share_of_wallet
        ci.customer_margin_rate = float(np.clip(
            ci.customer_margin_rate + share * (ci.item.item_margin_rate - prev_item_margin),
            0.01, 0.60,
        ))
        ci.category_margin_rate = float(np.clip(
            ci.category_margin_rate + (ci.item.item_margin_rate - prev_item_margin) / max(ci.n_items_in_category, 1),
            0.01, 0.60,
        ))

        # Update item share of wallet
        if ci.weekly_sales > 0:
            ci.item_share_of_wallet = float(np.clip(
                ci.item.weekly_revenue / ci.weekly_sales, 0.0, 1.0
            ))

        # Cross-item churn effect
        cross_effect = self._market_sim.compute_cross_item_effect(
            price_change, ci.item_share_of_wallet, ci.item.category,
        )

        # Customer-level churn (churn is a CUSTOMER event, not item)
        delivery_delay_discount = self.business_rules.get("delivery_delay_reward_discount", 0.3)
        churn_thresholds = self.config.get("reward", {}).get("clv_optimizer", {}).get(
            "churn_thresholds", {}
        )
        css_key = f"css_{ci.css_score}"
        threshold = churn_thresholds.get(css_key, 0.25)

        ci.churn_probability = self._market_sim.compute_churn_probability(
            margin_rate=ci.customer_margin_rate,
            css_score=ci.css_score,
            syw=ci.syw_flag,
            periods_stable=ci.periods_since_last_change,
            threshold=threshold,
        )
        # Cross-item amplification
        ci.churn_probability = float(np.clip(ci.churn_probability * cross_effect, 0.0, 1.0))

        churned = self._market_sim.check_churn(ci.churn_probability)

        # Track consecutive discounts
        self._action_history.append(action)
        if action in _DISCOUNT_ACTIONS:
            self._consecutive_discounts += 1
        else:
            self._consecutive_discounts = 0

        # Advance period
        ci.current_period = min(ci.current_period + 1, 51)

        # Push observation
        obs = ci.to_observation()
        self._obs_buffer.append(obs)

        # Compute reward
        if self.reward_fn is not None:
            reward = float(self.reward_fn.compute_item(
                prev_item_margin=prev_item_margin,
                prev_item_units=prev_item_units,
                prev_customer_margin=prev_customer_margin,
                prev_churn=prev_churn,
                action=action,
                state=ci,
            ))
        else:
            margin_delta = ci.item.weekly_revenue * ci.item.item_margin_rate - prev_item_revenue * prev_item_margin
            reward = float(margin_delta)

        self._step_count += 1
        terminated = churned
        truncated = self._step_count >= self.episode_length

        info = {
            "customer_state": {
                "css_score": ci.css_score,
                "margin_rate": ci.customer_margin_rate,
                "weekly_cases": ci.weekly_cases,
                "churn_probability": ci.churn_probability,
            },
            "item_state": {
                "category": ci.item.category,
                "item_margin_rate": ci.item.item_margin_rate,
                "weekly_units": ci.item.weekly_units,
                "weekly_revenue": ci.item.weekly_revenue,
            },
            "margin_dollars": ci.item.weekly_revenue * ci.item.item_margin_rate,
            "volume": ci.item.weekly_units,
            "step": self._step_count,
            "churned": churned,
        }

        return self._get_lagged_obs(), reward, terminated, truncated, info

    def _step_legacy(self, action: int):
        """Legacy 17-dim customer-level step (backward compatible)."""
        assert self._customer is not None, "Must call reset() before step()"

        prev_state = CustomerState(
            css_score=self._customer.css_score,
            performance_percentile=self._customer.performance_percentile,
            potential_tier=self._customer.potential_tier,
            current_margin_rate=self._customer.current_margin_rate,
            current_margin_dollars=self._customer.current_margin_dollars,
            weekly_cases=self._customer.weekly_cases,
            weekly_sales=self._customer.weekly_sales,
            deliveries_per_week=self._customer.deliveries_per_week,
            elasticity_estimate=self._customer.elasticity_estimate,
            price_change_history=list(self._customer.price_change_history),
            periods_since_last_change=self._customer.periods_since_last_change,
            syw_flag=self._customer.syw_flag,
            perks_flag=self._customer.perks_flag,
            churn_probability=self._customer.churn_probability,
        )

        price_change = self.action_map.get(action, 0.0)

        delta_vol = self._market_sim.compute_volume_response(
            price_change=price_change,
            elasticity=self._customer.elasticity_estimate,
            base_volume=self._base_volume,
            css_score=self._customer.css_score,
            periods_stable=self._customer.periods_since_last_change,
        )

        self._customer.weekly_cases = max(0.1, self._customer.weekly_cases + delta_vol)
        self._customer.weekly_sales = self._customer.weekly_cases * (
            self._base_sales / max(self._base_volume, 0.1)
        )

        self._customer.current_margin_rate = np.clip(
            self._customer.current_margin_rate * (1 + price_change), 0.01, 0.60
        )
        self._customer.current_margin_dollars = (
            self._customer.weekly_sales * self._customer.current_margin_rate
        )

        self._action_history.append(action)
        self._customer.price_change_history = [action] + self._customer.price_change_history[:3]

        if action == 0:
            self._customer.periods_since_last_change += 1
        else:
            self._customer.periods_since_last_change = 0

        sd = self.config.get("synthetic_data", {})
        churn_thresholds = self.config.get("reward", {}).get("clv_optimizer", {}).get(
            "churn_thresholds", {}
        )
        css_key = f"css_{self._customer.css_score}"
        threshold = churn_thresholds.get(css_key, 0.25)

        self._customer.churn_probability = self._market_sim.compute_churn_probability(
            margin_rate=self._customer.current_margin_rate,
            css_score=self._customer.css_score,
            syw=self._customer.syw_flag,
            periods_stable=self._customer.periods_since_last_change,
            threshold=threshold,
        )

        churned = self._market_sim.check_churn(self._customer.churn_probability)

        obs = self._customer.to_observation()
        self._obs_buffer.append(obs)

        if self.reward_fn is not None:
            reward = float(self.reward_fn.compute(prev_state, action, self._customer))
        else:
            reward = float(
                self._customer.current_margin_dollars - prev_state.current_margin_dollars
            )

        self._step_count += 1
        terminated = churned
        truncated = self._step_count >= self.episode_length

        info = {
            "customer_state": {
                "css_score": self._customer.css_score,
                "margin_rate": self._customer.current_margin_rate,
                "weekly_cases": self._customer.weekly_cases,
                "churn_probability": self._customer.churn_probability,
            },
            "margin_dollars": self._customer.current_margin_dollars,
            "volume": self._customer.weekly_cases,
            "step": self._step_count,
            "churned": churned,
        }

        return self._get_lagged_obs(), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return binary mask of valid actions. 1 = allowed, 0 = masked."""
        mask = np.ones(self.action_space.n, dtype=np.int8)

        if not self._action_history:
            return mask

        last_action = self._action_history[-1]

        # No consecutive deep cuts
        if self.no_consecutive_deep_cuts and last_action in _DEEP_CUT_ACTIONS:
            for a in _DEEP_CUT_ACTIONS:
                if a < len(mask):
                    mask[a] = 0

        # No up-down oscillation within window
        if self.no_oscillation_periods > 0 and len(self._action_history) >= self.no_oscillation_periods:
            recent = self._action_history[-self.no_oscillation_periods:]
            has_up = any(self.action_map.get(a, 0) > 0 for a in recent)
            has_down = any(self.action_map.get(a, 0) < 0 for a in recent)
            if has_up and has_down:
                # If oscillating, only allow hold and continuing in current direction
                last_dir = self.action_map.get(last_action, 0)
                if last_dir > 0:
                    # Was going up, mask all downs
                    for a in _DISCOUNT_ACTIONS:
                        mask[a] = 0
                elif last_dir < 0:
                    # Was going down, mask all ups
                    for a in (1, 2):
                        if a < len(mask):
                            mask[a] = 0

        # Item-level masks (only in non-legacy mode)
        if not self.legacy_mode and self._customer_item is not None:
            ci = self._customer_item
            cat_name = CATEGORIES.get(ci.item.category, "misc")

            # Margin floor mask: if item margin near category floor, mask discounts
            cat_floors = self.business_rules.get("category_margin_floors", {})
            floor = cat_floors.get(cat_name, 0.10)
            if ci.item.item_margin_rate <= floor + 0.02:
                for a in _DISCOUNT_ACTIONS:
                    mask[a] = 0

            # Consecutive discount limit
            max_consec = self.business_rules.get("max_consecutive_discounts", 3)
            if self._consecutive_discounts >= max_consec:
                for a in _DISCOUNT_ACTIONS:
                    mask[a] = 0

            # Margin velocity mask: if margin declined too fast, mask deep cuts
            max_decline = self.business_rules.get("margin_velocity_max_decline", -0.05)
            if len(self._action_history) >= 4:
                recent_changes = [self.action_map.get(a, 0) for a in self._action_history[-4:]]
                cumulative = sum(recent_changes)
                if cumulative < max_decline:
                    for a in _DEEP_CUT_ACTIONS:
                        mask[a] = 0

            # Low-substitutability items: mask deep cuts
            if ci.item.substitutability < 0.3:
                for a in _DEEP_CUT_ACTIONS:
                    mask[a] = 0

            # Holiday margin floor boost
            holiday_periods = self.business_rules.get("holiday_periods", [])
            if ci.current_period in holiday_periods:
                boost = self.business_rules.get("holiday_margin_floor_boost", 0.03)
                if ci.item.item_margin_rate <= floor + boost + 0.02:
                    for a in _DISCOUNT_ACTIONS:
                        mask[a] = 0

        # Always allow hold
        mask[0] = 1

        return mask

    def _get_lagged_obs(self) -> np.ndarray:
        """Return the oldest observation in the lag buffer."""
        return self._obs_buffer[0].copy()

    def _random_customer_item(self) -> CustomerItemState:
        """Generate a random customer-item pair for a new episode."""
        rng = self._market_sim.rng
        sd = self.config.get("synthetic_data", {})
        items_cfg = self.config.get("items", {})

        # Customer attributes
        css_dist = sd.get("css_distribution", {})
        css_probs = [css_dist.get(f"css_{i}", 0.2) for i in range(1, 6)]
        total = sum(css_probs)
        css_probs = [p / total for p in css_probs]
        css_score = int(rng.choice([1, 2, 3, 4, 5], p=css_probs))

        bench = sd.get("percentile_benchmarks", {})
        css_scale = {1: 0.4, 2: 0.6, 3: 1.0, 4: 1.5, 5: 2.0}
        scale = css_scale[css_score]

        cases_monthly = max(1.0, rng.lognormal(np.log(bench.get("cases_p50_monthly", 60) * scale), 0.5))
        sales_monthly = max(10.0, rng.lognormal(np.log(bench.get("sales_p50_monthly", 3000) * scale), 0.5))
        margin_rate = float(np.clip(rng.lognormal(np.log(bench.get("dm_pct_p50", 0.24)), 0.3), 0.05, 0.55))

        weekly_cases = cases_monthly / 4.33
        weekly_sales = sales_monthly / 4.33
        deliveries = float(np.clip(weekly_cases / 6.0 + rng.normal(0, 0.3), 0.5, 7.0))

        # Concept
        concepts = sd.get("concepts", [])
        concept_names = [c["name"] for c in concepts]
        concept_weights = [c["weight"] for c in concepts]
        concept_name = rng.choice(concept_names, p=concept_weights) if concepts else "casual_dining"
        concept_id = CONCEPT_IDS.get(concept_name, 1)

        # Elasticity
        elast_cfg = sd.get("elasticity", {}).get("by_css", {})
        css_key = f"css_{css_score}"
        elast_mean = elast_cfg.get(css_key, {}).get("mean", -1.5)
        elast_std = elast_cfg.get(css_key, {}).get("std", 0.3)
        concept_modifier = next(
            (c["elasticity_modifier"] for c in concepts if c["name"] == concept_name),
            1.0
        )
        elasticity = float(np.clip(rng.normal(elast_mean, elast_std) * concept_modifier, -5.0, -0.1))

        # SYW/Perks
        syw_skew = sd.get("syw_css_skew", [0.1, 0.15, 0.3, 0.4, 0.5])
        syw = bool(rng.random() < syw_skew[css_score - 1])
        perks = bool(rng.random() < sd.get("perks_penetration", 0.15))

        performance_percentile = float(rng.random())
        if css_score >= 4 and performance_percentile >= 0.5:
            potential_tier = 2
        elif css_score >= 3 or performance_percentile >= 0.3:
            potential_tier = 1
        else:
            potential_tier = 0

        # Generate random item
        categories = items_cfg.get("categories", {})
        cat_names = list(categories.keys())
        if not cat_names:
            cat_names = ["protein"]

        # Weight by concept affinity
        affinity = items_cfg.get("concept_category_affinity", {}).get(concept_name, {})
        cat_weights_raw = [affinity.get(c, 1.0) for c in cat_names]
        cat_total = sum(cat_weights_raw)
        cat_probs = [w / cat_total for w in cat_weights_raw]

        cat_name = rng.choice(cat_names, p=cat_probs)
        cat_cfg = categories.get(cat_name, {})
        cat_id = cat_cfg.get("id", 0)

        subcats = cat_cfg.get("subcategories", ["default"])
        subcat_idx = int(rng.integers(0, len(subcats)))

        # Item properties from category config
        unit_cost = max(0.50, float(rng.lognormal(np.log(8.0), 0.6)))
        margin_cfg = cat_cfg.get("margin_rate", {"mean": 0.24, "std": 0.05})
        item_margin = float(np.clip(rng.normal(margin_cfg["mean"], margin_cfg["std"]), 0.03, 0.50))
        unit_price = unit_cost / max(1 - item_margin, 0.01)

        perish_cfg = cat_cfg.get("perishability", {"mean": 0.5, "std": 0.1})
        perishability = float(np.clip(rng.normal(perish_cfg["mean"], perish_cfg["std"]), 0.0, 1.0))

        sub_cfg = cat_cfg.get("substitutability", {"mean": 0.5, "std": 0.15})
        substitutability = float(np.clip(rng.normal(sub_cfg["mean"], sub_cfg["std"]), 0.0, 1.0))

        competitive_index = float(np.clip(0.5 + 0.3 * (substitutability - 0.5) + rng.normal(0, 0.15), 0.0, 1.0))

        seasonal_base = 1.0
        if cat_name in ("produce", "protein", "bakery"):
            seasonal_base = 1.3
        elif cat_name in ("frozen", "paper", "misc"):
            seasonal_base = 0.7
        seasonal_index = float(np.clip(rng.normal(seasonal_base, 0.3), 0.0, 2.0))

        # Item volume: fraction of customer's total
        n_items_approx = int(np.clip(rng.normal(30, 15), 5, 100))
        item_share = float(np.clip(rng.dirichlet(np.ones(n_items_approx))[0], 0.01, 0.5))
        item_revenue = weekly_sales * item_share
        item_units = max(0.1, item_revenue / max(unit_price, 0.01))

        is_loss_leader = bool(rng.random() < items_cfg.get("loss_leader_pct", 0.05))

        item = ItemState(
            category=cat_id,
            subcategory=subcat_idx,
            unit_cost=unit_cost,
            unit_price=unit_price,
            item_margin_rate=item_margin,
            weekly_units=item_units,
            weekly_revenue=item_revenue,
            perishability=perishability,
            substitutability=substitutability,
            competitive_index=competitive_index,
            seasonal_index=seasonal_index,
            is_loss_leader=is_loss_leader,
            price_change_history=[0, 0, 0, 0],
            periods_since_last_change=int(rng.integers(0, 20)),
        )

        # Category margin: average across items in category
        cat_margin = float(np.clip(margin_cfg["mean"] + rng.normal(0, 0.02), 0.03, 0.50))

        # Item-specific elasticity
        item_elasticity = self._market_sim.compute_item_elasticity(
            base_elasticity=elasticity,
            category=cat_id,
            concept=concept_id,
            substitutability=substitutability,
            perishability=perishability,
            config=items_cfg,
        )

        return CustomerItemState(
            css_score=css_score,
            performance_percentile=performance_percentile,
            potential_tier=potential_tier,
            customer_margin_rate=margin_rate,
            weekly_cases=weekly_cases,
            weekly_sales=weekly_sales,
            deliveries_per_week=deliveries,
            concept=concept_id,
            syw_flag=syw,
            perks_flag=perks,
            churn_probability=0.05,
            current_period=int(rng.integers(0, 52)),
            customer_elasticity=elasticity,
            item=item,
            item_share_of_wallet=item_share,
            category_margin_rate=cat_margin,
            customer_item_elasticity=item_elasticity,
            n_items_in_category=int(np.clip(rng.normal(15, 5), 3, 40)),
            is_loss_leader=is_loss_leader,
            price_change_history=[0, 0, 0, 0],
            periods_since_last_change=item.periods_since_last_change,
        )

    def _random_customer(self) -> CustomerState:
        """Generate a random customer for a new episode (legacy mode)."""
        rng = self._market_sim.rng

        sd = self.config.get("synthetic_data", {})
        css_dist = sd.get("css_distribution", {})
        css_probs = [css_dist.get(f"css_{i}", 0.2) for i in range(1, 6)]
        total = sum(css_probs)
        css_probs = [p / total for p in css_probs]
        css_score = int(rng.choice([1, 2, 3, 4, 5], p=css_probs))

        bench = sd.get("percentile_benchmarks", {})
        css_scale = {1: 0.4, 2: 0.6, 3: 1.0, 4: 1.5, 5: 2.0}
        scale = css_scale[css_score]

        cases_monthly = max(1.0, rng.lognormal(np.log(bench.get("cases_p50_monthly", 60) * scale), 0.5))
        sales_monthly = max(10.0, rng.lognormal(np.log(bench.get("sales_p50_monthly", 3000) * scale), 0.5))
        margin_rate = float(np.clip(rng.lognormal(np.log(bench.get("dm_pct_p50", 0.24)), 0.3), 0.05, 0.55))

        weekly_cases = cases_monthly / 4.33
        weekly_sales = sales_monthly / 4.33
        margin_dollars = weekly_sales * margin_rate

        deliveries = float(np.clip(weekly_cases / 6.0 + rng.normal(0, 0.3), 0.5, 7.0))

        elast_cfg = sd.get("elasticity", {}).get("by_css", {})
        css_key = f"css_{css_score}"
        elast_mean = elast_cfg.get(css_key, {}).get("mean", -1.5)
        elast_std = elast_cfg.get(css_key, {}).get("std", 0.3)
        elasticity = float(np.clip(rng.normal(elast_mean, elast_std), -5.0, -0.1))

        syw_skew = sd.get("syw_css_skew", [0.1, 0.15, 0.3, 0.4, 0.5])
        syw = bool(rng.random() < syw_skew[css_score - 1])
        perks = bool(rng.random() < sd.get("perks_penetration", 0.15))

        performance_percentile = float(rng.random())
        if css_score >= 4 and performance_percentile >= 0.5:
            potential_tier = 2
        elif css_score >= 3 or performance_percentile >= 0.3:
            potential_tier = 1
        else:
            potential_tier = 0

        return CustomerState(
            css_score=css_score,
            performance_percentile=performance_percentile,
            potential_tier=potential_tier,
            current_margin_rate=margin_rate,
            current_margin_dollars=margin_dollars,
            weekly_cases=weekly_cases,
            weekly_sales=weekly_sales,
            deliveries_per_week=deliveries,
            elasticity_estimate=elasticity,
            price_change_history=[0, 0, 0, 0],
            periods_since_last_change=int(rng.integers(0, 20)),
            syw_flag=syw,
            perks_flag=perks,
            churn_probability=0.05,
        )

    @staticmethod
    def _deep_merge(base: dict, overrides: dict):
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                DynamicPricingEnv._deep_merge(base[k], v)
            else:
                base[k] = v
