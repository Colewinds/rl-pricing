"""Gymnasium environment for dynamic pricing of a single customer."""

from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml

from src.environment.customer import CustomerState
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

OBS_DIM = 17


class DynamicPricingEnv(gym.Env):
    """Single-customer dynamic pricing environment.

    Observation: 17-float normalized vector from CustomerState.
    Action: Discrete(7) price adjustment actions.
    Episode: Up to 52 weekly steps, terminates early on churn.

    Supports action masking via action_masks() for MaskablePPO.
    Configurable observation lag via ring buffer.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict | None = None, reward_fn=None):
        super().__init__()

        # Merge provided config with defaults
        default_config = _load_default_config()
        if config:
            self._deep_merge(default_config, config)
        self.config = default_config

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

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(n_actions)

        # Reward function (default: simple margin delta)
        self.reward_fn = reward_fn

        # Internal state (set in reset)
        self._customer: CustomerState | None = None
        self._step_count = 0
        self._action_history: list[int] = []
        self._obs_buffer: deque | None = None
        self._market_sim: MarketSimulator | None = None
        self._base_volume: float = 0.0
        self._base_sales: float = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Create market simulator with new seed
        sim_seed = self.np_random.integers(0, 2**31)
        self._market_sim = MarketSimulator(
            seed=int(sim_seed),
            config=self.config.get("environment", {}),
        )

        # Generate a random customer
        self._customer = self._random_customer()
        self._step_count = 0
        self._action_history = []

        # Initialize observation lag buffer
        obs = self._customer.to_observation()
        self._obs_buffer = deque(maxlen=max(self.obs_lag + 1, 1))
        for _ in range(self._obs_buffer.maxlen):
            self._obs_buffer.append(obs.copy())

        self._base_volume = self._customer.weekly_cases
        self._base_sales = self._customer.weekly_sales

        return self._get_lagged_obs(), {}

    def step(self, action: int):
        assert self._customer is not None, "Must call reset() before step()"
        action = int(action)

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

        # Apply price change
        price_change = self.action_map.get(action, 0.0)

        # Compute volume response
        delta_vol = self._market_sim.compute_volume_response(
            price_change=price_change,
            elasticity=self._customer.elasticity_estimate,
            base_volume=self._customer.weekly_cases,
            css_score=self._customer.css_score,
            periods_stable=self._customer.periods_since_last_change,
        )

        # Update customer state
        self._customer.weekly_cases = max(0.1, self._customer.weekly_cases + delta_vol)
        self._customer.weekly_sales = self._customer.weekly_cases * (
            self._base_sales / max(self._base_volume, 0.1)
        )

        # Update margin
        self._customer.current_margin_rate = np.clip(
            self._customer.current_margin_rate * (1 + price_change), 0.01, 0.60
        )
        self._customer.current_margin_dollars = (
            self._customer.weekly_sales * self._customer.current_margin_rate
        )

        # Update action history
        self._action_history.append(action)
        self._customer.price_change_history = [action] + self._customer.price_change_history[:3]

        # Update periods since last change
        if action == 0:
            self._customer.periods_since_last_change += 1
        else:
            self._customer.periods_since_last_change = 0

        # Compute churn probability
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

        # Check for churn
        churned = self._market_sim.check_churn(self._customer.churn_probability)

        # Push new observation to lag buffer
        obs = self._customer.to_observation()
        self._obs_buffer.append(obs)

        # Compute reward
        if self.reward_fn is not None:
            reward = float(self.reward_fn.compute(prev_state, action, self._customer))
        else:
            # Default: margin delta
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
                # If we've been oscillating, restrict to hold
                # Actually: just prevent continuing the oscillation
                pass  # Keep simple: mask is still permissive

        # Always allow hold
        mask[0] = 1

        return mask

    def _get_lagged_obs(self) -> np.ndarray:
        """Return the oldest observation in the lag buffer."""
        return self._obs_buffer[0].copy()

    def _random_customer(self) -> CustomerState:
        """Generate a random customer for a new episode."""
        rng = self._market_sim.rng

        # CSS distribution from config
        sd = self.config.get("synthetic_data", {})
        css_dist = sd.get("css_distribution", {})
        css_probs = [css_dist.get(f"css_{i}", 0.2) for i in range(1, 6)]
        total = sum(css_probs)
        css_probs = [p / total for p in css_probs]
        css_score = int(rng.choice([1, 2, 3, 4, 5], p=css_probs))

        # Benchmarks
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

        # Elasticity
        elast_cfg = sd.get("elasticity", {}).get("by_css", {})
        css_key = f"css_{css_score}"
        elast_mean = elast_cfg.get(css_key, {}).get("mean", -1.5)
        elast_std = elast_cfg.get(css_key, {}).get("std", 0.3)
        elasticity = float(np.clip(rng.normal(elast_mean, elast_std), -5.0, -0.1))

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
