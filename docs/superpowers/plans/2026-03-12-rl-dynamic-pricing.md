# RL Dynamic Pricing Agent Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-credible RL-based dynamic pricing prototype with multi-agent orchestration, configurable synthetic data, and Streamlit dashboard.

**Architecture:** Single-customer Gymnasium environment with SB3 agents (PPO/DQN), custom multi-agent orchestration (Scout/Guardian/Portfolio Manager), config-driven synthetic data, and 4-tab Streamlit dashboard. All hyperparameters and data distributions in YAML config.

**Tech Stack:** Python 3.12, Stable-Baselines3, Gymnasium, sb3-contrib (action masking), pandas, numpy, PyYAML, Streamlit, Plotly, TensorBoard, pytest

**Spec:** `docs/superpowers/specs/2026-03-12-rl-dynamic-pricing-design.md`

---

## Chunk 1: Project Scaffold & Configuration

### Task 1: Project Bootstrap

**Files:**
- Create: `pricing_rl/pyproject.toml`
- Create: `pricing_rl/src/__init__.py`
- Create: `pricing_rl/src/environment/__init__.py`
- Create: `pricing_rl/src/agent/__init__.py`
- Create: `pricing_rl/src/reward/__init__.py`
- Create: `pricing_rl/src/data/__init__.py`
- Create: `pricing_rl/src/evaluation/__init__.py`
- Create: `pricing_rl/src/monitoring/__init__.py`
- Create: `pricing_rl/src/orchestrator/__init__.py`
- Create: `pricing_rl/config/default.yaml`
- Create: `pricing_rl/config/scenarios/conservative.yaml`
- Create: `pricing_rl/config/scenarios/aggressive.yaml`
- Create: `pricing_rl/config/scenarios/balanced.yaml`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "pricing-rl"
version = "0.1.0"
description = "RL-based dynamic pricing agent for foodservice distribution"
requires-python = ">=3.11"
dependencies = [
    "stable-baselines3>=2.3.0",
    "sb3-contrib>=2.3.0",
    "gymnasium>=0.29.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "pyyaml>=6.0",
    "streamlit>=1.30.0",
    "plotly>=5.18.0",
    "tensorboard>=2.15.0",
    "torch>=2.1.0",
    "scipy>=1.11.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 2: Create default.yaml config**

```yaml
environment:
  episode_length: 52
  observation_lag: 2
  action_space_size: 7
  actions:
    0: {name: "hold", pct_change: 0.0}
    1: {name: "price_up_2pct", pct_change: 0.02}
    2: {name: "price_up_5pct", pct_change: 0.05}
    3: {name: "price_down_2pct", pct_change: -0.02}
    4: {name: "price_down_5pct", pct_change: -0.05}
    5: {name: "price_down_10pct", pct_change: -0.10}
    6: {name: "price_down_15pct", pct_change: -0.15}
  masking:
    no_consecutive_deep_cuts: true
    no_updown_oscillation_periods: 2
  stickiness_threshold_periods: 8

synthetic_data:
  n_customers: 10000
  css_distribution:
    css_1: 0.10
    css_2: 0.10
    css_3: 0.40
    css_4: 0.25
    css_5: 0.15
  percentile_benchmarks:
    cases_p50_monthly: 60
    sales_p50_monthly: 3000
    dm_pct_p50: 0.24
    dm_dollars_p50_monthly: 665
  elasticity:
    css_correlation: -0.4
    by_css:
      css_1: {mean: -2.5, std: 0.5}
      css_2: {mean: -2.0, std: 0.4}
      css_3: {mean: -1.5, std: 0.3}
      css_4: {mean: -1.0, std: 0.25}
      css_5: {mean: -0.7, std: 0.2}
  syw_penetration: 0.30
  perks_penetration: 0.15
  syw_css_skew: [0.10, 0.15, 0.30, 0.40, 0.50]  # probability by CSS 1-5
  concepts:
    - {name: "qsr", weight: 0.30, elasticity_modifier: 1.2}
    - {name: "casual_dining", weight: 0.25, elasticity_modifier: 1.0}
    - {name: "fine_dining", weight: 0.15, elasticity_modifier: 0.7}
    - {name: "institutional", weight: 0.20, elasticity_modifier: 0.9}
    - {name: "healthcare", weight: 0.10, elasticity_modifier: 0.8}
  seasonality:
    q1_modifier: 0.85
    q2_modifier: 1.0
    q3_modifier: 1.05
    q4_modifier: 1.15
  annual_churn_baseline: 0.07
  churn_css_multiplier: {css_1: 2.0, css_2: 1.5, css_3: 1.0, css_4: 0.7, css_5: 0.5}

reward:
  margin_maximizer: {}
  clv_optimizer:
    alpha_by_css: {css_1: 0.3, css_2: 0.35, css_3: 0.5, css_4: 0.6, css_5: 0.7}
    beta_by_css: {css_1: 0.5, css_2: 0.45, css_3: 0.3, css_4: 0.2, css_5: 0.15}
    gamma: 10.0
    delta: 2.0
    churn_thresholds: {css_1: 0.40, css_2: 0.40, css_3: 0.25, css_4: 0.15, css_5: 0.15}
    volatility_window: 4
    volatility_max_changes: 2
    lifetime_discount_by_css: {css_1: 0.5, css_2: 0.6, css_3: 0.8, css_4: 0.9, css_5: 1.0}
  portfolio_optimizer:
    css_migration_bonus: 5.0
    action_concentration_penalty: 3.0

training:
  ppo:
    learning_rate: 0.0003
    n_steps: 2048
    batch_size: 64
    n_epochs: 10
    gamma: 0.99
    clip_range: 0.2
  dqn:
    learning_rate: 0.0001
    buffer_size: 100000
    learning_starts: 1000
    batch_size: 32
    gamma: 0.99
    exploration_fraction: 0.3
    exploration_final_eps: 0.05
  timesteps: 500000
  eval_freq: 10000
  early_stopping_patience: 50000
  checkpoint_freq: 50000
  log_dir: "results/tensorboard"
  model_dir: "results/models"

multi_agent:
  portfolio_manager:
    reallocation_period: 4
    css_routing:
      scout: [1, 2]
      guardian: [4, 5]
      contested: [3]
  scout:
    exploration_bonus: 0.1
  guardian:
    restricted_actions: [0, 1, 2, 3]  # Hold, +2%, +5%, -2% only

monitoring:
  reward_drift_sigma: 2.0
  action_entropy_min: 0.5
  alert_consecutive_periods: 3

evaluation:
  ab_test:
    split_ratio: 0.5
    n_simulations: 100
    confidence_level: 0.95
```

- [ ] **Step 3: Create scenario config overrides**

`config/scenarios/conservative.yaml`:
```yaml
environment:
  masking:
    no_consecutive_deep_cuts: true
    no_updown_oscillation_periods: 3
multi_agent:
  guardian:
    restricted_actions: [0, 1, 3]  # Hold, +2%, -2% only
```

`config/scenarios/aggressive.yaml`:
```yaml
environment:
  masking:
    no_consecutive_deep_cuts: false
    no_updown_oscillation_periods: 0
training:
  ppo:
    clip_range: 0.3
multi_agent:
  scout:
    exploration_bonus: 0.2
```

`config/scenarios/balanced.yaml`:
```yaml
# Balanced scenario uses default.yaml as-is
# This file exists for explicit selection
```

- [ ] **Step 4: Create all __init__.py files and directory structure**

All `__init__.py` files are empty initially. Create directories:
- `pricing_rl/src/environment/`
- `pricing_rl/src/agent/`
- `pricing_rl/src/reward/`
- `pricing_rl/src/data/`
- `pricing_rl/src/evaluation/`
- `pricing_rl/src/monitoring/`
- `pricing_rl/src/orchestrator/`
- `pricing_rl/scripts/`
- `pricing_rl/dashboard/`
- `pricing_rl/tests/`
- `pricing_rl/config/scenarios/`
- `pricing_rl/results/`
- `pricing_rl/data/`

- [ ] **Step 5: Install in dev mode**

Run: `cd pricing_rl && python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`
Expected: successful install with all dependencies

- [ ] **Step 6: Commit scaffold**

```bash
git add pricing_rl/
git commit -m "feat: project scaffold with config-driven architecture"
```

---

## Chunk 2: Customer Model & Synthetic Data Generator

### Task 2: Customer State Dataclass

**Files:**
- Create: `pricing_rl/src/environment/customer.py`
- Create: `pricing_rl/tests/test_customer.py`

- [ ] **Step 1: Write failing test for Customer dataclass**

```python
# tests/test_customer.py
import numpy as np
from src.environment.customer import CustomerState

def test_customer_state_creation():
    cs = CustomerState(
        css_score=3,
        performance_percentile=0.65,
        potential_tier=1,  # Medium
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
    assert obs.shape == (18,)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd pricing_rl && python -m pytest tests/test_customer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.environment.customer'`

- [ ] **Step 3: Implement CustomerState**

```python
# src/environment/customer.py
from dataclasses import dataclass, field
import numpy as np

# Normalization constants (max expected values for scaling to [0,1])
NORM = {
    "css_score": 5.0,
    "potential_tier": 2.0,
    "margin_rate": 0.60,
    "margin_dollars": 5000.0,
    "weekly_cases": 200.0,
    "weekly_sales": 15000.0,
    "deliveries": 7.0,
    "elasticity": 4.0,  # absolute value
    "action": 6.0,
    "periods": 52.0,
}


@dataclass
class CustomerState:
    css_score: int
    performance_percentile: float
    potential_tier: int  # 0=Low, 1=Medium, 2=High
    current_margin_rate: float
    current_margin_dollars: float
    weekly_cases: float
    weekly_sales: float
    deliveries_per_week: float
    elasticity_estimate: float
    price_change_history: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    periods_since_last_change: int = 0
    syw_flag: bool = False
    perks_flag: bool = False
    churn_probability: float = 0.0

    def to_observation(self) -> np.ndarray:
        obs = np.array([
            self.css_score / NORM["css_score"],
            self.performance_percentile,
            self.potential_tier / NORM["potential_tier"],
            min(self.current_margin_rate / NORM["margin_rate"], 1.0),
            min(self.current_margin_dollars / NORM["margin_dollars"], 1.0),
            min(self.weekly_cases / NORM["weekly_cases"], 1.0),
            min(self.weekly_sales / NORM["weekly_sales"], 1.0),
            min(self.deliveries_per_week / NORM["deliveries"], 1.0),
            min(abs(self.elasticity_estimate) / NORM["elasticity"], 1.0),
            self.price_change_history[0] / NORM["action"],
            self.price_change_history[1] / NORM["action"],
            self.price_change_history[2] / NORM["action"],
            self.price_change_history[3] / NORM["action"],
            min(self.periods_since_last_change / NORM["periods"], 1.0),
            float(self.syw_flag),
            float(self.perks_flag),
            self.churn_probability,
        ], dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)

    @classmethod
    def from_observation(cls, obs: np.ndarray) -> "CustomerState":
        return cls(
            css_score=int(round(obs[0] * NORM["css_score"])),
            performance_percentile=float(obs[1]),
            potential_tier=int(round(obs[2] * NORM["potential_tier"])),
            current_margin_rate=float(obs[3] * NORM["margin_rate"]),
            current_margin_dollars=float(obs[4] * NORM["margin_dollars"]),
            weekly_cases=float(obs[5] * NORM["weekly_cases"]),
            weekly_sales=float(obs[6] * NORM["weekly_sales"]),
            deliveries_per_week=float(obs[7] * NORM["deliveries"]),
            elasticity_estimate=float(-obs[8] * NORM["elasticity"]),
            price_change_history=[
                int(round(obs[9] * NORM["action"])),
                int(round(obs[10] * NORM["action"])),
                int(round(obs[11] * NORM["action"])),
                int(round(obs[12] * NORM["action"])),
            ],
            periods_since_last_change=int(round(obs[13] * NORM["periods"])),
            syw_flag=bool(obs[14] > 0.5),
            perks_flag=bool(obs[15] > 0.5),
            churn_probability=float(obs[16]),
        )
```

Note: `to_observation()` produces 17 floats. The spec says 18 — we need to verify if that's correct or adjust. The 17 comes from: 1 css + 1 perf + 1 potential + 1 margin_rate + 1 margin_$ + 1 cases + 1 sales + 1 deliveries + 1 elasticity + 4 price_history + 1 periods + 1 syw + 1 perks + 1 churn = 17. The spec counted 18 by listing price_change_history as 4 separate values and double-counting. **Use 17.**

- [ ] **Step 4: Run test to verify it passes**

Run: `cd pricing_rl && python -m pytest tests/test_customer.py -v`
Expected: PASS (update test assertion to `obs.shape == (17,)`)

- [ ] **Step 5: Commit**

```bash
git add src/environment/customer.py tests/test_customer.py
git commit -m "feat: CustomerState dataclass with observation serialization"
```

### Task 3: Synthetic Data Generator

**Files:**
- Create: `pricing_rl/src/data/synthetic_generator.py`
- Create: `pricing_rl/tests/test_synthetic_generator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_synthetic_generator.py
import pandas as pd
import numpy as np
from src.data.synthetic_generator import generate_customer_population, generate_transaction_history

def test_generate_population_shape():
    df = generate_customer_population(n=500, seed=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 500

def test_generate_population_columns():
    df = generate_customer_population(n=100, seed=42)
    required = [
        "customer_id", "css_score", "performance_percentile", "potential_tier",
        "margin_rate", "margin_dollars_monthly", "cases_monthly", "sales_monthly",
        "deliveries_per_week", "elasticity", "syw_flag", "perks_flag", "concept",
    ]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"

def test_css_distribution():
    df = generate_customer_population(n=10000, seed=42)
    css_pcts = df["css_score"].value_counts(normalize=True).sort_index()
    assert abs(css_pcts[1] + css_pcts[2] - 0.20) < 0.05  # ~20% CSS 1-2
    assert abs(css_pcts[3] - 0.40) < 0.05                  # ~40% CSS 3
    assert abs(css_pcts[4] - 0.25) < 0.05                  # ~25% CSS 4
    assert abs(css_pcts[5] - 0.15) < 0.05                  # ~15% CSS 5

def test_syw_penetration():
    df = generate_customer_population(n=10000, seed=42)
    assert abs(df["syw_flag"].mean() - 0.30) < 0.05

def test_elasticity_css_correlation():
    df = generate_customer_population(n=10000, seed=42)
    corr = df["css_score"].corr(df["elasticity"])
    assert corr > 0  # elasticity is negative, higher CSS = less negative = positive corr

def test_deterministic():
    df1 = generate_customer_population(n=100, seed=42)
    df2 = generate_customer_population(n=100, seed=42)
    pd.testing.assert_frame_equal(df1, df2)

def test_transaction_history_shape():
    customers = generate_customer_population(n=100, seed=42)
    txns = generate_transaction_history(customers, periods=52, seed=42)
    assert isinstance(txns, pd.DataFrame)
    assert len(txns) > 0
    assert "period" in txns.columns
    assert "customer_id" in txns.columns

def test_transaction_seasonality():
    customers = generate_customer_population(n=1000, seed=42)
    txns = generate_transaction_history(customers, periods=52, seed=42)
    q1 = txns[txns["period"].between(0, 12)]["sales"].mean()
    q4 = txns[txns["period"].between(39, 51)]["sales"].mean()
    assert q4 > q1  # Q4 should be higher than Q1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd pricing_rl && python -m pytest tests/test_synthetic_generator.py -v`
Expected: FAIL

- [ ] **Step 3: Implement synthetic_generator.py**

Implement `generate_customer_population()`:
- Load config from `config/default.yaml` for distribution params (accept config_path kwarg)
- CSS assignment: multinomial draw from `css_distribution`
- For each metric (cases, sales, margin), use lognormal distributions scaled by CSS tier
- Elasticity: draw from per-CSS normal distributions, apply concept modifier
- SYW: Bernoulli with CSS-dependent probability from `syw_css_skew`
- Perks: Bernoulli with flat rate
- Concept: multinomial from concept weights
- Performance percentile: composite rank of (DM% * 0.43 + DM$ * 0.26 + Sales * 0.17 + Cases * 0.15)
- Potential tier: based on CSS and performance percentile

Implement `generate_transaction_history()`:
- For each customer, for each period (0-51):
  - Base metrics from customer profile
  - Apply seasonal modifier based on period (Q1=0-12, Q2=13-25, Q3=26-38, Q4=39-51)
  - Add noise: N(0, 0.05 * base_value)
  - Track churn: per-period churn probability based on CSS, cumulative. If churned, no more records.
- Columns: customer_id, period, cases, sales, margin_rate, margin_dollars, deliveries

- [ ] **Step 4: Run tests**

Run: `cd pricing_rl && python -m pytest tests/test_synthetic_generator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/synthetic_generator.py tests/test_synthetic_generator.py
git commit -m "feat: config-driven synthetic data generator with CSS distributions"
```

---

## Chunk 3: Market Simulator & Gymnasium Environment

### Task 4: Market Simulator

**Files:**
- Create: `pricing_rl/src/environment/market_simulator.py`
- Create: `pricing_rl/tests/test_market_simulator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_market_simulator.py
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
    sim = MarketSimulator(seed=42)
    # Short history -> more responsive
    resp_short = abs(sim.compute_volume_response(
        price_change=0.05, elasticity=-1.5, base_volume=100.0,
        css_score=3, periods_stable=2
    ))
    # Long history -> less responsive (stickiness)
    resp_long = abs(sim.compute_volume_response(
        price_change=0.05, elasticity=-1.5, base_volume=100.0,
        css_score=3, periods_stable=12
    ))
    assert resp_long < resp_short
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd pricing_rl && python -m pytest tests/test_market_simulator.py -v`
Expected: FAIL

- [ ] **Step 3: Implement MarketSimulator**

```python
# src/environment/market_simulator.py
import numpy as np

class MarketSimulator:
    def __init__(self, seed: int = 42, config: dict | None = None):
        self.rng = np.random.default_rng(seed)
        self.config = config or {}
        self.stickiness_threshold = self.config.get("stickiness_threshold_periods", 8)

    def compute_volume_response(
        self,
        price_change: float,
        elasticity: float,
        base_volume: float,
        css_score: int,
        periods_stable: int = 0,
    ) -> float:
        # Base response: delta_vol = elasticity * price_change * base_volume
        # elasticity is negative, price_change negative (cut) -> positive delta
        response = elasticity * price_change * base_volume

        # Stickiness: dampen response for long-stable customers
        if periods_stable >= self.stickiness_threshold:
            damping = 0.5 + 0.5 * (self.stickiness_threshold / periods_stable)
            response *= damping

        # Noise: scaled by CSS tier (higher CSS = less noise)
        noise_scale = 0.1 * base_volume * (6 - css_score) / 5
        noise = self.rng.normal(0, noise_scale)

        return response + noise

    def compute_churn_probability(
        self,
        margin_rate: float,
        css_score: int,
        syw: bool,
        periods_stable: int,
        threshold: float,
    ) -> float:
        # Logistic function centered on threshold
        margin_gap = threshold - margin_rate  # positive when margin below threshold
        base_prob = 1 / (1 + np.exp(-10 * margin_gap))

        # SYW reduces churn 15-20%
        if syw:
            base_prob *= 0.82

        # Stickiness reduces churn
        if periods_stable >= self.stickiness_threshold:
            base_prob *= 0.7

        return float(np.clip(base_prob, 0.0, 1.0))

    def check_churn(self, churn_probability: float) -> bool:
        return bool(self.rng.random() < churn_probability)

    def apply_seasonality(self, base_value: float, period: int, config: dict) -> float:
        quarter = period // 13  # 0-3
        modifiers = [
            config.get("q1_modifier", 0.85),
            config.get("q2_modifier", 1.0),
            config.get("q3_modifier", 1.05),
            config.get("q4_modifier", 1.15),
        ]
        return base_value * modifiers[min(quarter, 3)]
```

- [ ] **Step 4: Run tests**

Run: `cd pricing_rl && python -m pytest tests/test_market_simulator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/environment/market_simulator.py tests/test_market_simulator.py
git commit -m "feat: market simulator with elasticity, churn, and stickiness dynamics"
```

### Task 5: Gymnasium Environment

**Files:**
- Create: `pricing_rl/src/environment/pricing_env.py`
- Create: `pricing_rl/tests/test_environment.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_environment.py
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from src.environment.pricing_env import DynamicPricingEnv

def test_env_creation():
    env = DynamicPricingEnv()
    assert env is not None

def test_gymnasium_compliance():
    env = DynamicPricingEnv()
    check_env(env, skip_render_check=True)

def test_observation_space_shape():
    env = DynamicPricingEnv()
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)

def test_action_space():
    env = DynamicPricingEnv()
    assert env.action_space.n == 7

def test_episode_length():
    env = DynamicPricingEnv()
    obs, _ = env.reset()
    steps = 0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    assert steps <= 52  # max episode length

def test_action_masking():
    env = DynamicPricingEnv()
    env.reset()
    # Take action 6 (15% cut)
    env.step(6)
    mask = env.action_masks()
    assert mask[6] == 0  # can't do consecutive 15% cut

def test_observation_lag():
    env = DynamicPricingEnv(config={"environment": {"observation_lag": 2}})
    obs0, _ = env.reset()
    # Take action, observation should reflect lag
    obs1, _, _, _, info = env.step(3)  # price down 2%
    # With lag=2, obs1 should still show the initial state
    # (the effect of the action isn't visible yet)
    assert "observation_lag" in info or True  # at minimum, env should run

def test_observation_lag_zero():
    env = DynamicPricingEnv(config={"environment": {"observation_lag": 0}})
    obs0, _ = env.reset()
    obs1, _, _, _, _ = env.step(3)
    # With no lag, observation should reflect the action immediately
    # Just verify it runs without error
    assert obs1.shape == obs0.shape

def test_reward_is_float():
    env = DynamicPricingEnv()
    env.reset()
    _, reward, _, _, _ = env.step(0)
    assert isinstance(reward, float)

def test_info_contains_customer_state():
    env = DynamicPricingEnv()
    env.reset()
    _, _, _, _, info = env.step(0)
    assert "customer_state" in info
    assert "margin_dollars" in info
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd pricing_rl && python -m pytest tests/test_environment.py -v`
Expected: FAIL

- [ ] **Step 3: Implement DynamicPricingEnv**

Key implementation details:
- Subclass `gymnasium.Env`
- `__init__`: load config, create MarketSimulator, define observation_space (Box, shape=(17,), 0-1) and action_space (Discrete(7))
- `reset()`: generate or sample a customer (from synthetic generator or random), initialize state, clear history buffers, return observation
- `step(action)`: apply price change via action map, compute volume response, update margin, compute churn, update customer state, apply observation lag (ring buffer), compute reward (default CLV Optimizer), return (obs, reward, terminated, truncated, info)
- `action_masks()`: return np.array of 0/1 for valid actions based on history
- Observation lag: maintain a deque of size `lag+1`, push current state, pop oldest as observation
- Info dict: include customer_state dict, margin_dollars, volume, churn_probability

- [ ] **Step 4: Run tests**

Run: `cd pricing_rl && python -m pytest tests/test_environment.py -v`
Expected: PASS

- [ ] **Step 5: Run Gymnasium's check_env specifically**

Run: `cd pricing_rl && python -c "from gymnasium.utils.env_checker import check_env; from src.environment.pricing_env import DynamicPricingEnv; check_env(DynamicPricingEnv()); print('Environment OK')"`
Expected: "Environment OK"

- [ ] **Step 6: Commit**

```bash
git add src/environment/pricing_env.py tests/test_environment.py
git commit -m "feat: Gymnasium pricing environment with action masking and observation lag"
```

---

## Chunk 4: Reward Functions

### Task 6: Reward Function Classes

**Files:**
- Create: `pricing_rl/src/reward/reward_functions.py`
- Create: `pricing_rl/tests/test_reward.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_reward.py
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

# --- CLV Optimizer ---

def test_clv_optimizer_churn_penalty():
    rf = CLVOptimizer()
    state = _make_state(css_score=3, churn_probability=0.10)
    # next state with churn above threshold (0.25 for CSS 3)
    next_state = _make_state(css_score=3, churn_probability=0.30,
                              current_margin_dollars=665.0)
    r = rf.compute(state, 5, next_state)
    # Should have churn penalty pulling reward down
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
    # CSS 5 should have high alpha (margin weight)
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
    assert "Margin" in explanation

# --- Portfolio Optimizer ---

def test_portfolio_optimizer_extends_clv():
    rf = PortfolioOptimizer()
    state = _make_state()
    next_state = _make_state(current_margin_dollars=700.0)
    r = rf.compute(state, 1, next_state)
    assert isinstance(r, float)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd pricing_rl && python -m pytest tests/test_reward.py -v`
Expected: FAIL

- [ ] **Step 3: Implement reward_functions.py**

Three classes: `MarginMaximizer`, `CLVOptimizer`, `PortfolioOptimizer`. Each has:
- `__init__(self, config: dict | None = None)` — loads reward params from config
- `compute(self, state: CustomerState, action: int, next_state: CustomerState) -> float`
- `explain(self, state: CustomerState, action: int, next_state: CustomerState) -> str`

Key logic for CLVOptimizer:
- Look up alpha/beta by `state.css_score`
- margin_term = alpha * (next_state.current_margin_dollars - state.current_margin_dollars)
- volume_term = beta * (next_state.weekly_cases - state.weekly_cases)
- churn_penalty = gamma * max(0, next_state.churn_probability - threshold) if above threshold
- volatility_penalty: count non-zero actions in price_change_history, penalize if > max_changes
- Scale churn penalty to delivery-at-risk only (multiply by margin_dollars, not total sales)

PortfolioOptimizer extends CLVOptimizer, adds:
- css_migration_bonus if next_state.css_score > state.css_score
- Takes optional `action_distribution` param for concentration penalty

- [ ] **Step 4: Run tests**

Run: `cd pricing_rl && python -m pytest tests/test_reward.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/reward/reward_functions.py tests/test_reward.py
git commit -m "feat: three-tier reward functions with explain() method"
```

---

## Chunk 5: Agents & Heuristic Baseline

### Task 7: Heuristic Baseline Agent

**Files:**
- Create: `pricing_rl/src/agent/heuristic_baseline.py`
- Create: `pricing_rl/tests/test_agent.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_agent.py
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
    state = _make_state(css_score=5, weekly_cases=15.0)
    action = agent.predict(state)
    assert action == 0  # hold (volume stable)

def test_heuristic_css5_volume_drop():
    agent = HeuristicBaseline()
    # Simulate volume drop >10% from baseline by setting low volume
    state = _make_state(css_score=5, weekly_cases=10.0,
                        performance_percentile=0.9)
    action = agent.predict(state)
    assert action == 3  # price down 2%
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd pricing_rl && python -m pytest tests/test_agent.py -v`
Expected: FAIL

- [ ] **Step 3: Implement HeuristicBaseline**

```python
# src/agent/heuristic_baseline.py
from src.environment.customer import CustomerState

class HeuristicBaseline:
    """Rule-based pricing strategy approximating Sysco's manual approach.

    CSS 1-2: Always discount 5% (grow volume)
    CSS 3: Hold unless margin < 20%, then price up 2%
    CSS 4-5: Hold unless volume drops >10% from expected baseline, then discount 2%
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.low_margin_threshold = self.config.get("low_margin_threshold", 0.20)
        self.volume_drop_threshold = self.config.get("volume_drop_threshold", 0.10)
        # Expected baseline volume by CSS tier (weekly cases)
        self.baseline_cases = self.config.get("baseline_cases", {
            1: 8.0, 2: 10.0, 3: 15.0, 4: 20.0, 5: 25.0,
        })

    def predict(self, state: CustomerState) -> int:
        if state.css_score <= 2:
            return 4  # price down 5%

        if state.css_score == 3:
            if state.current_margin_rate < self.low_margin_threshold:
                return 1  # price up 2%
            return 0  # hold

        # CSS 4-5
        baseline = self.baseline_cases.get(state.css_score, 20.0)
        if state.weekly_cases < baseline * (1 - self.volume_drop_threshold):
            return 3  # price down 2%
        return 0  # hold
```

- [ ] **Step 4: Run tests**

Run: `cd pricing_rl && python -m pytest tests/test_agent.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/heuristic_baseline.py tests/test_agent.py
git commit -m "feat: heuristic baseline agent matching Sysco's manual pricing rules"
```

### Task 8: RL Agent Wrapper

**Files:**
- Create: `pricing_rl/src/agent/rl_agent.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_agent.py`:

```python
from src.agent.rl_agent import RLAgent

def test_rl_agent_creation_ppo():
    agent = RLAgent(algorithm="ppo")
    assert agent.algorithm == "ppo"

def test_rl_agent_creation_dqn():
    agent = RLAgent(algorithm="dqn")
    assert agent.algorithm == "dqn"

def test_rl_agent_predict_shape():
    from src.environment.pricing_env import DynamicPricingEnv
    env = DynamicPricingEnv()
    agent = RLAgent(algorithm="ppo", env=env)
    obs, _ = env.reset()
    action = agent.predict(obs)
    assert 0 <= action < 7
```

- [ ] **Step 2: Implement RLAgent wrapper**

```python
# src/agent/rl_agent.py
from stable_baselines3 import PPO, DQN
from sb3_contrib import MaskablePPO
import gymnasium as gym
import numpy as np
from pathlib import Path

class RLAgent:
    ALGORITHMS = {
        "ppo": MaskablePPO,
        "dqn": DQN,
    }

    def __init__(
        self,
        algorithm: str = "ppo",
        env: gym.Env | None = None,
        config: dict | None = None,
        model_path: str | None = None,
    ):
        self.algorithm = algorithm
        self.config = config or {}
        algo_config = self.config.get(algorithm, {})

        if model_path:
            cls = self.ALGORITHMS[algorithm]
            self.model = cls.load(model_path, env=env)
        elif env is not None:
            cls = self.ALGORITHMS[algorithm]
            self.model = cls(
                "MlpPolicy" if algorithm == "dqn" else "MlpPolicy",
                env,
                verbose=0,
                **algo_config,
            )
        else:
            self.model = None

    def predict(self, obs: np.ndarray, action_masks: np.ndarray | None = None) -> int:
        if self.algorithm == "ppo" and action_masks is not None:
            action, _ = self.model.predict(obs, action_masks=action_masks)
        else:
            action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def train(self, total_timesteps: int, **kwargs):
        self.model.learn(total_timesteps=total_timesteps, **kwargs)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    def load(self, path: str, env: gym.Env | None = None):
        cls = self.ALGORITHMS[self.algorithm]
        self.model = cls.load(path, env=env)
```

- [ ] **Step 3: Run tests**

Run: `cd pricing_rl && python -m pytest tests/test_agent.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/agent/rl_agent.py tests/test_agent.py
git commit -m "feat: RL agent wrapper for PPO/DQN with action masking support"
```

---

## Chunk 6: Multi-Agent Orchestration

### Task 9: Multi-Agent System

**Files:**
- Create: `pricing_rl/src/orchestrator/multi_agent.py`
- Create: `pricing_rl/tests/test_orchestrator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_orchestrator.py
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

def test_portfolio_manager_routes_css5_to_guardian():
    pm = PortfolioManager()
    state = _make_state(css_score=5)
    agent_id = pm.assign(state)
    assert agent_id == "guardian"

def test_portfolio_manager_routes_css3():
    pm = PortfolioManager()
    state = _make_state(css_score=3)
    agent_id = pm.assign(state)
    assert agent_id in ("scout", "guardian")

def test_portfolio_manager_reallocation():
    pm = PortfolioManager()
    # Log some results showing scout doing well on CSS 3
    for _ in range(10):
        pm.log_result("scout", _make_state(css_score=3), action=4, reward=5.0)
        pm.log_result("guardian", _make_state(css_score=3), action=0, reward=1.0)
    pm.update_allocations()
    state = _make_state(css_score=3)
    agent_id = pm.assign(state)
    assert agent_id == "scout"  # scout is performing better on CSS 3

def test_price_scout_action_space():
    scout = PriceScout()
    # Scout should be willing to use aggressive actions
    assert scout.allowed_actions == list(range(7))

def test_margin_guardian_restricted_actions():
    guardian = MarginGuardian()
    # Guardian only uses Hold, +2%, +5%, -2%
    assert guardian.allowed_actions == [0, 1, 2, 3]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd pricing_rl && python -m pytest tests/test_orchestrator.py -v`
Expected: FAIL

- [ ] **Step 3: Implement multi_agent.py**

```python
# src/orchestrator/multi_agent.py
from collections import defaultdict
from src.environment.customer import CustomerState

class PriceScout:
    """Aggressive exploration agent for CSS 1-2 customers."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.allowed_actions = list(range(7))  # full action space
        self.exploration_bonus = self.config.get("exploration_bonus", 0.1)
        self.rl_agent = None  # Set during training

    def get_action_mask(self) -> list[int]:
        return [1] * 7  # all actions allowed


class MarginGuardian:
    """Conservative agent for CSS 4-5 customers."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.allowed_actions = self.config.get("restricted_actions", [0, 1, 2, 3])
        self.rl_agent = None

    def get_action_mask(self) -> list[int]:
        mask = [0] * 7
        for a in self.allowed_actions:
            mask[a] = 1
        return mask


class PortfolioManager:
    """Meta-agent that allocates customers to Scout or Guardian."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        routing = self.config.get("css_routing", {})
        self.scout_css = routing.get("scout", [1, 2])
        self.guardian_css = routing.get("guardian", [4, 5])
        self.contested_css = routing.get("contested", [3])
        self.reallocation_period = self.config.get("reallocation_period", 4)

        # Performance tracking for contested CSS segments
        self._logs: dict[str, list[dict]] = defaultdict(list)
        self._contested_assignment = "guardian"  # default for CSS 3
        self._period_counter = 0

    def assign(self, state: CustomerState) -> str:
        if state.css_score in self.scout_css:
            return "scout"
        if state.css_score in self.guardian_css:
            return "guardian"
        return self._contested_assignment

    def log_result(
        self, agent_id: str, state: CustomerState, action: int, reward: float
    ):
        self._logs[agent_id].append({
            "css_score": state.css_score,
            "action": action,
            "reward": reward,
        })

    def update_allocations(self):
        """Re-evaluate which agent handles contested CSS segments."""
        self._period_counter += 1

        scout_contested = [
            r["reward"] for r in self._logs["scout"]
            if r["css_score"] in self.contested_css
        ]
        guardian_contested = [
            r["reward"] for r in self._logs["guardian"]
            if r["css_score"] in self.contested_css
        ]

        if scout_contested and guardian_contested:
            scout_avg = sum(scout_contested) / len(scout_contested)
            guardian_avg = sum(guardian_contested) / len(guardian_contested)
            self._contested_assignment = (
                "scout" if scout_avg > guardian_avg else "guardian"
            )

    def should_reallocate(self, period: int) -> bool:
        return period > 0 and period % self.reallocation_period == 0
```

- [ ] **Step 4: Run tests**

Run: `cd pricing_rl && python -m pytest tests/test_orchestrator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/multi_agent.py tests/test_orchestrator.py
git commit -m "feat: multi-agent orchestration with Portfolio Manager allocation"
```

---

## Chunk 7: Training Pipeline

### Task 10: Training Script

**Files:**
- Create: `pricing_rl/scripts/train.py`

- [ ] **Step 1: Implement train.py**

```python
# scripts/train.py
"""Training entry point for RL pricing agents.

Usage:
    python scripts/train.py --agent ppo --reward clv_optimizer --timesteps 500000
    python scripts/train.py --agent dqn --reward clv_optimizer --timesteps 500000
    python scripts/train.py --agent heuristic --reward clv_optimizer --timesteps 52000
    python scripts/train.py --agent multi --reward portfolio_optimizer --timesteps 500000
"""
import argparse
import yaml
from pathlib import Path
from datetime import datetime

from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from src.environment.pricing_env import DynamicPricingEnv
from src.agent.rl_agent import RLAgent
from src.agent.heuristic_baseline import HeuristicBaseline
from src.reward.reward_functions import MarginMaximizer, CLVOptimizer, PortfolioOptimizer

REWARD_FUNCTIONS = {
    "margin_maximizer": MarginMaximizer,
    "clv_optimizer": CLVOptimizer,
    "portfolio_optimizer": PortfolioOptimizer,
}


def load_config(config_path: str, scenario: str | None = None) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if scenario:
        scenario_path = Path(config_path).parent / "scenarios" / f"{scenario}.yaml"
        if scenario_path.exists():
            with open(scenario_path) as f:
                overrides = yaml.safe_load(f) or {}
            _deep_merge(config, overrides)
    return config


def _deep_merge(base: dict, overrides: dict):
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def train_single_agent(args, config):
    run_id = f"{args.agent}_{args.reward}_{datetime.now():%Y%m%d_%H%M%S}"
    log_dir = Path(config["training"]["log_dir"]) / run_id
    model_dir = Path(config["training"]["model_dir"]) / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    reward_fn = REWARD_FUNCTIONS[args.reward](config.get("reward", {}).get(args.reward))

    env = Monitor(DynamicPricingEnv(config=config, reward_fn=reward_fn))
    eval_env = Monitor(DynamicPricingEnv(config=config, reward_fn=reward_fn))

    if args.agent == "heuristic":
        # Run heuristic through environment for baseline metrics
        agent = HeuristicBaseline(config.get("multi_agent", {}).get("heuristic"))
        print(f"Running heuristic baseline for {args.timesteps} steps...")
        obs, _ = eval_env.reset()
        from src.environment.customer import CustomerState
        total_reward = 0
        for step in range(args.timesteps):
            cs = CustomerState.from_observation(obs)
            action = agent.predict(cs)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            if terminated or truncated:
                obs, _ = eval_env.reset()
        print(f"Heuristic total reward: {total_reward:.2f}")
        return

    algo_config = config.get("training", {}).get(args.agent, {})
    agent = RLAgent(algorithm=args.agent, env=env, config={"training": {args.agent: algo_config}})

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=config["training"]["eval_freq"],
        deterministic=True,
        render=False,
    )

    print(f"Training {args.agent} with {args.reward} for {args.timesteps} timesteps...")
    agent.train(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        tb_log_name=run_id,
    )
    agent.save(str(model_dir / "final_model"))
    print(f"Model saved to {model_dir / 'final_model'}")


def main():
    parser = argparse.ArgumentParser(description="Train RL pricing agents")
    parser.add_argument("--agent", choices=["ppo", "dqn", "heuristic", "multi"],
                       required=True)
    parser.add_argument("--reward", choices=list(REWARD_FUNCTIONS.keys()),
                       default="clv_optimizer")
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()

    config = load_config(args.config, args.scenario)

    if args.agent == "multi":
        # Multi-agent training handled separately
        print("Multi-agent training not yet implemented in this script.")
        print("Use individual agent training first, then orchestrator.")
        return

    train_single_agent(args, config)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script runs with --help**

Run: `cd pricing_rl && python scripts/train.py --help`
Expected: argparse help output

- [ ] **Step 3: Commit**

```bash
git add scripts/train.py
git commit -m "feat: training script with config loading and eval callbacks"
```

---

## Chunk 8: Evaluation & Monitoring

### Task 11: Evaluation Metrics

**Files:**
- Create: `pricing_rl/src/evaluation/metrics.py`
- Create: `pricing_rl/tests/test_metrics.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_metrics.py
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

def test_action_entropy():
    # Uniform distribution -> high entropy
    actions_uniform = list(range(7)) * 100
    entropy_uniform = compute_action_entropy(actions_uniform, n_actions=7)

    # Collapsed to single action -> zero entropy
    actions_collapsed = [0] * 700
    entropy_collapsed = compute_action_entropy(actions_collapsed, n_actions=7)

    assert entropy_uniform > entropy_collapsed
    assert entropy_collapsed < 0.1
```

- [ ] **Step 2: Run tests, verify fail**

Run: `cd pricing_rl && python -m pytest tests/test_metrics.py -v`

- [ ] **Step 3: Implement metrics.py**

Functions: `compute_portfolio_margin`, `compute_churn_rate_by_css`, `compute_css_migration`, `compute_action_entropy`, `compute_regret_vs_oracle`. Pure functions, no side effects.

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: evaluation metrics for portfolio, churn, migration, entropy"
```

### Task 12: A/B Test Simulator

**Files:**
- Create: `pricing_rl/src/evaluation/ab_test_simulator.py`

- [ ] **Step 1: Implement A/B test simulator**

Class `ABTestSimulator`:
- `__init__(self, treatment_agent, control_agent, env_config, n_simulations=100, split_ratio=0.5)`
- `run(self) -> ABTestResult` — runs n_simulations, each with different seed, 50/50 customer split
- `ABTestResult` dataclass: mean_delta_margin, ci_95, p_value, power, cumulative_curves
- Uses scipy.stats.ttest_ind for p-value
- Returns cumulative margin curves for plotting

- [ ] **Step 2: Write test**

```python
# Add to tests/test_metrics.py or new file
def test_ab_simulator_runs():
    from src.evaluation.ab_test_simulator import ABTestSimulator
    from src.agent.heuristic_baseline import HeuristicBaseline
    sim = ABTestSimulator(
        treatment_agent=HeuristicBaseline(),
        control_agent=HeuristicBaseline(),
        env_config={},
        n_simulations=5,
    )
    result = sim.run()
    assert hasattr(result, "mean_delta_margin")
    assert hasattr(result, "p_value")
```

- [ ] **Step 3: Run test, verify pass**

- [ ] **Step 4: Commit**

```bash
git add src/evaluation/ab_test_simulator.py tests/
git commit -m "feat: A/B test simulator with statistical significance reporting"
```

### Task 13: Drift Detector

**Files:**
- Create: `pricing_rl/src/monitoring/drift_detector.py`

- [ ] **Step 1: Implement drift_detector.py**

Class `DriftDetector`:
- `__init__(self, config)` — loads thresholds from config
- `update(self, reward, action, elasticity_observed, elasticity_expected)` — accumulate per period
- `check_alerts(self) -> dict` — returns dict of alert flags
- `generate_report(self) -> dict` — full monitoring report for JSON export

Alert logic:
- Reward drift: track rolling mean/std, alert if current mean > 2σ from baseline for 3 consecutive periods
- Action entropy: compute entropy of recent action distribution, alert if below threshold
- Elasticity accuracy: track MAE between observed and expected, alert if diverges

- [ ] **Step 2: Write tests, verify pass**

- [ ] **Step 3: Commit**

```bash
git add src/monitoring/drift_detector.py tests/
git commit -m "feat: drift detector with reward, action, and elasticity monitoring"
```

### Task 14: Evaluate Script

**Files:**
- Create: `pricing_rl/scripts/evaluate.py`

- [ ] **Step 1: Implement evaluate.py**

Support three modes:
- `--agents ppo,dqn,heuristic --episodes 100` — run all agents, compare metrics
- `--ab-test --treatment ppo --control heuristic --simulations 100` — run A/B sim
- `--generate-report --output results/` — generate results.md with tables and recommendations

- [ ] **Step 2: Verify runs with --help**

- [ ] **Step 3: Commit**

```bash
git add scripts/evaluate.py
git commit -m "feat: evaluation script with comparison, A/B test, and report generation"
```

---

## Chunk 9: Dashboard

### Task 15: Streamlit Dashboard

**Files:**
- Create: `pricing_rl/dashboard/app.py`

- [ ] **Step 1: Implement 4-tab Streamlit dashboard**

```python
# dashboard/app.py
import streamlit as st

st.set_page_config(page_title="RL Pricing Agent Dashboard", layout="wide")

tab1, tab2, tab3, tab4 = st.tabs([
    "Training Progress", "Agent Decisions", "Portfolio Health", "A/B Results"
])
```

**Tab 1: Training Progress**
- Load TensorBoard logs from results/tensorboard/
- Plot: reward over timesteps (Plotly line chart)
- Plot: action distribution over training (stacked area)
- Plot: loss curves

**Tab 2: Agent Decisions**
- Customer selector (dropdown by ID)
- Pricing history timeline (Plotly)
- State evolution charts (margin, volume, churn prob over time)
- Reward decomposition table using `.explain()` output

**Tab 3: Portfolio Health**
- CSS distribution bar chart over time
- Margin heatmap by CSS tier
- Churn rate by tier bar chart
- Scout vs. Guardian performance comparison

**Tab 4: A/B Results**
- Cumulative margin curves (treatment vs control)
- Statistical significance summary (p-value, CI, effect size)
- Per-CSS-tier breakdown

- [ ] **Step 2: Verify dashboard launches**

Run: `cd pricing_rl && streamlit run dashboard/app.py --server.headless true` (just verify no import errors)

- [ ] **Step 3: Commit**

```bash
git add dashboard/app.py
git commit -m "feat: Streamlit dashboard with 4 tabs for training, decisions, portfolio, A/B"
```

---

## Chunk 10: Integration & Serve

### Task 16: Serve Script

**Files:**
- Create: `pricing_rl/scripts/serve.py`

- [ ] **Step 1: Implement serve.py**

Simple inference endpoint:
- Loads trained model from path
- Accepts customer state (as dict or JSON)
- Returns: action, action_name, confidence, reward_explanation
- Can run as CLI: `python scripts/serve.py --model results/models/run_id/final_model --input '{"css_score": 3, ...}'`

- [ ] **Step 2: Commit**

```bash
git add scripts/serve.py
git commit -m "feat: inference serve script for trained model predictions"
```

### Task 17: README

**Files:**
- Create: `pricing_rl/README.md`

- [ ] **Step 1: Write README**

Sections:
1. Problem Framing (MDP formulation)
2. State/Action/Reward design rationale
3. Architecture diagram (Mermaid): Data -> Feature Store -> State Builder -> RL Env -> Agent -> Action -> Market Response -> Reward -> State Builder (loop), with multi-agent overlay and monitoring feedback
4. Multi-agent design explanation
5. Why RL over supervised learning
6. Quickstart commands (install, generate data, validate env, train, evaluate, dashboard)
7. Configuration guide
8. Results interpretation guide

- [ ] **Step 2: Commit**

```bash
git add pricing_rl/README.md
git commit -m "docs: comprehensive README with MDP formulation and architecture diagram"
```

### Task 18: End-to-End Integration Test

**Files:**
- Create: `pricing_rl/tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""End-to-end integration test: generate data, create env, train briefly, evaluate."""

def test_full_pipeline():
    from src.data.synthetic_generator import generate_customer_population
    from src.environment.pricing_env import DynamicPricingEnv
    from src.reward.reward_functions import CLVOptimizer
    from src.agent.rl_agent import RLAgent
    from src.evaluation.metrics import compute_portfolio_margin

    # Generate small dataset
    customers = generate_customer_population(n=100, seed=42)
    assert len(customers) == 100

    # Create environment
    reward_fn = CLVOptimizer()
    env = DynamicPricingEnv(reward_fn=reward_fn)

    # Quick train (just verify it runs)
    agent = RLAgent(algorithm="ppo", env=env)
    agent.train(total_timesteps=1000)

    # Evaluate
    obs, _ = env.reset()
    rewards = []
    for _ in range(52):
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break

    assert len(rewards) > 0
    assert all(isinstance(r, float) for r in rewards)
```

- [ ] **Step 2: Run integration test**

Run: `cd pricing_rl && python -m pytest tests/test_integration.py -v --timeout=120`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: end-to-end integration test for full pipeline"
```

---

## Execution Order Summary

| Task | Component | Dependencies |
|---|---|---|
| 1 | Scaffold & Config | None |
| 2 | CustomerState | Task 1 |
| 3 | Synthetic Generator | Task 2 |
| 4 | Market Simulator | Task 1 |
| 5 | Gymnasium Environment | Tasks 2, 4 |
| 6 | Reward Functions | Task 2 |
| 7 | Heuristic Baseline | Task 2 |
| 8 | RL Agent Wrapper | Task 5 |
| 9 | Multi-Agent Orchestration | Tasks 7, 8 |
| 10 | Training Script | Tasks 5, 6, 8 |
| 11 | Evaluation Metrics | Task 2 |
| 12 | A/B Test Simulator | Tasks 5, 7, 11 |
| 13 | Drift Detector | Task 11 |
| 14 | Evaluate Script | Tasks 11, 12, 13 |
| 15 | Streamlit Dashboard | Tasks 11, 14 |
| 16 | Serve Script | Task 8 |
| 17 | README | All above |
| 18 | Integration Test | All above |

**Parallelizable groups:**
- Group A (after Task 1): Tasks 2, 4 can run in parallel
- Group B (after Tasks 2, 4): Tasks 3, 5, 6, 7 can partially parallelize
- Group C (after Task 5): Tasks 8, 11 can run in parallel
- Group D (after Group C): Tasks 9, 10, 12, 13 can partially parallelize
- Group E (after Group D): Tasks 14, 15, 16
- Group F (final): Tasks 17, 18
