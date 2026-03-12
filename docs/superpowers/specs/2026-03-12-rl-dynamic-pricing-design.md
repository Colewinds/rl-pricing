# RL-Based Dynamic Pricing Agent — Design Spec

**Date:** 2026-03-12
**Scope:** Production-credible prototype for Sysco dynamic pricing case study
**Framework:** Stable-Baselines3 + custom multi-agent orchestration + Gymnasium

---

## 1. Project Structure

```
pricing_rl/
├── pyproject.toml
├── config/
│   ├── default.yaml
│   └── scenarios/
│       ├── conservative.yaml
│       ├── aggressive.yaml
│       └── balanced.yaml
├── src/
│   ├── __init__.py
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── pricing_env.py
│   │   ├── customer.py
│   │   └── market_simulator.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── rl_agent.py
│   │   └── heuristic_baseline.py
│   ├── reward/
│   │   ├── __init__.py
│   │   └── reward_functions.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── synthetic_generator.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── ab_test_simulator.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── drift_detector.py
│   └── orchestrator/
│       ├── __init__.py
│       └── multi_agent.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── serve.py
├── dashboard/
│   └── app.py
└── tests/
    ├── test_environment.py
    ├── test_reward.py
    ├── test_agent.py
    └── test_orchestrator.py
```

All hyperparameters, environment params, synthetic data distributions, reward weights, and monitoring thresholds live in `config/default.yaml`. Scenario files override specific sections.

---

## 2. Environment Design (MDP Formulation)

### State Space (18 floats, normalized to [0,1])

| Feature | Type | Range | Description |
|---|---|---|---|
| css_score | int | 1-5 | Customer Segmentation Score |
| performance_percentile | float | 0-1 | Composite performance rank |
| potential_tier | int | 0-2 | Low/Medium/High encoded |
| current_margin_rate | float | 0-1 | Current DM% |
| current_margin_dollars | float | 0-1 | Current DM$ (normalized) |
| weekly_cases | float | 0-1 | Avg cases/week (normalized) |
| weekly_sales | float | 0-1 | Avg sales $/week (normalized) |
| deliveries_per_week | float | 0-1 | Delivery frequency |
| elasticity_estimate | float | 0-1 | Estimated price sensitivity |
| price_change_history[0-3] | float | 0-1 | Last 4 actions (encoded) |
| periods_since_last_change | float | 0-1 | Cooldown tracker (normalized) |
| syw_flag | float | 0/1 | Sysco Your Way enrollment |
| perks_flag | float | 0/1 | Perks program enrollment |
| churn_probability | float | 0-1 | Predicted churn risk |

### Action Space — Discrete(7)

| Action | Effect |
|---|---|
| 0 | Hold (no change) |
| 1 | Price up 2% |
| 2 | Price up 5% |
| 3 | Price down 2% |
| 4 | Price down 5% |
| 5 | Price down 10% |
| 6 | Price down 15% |

### Action Masking

- No consecutive 15% cuts
- No price-up within 2 periods of a price-down
- Implemented via SB3 `MaskableMultiInputPolicy`

### Transition Dynamics

- **Volume:** `delta_volume = elasticity * delta_price + N(0, sigma)` where sigma scales with CSS tier
- **Elasticity:** CSS 1-2 more elastic, CSS 4-5 less elastic (configured in YAML)
- **Churn:** Logistic function of margin squeeze below segment threshold. SYW flag reduces churn rate 15-20%. Stickiness effect: 8+ stable periods reduces sensitivity.
- **Observation lag:** Configurable ring buffer (default 2 periods). Agent sees state from N steps ago.

### Episode Structure

- 52 timesteps = 1 year of weekly decisions
- Early termination on customer churn
- Single-customer env — portfolio handled by multi-agent layer running multiple instances

---

## 3. Reward Functions

Three callable classes, each with `.compute(state, action, next_state)` and `.explain(state, action, next_state)`.

### Reward 1: Margin Maximizer (myopic baseline)

`R = delta(margin_dollars)`

Exists to demonstrate why naive reward design fails.

### Reward 2: CLV Optimizer (primary)

`R = alpha * delta(margin$) + beta * delta(volume) - gamma * churn_penalty - delta_coeff * volatility_penalty`

- **alpha/beta:** CSS-tier-dependent. CSS 4-5: alpha high (protect margin). CSS 1-2: beta high (grow wallet share).
- **churn_penalty:** Large negative when churn_prob exceeds tier threshold (CSS 1-2: 40%, CSS 3: 25%, CSS 4-5: 15%). Scoped to at-risk delivery sales only, not total customer sales.
- **volatility_penalty:** Triggered when >2 price changes in last 4 periods.
- **Lifetime discount:** Retaining CSS 5 over 3 years valued higher than CSS 1.

### Reward 3: Portfolio Optimizer (strategic)

Extends CLV Optimizer with:
- Bonus for CSS migration upward
- Penalty for action concentration (too many customers getting the same action)

### `.explain()` output format

> "Margin: +$42 (alpha=0.6) | Volume: +3 cases (beta=0.3) | Churn: no penalty (prob=0.12 < threshold 0.25) | Volatility: -0.5 (3 changes in 4 periods) | Total: +24.1"

---

## 4. Multi-Agent Orchestration

### Agent 1: Price Scout

- Trained on CSS 1-2 customers
- Higher exploration rate
- Uses Reward 1 (Margin Maximizer) — intentionally aggressive
- Outputs: action + elasticity signal from observed volume response

### Agent 2: Margin Guardian

- Trained on CSS 4-5 customers
- Restricted action space: Hold, +2%, +5%, -2% (no deep discounts)
- Uses Reward 2 (CLV Optimizer) with high alpha weight

### Agent 3: Portfolio Manager

- Starts as rule-based allocator:
  - CSS 1-2 -> Scout
  - CSS 4-5 -> Guardian
  - CSS 3 -> whichever agent has better recent performance on similar customers
- Interface: `manager.assign(customer_state) -> agent_id`
- Reads action logs every 4 periods, updates allocation rules
- Upgrade path: replaceable with contextual bandit or RL policy (same interface)

### Coordination Flow (per timestep)

1. Portfolio Manager receives batch of customer states
2. Assigns each customer to Scout or Guardian
3. Each agent runs `predict(obs)` on assigned customers
4. Actions execute in environment
5. Results logged — Manager reads logs every 4 periods

### Training Strategy

- Phase 1: Train Scout and Guardian independently on their CSS segments
- Phase 2: Fine-tune with Portfolio Manager allocating in the loop
- Phase 3: (optional) Replace rule-based Manager with learned allocator

---

## 5. Synthetic Data Generator

### `generate_customer_population(n=10000, seed=42) -> pd.DataFrame`

- All distributions from config (CSS splits, percentiles, correlations)
- Defaults: Cases P50~60/mo, Sales P50~$3k/mo, DM% P50~24%, DM$ P50~$665/mo
- Lognormal distributions for cases, sales, margin dollars
- CSS distribution: ~20% CSS 1-2, ~40% CSS 3, ~25% CSS 4, ~15% CSS 5
- Elasticity inversely correlated with CSS (r~-0.4)
- SYW: 30% overall, skewing CSS 3-5. Perks: 15% overall
- Concept field: QSR, casual dining, fine dining, institutional, healthcare
- Deterministic given seed

### `generate_transaction_history(customers_df, periods=52, seed=42) -> pd.DataFrame`

- Weekly records per customer with seasonal patterns (Q4 bump, Q1 dip)
- Natural churn events (~5-8% annual baseline, higher for CSS 1-2)
- Deterministic given seed

The generator bootstraps the environment. The Gymnasium env then evolves states via its own transition dynamics.

---

## 6. Evaluation & Monitoring

### Metrics

- Total portfolio margin $ (annualized)
- Average margin rate by CSS tier
- Churn rate by CSS tier
- CSS migration: % up vs. down over 52 weeks
- Action entropy (collapse detection)
- Regret vs. oracle (perfect elasticity knowledge)
- All broken down by agent, reward function, scenario

### A/B Test Simulator

- 50/50 split: RL agent vs. heuristic baseline
- 100 simulations, different seeds
- Output: mean delta-margin, 95% CI, p-value, power analysis
- Viz: cumulative margin curves, treatment vs. control

### Drift Detector

- Reward distribution: alert if mean drifts >2sigma from training baseline
- Action distribution: alert if entropy drops below threshold
- Elasticity accuracy: observed vs. assumed
- 3 consecutive breaching periods required to fire alert
- Output: `monitoring_report.json`

### Observation Lag Analysis

Train identical agents with lag=0, 1, 2, 4 and plot performance degradation.

---

## 7. Dashboard (Streamlit)

**Tab 1: Training Progress** — Reward curves, loss, action distributions over training

**Tab 2: Agent Decisions** — Select a customer, see pricing history, state evolution, reward decomposition via `.explain()`

**Tab 3: Portfolio Health** — CSS distribution heatmap, margin by tier, churn rates, Scout vs. Guardian performance

**Tab 4: A/B Results** — Cumulative margin curves, significance tracker, confidence intervals

---

## 8. Scripts & Entry Points

### train.py

```bash
python scripts/train.py --agent ppo --reward clv_optimizer --timesteps 500000 --config config/default.yaml
python scripts/train.py --agent multi --reward portfolio_optimizer --timesteps 500000
```

- TensorBoard logging, checkpoints every 50k, eval every 10k, early stopping after 50k plateau

### evaluate.py

```bash
python scripts/evaluate.py --agents ppo,dqn,heuristic --episodes 100
python scripts/evaluate.py --ab-test --treatment ppo --control heuristic --simulations 100
python scripts/evaluate.py --generate-report --output results/
```

### serve.py

- Loads trained model, input customer state -> output action + confidence + reward explanation

---

## 9. Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| RL Framework | SB3 | Fast, debuggable, well-documented. RLlib is upgrade path. |
| Multi-agent | Custom Python over SB3 | Orchestration pattern, not MARL. Cleaner, more explainable. |
| Environment | Single-customer Gymnasium | Keeps env standard. Portfolio handled by orchestrator. |
| Observation lag | Configurable (default 2) | Enables lag-sensitivity analysis for presentations. |
| Synthetic data | Config-driven | All distributions in YAML for easy swap to real data. |
| Reward scoping | Delivery sales only | Avoids Sysco's known analytical trap of overweighting churn risk. |
| Portfolio Manager | Rule-based with learned upgrade path | Start simple, prove value, then upgrade interface stays same. |
