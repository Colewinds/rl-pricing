# RL Dynamic Pricing - Evaluation Results

Generated: 2026-03-12 13:46:42

## Agent Comparison (100 episodes each)

| Metric | heuristic | ppo | dqn |
|--------|------|------|------|
| Mean Reward | -2.1871 | 1.6193 | -2.0958 |
| Total Reward | -774 | 423 | -1,295 |
| Mean Episode Margin ($) | 1,065.50 | 716.91 | 2,348.42 |
| Action Entropy | 0.534 | 0.552 | 0.377 |
| Episodes | 100 | 100 | 100 |
| Total Steps | 354 | 261 | 618 |

## CSS Migration

| Agent | Upgrades | Downgrades | Same |
|-------|----------|------------|------|
| heuristic | 0 | 0 | 100 |
| ppo | 0 | 0 | 100 |
| dqn | 0 | 0 | 100 |

## Churn Rate by CSS Tier

| CSS Tier | heuristic | ppo | dqn |
|----------|------|------|------|
| CSS 1 | 1.000 | 1.000 | 1.000 |
| CSS 2 | 1.000 | 1.000 | 1.000 |
| CSS 3 | 1.000 | 1.000 | 1.000 |
| CSS 4 | 1.000 | 1.000 | 0.966 |
| CSS 5 | 1.000 | 1.000 | 0.909 |

## Recommendations

- **DQN** achieves the highest mean episode margin ($2,348.42), a +120.4% difference vs heuristic baseline.
- Heuristic baseline mean episode margin: $1,065.50
