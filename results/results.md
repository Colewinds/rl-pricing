# RL Dynamic Pricing - Evaluation Results

Generated: 2026-03-12 19:38:18

## Agent Comparison (100 episodes each)

| Metric | heuristic | ppo | dqn |
|--------|------|------|------|
| Mean Reward | -1.3897 | 0.2154 | -0.1943 |
| Total Reward | -6,917 | 1,065 | -914 |
| Mean Episode Margin ($) | 10,439.27 | 13,467.45 | 12,384.32 |
| Action Entropy | 0.497 | 0.166 | 0.289 |
| Episodes | 100 | 100 | 100 |
| Total Steps | 4,977 | 4,945 | 4,706 |

## CSS Migration

| Agent | Upgrades | Downgrades | Same |
|-------|----------|------------|------|
| heuristic | 0 | 0 | 100 |
| ppo | 0 | 0 | 100 |
| dqn | 0 | 0 | 100 |

## Churn Rate by CSS Tier

| CSS Tier | heuristic | ppo | dqn |
|----------|------|------|------|
| CSS 1 | 0.286 | 0.333 | 0.583 |
| CSS 2 | 0.235 | 0.000 | 0.222 |
| CSS 3 | 0.147 | 0.053 | 0.128 |
| CSS 4 | 0.000 | 0.077 | 0.095 |
| CSS 5 | 0.000 | 0.000 | 0.053 |

## Recommendations

- **PPO** achieves the highest mean episode margin ($13,467.45), a +29.0% difference vs heuristic baseline.
- Heuristic baseline mean episode margin: $10,439.27
- WARNING: PPO has low action entropy (0.166), suggesting policy collapse.
- WARNING: DQN has low action entropy (0.289), suggesting policy collapse.
