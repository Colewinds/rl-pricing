# Observation Lag Analysis

## Setup

PPO agents were trained with identical hyperparameters (200k timesteps for lag=0 and lag=4;
500k for lag=2 default) under three observation lag settings:

- **lag=0**: Agent sees the current state immediately
- **lag=2**: Agent sees state from 2 periods ago (default, simulates real-world data delay)
- **lag=4**: Agent sees state from 4 periods ago (high latency scenario)

Each agent was evaluated over 100 episodes using the CLV Optimizer reward function.

## Results

| Lag | Mean Reward | Mean Episode Margin ($) | Mean Episode Length |
|-----|-------------|-------------------------|-------------------|
| 0   | 2.7433      | $1,899.31               | 4.7 steps         |
| 2   | 1.2654      | $1,532.54               | 4.4 steps         |
| 4   | -0.5722     | $1,412.47               | 4.0 steps         |

## Key Findings

1. **Performance degrades monotonically with lag.** Each 2-period increase in observation
   lag reduces mean reward by approximately 1.5-1.8 points.

2. **Margin impact is substantial.** Moving from lag=0 to lag=4 costs ~$487/episode
   (~26% reduction), representing real revenue at risk from stale data.

3. **Episode length shortens with lag.** Higher lag leads to more churned customers
   (shorter episodes), suggesting delayed observations cause the agent to react too
   slowly to deteriorating customer health.

4. **Lag=2 is a reasonable production default.** It balances realism (Sysco's data
   pipelines have ~2 week latency) with acceptable performance degradation (~19% vs
   perfect information).

## Implications for Production

- Investing in reducing data pipeline latency from 2 weeks to near-real-time could
  yield a ~24% improvement in margin outcomes.
- If lag cannot be reduced, the agent could be retrained with augmented lag values
  (curriculum learning from lag=0 to lag=2) to build robustness.
- The lag=4 results suggest that any system with more than 4 weeks of data staleness
  may not benefit from RL pricing over a simple heuristic.
