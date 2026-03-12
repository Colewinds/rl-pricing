"""A/B test simulator for comparing pricing agents."""

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from src.environment.customer import CustomerState
from src.environment.pricing_env import DynamicPricingEnv


@dataclass
class ABTestResult:
    """Results from an A/B test simulation."""

    mean_delta_margin: float
    ci_95: tuple[float, float]
    p_value: float
    power: float
    treatment_margins: list[float] = field(default_factory=list)
    control_margins: list[float] = field(default_factory=list)
    cumulative_curves: dict = field(default_factory=dict)


class ABTestSimulator:
    """Simulates A/B tests between treatment and control agents.

    Runs n_simulations episodes, splitting customers 50/50 between
    treatment and control. Reports statistical significance via t-test.
    """

    def __init__(
        self,
        treatment_agent,
        control_agent,
        env_config: dict | None = None,
        n_simulations: int = 100,
        split_ratio: float = 0.5,
    ):
        self.treatment_agent = treatment_agent
        self.control_agent = control_agent
        self.env_config = env_config or {}
        self.n_simulations = n_simulations
        self.split_ratio = split_ratio

    def _run_episode(self, agent, env) -> float:
        """Run one episode and return total margin."""
        obs, _ = env.reset()
        total_margin = 0.0
        done = False
        while not done:
            # Get action from agent
            if hasattr(agent, "predict") and hasattr(agent, "allowed_actions"):
                # Heuristic or similar agent expecting CustomerState
                cs = CustomerState.from_observation(obs)
                action = agent.predict(cs)
            elif hasattr(agent, "predict"):
                # Could be heuristic or RL agent
                try:
                    cs = CustomerState.from_observation(obs)
                    action = agent.predict(cs)
                except TypeError:
                    action = agent.predict(obs)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            total_margin += info.get("margin_dollars", 0.0)
            done = terminated or truncated

        return total_margin

    def run(self) -> ABTestResult:
        """Execute the A/B test simulation.

        Returns:
            ABTestResult with statistical summary.
        """
        treatment_margins = []
        control_margins = []

        for i in range(self.n_simulations):
            env = DynamicPricingEnv(config=self.env_config)

            # Treatment
            t_margin = self._run_episode(self.treatment_agent, env)
            treatment_margins.append(t_margin)

            # Control (same env config, different seed via reset)
            c_margin = self._run_episode(self.control_agent, env)
            control_margins.append(c_margin)

        treatment_arr = np.array(treatment_margins)
        control_arr = np.array(control_margins)

        mean_delta = float(treatment_arr.mean() - control_arr.mean())

        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(treatment_arr, control_arr)

        # 95% CI for the difference
        diff = treatment_arr.mean() - control_arr.mean()
        se = np.sqrt(treatment_arr.var() / len(treatment_arr) + control_arr.var() / len(control_arr))
        ci_95 = (float(diff - 1.96 * se), float(diff + 1.96 * se))

        # Power estimate (proportion of simulations showing treatment > control)
        power = float(np.mean(treatment_arr > control_arr.mean()))

        # Cumulative curves
        cumulative_curves = {
            "treatment": np.cumsum(treatment_margins).tolist(),
            "control": np.cumsum(control_margins).tolist(),
        }

        return ABTestResult(
            mean_delta_margin=mean_delta,
            ci_95=ci_95,
            p_value=float(p_value),
            power=power,
            treatment_margins=treatment_margins,
            control_margins=control_margins,
            cumulative_curves=cumulative_curves,
        )
