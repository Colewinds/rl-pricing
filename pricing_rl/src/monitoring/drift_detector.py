"""Drift detector for monitoring RL agent behavior over time."""

from collections import deque

import numpy as np


class DriftDetector:
    """Monitors reward drift, action entropy collapse, and elasticity accuracy.

    Alerts after consecutive breaching periods exceed threshold.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.reward_drift_sigma = self.config.get("reward_drift_sigma", 2.0)
        self.action_entropy_min = self.config.get("action_entropy_min", 0.5)
        self.alert_consecutive = self.config.get("alert_consecutive_periods", 3)

        # Rolling windows
        self._reward_history: list[float] = []
        self._action_history: list[int] = []
        self._elasticity_errors: list[float] = []

        # Per-period aggregates
        self._period_rewards: list[float] = []
        self._period_actions: list[int] = []
        self._period_elast_errors: list[float] = []

        # Alert counters
        self._reward_breach_count = 0
        self._entropy_breach_count = 0
        self._elasticity_breach_count = 0

        # Baseline stats (computed after warmup)
        self._baseline_reward_mean: float | None = None
        self._baseline_reward_std: float | None = None
        self._warmup_periods = 10

        # Period tracking
        self._period_count = 0

    def update(
        self,
        reward: float,
        action: int,
        elasticity_observed: float | None = None,
        elasticity_expected: float | None = None,
    ):
        """Record a single step's data.

        Args:
            reward: Reward received this step.
            action: Action taken this step.
            elasticity_observed: Measured elasticity (optional).
            elasticity_expected: Model's expected elasticity (optional).
        """
        self._period_rewards.append(reward)
        self._period_actions.append(action)

        if elasticity_observed is not None and elasticity_expected is not None:
            error = abs(elasticity_observed - elasticity_expected)
            self._period_elast_errors.append(error)

    def end_period(self):
        """Finalize current period and check for alerts.

        Call this at the end of each evaluation period (e.g., weekly).
        """
        self._period_count += 1

        # Aggregate period
        if self._period_rewards:
            period_mean_reward = np.mean(self._period_rewards)
            self._reward_history.append(period_mean_reward)
        else:
            self._reward_history.append(0.0)

        self._action_history.extend(self._period_actions)

        if self._period_elast_errors:
            self._elasticity_errors.append(np.mean(self._period_elast_errors))

        # Update baseline after warmup
        if self._period_count == self._warmup_periods:
            self._baseline_reward_mean = np.mean(self._reward_history)
            self._baseline_reward_std = max(np.std(self._reward_history), 1e-6)

        # Check reward drift
        if self._baseline_reward_mean is not None and len(self._reward_history) > self._warmup_periods:
            recent_mean = np.mean(self._reward_history[-3:])
            z_score = abs(recent_mean - self._baseline_reward_mean) / self._baseline_reward_std
            if z_score > self.reward_drift_sigma:
                self._reward_breach_count += 1
            else:
                self._reward_breach_count = 0

        # Check action entropy
        if len(self._period_actions) >= 7:
            entropy = self._compute_entropy(self._period_actions)
            if entropy < self.action_entropy_min:
                self._entropy_breach_count += 1
            else:
                self._entropy_breach_count = 0

        # Check elasticity accuracy
        if self._elasticity_errors:
            recent_mae = np.mean(self._elasticity_errors[-3:]) if len(self._elasticity_errors) >= 3 else self._elasticity_errors[-1]
            # Alert if MAE > 1.0 (significant divergence)
            if recent_mae > 1.0:
                self._elasticity_breach_count += 1
            else:
                self._elasticity_breach_count = 0

        # Reset period buffers
        self._period_rewards = []
        self._period_actions = []
        self._period_elast_errors = []

    def check_alerts(self) -> dict:
        """Check if any monitoring thresholds are breached.

        Returns:
            Dict with alert flags for each monitoring dimension.
        """
        alerts = {
            "reward_drift": self._reward_breach_count >= self.alert_consecutive,
            "action_entropy_collapse": self._entropy_breach_count >= self.alert_consecutive,
            "elasticity_divergence": self._elasticity_breach_count >= self.alert_consecutive,
            "reward_breach_periods": self._reward_breach_count,
            "entropy_breach_periods": self._entropy_breach_count,
            "elasticity_breach_periods": self._elasticity_breach_count,
        }
        alerts["any_alert"] = (
            alerts["reward_drift"]
            or alerts["action_entropy_collapse"]
            or alerts["elasticity_divergence"]
        )
        return alerts

    def generate_report(self) -> dict:
        """Generate a full monitoring report.

        Returns:
            Dict with monitoring statistics and alert status.
        """
        alerts = self.check_alerts()

        report = {
            "period_count": self._period_count,
            "alerts": alerts,
            "reward_history": list(self._reward_history),
            "baseline_reward_mean": self._baseline_reward_mean,
            "baseline_reward_std": self._baseline_reward_std,
        }

        if self._action_history:
            report["action_entropy"] = self._compute_entropy(self._action_history)

        if self._elasticity_errors:
            report["elasticity_mae"] = float(np.mean(self._elasticity_errors))

        return report

    @staticmethod
    def _compute_entropy(actions: list[int], n_actions: int = 7) -> float:
        """Compute normalized Shannon entropy of action distribution."""
        counts = np.zeros(n_actions)
        for a in actions:
            if 0 <= a < n_actions:
                counts[a] += 1
        if counts.sum() == 0:
            return 0.0
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(n_actions)
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0
