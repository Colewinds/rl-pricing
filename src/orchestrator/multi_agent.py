"""Multi-agent orchestration: PriceScout, MarginGuardian, PortfolioManager."""

from collections import defaultdict

from src.environment.customer import CustomerState


class PriceScout:
    """Aggressive exploration agent for CSS 1-2 customers.

    Has access to the full action space including deep discounts.
    Optimized for volume growth and customer retention.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.allowed_actions = list(range(7))  # full action space
        self.exploration_bonus = self.config.get("exploration_bonus", 0.1)
        self.rl_agent = None  # Set during training

    def get_action_mask(self) -> list[int]:
        """Return action mask - all actions allowed for Scout."""
        return [1] * 7


class MarginGuardian:
    """Conservative agent for CSS 4-5 customers.

    Restricted to safe actions: Hold, +2%, +5%, -2% only.
    Optimized for margin protection with minimal churn risk.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.allowed_actions = self.config.get("restricted_actions", [0, 1, 2, 3])
        self.rl_agent = None

    def get_action_mask(self) -> list[int]:
        """Return action mask - only allowed actions enabled."""
        mask = [0] * 7
        for a in self.allowed_actions:
            mask[a] = 1
        return mask


class PortfolioManager:
    """Meta-agent that allocates customers to Scout or Guardian.

    Routes customers based on CSS tier:
    - CSS 1-2: PriceScout
    - CSS 4-5: MarginGuardian
    - CSS 3 (contested): assigned to whichever agent performs better,
      re-evaluated every `reallocation_period` periods.
    """

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
        """Determine which agent should handle this customer.

        Args:
            state: Current customer state.

        Returns:
            Agent identifier: "scout" or "guardian".
        """
        if state.css_score in self.scout_css:
            return "scout"
        if state.css_score in self.guardian_css:
            return "guardian"
        return self._contested_assignment

    def log_result(
        self, agent_id: str, state: CustomerState, action: int, reward: float
    ):
        """Record a pricing result for performance tracking.

        Args:
            agent_id: Which agent made the decision.
            state: Customer state at decision time.
            action: Action taken.
            reward: Reward received.
        """
        self._logs[agent_id].append({
            "css_score": state.css_score,
            "action": action,
            "reward": reward,
        })

    def update_allocations(self):
        """Re-evaluate which agent handles contested CSS segments.

        Compares average reward per agent on contested CSS customers
        and assigns the segment to the better performer.
        """
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
        """Check if it's time to re-evaluate agent allocations."""
        return period > 0 and period % self.reallocation_period == 0

    def get_performance_summary(self) -> dict:
        """Get summary statistics for each agent."""
        summary = {}
        for agent_id in ["scout", "guardian"]:
            logs = self._logs[agent_id]
            if logs:
                rewards = [r["reward"] for r in logs]
                summary[agent_id] = {
                    "n_decisions": len(logs),
                    "mean_reward": sum(rewards) / len(rewards),
                    "total_reward": sum(rewards),
                }
            else:
                summary[agent_id] = {
                    "n_decisions": 0,
                    "mean_reward": 0.0,
                    "total_reward": 0.0,
                }
        summary["contested_assignment"] = self._contested_assignment
        return summary
