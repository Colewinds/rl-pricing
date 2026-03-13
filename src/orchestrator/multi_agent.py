"""Multi-agent orchestration: PriceScout, MarginGuardian, PortfolioManager.

Supports both customer-level and customer-item-level routing.
"""

from collections import defaultdict

from src.environment.customer import CustomerState, CustomerItemState
from src.environment.item import CATEGORIES


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

    Routes customers based on CSS tier and item properties:
    - CSS 1-2: PriceScout
    - CSS 4-5: MarginGuardian
    - CSS 3 (contested): assigned to whichever agent performs better,
      re-evaluated every `reallocation_period` periods.

    Item-level routing overrides:
    - High-substitutability items go to Scout regardless of CSS (exploration needed)
    - Items near margin floor go to Guardian (protection needed)
    """

    def __init__(self, config: dict | None = None, business_rules: dict | None = None):
        self.config = config or {}
        self.business_rules = business_rules or {}
        routing = self.config.get("css_routing", {})
        self.scout_css = routing.get("scout", [1, 2])
        self.guardian_css = routing.get("guardian", [4, 5])
        self.contested_css = routing.get("contested", [3])
        self.reallocation_period = self.config.get("reallocation_period", 4)

        # Performance tracking for contested CSS segments
        self._logs: dict[str, list[dict]] = defaultdict(list)
        self._contested_assignment = "guardian"  # default for CSS 3
        self._period_counter = 0

        # Category-level tracking
        self._category_discount_counts: dict[str, int] = defaultdict(int)
        self._category_margins: dict[str, list[float]] = defaultdict(list)
        self._category_action_counts: dict[str, int] = defaultdict(int)

    def assign(self, state: CustomerState | CustomerItemState) -> str:
        """Determine which agent should handle this customer/item.

        Args:
            state: Current customer or customer-item state.

        Returns:
            Agent identifier: "scout" or "guardian".
        """
        css_score = state.css_score

        # Item-level routing overrides
        if isinstance(state, CustomerItemState):
            cat_name = CATEGORIES.get(state.item.category, "misc")

            # High-substitutability items: always scout (need exploration)
            if state.item.substitutability > 0.8:
                return "scout"

            # Items near margin floor: always guardian (need protection)
            cat_floors = self.business_rules.get("category_margin_floors", {})
            floor = cat_floors.get(cat_name, 0.10)
            if state.item.item_margin_rate <= floor + 0.03:
                return "guardian"

        # Standard CSS-based routing
        if css_score in self.scout_css:
            return "scout"
        if css_score in self.guardian_css:
            return "guardian"
        return self._contested_assignment

    def log_result(
        self, agent_id: str, state: CustomerState | CustomerItemState,
        action: int, reward: float
    ):
        """Record a pricing result for performance tracking."""
        entry = {
            "css_score": state.css_score,
            "action": action,
            "reward": reward,
        }

        if isinstance(state, CustomerItemState):
            cat_name = CATEGORIES.get(state.item.category, "misc")
            entry["category"] = cat_name
            entry["item_margin"] = state.item.item_margin_rate

            # Track category-level stats
            self._category_action_counts[cat_name] += 1
            self._category_margins[cat_name].append(state.item.item_margin_rate)
            if action in (3, 4, 5, 6):  # discount actions
                self._category_discount_counts[cat_name] += 1

        self._logs[agent_id].append(entry)

    def check_customer_margin_floor(self, state: CustomerItemState, proposed_action: int) -> int:
        """Enforce customer-level margin floor. Override action to hold if needed.

        Args:
            state: Current customer-item state.
            proposed_action: Agent's proposed action.

        Returns:
            Approved action (may be overridden to 0=hold).
        """
        css_key = f"css_{state.css_score}"
        floor = self.business_rules.get("customer_margin_floor_by_css", {}).get(
            css_key, self.business_rules.get("customer_margin_floor", 0.15)
        )

        # Check holiday boost
        holiday_periods = self.business_rules.get("holiday_periods", [])
        if state.current_period in holiday_periods:
            floor += self.business_rules.get("holiday_margin_floor_boost", 0.03)

        if state.customer_margin_rate <= floor and proposed_action in (3, 4, 5, 6):
            return 0  # force hold
        return proposed_action

    def check_category_discount_share(self, state: CustomerItemState) -> bool:
        """Check if a category has too many discounted items.

        Returns True if the category discount share is below the limit.
        """
        cat_name = CATEGORIES.get(state.item.category, "misc")
        total = self._category_action_counts.get(cat_name, 0)
        discounts = self._category_discount_counts.get(cat_name, 0)

        if total == 0:
            return True

        max_share = self.business_rules.get("max_category_discount_share", 0.20)
        return (discounts / total) <= max_share

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

        # Add category-level stats
        summary["category_stats"] = {}
        for cat_name in self._category_action_counts:
            total = self._category_action_counts[cat_name]
            discounts = self._category_discount_counts.get(cat_name, 0)
            margins = self._category_margins.get(cat_name, [])
            summary["category_stats"][cat_name] = {
                "total_actions": total,
                "discount_share": discounts / total if total > 0 else 0.0,
                "mean_margin": sum(margins) / len(margins) if margins else 0.0,
            }

        return summary
