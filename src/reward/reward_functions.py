"""Three-tier reward functions for the RL pricing environment.

Supports both legacy customer-level and new item-level reward computation.
"""

from __future__ import annotations

from src.environment.customer import CustomerState, CustomerItemState
from src.environment.item import CATEGORIES

# Default config values matching config/default.yaml
_DEFAULT_CLV_CONFIG = {
    "alpha_by_css": {"css_1": 0.3, "css_2": 0.35, "css_3": 0.5, "css_4": 0.6, "css_5": 0.7},
    "beta_by_css": {"css_1": 0.5, "css_2": 0.45, "css_3": 0.3, "css_4": 0.2, "css_5": 0.15},
    "gamma": 10.0,
    "delta": 2.0,
    "epsilon": 3.0,
    "zeta": 15.0,
    "eta": 1.0,
    "theta": 5.0,
    "churn_thresholds": {"css_1": 0.40, "css_2": 0.40, "css_3": 0.25, "css_4": 0.15, "css_5": 0.15},
    "volatility_window": 4,
    "volatility_max_changes": 2,
    "lifetime_discount_by_css": {"css_1": 0.5, "css_2": 0.6, "css_3": 0.8, "css_4": 0.9, "css_5": 1.0},
    "alpha_concept_modifier": {"qsr": 0.8, "casual_dining": 1.0, "fine_dining": 1.3, "institutional": 0.9, "healthcare": 1.1},
    "beta_concept_modifier": {"qsr": 1.3, "casual_dining": 1.0, "fine_dining": 0.6, "institutional": 1.1, "healthcare": 0.8},
    "cross_sell_affinities": {
        "protein": {"paper": 0.15, "beverages": 0.10},
        "produce": {"dairy": 0.08},
        "bakery": {"dairy": 0.12, "beverages": 0.06},
        "frozen": {"paper": 0.05},
    },
}

_DEFAULT_PORTFOLIO_CONFIG = {
    "css_migration_bonus": 5.0,
    "action_concentration_penalty": 3.0,
}

_DEFAULT_BUSINESS_RULES = {
    "category_margin_floors": {
        "protein": 0.12, "produce": 0.08, "paper": 0.18, "dairy": 0.10,
        "beverages": 0.15, "frozen": 0.12, "bakery": 0.14, "misc": 0.14,
    },
    "customer_margin_floor": 0.15,
    "customer_margin_floor_by_css": {"css_1": 0.12, "css_2": 0.12, "css_3": 0.15, "css_4": 0.18, "css_5": 0.20},
    "max_consecutive_discounts": 3,
    "max_category_discount_share": 0.20,
    "delivery_delay_reward_discount": 0.3,
}

# Concept ID to name mapping
_CONCEPT_NAMES = {0: "qsr", 1: "casual_dining", 2: "fine_dining", 3: "institutional", 4: "healthcare"}


class MarginMaximizer:
    """Simple reward: R = delta margin dollars.

    Best for pure margin optimization without customer retention concerns.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def compute(
        self, state: CustomerState, action: int, next_state: CustomerState
    ) -> float:
        return next_state.current_margin_dollars - state.current_margin_dollars

    def compute_item(self, prev_item_margin, prev_item_units, prev_customer_margin,
                     prev_churn, action, state: CustomerItemState) -> float:
        """Item-level margin delta."""
        current_margin_dollars = state.item.weekly_revenue * state.item.item_margin_rate
        prev_margin_dollars = (prev_item_units * state.item.unit_price) * prev_item_margin
        return current_margin_dollars - prev_margin_dollars

    def explain(
        self, state: CustomerState, action: int, next_state: CustomerState
    ) -> str:
        delta = next_state.current_margin_dollars - state.current_margin_dollars
        return (
            f"Margin delta: ${delta:+.2f} "
            f"(${state.current_margin_dollars:.2f} -> ${next_state.current_margin_dollars:.2f})"
        )


class CLVOptimizer:
    """Customer Lifetime Value reward with CSS-dependent and concept-dependent weights.

    Legacy (customer-level):
        R = alpha * margin_delta + beta * volume_delta - gamma * churn_penalty - delta * volatility_penalty

    Item-level:
        R = alpha * margin_delta          (item-level)
          + beta * volume_delta           (item-level)
          - gamma * churn_penalty         (CUSTOMER-level)
          - delta * volatility_penalty    (item-level action history)
          + epsilon * strategic_bonus     (loss leader cross-sell)
          - zeta * margin_floor_penalty   (hard penalty below category floor)
          + eta * cross_sell_bonus        (complementary item uplift)
          - theta * category_concentration_penalty
    """

    def __init__(self, config: dict | None = None, business_rules: dict | None = None):
        self.config = {**_DEFAULT_CLV_CONFIG, **(config or {})}
        self.rules = {**_DEFAULT_BUSINESS_RULES, **(business_rules or {})}

    def _get_css_key(self, css_score: int) -> str:
        return f"css_{css_score}"

    def _get_alpha_beta(self, css_score: int, concept: int | None = None):
        """Get CSS-dependent alpha/beta, optionally modulated by concept."""
        css_key = self._get_css_key(css_score)
        alpha = self.config["alpha_by_css"].get(css_key, 0.5)
        beta = self.config["beta_by_css"].get(css_key, 0.3)

        if concept is not None:
            concept_name = _CONCEPT_NAMES.get(concept, "casual_dining")
            alpha *= self.config.get("alpha_concept_modifier", {}).get(concept_name, 1.0)
            beta *= self.config.get("beta_concept_modifier", {}).get(concept_name, 1.0)

        return alpha, beta

    def compute(
        self, state: CustomerState, action: int, next_state: CustomerState
    ) -> float:
        """Legacy customer-level reward computation."""
        css_key = self._get_css_key(state.css_score)
        alpha, beta = self._get_alpha_beta(state.css_score)
        gamma = self.config["gamma"]
        delta = self.config["delta"]

        margin_delta = next_state.current_margin_dollars - state.current_margin_dollars
        margin_term = alpha * margin_delta

        volume_delta = next_state.weekly_cases - state.weekly_cases
        volume_term = beta * volume_delta

        churn_threshold = self.config["churn_thresholds"].get(css_key, 0.25)
        churn_excess = max(0.0, next_state.churn_probability - churn_threshold)
        churn_penalty = gamma * churn_excess

        volatility_window = self.config["volatility_window"]
        max_changes = self.config["volatility_max_changes"]
        recent_actions = next_state.price_change_history[:volatility_window]
        n_changes = sum(1 for a in recent_actions if a != 0)
        volatility_penalty = delta * max(0, n_changes - max_changes)

        return margin_term + volume_term - churn_penalty - volatility_penalty

    def compute_item(
        self,
        prev_item_margin: float,
        prev_item_units: float,
        prev_customer_margin: float,
        prev_churn: float,
        action: int,
        state: CustomerItemState,
    ) -> float:
        """Item-level reward with all 8 terms."""
        css_key = self._get_css_key(state.css_score)
        alpha, beta = self._get_alpha_beta(state.css_score, state.concept)
        gamma = self.config["gamma"]
        delta = self.config["delta"]
        epsilon = self.config.get("epsilon", 3.0)
        zeta = self.config.get("zeta", 15.0)
        eta = self.config.get("eta", 1.0)
        theta = self.config.get("theta", 5.0)

        # 1. Margin term (item-level)
        current_margin_dollars = state.item.weekly_revenue * state.item.item_margin_rate
        prev_margin_dollars = prev_item_units * (state.item.unit_cost / max(1 - prev_item_margin, 0.01)) * prev_item_margin
        margin_delta = current_margin_dollars - prev_margin_dollars
        margin_term = alpha * margin_delta

        # 2. Volume term (item-level)
        volume_delta = state.item.weekly_units - prev_item_units
        volume_term = beta * volume_delta

        # 3. Churn penalty (customer-level)
        churn_threshold = self.config["churn_thresholds"].get(css_key, 0.25)
        churn_excess = max(0.0, state.churn_probability - churn_threshold)
        churn_penalty = gamma * churn_excess

        # 4. Volatility penalty (item action history)
        volatility_window = self.config["volatility_window"]
        max_changes = self.config["volatility_max_changes"]
        recent_actions = state.item.price_change_history[:volatility_window]
        n_changes = sum(1 for a in recent_actions if a != 0)
        volatility_penalty = delta * max(0, n_changes - max_changes)

        # 5. Strategic bonus (loss leader cross-sell)
        strategic_bonus = 0.0
        if state.is_loss_leader and action in (3, 4, 5, 6):  # discount on loss leader
            strategic_bonus = epsilon * abs(state.item_share_of_wallet)

        # 6. Margin floor penalty
        cat_name = CATEGORIES.get(state.item.category, "misc")
        cat_floors = self.rules.get("category_margin_floors", {})
        floor = cat_floors.get(cat_name, 0.10)
        margin_floor_penalty = 0.0
        if state.item.item_margin_rate < floor:
            margin_floor_penalty = zeta * (floor - state.item.item_margin_rate)

        # 7. Cross-sell bonus
        cross_sell_bonus = 0.0
        affinities = self.config.get("cross_sell_affinities", {})
        if cat_name in affinities and action in (3, 4, 5, 6):
            for _related_cat, affinity_strength in affinities[cat_name].items():
                cross_sell_bonus += eta * affinity_strength * abs(volume_delta)

        # 8. Category concentration penalty
        # Penalize if too many discounts in this item's category (approximation)
        discount_count = sum(1 for a in recent_actions if a in (3, 4, 5, 6))
        max_share = self.rules.get("max_category_discount_share", 0.20)
        category_penalty = 0.0
        if discount_count > volatility_window * max_share:
            category_penalty = theta * (discount_count / volatility_window - max_share)

        total = (
            margin_term
            + volume_term
            - churn_penalty
            - volatility_penalty
            + strategic_bonus
            - margin_floor_penalty
            + cross_sell_bonus
            - category_penalty
        )
        return total

    def explain(
        self, state: CustomerState, action: int, next_state: CustomerState
    ) -> str:
        css_key = self._get_css_key(state.css_score)
        alpha, beta = self._get_alpha_beta(state.css_score)
        gamma = self.config["gamma"]
        delta = self.config["delta"]

        margin_delta = next_state.current_margin_dollars - state.current_margin_dollars
        volume_delta = next_state.weekly_cases - state.weekly_cases
        churn_threshold = self.config["churn_thresholds"].get(css_key, 0.25)
        churn_excess = max(0.0, next_state.churn_probability - churn_threshold)

        recent_actions = next_state.price_change_history[: self.config["volatility_window"]]
        n_changes = sum(1 for a in recent_actions if a != 0)
        vol_excess = max(0, n_changes - self.config["volatility_max_changes"])

        lines = [
            f"Margin term: {alpha:.2f} * ${margin_delta:+.2f} = ${alpha * margin_delta:+.2f}",
            f"Volume term: {beta:.2f} * {volume_delta:+.1f} cases = {beta * volume_delta:+.2f}",
            f"Churn penalty: -{gamma:.1f} * {churn_excess:.3f} = {-gamma * churn_excess:+.2f}",
            f"Volatility penalty: -{delta:.1f} * {vol_excess} = {-delta * vol_excess:+.2f}",
            f"Total: {self.compute(state, action, next_state):+.2f}",
        ]
        return "\n".join(lines)

    def explain_item(self, state: CustomerItemState) -> str:
        """Explain item-level reward components for the current state."""
        cat_name = CATEGORIES.get(state.item.category, "misc")
        concept_name = _CONCEPT_NAMES.get(state.concept, "casual_dining")
        alpha, beta = self._get_alpha_beta(state.css_score, state.concept)

        lines = [
            f"CSS {state.css_score} / {concept_name} / {cat_name}",
            f"  alpha={alpha:.3f} (margin weight), beta={beta:.3f} (volume weight)",
            f"  Item margin: {state.item.item_margin_rate:.1%}, Customer margin: {state.customer_margin_rate:.1%}",
            f"  Category floor: {self.rules.get('category_margin_floors', {}).get(cat_name, 0.10):.1%}",
            f"  Share of wallet: {state.item_share_of_wallet:.1%}",
            f"  Loss leader: {state.is_loss_leader}",
        ]
        return "\n".join(lines)


class PortfolioOptimizer(CLVOptimizer):
    """Extends CLV optimizer with CSS migration bonus and concentration penalty."""

    def __init__(self, config: dict | None = None, business_rules: dict | None = None):
        merged = {**_DEFAULT_CLV_CONFIG, **_DEFAULT_PORTFOLIO_CONFIG, **(config or {})}
        super().__init__(merged, business_rules)

    def compute(
        self,
        state: CustomerState,
        action: int,
        next_state: CustomerState,
        action_distribution: dict[int, float] | None = None,
    ) -> float:
        base_reward = super().compute(state, action, next_state)

        migration_bonus = 0.0
        if next_state.css_score > state.css_score:
            migration_bonus = self.config["css_migration_bonus"]

        concentration_penalty = 0.0
        if action_distribution is not None:
            max_share = max(action_distribution.values())
            if max_share > 0.6:
                concentration_penalty = (
                    self.config["action_concentration_penalty"] * (max_share - 0.6)
                )

        return base_reward + migration_bonus - concentration_penalty

    def explain(
        self, state: CustomerState, action: int, next_state: CustomerState
    ) -> str:
        base_explanation = super().explain(state, action, next_state)
        migration = ""
        if next_state.css_score > state.css_score:
            migration = f"\nCSS migration bonus: +{self.config['css_migration_bonus']:.2f}"
        return base_explanation + migration
