"""Three-tier reward functions for the RL pricing environment."""

from src.environment.customer import CustomerState

# Default config values matching config/default.yaml
_DEFAULT_CLV_CONFIG = {
    "alpha_by_css": {"css_1": 0.3, "css_2": 0.35, "css_3": 0.5, "css_4": 0.6, "css_5": 0.7},
    "beta_by_css": {"css_1": 0.5, "css_2": 0.45, "css_3": 0.3, "css_4": 0.2, "css_5": 0.15},
    "gamma": 10.0,
    "delta": 2.0,
    "churn_thresholds": {"css_1": 0.40, "css_2": 0.40, "css_3": 0.25, "css_4": 0.15, "css_5": 0.15},
    "volatility_window": 4,
    "volatility_max_changes": 2,
    "lifetime_discount_by_css": {"css_1": 0.5, "css_2": 0.6, "css_3": 0.8, "css_4": 0.9, "css_5": 1.0},
}

_DEFAULT_PORTFOLIO_CONFIG = {
    "css_migration_bonus": 5.0,
    "action_concentration_penalty": 3.0,
}


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

    def explain(
        self, state: CustomerState, action: int, next_state: CustomerState
    ) -> str:
        delta = next_state.current_margin_dollars - state.current_margin_dollars
        return (
            f"Margin delta: ${delta:+.2f} "
            f"(${state.current_margin_dollars:.2f} -> ${next_state.current_margin_dollars:.2f})"
        )


class CLVOptimizer:
    """Customer Lifetime Value reward with CSS-dependent weights.

    R = alpha * margin_delta + beta * volume_delta - gamma * churn_penalty - delta * volatility_penalty

    Alpha/beta weights vary by CSS tier:
    - CSS 1-2 (at-risk): high beta (volume weight), low alpha (margin weight)
    - CSS 4-5 (loyal): high alpha (protect margin), low beta
    """

    def __init__(self, config: dict | None = None):
        self.config = {**_DEFAULT_CLV_CONFIG, **(config or {})}

    def _get_css_key(self, css_score: int) -> str:
        return f"css_{css_score}"

    def compute(
        self, state: CustomerState, action: int, next_state: CustomerState
    ) -> float:
        css_key = self._get_css_key(state.css_score)

        # CSS-dependent weights
        alpha = self.config["alpha_by_css"].get(css_key, 0.5)
        beta = self.config["beta_by_css"].get(css_key, 0.3)
        gamma = self.config["gamma"]
        delta = self.config["delta"]

        # Margin term
        margin_delta = next_state.current_margin_dollars - state.current_margin_dollars
        margin_term = alpha * margin_delta

        # Volume term
        volume_delta = next_state.weekly_cases - state.weekly_cases
        volume_term = beta * volume_delta

        # Churn penalty: penalize when churn exceeds CSS-specific threshold
        churn_threshold = self.config["churn_thresholds"].get(css_key, 0.25)
        churn_excess = max(0.0, next_state.churn_probability - churn_threshold)
        churn_penalty = gamma * churn_excess

        # Volatility penalty: count non-hold actions in recent history
        volatility_window = self.config["volatility_window"]
        max_changes = self.config["volatility_max_changes"]
        recent_actions = next_state.price_change_history[:volatility_window]
        n_changes = sum(1 for a in recent_actions if a != 0)
        volatility_penalty = delta * max(0, n_changes - max_changes)

        return margin_term + volume_term - churn_penalty - volatility_penalty

    def explain(
        self, state: CustomerState, action: int, next_state: CustomerState
    ) -> str:
        css_key = self._get_css_key(state.css_score)
        alpha = self.config["alpha_by_css"].get(css_key, 0.5)
        beta = self.config["beta_by_css"].get(css_key, 0.3)
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


class PortfolioOptimizer(CLVOptimizer):
    """Extends CLV optimizer with CSS migration bonus and concentration penalty.

    Additional terms:
    - CSS migration bonus: reward when customer's CSS improves
    - Action concentration penalty: penalize when action distribution is too narrow
    """

    def __init__(self, config: dict | None = None):
        merged = {**_DEFAULT_CLV_CONFIG, **_DEFAULT_PORTFOLIO_CONFIG, **(config or {})}
        super().__init__(merged)

    def compute(
        self,
        state: CustomerState,
        action: int,
        next_state: CustomerState,
        action_distribution: dict[int, float] | None = None,
    ) -> float:
        base_reward = super().compute(state, action, next_state)

        # CSS migration bonus
        migration_bonus = 0.0
        if next_state.css_score > state.css_score:
            migration_bonus = self.config["css_migration_bonus"]

        # Action concentration penalty (optional)
        concentration_penalty = 0.0
        if action_distribution is not None:
            max_share = max(action_distribution.values())
            if max_share > 0.6:  # More than 60% of actions are the same
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
