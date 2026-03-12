"""Heuristic baseline agent approximating manual pricing rules.

CSS 1-2: Always discount 5% (grow volume for at-risk customers)
CSS 3:   Hold unless margin < 20%, then price up 2%
CSS 4-5: Hold unless volume drops >10% from expected baseline, then discount 2%
"""

from src.environment.customer import CustomerState


class HeuristicBaseline:
    """Rule-based pricing strategy approximating Sysco's manual approach."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.low_margin_threshold = self.config.get("low_margin_threshold", 0.20)
        self.volume_drop_threshold = self.config.get("volume_drop_threshold", 0.10)
        # Expected baseline volume by CSS tier (weekly cases)
        self.baseline_cases = self.config.get("baseline_cases", {
            1: 8.0, 2: 10.0, 3: 15.0, 4: 20.0, 5: 25.0,
        })

    def predict(self, state: CustomerState) -> int:
        """Select a pricing action based on heuristic rules.

        Args:
            state: Current customer state.

        Returns:
            Action index (0-6) from the action space.
        """
        if state.css_score <= 2:
            return 4  # price down 5%

        if state.css_score == 3:
            if state.current_margin_rate < self.low_margin_threshold:
                return 1  # price up 2%
            return 0  # hold

        # CSS 4-5
        baseline = self.baseline_cases.get(state.css_score, 20.0)
        if state.weekly_cases < baseline * (1 - self.volume_drop_threshold):
            return 3  # price down 2%
        return 0  # hold
