"""CustomerState dataclass with observation vector serialization."""

from dataclasses import dataclass, field
import numpy as np

# Normalization constants (max expected values for scaling to [0,1])
NORM = {
    "css_score": 5.0,
    "potential_tier": 2.0,
    "margin_rate": 0.60,
    "margin_dollars": 5000.0,
    "weekly_cases": 200.0,
    "weekly_sales": 15000.0,
    "deliveries": 7.0,
    "elasticity": 4.0,  # absolute value
    "action": 6.0,
    "periods": 52.0,
}


@dataclass
class CustomerState:
    """Represents the observable state of a single customer for the RL environment.

    Fields are chosen to capture pricing-relevant signals: satisfaction (CSS),
    financial metrics, behavioral patterns, and loyalty program membership.
    """

    css_score: int  # 1-5, Customer Satisfaction Score
    performance_percentile: float  # 0-1
    potential_tier: int  # 0=Low, 1=Medium, 2=High
    current_margin_rate: float  # e.g. 0.24 = 24%
    current_margin_dollars: float  # weekly margin in dollars
    weekly_cases: float
    weekly_sales: float
    deliveries_per_week: float
    elasticity_estimate: float  # negative value, e.g. -1.5
    price_change_history: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    periods_since_last_change: int = 0
    syw_flag: bool = False
    perks_flag: bool = False
    churn_probability: float = 0.0

    def to_observation(self) -> np.ndarray:
        """Convert to a normalized 17-float observation vector in [0, 1]."""
        obs = np.array(
            [
                self.css_score / NORM["css_score"],
                self.performance_percentile,
                self.potential_tier / NORM["potential_tier"],
                min(self.current_margin_rate / NORM["margin_rate"], 1.0),
                min(self.current_margin_dollars / NORM["margin_dollars"], 1.0),
                min(self.weekly_cases / NORM["weekly_cases"], 1.0),
                min(self.weekly_sales / NORM["weekly_sales"], 1.0),
                min(self.deliveries_per_week / NORM["deliveries"], 1.0),
                min(abs(self.elasticity_estimate) / NORM["elasticity"], 1.0),
                self.price_change_history[0] / NORM["action"],
                self.price_change_history[1] / NORM["action"],
                self.price_change_history[2] / NORM["action"],
                self.price_change_history[3] / NORM["action"],
                min(self.periods_since_last_change / NORM["periods"], 1.0),
                float(self.syw_flag),
                float(self.perks_flag),
                self.churn_probability,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, 0.0, 1.0)

    @classmethod
    def from_observation(cls, obs: np.ndarray) -> "CustomerState":
        """Reconstruct a CustomerState from an observation vector.

        Note: This is lossy due to normalization rounding.
        """
        return cls(
            css_score=int(round(obs[0] * NORM["css_score"])),
            performance_percentile=float(obs[1]),
            potential_tier=int(round(obs[2] * NORM["potential_tier"])),
            current_margin_rate=float(obs[3] * NORM["margin_rate"]),
            current_margin_dollars=float(obs[4] * NORM["margin_dollars"]),
            weekly_cases=float(obs[5] * NORM["weekly_cases"]),
            weekly_sales=float(obs[6] * NORM["weekly_sales"]),
            deliveries_per_week=float(obs[7] * NORM["deliveries"]),
            elasticity_estimate=float(-obs[8] * NORM["elasticity"]),
            price_change_history=[
                int(round(obs[9] * NORM["action"])),
                int(round(obs[10] * NORM["action"])),
                int(round(obs[11] * NORM["action"])),
                int(round(obs[12] * NORM["action"])),
            ],
            periods_since_last_change=int(round(obs[13] * NORM["periods"])),
            syw_flag=bool(obs[14] > 0.5),
            perks_flag=bool(obs[15] > 0.5),
            churn_probability=float(obs[16]),
        )
