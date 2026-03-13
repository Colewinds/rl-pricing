"""CustomerState and CustomerItemState dataclasses with observation vector serialization."""

from dataclasses import dataclass, field
import numpy as np

from src.environment.item import ItemState

# Normalization constants for legacy 17-dim CustomerState
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

# Normalization constants for 33-dim CustomerItemState
ITEM_NORM = {
    **NORM,
    "concept": 4.0,
    "category": 7.0,
    "subcategory": 20.0,
    "unit_cost": 50.0,
    "unit_price": 80.0,
    "weekly_units": 500.0,
    "weekly_revenue": 10000.0,
    "n_items_in_category": 50.0,
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


@dataclass
class CustomerItemState:
    """Represents the observable state of a customer-item pair for the RL environment.

    Extends the customer-level view with item-level pricing signals and
    cross-level features (share of wallet, category margin, item elasticity).

    Observation: 33-float normalized vector.
    Customer block (13): css, perf%, potential, margin_rate, weekly_cases, weekly_sales,
                         deliveries, concept, syw, perks, churn_prob, current_period, elasticity
    Item block (15):     category, subcategory, unit_cost, unit_price, item_margin_rate,
                         weekly_units, weekly_revenue, perishability, substitutability,
                         competitive_index, seasonal_index, price_history[0:4], periods_since
    Cross block (5):     item_share_of_wallet, category_margin_rate, customer_item_elasticity,
                         n_items_in_category, is_loss_leader
    """

    # Customer block
    css_score: int
    performance_percentile: float
    potential_tier: int
    customer_margin_rate: float  # aggregate across all items
    weekly_cases: float
    weekly_sales: float
    deliveries_per_week: float
    concept: int  # 0-4: QSR, casual, fine, institutional, healthcare
    syw_flag: bool
    perks_flag: bool
    churn_probability: float
    current_period: int  # 0-51, for seasonality
    customer_elasticity: float  # base elasticity for this customer

    # Item block
    item: ItemState = field(default_factory=lambda: ItemState(
        category=0, subcategory=0, unit_cost=5.0, unit_price=8.0,
        item_margin_rate=0.24, weekly_units=10.0, weekly_revenue=80.0,
        perishability=0.5, substitutability=0.5, competitive_index=0.5,
        seasonal_index=1.0,
    ))

    # Cross-level block
    item_share_of_wallet: float = 0.05  # item revenue / customer total
    category_margin_rate: float = 0.24  # avg margin across category for this customer
    customer_item_elasticity: float = -1.5  # item-specific elasticity
    n_items_in_category: int = 10
    is_loss_leader: bool = False

    # Legacy compat fields
    price_change_history: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    periods_since_last_change: int = 0

    OBS_DIM = 33

    def to_observation(self) -> np.ndarray:
        """Convert to a normalized 33-float observation vector in [0, 1]."""
        N = ITEM_NORM
        obs = np.array(
            [
                # Customer block (13)
                self.css_score / N["css_score"],
                self.performance_percentile,
                self.potential_tier / N["potential_tier"],
                min(self.customer_margin_rate / N["margin_rate"], 1.0),
                min(self.weekly_cases / N["weekly_cases"], 1.0),
                min(self.weekly_sales / N["weekly_sales"], 1.0),
                min(self.deliveries_per_week / N["deliveries"], 1.0),
                self.concept / N["concept"],
                float(self.syw_flag),
                float(self.perks_flag),
                self.churn_probability,
                self.current_period / N["periods"],
                min(abs(self.customer_elasticity) / N["elasticity"], 1.0),
                # Item block (15)
                self.item.category / N["category"],
                self.item.subcategory / N["subcategory"],
                min(self.item.unit_cost / N["unit_cost"], 1.0),
                min(self.item.unit_price / N["unit_price"], 1.0),
                min(self.item.item_margin_rate / N["margin_rate"], 1.0),
                min(self.item.weekly_units / N["weekly_units"], 1.0),
                min(self.item.weekly_revenue / N["weekly_revenue"], 1.0),
                self.item.perishability,
                self.item.substitutability,
                self.item.competitive_index,
                min(self.item.seasonal_index / 2.0, 1.0),
                self.item.price_change_history[0] / N["action"],
                self.item.price_change_history[1] / N["action"],
                self.item.price_change_history[2] / N["action"],
                self.item.price_change_history[3] / N["action"],
                min(self.item.periods_since_last_change / N["periods"], 1.0),
                # Cross-level block (4)
                min(self.item_share_of_wallet, 1.0),
                min(self.category_margin_rate / N["margin_rate"], 1.0),
                min(abs(self.customer_item_elasticity) / N["elasticity"], 1.0),
                min(self.n_items_in_category / N["n_items_in_category"], 1.0),
            ],
            dtype=np.float32,
        )
        return np.clip(obs, 0.0, 1.0)

    @classmethod
    def from_observation(cls, obs: np.ndarray) -> "CustomerItemState":
        """Reconstruct from a 33-dim observation vector. Lossy due to normalization."""
        N = ITEM_NORM
        item = ItemState(
            category=int(round(obs[13] * N["category"])),
            subcategory=int(round(obs[14] * N["subcategory"])),
            unit_cost=float(obs[15] * N["unit_cost"]),
            unit_price=float(obs[16] * N["unit_price"]),
            item_margin_rate=float(obs[17] * N["margin_rate"]),
            weekly_units=float(obs[18] * N["weekly_units"]),
            weekly_revenue=float(obs[19] * N["weekly_revenue"]),
            perishability=float(obs[20]),
            substitutability=float(obs[21]),
            competitive_index=float(obs[22]),
            seasonal_index=float(obs[23] * 2.0),
            price_change_history=[
                int(round(obs[24] * N["action"])),
                int(round(obs[25] * N["action"])),
                int(round(obs[26] * N["action"])),
                int(round(obs[27] * N["action"])),
            ],
            periods_since_last_change=int(round(obs[28] * N["periods"])),
        )
        return cls(
            css_score=int(round(obs[0] * N["css_score"])),
            performance_percentile=float(obs[1]),
            potential_tier=int(round(obs[2] * N["potential_tier"])),
            customer_margin_rate=float(obs[3] * N["margin_rate"]),
            weekly_cases=float(obs[4] * N["weekly_cases"]),
            weekly_sales=float(obs[5] * N["weekly_sales"]),
            deliveries_per_week=float(obs[6] * N["deliveries"]),
            concept=int(round(obs[7] * N["concept"])),
            syw_flag=bool(obs[8] > 0.5),
            perks_flag=bool(obs[9] > 0.5),
            churn_probability=float(obs[10]),
            current_period=int(round(obs[11] * N["periods"])),
            customer_elasticity=float(-obs[12] * N["elasticity"]),
            item=item,
            item_share_of_wallet=float(obs[29]),
            category_margin_rate=float(obs[30] * N["margin_rate"]),
            customer_item_elasticity=float(-obs[31] * N["elasticity"]),
            n_items_in_category=int(round(obs[32] * N["n_items_in_category"])),
            price_change_history=item.price_change_history,
            periods_since_last_change=item.periods_since_last_change,
        )

    def to_legacy_customer_state(self) -> CustomerState:
        """Convert to a legacy 17-dim CustomerState for backward compatibility."""
        return CustomerState(
            css_score=self.css_score,
            performance_percentile=self.performance_percentile,
            potential_tier=self.potential_tier,
            current_margin_rate=self.customer_margin_rate,
            current_margin_dollars=self.weekly_sales * self.customer_margin_rate,
            weekly_cases=self.weekly_cases,
            weekly_sales=self.weekly_sales,
            deliveries_per_week=self.deliveries_per_week,
            elasticity_estimate=self.customer_elasticity,
            price_change_history=self.price_change_history,
            periods_since_last_change=self.periods_since_last_change,
            syw_flag=self.syw_flag,
            perks_flag=self.perks_flag,
            churn_probability=self.churn_probability,
        )
