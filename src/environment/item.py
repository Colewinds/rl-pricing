"""ItemState dataclass for item-level pricing decisions."""

from dataclasses import dataclass, field


# Category constants
CATEGORIES = {
    0: "protein",
    1: "produce",
    2: "paper",
    3: "dairy",
    4: "beverages",
    5: "frozen",
    6: "bakery",
    7: "misc",
}

CATEGORY_IDS = {v: k for k, v in CATEGORIES.items()}

# Concept constants
CONCEPTS = {
    0: "qsr",
    1: "casual_dining",
    2: "fine_dining",
    3: "institutional",
    4: "healthcare",
}

CONCEPT_IDS = {v: k for k, v in CONCEPTS.items()}


@dataclass
class ItemState:
    """Represents the observable state of a single item for the RL environment.

    Fields capture item-level pricing signals: category, cost structure,
    volume, perishability, substitutability, and competitive positioning.
    """

    category: int  # 0-7, see CATEGORIES
    subcategory: int  # 0-N within category
    unit_cost: float  # cost per unit in dollars
    unit_price: float  # selling price per unit
    item_margin_rate: float  # (price - cost) / price
    weekly_units: float
    weekly_revenue: float  # units * price
    perishability: float  # 0-1, how perishable (0=shelf-stable, 1=next-day)
    substitutability: float  # 0-1, how easily substituted (0=unique, 1=commodity)
    competitive_index: float  # 0-1, competitiveness of current price vs market
    seasonal_index: float  # 0-2, how seasonal this item is (0=stable, 2=highly seasonal)
    is_loss_leader: bool = False
    price_change_history: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    periods_since_last_change: int = 0
