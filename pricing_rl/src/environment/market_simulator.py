"""Market simulator for computing volume response, churn, and seasonality."""

import numpy as np

from src.environment.item import CATEGORIES


class MarketSimulator:
    """Simulates market dynamics for pricing decisions.

    Models three key phenomena:
    - Volume response: elasticity-based with stickiness dampening
    - Churn probability: logistic function of margin gap with SYW discount
    - Seasonality: quarterly modifiers on base values

    Extended for item-level:
    - Item elasticity: modified by category, concept, substitutability, perishability
    - Cross-item effects: price changes on high-share items affect related categories
    - Seasonal elasticity: perishable seasonal items become inelastic in peak season
    """

    def __init__(self, seed: int = 42, config: dict | None = None):
        self.rng = np.random.default_rng(seed)
        self.config = config or {}
        self.stickiness_threshold = self.config.get("stickiness_threshold_periods", 8)

    def compute_volume_response(
        self,
        price_change: float,
        elasticity: float,
        base_volume: float,
        css_score: int,
        periods_stable: int = 0,
        seasonal_index: float = 1.0,
        perishability: float = 0.5,
    ) -> float:
        """Compute the volume change from a price action.

        Args:
            price_change: Fractional price change (e.g., -0.05 for 5% cut).
            elasticity: Price elasticity (negative, e.g., -1.5).
            base_volume: Current volume level.
            css_score: Customer satisfaction score 1-5.
            periods_stable: Number of periods since last price change.
            seasonal_index: How seasonal this item is (0-2).
            perishability: How perishable the item is (0-1).

        Returns:
            Delta volume (positive means volume increase).
        """
        # Base response: elasticity * price_change * base_volume
        # elasticity is negative, price_change negative (cut) -> positive delta
        response = elasticity * price_change * base_volume

        # Stickiness: dampen response for long-stable customers
        if periods_stable >= self.stickiness_threshold:
            damping = 0.5 + 0.5 * (self.stickiness_threshold / periods_stable)
            response *= damping

        # Perishability effect: highly perishable items have stronger response
        # (customers can't stockpile, so volume shifts more with price)
        if perishability > 0.7:
            response *= 1.0 + 0.3 * (perishability - 0.7) / 0.3

        # Noise: scaled by CSS tier (higher CSS = less noise)
        noise_scale = 0.1 * base_volume * (6 - css_score) / 5
        noise = self.rng.normal(0, noise_scale)

        return response + noise

    def compute_item_elasticity(
        self,
        base_elasticity: float,
        category: int,
        concept: int,
        substitutability: float,
        perishability: float,
        config: dict | None = None,
    ) -> float:
        """Compute item-specific elasticity from customer base elasticity.

        Modifies base elasticity by category properties and concept affinity.

        Args:
            base_elasticity: Customer's base elasticity (negative).
            category: Category ID (0-7).
            concept: Concept ID (0-4).
            substitutability: Item substitutability (0-1).
            perishability: Item perishability (0-1).
            config: Items config with category elasticity modifiers.

        Returns:
            Item-specific elasticity (negative).
        """
        config = config or {}
        categories = config.get("categories", {})
        cat_name = CATEGORIES.get(category, "misc")
        cat_cfg = categories.get(cat_name, {})
        cat_modifier = cat_cfg.get("elasticity_modifier", 1.0)

        # High substitutability -> more elastic (customers switch easily)
        sub_modifier = 0.8 + 0.4 * substitutability  # 0.8 to 1.2

        # Perishability slightly increases elasticity (must buy somewhere)
        perish_modifier = 1.0 - 0.15 * perishability  # 1.0 to 0.85 (less elastic)

        item_elasticity = base_elasticity * cat_modifier * sub_modifier * perish_modifier
        return float(np.clip(item_elasticity, -5.0, -0.1))

    def compute_cross_item_effect(
        self,
        price_change: float,
        item_share_of_wallet: float,
        category: int,
        config: dict | None = None,
    ) -> float:
        """Compute cross-item effect of a price change.

        Large price changes on high-share items can affect overall customer
        behavior. Returns a multiplier on the customer-level churn impact.

        Args:
            price_change: Fractional price change applied.
            item_share_of_wallet: This item's share of customer's total revenue.
            category: Category ID.
            config: Cross-sell affinities config.

        Returns:
            Cross-item effect multiplier (1.0 = neutral, >1 = amplified).
        """
        # Only significant items create cross-effects
        if abs(item_share_of_wallet) < 0.03:
            return 1.0

        # Larger share + larger change = bigger effect
        effect = 1.0 + abs(price_change) * item_share_of_wallet * 2.0

        return float(np.clip(effect, 1.0, 1.5))

    def compute_churn_probability(
        self,
        margin_rate: float,
        css_score: int,
        syw: bool,
        periods_stable: int,
        threshold: float,
    ) -> float:
        """Compute per-step probability of customer churning.

        Computes an annual churn rate via logistic function, then converts
        to a per-week probability so that compounding over 52 weeks yields
        realistic annual churn rates (5-8% baseline, up to ~30% for at-risk).

        Args:
            margin_rate: Current margin rate (e.g., 0.24).
            css_score: Customer satisfaction score 1-5.
            syw: Whether customer is in SYW loyalty program.
            periods_stable: Periods since last price change.
            threshold: Margin threshold below which churn risk increases.

        Returns:
            Per-step churn probability in [0, 1].
        """
        # Logistic function gives annual churn rate centered on threshold.
        # When margin_rate == threshold, annual churn ~= base_annual_rate.
        # Steepness=8 gives a gradual curve; margins well above threshold -> low churn.
        margin_gap = threshold - margin_rate  # positive when margin below threshold
        logistic = 1 / (1 + np.exp(-8 * margin_gap))

        # Scale logistic output to realistic annual churn range.
        # CSS 1-2: baseline ~15% annual, max ~40%.  CSS 4-5: baseline ~3%, max ~20%.
        annual_baseline = {1: 0.12, 2: 0.10, 3: 0.06, 4: 0.03, 5: 0.02}
        annual_max = {1: 0.40, 2: 0.35, 3: 0.25, 4: 0.15, 5: 0.10}

        base = annual_baseline.get(css_score, 0.06)
        cap = annual_max.get(css_score, 0.25)
        annual_rate = base + (cap - base) * logistic

        # SYW reduces annual churn 15-20%
        if syw:
            annual_rate *= 0.82

        # Stickiness reduces churn
        if periods_stable >= self.stickiness_threshold:
            annual_rate *= 0.7

        annual_rate = float(np.clip(annual_rate, 0.0, 0.5))

        # Convert annual rate to per-week probability:
        # annual = 1 - (1 - weekly)^52  =>  weekly = 1 - (1 - annual)^(1/52)
        per_week = 1.0 - (1.0 - annual_rate) ** (1.0 / 52.0)

        return float(np.clip(per_week, 0.0, 1.0))

    def check_churn(self, churn_probability: float) -> bool:
        """Stochastic churn check based on probability."""
        return bool(self.rng.random() < churn_probability)

    def apply_seasonality(self, base_value: float, period: int, config: dict) -> float:
        """Apply quarterly seasonal modifier to a base value.

        Args:
            base_value: The baseline metric value.
            period: The current period (0-51, mapped to quarters).
            config: Seasonality config with q1-q4 modifiers.

        Returns:
            Seasonally adjusted value.
        """
        quarter = period // 13  # 0-3
        modifiers = [
            config.get("q1_modifier", 0.85),
            config.get("q2_modifier", 1.0),
            config.get("q3_modifier", 1.05),
            config.get("q4_modifier", 1.15),
        ]
        return base_value * modifiers[min(quarter, 3)]

    def apply_seasonal_elasticity(
        self,
        elasticity: float,
        seasonal_index: float,
        period: int,
        config: dict,
    ) -> float:
        """Modulate elasticity based on seasonality.

        Perishable seasonal items become less elastic in peak season
        (customers must buy regardless of price) and more elastic off-season.

        Args:
            elasticity: Base elasticity (negative).
            seasonal_index: How seasonal this item is (0-2).
            period: Current period (0-51).
            config: Seasonality config.

        Returns:
            Seasonally adjusted elasticity (negative).
        """
        quarter = period // 13
        modifiers = [
            config.get("q1_modifier", 0.85),
            config.get("q2_modifier", 1.0),
            config.get("q3_modifier", 1.05),
            config.get("q4_modifier", 1.15),
        ]
        seasonal_mod = modifiers[min(quarter, 3)]

        # In peak season (mod > 1), reduce elasticity for seasonal items
        # In off season (mod < 1), increase elasticity
        if seasonal_mod > 1.0:
            dampening = 1.0 - seasonal_index * (seasonal_mod - 1.0) * 0.5
        else:
            dampening = 1.0 + seasonal_index * (1.0 - seasonal_mod) * 0.3

        dampening = max(0.5, min(dampening, 1.5))
        return float(elasticity * dampening)
