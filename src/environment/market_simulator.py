"""Market simulator for computing volume response, churn, and seasonality."""

import numpy as np


class MarketSimulator:
    """Simulates market dynamics for pricing decisions.

    Models three key phenomena:
    - Volume response: elasticity-based with stickiness dampening
    - Churn probability: logistic function of margin gap with SYW discount
    - Seasonality: quarterly modifiers on base values
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
    ) -> float:
        """Compute the volume change from a price action.

        Args:
            price_change: Fractional price change (e.g., -0.05 for 5% cut).
            elasticity: Price elasticity (negative, e.g., -1.5).
            base_volume: Current volume level.
            css_score: Customer satisfaction score 1-5.
            periods_stable: Number of periods since last price change.

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

        # Noise: scaled by CSS tier (higher CSS = less noise)
        noise_scale = 0.1 * base_volume * (6 - css_score) / 5
        noise = self.rng.normal(0, noise_scale)

        return response + noise

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
