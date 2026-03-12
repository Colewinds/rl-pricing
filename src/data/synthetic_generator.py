"""Config-driven synthetic data generator for customer populations and transactions."""

from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def _load_config(config_path: str | None = None) -> dict:
    """Load config from YAML file."""
    if config_path is None:
        config_path = str(Path(__file__).parent.parent.parent / "config" / "default.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_customer_population(
    n: int = 10000,
    seed: int = 42,
    config_path: str | None = None,
) -> pd.DataFrame:
    """Generate a synthetic customer population.

    Args:
        n: Number of customers to generate.
        seed: Random seed for reproducibility.
        config_path: Path to YAML config. Defaults to config/default.yaml.

    Returns:
        DataFrame with one row per customer.
    """
    config = _load_config(config_path)
    sd = config["synthetic_data"]
    rng = np.random.default_rng(seed)

    # CSS assignment: multinomial draw
    css_probs = [sd["css_distribution"][f"css_{i}"] for i in range(1, 6)]
    css_scores = rng.choice([1, 2, 3, 4, 5], size=n, p=css_probs)

    # Concept assignment
    concept_names = [c["name"] for c in sd["concepts"]]
    concept_weights = [c["weight"] for c in sd["concepts"]]
    concept_modifiers = {c["name"]: c["elasticity_modifier"] for c in sd["concepts"]}
    concepts = rng.choice(concept_names, size=n, p=concept_weights)

    # Benchmarks
    bench = sd["percentile_benchmarks"]

    # Lognormal distributions scaled by CSS tier
    # Higher CSS -> higher volume/sales (CSS acts as a multiplier)
    css_scale = {1: 0.4, 2: 0.6, 3: 1.0, 4: 1.5, 5: 2.0}

    cases_monthly = np.array([
        max(1.0, rng.lognormal(np.log(bench["cases_p50_monthly"] * css_scale[cs]), 0.5))
        for cs in css_scores
    ])

    sales_monthly = np.array([
        max(10.0, rng.lognormal(np.log(bench["sales_p50_monthly"] * css_scale[cs]), 0.5))
        for cs in css_scores
    ])

    # Margin rate: lognormal around benchmark, less variation
    margin_rate = np.array([
        np.clip(rng.lognormal(np.log(bench["dm_pct_p50"]), 0.3), 0.05, 0.55)
        for _ in range(n)
    ])

    margin_dollars_monthly = sales_monthly * margin_rate

    # Deliveries per week: correlated with cases
    deliveries_per_week = np.clip(
        cases_monthly / (bench["cases_p50_monthly"] / 2.5) + rng.normal(0, 0.3, n),
        0.5, 7.0,
    )

    # Elasticity: per-CSS normal distributions
    elasticity_config = sd["elasticity"]["by_css"]
    elasticities = np.array([
        rng.normal(
            elasticity_config[f"css_{cs}"]["mean"],
            elasticity_config[f"css_{cs}"]["std"],
        ) * concept_modifiers[concepts[i]]
        for i, cs in enumerate(css_scores)
    ])
    # Ensure all elasticities are negative
    elasticities = np.clip(elasticities, -5.0, -0.1)

    # SYW: CSS-dependent probability
    syw_skew = sd["syw_css_skew"]  # list indexed 0-4 for CSS 1-5
    syw_flags = np.array([
        rng.random() < syw_skew[cs - 1]
        for cs in css_scores
    ])

    # Perks: flat rate
    perks_flags = rng.random(n) < sd["perks_penetration"]

    # Performance percentile: composite rank
    # DM% * 0.43 + DM$ * 0.26 + Sales * 0.17 + Cases * 0.15
    composite = (
        _rank_normalize(margin_rate) * 0.43
        + _rank_normalize(margin_dollars_monthly) * 0.26
        + _rank_normalize(sales_monthly) * 0.17
        + _rank_normalize(cases_monthly) * 0.15
    )
    performance_percentile = _rank_normalize(composite)

    # Potential tier: based on CSS and performance
    potential_tier = np.zeros(n, dtype=int)
    for i in range(n):
        if css_scores[i] >= 4 and performance_percentile[i] >= 0.5:
            potential_tier[i] = 2  # High
        elif css_scores[i] >= 3 or performance_percentile[i] >= 0.3:
            potential_tier[i] = 1  # Medium
        else:
            potential_tier[i] = 0  # Low

    df = pd.DataFrame({
        "customer_id": [f"C{i:06d}" for i in range(n)],
        "css_score": css_scores,
        "performance_percentile": performance_percentile,
        "potential_tier": potential_tier,
        "margin_rate": margin_rate,
        "margin_dollars_monthly": margin_dollars_monthly,
        "cases_monthly": cases_monthly,
        "sales_monthly": sales_monthly,
        "deliveries_per_week": deliveries_per_week,
        "elasticity": elasticities,
        "syw_flag": syw_flags,
        "perks_flag": perks_flags,
        "concept": concepts,
    })

    return df


def generate_transaction_history(
    customers_df: pd.DataFrame,
    periods: int = 52,
    seed: int = 42,
    config_path: str | None = None,
) -> pd.DataFrame:
    """Generate weekly transaction history for a customer population.

    Args:
        customers_df: Customer population DataFrame from generate_customer_population.
        periods: Number of weekly periods to simulate.
        seed: Random seed.
        config_path: Path to config YAML.

    Returns:
        DataFrame with one row per customer per period (until churn).
    """
    config = _load_config(config_path)
    sd = config["synthetic_data"]
    rng = np.random.default_rng(seed)

    seasonality = sd["seasonality"]
    annual_churn_baseline = sd["annual_churn_baseline"]
    churn_multiplier = sd["churn_css_multiplier"]

    records = []
    for _, cust in customers_df.iterrows():
        weekly_churn_base = 1 - (1 - annual_churn_baseline) ** (1 / 52)
        css_mult = churn_multiplier.get(f"css_{cust['css_score']}", 1.0)
        churned = False

        for period in range(periods):
            if churned:
                break

            # Seasonal modifier
            quarter = period // 13
            q_key = f"q{quarter + 1}_modifier"
            seasonal_mod = seasonality.get(q_key, 1.0)

            # Base metrics with seasonal adjustment and noise
            base_cases = cust["cases_monthly"] / 4.33  # monthly -> weekly
            base_sales = cust["sales_monthly"] / 4.33
            noise_scale = 0.05

            cases = max(0.0, base_cases * seasonal_mod + rng.normal(0, noise_scale * base_cases))
            sales = max(0.0, base_sales * seasonal_mod + rng.normal(0, noise_scale * base_sales))
            margin_rate = max(0.01, cust["margin_rate"] + rng.normal(0, 0.01))
            margin_dollars = sales * margin_rate
            deliveries = max(0.0, cust["deliveries_per_week"] + rng.normal(0, 0.2))

            records.append({
                "customer_id": cust["customer_id"],
                "period": period,
                "cases": cases,
                "sales": sales,
                "margin_rate": margin_rate,
                "margin_dollars": margin_dollars,
                "deliveries": deliveries,
            })

            # Churn check
            churn_prob = weekly_churn_base * css_mult
            if rng.random() < churn_prob:
                churned = True

    return pd.DataFrame(records)


def _rank_normalize(arr: np.ndarray) -> np.ndarray:
    """Convert array values to their percentile rank in [0, 1]."""
    from scipy.stats import rankdata
    ranks = rankdata(arr, method="average")
    return (ranks - 1) / max(len(ranks) - 1, 1)
