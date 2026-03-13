"""Config-driven synthetic data generator for customer populations, items, and transactions."""

from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from src.environment.item import CATEGORIES, CATEGORY_IDS, CONCEPT_IDS


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


def generate_item_catalog(
    seed: int = 42,
    config_path: str | None = None,
) -> pd.DataFrame:
    """Generate a master item catalog with category, cost, and item properties.

    Args:
        seed: Random seed for reproducibility.
        config_path: Path to YAML config.

    Returns:
        DataFrame with one row per item (n_catalog_items rows).
    """
    config = _load_config(config_path)
    items_cfg = config["items"]
    n_items = items_cfg["n_catalog_items"]
    rng = np.random.default_rng(seed)

    categories_cfg = items_cfg["categories"]
    cat_names = list(categories_cfg.keys())
    cat_weights = [categories_cfg[c]["weight"] for c in cat_names]

    # Assign items to categories proportionally
    cat_assignments = rng.choice(cat_names, size=n_items, p=cat_weights)

    records = []
    subcat_counters = {c: 0 for c in cat_names}
    for i in range(n_items):
        cat_name = cat_assignments[i]
        cat_cfg = categories_cfg[cat_name]
        cat_id = cat_cfg["id"]

        # Subcategory: cycle through available subcategories
        subcats = cat_cfg.get("subcategories", ["default"])
        subcat_idx = subcat_counters[cat_name] % len(subcats)
        subcat_counters[cat_name] += 1

        # Unit cost: lognormal, category-dependent
        base_cost = rng.lognormal(np.log(8.0), 0.6)
        if cat_name == "protein":
            base_cost *= 1.5
        elif cat_name == "produce":
            base_cost *= 0.8
        elif cat_name == "beverages":
            base_cost *= 0.6

        unit_cost = max(0.50, float(base_cost))

        # Margin rate
        margin_mean = cat_cfg["margin_rate"]["mean"]
        margin_std = cat_cfg["margin_rate"]["std"]
        item_margin = float(np.clip(rng.normal(margin_mean, margin_std), 0.03, 0.50))
        unit_price = unit_cost / (1 - item_margin)

        # Perishability
        perish_mean = cat_cfg["perishability"]["mean"]
        perish_std = cat_cfg["perishability"]["std"]
        perishability = float(np.clip(rng.normal(perish_mean, perish_std), 0.0, 1.0))

        # Substitutability
        sub_mean = cat_cfg["substitutability"]["mean"]
        sub_std = cat_cfg["substitutability"]["std"]
        substitutability = float(np.clip(rng.normal(sub_mean, sub_std), 0.0, 1.0))

        # Competitive index: random, slight correlation with substitutability
        competitive_index = float(np.clip(
            0.5 + 0.3 * (substitutability - 0.5) + rng.normal(0, 0.15), 0.0, 1.0
        ))

        # Seasonal index: produce and protein more seasonal
        seasonal_base = 1.0
        if cat_name in ("produce", "protein", "bakery"):
            seasonal_base = 1.3
        elif cat_name in ("frozen", "paper", "misc"):
            seasonal_base = 0.7
        seasonal_index = float(np.clip(rng.normal(seasonal_base, 0.3), 0.0, 2.0))

        records.append({
            "item_id": f"SKU{i:05d}",
            "category": cat_name,
            "category_id": cat_id,
            "subcategory": subcats[subcat_idx],
            "subcategory_id": subcat_idx,
            "unit_cost": round(unit_cost, 2),
            "unit_price": round(unit_price, 2),
            "item_margin_rate": round(item_margin, 4),
            "perishability": round(perishability, 3),
            "substitutability": round(substitutability, 3),
            "competitive_index": round(competitive_index, 3),
            "seasonal_index": round(seasonal_index, 3),
        })

    return pd.DataFrame(records)


def generate_customer_items(
    customers_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    seed: int = 42,
    config_path: str | None = None,
) -> pd.DataFrame:
    """Assign items to customers and generate per-customer-item metrics.

    Items are assigned based on concept-category affinity. Each customer gets
    a subset of the catalog, with volume and revenue proportional to their
    overall spending level.

    Args:
        customers_df: Customer population from generate_customer_population.
        catalog_df: Item catalog from generate_item_catalog.
        seed: Random seed.
        config_path: Path to config YAML.

    Returns:
        DataFrame with one row per customer-item pair.
    """
    config = _load_config(config_path)
    items_cfg = config["items"]
    rng = np.random.default_rng(seed)

    ipc = items_cfg["items_per_customer"]
    min_items = ipc["min"]
    max_items = ipc["max"]
    css_scale = ipc["css_scale"]
    affinity = items_cfg["concept_category_affinity"]
    loss_leader_pct = items_cfg.get("loss_leader_pct", 0.05)

    records = []
    for _, cust in customers_df.iterrows():
        concept = cust["concept"]
        css = cust["css_score"]
        scale = css_scale.get(f"css_{css}", 1.0)

        # Number of items for this customer
        n_items = int(np.clip(
            rng.normal((min_items + max_items) / 2 * scale, (max_items - min_items) / 6),
            min_items, max_items,
        ))
        n_items = min(n_items, len(catalog_df))

        # Build item selection probabilities based on concept-category affinity
        concept_aff = affinity.get(concept, {})
        item_probs = np.array([
            concept_aff.get(row["category"], 1.0)
            for _, row in catalog_df.iterrows()
        ])
        item_probs /= item_probs.sum()

        # Select items without replacement
        selected_indices = rng.choice(
            len(catalog_df), size=n_items, replace=False, p=item_probs
        )
        selected_items = catalog_df.iloc[selected_indices]

        # Distribute customer's weekly revenue across items
        weekly_sales = cust["sales_monthly"] / 4.33
        weekly_cases = cust["cases_monthly"] / 4.33

        # Generate Dirichlet-distributed revenue shares
        alpha_vec = np.ones(n_items) * 0.5  # sparse distribution
        revenue_shares = rng.dirichlet(alpha_vec)

        for idx, (_, item_row) in enumerate(selected_items.iterrows()):
            item_revenue = weekly_sales * revenue_shares[idx]
            item_units = max(0.1, item_revenue / max(item_row["unit_price"], 0.01))

            # Item-specific margin: base from catalog + noise
            item_margin = float(np.clip(
                item_row["item_margin_rate"] + rng.normal(0, 0.02),
                0.03, 0.50,
            ))

            is_ll = bool(rng.random() < loss_leader_pct)

            records.append({
                "customer_id": cust["customer_id"],
                "item_id": item_row["item_id"],
                "category": item_row["category"],
                "category_id": item_row["category_id"],
                "subcategory": item_row["subcategory"],
                "subcategory_id": item_row["subcategory_id"],
                "concept": concept,
                "css_score": css,
                "unit_cost": item_row["unit_cost"],
                "unit_price": item_row["unit_price"],
                "item_margin_rate": round(item_margin, 4),
                "weekly_units": round(float(item_units), 2),
                "weekly_revenue": round(float(item_revenue), 2),
                "revenue_share": round(float(revenue_shares[idx]), 4),
                "perishability": item_row["perishability"],
                "substitutability": item_row["substitutability"],
                "competitive_index": item_row["competitive_index"],
                "seasonal_index": item_row["seasonal_index"],
                "is_loss_leader": is_ll,
            })

    return pd.DataFrame(records)


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
