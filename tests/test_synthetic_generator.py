import pandas as pd
import numpy as np
from src.data.synthetic_generator import generate_customer_population, generate_transaction_history


def test_generate_population_shape():
    df = generate_customer_population(n=500, seed=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 500


def test_generate_population_columns():
    df = generate_customer_population(n=100, seed=42)
    required = [
        "customer_id", "css_score", "performance_percentile", "potential_tier",
        "margin_rate", "margin_dollars_monthly", "cases_monthly", "sales_monthly",
        "deliveries_per_week", "elasticity", "syw_flag", "perks_flag", "concept",
    ]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_css_distribution():
    df = generate_customer_population(n=10000, seed=42)
    css_pcts = df["css_score"].value_counts(normalize=True).sort_index()
    assert abs(css_pcts[1] + css_pcts[2] - 0.20) < 0.05
    assert abs(css_pcts[3] - 0.40) < 0.05
    assert abs(css_pcts[4] - 0.25) < 0.05
    assert abs(css_pcts[5] - 0.15) < 0.05


def test_syw_penetration():
    df = generate_customer_population(n=10000, seed=42)
    assert abs(df["syw_flag"].mean() - 0.30) < 0.05


def test_elasticity_css_correlation():
    df = generate_customer_population(n=10000, seed=42)
    corr = df["css_score"].corr(df["elasticity"])
    assert corr > 0  # elasticity is negative, higher CSS = less negative = positive corr


def test_deterministic():
    df1 = generate_customer_population(n=100, seed=42)
    df2 = generate_customer_population(n=100, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_transaction_history_shape():
    customers = generate_customer_population(n=100, seed=42)
    txns = generate_transaction_history(customers, periods=52, seed=42)
    assert isinstance(txns, pd.DataFrame)
    assert len(txns) > 0
    assert "period" in txns.columns
    assert "customer_id" in txns.columns


def test_transaction_seasonality():
    customers = generate_customer_population(n=1000, seed=42)
    txns = generate_transaction_history(customers, periods=52, seed=42)
    q1 = txns[txns["period"].between(0, 12)]["sales"].mean()
    q4 = txns[txns["period"].between(39, 51)]["sales"].mean()
    assert q4 > q1  # Q4 should be higher than Q1


def test_margin_rate_reasonable():
    df = generate_customer_population(n=1000, seed=42)
    assert df["margin_rate"].min() > 0
    assert df["margin_rate"].max() < 1.0
    assert 0.15 < df["margin_rate"].mean() < 0.35
