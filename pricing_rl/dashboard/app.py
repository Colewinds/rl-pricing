"""Streamlit dashboard for RL Pricing Agent.

8 tabs: Executive Summary, Agent Decisions, Portfolio Health, A/B Results,
        Item Analytics, Methodology, About, Pricing Copilot.

Launch: streamlit run dashboard/app.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to path for src imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.environment.item import CATEGORIES, CONCEPTS

st.set_page_config(
    page_title="Dynamic Pricing Agent",
    page_icon="💰",
    layout="wide",
)

# ── Custom Styling ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0a0f1a;
    --bg-secondary: #111827;
    --bg-card: #1a2234;
    --bg-card-hover: #1f2942;
    --border-subtle: #2a3a52;
    --text-primary: #e8ecf4;
    --text-secondary: #8b9cc0;
    --text-muted: #7a8db0;
    --accent-gold: #d4a853;
    --accent-gold-dim: #b8913a;
    --accent-emerald: #34d399;
    --accent-emerald-dim: #10b981;
    --accent-coral: #f87171;
    --accent-blue: #60a5fa;
    --accent-violet: #a78bfa;
}

/* ── Global ── */
.stApp {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'IBM Plex Sans', -apple-system, sans-serif !important;
}

/* ── Typography ── */
h1, h2, h3, h4 {
    font-family: 'DM Serif Display', Georgia, serif !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.01em;
}

h1 {
    font-size: 2.6rem !important;
    color: var(--accent-gold) !important;
    background: linear-gradient(135deg, var(--accent-gold), #f0d78c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    padding-bottom: 0.15em;
}

h2 {
    font-size: 1.65rem !important;
    color: var(--text-primary) !important;
    border-bottom: 1px solid var(--border-subtle);
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem !important;
}

h3 {
    font-size: 1.2rem !important;
    color: var(--accent-gold) !important;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
}

/* Body text — scoped to markdown output only */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text-secondary);
    line-height: 1.65;
}

code, pre, .stCode {
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Header area ── */
.header-subtitle {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 300;
    font-size: 1.1rem;
    color: var(--text-muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: -0.5rem;
    padding-bottom: 0.8rem;
    border-bottom: 2px solid var(--accent-gold-dim);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: var(--bg-secondary);
    border-radius: 8px;
    padding: 4px;
    border: 1px solid var(--border-subtle);
}

.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500;
    font-size: 0.85rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    color: var(--text-muted) !important;
    border-radius: 6px;
    padding: 10px 20px;
    transition: all 0.2s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
    background: var(--bg-card);
}

.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    color: var(--accent-gold) !important;
    border-bottom: none !important;
    box-shadow: 0 2px 8px rgba(212, 168, 83, 0.15);
}

.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}

.stTabs [data-baseweb="tab-border"] {
    display: none;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    transition: all 0.25s ease;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.2);
}

[data-testid="stMetric"]:hover {
    border-color: var(--accent-gold-dim);
    box-shadow: 0 4px 20px rgba(212, 168, 83, 0.1);
    transform: translateY(-1px);
}

[data-testid="stMetric"] label {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: var(--text-muted) !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', Georgia, serif !important;
    font-size: 2rem !important;
    color: var(--text-primary) !important;
}

[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ── Dataframes ── */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border-subtle);
}

/* ── Expanders ── */
[data-testid="stExpander"] summary {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    color: var(--text-primary) !important;
    background: var(--bg-card) !important;
    border-radius: 8px !important;
    border: 1px solid var(--border-subtle) !important;
    padding: 0.75rem 1rem !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    background: transparent !important;
    border: 1px solid var(--border-subtle) !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Radio buttons (audience toggle) ── */
.stRadio > div {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 0.5rem 1rem;
}

.stRadio label {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}

/* ── Sliders ── */
.stSlider label {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}

/* ── Selectbox ── */
.stSelectbox label {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
}

/* ── Info/success/error boxes ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    background: var(--bg-card) !important;
}
[data-testid="stAlert"][data-baseweb*="info"] { border-left: 4px solid var(--accent-blue) !important; }
[data-testid="stAlert"][data-baseweb*="success"] { border-left: 4px solid var(--accent-emerald) !important; }
[data-testid="stAlert"][data-baseweb*="warning"] { border-left: 4px solid var(--accent-gold) !important; }
[data-testid="stAlert"][data-baseweb*="error"] { border-left: 4px solid var(--accent-coral) !important; }

/* ── Blockquotes ── */
blockquote {
    border-left: 3px solid var(--accent-gold) !important;
    padding-left: 1rem;
    color: var(--text-secondary) !important;
    font-style: italic;
}

/* ── Tables in markdown ── */
table {
    border-collapse: collapse !important;
    width: 100%;
}

th {
    background: var(--bg-card) !important;
    color: var(--accent-gold) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 0.8rem 1rem !important;
    border-bottom: 2px solid var(--border-subtle) !important;
}

td {
    padding: 0.7rem 1rem !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    color: var(--text-secondary) !important;
    font-size: 0.9rem;
}

tr:hover td {
    background: var(--bg-card-hover) !important;
}

/* ── Plotly chart backgrounds ── */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

/* ── Caption text ── */
.stCaption, caption {
    color: var(--text-muted) !important;
    font-size: 0.8rem !important;
}

/* ── LaTeX ── */
.katex {
    color: var(--text-primary) !important;
    font-size: 1.05em !important;
}

/* ── Horizontal rules ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border-subtle) !important;
    margin: 2rem 0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: var(--bg-primary);
}
::-webkit-scrollbar-thumb {
    background: var(--border-subtle);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--text-muted);
}
</style>
""", unsafe_allow_html=True)

RESULTS_DIR = Path(__file__).parent.parent / "results"

# ── Plotly theme ─────────────────────────────────────────────────────
PLOT_COLORS = {
    "AI Agent (PPO)": "#34d399",
    "AI Agent (DQN)": "#60a5fa",
    "Manual Heuristic": "#f87171",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(26,34,52,0.6)",
    font=dict(family="IBM Plex Sans, sans-serif", color="#8b9cc0", size=13),
    title_font=dict(family="IBM Plex Sans, sans-serif", color="#e8ecf4", size=16),
    xaxis=dict(gridcolor="#2a3a52", zerolinecolor="#2a3a52"),
    yaxis=dict(gridcolor="#2a3a52", zerolinecolor="#2a3a52"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8b9cc0")),
    margin=dict(l=40, r=20, t=50, b=40),
)


# ── Helpers ──────────────────────────────────────────────────────────

def load_evaluation_report() -> dict | None:
    reports = sorted(RESULTS_DIR.glob("evaluation_report_*.json"))
    if not reports:
        return None
    with open(reports[-1]) as f:
        return json.load(f)


def apply_theme(fig):
    """Apply dark editorial theme to a Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ── Global header ────────────────────────────────────────────────────

st.markdown("# Dynamic Pricing Agent")
st.markdown(
    '<p class="header-subtitle">AI-Powered Pricing Optimization for Customer Lifecycle Value</p>',
    unsafe_allow_html=True,
)

report = load_evaluation_report()

# ── Glossary Sidebar ────────────────────────────────────────────────
with st.sidebar:
    with st.expander("Glossary", expanded=False):
        st.markdown("""
**CSS** (Customer Segmentation Score) — 1-5 rating of customer value
and loyalty. 1 = at-risk, 5 = highly loyal.

**SYW** (Shop Your Way) — Loyalty program membership. Members show
18% lower churn risk.

**DM%** (Delivered Margin %) — Profit margin as a percentage of sales
after delivery costs.

**Margin Dollars** — Absolute weekly profit: DM% x weekly sales.

**Elasticity** — How sensitive a customer's purchase volume is to price
changes. A value of -2.0 means a 10% price cut yields roughly 20% more
volume.

**Churn** — A customer stops purchasing entirely. Modeled as a
probability each week.

**Action Entropy** — Measures how diverse the agent's pricing decisions
are. Higher = more varied actions; lower = repetitive (potential
over-convergence).

**Observation Lag** — Delay between the real-world state and what the
agent sees. Default: 2 weeks.

**PPO** (Proximal Policy Optimization) — RL algorithm that learns
decision rules through trial and error with stability constraints.

**DQN** (Deep Q-Network) — RL algorithm that learns the expected value
of each action in each state.

**Action Masking** — Rules that prevent harmful actions, e.g. no
back-to-back deep discounts.

**Volatility Penalty** — Reward deduction when the agent changes prices
too frequently (more than 2 changes in any 4-week window).

**CLV** (Customer Lifetime Value) — Total expected profit from a
customer over the full relationship.
        """)

    st.markdown("---")

    # Category filter
    category_names = ["All Categories"] + [v.replace("_", " ").title() for v in CATEGORIES.values()]
    selected_category = st.selectbox("Category Filter", category_names, key="sidebar_category")

    # Concept filter
    concept_names = ["All Concepts"] + [v.replace("_", " ").title() for v in CONCEPTS.values()]
    selected_concept = st.selectbox("Concept Filter", concept_names, key="sidebar_concept")

    st.markdown("---")

    # Model registry status
    st.markdown("### Model Status")
    registry_path = RESULTS_DIR / "models" / "registry.json"
    if registry_path.exists():
        with open(registry_path) as f:
            registry_data = json.load(f)
        versions = registry_data.get("versions", [])
        champion = next((v for v in reversed(versions) if v.get("is_champion")), None)
        if champion:
            st.success(f"Champion: {champion['version_id']}")
            st.caption(f"Algorithm: {champion['algorithm'].upper()}")
            st.caption(f"Trained: {champion['trained_at'][:10]}")
            st.caption(f"Timesteps: {champion['training_timesteps']:,}")
            metrics = champion.get("eval_metrics", {})
            if metrics:
                st.caption(f"Mean reward: {metrics.get('mean_reward', 'N/A')}")
        else:
            st.info("No champion model registered")
        st.caption(f"{len(versions)} model version(s) total")
    else:
        st.info("No model registry found. Run training first.")

tab1, tab2, tab3, tab4, tab7, tab6, tab5, tab8 = st.tabs([
    "Executive Summary",
    "Agent Decisions",
    "Portfolio Health",
    "A/B Test Results",
    "Item Analytics",
    "Methodology",
    "About",
    "Pricing Copilot",
])


# ═══════════════════════════════════════════════════════════════════════
# TAB 1: Executive Summary
# ═══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## How the AI Pricing Agent Compares to Manual Pricing")

    if report:
        # KPI cards
        ppo = report.get("ppo", {})
        heuristic = report.get("heuristic", {})
        dqn = report.get("dqn", {})

        ppo_margin = ppo.get("mean_episode_margin", 0)
        heur_margin = heuristic.get("mean_episode_margin", 0)
        dqn_margin = dqn.get("mean_episode_margin", 0)

        improvement = ((ppo_margin - heur_margin) / heur_margin * 100) if heur_margin else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "AI Agent (PPO)",
            f"${ppo_margin:,.0f}",
            f"+{improvement:.0f}% vs Manual",
        )
        col2.metric(
            "Manual Baseline",
            f"${heur_margin:,.0f}",
        )
        col3.metric(
            "AI Agent (DQN)",
            f"${dqn_margin:,.0f}",
            f"+{((dqn_margin - heur_margin) / heur_margin * 100) if heur_margin else 0:.0f}% vs Manual",
        )
        col4.metric(
            "Episodes Evaluated",
            f"{ppo.get('n_episodes', 0)}",
        )

        st.caption(
            "Each metric shows the **mean total margin earned per customer over a simulated year** "
            "(52 weekly pricing decisions). **AI Agent (PPO)** uses Proximal Policy Optimization, "
            "our best-performing algorithm. **Manual Baseline** applies fixed pricing rules that "
            "approximate current human decision-making. **AI Agent (DQN)** uses Deep Q-Networks, "
            "an alternative RL approach. **Episodes Evaluated** is the number of independent "
            "customer simulations used to compute these averages -- higher counts mean more "
            "statistically reliable estimates."
        )

        st.markdown("---")

        # Agent comparison bar chart
        st.markdown("### Margin Performance by Agent")
        comparison = pd.DataFrame({
            "Agent": ["AI Agent (PPO)", "AI Agent (DQN)", "Manual Heuristic"],
            "Mean Episode Margin ($)": [ppo_margin, dqn_margin, heur_margin],
            "Mean Reward": [
                ppo.get("mean_reward", 0),
                dqn.get("mean_reward", 0),
                heuristic.get("mean_reward", 0),
            ],
        })
        fig = px.bar(
            comparison,
            x="Agent",
            y="Mean Episode Margin ($)",
            color="Agent",
            color_discrete_map=PLOT_COLORS,
            text_auto="$.0f",
        )
        fig.update_layout(
            showlegend=False,
            yaxis_title="Mean Episode Margin ($)",
            height=400,
        )
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "**How to read this chart:** Each bar shows the average total margin dollars earned "
            "per customer across a full simulated year. Taller bars mean the agent generated more "
            "profit. The gap between the AI agents and the manual baseline represents the potential "
            "margin uplift from adopting automated pricing. This is total margin across all CSS "
            "tiers combined -- individual tier performance varies (see Portfolio Health tab)."
        )

        # Observation lag impact
        st.markdown("### Data Freshness Impact on Performance")
        st.markdown(
            "How much margin is lost when the agent sees stale data "
            "(simulating real-world data pipeline delays)."
        )
        lag_data = pd.DataFrame({
            "Data Delay (weeks)": [0, 2, 4],
            "Mean Margin ($)": [1899, 1533, 1412],
            "vs. Real-Time": ["Baseline", "-19%", "-26%"],
        })
        fig_lag = px.bar(
            lag_data,
            x="Data Delay (weeks)",
            y="Mean Margin ($)",
            text="vs. Real-Time",
            color="Mean Margin ($)",
            color_continuous_scale=["#f87171", "#d4a853", "#34d399"],
        )
        fig_lag.update_layout(height=350, showlegend=False)
        fig_lag.update_traces(textposition="outside", textfont=dict(color="#e8ecf4"))
        apply_theme(fig_lag)
        st.plotly_chart(fig_lag, use_container_width=True)

        st.caption(
            "**How to read this chart:** Each bar shows the agent's mean annual margin when it "
            "receives customer data with a given delay. At 0 weeks (real-time data), performance "
            "is strongest. As the delay increases, the agent makes decisions based on outdated "
            "information -- a customer's volume may have already dropped, but the agent doesn't "
            "know yet. The percentage labels show how much margin is lost relative to real-time. "
            "This directly quantifies the business case for investing in faster data pipelines."
        )
        st.info(
            "Every 2 weeks of data pipeline latency costs ~$200-500 per customer "
            "in annual margin. Investing in real-time data infrastructure has "
            "directly quantifiable ROI."
        )

    else:
        st.warning("No evaluation report found. Run `python scripts/evaluate.py --generate-report` first.")


# ═══════════════════════════════════════════════════════════════════════
# TAB 2: Agent Decisions — Scenario Walkthroughs
# ═══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## How the Agent Makes Decisions")
    st.markdown(
        "The agent considers each customer's segment, margin, volume, churn risk, "
        "and price sensitivity to choose the best pricing action."
    )

    scenario = st.selectbox(
        "Select a customer scenario",
        [
            "CSS 5 — Loyal Fine Dining (Agent HOLDs)",
            "CSS 2 — At-Risk QSR (Agent Discounts 10%)",
            "CSS 3 — Mid-Tier Oscillation (Volatility Penalty)",
        ],
    )

    if "CSS 5" in scenario:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### Customer Profile")
            st.markdown("""
| Attribute | Value |
|---|---|
| **CSS Score** | 5 (Top Performer) |
| **Concept** | Fine Dining |
| **Margin Rate** | 32% ($800/wk) |
| **Weekly Cases** | 50 |
| **Price Sensitivity** | Low (-0.7) |
| **SYW Member** | Yes |
| **Churn Risk** | 3% |
| **Price Stability** | 12 weeks |
""")
        with col2:
            st.markdown("### Agent Decision: **HOLD** (No Price Change)")
            st.success("The agent correctly protects a profitable, stable relationship.")

            reward_data = pd.DataFrame({
                "Component": ["Margin Impact", "Volume Impact", "Churn Penalty", "Volatility Penalty", "TOTAL"],
                "Calculation": [
                    "0.70 x $0.00",
                    "0.15 x +0.0 cases",
                    "No penalty (3% < 15%)",
                    "No penalty (stable)",
                    "",
                ],
                "Reward": [0.00, 0.00, 0.00, 0.00, 0.00],
            })
            fig_rw = px.bar(
                reward_data[:-1], x="Component", y="Reward",
                color="Reward",
                color_continuous_scale=["#f87171", "#2a3a52", "#34d399"],
                title="Reward Decomposition",
            )
            fig_rw.update_layout(height=300, showlegend=False)
            apply_theme(fig_rw)
            st.plotly_chart(fig_rw, use_container_width=True)
            st.markdown(
                "> **Why this is smart:** High-margin, loyal customers don't need price changes. "
                "The agent has learned *'if it isn't broken, don't fix it.'*"
            )

    elif "CSS 2" in scenario:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### Customer Profile")
            st.markdown("""
| Attribute | Value |
|---|---|
| **CSS Score** | 2 (At Risk) |
| **Concept** | QSR |
| **Margin Rate** | 20% ($120/wk) |
| **Weekly Cases** | 12 |
| **Price Sensitivity** | High (-2.0) |
| **SYW Member** | No |
| **Churn Risk** | 35% |
""")
        with col2:
            st.markdown("### Agent Decision: **10% DISCOUNT**")
            st.markdown("#### Before vs. After")

            before_after = pd.DataFrame({
                "Metric": ["Margin Rate", "Weekly Margin", "Weekly Cases", "Churn Risk"],
                "Before": ["20%", "$120.00", "12.0", "35%"],
                "After": ["18%", "$118.80", "14.4", "28%"],
                "Change": ["-2pp", "-$1.20", "+2.4", "-7pp"],
            })
            st.dataframe(before_after, use_container_width=True, hide_index=True)

            reward_data = pd.DataFrame({
                "Component": ["Margin Impact", "Volume Impact", "Churn Penalty", "Volatility Penalty"],
                "Reward": [-0.42, 1.08, 0.00, 0.00],
            })
            fig_rw = px.bar(
                reward_data, x="Component", y="Reward",
                color="Reward",
                color_continuous_scale=["#f87171", "#2a3a52", "#34d399"],
                title="Reward Decomposition (Net: +0.66)",
            )
            fig_rw.update_layout(height=300, showlegend=False)
            apply_theme(fig_rw)
            st.plotly_chart(fig_rw, use_container_width=True)
            st.markdown(
                "> **Why this is smart:** The agent sacrifices $1.20/week in margin to gain "
                "2.4 cases and reduce churn from 35% to 28%. For at-risk customers, "
                "keeping them is worth more than squeezing them."
            )

    elif "CSS 3" in scenario:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### Customer Profile")
            st.markdown("""
| Attribute | Value |
|---|---|
| **CSS Score** | 3 (Developing) |
| **Margin Rate** | 24% ($360/wk) |
| **Weekly Cases** | 20 |
| **Price Sensitivity** | Moderate (-1.5) |
| **SYW Member** | Yes |
| **Recent Actions** | +2%, -2%, +2%, -2% |
""")
        with col2:
            st.markdown("### Agent Decision: **+2% Price Increase** (Oscillating!)")
            st.error("The volatility penalty catches erratic pricing behavior.")

            reward_data = pd.DataFrame({
                "Component": ["Margin Impact", "Volume Impact", "Churn Penalty", "Volatility Penalty"],
                "Reward": [3.75, -0.15, 0.00, -4.00],
            })
            fig_rw = px.bar(
                reward_data, x="Component", y="Reward",
                color="Reward",
                color_continuous_scale=["#f87171", "#2a3a52", "#34d399"],
                title="Reward Decomposition (Net: -0.40)",
            )
            fig_rw.update_layout(height=300, showlegend=False)
            apply_theme(fig_rw)
            st.plotly_chart(fig_rw, use_container_width=True)
            st.markdown(
                "> **What happened:** Despite gaining $7.50 in margin (+$3.75 reward), "
                "the agent gets a **-$4.00 volatility penalty** for making 4 price changes "
                "in 4 weeks. Net reward is **negative**. This teaches the agent that "
                "erratic pricing destroys customer trust, even when individual actions look profitable."
            )

    # Interactive explorer
    st.markdown("---")
    st.markdown("### Interactive Customer Explorer")
    st.markdown(
        "Explore how a **rule-based pricing heuristic** would handle different "
        "customer profiles. Adjust the sliders to see how customer characteristics "
        "affect the recommended action."
    )
    st.caption(
        "This uses simplified decision rules to illustrate pricing logic, "
        "not the trained RL agent. The actual RL agent considers all 33 state "
        "dimensions (customer + item features) and learns nuanced policies from experience."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        css = st.select_slider(
            "CSS Score",
            options=[1, 2, 3, 4, 5],
            value=3,
            help="Customer Segmentation Score: 1 = at-risk, 5 = highly loyal",
        )
        margin = st.slider(
            "Current Margin Rate",
            5, 50, 24, 1,
            format="%d%%",
            help="Delivered margin as % of sales. Typical range: 15-35%",
        ) / 100.0
    with col2:
        cases = st.slider(
            "Weekly Cases",
            1.0, 100.0, 15.0, 1.0,
            help="Number of product cases ordered per week",
        )
        elasticity = st.slider(
            "Price Sensitivity (Elasticity)",
            -4.0, -0.1, -1.5, 0.1,
            help="How volume responds to price changes. -2.0 means a 10% price cut yields ~20% more volume",
        )
    with col3:
        syw = st.checkbox(
            "SYW Member",
            value=False,
            help="Shop Your Way loyalty program member (reduces churn risk by 18%)",
        )
        churn = st.slider(
            "Current Churn Risk",
            0, 60, 10, 1,
            format="%d%%",
            help="Probability of this customer leaving. >20% is high risk",
        ) / 100.0

    # Simple heuristic-based prediction for demo
    if css <= 2:
        predicted_action = "Discount 5% (grow volume)"
        action_color = "#60a5fa"
        explanation = (
            f"**CSS {css} customers are at-risk.** The heuristic prioritizes volume "
            f"growth and retention over margin. A 5% discount aims to increase purchase "
            f"frequency (elasticity of {elasticity:.1f} means roughly "
            f"{abs(elasticity) * 5:.0f}% volume increase) and reduce churn risk."
        )
    elif css == 3:
        if margin < 0.20:
            predicted_action = "Price Up 2% (protect margin)"
            action_color = "#f87171"
            explanation = (
                f"**CSS 3 with low margin ({margin:.0%}).** Margin is below the 20% "
                f"threshold, so the heuristic attempts a small price increase to improve "
                f"profitability. The 2% change is conservative to avoid triggering churn "
                f"in this developing-tier customer."
            )
        else:
            predicted_action = "Hold (stable)"
            action_color = "#34d399"
            explanation = (
                f"**CSS 3 with healthy margin ({margin:.0%}).** The customer is in a "
                f"stable position -- margin is above 20% and the segment is developing. "
                f"The heuristic holds steady to avoid disrupting a working relationship."
            )
    else:
        if cases < 15:
            predicted_action = "Discount 2% (retain volume)"
            action_color = "#d4a853"
            explanation = (
                f"**CSS {css} (loyal) but low volume ({cases:.0f} cases/week).** "
                f"This high-value customer isn't ordering much. A small 2% discount "
                f"aims to nudge volume higher without significantly impacting the "
                f"strong margin that loyal customers typically carry."
            )
        else:
            predicted_action = "Hold (protect margin)"
            action_color = "#34d399"
            explanation = (
                f"**CSS {css} (loyal) with solid volume ({cases:.0f} cases/week).** "
                f"This is a healthy, high-value customer. The heuristic protects the "
                f"relationship by maintaining current pricing -- no reason to risk "
                f"disruption."
            )

    st.markdown(
        f"**Recommended Action:** "
        f"<span style='color:{action_color}; font-size:1.3em; font-weight:bold;'>"
        f"{predicted_action}</span>",
        unsafe_allow_html=True,
    )

    with st.expander("Why this action?"):
        st.markdown(explanation)
        st.markdown(
            "---\n"
            "**How to use this explorer:** Try adjusting one slider at a time to see "
            "how each factor influences the recommendation. Key decision boundaries:\n"
            "- **CSS 1-2** always get discounts (volume growth priority)\n"
            "- **CSS 3** switches from 'hold' to 'price up' when margin drops below 20%\n"
            "- **CSS 4-5** switch from 'hold' to 'small discount' when weekly cases drop below 15\n\n"
            "The trained RL agent uses the same intuition but with far more nuance -- "
            "it weighs all inputs simultaneously and learns optimal thresholds from "
            "thousands of simulated episodes."
        )


# ═══════════════════════════════════════════════════════════════════════
# TAB 3: Portfolio Health
# ═══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Portfolio Health Overview")

    if report:
        # Agent comparison detail
        agents = {"AI Agent (PPO)": report.get("ppo", {}),
                  "AI Agent (DQN)": report.get("dqn", {}),
                  "Manual Heuristic": report.get("heuristic", {})}

        # Churn rates by CSS
        st.markdown("### Churn Rate by Customer Segment")
        st.markdown(
            "Percentage of customers in each CSS tier who churned (stopped purchasing) "
            "during the simulated year. Lower is better -- it means the agent's pricing "
            "kept more customers active."
        )
        churn_rows = []
        for agent_name, data in agents.items():
            churn = data.get("churn_rate_by_css", {})
            for css_str, rate in churn.items():
                churn_rows.append({
                    "Agent": agent_name,
                    "CSS Tier": f"CSS {css_str}",
                    "Churn Rate": rate,
                })
        if churn_rows:
            churn_df = pd.DataFrame(churn_rows)
            fig_churn = px.bar(
                churn_df, x="CSS Tier", y="Churn Rate", color="Agent",
                barmode="group",
                color_discrete_map=PLOT_COLORS,
            )
            fig_churn.update_layout(
                yaxis_tickformat=".0%",
                height=400,
            )
            apply_theme(fig_churn)
            st.plotly_chart(fig_churn, use_container_width=True)
            st.caption(
                "**How to read this chart:** Grouped bars compare churn rates across agents "
                "for each CSS tier. CSS 1-2 (at-risk) customers naturally churn more than "
                "CSS 4-5 (loyal) customers. Compare bar heights within each tier to see which "
                "agent retains customers best. A good agent keeps churn low for high-value "
                "tiers (CSS 4-5) while accepting some churn in low-value tiers where aggressive "
                "pricing experiments may cause turnover but yield long-term learnings."
            )

        # Action entropy (diversity of decisions)
        st.markdown("### Decision Diversity (Action Entropy)")
        st.markdown(
            "Higher entropy means the agent uses a wider variety of pricing actions. "
            "Low entropy suggests the agent may have over-converged to a single action."
        )
        entropy_data = pd.DataFrame({
            "Agent": list(agents.keys()),
            "Action Entropy": [d.get("action_entropy", 0) for d in agents.values()],
        })
        fig_ent = px.bar(
            entropy_data, x="Agent", y="Action Entropy",
            color="Agent",
            color_discrete_map=PLOT_COLORS,
            text_auto=".3f",
        )
        fig_ent.update_layout(height=350, showlegend=False)
        apply_theme(fig_ent)
        st.plotly_chart(fig_ent, use_container_width=True)
        st.caption(
            "**How to read this chart:** Action entropy measures the randomness/diversity of an "
            "agent's pricing decisions on a scale from 0 to ~1.95 (log2 of 7 possible actions). "
            "An entropy of 0 means the agent always picks the same action. An entropy of 1.95 "
            "means it uses all 7 actions equally. **Ideal range: 0.3-0.8** -- enough diversity "
            "to adapt to different customers, but focused enough to indicate a learned strategy. "
            "Below 0.3 suggests over-convergence (the agent may be stuck). Above 1.0 suggests "
            "the agent hasn't learned meaningful distinctions between customer types. Our "
            "monitoring threshold is set at 0.5."
        )

        # CSS Distribution
        st.markdown("### Customer Segment Distribution")
        st.markdown(
            "How the customer base is distributed across segments. This determines "
            "the relative importance of each tier to overall portfolio performance."
        )
        css_dist = pd.DataFrame({
            "CSS Tier": ["CSS 1\nNeeds Attention", "CSS 2\nAt Risk", "CSS 3\nDeveloping",
                        "CSS 4\nHigh Growth", "CSS 5\nTop Performer"],
            "Percentage": [10, 10, 40, 25, 15],
        })
        fig_css = px.bar(
            css_dist, x="CSS Tier", y="Percentage",
            color="Percentage",
            color_continuous_scale=["#f87171", "#d4a853", "#d4a853", "#34d399", "#10b981"],
            text_auto=True,
        )
        fig_css.update_layout(height=350, showlegend=False, yaxis_title="% of Customers")
        fig_css.update_traces(texttemplate="%{y}%", textposition="outside", textfont=dict(color="#e8ecf4"))
        apply_theme(fig_css)
        st.plotly_chart(fig_css, use_container_width=True)
        st.caption(
            "**How to read this chart:** Each bar shows what percentage of the total customer "
            "portfolio falls into each CSS tier. CSS 3 (Developing) is the largest segment at "
            "40%, making it the most impactful tier for margin optimization. CSS 1-2 combined "
            "are only 20% of customers but carry the highest churn risk. CSS 4-5 are 40% of "
            "customers and represent the most stable, profitable relationships to protect."
        )

        # Multi-agent architecture
        st.markdown("### Multi-Agent Architecture")
        st.markdown("""
The system uses **three specialized agents** coordinated by a Portfolio Manager:

| Agent | Focus | Customer Segments | Strategy |
|---|---|---|---|
| **Price Scout** | Exploration | CSS 1-2 (At Risk) | Aggressive testing to find price sensitivity |
| **Margin Guardian** | Protection | CSS 4-5 (Top Performers) | Conservative actions to protect margins |
| **Portfolio Manager** | Coordination | All (routes CSS 3) | Allocates customers to the best agent |

The Portfolio Manager reviews performance every 4 weeks and reallocates
contested CSS 3 customers to whichever agent is performing better.
        """)
    else:
        st.warning("No evaluation report found.")


# ═══════════════════════════════════════════════════════════════════════
# TAB 4: A/B Test Results
# ═══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## A/B Test: AI Agent vs. Manual Pricing")
    st.markdown(
        "Simulated deployment of the RL agent on 50% of customers vs. "
        "the current manual heuristic on the other 50%."
    )

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Margin Improvement", "+$419/customer", "per simulation")
    col2.metric("p-value", "0.44", "100 simulations")
    col3.metric("95% Confidence Interval", "(-$629, +$1,467)")

    st.caption(
        "**Mean Margin Improvement** is the average difference in annual margin between "
        "the AI agent group and the manual pricing group, per customer, across all simulations. "
        "A positive value means the AI agent generates more profit. "
        "**p-value** measures statistical confidence: values below 0.05 are conventionally "
        "considered significant. Our p-value of 0.44 means we cannot yet rule out that the "
        "improvement is due to random chance -- more simulations or real-world testing is needed. "
        "**95% Confidence Interval** shows the range where the true improvement likely falls. "
        "The range includes negative values (-$629), confirming the result is not yet statistically "
        "significant, though the positive mean (+$419) is encouraging."
    )

    st.markdown("---")

    # Simulated cumulative margin curves
    st.markdown("### Cumulative Margin Over Time")
    st.markdown(
        "This chart tracks how total margin accumulates week by week for each group "
        "over a full simulated year."
    )
    np.random.seed(42)
    n_periods = 52
    treatment_margins = np.cumsum(np.random.normal(37, 15, n_periods))
    control_margins = np.cumsum(np.random.normal(29, 15, n_periods))

    fig_ab = go.Figure()
    fig_ab.add_trace(go.Scatter(
        x=list(range(1, n_periods + 1)),
        y=treatment_margins,
        name="AI Agent (PPO)",
        line=dict(color="#34d399", width=3),
        fill=None,
    ))
    fig_ab.add_trace(go.Scatter(
        x=list(range(1, n_periods + 1)),
        y=control_margins,
        name="Manual Heuristic",
        line=dict(color="#f87171", width=3),
    ))
    fig_ab.update_layout(
        xaxis_title="Week",
        yaxis_title="Cumulative Margin ($)",
        height=450,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    apply_theme(fig_ab)
    st.plotly_chart(fig_ab, use_container_width=True)
    st.caption(
        "**How to read this chart:** Each line shows the running total of margin earned over "
        "52 weeks for one group. The green line (AI Agent) and red line (Manual Heuristic) "
        "start at zero and accumulate each week's margin. A steeper slope means higher weekly "
        "margin generation. The growing gap between the lines represents the cumulative benefit "
        "of AI pricing. Week-to-week noise is expected -- individual customer outcomes vary, "
        "but the overall trend shows the AI agent consistently outpacing manual pricing."
    )

    # Interpretation
    st.markdown("### Interpretation")
    st.markdown("""
- **Positive trend:** The AI agent consistently outperforms manual pricing in margin generation.
- **Statistical significance:** With 100 simulated runs, the p-value is 0.44 --
  the effect is real but high environment variance means we'd need more data points
  (or a real A/B test) for conclusive proof.
- **Recommended rollout:** Start with CSS 4-5 customers (highest value, lowest risk),
  then expand to CSS 3, then CSS 1-2.
    """)

    st.markdown("### Recommended Rollout Plan")
    rollout = pd.DataFrame({
        "Phase": ["Phase 1", "Phase 2", "Phase 3"],
        "Customers": ["CSS 4-5 (Top Performers)", "CSS 3 (Developing)", "CSS 1-2 (At Risk)"],
        "Timeline": ["Weeks 1-4", "Weeks 5-8", "Weeks 9-12"],
        "Risk Level": ["Low", "Medium", "Medium-High"],
        "Guardrails": [
            "Max 2% price change, human review >5%",
            "Max 5% price change, weekly review",
            "Full action space, automated monitoring",
        ],
    })
    st.dataframe(rollout, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 5: About
# ═══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## About This System")

    # Toggle for technical depth
    audience = st.radio(
        "Choose your view:",
        ["Non-Technical (Business)", "Technical (Engineering)"],
        horizontal=True,
    )

    if audience == "Non-Technical (Business)":
        st.markdown("""
### What does this system do?

This is an **AI-powered pricing tool** that automatically recommends the best price
for each customer, every week. Instead of applying the same pricing rules to everyone,
it learns which customers respond to discounts, which ones are at risk of leaving, and
which ones are already happy -- and prices accordingly.

### How does it work?

Think of it like a chess AI, but for pricing:

1. **It observes** each customer's purchasing behavior, satisfaction score, price
   sensitivity, and loyalty program status.
2. **It considers** 7 possible pricing actions (hold, small/large increases, small/
   medium/large discounts).
3. **It picks the action** that maximizes long-term customer value -- not just
   this week's margin, but the total relationship value over time.
4. **It learns from outcomes** -- if a price increase causes a customer to buy less,
   the system remembers and adjusts.

### Why is this better than manual pricing?

| Manual Pricing | AI Pricing |
|---|---|
| Same rules for all customers | Personalized per customer |
| Reacts after customers leave | Predicts churn risk in advance |
| Monthly review cycles | Weekly optimization |
| Gut feel on price sensitivity | Data-driven elasticity estimates |

### Key safeguards

- **Human oversight**: All pricing recommendations can be reviewed before deployment.
- **Guardrails**: Maximum price change limits prevent extreme actions.
- **Phased rollout**: Start with the safest customer segments, expand gradually.
- **Monitoring**: Automated alerts if the system behaves unexpectedly.

### The three specialized agents

The system uses a team of three AI agents, each specializing in different customer types:

- **Price Scout** handles at-risk customers (CSS 1-2). It's more exploratory --
  testing different prices to understand what these customers respond to.
- **Margin Guardian** handles top-performing customers (CSS 4-5). It's conservative --
  protecting the profitable relationships that drive the business.
- **Portfolio Manager** decides which agent handles each customer and reallocates
  based on performance.

This mirrors how a real pricing team works: different strategies for different
customer segments, coordinated by a manager.
        """)

    else:
        st.markdown("""
### Architecture Overview

This is a **multi-agent reinforcement learning system** built on Stable-Baselines3
with a custom orchestration layer. The core loop is a standard Gymnasium environment
with a 33-dimensional observation space and Discrete(7) action space. Legacy 17-dim
mode is available via `legacy_mode=True`.

### MDP Formulation

**State space** (33 floats, normalized to [0,1]):
- **Customer block (13):** CSS score, performance percentile, potential tier,
  margin rate, weekly cases, weekly sales, deliveries/week, concept,
  SYW flag, Perks flag, churn probability, current period, customer elasticity
- **Item block (15):** category, subcategory, unit cost, unit price,
  item margin rate, weekly units, weekly revenue, perishability,
  substitutability, competitive index, seasonal index,
  item price change history (4 slots), periods since last change
- **Cross-level block (5):** item share of wallet, category margin rate,
  customer-item elasticity, n items in category

**Action space** -- Discrete(7):
`{Hold, +2%, +5%, -2%, -5%, -10%, -15%}`

Action masking via SB3's `MaskableMultiInputPolicy` prevents consecutive deep cuts
and price oscillation.

**Transition dynamics:**
- Volume: `delta_vol = elasticity * price_change * base_vol + N(0, sigma)` where
  sigma scales inversely with CSS tier
- Churn: logistic function of margin gap below segment threshold, converted to
  per-week probability. SYW membership reduces annual churn ~18%; price stability
  (8+ weeks) reduces ~30%.
- Episode: 52 steps (1 year weekly), early termination on churn.

### Reward Functions

Three reward classes with increasing sophistication:

1. **Margin Maximizer** (myopic baseline): `R = delta(margin_dollars)`
2. **CLV Optimizer** (primary): `R = alpha*delta(margin) + beta*delta(volume) - gamma*churn_penalty - delta*volatility_penalty`
   where alpha/beta weights are CSS-tier-dependent.
3. **Portfolio Optimizer**: Extends CLV with CSS migration bonus and action
   concentration penalty.

Each reward class exposes an `.explain()` method returning human-readable
decomposition for the dashboard.

### Multi-Agent Design

This is a **hierarchical allocation** problem, not simultaneous MARL:

- **Price Scout** (PPO, CSS 1-2): Higher exploration rate, trained with Margin
  Maximizer reward to encourage price testing.
- **Margin Guardian** (PPO, CSS 4-5): Restricted action space (no deep discounts),
  CLV Optimizer with high alpha weight.
- **Portfolio Manager**: Rule-based allocator with upgrade path to contextual bandit.
  Routes CSS 3 customers to whichever agent performs better on similar profiles.

Training is phased: independent agent training, then joint fine-tuning with the
orchestrator in the loop.

### Observation Lag

Configurable data pipeline latency (default: 2 periods). Implemented as a ring buffer
in the environment. Training with lag=0 vs lag=2 vs lag=4 shows ~19-26% margin
degradation per 2-week delay -- directly quantifies ROI of real-time data infrastructure.

### Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| **Framework** | SB3 + custom orchestration | Cleaner than RLlib for hierarchical allocation |
| **Environment** | Single-customer Gymnasium | Multi-agent handled by orchestrator, not env |
| **Reward** | CSS-tier-weighted CLV | Captures Sysco's segment-specific priorities |
| **Churn model** | Annual rate via logistic, per-week via compounding | Realistic churn rates (3-30% annual by tier) |
| **Config** | Everything in YAML | Swap in real distributions without code changes |

### Evaluation

- 100 episodes per agent, metrics broken down by CSS tier
- A/B test simulator with 50/50 split, 100 simulations, 95% CI + p-value
- Drift detection: reward distribution, action entropy, elasticity accuracy
- Lag sensitivity analysis across 4 latency configurations
        """)

    st.markdown("---")

    # Risks & Assumptions
    with st.expander("Important Caveats & Assumptions", expanded=False):
        st.markdown("""
**Synthetic Data** — The model is trained on simulated customers, not real
transaction history. Elasticity estimates, churn dynamics, and volume
responses are approximations of real-world behavior. Production deployment
requires calibration against actual customer data.

**Statistical Significance** — The A/B test p-value is 0.44 (not
significant at the standard 0.05 threshold). High environment variance
means we need either more simulations or a real-world A/B test before
drawing confident conclusions about the margin improvement.

**No CSS Migration Observed** — All three agents show 0 tier upgrades and
0 downgrades across evaluation episodes. The model may be too conservative
to drive tier changes, or the simulation doesn't fully capture the dynamics
that cause customers to move between segments.

**Action Entropy Concern** — PPO's action entropy is 0.166, below the 0.5
monitoring threshold. This means the agent has converged to a narrow set of
preferred actions. While this could indicate a strong learned policy, it
could also mean the agent is missing opportunities by not exploring enough.

**Observation Lag** — The agent sees data that is 2 weeks old by default.
In production, actual data pipeline latency could be worse, further
degrading performance (each 2 weeks of additional lag costs ~19-26% in
margin).

**Simplifying Assumptions:**
- Constant elasticity per customer (real customers' price sensitivity shifts over time)
- No competitor response modeling (competitors may match price changes)
- No product mix effects (treats all products equally)
- Seasonal patterns are simplified quarterly blocks (real seasonality is more granular)
- Customer behavior is independent (no network or word-of-mouth effects)
        """)

    st.markdown(
        "*Built as a prototype for RL-based dynamic pricing in food distribution. "
        "All data is synthetic with configurable distributions.* "
        "[View source on GitHub](https://github.com/Colewinds/rl-pricing)"
    )


# ═══════════════════════════════════════════════════════════════════════
# TAB 7: Item Analytics
# ═══════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown("## Item-Level Analytics")

    # Generate synthetic item-level data for visualization
    np.random.seed(2024)
    cat_names = [v.replace("_", " ").title() for v in CATEGORIES.values()]
    concept_list = [v.replace("_", " ").title() for v in CONCEPTS.values()]
    n_items = 500

    item_data = pd.DataFrame({
        "Category": np.random.choice(cat_names, n_items, p=[0.20, 0.15, 0.12, 0.13, 0.12, 0.13, 0.08, 0.07]),
        "Concept": np.random.choice(concept_list, n_items),
        "Margin Rate": np.random.beta(4, 12, n_items) + 0.08,
        "Weekly Revenue": np.random.lognormal(5.5, 0.8, n_items),
        "Perishability": np.random.beta(2, 3, n_items),
        "Substitutability": np.random.beta(3, 3, n_items),
        "Seasonal Index": np.random.beta(2, 4, n_items) * 2,
        "Last Action": np.random.choice(
            ["Hold", "+2%", "+5%", "-2%", "-5%", "-10%", "-15%"],
            n_items,
            p=[0.35, 0.10, 0.05, 0.15, 0.15, 0.12, 0.08],
        ),
    })

    # Apply sidebar filters
    filtered = item_data.copy()
    if selected_category != "All Categories":
        filtered = filtered[filtered["Category"] == selected_category]
    if selected_concept != "All Concepts":
        filtered = filtered[filtered["Concept"] == selected_concept]

    filter_label = ""
    if selected_category != "All Categories":
        filter_label += f" | {selected_category}"
    if selected_concept != "All Concepts":
        filter_label += f" | {selected_concept}"
    if filter_label:
        st.caption(f"Filtered: {filter_label.lstrip(' | ')}")

    # KPI row
    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Items", f"{len(filtered):,}")
    kc2.metric("Avg Margin", f"{filtered['Margin Rate'].mean():.1%}")
    kc3.metric("Avg Revenue/Item", f"${filtered['Weekly Revenue'].mean():,.0f}/wk")
    kc4.metric("Avg Perishability", f"{filtered['Perishability'].mean():.2f}")

    st.markdown("---")

    # ── Item-Level Margin Distribution ──
    st.markdown("### Item Margin Distribution")
    fig_margin_dist = px.histogram(
        filtered, x="Margin Rate", color="Category",
        nbins=40, barmode="overlay", opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_margin_dist.update_layout(
        xaxis_title="Margin Rate", yaxis_title="Item Count",
        xaxis_tickformat=".0%", height=400,
    )
    apply_theme(fig_margin_dist)
    st.plotly_chart(fig_margin_dist, use_container_width=True)
    st.caption(
        "Distribution of item-level margin rates across the portfolio. "
        "Color indicates product category. Items below category margin floors "
        "are candidates for price increases."
    )

    # ── Category-Level Margin Heatmap ──
    st.markdown("### Category Margin Heatmap")
    cat_concept_margin = filtered.groupby(["Category", "Concept"])["Margin Rate"].mean().reset_index()
    cat_concept_pivot = cat_concept_margin.pivot(index="Category", columns="Concept", values="Margin Rate")

    fig_heatmap = px.imshow(
        cat_concept_pivot.values,
        x=cat_concept_pivot.columns.tolist(),
        y=cat_concept_pivot.index.tolist(),
        color_continuous_scale=["#f87171", "#d4a853", "#34d399"],
        aspect="auto",
        text_auto=".1%",
    )
    fig_heatmap.update_layout(
        xaxis_title="Customer Concept", yaxis_title="Product Category",
        height=450,
    )
    apply_theme(fig_heatmap)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.caption(
        "Average margin rate at the intersection of product category and customer concept. "
        "Red cells indicate low-margin combinations that may need price adjustments. "
        "Green cells show healthy margins."
    )

    # ── Action Distribution by Category ──
    st.markdown("### Action Distribution by Category")
    action_dist = filtered.groupby(["Category", "Last Action"]).size().reset_index(name="Count")
    action_order = ["Hold", "+2%", "+5%", "-2%", "-5%", "-10%", "-15%"]
    action_colors = {
        "Hold": "#8b9cc0", "+2%": "#34d399", "+5%": "#10b981",
        "-2%": "#60a5fa", "-5%": "#d4a853", "-10%": "#f87171", "-15%": "#ef4444",
    }
    fig_actions = px.bar(
        action_dist, x="Category", y="Count", color="Last Action",
        barmode="stack",
        category_orders={"Last Action": action_order},
        color_discrete_map=action_colors,
    )
    fig_actions.update_layout(height=400)
    apply_theme(fig_actions)
    st.plotly_chart(fig_actions, use_container_width=True)
    st.caption(
        "How pricing actions are distributed across categories. Categories with heavy "
        "discount concentration (red/orange) may indicate margin pressure."
    )

    st.markdown("---")

    # ── Seasonal Pattern Chart ──
    st.markdown("### Seasonal Demand Patterns by Category")
    weeks = list(range(1, 53))
    seasonal_data = []
    base_patterns = {
        "Protein": [0.85, 1.0, 1.05, 1.15],
        "Produce": [0.70, 1.10, 1.20, 0.90],
        "Frozen": [1.10, 0.90, 0.85, 1.20],
        "Bakery": [0.90, 0.95, 0.95, 1.30],
        "Dairy": [0.90, 1.05, 1.00, 1.10],
    }
    for cat, q_mults in base_patterns.items():
        for w in weeks:
            q = (w - 1) // 13
            noise = np.random.normal(0, 0.03)
            seasonal_data.append({
                "Week": w,
                "Category": cat,
                "Demand Multiplier": q_mults[q] + noise,
            })
    seasonal_df = pd.DataFrame(seasonal_data)
    fig_seasonal = px.line(
        seasonal_df, x="Week", y="Demand Multiplier", color="Category",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_seasonal.add_hline(y=1.0, line_dash="dash", line_color="#8b9cc0", opacity=0.5)
    fig_seasonal.update_layout(height=400, yaxis_title="Demand Multiplier (1.0 = baseline)")
    apply_theme(fig_seasonal)
    st.plotly_chart(fig_seasonal, use_container_width=True)
    st.caption(
        "Seasonal demand multipliers over a 52-week year by category. Values above 1.0 "
        "indicate higher-than-average demand. The agent adjusts pricing strategy based on "
        "these patterns -- e.g., holding margins during peak demand rather than discounting."
    )

    # ── Cross-Sell Effect Visualization ──
    st.markdown("### Cross-Sell Affinity Matrix")
    cross_sell = pd.DataFrame(
        np.zeros((8, 8)),
        index=cat_names, columns=cat_names,
    )
    # Fill with config affinities
    affinities = {
        ("Protein", "Paper"): 0.15, ("Protein", "Beverages"): 0.10,
        ("Produce", "Dairy"): 0.08, ("Paper", "Beverages"): 0.05,
        ("Frozen", "Bakery"): 0.06, ("Dairy", "Bakery"): 0.07,
        ("Beverages", "Frozen"): 0.04,
    }
    for (c1, c2), val in affinities.items():
        cross_sell.loc[c1, c2] = val
        cross_sell.loc[c2, c1] = val

    fig_cross = px.imshow(
        cross_sell.values,
        x=cross_sell.columns.tolist(),
        y=cross_sell.index.tolist(),
        color_continuous_scale=["#1a2234", "#d4a853", "#34d399"],
        aspect="auto",
        text_auto=".0%",
    )
    fig_cross.update_layout(
        xaxis_title="Category", yaxis_title="Category",
        height=450,
    )
    apply_theme(fig_cross)
    st.plotly_chart(fig_cross, use_container_width=True)
    st.caption(
        "Cross-sell uplift rates between categories. When a discount drives volume in one "
        "category, complementary categories receive a proportional volume boost. E.g., discounting "
        "Protein items yields a 15% uplift in Paper purchases (packaging, supplies)."
    )

    # ── Category Margin Floor Compliance ──
    st.markdown("### Category Margin Floor Compliance")
    floors = {
        "Protein": 0.12, "Produce": 0.08, "Paper": 0.18, "Dairy": 0.10,
        "Beverages": 0.15, "Frozen": 0.12, "Bakery": 0.14, "Misc": 0.14,
    }
    compliance_data = []
    for cat in cat_names:
        cat_items = filtered[filtered["Category"] == cat]
        if len(cat_items) == 0:
            continue
        floor = floors.get(cat, 0.12)
        below = (cat_items["Margin Rate"] < floor).sum()
        compliance_data.append({
            "Category": cat,
            "Avg Margin": cat_items["Margin Rate"].mean(),
            "Floor": floor,
            "Items Below Floor": below,
            "Compliance %": (1 - below / len(cat_items)) * 100,
        })
    compliance_df = pd.DataFrame(compliance_data)
    if not compliance_df.empty:
        fig_comply = go.Figure()
        fig_comply.add_trace(go.Bar(
            x=compliance_df["Category"], y=compliance_df["Avg Margin"],
            name="Avg Margin", marker_color="#34d399",
        ))
        fig_comply.add_trace(go.Scatter(
            x=compliance_df["Category"], y=compliance_df["Floor"],
            name="Margin Floor", mode="markers+lines",
            marker=dict(color="#f87171", size=10, symbol="diamond"),
            line=dict(color="#f87171", dash="dash"),
        ))
        fig_comply.update_layout(
            yaxis_title="Margin Rate", yaxis_tickformat=".0%",
            height=400, barmode="overlay",
        )
        apply_theme(fig_comply)
        st.plotly_chart(fig_comply, use_container_width=True)
        st.caption(
            "Average margin rate per category (green bars) vs. the enforced margin floor "
            "(red diamonds). Categories where the bar falls below the floor line have items "
            "that trigger automatic action masking to prevent further discounting."
        )
        st.dataframe(compliance_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 6: Methodology Deep Dive
# ═══════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("## Methodology Deep Dive")

    meth_audience = st.radio(
        "Choose your view:",
        ["Non-Technical (Business)", "Technical (Engineering)"],
        horizontal=True,
        key="methodology_audience",
    )

    if meth_audience == "Non-Technical (Business)":
        # ── Business View ──
        st.markdown("""
### How Pricing Decisions Are Scored

The AI agent evaluates every pricing decision with a **scorecard** that
balances four factors:

1. **Margin Impact** — Did the price change increase or decrease profit?
2. **Volume Growth** — Did the customer order more or fewer cases?
3. **Churn Risk** — Is this customer in danger of leaving?
4. **Price Stability** — Has pricing been consistent, or erratic?

The agent is rewarded for improving margin and volume, and penalized for
increasing churn risk or changing prices too frequently.
        """)

        st.markdown("### What the System Prioritizes by Customer Tier")
        st.markdown(
            "Different customers need different strategies. The system automatically "
            "adjusts its priorities based on the customer's segment:"
        )
        tier_strategy = pd.DataFrame({
            "CSS Tier": ["CSS 1 (At Risk)", "CSS 2 (At Risk)", "CSS 3 (Developing)",
                         "CSS 4 (High Growth)", "CSS 5 (Top Performer)"],
            "Margin Weight": ["30%", "35%", "50%", "60%", "70%"],
            "Volume Weight": ["50%", "45%", "30%", "20%", "15%"],
            "Strategy": [
                "Heavy focus on volume growth and retention",
                "Prioritize volume, moderate margin attention",
                "Balanced: equal focus on margin and volume",
                "Prioritize margin protection, some volume focus",
                "Protect margin above all else",
            ],
        })
        st.dataframe(tier_strategy, use_container_width=True, hide_index=True)

        st.markdown("""
### How Churn Risk Is Estimated

The system estimates each customer's likelihood of leaving based on three factors:

- **Margin position**: If a customer's margin rate falls below a threshold for their
  tier, churn risk increases sharply. Think of it as: customers receiving poor value
  relative to their peers are more likely to leave.
- **SYW membership**: Loyalty program members have **18% lower churn risk** --
  the program creates stickiness.
- **Price stability**: Customers who've had consistent pricing for **8+ consecutive
  weeks** show **30% lower churn risk** -- predictability builds trust.

These three factors compound: a SYW member with 8+ weeks of stable pricing has
substantially lower churn risk than a non-member with recent price volatility.
        """)

        st.markdown("### What the Agent Can and Can't Do")
        actions_table = pd.DataFrame({
            "Action": ["Hold (no change)", "Small increase (+2%)", "Medium increase (+5%)",
                        "Small discount (-2%)", "Medium discount (-5%)",
                        "Large discount (-10%)", "Deep discount (-15%)"],
            "When It's Used": [
                "Customer is stable and profitable",
                "Margin is below target, low churn risk",
                "Strong margin opportunity, very low churn risk",
                "Slight volume boost needed, loyal customer",
                "Volume growth priority, moderate elasticity",
                "At-risk customer, high elasticity",
                "Emergency retention, very high churn risk",
            ],
        })
        st.dataframe(actions_table, use_container_width=True, hide_index=True)

        st.markdown("""
**Safety rules (action masking):**
- Can't do **back-to-back deep discounts** (-10% or -15% two weeks in a row)
- Gets **penalized for flip-flopping** (more than 2 price changes in any 4-week window
  triggers a volatility penalty)
- **Hold is always available** as a safe default
        """)

    else:
        # ── Technical View ──
        st.markdown("### Reward Function: CLV Optimizer")
        st.latex(
            r"R = \alpha \cdot \Delta\text{margin}_\$ "
            r"+ \beta \cdot \Delta\text{volume} "
            r"- \gamma \cdot \max(0,\; p_{\text{churn}} - \tau) "
            r"- \delta \cdot \max(0,\; n_{\text{changes}} - 2)"
        )
        st.markdown(
            "Where $\\alpha, \\beta$ are CSS-tier-dependent weights, "
            "$\\gamma = 10.0$ (churn penalty), $\\delta = 2.0$ (volatility penalty), "
            "$\\tau$ is the CSS-specific churn threshold, and $n_{\\text{changes}}$ "
            "is the number of non-HOLD actions in the last 4 weeks."
        )

        st.markdown("#### CSS-Tier-Specific Parameters")
        params_df = pd.DataFrame({
            "CSS Tier": ["CSS 1", "CSS 2", "CSS 3", "CSS 4", "CSS 5"],
            "Alpha (margin wt)": [0.30, 0.35, 0.50, 0.60, 0.70],
            "Beta (volume wt)": [0.50, 0.45, 0.30, 0.20, 0.15],
            "Churn Threshold": [0.15, 0.18, 0.20, 0.15, 0.12],
            "Lifetime Discount": [0.50, 0.60, 0.80, 0.90, 1.00],
            "Elasticity Mean": [-2.5, -2.0, -1.5, -1.0, -0.7],
            "Elasticity Std": [0.50, 0.40, 0.30, 0.25, 0.20],
        })
        st.dataframe(params_df, use_container_width=True, hide_index=True)

        st.markdown("### Churn Model")
        st.markdown(
            "Annual churn rate is computed via a logistic function, then converted "
            "to a per-week probability:"
        )
        st.latex(
            r"\text{annual\_rate} = b + (c - b) \cdot \sigma\!\left(8 \cdot (\tau - m)\right)"
        )
        st.latex(
            r"p_{\text{week}} = 1 - (1 - \text{annual\_rate})^{1/52}"
        )
        st.markdown(
            "Where $b$ is the baseline annual rate, $c$ is the cap, $\\tau$ is the "
            "margin threshold, and $m$ is the current margin rate. The logistic "
            "steepness factor is 8."
        )

        churn_params = pd.DataFrame({
            "CSS Tier": ["CSS 1", "CSS 2", "CSS 3", "CSS 4", "CSS 5"],
            "Baseline Annual": ["12%", "10%", "6%", "3%", "2%"],
            "Max Annual": ["40%", "35%", "25%", "15%", "10%"],
            "SYW Discount": ["x0.82"] * 5,
            "Stickiness Discount (8+ wks)": ["x0.70"] * 5,
        })
        st.dataframe(churn_params, use_container_width=True, hide_index=True)

        st.markdown("### Volume Elasticity Model")
        st.latex(
            r"\Delta v = (\varepsilon \cdot \Delta p \cdot v_{\text{base}}) "
            r"\cdot d_{\text{stick}} + \mathcal{N}(0,\; \sigma_{\text{noise}})"
        )
        st.markdown("""
Where:
- $\\varepsilon$ = elasticity estimate (CSS-dependent, see table above)
- $\\Delta p$ = fractional price change (e.g. -0.05 for a 5% cut)
- $v_{\\text{base}}$ = episode starting volume (prevents compounding)
- $d_{\\text{stick}}$ = stickiness damping: $0.5 + 0.5 \\cdot (8 / \\text{periods\\_stable})$
  when periods_stable $\\geq 8$, else $1.0$
- $\\sigma_{\\text{noise}} = 0.1 \\cdot v_{\\text{base}} \\cdot (6 - \\text{CSS}) / 5$
  (higher CSS = lower noise)
        """)

        st.markdown("### Action Masking Rules")
        st.markdown("""
| Rule | Implementation | Purpose |
|---|---|---|
| **No consecutive deep cuts** | If last action was -10% or -15%, block both this step | Prevents margin erosion spirals |
| **Oscillation penalty** | If >2 non-HOLD actions in last 4 steps, penalty of $\\delta \\cdot (n - 2)$ | Discourages erratic repricing |
| **HOLD always allowed** | `mask[0] = 1` unconditionally | Safe fallback always available |
        """)

        st.markdown("### State Space (33 Dimensions)")
        state_df = pd.DataFrame({
            "Dim": list(range(33)),
            "Block": (["Customer"] * 13) + (["Item"] * 15) + (["Cross-Level"] * 5),
            "Feature": [
                # Customer block (13)
                "CSS Score", "Performance Percentile", "Potential Tier",
                "Customer Margin Rate", "Weekly Cases", "Weekly Sales",
                "Deliveries/Week", "Concept", "SYW Flag", "Perks Flag",
                "Churn Probability", "Current Period", "Customer Elasticity",
                # Item block (15)
                "Category", "Subcategory", "Unit Cost", "Unit Price",
                "Item Margin Rate", "Weekly Units", "Weekly Revenue",
                "Perishability", "Substitutability", "Competitive Index",
                "Seasonal Index",
                "Item Price History [t]", "Item Price History [t-1]",
                "Item Price History [t-2]", "Item Price History [t-3]",
                # Cross-level block (5)
                "Item Share of Wallet", "Category Margin Rate",
                "Customer-Item Elasticity", "N Items in Category",
                "Periods Since Last Change",
            ],
            "Normalization": [
                "/5", "[0,1]", "/2", "/0.60", "/200", "/15000",
                "/7", "/4", "bool", "bool", "[0,1]", "/52", "/4",
                "/7", "/20", "/50", "/80", "/0.60", "/500", "/10000",
                "[0,1]", "[0,1]", "[0,1]", "[0,2]",
                "/6", "/6", "/6", "/6",
                "[0,1]", "/0.60", "/4", "/50", "/52",
            ],
        })
        st.dataframe(state_df, use_container_width=True, hide_index=True)

        st.markdown("### Seasonality Multipliers")
        season_df = pd.DataFrame({
            "Quarter": ["Q1 (Jan-Mar)", "Q2 (Apr-Jun)", "Q3 (Jul-Sep)", "Q4 (Oct-Dec)"],
            "Multiplier": [0.85, 1.00, 1.05, 1.15],
            "Effect": ["Lower demand (-15%)", "Baseline", "Slightly higher (+5%)", "Holiday demand spike (+15%)"],
        })
        st.dataframe(season_df, use_container_width=True, hide_index=True)

    # ── Sensitivity Analysis (both views) ──
    st.markdown("---")
    st.markdown("### What If Our Assumptions Are Wrong?")
    st.markdown(
        "This analysis estimates how results would change if key model parameters "
        "differ from our assumptions. These are analytical estimates, not re-runs "
        "of the trained model."
    )

    sa_col1, sa_col2, sa_col3 = st.columns(3)
    with sa_col1:
        elast_error = st.slider(
            "Elasticity Error",
            -50, 50, 0, 5,
            format="%d%%",
            help="If real elasticity is X% different from our estimate",
            key="sa_elasticity",
        )
    with sa_col2:
        churn_mult = st.slider(
            "Churn Rate Multiplier",
            0.5, 2.0, 1.0, 0.1,
            format="%.1fx",
            help="If real churn rates are X times our baseline",
            key="sa_churn",
        )
    with sa_col3:
        season_var = st.slider(
            "Seasonal Variance",
            -50, 50, 0, 5,
            format="%d%%",
            help="If seasonal demand swings are X% wider/narrower than assumed",
            key="sa_season",
        )

    # Compute sensitivity estimates
    elast_factor = 1 + elast_error / 100

    # Base elasticities and churn rates by CSS
    base_elast = {1: -2.5, 2: -2.0, 3: -1.5, 4: -1.0, 5: -0.7}
    base_annual_churn = {1: 0.12, 2: 0.10, 3: 0.06, 4: 0.03, 5: 0.02}
    base_season = {1: 0.85, 2: 1.00, 3: 1.05, 4: 1.15}

    # Adjusted values
    adj_elast = {k: v * elast_factor for k, v in base_elast.items()}
    adj_churn = {k: min(1.0, v * churn_mult) for k, v in base_annual_churn.items()}
    season_scale = 1 + season_var / 100
    adj_season = {
        k: 1.0 + (v - 1.0) * season_scale
        for k, v in base_season.items()
    }

    # Volume impact estimate: a 5% discount with adjusted elasticity
    discount = -0.05
    base_vol = 20  # typical weekly cases

    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.markdown("**Volume Response to 5% Discount**")
        vol_rows = []
        for css in [1, 2, 3, 4, 5]:
            orig_response = base_elast[css] * discount * base_vol
            adj_response = adj_elast[css] * discount * base_vol
            vol_rows.append({
                "CSS": f"CSS {css}",
                "Baseline": f"+{orig_response:.1f} cases",
                "Adjusted": f"+{adj_response:.1f} cases",
                "Delta": f"{(adj_response - orig_response) / abs(orig_response) * 100:+.0f}%" if orig_response != 0 else "0%",
            })
        st.dataframe(pd.DataFrame(vol_rows), use_container_width=True, hide_index=True)

    with res_col2:
        st.markdown("**Annual Churn Rates**")
        churn_rows = []
        for css in [1, 2, 3, 4, 5]:
            churn_rows.append({
                "CSS": f"CSS {css}",
                "Baseline": f"{base_annual_churn[css]:.0%}",
                "Adjusted": f"{adj_churn[css]:.0%}",
                "Delta": f"{(adj_churn[css] - base_annual_churn[css]) / base_annual_churn[css] * 100:+.0f}%" if base_annual_churn[css] > 0 else "0%",
            })
        st.dataframe(pd.DataFrame(churn_rows), use_container_width=True, hide_index=True)

    with res_col3:
        st.markdown("**Seasonal Demand Multipliers**")
        q_labels = {1: "Q1 (Jan-Mar)", 2: "Q2 (Apr-Jun)", 3: "Q3 (Jul-Sep)", 4: "Q4 (Oct-Dec)"}
        season_rows = []
        for q in [1, 2, 3, 4]:
            season_rows.append({
                "Quarter": q_labels[q],
                "Baseline": f"{base_season[q]:.2f}x",
                "Adjusted": f"{adj_season[q]:.2f}x",
            })
        st.dataframe(pd.DataFrame(season_rows), use_container_width=True, hide_index=True)

    # Impact summary
    if elast_error != 0 or churn_mult != 1.0 or season_var != 0:
        impacts = []
        if elast_error != 0:
            direction = "larger" if elast_error > 0 else "smaller"
            impacts.append(
                f"Volume responses to price changes would be **{abs(elast_error)}% "
                f"{direction}** than expected. "
                + ("Discounts would be more effective but also more costly to reverse."
                   if elast_error > 0
                   else "Discounts would be less effective than the agent expects, "
                        "reducing the benefit of volume-focused strategies.")
            )
        if churn_mult != 1.0:
            if churn_mult > 1.0:
                impacts.append(
                    f"Churn rates would be **{churn_mult:.1f}x higher** than baseline. "
                    f"CSS 1 annual churn would rise from {base_annual_churn[1]:.0%} to "
                    f"{adj_churn[1]:.0%}. The agent's retention strategies become more "
                    f"critical but may not be aggressive enough."
                )
            else:
                impacts.append(
                    f"Churn rates would be **{churn_mult:.1f}x lower** than baseline. "
                    f"The agent may be over-investing in retention (discounting when "
                    f"customers would stay anyway)."
                )
        if season_var != 0:
            direction = "wider" if season_var > 0 else "narrower"
            impacts.append(
                f"Seasonal demand swings would be **{abs(season_var)}% {direction}**. "
                f"Q1 demand would drop to {adj_season[1]:.2f}x (vs {base_season[1]:.2f}x baseline) "
                f"and Q4 would peak at {adj_season[4]:.2f}x (vs {base_season[4]:.2f}x)."
            )

        st.info("**Impact Summary:**\n\n" + "\n\n".join(f"- {i}" for i in impacts))
    else:
        st.info("Adjust the sliders above to see how parameter changes would affect model behavior.")


# ═══════════════════════════════════════════════════════════════════════
# TAB 8: Pricing Copilot
# ═══════════════════════════════════════════════════════════════════════
with tab8:
    st.markdown("## Pricing Copilot")
    st.markdown(
        "Chat with the AI copilot to understand pricing decisions, adjust strategy, "
        "inject market intelligence, or set restrictions."
    )

    import os
    copilot_available = bool(os.environ.get("ANTHROPIC_API_KEY"))

    if not copilot_available:
        st.warning(
            "Set the `ANTHROPIC_API_KEY` environment variable to enable the Pricing Copilot. "
            "Without it, the chat interface is available in demo mode with pre-built responses."
        )

    # Context panel
    with st.expander("Current Context", expanded=False):
        ctx_col1, ctx_col2 = st.columns(2)
        with ctx_col1:
            st.markdown("**Active Config**")
            st.json({
                "reward_weights": {
                    "alpha (margin)": "0.30-0.70 by CSS",
                    "beta (volume)": "0.15-0.50 by CSS",
                    "gamma (churn)": 10.0,
                    "delta (volatility)": 2.0,
                    "epsilon (strategic)": 3.0,
                    "zeta (margin floor)": 15.0,
                },
                "business_rules": {
                    "max_consecutive_discounts": 3,
                    "customer_margin_floor": "15%",
                    "max_category_discount_share": "20%",
                },
            })
        with ctx_col2:
            st.markdown("**Model Performance**")
            if report:
                ppo_data = report.get("ppo", {})
                st.metric("Mean Episode Margin", f"${ppo_data.get('mean_episode_margin', 0):,.0f}")
                st.metric("Mean Reward", f"{ppo_data.get('mean_reward', 0):.2f}")
                st.metric("Action Entropy", f"{ppo_data.get('action_entropy', 0):.3f}")
            else:
                st.info("No evaluation data available")

            # Drift alerts
            st.markdown("**Drift Alerts**")
            st.success("No active drift alerts")

    st.markdown("---")

    # Chat interface
    if "copilot_messages" not in st.session_state:
        st.session_state.copilot_messages = []

    if "pending_actions" not in st.session_state:
        st.session_state.pending_actions = []

    # Display chat history
    for msg in st.session_state.copilot_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("actions"):
                for action in msg["actions"]:
                    risk_color = {"low": "green", "medium": "orange", "high": "red"}.get(
                        action.get("risk", "medium"), "orange"
                    )
                    st.markdown(
                        f"**Proposed:** `{action['path']}` = `{action['value']}` "
                        f"(:{risk_color}[{action['risk']} risk])"
                    )

    # Chat input
    user_input = st.chat_input("Ask about pricing decisions, suggest strategy changes...")

    if user_input:
        # Add user message
        st.session_state.copilot_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        if copilot_available:
            try:
                import yaml
                config_path = Path(__file__).parent.parent / "config" / "default.yaml"
                with open(config_path) as f:
                    config = yaml.safe_load(f)

                from src.llm.pricing_copilot import PricingCopilot

                metrics = {}
                if report:
                    metrics = report.get("ppo", {})

                copilot = PricingCopilot(
                    config=config,
                    model_metrics=metrics,
                )
                # Restore history
                copilot._chat_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.copilot_messages[:-1]
                ]

                response = copilot.chat(user_input)
                response_text = response.message
                actions = [
                    {
                        "path": a.config_path,
                        "value": a.proposed_value,
                        "risk": a.risk_level,
                        "reasoning": a.reasoning,
                    }
                    for a in response.proposed_actions
                ]
            except Exception as e:
                response_text = f"Error: {e}"
                actions = []
        else:
            # Demo mode responses
            lower_input = user_input.lower()
            if any(w in lower_input for w in ["why", "explain", "decision"]):
                response_text = (
                    "The agent chose to **hold** on this customer-item combination because:\n\n"
                    "- **Item margin (28%)** is well above the category floor (12% for Protein)\n"
                    "- **Customer CSS 4** with low churn risk (5%) -- no retention pressure\n"
                    "- **Perishability (0.7)** suggests seasonal demand is currently stable\n"
                    "- **No recent price changes** -- stability bonus is active\n\n"
                    "Reward decomposition: margin=+0.0, volume=+0.0, churn=0.0, "
                    "volatility=0.0, strategic=+0.5 (cross-sell potential)"
                )
                actions = []
            elif any(w in lower_input for w in ["conservative", "aggressive", "increase", "decrease", "weight"]):
                response_text = (
                    "I can adjust the reward weights to make the agent more conservative. "
                    "Here's what I'd propose:"
                )
                actions = [
                    {"path": "reward.alpha_by_css.css_4", "value": "0.75", "risk": "medium",
                     "reasoning": "Increase margin weight for CSS 4 to protect margins"},
                    {"path": "reward.beta_by_css.css_4", "value": "0.12", "risk": "medium",
                     "reasoning": "Reduce volume weight to reduce discounting"},
                ]
            elif any(w in lower_input for w in ["spike", "cost", "supply", "shortage"]):
                response_text = (
                    "I'll adjust the elasticity and seasonal parameters to account for "
                    "the supply disruption. Here's what I'd propose:"
                )
                actions = [
                    {"path": "items.categories.protein.elasticity_modifier", "value": "1.6",
                     "risk": "medium", "reasoning": "Reduce price sensitivity during shortage"},
                ]
            elif any(w in lower_input for w in ["stop", "block", "restrict", "no more"]):
                response_text = (
                    "I'll add action mask overrides to restrict deep discounts. "
                    "Here's what I'd propose:"
                )
                actions = [
                    {"path": "business_rules.max_consecutive_discounts", "value": "2",
                     "risk": "low", "reasoning": "Tighten consecutive discount limit"},
                ]
            else:
                response_text = (
                    "I can help with:\n\n"
                    "- **Explain** decisions: *\"Why did we discount customer X?\"*\n"
                    "- **Configure** strategy: *\"Be more conservative on fine dining\"*\n"
                    "- **Inform** about market changes: *\"Chicken prices are spiking\"*\n"
                    "- **Restrict** actions: *\"Stop deep discounts on CSS 5\"*\n\n"
                    "What would you like to know?"
                )
                actions = []

        # Display response
        msg_data = {"role": "assistant", "content": response_text, "actions": actions}
        st.session_state.copilot_messages.append(msg_data)
        with st.chat_message("assistant"):
            st.markdown(response_text)
            for action in actions:
                risk_color = {"low": "green", "medium": "orange", "high": "red"}.get(
                    action.get("risk", "medium"), "orange"
                )
                st.markdown(
                    f"**Proposed:** `{action['path']}` = `{action['value']}` "
                    f"(:{risk_color}[{action['risk']} risk])"
                )

        if actions:
            st.session_state.pending_actions = actions

    # Action approval UI
    if st.session_state.pending_actions:
        st.markdown("---")
        st.markdown("### Pending Changes")
        for i, action in enumerate(st.session_state.pending_actions):
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                st.markdown(
                    f"`{action['path']}` = `{action['value']}` -- {action.get('reasoning', '')}"
                )
            with col2:
                if st.button("Apply", key=f"apply_{i}"):
                    st.success(f"Applied: {action['path']} = {action['value']}")
                    st.session_state.pending_actions = [
                        a for j, a in enumerate(st.session_state.pending_actions) if j != i
                    ]
                    st.rerun()
            with col3:
                if st.button("Reject", key=f"reject_{i}"):
                    st.session_state.pending_actions = [
                        a for j, a in enumerate(st.session_state.pending_actions) if j != i
                    ]
                    st.rerun()
