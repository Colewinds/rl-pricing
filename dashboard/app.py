"""Streamlit dashboard for RL Pricing Agent.

4 tabs: Training Progress, Agent Decisions, Portfolio Health, A/B Results.

Launch: streamlit run dashboard/app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="RL Pricing Agent Dashboard", layout="wide")
st.title("RL Dynamic Pricing Agent Dashboard")

# Sidebar config
st.sidebar.header("Configuration")
results_dir = st.sidebar.text_input("Results directory", value="results/")
results_path = Path(results_dir)

tab1, tab2, tab3, tab4 = st.tabs([
    "Training Progress", "Agent Decisions", "Portfolio Health", "A/B Results"
])


# --- Tab 1: Training Progress ---
with tab1:
    st.header("Training Progress")

    tb_dir = results_path / "tensorboard"
    if tb_dir.exists():
        runs = sorted(tb_dir.iterdir()) if tb_dir.is_dir() else []
        if runs:
            selected_run = st.selectbox("Select training run", [r.name for r in runs])
            st.info(f"TensorBoard logs available at: {tb_dir / selected_run}")
            st.markdown("Launch TensorBoard: `tensorboard --logdir " + str(tb_dir) + "`")
        else:
            st.info("No training runs found. Train a model first.")
    else:
        st.info("No training results yet. Run `python scripts/train.py` first.")

    # Placeholder charts for demo
    st.subheader("Reward Over Training (Demo)")
    demo_steps = np.arange(0, 10000, 100)
    demo_rewards = np.cumsum(np.random.randn(len(demo_steps)) * 0.5 + 0.1)
    fig = px.line(x=demo_steps, y=demo_rewards, labels={"x": "Timesteps", "y": "Cumulative Reward"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Action Distribution Over Training (Demo)")
    action_names = ["Hold", "+2%", "+5%", "-2%", "-5%", "-10%", "-15%"]
    demo_dist = pd.DataFrame({
        "Period": list(range(52)) * 7,
        "Action": [a for a in action_names for _ in range(52)],
        "Count": np.random.randint(1, 20, 52 * 7),
    })
    fig2 = px.area(demo_dist, x="Period", y="Count", color="Action",
                   title="Action Distribution Over Time")
    st.plotly_chart(fig2, use_container_width=True)


# --- Tab 2: Agent Decisions ---
with tab2:
    st.header("Agent Decisions")

    st.subheader("Customer State Inspector")
    col1, col2 = st.columns(2)
    with col1:
        css_score = st.slider("CSS Score", 1, 5, 3)
        margin_rate = st.slider("Margin Rate", 0.05, 0.55, 0.24, 0.01)
        weekly_cases = st.slider("Weekly Cases", 1.0, 100.0, 15.0)
    with col2:
        elasticity = st.slider("Elasticity", -4.0, -0.1, -1.5, 0.1)
        syw = st.checkbox("SYW Member")
        periods_stable = st.slider("Periods Since Last Change", 0, 52, 4)

    st.json({
        "css_score": css_score,
        "margin_rate": margin_rate,
        "weekly_cases": weekly_cases,
        "elasticity": elasticity,
        "syw_flag": syw,
        "periods_since_last_change": periods_stable,
    })

    st.subheader("Pricing History Timeline (Demo)")
    demo_periods = list(range(52))
    demo_prices = [0.24]
    actions_taken = []
    for i in range(51):
        change = np.random.choice([0, 0.02, -0.02, -0.05], p=[0.5, 0.2, 0.2, 0.1])
        demo_prices.append(np.clip(demo_prices[-1] + change, 0.05, 0.55))
        actions_taken.append(change)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=demo_periods, y=demo_prices, mode="lines+markers",
                               name="Margin Rate"))
    fig3.update_layout(title="Margin Rate Over Episode", xaxis_title="Period",
                       yaxis_title="Margin Rate")
    st.plotly_chart(fig3, use_container_width=True)


# --- Tab 3: Portfolio Health ---
with tab3:
    st.header("Portfolio Health")

    # Load evaluation reports if available
    report_files = list(results_path.glob("evaluation_report_*.json"))

    if report_files:
        latest = sorted(report_files)[-1]
        with open(latest) as f:
            report_data = json.load(f)
        st.success(f"Loaded report: {latest.name}")

        for agent_name, metrics in report_data.items():
            st.subheader(f"Agent: {agent_name}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Reward", f"{metrics.get('mean_reward', 0):.4f}")
            col2.metric("Action Entropy", f"{metrics.get('action_entropy', 0):.3f}")
            col3.metric("Mean Episode Margin",
                        f"${metrics.get('mean_episode_margin', 0):.2f}")

            if "churn_rate_by_css" in metrics:
                churn_data = metrics["churn_rate_by_css"]
                fig_churn = px.bar(
                    x=list(churn_data.keys()),
                    y=list(churn_data.values()),
                    labels={"x": "CSS Tier", "y": "Churn Rate"},
                    title="Churn Rate by CSS Tier",
                )
                st.plotly_chart(fig_churn, use_container_width=True)
    else:
        st.info("No evaluation reports found. Run `python scripts/evaluate.py --generate-report`.")

    # Demo CSS distribution
    st.subheader("CSS Distribution (Demo)")
    css_data = pd.DataFrame({
        "CSS": [1, 2, 3, 4, 5],
        "Count": [100, 100, 400, 250, 150],
    })
    fig4 = px.bar(css_data, x="CSS", y="Count", title="Customer Distribution by CSS Tier")
    st.plotly_chart(fig4, use_container_width=True)

    # Margin heatmap
    st.subheader("Margin by CSS Tier (Demo)")
    margin_data = pd.DataFrame({
        "CSS": [1, 2, 3, 4, 5],
        "Mean Margin %": [18, 20, 24, 28, 32],
    })
    fig5 = px.bar(margin_data, x="CSS", y="Mean Margin %",
                  title="Average Margin Rate by CSS Tier",
                  color="Mean Margin %", color_continuous_scale="RdYlGn")
    st.plotly_chart(fig5, use_container_width=True)


# --- Tab 4: A/B Results ---
with tab4:
    st.header("A/B Test Results")

    st.info("Run an A/B test: `python scripts/evaluate.py --ab-test --treatment ppo --control heuristic`")

    # Demo results
    st.subheader("Demo A/B Comparison")
    n_sims = 50
    np.random.seed(42)
    treatment = np.cumsum(np.random.normal(100, 30, n_sims))
    control = np.cumsum(np.random.normal(95, 30, n_sims))

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=list(range(n_sims)), y=treatment,
                               name="Treatment (RL)", mode="lines"))
    fig6.add_trace(go.Scatter(x=list(range(n_sims)), y=control,
                               name="Control (Heuristic)", mode="lines"))
    fig6.update_layout(title="Cumulative Margin: Treatment vs Control",
                       xaxis_title="Simulation", yaxis_title="Cumulative Margin ($)")
    st.plotly_chart(fig6, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Delta Margin", "+$5.23")
    col2.metric("p-value", "0.032")
    col3.metric("95% CI", "($1.12, $9.34)")
