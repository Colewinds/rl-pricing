"""Inference serve script for trained model predictions.

Usage:
    python scripts/serve.py --model results/models/run_id/final_model --algorithm ppo
    python scripts/serve.py --model results/models/run_id/final_model --algorithm ppo \\
        --input '{"css_score": 3, "margin_rate": 0.24, "weekly_cases": 15}'
"""

import argparse
import json

import numpy as np

from src.agent.rl_agent import RLAgent
from src.environment.customer import CustomerState
from src.environment.pricing_env import DynamicPricingEnv
from src.reward.reward_functions import CLVOptimizer

ACTION_NAMES = {
    0: "hold",
    1: "price_up_2pct",
    2: "price_up_5pct",
    3: "price_down_2pct",
    4: "price_down_5pct",
    5: "price_down_10pct",
    6: "price_down_15pct",
}


def build_customer_state(data: dict) -> CustomerState:
    """Build a CustomerState from a partial input dict with sensible defaults."""
    return CustomerState(
        css_score=data.get("css_score", 3),
        performance_percentile=data.get("performance_percentile", 0.5),
        potential_tier=data.get("potential_tier", 1),
        current_margin_rate=data.get("margin_rate", 0.24),
        current_margin_dollars=data.get("margin_dollars", 665.0),
        weekly_cases=data.get("weekly_cases", 15.0),
        weekly_sales=data.get("weekly_sales", 750.0),
        deliveries_per_week=data.get("deliveries_per_week", 2.5),
        elasticity_estimate=data.get("elasticity", -1.5),
        price_change_history=data.get("price_change_history", [0, 0, 0, 0]),
        periods_since_last_change=data.get("periods_since_last_change", 4),
        syw_flag=data.get("syw_flag", False),
        perks_flag=data.get("perks_flag", False),
        churn_probability=data.get("churn_probability", 0.10),
    )


def predict_action(agent: RLAgent, customer_state: CustomerState) -> dict:
    """Get a prediction with explanation for a customer state.

    Returns:
        Dict with action, action_name, confidence info, and reward explanation.
    """
    obs = customer_state.to_observation()
    action = agent.predict(obs)
    action_name = ACTION_NAMES.get(action, f"unknown_{action}")

    # Get reward explanation
    reward_fn = CLVOptimizer()
    # Simulate what happens with this action (approximate next state)
    next_state = CustomerState(
        css_score=customer_state.css_score,
        performance_percentile=customer_state.performance_percentile,
        potential_tier=customer_state.potential_tier,
        current_margin_rate=customer_state.current_margin_rate,
        current_margin_dollars=customer_state.current_margin_dollars,
        weekly_cases=customer_state.weekly_cases,
        weekly_sales=customer_state.weekly_sales,
        deliveries_per_week=customer_state.deliveries_per_week,
        elasticity_estimate=customer_state.elasticity_estimate,
        price_change_history=[action] + customer_state.price_change_history[:3],
        periods_since_last_change=(
            customer_state.periods_since_last_change + 1 if action == 0 else 0
        ),
        syw_flag=customer_state.syw_flag,
        perks_flag=customer_state.perks_flag,
        churn_probability=customer_state.churn_probability,
    )

    explanation = reward_fn.explain(customer_state, action, next_state)

    return {
        "action": action,
        "action_name": action_name,
        "customer_css": customer_state.css_score,
        "current_margin_rate": customer_state.current_margin_rate,
        "explanation": explanation,
    }


def main():
    parser = argparse.ArgumentParser(description="Serve trained model predictions")
    parser.add_argument("--model", type=str, help="Path to saved model")
    parser.add_argument("--algorithm", type=str, default="ppo", choices=["ppo", "dqn"])
    parser.add_argument("--input", type=str, default=None, help="JSON customer state")
    args = parser.parse_args()

    # Load model
    if args.model:
        env = DynamicPricingEnv()
        agent = RLAgent(algorithm=args.algorithm, model_path=args.model, env=env)
    else:
        # Demo mode with untrained model
        env = DynamicPricingEnv()
        agent = RLAgent(algorithm=args.algorithm, env=env)
        print("WARNING: No model path provided. Using untrained model for demo.\n")

    if args.input:
        data = json.loads(args.input)
        customer = build_customer_state(data)
        result = predict_action(agent, customer)
        print(json.dumps(result, indent=2))
    else:
        # Interactive demo with a sample customer
        sample = {
            "css_score": 3,
            "margin_rate": 0.24,
            "weekly_cases": 15.0,
            "weekly_sales": 750.0,
        }
        print(f"Sample customer: {json.dumps(sample)}")
        customer = build_customer_state(sample)
        result = predict_action(agent, customer)
        print(f"\nPrediction:")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
