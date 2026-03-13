"""Intent parser using Claude tool_use for structured config change extraction."""

from dataclasses import dataclass


@dataclass
class ProposedAction:
    """A proposed configuration change from the copilot."""
    action_type: str  # "configure", "restrict", "inform"
    config_path: str  # e.g., "reward.clv_optimizer.alpha_by_css.css_5"
    current_value: str
    proposed_value: str
    reasoning: str
    risk_level: str  # "low", "medium", "high"


@dataclass
class CopilotResponse:
    """Structured response from the pricing copilot."""
    message: str
    intent: str  # "explain", "configure", "inform", "restrict"
    proposed_actions: list[ProposedAction]
    requires_approval: bool
    confidence: float  # 0-1


# Config validation ranges
CONFIG_RANGES = {
    "reward.clv_optimizer.alpha_by_css": (0.0, 2.0),
    "reward.clv_optimizer.beta_by_css": (0.0, 2.0),
    "reward.clv_optimizer.gamma": (0.0, 50.0),
    "reward.clv_optimizer.delta": (0.0, 20.0),
    "reward.clv_optimizer.epsilon": (0.0, 20.0),
    "reward.clv_optimizer.zeta": (0.0, 50.0),
    "reward.clv_optimizer.eta": (0.0, 10.0),
    "reward.clv_optimizer.theta": (0.0, 20.0),
    "reward.clv_optimizer.alpha_concept_modifier": (0.3, 3.0),
    "reward.clv_optimizer.beta_concept_modifier": (0.3, 3.0),
    "business_rules.category_margin_floors": (0.0, 0.50),
    "business_rules.customer_margin_floor_by_css": (0.0, 0.50),
    "business_rules.max_consecutive_discounts": (1, 10),
}

# Hard blocks: never allow these changes
BLOCKED_CHANGES = [
    "monitoring.reward_drift_sigma",  # never disable monitoring
    "monitoring.action_entropy_min",
    "monitoring.alert_consecutive_periods",
]

BLOCKED_VALUES = {
    "reward.clv_optimizer.gamma": 0.0,  # never remove churn penalty
}


def validate_config_change(config_path: str, proposed_value: float) -> tuple[bool, str]:
    """Validate a proposed config change against safety rules.

    Args:
        config_path: Dot-separated config path.
        proposed_value: The proposed new value.

    Returns:
        (is_valid, reason) tuple.
    """
    # Check blocked paths
    for blocked in BLOCKED_CHANGES:
        if config_path.startswith(blocked):
            return False, f"Cannot modify monitoring config: {config_path}"

    # Check blocked values
    base_path = ".".join(config_path.split(".")[:3])  # e.g., "reward.clv_optimizer.gamma"
    if base_path in BLOCKED_VALUES:
        if proposed_value == BLOCKED_VALUES[base_path]:
            return False, f"Cannot set {config_path} to {proposed_value} (safety guardrail)"

    # Check ranges
    for range_path, (min_val, max_val) in CONFIG_RANGES.items():
        if config_path.startswith(range_path):
            if not (min_val <= proposed_value <= max_val):
                return False, f"{config_path} must be between {min_val} and {max_val}, got {proposed_value}"
            return True, "OK"

    return True, "OK (no range constraint defined)"


def classify_risk(config_path: str, current_value: float, proposed_value: float) -> str:
    """Classify the risk level of a config change.

    Args:
        config_path: The config path being changed.
        current_value: Current value.
        proposed_value: Proposed new value.

    Returns:
        Risk level: "low", "medium", or "high".
    """
    if current_value == 0:
        return "high" if proposed_value != 0 else "low"

    pct_change = abs(proposed_value - current_value) / abs(current_value)

    if pct_change < 0.10:
        return "low"
    elif pct_change < 0.30:
        return "medium"
    else:
        return "high"


def get_config_value(config: dict, path: str):
    """Get a value from nested config using dot-separated path."""
    keys = path.split(".")
    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def set_config_value(config: dict, path: str, value) -> bool:
    """Set a value in nested config using dot-separated path.

    Returns True if the value was set successfully.
    """
    keys = path.split(".")
    current = config
    for key in keys[:-1]:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return False
    if isinstance(current, dict):
        current[keys[-1]] = value
        return True
    return False
