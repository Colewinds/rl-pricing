"""Tests for the LLM pricing copilot, intent parser, and prompts."""

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.llm.intent_parser import (
    CONFIG_RANGES,
    BLOCKED_CHANGES,
    BLOCKED_VALUES,
    CopilotResponse,
    ProposedAction,
    classify_risk,
    get_config_value,
    set_config_value,
    validate_config_change,
)
from src.llm.pricing_copilot import PricingCopilot
from src.llm.prompts import FEW_SHOT_EXAMPLES, INTENT_TOOLS, build_system_prompt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config():
    """Minimal config dict that mirrors the structure of default.yaml."""
    return {
        "reward": {
            "clv_optimizer": {
                "alpha_by_css": {"css_1": 0.3, "css_2": 0.35, "css_3": 0.5, "css_4": 0.6, "css_5": 0.7},
                "beta_by_css": {"css_1": 0.5, "css_2": 0.45, "css_3": 0.3, "css_4": 0.2, "css_5": 0.15},
                "gamma": 10.0,
                "delta": 2.0,
                "epsilon": 3.0,
                "zeta": 15.0,
                "eta": 1.0,
                "theta": 5.0,
                "alpha_concept_modifier": {"qsr": 0.8, "fine_dining": 1.3},
                "beta_concept_modifier": {"qsr": 1.3, "fine_dining": 0.6},
            },
        },
        "business_rules": {
            "category_margin_floors": {"protein": 0.12, "produce": 0.08},
            "customer_margin_floor_by_css": {"css_1": 0.12, "css_5": 0.20},
            "max_consecutive_discounts": 3,
        },
        "items": {
            "categories": {"protein": {}, "produce": {}, "paper": {}},
        },
        "monitoring": {
            "reward_drift_sigma": 2.0,
            "action_entropy_min": 0.5,
            "alert_consecutive_periods": 3,
        },
        "llm": {
            "model": "claude-sonnet-4-20250514",
            "temperature": 0.3,
            "auto_approve_low_risk": False,
            "guardrails": {
                "max_weight_change_pct": 0.30,
                "always_require_approval": ["reward_weight", "action_mask"],
                "never_allow": ["disable_monitoring", "remove_churn_penalty"],
            },
        },
    }


@pytest.fixture
def copilot(sample_config):
    """PricingCopilot instance (no real API client)."""
    return PricingCopilot(sample_config)


def _make_action(config_path="reward.clv_optimizer.gamma", current="10.0",
                 proposed="12.0", reasoning="test", risk="low"):
    return ProposedAction(
        action_type="configure",
        config_path=config_path,
        current_value=current,
        proposed_value=proposed,
        reasoning=reasoning,
        risk_level=risk,
    )


# ---------------------------------------------------------------------------
# validate_config_change
# ---------------------------------------------------------------------------

class TestValidateConfigChange:
    """Tests for validate_config_change()."""

    def test_valid_in_range(self):
        ok, reason = validate_config_change("reward.clv_optimizer.gamma", 25.0)
        assert ok is True
        assert "OK" in reason

    def test_valid_alpha_css_leaf(self):
        ok, reason = validate_config_change("reward.clv_optimizer.alpha_by_css.css_5", 1.5)
        assert ok is True

    def test_out_of_range_below(self):
        ok, reason = validate_config_change("reward.clv_optimizer.gamma", -1.0)
        assert ok is False
        assert "between" in reason

    def test_out_of_range_above(self):
        ok, reason = validate_config_change("reward.clv_optimizer.gamma", 100.0)
        assert ok is False
        assert "between" in reason

    def test_out_of_range_margin_floor(self):
        ok, reason = validate_config_change("business_rules.category_margin_floors.protein", 0.75)
        assert ok is False

    def test_blocked_path_reward_drift(self):
        ok, reason = validate_config_change("monitoring.reward_drift_sigma", 5.0)
        assert ok is False
        assert "monitoring" in reason.lower()

    def test_blocked_path_entropy(self):
        ok, reason = validate_config_change("monitoring.action_entropy_min", 0.1)
        assert ok is False

    def test_blocked_path_alert_periods(self):
        ok, reason = validate_config_change("monitoring.alert_consecutive_periods", 10)
        assert ok is False

    def test_blocked_value_gamma_zero(self):
        ok, reason = validate_config_change("reward.clv_optimizer.gamma", 0.0)
        assert ok is False
        assert "safety guardrail" in reason.lower() or "cannot set" in reason.lower()

    def test_gamma_nonzero_is_fine(self):
        ok, _ = validate_config_change("reward.clv_optimizer.gamma", 0.5)
        assert ok is True

    def test_unknown_path_allowed(self):
        """Paths with no range constraint pass with a note."""
        ok, reason = validate_config_change("some.unknown.path", 42.0)
        assert ok is True
        assert "no range constraint" in reason.lower()

    def test_boundary_values(self):
        """Exact boundary values should be accepted."""
        ok, _ = validate_config_change("reward.clv_optimizer.gamma", 0.0)
        # 0.0 is blocked by BLOCKED_VALUES, skip that case; test upper bound
        ok_upper, _ = validate_config_change("reward.clv_optimizer.gamma", 50.0)
        assert ok_upper is True

    def test_max_consecutive_discounts_in_range(self):
        ok, _ = validate_config_change("business_rules.max_consecutive_discounts", 5)
        assert ok is True

    def test_max_consecutive_discounts_out_of_range(self):
        ok, _ = validate_config_change("business_rules.max_consecutive_discounts", 0)
        assert ok is False


# ---------------------------------------------------------------------------
# classify_risk
# ---------------------------------------------------------------------------

class TestClassifyRisk:
    """Tests for classify_risk()."""

    def test_low_risk_small_change(self):
        # 5% change
        assert classify_risk("reward.clv_optimizer.gamma", 10.0, 10.5) == "low"

    def test_medium_risk_moderate_change(self):
        # 20% change
        assert classify_risk("reward.clv_optimizer.gamma", 10.0, 12.0) == "medium"

    def test_high_risk_large_change(self):
        # 50% change
        assert classify_risk("reward.clv_optimizer.gamma", 10.0, 15.0) == "high"

    def test_boundary_10pct_is_medium(self):
        # Exactly 10% -> should be medium (>= 0.10)
        assert classify_risk("x", 100.0, 110.0) == "medium"

    def test_boundary_30pct_is_high(self):
        # Exactly 30% -> should be high (>= 0.30)
        assert classify_risk("x", 100.0, 130.0) == "high"

    def test_just_under_10pct_is_low(self):
        assert classify_risk("x", 100.0, 109.0) == "low"

    def test_just_under_30pct_is_medium(self):
        assert classify_risk("x", 100.0, 129.0) == "medium"

    def test_zero_current_nonzero_proposed_is_high(self):
        assert classify_risk("x", 0.0, 5.0) == "high"

    def test_zero_to_zero_is_low(self):
        assert classify_risk("x", 0.0, 0.0) == "low"

    def test_negative_direction_same_magnitude(self):
        # 20% decrease
        assert classify_risk("x", 10.0, 8.0) == "medium"


# ---------------------------------------------------------------------------
# get_config_value / set_config_value
# ---------------------------------------------------------------------------

class TestConfigAccess:
    """Tests for get_config_value() and set_config_value()."""

    def test_get_top_level(self, sample_config):
        result = get_config_value(sample_config, "reward")
        assert isinstance(result, dict)
        assert "clv_optimizer" in result

    def test_get_nested_scalar(self, sample_config):
        assert get_config_value(sample_config, "reward.clv_optimizer.gamma") == 10.0

    def test_get_deeply_nested(self, sample_config):
        assert get_config_value(sample_config, "reward.clv_optimizer.alpha_by_css.css_5") == 0.7

    def test_get_missing_key_returns_none(self, sample_config):
        assert get_config_value(sample_config, "reward.nonexistent.key") is None

    def test_get_empty_path_segment(self, sample_config):
        # Entirely unknown top-level
        assert get_config_value(sample_config, "nope") is None

    def test_set_nested_scalar(self, sample_config):
        cfg = copy.deepcopy(sample_config)
        ok = set_config_value(cfg, "reward.clv_optimizer.gamma", 20.0)
        assert ok is True
        assert cfg["reward"]["clv_optimizer"]["gamma"] == 20.0

    def test_set_deeply_nested(self, sample_config):
        cfg = copy.deepcopy(sample_config)
        ok = set_config_value(cfg, "reward.clv_optimizer.alpha_by_css.css_5", 1.0)
        assert ok is True
        assert cfg["reward"]["clv_optimizer"]["alpha_by_css"]["css_5"] == 1.0

    def test_set_creates_new_leaf_key(self, sample_config):
        cfg = copy.deepcopy(sample_config)
        ok = set_config_value(cfg, "reward.clv_optimizer.new_key", 99)
        assert ok is True
        assert cfg["reward"]["clv_optimizer"]["new_key"] == 99

    def test_set_missing_intermediate_returns_false(self, sample_config):
        cfg = copy.deepcopy(sample_config)
        ok = set_config_value(cfg, "reward.nonexistent.key", 1)
        assert ok is False

    def test_set_on_non_dict_returns_false(self, sample_config):
        cfg = copy.deepcopy(sample_config)
        # gamma is a float, can't traverse into it
        ok = set_config_value(cfg, "reward.clv_optimizer.gamma.sub", 1)
        assert ok is False


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------

class TestBuildSystemPrompt:
    """Tests for build_system_prompt()."""

    def test_returns_nonempty_string(self, sample_config):
        prompt = build_system_prompt(sample_config)
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_contains_config_values(self, sample_config):
        prompt = build_system_prompt(sample_config)
        # Should mention gamma value
        assert "10.0" in prompt
        # Should mention category names from items
        assert "protein" in prompt

    def test_includes_metrics_section(self, sample_config):
        metrics = {"mean_reward": 42.5, "action_entropy": 1.2}
        prompt = build_system_prompt(sample_config, model_metrics=metrics)
        assert "42.5" in prompt
        assert "1.2" in prompt

    def test_includes_drift_alerts(self, sample_config):
        alerts = {"reward_drift": True, "action_entropy": True, "any_alert": True}
        prompt = build_system_prompt(sample_config, drift_alerts=alerts)
        assert "reward_drift" in prompt
        assert "action_entropy" in prompt

    def test_no_metrics_no_crash(self, sample_config):
        prompt = build_system_prompt(sample_config, model_metrics=None, drift_alerts=None)
        assert isinstance(prompt, str)


# ---------------------------------------------------------------------------
# INTENT_TOOLS / FEW_SHOT_EXAMPLES sanity checks
# ---------------------------------------------------------------------------

class TestPromptConstants:
    """Sanity checks on INTENT_TOOLS and FEW_SHOT_EXAMPLES."""

    def test_intent_tools_is_list(self):
        assert isinstance(INTENT_TOOLS, list)
        assert len(INTENT_TOOLS) >= 1

    def test_intent_tool_has_name_and_schema(self):
        tool = INTENT_TOOLS[0]
        assert tool["name"] == "classify_intent"
        assert "input_schema" in tool

    def test_few_shot_examples_alternating_roles(self):
        assert len(FEW_SHOT_EXAMPLES) >= 2
        for i in range(0, len(FEW_SHOT_EXAMPLES), 2):
            assert FEW_SHOT_EXAMPLES[i]["role"] == "user"
            assert FEW_SHOT_EXAMPLES[i + 1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# PricingCopilot.apply_action
# ---------------------------------------------------------------------------

class TestApplyAction:
    """Tests for PricingCopilot.apply_action()."""

    def test_apply_modifies_config(self, copilot):
        action = _make_action(
            config_path="reward.clv_optimizer.gamma",
            current="10.0",
            proposed="12.0",
        )
        ok = copilot.apply_action(action)
        assert ok is True
        assert copilot.config["reward"]["clv_optimizer"]["gamma"] == 12.0

    def test_apply_logs_override(self, copilot):
        action = _make_action()
        copilot.apply_action(action)
        log = copilot.get_override_log()
        assert len(log) == 1
        entry = log[0]
        assert entry["config_path"] == "reward.clv_optimizer.gamma"
        assert entry["old_value"] == "10.0"
        assert entry["new_value"] == "12.0"
        assert "timestamp" in entry
        assert "risk_level" in entry

    def test_apply_bad_path_returns_false(self, copilot):
        action = _make_action(config_path="totally.bogus.path")
        ok = copilot.apply_action(action)
        assert ok is False
        assert len(copilot.get_override_log()) == 0

    def test_apply_non_numeric_value(self, copilot):
        action = _make_action(
            config_path="reward.clv_optimizer.alpha_concept_modifier.qsr",
            current="0.8",
            proposed="aggressive",
        )
        ok = copilot.apply_action(action)
        assert ok is True
        assert copilot.config["reward"]["clv_optimizer"]["alpha_concept_modifier"]["qsr"] == "aggressive"

    def test_apply_multiple_actions_accumulates_log(self, copilot):
        a1 = _make_action(config_path="reward.clv_optimizer.gamma", proposed="12.0")
        a2 = _make_action(config_path="reward.clv_optimizer.delta", proposed="3.0")
        copilot.apply_action(a1)
        copilot.apply_action(a2)
        assert len(copilot.get_override_log()) == 2

    def test_get_override_log_returns_copy(self, copilot):
        copilot.apply_action(_make_action())
        log = copilot.get_override_log()
        log.clear()
        assert len(copilot.get_override_log()) == 1


# ---------------------------------------------------------------------------
# PricingCopilot.chat  (mocked Anthropic client)
# ---------------------------------------------------------------------------

def _mock_anthropic_response(text="Here is my analysis.", tool_input=None):
    """Build a mock response object matching the Anthropic API structure."""
    blocks = []

    text_block = SimpleNamespace(type="text", text=text)
    blocks.append(text_block)

    if tool_input is not None:
        tool_block = SimpleNamespace(
            type="tool_use",
            name="classify_intent",
            input=tool_input,
        )
        blocks.append(tool_block)

    return SimpleNamespace(content=blocks)


class TestChatMocked:
    """Tests for PricingCopilot.chat() with a mocked Anthropic client."""

    def test_chat_explain_no_actions(self, copilot):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response(
            text="The agent discounted because volume was declining.",
            tool_input={"intent": "explain", "target": "css_3 protein"},
        )
        copilot._client = mock_client

        resp = copilot.chat("Why did we discount CSS 3 on protein?")

        assert isinstance(resp, CopilotResponse)
        assert resp.intent == "explain"
        assert resp.message == "The agent discounted because volume was declining."
        assert resp.proposed_actions == []
        assert resp.requires_approval is False

    def test_chat_configure_with_valid_change(self, copilot):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response(
            text="I'll increase gamma to 12.",
            tool_input={
                "intent": "configure",
                "target": "gamma",
                "proposed_changes": [
                    {
                        "config_path": "reward.clv_optimizer.gamma",
                        "proposed_value": "12.0",
                        "reasoning": "Strengthen churn penalty",
                    }
                ],
            },
        )
        copilot._client = mock_client

        resp = copilot.chat("Increase churn penalty")

        assert resp.intent == "configure"
        assert len(resp.proposed_actions) == 1
        action = resp.proposed_actions[0]
        assert action.config_path == "reward.clv_optimizer.gamma"
        assert action.proposed_value == "12.0"
        assert action.current_value == "10.0"
        assert action.risk_level == "medium"  # 20% change
        assert resp.requires_approval is True

    def test_chat_rejected_change_not_in_actions(self, copilot):
        """A proposed change that fails validation should be silently dropped."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response(
            text="Setting gamma to 0.",
            tool_input={
                "intent": "configure",
                "target": "gamma",
                "proposed_changes": [
                    {
                        "config_path": "reward.clv_optimizer.gamma",
                        "proposed_value": "0.0",
                        "reasoning": "Remove churn penalty",
                    }
                ],
            },
        )
        copilot._client = mock_client

        resp = copilot.chat("Remove churn penalty")

        # gamma=0 is blocked, so proposed_actions should be empty
        assert len(resp.proposed_actions) == 0
        assert resp.requires_approval is False

    def test_chat_out_of_range_change_dropped(self, copilot):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response(
            text="Setting gamma to 100.",
            tool_input={
                "intent": "configure",
                "target": "gamma",
                "proposed_changes": [
                    {
                        "config_path": "reward.clv_optimizer.gamma",
                        "proposed_value": "100.0",
                        "reasoning": "Extreme value",
                    }
                ],
            },
        )
        copilot._client = mock_client

        resp = copilot.chat("Set gamma to 100")
        assert len(resp.proposed_actions) == 0

    def test_chat_auto_approve_low_risk(self, sample_config):
        """When auto_approve_low_risk is True and all actions are low risk, no approval needed."""
        sample_config["llm"]["auto_approve_low_risk"] = True
        # Remove guardrails that would force approval
        sample_config["llm"]["guardrails"]["always_require_approval"] = []
        cop = PricingCopilot(sample_config)

        mock_client = MagicMock()
        # 5% change => low risk
        mock_client.messages.create.return_value = _mock_anthropic_response(
            text="Small tweak.",
            tool_input={
                "intent": "configure",
                "target": "gamma",
                "proposed_changes": [
                    {
                        "config_path": "reward.clv_optimizer.gamma",
                        "proposed_value": "10.5",
                        "reasoning": "Minor adjustment",
                    }
                ],
            },
        )
        cop._client = mock_client

        resp = cop.chat("Tiny gamma bump")
        assert len(resp.proposed_actions) == 1
        assert resp.proposed_actions[0].risk_level == "low"
        assert resp.requires_approval is False

    def test_chat_api_error_returns_error_response(self, copilot):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API connection failed")
        copilot._client = mock_client

        resp = copilot.chat("Hello")

        assert resp.intent == "error"
        assert resp.confidence == 0.0
        assert "Error communicating" in resp.message
        assert resp.proposed_actions == []
        assert resp.requires_approval is False

    def test_chat_updates_history(self, copilot):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response(text="Reply 1.")
        copilot._client = mock_client

        copilot.chat("Hello")
        assert len(copilot._chat_history) == 2  # user + assistant
        assert copilot._chat_history[0]["role"] == "user"
        assert copilot._chat_history[1]["role"] == "assistant"

    def test_chat_history_truncated(self, copilot):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response(text="ok")
        copilot._client = mock_client

        for i in range(15):
            copilot.chat(f"msg {i}")

        # 15 chats = 30 entries, but capped at 20
        assert len(copilot._chat_history) <= 20

    def test_chat_passes_correct_params_to_api(self, copilot):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response(text="ok")
        copilot._client = mock_client

        copilot.chat("test message")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs["max_tokens"] == 2000
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["tools"] == INTENT_TOOLS
        assert isinstance(call_kwargs["system"], str)
        # messages should include few-shot examples + user message
        assert any(m["content"] == "test message" for m in call_kwargs["messages"])


# ---------------------------------------------------------------------------
# Guardrails: always_require_approval
# ---------------------------------------------------------------------------

class TestGuardrails:
    """Tests that always_require_approval paths force requires_approval=True."""

    def test_reward_weight_keyword_forces_approval(self, sample_config):
        sample_config["llm"]["auto_approve_low_risk"] = True
        sample_config["llm"]["guardrails"]["always_require_approval"] = ["reward_weight"]
        cop = PricingCopilot(sample_config)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response(
            text="Adjusting reward_weight param.",
            tool_input={
                "intent": "configure",
                "target": "reward_weight.gamma",
                "proposed_changes": [
                    {
                        "config_path": "reward.clv_optimizer.reward_weight.gamma",
                        "proposed_value": "10.5",
                        "reasoning": "small",
                    }
                ],
            },
        )
        cop._client = mock_client

        resp = cop.chat("tweak reward_weight")
        # Even though low risk & auto_approve, the guardrail keyword matches
        assert resp.requires_approval is True

    def test_action_mask_keyword_forces_approval(self, sample_config):
        sample_config["llm"]["auto_approve_low_risk"] = True
        cop = PricingCopilot(sample_config)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response(
            text="Adding action_mask.",
            tool_input={
                "intent": "restrict",
                "target": "action_mask",
                "proposed_changes": [
                    {
                        "config_path": "environment.action_mask.protein",
                        "proposed_value": "true",
                        "reasoning": "restrict deep discounts",
                    }
                ],
            },
        )
        cop._client = mock_client

        resp = cop.chat("restrict protein discounts")
        assert resp.requires_approval is True

    def test_no_guardrail_match_allows_auto_approve(self, sample_config):
        sample_config["llm"]["auto_approve_low_risk"] = True
        sample_config["llm"]["guardrails"]["always_require_approval"] = ["reward_weight"]
        cop = PricingCopilot(sample_config)

        mock_client = MagicMock()
        # Path doesn't contain "reward_weight"
        mock_client.messages.create.return_value = _mock_anthropic_response(
            text="Small margin floor change.",
            tool_input={
                "intent": "configure",
                "target": "margin floor",
                "proposed_changes": [
                    {
                        "config_path": "business_rules.category_margin_floors.protein",
                        "proposed_value": "0.13",
                        "reasoning": "tiny bump",
                    }
                ],
            },
        )
        cop._client = mock_client

        resp = cop.chat("bump protein floor slightly")
        assert len(resp.proposed_actions) == 1
        assert resp.proposed_actions[0].risk_level == "low"  # ~8% change
        assert resp.requires_approval is False


# ---------------------------------------------------------------------------
# PricingCopilot.update_context
# ---------------------------------------------------------------------------

class TestUpdateContext:
    """Tests for PricingCopilot.update_context()."""

    def test_updates_metrics(self, copilot):
        copilot.update_context(model_metrics={"mean_reward": 50.0})
        assert copilot.model_metrics["mean_reward"] == 50.0

    def test_updates_drift_alerts(self, copilot):
        copilot.update_context(drift_alerts={"reward_drift": True})
        assert copilot.drift_alerts["reward_drift"] is True

    def test_none_does_not_overwrite(self, copilot):
        copilot.update_context(model_metrics={"a": 1})
        copilot.update_context(model_metrics=None)
        assert copilot.model_metrics == {"a": 1}


# ---------------------------------------------------------------------------
# Lazy client initialization
# ---------------------------------------------------------------------------

class TestLazyClient:
    """Tests for lazy Anthropic client init."""

    def test_client_none_at_construction(self, copilot):
        assert copilot._client is None

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-123"})
    @patch("src.llm.pricing_copilot.anthropic", create=True)
    def test_get_client_creates_instance(self, mock_anthropic_module, copilot):
        mock_anthropic_module.Anthropic.return_value = MagicMock()
        # Patch the import inside _get_client
        with patch("src.llm.pricing_copilot.anthropic", mock_anthropic_module, create=True):
            # Force lazy import path
            copilot._client = None
            # Directly mock the import within the method
            import importlib
            with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
                client = copilot._get_client()
                assert client is not None
                assert copilot._client is not None
