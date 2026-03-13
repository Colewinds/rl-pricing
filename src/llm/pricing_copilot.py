"""LLM Pricing Copilot using Claude API for pricing manager interaction."""

import os
from datetime import datetime

from src.llm.intent_parser import (
    CopilotResponse,
    ProposedAction,
    validate_config_change,
    classify_risk,
    get_config_value,
    set_config_value,
)
from src.llm.prompts import build_system_prompt, INTENT_TOOLS, FEW_SHOT_EXAMPLES


class PricingCopilot:
    """LLM-powered copilot for pricing managers.

    Provides a chat interface where managers can:
    - Ask why the agent made specific decisions
    - Adjust pricing strategy (reward weights, constraints)
    - Inject market intelligence (cost spikes, demand shifts)
    - Set restrictions (action masks, margin floors)

    Requires ANTHROPIC_API_KEY environment variable.
    """

    def __init__(self, config: dict, model_metrics: dict | None = None,
                 drift_alerts: dict | None = None):
        self.config = config
        self.model_metrics = model_metrics or {}
        self.drift_alerts = drift_alerts or {}

        llm_cfg = config.get("llm", {})
        self.model = llm_cfg.get("model", "claude-sonnet-4-20250514")
        self.temperature = llm_cfg.get("temperature", 0.3)
        self.auto_approve_low_risk = llm_cfg.get("auto_approve_low_risk", False)
        self.guardrails = llm_cfg.get("guardrails", {})

        self._chat_history: list[dict] = []
        self._override_log: list[dict] = []
        self._client = None

    def _get_client(self):
        """Lazy-init Anthropic client."""
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            )
        return self._client

    def chat(self, message: str) -> CopilotResponse:
        """Process a chat message from a pricing manager.

        Args:
            message: Natural language message from the manager.

        Returns:
            CopilotResponse with message, intent, proposed actions, and approval status.
        """
        system_prompt = build_system_prompt(
            self.config, self.model_metrics, self.drift_alerts
        )

        # Build messages with few-shot examples and history
        messages = list(FEW_SHOT_EXAMPLES) + list(self._chat_history)
        messages.append({"role": "user", "content": message})

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=self.temperature,
                system=system_prompt,
                messages=messages,
                tools=INTENT_TOOLS,
            )

            # Extract response text and tool use
            response_text = ""
            proposed_actions = []
            intent = "explain"

            for block in response.content:
                if block.type == "text":
                    response_text = block.text
                elif block.type == "tool_use" and block.name == "classify_intent":
                    tool_input = block.input
                    intent = tool_input.get("intent", "explain")

                    for change in tool_input.get("proposed_changes", []):
                        config_path = change.get("config_path", "")
                        current = get_config_value(self.config, config_path)
                        proposed = change.get("proposed_value", "")

                        try:
                            proposed_float = float(proposed)
                            risk = classify_risk(config_path, float(current or 0), proposed_float)
                            valid, reason = validate_config_change(config_path, proposed_float)
                        except (ValueError, TypeError):
                            risk = "medium"
                            valid = True
                            reason = "Non-numeric value"

                        if valid:
                            proposed_actions.append(ProposedAction(
                                action_type=intent,
                                config_path=config_path,
                                current_value=str(current),
                                proposed_value=str(proposed),
                                reasoning=change.get("reasoning", ""),
                                risk_level=risk,
                            ))

            # Determine if approval is required
            requires_approval = True
            if self.auto_approve_low_risk and all(
                a.risk_level == "low" for a in proposed_actions
            ):
                requires_approval = False
            if not proposed_actions:
                requires_approval = False

            # Check guardrails
            always_approve = self.guardrails.get("always_require_approval", [])
            for action in proposed_actions:
                for keyword in always_approve:
                    if keyword in action.config_path:
                        requires_approval = True
                        break

            # Update chat history
            self._chat_history.append({"role": "user", "content": message})
            self._chat_history.append({"role": "assistant", "content": response_text})

            # Keep history manageable
            if len(self._chat_history) > 20:
                self._chat_history = self._chat_history[-20:]

            return CopilotResponse(
                message=response_text,
                intent=intent,
                proposed_actions=proposed_actions,
                requires_approval=requires_approval,
                confidence=0.85,
            )

        except Exception as e:
            return CopilotResponse(
                message=f"Error communicating with LLM: {str(e)}. "
                        "Please check your ANTHROPIC_API_KEY environment variable.",
                intent="error",
                proposed_actions=[],
                requires_approval=False,
                confidence=0.0,
            )

    def apply_action(self, action: ProposedAction) -> bool:
        """Apply a proposed config change.

        Args:
            action: The approved ProposedAction to apply.

        Returns:
            True if the change was applied successfully.
        """
        try:
            value = float(action.proposed_value)
        except ValueError:
            value = action.proposed_value

        success = set_config_value(self.config, action.config_path, value)

        if success:
            self._override_log.append({
                "timestamp": datetime.now().isoformat(),
                "config_path": action.config_path,
                "old_value": action.current_value,
                "new_value": action.proposed_value,
                "reasoning": action.reasoning,
                "risk_level": action.risk_level,
            })

        return success

    def get_override_log(self) -> list[dict]:
        """Get the audit trail of all config overrides."""
        return list(self._override_log)

    def update_context(self, model_metrics: dict | None = None,
                       drift_alerts: dict | None = None):
        """Update the copilot's context with fresh metrics."""
        if model_metrics:
            self.model_metrics = model_metrics
        if drift_alerts:
            self.drift_alerts = drift_alerts
