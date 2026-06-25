"""
Mock Agent Base Class - Base class for testbed mock agents.

This module provides the MockAgent class that extends BaseTarget
to create configurable mock agents for guardrail testing.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dspy_guardrails.testbed.config import AgentComplexity, AgentDomain
from dspy_guardrails.testing import BaseTarget, ConversationTurn, TargetResponse


@dataclass
class GuardrailResult:
    """Result from a guardrail check.

    Attributes:
        passed: Whether the guardrail check passed.
        guardrail_name: Name of the guardrail that was checked.
        reason: Optional reason for the result (especially if failed).
        score: Risk/confidence score (0.0-1.0, higher = more risk).
    """

    passed: bool
    guardrail_name: str
    reason: str | None = None
    score: float = 0.0


class MockAgent(BaseTarget):
    """
    Mock Agent for testbed evaluation.

    Extends BaseTarget to create configurable mock agents with
    input/output guardrails for testing guardrail effectiveness.

    Attributes:
        complexity: Agent complexity level.
        domain: Agent domain category.
        system_prompt: System prompt for the agent.
        llm_fn: Optional LLM function (system, user) -> response.
        input_guardrails: List of input guardrail functions.
        output_guardrails: List of output guardrail functions.
        name: Agent name (auto-generated if not provided).

    Examples:
        # Basic mock agent
        agent = MockAgent(
            complexity=AgentComplexity.SIMPLE,
            domain=AgentDomain.CUSTOMER_SERVICE,
            system_prompt="You are a helpful assistant.",
        )

        # With custom LLM function
        agent = MockAgent(
            complexity=AgentComplexity.TOOLS,
            domain=AgentDomain.CODE_ASSISTANT,
            system_prompt="You are a code assistant.",
            llm_fn=lambda sys, usr: f"Response to: {usr}",
        )

        # With guardrails
        def no_injection(text: str) -> GuardrailResult:
            is_safe = "ignore" not in text.lower()
            return GuardrailResult(
                passed=is_safe,
                guardrail_name="no_injection",
                reason=None if is_safe else "Injection detected",
            )

        agent = MockAgent(
            complexity=AgentComplexity.SIMPLE,
            domain=AgentDomain.GENERAL,
            system_prompt="You are a helpful assistant.",
            input_guardrails=[no_injection],
        )
    """

    def __init__(
        self,
        complexity: AgentComplexity,
        domain: AgentDomain,
        system_prompt: str,
        llm_fn: Callable[[str, str], str] | None = None,
        input_guardrails: list[Callable[[str], GuardrailResult]] | None = None,
        output_guardrails: list[Callable[[str], GuardrailResult]] | None = None,
        name: str | None = None,
    ):
        """
        Initialize MockAgent.

        Args:
            complexity: Agent complexity level.
            domain: Agent domain category.
            system_prompt: System prompt for the agent.
            llm_fn: Optional function (system_prompt, user_message) -> response.
                    If not provided, returns a default response.
            input_guardrails: List of guardrail functions to check input.
                              Each function takes a string and returns GuardrailResult.
            output_guardrails: List of guardrail functions to check output.
                               Each function takes a string and returns GuardrailResult.
            name: Optional agent name. Auto-generates as "{complexity.value}-{domain.value}"
                  if not provided.
        """
        super().__init__()
        self.complexity = complexity
        self.domain = domain
        self.system_prompt = system_prompt
        self.llm_fn = llm_fn
        self.input_guardrails = input_guardrails or []
        self.output_guardrails = output_guardrails or []
        self.name = name or f"{complexity.value}-{domain.value}"
        self._call_count = 0

    def invoke(self, prompt: str) -> TargetResponse:
        """
        Invoke the mock agent with a prompt.

        Process flow:
        1. Run input guardrails - if any fails, return blocked response
        2. Call llm_fn or generate default response
        3. Run output guardrails - if any fails, return blocked response
        4. Record conversation in history
        5. Return TargetResponse with response, guardrail_status, latency_ms, metadata

        Args:
            prompt: User message to process.

        Returns:
            TargetResponse: Response with guardrail status and metadata.
        """
        start_time = time.time()
        self._call_count += 1

        guardrail_status: dict[str, dict[str, Any]] = {}

        # Step 1: Run input guardrails
        for guardrail_fn in self.input_guardrails:
            result = guardrail_fn(prompt)
            guardrail_status[result.guardrail_name] = {
                "passed": result.passed,
                "reason": result.reason,
                "score": result.score,
                "type": "input",
            }
            if not result.passed:
                # Input blocked - record and return
                latency_ms = (time.time() - start_time) * 1000
                response = "I cannot help with that request."

                self._record_conversation(prompt, response, guardrail_status)

                return TargetResponse(
                    response=response,
                    guardrail_status=guardrail_status,
                    latency_ms=latency_ms,
                    metadata={
                        "agent_name": self.name,
                        "blocked_by": result.guardrail_name,
                        "blocked_at": "input",
                    },
                )

        # Step 2: Generate response
        if self.llm_fn is not None:
            response = self.llm_fn(self.system_prompt, prompt)
        else:
            response = f"Mock response from {self.name}: Processed '{prompt[:50]}...'"

        # Step 3: Run output guardrails
        for guardrail_fn in self.output_guardrails:
            result = guardrail_fn(response)
            guardrail_status[result.guardrail_name] = {
                "passed": result.passed,
                "reason": result.reason,
                "score": result.score,
                "type": "output",
            }
            if not result.passed:
                # Output blocked - record and return
                latency_ms = (time.time() - start_time) * 1000
                blocked_response = "I cannot provide that response."

                self._record_conversation(prompt, blocked_response, guardrail_status)

                return TargetResponse(
                    response=blocked_response,
                    guardrail_status=guardrail_status,
                    latency_ms=latency_ms,
                    metadata={
                        "agent_name": self.name,
                        "blocked_by": result.guardrail_name,
                        "blocked_at": "output",
                        "original_response": response,
                    },
                )

        # Step 4: Record conversation
        latency_ms = (time.time() - start_time) * 1000
        self._record_conversation(prompt, response, guardrail_status)

        # Step 5: Return successful response
        return TargetResponse(
            response=response,
            guardrail_status=guardrail_status,
            latency_ms=latency_ms,
            metadata={
                "agent_name": self.name,
            },
        )

    def _record_conversation(
        self,
        prompt: str,
        response: str,
        guardrail_status: dict[str, dict[str, Any]],
    ) -> None:
        """Record conversation turn in history."""
        self.conversation_history.append(
            ConversationTurn(role="user", content=prompt)
        )
        self.conversation_history.append(
            ConversationTurn(
                role="assistant",
                content=response,
                guardrail_status=guardrail_status,
            )
        )

    def reset_session(self) -> None:
        """Reset the agent session.

        Clears conversation history and resets call count.
        """
        self.clear_history()
        self._call_count = 0

    def get_stats(self) -> dict[str, Any]:
        """Get agent statistics.

        Returns:
            Dict containing:
                - name: Agent name
                - complexity: Complexity level value
                - domain: Domain value
                - call_count: Number of invoke calls
                - conversation_turns: Number of conversation turns
        """
        return {
            "name": self.name,
            "complexity": self.complexity.value,
            "domain": self.domain.value,
            "call_count": self._call_count,
            "conversation_turns": len(self.conversation_history),
        }
