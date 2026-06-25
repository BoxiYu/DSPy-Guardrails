"""Base target interface for security testing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TargetResponse:
    """Response from a target invocation."""

    response: str
    guardrail_status: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

    @property
    def was_blocked(self) -> bool:
        """Check if any guardrail blocked the request."""
        for name, status in self.guardrail_status.items():
            if not status.get("passed", True):
                return True
        return False

    @property
    def blocking_guardrail(self) -> Optional[str]:
        """Get the name of the guardrail that blocked, if any."""
        for name, status in self.guardrail_status.items():
            if not status.get("passed", True):
                return name
        return None


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    role: str  # "user" or "assistant"
    content: str
    guardrail_status: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class BaseTarget(ABC):
    """Abstract base class for test targets."""

    def __init__(self):
        self.conversation_history: List[ConversationTurn] = []

    @abstractmethod
    def invoke(self, prompt: str) -> TargetResponse:
        """
        Invoke the target with a prompt.

        Args:
            prompt: The user message to send

        Returns:
            TargetResponse containing the response and guardrail status
        """
        pass

    @abstractmethod
    def reset_session(self) -> None:
        """Reset the conversation session."""
        pass

    def get_conversation_history(self) -> List[ConversationTurn]:
        """Get the conversation history."""
        return self.conversation_history.copy()

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
