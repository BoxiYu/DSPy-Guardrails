"""Unified target abstraction for security testing."""

from .dspy_module import DSPyModuleTarget
from .guardrail import GuardrailTarget
from .http import HTTPTarget
from .mcp import MCPTarget
from .protocol import (
    TargetCapability,
    TargetResponse,
    TargetType,
    UnifiedTarget,
)

__all__ = [
    "UnifiedTarget",
    "TargetType",
    "TargetCapability",
    "TargetResponse",
    "GuardrailTarget",
    "HTTPTarget",
    "MCPTarget",
    "DSPyModuleTarget",
]
