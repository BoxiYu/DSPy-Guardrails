"""
Compatibility module for integrating with external evaluation frameworks.

This module provides adapters for converting configurations from external
evaluation frameworks (like Amazon Agent Evaluation) to dspyGuardrails
testbed configurations.
"""

from dspy_guardrails.testbed.compat.agent_eval import (
    AgentEvalAdapter,
    AgentEvalConfig,
    AgentEvalTest,
)

__all__ = [
    "AgentEvalAdapter",
    "AgentEvalConfig",
    "AgentEvalTest",
]
