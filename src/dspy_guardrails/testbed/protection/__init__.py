"""
Protection module for testbed guardrails.

This module provides guardrail wrappers for different protection levels,
enabling easy integration of dspy_guardrails detection into the testbed.
"""

from .wrapper import GuardrailWrapper, create_guardrails_for_level

__all__ = [
    "GuardrailWrapper",
    "create_guardrails_for_level",
]
