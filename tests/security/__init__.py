"""
Security Testing Framework for OpenAI CS Agents Demo

A comprehensive attack and defense testing framework using dspy-guardrails.
"""

from .runner import SecurityTestRunner
from .targets import OpenAICSAgentTarget
from .evaluators import RedTeamEvaluator, BlueTeamEvaluator, HallucinationEvaluator

__all__ = [
    "SecurityTestRunner",
    "OpenAICSAgentTarget",
    "RedTeamEvaluator",
    "BlueTeamEvaluator",
    "HallucinationEvaluator",
]
