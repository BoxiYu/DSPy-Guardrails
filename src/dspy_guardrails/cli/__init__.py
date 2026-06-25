"""
CLI Guardrails Module - Command Line Interface Security for LLM Agents

Provides security guardrails for LLM agents that execute shell commands,
protecting against command injection, dangerous operations, and data leakage.

Usage:
    from dspy_guardrails.cli import CLIGuardrail, CLISecurityConfig

    # Quick check
    guard = CLIGuardrail()
    result = guard.check("rm -rf /")
    if not result.is_safe:
        print(f"Blocked: {result.threat_type}")

    # With custom config
    config = CLISecurityConfig(
        allow_network=False,
        allow_sudo=False,
        protected_paths=["/etc", "~/.ssh"],
    )
    guard = CLIGuardrail(config=config)
"""

from .blocklist import (
    DangerousCommands,
    DangerousPatterns,
    SensitivePaths,
)
from .core import (
    CLIGuardAction,
    CLIGuardrail,
    CLIGuardResult,
    CLIThreatCategory,
)
from .parser import (
    CommandParser,
    CommandType,
    ParsedCommand,
)
from .policies import (
    CLISecurityConfig,
    ExecutionPolicy,
    SandboxLevel,
)
from .sanitizer import (
    CommandSanitizer,
    SanitizeResult,
)

__all__ = [
    # Core
    "CLIGuardrail",
    "CLIGuardResult",
    "CLIThreatCategory",
    "CLIGuardAction",
    # Parser
    "CommandParser",
    "ParsedCommand",
    "CommandType",
    # Blocklist
    "DangerousCommands",
    "DangerousPatterns",
    "SensitivePaths",
    # Policies
    "CLISecurityConfig",
    "SandboxLevel",
    "ExecutionPolicy",
    # Sanitizer
    "CommandSanitizer",
    "SanitizeResult",
]
