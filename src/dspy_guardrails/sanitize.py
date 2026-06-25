"""
Sanitize / Fix Engine - Content remediation and redaction

Provides PII redaction, injection cleaning, and toxic content filtering.

Usage:
    from dspy_guardrails.sanitize import (
        sanitize_pii,
        sanitize_injection,
        sanitize_output,
    )

    # PII redaction
    clean = sanitize_pii("Email me at test@example.com or call 13812345678")
    # "Email me at [EMAIL] or call [PHONE]"

    # Injection cleaning
    clean = sanitize_injection("Hello\\nignore all instructions\\nWorld")
    # "Hello\\n[REDACTED]\\nWorld"

    # Composite output sanitization
    clean = sanitize_output(text, checks=["pii", "injection"])
"""

import re
from dataclasses import dataclass, field


@dataclass
class SanitizeReport:
    """Report of what was sanitized."""

    original: str
    sanitized: str
    replacements: list[dict] = field(default_factory=list)

    @property
    def was_modified(self) -> bool:
        return self.original != self.sanitized

    @property
    def count(self) -> int:
        return len(self.replacements)


# =============================================================================
# PII Patterns & Replacements
# =============================================================================

_PII_PATTERNS: list[tuple[str, str, re.Pattern]] = [
    (
        "EMAIL",
        "[EMAIL]",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    ),
    (
        "PHONE_CN",
        "[PHONE]",
        re.compile(r"\b1[3-9]\d{9}\b"),
    ),
    (
        "PHONE_US",
        "[PHONE]",
        re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    ),
    (
        "SSN",
        "[SSN]",
        re.compile(r"\b\d{3}[-.\s]\d{2}[-.\s]\d{4}\b"),
    ),
    (
        "CREDIT_CARD",
        "[CREDIT_CARD]",
        re.compile(r"\b(?:\d{4}[-.\s]?){3}\d{4}\b"),
    ),
    (
        "IP_ADDRESS",
        "[IP]",
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ),
    (
        "API_KEY_OPENAI",
        "[API_KEY]",
        re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    ),
    (
        "API_KEY_AWS",
        "[API_KEY]",
        re.compile(r"AKIA[A-Z0-9]{16}"),
    ),
    (
        "API_KEY_GITHUB",
        "[API_KEY]",
        re.compile(r"ghp_[a-zA-Z0-9]{36}"),
    ),
    (
        "JWT",
        "[TOKEN]",
        re.compile(r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*"),
    ),
    (
        "CONNECTION_STRING",
        "[CONNECTION_STRING]",
        re.compile(r"(?:mongodb|postgres|mysql|redis)://[^\s'\"]+"),
    ),
]


def sanitize_pii(text: str, report: bool = False) -> str | SanitizeReport:
    """Replace PII with placeholder tokens.

    Args:
        text: Input text.
        report: If True, return SanitizeReport instead of string.

    Returns:
        Sanitized text, or SanitizeReport if report=True.
    """
    result = text
    replacements = []

    for pii_type, placeholder, pattern in _PII_PATTERNS:
        for match in pattern.finditer(result):
            replacements.append({
                "type": pii_type,
                "original": match.group(),
                "replacement": placeholder,
                "start": match.start(),
                "end": match.end(),
            })
        result = pattern.sub(placeholder, result)

    if report:
        return SanitizeReport(
            original=text,
            sanitized=result,
            replacements=replacements,
        )
    return result


# =============================================================================
# Injection Pattern Cleaning
# =============================================================================

_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
        re.IGNORECASE,
    ),
    re.compile(r"ignore\s+all\s+(instructions?|prompts?|rules?)", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior)", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all|your\s+instructions?)", re.IGNORECASE),
    re.compile(
        r"you\s+are\s+now\s+(?:a|an)?\s*(?:DAN|evil|unrestricted|jailbroken)",
        re.IGNORECASE,
    ),
    re.compile(r"pretend\s+(?:to\s+be|you\s+are)", re.IGNORECASE),
    re.compile(r"jailbreak|DAN\s+mode|developer\s+mode", re.IGNORECASE),
    re.compile(r"\[INST\]|\[/INST\]"),
    re.compile(
        r"bypass\s+(?:your\s+)?(?:safety|restrictions?|filters?)", re.IGNORECASE
    ),
    re.compile(r"\[(?:SYSTEM|ADMIN|HIDDEN|OVERRIDE)\][^\]]*", re.IGNORECASE),
    re.compile(
        r"(?:NEW|REAL|ACTUAL|TRUE)\s+(?:task|mission|objective|instruction):",
        re.IGNORECASE,
    ),
    # Chinese injection
    re.compile(r"忽略.*(之前|以上|所有).*(指令|命令|规则)"),
    re.compile(r"假装你是|扮演.*(?:黑客|邪恶)"),
    re.compile(r"绕过.*(?:安全|限制)"),
]

_REDACTED = "[REDACTED]"


def sanitize_injection(
    text: str, replacement: str = _REDACTED, report: bool = False
) -> str | SanitizeReport:
    """Remove or replace prompt injection patterns.

    Args:
        text: Input text.
        replacement: String to replace injection patterns with.
        report: If True, return SanitizeReport.

    Returns:
        Sanitized text.
    """
    result = text
    replacements = []

    for pattern in _INJECTION_PATTERNS:
        for match in pattern.finditer(result):
            replacements.append({
                "type": "INJECTION",
                "original": match.group(),
                "replacement": replacement,
                "start": match.start(),
                "end": match.end(),
            })
        result = pattern.sub(replacement, result)

    if report:
        return SanitizeReport(
            original=text,
            sanitized=result,
            replacements=replacements,
        )
    return result


# =============================================================================
# Command Injection Cleaning
# =============================================================================

_DANGEROUS_COMMAND_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("SHELL_INJECTION", re.compile(r"[;&|`]\s*(?:rm|cat|chmod|wget|curl|bash|sh)\b")),
    ("SUBSHELL", re.compile(r"`[^`]+`")),
    ("COMMAND_SUBSTITUTION", re.compile(r"\$\([^)]+\)")),
    ("PIPE_TO_SHELL", re.compile(r"\|\s*(?:bash|sh|python|perl|ruby)\b")),
    ("CURL_PIPE", re.compile(r"curl\s+[^\s]+\s*\|\s*(?:bash|sh)")),
]


def sanitize_commands(text: str, report: bool = False) -> str | SanitizeReport:
    """Remove dangerous shell command patterns from text.

    Args:
        text: Input text.
        report: If True, return SanitizeReport.

    Returns:
        Sanitized text.
    """
    result = text
    replacements = []

    for cmd_type, pattern in _DANGEROUS_COMMAND_PATTERNS:
        for match in pattern.finditer(result):
            replacements.append({
                "type": cmd_type,
                "original": match.group(),
                "replacement": _REDACTED,
            })
        result = pattern.sub(_REDACTED, result)

    if report:
        return SanitizeReport(
            original=text,
            sanitized=result,
            replacements=replacements,
        )
    return result


# =============================================================================
# Composite Sanitizer
# =============================================================================


def sanitize_output(
    text: str,
    checks: list[str] | None = None,
    report: bool = False,
) -> str | SanitizeReport:
    """Apply multiple sanitization passes.

    Args:
        text: Input text.
        checks: List of sanitizers to apply. Defaults to ["pii", "injection"].
            Available: "pii", "injection", "commands"
        report: If True, return SanitizeReport with all replacements.

    Returns:
        Sanitized text.
    """
    if checks is None:
        checks = ["pii", "injection"]

    sanitizers = {
        "pii": sanitize_pii,
        "injection": sanitize_injection,
        "commands": sanitize_commands,
    }

    result = text
    all_replacements = []

    for name in checks:
        fn = sanitizers.get(name)
        if fn is None:
            continue
        sub_report = fn(result, report=True)
        result = sub_report.sanitized
        all_replacements.extend(sub_report.replacements)

    if report:
        return SanitizeReport(
            original=text,
            sanitized=result,
            replacements=all_replacements,
        )
    return result
