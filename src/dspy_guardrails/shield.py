"""
Shield - Unified entry point for dspy-guardrails.

Consolidates 150+ public classes into a single, intuitive API with 4 methods.

Usage:
    from dspy_guardrails import Shield

    # Zero-config (strong defaults)
    shield = Shield()
    result = shield.check("Hello world")
    if result:
        print("Safe!")

    # With specific checks
    shield = Shield(checks=["injection", "pii"], on_fail="fix")
    result = shield.check("Email me at test@example.com")
    print(result.output)  # "Email me at [EMAIL]"

    # LLM wrapping with reask
    result = shield.wrap(llm, prompt="Write a safe response")

    # Streaming
    async for token in shield.stream(token_iter):
        print(token, end="")
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dspy_guardrails.shield_config import (
    PRESETS,
    load_shield_config,
    normalize_check_config,
)

# =============================================================================
# Check name → Validator class mapping
# =============================================================================

_CHECK_REGISTRY: dict[str, str] = {
    "injection": "NoInjection",
    "pii": "NoPII",
    "toxicity": "NoToxicity",
    "mcp": "NoMCPAttack",
    "length": "ValidLength",
    "json": "ValidJSON",
    "regex": "ValidRegex",
    "choices": "ValidChoices",
    "range": "ValidRange",
}

# on_fail string aliases
_ON_FAIL_MAP = {
    "warn": "noop",
    "block": "exception",
    "fix": "fix",
    "exception": "exception",
    "noop": "noop",
    "reask": "reask",
    "refrain": "refrain",
    "filter": "filter",
}

_DEFAULT_CHECKS = ["injection", "pii", "toxicity", "mcp"]
_DEFAULT_THRESHOLD = 0.3
_DEFAULT_ON_FAIL: dict[str, str] = {
    "injection": "exception",
    "mcp": "exception",
    "pii": "fix",
    "toxicity": "warn",
}

# =============================================================================
# Domain-aware allowlists for reducing false positives
# =============================================================================

_DOMAIN_ALLOWLISTS: dict[str, dict[str, list[str]]] = {
    "technical": {
        "injection": [
            r"\bignore\s+(?:the\s+)?(?:noise|error|warning|exception|log|output|stderr)\b",
            r"\boverride\s+(?:the\s+)?(?:default|setting|config|option|parameter)\b",
            r"\bbypass\s+(?:the\s+)?(?:cache|proxy|firewall|filter|limit|queue|traffic)\b",
            r"\breset\s+(?:the\s+)?(?:connection|session|counter|timer|state)\b",
        ],
        "toxicity": [
            r"\bkill\s+(?:the\s+)?(?:process|thread|task|job|server|container|pod|worker|daemon)\b",
            r"\bdie\s*\(|die\s+(?:gracefully)\b",
            r"\battack\s+(?:vector|surface|pattern|scenario|simulation|model)\b",
            r"\babort\s+(?:the\s+)?(?:process|transaction|request|operation|task)\b",
            r"\bdead\s*(?:lock|letter|code)\b",
            r"\bexecute\s+(?:the\s+)?(?:command|query|script|task|function|test)\b",
        ],
    },
}


# =============================================================================
# Result types
# =============================================================================


@dataclass
class ShieldIssue:
    """A single issue found during checking."""

    check: str
    message: str
    severity: str = "medium"
    suggestion: str = ""
    fixed_value: Any = None


@dataclass
class ShieldResult:
    """Result of a Shield check."""

    safe: bool
    output: Any = None
    raw_output: Any = None
    issues: list[ShieldIssue] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.safe

    def __str__(self) -> str:
        return self.summary(color=False)

    def summary(self, color: bool = True) -> str:
        """Human-readable summary."""
        if self.safe:
            mark = "\033[32m✓ SAFE\033[0m" if color else "SAFE"
            return mark

        mark = "\033[31m✗ BLOCKED\033[0m" if color else "BLOCKED"
        lines = [mark]
        for issue in self.issues:
            sev = issue.severity.upper()
            lines.append(f"  [{sev}] {issue.check}: {issue.message}")
            if issue.suggestion:
                lines.append(f"         → {issue.suggestion}")
        return "\n".join(lines)


# =============================================================================
# Severity heuristic
# =============================================================================

_SEVERITY_MAP: dict[str, str] = {
    "injection": "critical",
    "mcp": "critical",
    "pii": "high",
    "toxicity": "high",
    "length": "low",
    "json": "medium",
    "regex": "medium",
    "choices": "low",
    "range": "low",
}


# =============================================================================
# Shield
# =============================================================================


class Shield:
    """Unified guardrail entry point.

    Args:
        checks: List of check names. Defaults to ["injection", "pii", "toxicity", "mcp"].
        on_fail: Global on_fail strategy or per-check dict.
            Default (strong): injection/mcp → block, pii → fix, toxicity → warn
            "warn"      — record issue, output passes through
            "block"     — raise exception / mark unsafe
            "fix"       — auto-fix (PII masking, etc.)
            "exception" — alias for block
        threshold: Global or per-check score threshold (reserved for future use).
        review_ratio: Hybrid LLM review threshold ratio. Review runs when
            rule-based score > threshold * review_ratio. Default: 0.7.
        max_reasks: Number of LLM reask attempts in wrap().
        domain: Domain-aware allowlist (e.g., "technical") to reduce false positives.
        allowlists: Custom per-check allowlists to reduce false positives.
            Dict mapping check name to list of regex patterns.
            Example: {"injection": [r"bypass\\s+cache", r"kill\\s+process"]}
        mode: Detection mode - "fast" (pattern-based) or "hybrid" (pattern + LLM).
        require_llm: If True, raise ValueError when mode="hybrid" but LLM is not configured.
            If False (default), silently falls back to "fast" mode.
    """

    def __init__(
        self,
        checks: list[str] | None = None,
        on_fail: str | dict[str, str] = _DEFAULT_ON_FAIL,
        threshold: float | dict[str, float] = 0.3,
        review_ratio: float = 0.7,
        max_reasks: int = 1,
        domain: str | None = None,
        allowlists: dict[str, list[str]] | None = None,
        mode: str = "fast",
        require_llm: bool = False,
    ):
        self._check_names = checks or list(_DEFAULT_CHECKS)
        if isinstance(on_fail, dict):
            self._on_fail = dict(on_fail)
        else:
            self._on_fail = on_fail
        self._threshold = threshold
        if not (0.0 < review_ratio <= 1.0):
            raise ValueError("review_ratio must be in (0.0, 1.0].")
        self._review_ratio = review_ratio
        self._max_reasks = max_reasks
        self._domain = domain
        self._custom_allowlists = allowlists or {}
        self._requested_mode = mode
        self._mode = mode
        self._llm_available = False
        self._llm_fallback_reason: str | None = None

        if mode == "hybrid":
            self._check_dspy_configured(require_llm=require_llm)

        # Build validators
        self._validators = self._build_validators()

    # -----------------------------------------------------------------
    # Construction helpers
    # -----------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> Shield:
        """Load configuration from a YAML file."""
        config = load_shield_config(path)
        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Shield:
        """Build from a config dict."""
        raw_checks = config.get("checks")
        on_fail = config.get("on_fail", "warn")
        threshold = config.get("threshold", 0.3)
        review_ratio = config.get("review_ratio", 0.7)
        max_reasks = config.get("max_reasks", 1)

        if raw_checks is not None:
            normalized = normalize_check_config(raw_checks)
            check_names = [c["name"] for c in normalized]
            # Build per-check on_fail overrides
            per_check_on_fail: dict[str, str] = {}
            per_check_threshold: dict[str, float] = {}
            for c in normalized:
                if "on_fail" in c:
                    per_check_on_fail[c["name"]] = c["on_fail"]
                if "threshold" in c:
                    per_check_threshold[c["name"]] = c["threshold"]

            final_on_fail: str | dict[str, str] = on_fail
            if per_check_on_fail:
                if isinstance(on_fail, str):
                    final_on_fail = {name: on_fail for name in check_names}
                else:
                    final_on_fail = dict(on_fail)
                for k, v in per_check_on_fail.items():
                    final_on_fail[k] = v

            final_threshold: float | dict[str, float] = threshold
            if per_check_threshold:
                if isinstance(threshold, (int, float)):
                    final_threshold = {name: float(threshold) for name in check_names}
                else:
                    final_threshold = dict(threshold)
                for k, v in per_check_threshold.items():
                    final_threshold[k] = v

            return cls(
                checks=check_names,
                on_fail=final_on_fail,
                threshold=final_threshold,
                review_ratio=review_ratio,
                max_reasks=max_reasks,
                domain=config.get("domain"),
                allowlists=config.get("allowlists"),
                mode=config.get("mode", "fast"),
            )

        return cls(
            on_fail=on_fail,
            threshold=threshold,
            review_ratio=review_ratio,
            max_reasks=max_reasks,
            domain=config.get("domain"),
            allowlists=config.get("allowlists"),
            mode=config.get("mode", "fast"),
        )

    @classmethod
    def preset(cls, name: str) -> Shield:
        """Load a named preset: 'strict', 'permissive', 'production'."""
        if name not in PRESETS:
            available = ", ".join(sorted(PRESETS))
            raise ValueError(f"Unknown preset '{name}'. Available: {available}")
        return cls.from_dict(PRESETS[name])

    # -----------------------------------------------------------------
    # Internal: build validator instances
    # -----------------------------------------------------------------

    def _check_dspy_configured(self, require_llm: bool = False) -> None:
        """Check if DSPy LM is configured; hybrid will fallback to pattern if not.

        Args:
            require_llm: If True, raise ValueError instead of falling back.
        """
        import logging
        import warnings

        logger = logging.getLogger(__name__)

        try:
            import dspy

            if not getattr(dspy.settings, "lm", None):
                self._llm_available = False
                self._llm_fallback_reason = "DSPy LM not configured"

                if require_llm:
                    raise ValueError(
                        "Shield mode='hybrid' requires a configured DSPy LM, but none found. "
                        "Configure with: dspy.configure(lm=dspy.LM('openai/gpt-4', api_key=...)). "
                        "Or set require_llm=False to fall back to pattern-based detection."
                    )

                msg = (
                    "Shield hybrid mode requested but DSPy LM not configured. "
                    "Falling back to pattern-based detection (lower accuracy). "
                    "To use hybrid mode: dspy.configure(lm=dspy.LM('openai/gpt-4', api_key=...)). "
                    "To suppress this warning: use mode='fast' or set require_llm=True to fail fast."
                )
                warnings.warn(msg, UserWarning, stacklevel=4)
                logger.warning(msg)
                self._mode = "fast"
            else:
                self._llm_available = True
        except ImportError:
            self._llm_available = False
            self._llm_fallback_reason = "dspy not installed"

            if require_llm:
                raise ValueError(
                    "Shield mode='hybrid' requires dspy package, but it's not installed. "
                    "Install with: pip install dspy"
                )

            msg = "dspy not available. Falling back to pattern-based detection."
            warnings.warn(msg, UserWarning, stacklevel=4)
            logger.warning(msg)
            self._mode = "fast"
            self._mode = "fast"

    def _resolve_threshold(self, check_name: str) -> float:
        """Get the threshold for a specific check."""
        defaults = {"injection": 0.5, "pii": 0.1, "toxicity": 0.3, "mcp": 0.25}
        if isinstance(self._threshold, dict):
            return self._threshold.get(check_name, defaults.get(check_name, 0.3))
        if self._threshold == _DEFAULT_THRESHOLD:
            return defaults.get(check_name, self._threshold)
        return self._threshold

    def _resolve_on_fail(self, check_name: str) -> str:
        """Get the on_fail action for a specific check."""
        if isinstance(self._on_fail, dict):
            raw = self._on_fail.get(check_name, "noop")
        else:
            raw = self._on_fail
        return _ON_FAIL_MAP.get(raw, raw)

    def _build_validators(self) -> list:
        """Instantiate Validator objects for each check."""
        from dspy_guardrails.validators.builtin import (
            NoInjection,
            NoMCPAttack,
            NoPII,
            NoToxicity,
            ValidChoices,
            ValidJSON,
            ValidLength,
            ValidRange,
            ValidRegex,
        )

        cls_map = {
            "injection": NoInjection,
            "pii": NoPII,
            "toxicity": NoToxicity,
            "mcp": NoMCPAttack,
            "length": ValidLength,
            "json": ValidJSON,
            "regex": ValidRegex,
            "choices": ValidChoices,
            "range": ValidRange,
        }

        if self._mode == "hybrid":
            from dspy_guardrails.validators.builtin import HybridInjection, HybridToxicity

            cls_map["injection"] = HybridInjection
            cls_map["toxicity"] = HybridToxicity

        # Merge domain allowlists with custom allowlists
        domain_lists = _DOMAIN_ALLOWLISTS.get(self._domain, {}) if self._domain else {}
        merged_allowlists: dict[str, list[str]] = {}
        for name in self._check_names:
            patterns = []
            if name in domain_lists:
                patterns.extend(domain_lists[name])
            if name in self._custom_allowlists:
                patterns.extend(self._custom_allowlists[name])
            if patterns:
                merged_allowlists[name] = patterns

        validators = []
        for name in self._check_names:
            cls = cls_map.get(name)
            if cls is None:
                raise ValueError(
                    f"Unknown check '{name}'. Available: {', '.join(sorted(cls_map))}"
                )
            on_fail = self._resolve_on_fail(name)
            threshold = self._resolve_threshold(name)
            allowlist = merged_allowlists.get(name)
            if name in ("injection", "toxicity"):
                kwargs = {
                    "threshold": threshold,
                    "allowlist": allowlist,
                    "on_fail": on_fail,
                }
                if self._mode == "hybrid":
                    kwargs["review_ratio"] = self._review_ratio
                validators.append(cls(**kwargs))
            elif name in ("pii", "mcp"):
                validators.append(cls(threshold=threshold, on_fail=on_fail))
            else:
                validators.append(cls(on_fail=on_fail))
        return validators

    def _build_guard(self):
        """Build a Guard instance from validators."""
        from dspy_guardrails.validators.guard import Guard

        return Guard(validators=list(self._validators))

    def _build_async_guard(self):
        """Build an AsyncGuard instance from validators."""
        from dspy_guardrails.validators.guard import AsyncGuard

        return AsyncGuard(validators=list(self._validators))

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    def diagnose(self) -> dict[str, Any]:
        """Return diagnostic information about the Shield's current state.

        Useful for debugging and verifying configuration.

        Returns:
            Dict with:
                - requested_mode: The mode passed to __init__
                - actual_mode: The mode actually being used
                - llm_available: Whether LLM is available for hybrid mode
                - fallback_reason: Why LLM is unavailable (if applicable)
                - checks: List of configured check names
                - on_fail: Current on_fail configuration
                - domain: Current domain allowlist (if set)

        Example:
            shield = Shield(mode="hybrid")
            print(shield.diagnose())
            # {'requested_mode': 'hybrid', 'actual_mode': 'fast',
            #  'llm_available': False, 'fallback_reason': 'DSPy LM not configured', ...}
        """
        return {
            "requested_mode": self._requested_mode,
            "actual_mode": self._mode,
            "llm_available": self._llm_available,
            "fallback_reason": self._llm_fallback_reason,
            "checks": list(self._check_names),
            "on_fail": self._on_fail,
            "threshold": self._threshold,
            "review_ratio": self._review_ratio,
            "domain": self._domain,
            "allowlists": self._custom_allowlists,
            "max_reasks": self._max_reasks,
        }

    # -----------------------------------------------------------------
    # Core API: check
    # -----------------------------------------------------------------

    def check(self, text: str) -> ShieldResult:
        """Synchronously check text against all configured checks.

        Args:
            text: The text to validate.

        Returns:
            ShieldResult with safe/issues/output.
        """
        from dspy_guardrails.validators.base import ValidationError

        guard = self._build_guard()
        try:
            result = guard.validate(text)
        except ValidationError as exc:
            issues = [
                ShieldIssue(
                    check=self._check_names[0] if self._check_names else "unknown",
                    message=e.error_message,
                    severity=_SEVERITY_MAP.get(
                        self._check_names[0] if self._check_names else "", "medium"
                    ),
                )
                for e in exc.errors
            ]
            return ShieldResult(
                safe=False,
                output=text,
                raw_output=text,
                issues=issues,
            )

        return self._guard_result_to_shield(result, text)

    # -----------------------------------------------------------------
    # Core API: acheck
    # -----------------------------------------------------------------

    async def acheck(self, text: str) -> ShieldResult:
        """Asynchronously check text against all configured checks.

        Args:
            text: The text to validate.

        Returns:
            ShieldResult with safe/issues/output.
        """
        from dspy_guardrails.validators.base import ValidationError

        guard = self._build_async_guard()
        try:
            result = await guard.validate(text)
        except ValidationError as exc:
            issues = [
                ShieldIssue(
                    check=self._check_names[0] if self._check_names else "unknown",
                    message=e.error_message,
                    severity="critical",
                )
                for e in exc.errors
            ]
            return ShieldResult(
                safe=False,
                output=text,
                raw_output=text,
                issues=issues,
            )

        return self._guard_result_to_shield(result, text)

    # -----------------------------------------------------------------
    # Core API: wrap
    # -----------------------------------------------------------------

    def wrap(
        self,
        llm: Any,
        *,
        messages: list[dict] | None = None,
        prompt: str | None = None,
    ) -> ShieldResult:
        """Call an LLM and validate+reask its output.

        Args:
            llm: A DSPy LM or callable.
            messages: Chat messages for the LLM.
            prompt: Simple prompt string (alternative to messages).

        Returns:
            ShieldResult after validation and optional reask.
        """
        from dspy_guardrails.validators.base import ValidationError

        guard = self._build_guard()
        try:
            result = guard(
                llm=llm,
                messages=messages,
                prompt=prompt,
                max_reasks=self._max_reasks,
            )
        except ValidationError as exc:
            issues = [
                ShieldIssue(
                    check="unknown",
                    message=e.error_message,
                    severity="critical",
                )
                for e in exc.errors
            ]
            return ShieldResult(
                safe=False,
                output=None,
                raw_output=None,
                issues=issues,
            )

        raw = result.raw_output
        return self._guard_result_to_shield(result, raw)

    # -----------------------------------------------------------------
    # Core API: stream
    # -----------------------------------------------------------------

    async def stream(self, token_iter: AsyncIterator[str]) -> AsyncIterator[str]:
        """Filter an async token stream through guardrail checks.

        Yields tokens that pass. Stops on violation if on_fail is block/exception.

        Args:
            token_iter: Async iterator of string tokens.

        Yields:
            Safe tokens.
        """
        from dspy_guardrails.streaming import StreamGuardrail

        sg = StreamGuardrail.from_validators(
            validators=list(self._validators),
            on_violation="block",
        )
        async for token in sg.filter(token_iter):
            yield token

    # -----------------------------------------------------------------
    # Internal conversion
    # -----------------------------------------------------------------

    def _guard_result_to_shield(self, result: Any, raw_output: Any) -> ShieldResult:
        """Convert a GuardResult to ShieldResult."""
        issues = []
        for i, err in enumerate(result.errors):
            check_name = (
                self._check_names[i]
                if i < len(self._check_names)
                else "unknown"
            )
            # Try to match error to the correct check name
            matched_name = self._match_error_to_check(err, check_name)
            issues.append(
                ShieldIssue(
                    check=matched_name,
                    message=err.error_message,
                    severity=_SEVERITY_MAP.get(matched_name, "medium"),
                    suggestion=self._suggestion_for(matched_name),
                    fixed_value=err.fix_value,
                )
            )

        return ShieldResult(
            safe=result.is_valid,
            output=result.output,
            raw_output=raw_output,
            issues=issues,
        )

    def _match_error_to_check(self, err: Any, fallback: str) -> str:
        """Try to match an error to the right check name based on message."""
        msg = err.error_message.lower()
        if "injection" in msg:
            return "injection"
        if "pii" in msg or "personal" in msg:
            return "pii"
        if "toxic" in msg:
            return "toxicity"
        if "mcp" in msg:
            return "mcp"
        if "length" in msg:
            return "length"
        if "json" in msg:
            return "json"
        return fallback

    @staticmethod
    def _suggestion_for(check_name: str) -> str:
        """Return a human-readable suggestion for a check."""
        suggestions = {
            "injection": "Remove prompt injection patterns from input",
            "pii": "Use on_fail='fix' to auto-mask PII",
            "toxicity": "Rephrase to remove toxic language",
            "mcp": "Remove MCP attack patterns",
            "length": "Adjust text length to meet constraints",
            "json": "Ensure output is valid JSON",
            "regex": "Ensure output matches the expected pattern",
            "choices": "Use one of the allowed values",
            "range": "Use a value within the allowed range",
        }
        return suggestions.get(check_name, "")
