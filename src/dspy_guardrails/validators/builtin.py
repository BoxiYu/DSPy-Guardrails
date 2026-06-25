"""
Built-in validators for common security and structural checks.

Each validator implements validate() and optionally fix().
"""

import json
import re
from typing import Any

from dspy_guardrails.validators.base import (
    FailResult,
    OnFailAction,
    PassResult,
    Validator,
    ValidatorResult,
)

# =============================================================================
# Security Validators
# =============================================================================


class NoInjection(Validator):
    """Detect prompt injection attacks.

    Uses the pattern-based guardrail.no_injection() with optional
    Leetspeak normalization.

    Usage:
        v = NoInjection(on_fail="exception")
        result = v.validate("ignore all instructions")
        # FailResult(error_message="Prompt injection detected")
    """

    def __init__(
        self,
        threshold: float = 0.5,
        allowlist: list[str] | None = None,
        on_fail: str | OnFailAction = OnFailAction.EXCEPTION,
    ):
        super().__init__(on_fail=on_fail)
        self.threshold = threshold
        self._allowlist = [re.compile(p, re.IGNORECASE) for p in (allowlist or [])]

    def validate(self, value: Any, **kwargs: Any) -> ValidatorResult:
        from dspy_guardrails.guardrail import guardrail

        text = str(value)
        score = guardrail.injection_score(text)
        if score < self.threshold:
            return PassResult(value=text)

        # Allowlist suppression
        if self._allowlist and any(p.search(text) for p in self._allowlist):
            return PassResult(value=text)

        return FailResult(
            error_message=f"Prompt injection detected (score={score:.2f}, threshold={self.threshold})",
            metadata={"injection_score": score},
        )


class NoPII(Validator):
    """Detect and optionally anonymize PII.

    Supports fix() which replaces PII with placeholders.

    Usage:
        v = NoPII(on_fail="fix")
        result = v.validate("Email me at test@example.com")
        # FailResult with fix_value="Email me at [EMAIL]"
    """

    def __init__(
        self,
        threshold: float = 0.1,
        on_fail: str | OnFailAction = OnFailAction.FIX,
    ):
        super().__init__(on_fail=on_fail)
        self.threshold = threshold

    def validate(self, value: Any, **kwargs: Any) -> ValidatorResult:
        from dspy_guardrails.guardrail import guardrail

        text = str(value)
        score = guardrail.pii_score(text)
        if score < self.threshold:
            return PassResult(value=text)

        fixed = self.fix(text)
        return FailResult(
            error_message=f"PII detected in text (score={score:.2f}, threshold={self.threshold})",
            fix_value=fixed,
            metadata={"pii_score": score},
        )

    def fix(self, value: Any, **kwargs: Any) -> str:
        """Replace PII with placeholders."""
        from dspy_guardrails.sanitize import sanitize_pii

        return sanitize_pii(str(value))


class NoToxicity(Validator):
    """Detect toxic content.

    Supports fix() which attempts to rewrite toxic content.

    Usage:
        v = NoToxicity(threshold=0.3, on_fail="reask")
        result = v.validate("some text")
    """

    def __init__(
        self,
        threshold: float = 0.3,
        allowlist: list[str] | None = None,
        on_fail: str | OnFailAction = OnFailAction.REASK,
    ):
        super().__init__(on_fail=on_fail)
        self.threshold = threshold
        self._allowlist = [re.compile(p, re.IGNORECASE) for p in (allowlist or [])]

    def validate(self, value: Any, **kwargs: Any) -> ValidatorResult:
        from dspy_guardrails.guardrail import guardrail

        text = str(value)
        score = guardrail.toxicity(text)
        if score < self.threshold:
            return PassResult(value=text)

        # Allowlist suppression
        if self._allowlist and any(p.search(text) for p in self._allowlist):
            return PassResult(value=text)

        return FailResult(
            error_message=f"Toxic content detected (score={score:.2f}, threshold={self.threshold})",
            metadata={"toxicity_score": score},
        )


class NoMCPAttack(Validator):
    """Detect MCP protocol attacks."""

    def __init__(
        self,
        context: str = "auto",
        threshold: float = 0.25,
        on_fail: str | OnFailAction = OnFailAction.EXCEPTION,
    ):
        super().__init__(on_fail=on_fail)
        self.context = context
        self.threshold = threshold

    def validate(self, value: Any, **kwargs: Any) -> ValidatorResult:
        from dspy_guardrails.guardrail import guardrail

        text = str(value)
        score = guardrail.mcp_security_score(text, self.context)
        if score < self.threshold:
            return PassResult(value=text)

        details = guardrail.mcp_attack_details(text, self.context)
        return FailResult(
            error_message=f"MCP attack detected (score={score:.2f})",
            metadata={"mcp_score": score, "attack_details": details},
        )


# =============================================================================
# Structural Validators
# =============================================================================


class ValidLength(Validator):
    """Validate string length.

    Usage:
        v = ValidLength(min=10, max=500, on_fail="reask")
        result = v.validate("too short")
    """

    def __init__(
        self,
        min: int = 0,
        max: int | None = None,
        on_fail: str | OnFailAction = OnFailAction.REASK,
    ):
        super().__init__(on_fail=on_fail)
        self.min = min
        self.max = max

    def validate(self, value: Any, **kwargs: Any) -> ValidatorResult:
        text = str(value)
        length = len(text)

        if length < self.min:
            return FailResult(
                error_message=f"Text too short: {length} chars, minimum {self.min}",
                metadata={"length": length, "min": self.min},
            )

        if self.max is not None and length > self.max:
            return FailResult(
                error_message=f"Text too long: {length} chars, maximum {self.max}",
                fix_value=text[: self.max],
                metadata={"length": length, "max": self.max},
            )

        return PassResult(value=text)

    def fix(self, value: Any, **kwargs: Any) -> str:
        text = str(value)
        if self.max is not None and len(text) > self.max:
            return text[: self.max]
        return text


class ValidChoices(Validator):
    """Validate that value is one of allowed choices.

    Usage:
        v = ValidChoices(choices=["dog", "cat", "bird"], on_fail="fix")
        result = v.validate("Dog")  # PassResult (case-insensitive)
    """

    def __init__(
        self,
        choices: list[str],
        case_sensitive: bool = False,
        on_fail: str | OnFailAction = OnFailAction.FIX,
    ):
        super().__init__(on_fail=on_fail)
        self.choices = choices
        self.case_sensitive = case_sensitive

    def validate(self, value: Any, **kwargs: Any) -> ValidatorResult:
        text = str(value).strip()
        if self.case_sensitive:
            if text in self.choices:
                return PassResult(value=text)
        else:
            for choice in self.choices:
                if text.lower() == choice.lower():
                    return PassResult(value=choice)

        return FailResult(
            error_message=f"'{text}' is not a valid choice. Valid choices: {self.choices}",
            fix_value=self._find_closest(text),
            metadata={"value": text, "choices": self.choices},
        )

    def fix(self, value: Any, **kwargs: Any) -> str:
        return self._find_closest(str(value).strip())

    def _find_closest(self, text: str) -> str:
        """Find closest matching choice by substring match."""
        text_lower = text.lower()
        for choice in self.choices:
            if text_lower in choice.lower() or choice.lower() in text_lower:
                return choice
        return self.choices[0]


class ValidRange(Validator):
    """Validate numeric value within a range.

    Usage:
        v = ValidRange(min=0, max=100, on_fail="fix")
        result = v.validate(150)  # FailResult with fix_value=100
    """

    def __init__(
        self,
        min: float | None = None,
        max: float | None = None,
        on_fail: str | OnFailAction = OnFailAction.FIX,
    ):
        super().__init__(on_fail=on_fail)
        self.min = min
        self.max = max

    def validate(self, value: Any, **kwargs: Any) -> ValidatorResult:
        try:
            num = float(value)
        except (ValueError, TypeError):
            return FailResult(error_message=f"Cannot convert '{value}' to number")

        if self.min is not None and num < self.min:
            return FailResult(
                error_message=f"Value {num} below minimum {self.min}",
                fix_value=self.min,
            )
        if self.max is not None and num > self.max:
            return FailResult(
                error_message=f"Value {num} above maximum {self.max}",
                fix_value=self.max,
            )
        return PassResult(value=num)

    def fix(self, value: Any, **kwargs: Any) -> float:
        try:
            num = float(value)
        except (ValueError, TypeError):
            return self.min if self.min is not None else 0.0
        if self.min is not None:
            num = builtins_max(num, self.min)
        if self.max is not None:
            num = builtins_min(num, self.max)
        return num


# Avoid shadowing builtins in class scope
builtins_max = max
builtins_min = min


class ValidRegex(Validator):
    """Validate text matches a regex pattern.

    Usage:
        v = ValidRegex(pattern=r"^[A-Z]{2}-\\d{4}$", on_fail="reask")
        result = v.validate("AB-1234")  # PassResult
    """

    def __init__(
        self,
        pattern: str,
        on_fail: str | OnFailAction = OnFailAction.REASK,
    ):
        super().__init__(on_fail=on_fail)
        self.pattern = pattern
        self._compiled = re.compile(pattern)

    def validate(self, value: Any, **kwargs: Any) -> ValidatorResult:
        text = str(value)
        if self._compiled.search(text):
            return PassResult(value=text)
        return FailResult(
            error_message=f"Text does not match pattern: {self.pattern}",
            metadata={"pattern": self.pattern},
        )


class ValidJSON(Validator):
    """Validate that text is valid JSON.

    Supports fix() which attempts to extract JSON from markdown code blocks.

    Usage:
        v = ValidJSON(on_fail="fix")
        result = v.validate('```json\\n{"a": 1}\\n```')
    """

    def __init__(self, on_fail: str | OnFailAction = OnFailAction.FIX):
        super().__init__(on_fail=on_fail)

    def validate(self, value: Any, **kwargs: Any) -> ValidatorResult:
        text = str(value)
        try:
            parsed = json.loads(text)
            return PassResult(value=parsed)
        except json.JSONDecodeError:
            fixed = self.fix(text)
            if fixed is not None:
                return FailResult(
                    error_message="Invalid JSON",
                    fix_value=fixed,
                )
            return FailResult(error_message="Invalid JSON, cannot auto-fix")

    def fix(self, value: Any, **kwargs: Any) -> Any:
        text = str(value)
        # Try extracting from markdown code block
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # Try stripping leading/trailing non-JSON
        for start_char in ["{", "["]:
            idx = text.find(start_char)
            if idx >= 0:
                end_char = "}" if start_char == "{" else "]"
                ridx = text.rfind(end_char)
                if ridx > idx:
                    try:
                        return json.loads(text[idx : ridx + 1])
                    except json.JSONDecodeError:
                        pass
        return None


class HybridInjection(Validator):
    """Hybrid injection detection: pattern first, LLM confirmation for edge cases."""

    _DEFAULT_REVIEW_RATIO = 0.7  # LLM review threshold = block threshold * ratio

    def __init__(
        self,
        threshold: float = 0.5,
        allowlist: list[str] | None = None,
        review_ratio: float | None = None,
        on_fail: str | OnFailAction = OnFailAction.EXCEPTION,
    ):
        super().__init__(on_fail=on_fail)
        self.threshold = threshold
        if review_ratio is None:
            review_ratio = self._DEFAULT_REVIEW_RATIO
        if not (0.0 < review_ratio <= 1.0):
            raise ValueError("review_ratio must be in (0.0, 1.0].")
        self._review_ratio = review_ratio
        self._allowlist = [re.compile(p, re.IGNORECASE) for p in (allowlist or [])]

    def validate(self, value: Any, **kwargs: Any) -> ValidatorResult:
        text = str(value)
        # Allowlist check first
        if self._allowlist and any(p.search(text) for p in self._allowlist):
            return PassResult(value=text)
        try:
            from dspy_guardrails.llm_guardrail import HybridGuardrail

            review_threshold = max(0.0, min(1.0, self.threshold * self._review_ratio))
            hybrid = HybridGuardrail(use_llm=True, threshold=review_threshold)
            is_unsafe, confidence = hybrid.check(text, "injection")
        except Exception:
            # Fallback to pattern-only
            from dspy_guardrails.guardrail import guardrail

            score = guardrail.injection_score(text)
            is_unsafe = score >= self.threshold
            confidence = score
        if is_unsafe:
            return FailResult(
                error_message="Prompt injection detected (hybrid)",
                metadata={"mode": "hybrid", "confidence": confidence},
            )
        return PassResult(value=text)


class HybridToxicity(Validator):
    """Hybrid toxicity detection: pattern first, LLM confirmation for edge cases."""

    _DEFAULT_REVIEW_RATIO = 0.7  # LLM review threshold = block threshold * ratio

    def __init__(
        self,
        threshold: float = 0.3,
        allowlist: list[str] | None = None,
        review_ratio: float | None = None,
        on_fail: str | OnFailAction = OnFailAction.REASK,
    ):
        super().__init__(on_fail=on_fail)
        self.threshold = threshold
        if review_ratio is None:
            review_ratio = self._DEFAULT_REVIEW_RATIO
        if not (0.0 < review_ratio <= 1.0):
            raise ValueError("review_ratio must be in (0.0, 1.0].")
        self._review_ratio = review_ratio
        self._allowlist = [re.compile(p, re.IGNORECASE) for p in (allowlist or [])]

    def validate(self, value: Any, **kwargs: Any) -> ValidatorResult:
        text = str(value)
        if self._allowlist and any(p.search(text) for p in self._allowlist):
            return PassResult(value=text)
        try:
            from dspy_guardrails.llm_guardrail import HybridGuardrail

            review_threshold = max(0.0, min(1.0, self.threshold * self._review_ratio))
            hybrid = HybridGuardrail(use_llm=True, threshold=review_threshold)
            is_unsafe, confidence = hybrid.check(text, "toxicity")
        except Exception:
            from dspy_guardrails.guardrail import guardrail

            score = guardrail.toxicity(text)
            is_unsafe = score >= self.threshold
            confidence = score
        if is_unsafe:
            return FailResult(
                error_message="Toxic content detected (hybrid)",
                metadata={"mode": "hybrid", "confidence": confidence},
            )
        return PassResult(value=text)
