"""
Structured Logging - JSON 格式结构化日志

使用 structlog 或 stdlib logging 输出结构化日志。
"""

import hashlib
import logging
from typing import Any

_structlog_available = False
_setup_done = False

try:
    import structlog

    _structlog_available = True
except ImportError:
    pass


def setup_logging(format: str = "json", level: int = logging.INFO) -> None:
    """Configure structured logging.

    Args:
        format: "json" for JSON output, "console" for human-readable.
        level: Logging level.
    """
    global _setup_done
    if _setup_done:
        return

    if _structlog_available:
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
        ]

        if format == "json":
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )

    _setup_done = True


def get_logger(name: str) -> Any:
    """Get a structured logger.

    Returns a structlog logger if available, otherwise stdlib logger.
    """
    if _structlog_available:
        return structlog.get_logger(name)
    return logging.getLogger(name)


def _hash_text(text: str) -> str:
    """Create a short hash of text for logging (privacy-safe)."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def log_check(
    logger: Any,
    check_name: str,
    text: str,
    result: bool | float,
    latency_ms: float,
    confidence: float | None = None,
    **extra: Any,
) -> None:
    """Log a guardrail check result with structured fields.

    Args:
        logger: Logger instance.
        check_name: Name of the check (e.g., "no_injection").
        text: Input text (will be hashed for privacy).
        result: Check result (bool or score).
        latency_ms: Check latency in milliseconds.
        confidence: Optional confidence score.
    """
    fields: dict[str, Any] = {
        "check": check_name,
        "input_hash": _hash_text(text),
        "input_length": len(text),
        "result": result,
        "latency_ms": round(latency_ms, 3),
    }
    if confidence is not None:
        fields["confidence"] = confidence
    fields.update(extra)

    if isinstance(result, bool) and not result:
        logger.warning("guardrail_blocked", **fields)
    else:
        logger.info("guardrail_check", **fields)
