"""Server configuration."""

from dataclasses import dataclass, field


@dataclass
class ServerConfig:
    """Configuration for the guardrail server."""

    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8000
    workers: int = 1
    reload: bool = False
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    enable_metrics: bool = True
    enable_docs: bool = True
    log_format: str = "json"
