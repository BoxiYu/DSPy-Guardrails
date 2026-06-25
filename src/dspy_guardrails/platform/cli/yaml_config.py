"""YAML configuration parser for security testing.

This module provides Pydantic models for parsing and validating YAML configuration
files used by the dspyGuardrails security testing CLI.

Example YAML:
    target:
      type: guardrail
      value: no_injection
      name: "Injection Guardrail Test"

    scan:
      enabled: true
      max_payloads: 50
      categories:
        - injection
        - jailbreak

    attack:
      enabled: true
      budget: 100
      use_llm: false

    report:
      formats:
        - console
        - json
      output_dir: ./reports
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class TargetConfig(BaseModel):
    """Target configuration for security testing.

    Attributes:
        type: Target type ('guardrail' or 'http')
        value: Target value (guardrail name or URL)
        name: Optional human-readable name for reports
    """
    type: str  # guardrail, http
    value: str  # guardrail:name or URL
    name: str | None = None

    def to_target_string(self) -> str:
        """Convert to CLI target string format.

        Returns:
            Target string in format 'guardrail:name' or URL
        """
        if self.type == "guardrail":
            return f"guardrail:{self.value}"
        elif self.type == "http":
            return self.value
        else:
            return f"{self.type}:{self.value}"


class ScanConfig(BaseModel):
    """Scan configuration for quick security scans.

    Attributes:
        enabled: Whether to run scan phase
        max_payloads: Maximum number of payloads to test
        categories: Attack categories to include
        severity: Minimum severity level to report
    """
    enabled: bool = True
    max_payloads: int = 20
    categories: list[str] = Field(default_factory=lambda: ["injection", "jailbreak"])
    severity: str = "medium"


class AttackConfig(BaseModel):
    """Attack configuration for security evaluation.

    Attributes:
        enabled: Whether to run attack phase
        budget: Total number of attacks to execute
        types: Attack types to include (injection, jailbreak, bypass, mcp)
        use_llm: Use LLM for attack generation (requires DSPy configuration)
        stop_on_success: Stop after first successful attack
    """
    enabled: bool = True
    budget: int = 100
    types: list[str] = Field(default_factory=lambda: ["injection", "jailbreak", "bypass"])
    use_llm: bool = False
    stop_on_success: bool = False


class ReportConfig(BaseModel):
    """Report configuration for output generation.

    Attributes:
        formats: Output formats (console, json, html)
        output_dir: Directory for saving reports
    """
    formats: list[str] = Field(default_factory=lambda: ["console"])
    output_dir: str = "./reports"


class SecurityConfig(BaseModel):
    """Full security test configuration.

    This is the top-level configuration model that contains all
    sub-configurations for target, scan, attack, and report settings.

    Example YAML:
        target:
          type: guardrail
          value: no_injection

        scan:
          enabled: true
          max_payloads: 50
          categories:
            - injection
            - jailbreak

        attack:
          enabled: true
          budget: 100
          use_llm: false

        report:
          formats:
            - console
            - json
          output_dir: ./reports

    Attributes:
        target: Target configuration (required)
        scan: Scan configuration (optional, uses defaults)
        attack: Attack configuration (optional, uses defaults)
        report: Report configuration (optional, uses defaults)
    """
    target: TargetConfig
    scan: ScanConfig = Field(default_factory=ScanConfig)
    attack: AttackConfig = Field(default_factory=AttackConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "SecurityConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            SecurityConfig instance

        Raises:
            FileNotFoundError: If file does not exist
            yaml.YAMLError: If YAML is invalid
            pydantic.ValidationError: If configuration is invalid
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"Empty configuration file: {path}")

        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SecurityConfig":
        """Load configuration from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            SecurityConfig instance
        """
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save YAML file
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return self.model_dump()


def create_sample_config() -> str:
    """Create a sample YAML configuration.

    Returns:
        Sample YAML configuration as string
    """
    return '''# dspyGuardrails Security Test Configuration
# Generated sample config

target:
  type: guardrail
  value: no_injection
  name: "Injection Guardrail Test"

scan:
  enabled: true
  max_payloads: 50
  categories:
    - injection
    - jailbreak
  severity: medium

attack:
  enabled: true
  budget: 100
  types:
    - injection
    - jailbreak
    - bypass
  use_llm: false
  stop_on_success: false

report:
  formats:
    - console
    - json
  output_dir: ./security_reports
'''


def validate_config_file(path: str) -> tuple[bool, str | None]:
    """Validate a YAML configuration file.

    Args:
        path: Path to YAML configuration file

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        SecurityConfig.from_yaml(path)
        return True, None
    except FileNotFoundError as e:
        return False, f"File not found: {e}"
    except yaml.YAMLError as e:
        return False, f"Invalid YAML: {e}"
    except Exception as e:
        return False, f"Configuration error: {e}"
