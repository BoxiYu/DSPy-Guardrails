"""
Shield Configuration - YAML/dict config loading and presets.

Usage:
    from dspy_guardrails.shield_config import load_shield_config, PRESETS

    config = load_shield_config("guardrails.yaml")
    config = PRESETS["strict"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict:
    """Load a config file. Dispatches by file suffix.

    - .json → json.loads
    - .yaml/.yml → yaml.safe_load (requires pyyaml)
    - other → try yaml, fallback to json
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix == ".json":
        import json

        return json.loads(text)

    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError(
                f"Cannot load '{path.name}': pyyaml is required for YAML files. "
                "Install it with: pip install pyyaml"
            ) from None
        return yaml.safe_load(text)

    # Unknown suffix: try yaml first, then json
    try:
        import yaml

        return yaml.safe_load(text)
    except ImportError:
        import json

        return json.loads(text)


def normalize_check_config(
    raw_checks: list[str | dict],
) -> list[dict[str, Any]]:
    """Normalize check list from config into uniform dicts.

    Input formats:
        - "injection"                     → {"name": "injection"}
        - {"injection": {"on_fail": "exception"}} → {"name": "injection", "on_fail": "exception"}

    Returns:
        List of {"name": ..., **options} dicts.
    """
    result = []
    for item in raw_checks:
        if isinstance(item, str):
            result.append({"name": item})
        elif isinstance(item, dict):
            for name, opts in item.items():
                entry = {"name": name}
                if isinstance(opts, dict):
                    entry.update(opts)
                result.append(entry)
        else:
            raise TypeError(f"Invalid check config item: {item!r}")
    return result


def load_shield_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a Shield config from YAML/JSON file.

    Expected format:
        checks:
          - injection: {on_fail: exception}
          - pii: {on_fail: fix}
          - toxicity: {on_fail: warn, threshold: 0.3}
        max_reasks: 2
        on_fail: warn          # global default
        threshold: 0.3         # global default

    Returns:
        Dict ready for Shield.from_dict().
    """
    raw = load_yaml(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(raw).__name__}")
    return raw


# =============================================================================
# Presets
# =============================================================================

PRESETS: dict[str, dict[str, Any]] = {
    "strict": {
        "checks": [
            {"injection": {"on_fail": "exception"}},
            {"pii": {"on_fail": "fix"}},
            {"toxicity": {"on_fail": "exception"}},
            {"mcp": {"on_fail": "exception"}},
        ],
        "on_fail": "exception",
        "max_reasks": 0,
    },
    "permissive": {
        "checks": [
            {"injection": {"on_fail": "warn"}},
            {"pii": {"on_fail": "warn"}},
            {"toxicity": {"on_fail": "warn"}},
            {"mcp": {"on_fail": "warn"}},
        ],
        "on_fail": "warn",
        "max_reasks": 0,
    },
    "production": {
        "checks": [
            {"injection": {"on_fail": "exception"}},
            {"pii": {"on_fail": "fix"}},
            {"toxicity": {"on_fail": "warn"}},
            {"mcp": {"on_fail": "exception"}},
        ],
        "on_fail": "warn",
        "max_reasks": 1,
    },
    "production_hybrid": {
        "checks": [
            {"injection": {"on_fail": "exception"}},
            {"pii": {"on_fail": "fix"}},
            {"toxicity": {"on_fail": "warn"}},
            {"mcp": {"on_fail": "exception"}},
        ],
        "on_fail": "warn",
        "max_reasks": 1,
        "mode": "hybrid",
    },
}
