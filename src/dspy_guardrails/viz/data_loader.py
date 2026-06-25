"""Data loading and type detection for security reports."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st


@st.cache_data
def load_report(path: str) -> dict:
    """Load a single JSON report file."""
    with open(path) as f:
        return json.load(f)


def detect_type(data: dict) -> str:
    """Detect report type from its keys.

    Returns one of: "security_report", "experiment", "trajectory", "unknown".
    """
    if "redteam" in data or "blueteam" in data or "overall_score" in data:
        return "security_report"
    if "experiment" in data or "conditions" in data:
        return "experiment"
    if "phase_reached" in data or "progress_scores" in data:
        return "trajectory"
    return "unknown"


@st.cache_data
def scan_directory(path: str) -> dict[str, dict]:
    """Scan a directory for JSON files and classify them.

    Returns:
        Dict mapping file path to {"data": ..., "type": ...}.
    """
    results: dict[str, dict] = {}
    root = Path(path)
    if not root.is_dir():
        return results

    for json_file in sorted(root.rglob("*.json")):
        try:
            data = json.loads(json_file.read_text())
            if isinstance(data, dict):
                results[str(json_file)] = {
                    "data": data,
                    "type": detect_type(data),
                    "name": json_file.name,
                }
        except (json.JSONDecodeError, OSError):
            continue

    return results
