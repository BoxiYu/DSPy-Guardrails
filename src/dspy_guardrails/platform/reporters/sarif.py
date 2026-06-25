"""SARIF Reporter - Static Analysis Results Interchange Format for GitHub/GitLab CI."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..plugins import BasePlugin, PluginConfig, PluginResult, PluginType


class SARIFReporter(BasePlugin):
    """
    SARIF 格式报告生成器

    生成符合 SARIF 2.1.0 标准的安全报告，可直接上传到:
    - GitHub Code Scanning
    - GitLab SAST
    - Azure DevOps

    Example:
        reporter = SARIFReporter()
        reporter.configure(PluginConfig(options={"output_path": "results.sarif"}))
        result = reporter.execute({"results": scan_results})
    """

    name = "sarif_reporter"
    version = "1.0.0"
    plugin_type = PluginType.REPORTER

    SARIF_VERSION = "2.1.0"
    SCHEMA_URI = "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json"

    def __init__(self) -> None:
        self._output_path: Path | None = None
        self._tool_name = "dspyGuardrails"
        self._tool_version = "0.3.0"
        self._config: PluginConfig | None = None

    def configure(self, config: PluginConfig) -> None:
        """Configure reporter."""
        self._config = config
        opts = config.options
        self._output_path = Path(opts.get("output_path", "security-results.sarif"))
        self._tool_name = opts.get("tool_name", "dspyGuardrails")
        self._tool_version = opts.get("tool_version", "0.3.0")

    def execute(self, context: dict[str, Any]) -> PluginResult:
        """Generate SARIF report."""
        results = context.get("results", [])

        sarif_report = self._build_sarif(results)

        # Write to file
        if self._output_path:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._output_path, "w", encoding="utf-8") as f:
                json.dump(sarif_report, f, indent=2)

        return PluginResult(
            success=True,
            data={
                "output_path": str(self._output_path),
                "sarif": sarif_report,
                "result_count": len(sarif_report["runs"][0]["results"]),
            },
            metrics={
                "vulnerabilities_reported": float(len(sarif_report["runs"][0]["results"])),
            },
        )

    def cleanup(self) -> None:
        """Cleanup resources."""
        pass

    def _build_sarif(self, results: list[Any]) -> dict[str, Any]:
        """Build SARIF document."""
        sarif_results = []
        rules: dict[str, dict[str, Any]] = {}

        for result_type, result in results:
            if result_type == "scan":
                vulns = result.data.get("vulnerabilities", [])
                for vuln in vulns:
                    rule_id = f"GUARD-{vuln.get('category', 'unknown').upper()}"

                    # Add rule if not exists
                    if rule_id not in rules:
                        rules[rule_id] = {
                            "id": rule_id,
                            "name": vuln.get("category", "Unknown"),
                            "shortDescription": {"text": f"{vuln.get('category', 'Security')} vulnerability"},
                            "fullDescription": {"text": f"Security vulnerability detected: {vuln.get('technique', 'unknown')}"},
                            "defaultConfiguration": {
                                "level": self._severity_to_sarif(vuln.get("severity", "medium")),
                            },
                            "properties": {
                                "category": vuln.get("category", "security"),
                            },
                        }

                    sarif_results.append({
                        "ruleId": rule_id,
                        "level": self._severity_to_sarif(vuln.get("severity", "medium")),
                        "message": {
                            "text": f"Vulnerability found: {vuln.get('technique', 'unknown')}. Payload: {vuln.get('prompt', '')[:100]}",
                        },
                        "properties": {
                            "payload_id": vuln.get("payload_id"),
                            "technique": vuln.get("technique"),
                        },
                    })

            elif result_type == "attack":
                attacks = result.data.get("attack_results", result.data.get("attacks", []))
                for atk in attacks:
                    if atk.get("success"):
                        rule_id = f"GUARD-{atk.get('category', 'unknown').upper()}-BYPASS"

                        if rule_id not in rules:
                            rules[rule_id] = {
                                "id": rule_id,
                                "name": f"{atk.get('category', 'Unknown')} Bypass",
                                "shortDescription": {"text": f"Guardrail bypass via {atk.get('category', 'unknown')}"},
                                "defaultConfiguration": {"level": "error"},
                            }

                        sarif_results.append({
                            "ruleId": rule_id,
                            "level": "error",
                            "message": {
                                "text": f"Attack succeeded: {atk.get('technique', 'unknown')}",
                            },
                            "properties": {
                                "payload_id": atk.get("payload_id"),
                                "latency_ms": atk.get("latency_ms"),
                            },
                        })

        return {
            "$schema": self.SCHEMA_URI,
            "version": self.SARIF_VERSION,
            "runs": [{
                "tool": {
                    "driver": {
                        "name": self._tool_name,
                        "version": self._tool_version,
                        "informationUri": "https://github.com/dspy-guardrails",
                        "rules": list(rules.values()),
                    },
                },
                "results": sarif_results,
                "invocations": [{
                    "executionSuccessful": True,
                    "endTimeUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                }],
            }],
        }

    def _severity_to_sarif(self, severity: str) -> str:
        """Convert severity to SARIF level."""
        mapping = {
            "critical": "error",
            "high": "error",
            "medium": "warning",
            "low": "note",
        }
        return mapping.get(severity.lower(), "warning")
