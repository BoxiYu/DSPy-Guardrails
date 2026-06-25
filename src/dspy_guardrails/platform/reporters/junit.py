"""JUnit Reporter - JUnit XML format for Jenkins CI integration."""

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.dom import minidom

from ..plugins import BasePlugin, PluginConfig, PluginResult, PluginType


class JUnitReporter(BasePlugin):
    """
    JUnit XML 格式报告生成器

    生成符合 JUnit XML 标准的测试报告，支持:
    - Jenkins
    - GitLab CI
    - CircleCI
    - Azure Pipelines

    Example:
        reporter = JUnitReporter()
        reporter.configure(PluginConfig(options={"output_path": "results.xml"}))
        result = reporter.execute({"results": scan_results})
    """

    name = "junit_reporter"
    version = "1.0.0"
    plugin_type = PluginType.REPORTER

    def __init__(self) -> None:
        self._output_path: Path | None = None
        self._suite_name = "dspyGuardrails Security Tests"
        self._config: PluginConfig | None = None

    def configure(self, config: PluginConfig) -> None:
        """Configure reporter."""
        self._config = config
        opts = config.options
        self._output_path = Path(opts.get("output_path", "security-results.xml"))
        self._suite_name = opts.get("suite_name", "dspyGuardrails Security Tests")

    def execute(self, context: dict[str, Any]) -> PluginResult:
        """Generate JUnit XML report."""
        results = context.get("results", [])

        xml_str = self._build_junit_xml(results)

        # Write to file
        if self._output_path:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._output_path, "w", encoding="utf-8") as f:
                f.write(xml_str)

        # Count failures
        failures = sum(1 for _, r in results
                       for v in r.data.get("vulnerabilities", []))
        failures += sum(1 for _, r in results
                        for a in r.data.get("attack_results", r.data.get("attacks", []))
                        if a.get("success"))

        return PluginResult(
            success=True,
            data={
                "output_path": str(self._output_path),
                "xml": xml_str,
                "failures": failures,
            },
            metrics={
                "test_failures": float(failures),
            },
        )

    def cleanup(self) -> None:
        """Cleanup resources."""
        pass

    def _build_junit_xml(self, results: list[Any]) -> str:
        """Build JUnit XML document."""
        testsuites = ET.Element("testsuites")
        testsuite = ET.SubElement(testsuites, "testsuite")
        testsuite.set("name", self._suite_name)
        testsuite.set("timestamp", datetime.now(timezone.utc).isoformat())

        tests = 0
        failures = 0
        time_total = 0.0

        for result_type, result in results:
            if result_type == "scan":
                vulns = result.data.get("vulnerabilities", [])
                _total_payloads = result.data.get("total_payloads", len(vulns))

                # Create test case for scan
                testcase = ET.SubElement(testsuite, "testcase")
                testcase.set("name", "Security Scan")
                testcase.set("classname", "dspyGuardrails.Scan")

                tests += 1

                if vulns:
                    failures += 1
                    failure = ET.SubElement(testcase, "failure")
                    failure.set("message", f"Found {len(vulns)} vulnerabilities")
                    failure.set("type", "SecurityVulnerability")
                    failure.text = "\n".join([
                        f"- {v.get('category')}: {v.get('technique')}"
                        for v in vulns[:10]
                    ])

            elif result_type == "attack":
                attacks = result.data.get("attack_results", result.data.get("attacks", []))
                successful = [a for a in attacks if a.get("success")]

                # Create test case for attack
                testcase = ET.SubElement(testsuite, "testcase")
                testcase.set("name", "Attack Resistance")
                testcase.set("classname", "dspyGuardrails.Attack")

                tests += 1
                time_total += sum(a.get("latency_ms", 0) for a in attacks) / 1000

                if successful:
                    failures += 1
                    failure = ET.SubElement(testcase, "failure")
                    failure.set("message", f"{len(successful)} attacks succeeded")
                    failure.set("type", "GuardrailBypass")
                    failure.text = "\n".join([
                        f"- {a.get('category')}: {a.get('technique')}"
                        for a in successful[:10]
                    ])

        testsuite.set("tests", str(tests))
        testsuite.set("failures", str(failures))
        testsuite.set("time", f"{time_total:.3f}")

        # Pretty print
        xml_str = ET.tostring(testsuites, encoding="unicode")
        return minidom.parseString(xml_str).toprettyxml(indent="  ")
