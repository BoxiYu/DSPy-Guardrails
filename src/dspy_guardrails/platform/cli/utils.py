"""CLI utility functions for target parsing and result formatting."""

import json
from urllib.parse import urlparse

from ..targets import GuardrailTarget, HTTPTarget, UnifiedTarget


def parse_target(target_str: str) -> UnifiedTarget:
    """
    Parse target string into UnifiedTarget.

    Formats:
        - guardrail:no_injection -> GuardrailTarget
        - guardrail:safe -> GuardrailTarget
        - http://localhost:8000 -> HTTPTarget
        - https://api.example.com/chat -> HTTPTarget

    Args:
        target_str: Target specification string

    Returns:
        UnifiedTarget instance

    Raises:
        ValueError: If target format is invalid

    Examples:
        >>> target = parse_target("guardrail:no_injection")
        >>> isinstance(target, GuardrailTarget)
        True

        >>> target = parse_target("http://localhost:8000/chat")
        >>> isinstance(target, HTTPTarget)
        True
    """
    if target_str.startswith("guardrail:"):
        return _parse_guardrail_target(target_str)
    elif target_str.startswith("http://") or target_str.startswith("https://"):
        return _parse_http_target(target_str)
    else:
        raise ValueError(
            f"Invalid target format: {target_str}. "
            "Use 'guardrail:name' or 'http://url'"
        )


def _parse_guardrail_target(target_str: str) -> GuardrailTarget:
    """
    Parse guardrail target string.

    Args:
        target_str: Target string in format 'guardrail:name'

    Returns:
        GuardrailTarget instance

    Raises:
        ValueError: If guardrail name is unknown
    """
    from ...guardrail import GuardrailFunctions

    name = target_str.split(":", 1)[1]
    guardrail_map = {
        "no_injection": GuardrailFunctions.no_injection,
        "no_pii": GuardrailFunctions.no_pii,
        "no_toxicity": GuardrailFunctions.no_toxicity,
        "safe": GuardrailFunctions.safe,
        "safe_input": GuardrailFunctions.safe_input,
        "safe_output": GuardrailFunctions.safe_output,
        "no_mcp_attack": GuardrailFunctions.no_mcp_attack,
        "safe_mcp": GuardrailFunctions.safe_mcp,
    }

    if name not in guardrail_map:
        available = ", ".join(guardrail_map.keys())
        raise ValueError(f"Unknown guardrail: {name}. Available: {available}")

    return GuardrailTarget(guardrail_fn=guardrail_map[name], name=name)


def _parse_http_target(target_str: str) -> HTTPTarget:
    """
    Parse HTTP target string.

    Args:
        target_str: Full URL string

    Returns:
        HTTPTarget instance
    """
    parsed = urlparse(target_str)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    endpoint = parsed.path or "/chat"

    return HTTPTarget(base_url=base_url, endpoint=endpoint)


def format_scan_result(result, output_format: str = "console") -> str:
    """
    Format scan result for output.

    Args:
        result: PluginResult from scanner
        output_format: Output format ('console', 'json', 'html')

    Returns:
        Formatted string
    """
    if output_format == "json":
        return _format_json(result)
    elif output_format == "html":
        return _format_html(result)
    else:
        return _format_console(result)


def _format_json(result) -> str:
    """Format result as JSON."""
    output = {
        "success": result.success,
        "metrics": result.metrics,
        **result.data,
    }
    if result.errors:
        output["errors"] = result.errors
    if result.warnings:
        output["warnings"] = result.warnings
    return json.dumps(output, indent=2)


def _format_console(result) -> str:
    """Format result for console output."""
    lines = []
    lines.append("=" * 60)
    lines.append("SCAN RESULTS")
    lines.append("=" * 60)

    metrics = result.metrics
    lines.append(f"\nSecurity Score: {metrics.get('security_score', 0):.1%}")
    lines.append(f"Attack Success Rate: {metrics.get('attack_success_rate', 0):.1%}")
    lines.append(f"Vulnerabilities Found: {metrics.get('vulnerability_count', 0)}")
    lines.append(f"Total Payloads Tested: {result.data.get('total_payloads', 0)}")

    vulns = result.data.get("vulnerabilities", [])
    if vulns:
        lines.append(f"\n{'-' * 60}")
        lines.append("VULNERABILITIES:")
        lines.append(f"{'-' * 60}")
        for i, vuln in enumerate(vulns[:10], 1):
            lines.append(f"\n[{i}] {vuln.get('category', 'unknown').upper()}")
            lines.append(f"    Severity: {vuln.get('severity', 'unknown')}")
            lines.append(f"    Technique: {vuln.get('technique', 'unknown')}")
            prompt = vuln.get("prompt", "")
            if len(prompt) > 60:
                prompt = prompt[:60] + "..."
            lines.append(f"    Payload: {prompt}")

        if len(vulns) > 10:
            lines.append(f"\n... and {len(vulns) - 10} more vulnerabilities")

    if result.errors:
        lines.append(f"\n{'-' * 60}")
        lines.append("ERRORS:")
        for error in result.errors:
            lines.append(f"  - {error}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def _format_html(result) -> str:
    """Format result as HTML."""
    metrics = result.metrics
    vulns = result.data.get("vulnerabilities", [])

    # Build HTML report
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "  <meta charset='utf-8'>",
        "  <title>Security Scan Report</title>",
        "  <style>",
        "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }",
        "    .header { background: #1a1a2e; color: white; padding: 20px; border-radius: 8px; }",
        "    .metrics { display: flex; gap: 20px; margin: 20px 0; }",
        "    .metric { background: #f0f0f0; padding: 15px; border-radius: 8px; flex: 1; }",
        "    .metric-value { font-size: 24px; font-weight: bold; }",
        "    .metric-label { color: #666; }",
        "    .vuln { background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 10px 0; }",
        "    .vuln.critical { background: #f8d7da; border-color: #dc3545; }",
        "    .vuln.high { background: #f8d7da; border-color: #dc3545; }",
        "    .vuln-title { font-weight: bold; }",
        "    pre { background: #f8f9fa; padding: 10px; overflow-x: auto; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <div class='header'>",
        "    <h1>Security Scan Report</h1>",
        "    <p>dspyGuardrails Security Platform</p>",
        "  </div>",
        "  <div class='metrics'>",
        f"    <div class='metric'><div class='metric-value'>{metrics.get('security_score', 0):.1%}</div><div class='metric-label'>Security Score</div></div>",
        f"    <div class='metric'><div class='metric-value'>{metrics.get('attack_success_rate', 0):.1%}</div><div class='metric-label'>Attack Success Rate</div></div>",
        f"    <div class='metric'><div class='metric-value'>{metrics.get('vulnerability_count', 0)}</div><div class='metric-label'>Vulnerabilities</div></div>",
        f"    <div class='metric'><div class='metric-value'>{result.data.get('total_payloads', 0)}</div><div class='metric-label'>Payloads Tested</div></div>",
        "  </div>",
    ]

    if vulns:
        html_parts.append("  <h2>Vulnerabilities</h2>")
        for vuln in vulns[:20]:
            severity = vuln.get("severity", "unknown")
            severity_class = "critical" if severity in ["critical", "high"] else ""
            html_parts.extend([
                f"  <div class='vuln {severity_class}'>",
                f"    <div class='vuln-title'>[{vuln.get('category', 'unknown').upper()}] {vuln.get('technique', 'unknown')}</div>",
                f"    <p>Severity: {severity}</p>",
                f"    <pre>{vuln.get('prompt', '')[:200]}</pre>",
                "  </div>",
            ])

    html_parts.extend([
        "</body>",
        "</html>",
    ])

    return "\n".join(html_parts)


def format_attack_result(result, output_format: str = "console") -> str:
    """
    Format attack result for output.

    Args:
        result: PluginResult from attacker
        output_format: Output format ('console', 'json', 'html')

    Returns:
        Formatted string
    """
    if output_format == "json":
        return _format_attack_json(result)
    elif output_format == "html":
        return _format_attack_html(result)
    else:
        return _format_attack_console(result)


def _format_attack_json(result) -> str:
    """Format attack result as JSON."""
    output = {
        "success": result.success,
        "metrics": result.metrics,
        **result.data,
    }
    if result.errors:
        output["errors"] = result.errors
    if result.warnings:
        output["warnings"] = result.warnings
    return json.dumps(output, indent=2)


def _format_attack_console(result) -> str:
    """Format attack result for console output."""
    lines = []
    lines.append("=" * 60)
    lines.append("ATTACK RESULTS")
    lines.append("=" * 60)

    metrics = result.metrics
    # Handle both StaticAttacker and LLMAttacker metric naming
    success_rate = metrics.get("success_rate", metrics.get("attack_success_rate", 0))
    successful_attacks = int(metrics.get("successful_attacks", 0))
    total_attacks = int(metrics.get("total_attacks", 0))
    avg_latency = metrics.get("avg_latency_ms", 0)

    lines.append(f"\nSuccess Rate: {success_rate:.1%}")
    lines.append(f"Successful Attacks: {successful_attacks}")
    lines.append(f"Total Attacks: {total_attacks}")
    lines.append(f"Average Latency: {avg_latency:.1f}ms")

    if result.data.get("llm_used"):
        lines.append("LLM Generation: Enabled")

    # Handle both StaticAttacker and LLMAttacker data formats
    attacks = result.data.get("attack_results", result.data.get("attacks", []))
    successful = [a for a in attacks if a.get("success", False)]

    if successful:
        lines.append(f"\n{'─' * 60}")
        lines.append("SUCCESSFUL ATTACKS:")
        lines.append(f"{'─' * 60}")
        for i, atk in enumerate(successful[:10], 1):
            lines.append(f"\n[{i}] {atk.get('category', 'unknown').upper()}")
            lines.append(f"    Technique: {atk.get('technique', 'unknown')}")
            lines.append(f"    Latency: {atk.get('latency_ms', 0):.1f}ms")
            prompt = atk.get("prompt", "")
            if len(prompt) > 60:
                prompt = prompt[:60] + "..."
            lines.append(f"    Payload: {prompt}")

        if len(successful) > 10:
            lines.append(f"\n... and {len(successful) - 10} more successful attacks")
    else:
        lines.append(f"\n{'─' * 60}")
        lines.append("No successful attacks - target appears secure!")
        lines.append(f"{'─' * 60}")

    if result.warnings:
        lines.append(f"\n{'─' * 60}")
        lines.append("WARNINGS:")
        for warning in result.warnings:
            lines.append(f"  - {warning}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def _format_attack_html(result) -> str:
    """Format attack result as HTML."""
    metrics = result.metrics
    success_rate = metrics.get("success_rate", metrics.get("attack_success_rate", 0))
    successful_attacks = int(metrics.get("successful_attacks", 0))
    total_attacks = int(metrics.get("total_attacks", 0))
    avg_latency = metrics.get("avg_latency_ms", 0)

    attacks = result.data.get("attack_results", result.data.get("attacks", []))
    successful = [a for a in attacks if a.get("success", False)]

    # Build HTML report
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "  <meta charset='utf-8'>",
        "  <title>Attack Results Report</title>",
        "  <style>",
        "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }",
        "    .header { background: #dc3545; color: white; padding: 20px; border-radius: 8px; }",
        "    .header.secure { background: #28a745; }",
        "    .metrics { display: flex; gap: 20px; margin: 20px 0; }",
        "    .metric { background: #f0f0f0; padding: 15px; border-radius: 8px; flex: 1; }",
        "    .metric-value { font-size: 24px; font-weight: bold; }",
        "    .metric-label { color: #666; }",
        "    .attack { background: #f8d7da; border-left: 4px solid #dc3545; padding: 15px; margin: 10px 0; }",
        "    .attack-title { font-weight: bold; }",
        "    pre { background: #f8f9fa; padding: 10px; overflow-x: auto; }",
        "  </style>",
        "</head>",
        "<body>",
        f"  <div class='header {'secure' if not successful else ''}'>",
        "    <h1>Attack Results Report</h1>",
        "    <p>dspyGuardrails Security Platform</p>",
        "  </div>",
        "  <div class='metrics'>",
        f"    <div class='metric'><div class='metric-value'>{success_rate:.1%}</div><div class='metric-label'>Success Rate</div></div>",
        f"    <div class='metric'><div class='metric-value'>{successful_attacks}</div><div class='metric-label'>Successful Attacks</div></div>",
        f"    <div class='metric'><div class='metric-value'>{total_attacks}</div><div class='metric-label'>Total Attacks</div></div>",
        f"    <div class='metric'><div class='metric-value'>{avg_latency:.1f}ms</div><div class='metric-label'>Avg Latency</div></div>",
        "  </div>",
    ]

    if successful:
        html_parts.append("  <h2>Successful Attacks</h2>")
        for atk in successful[:20]:
            html_parts.extend([
                "  <div class='attack'>",
                f"    <div class='attack-title'>[{atk.get('category', 'unknown').upper()}] {atk.get('technique', 'unknown')}</div>",
                f"    <p>Latency: {atk.get('latency_ms', 0):.1f}ms</p>",
                f"    <pre>{atk.get('prompt', '')[:200]}</pre>",
                "  </div>",
            ])
    else:
        html_parts.append("  <h2>Target Secure</h2>")
        html_parts.append("  <p>No successful attacks were found. The target appears to be secure against the tested payloads.</p>")

    html_parts.extend([
        "</body>",
        "</html>",
    ])

    return "\n".join(html_parts)
