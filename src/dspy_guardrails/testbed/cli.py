"""
CLI for dspyGuardrails Agent Testbed.

This module provides a command-line interface for running testbed evaluations,
initializing configuration files, and listing available resources.
"""

import json
import sys
from pathlib import Path

import click

from dspy_guardrails.testbed.config import (
    AgentComplexity,
    AgentConfig,
    AgentDomain,
    ProtectionLevel,
    TestbedConfig,
)
from dspy_guardrails.testbed.orchestrator import (
    ProgressEvent,
    TestbedOrchestrator,
)
from dspy_guardrails.testbed.results import TestbedResults


def _create_template_config(template: str) -> TestbedConfig:
    """
    Create a TestbedConfig based on template name.

    Args:
        template: Template name - "minimal", "airline", or "default".

    Returns:
        TestbedConfig configured according to the template.
    """
    if template == "minimal":
        # Minimal: 1 simple/general agent, 2 protections, 1 suite, max 10 attacks
        return TestbedConfig(
            agents=[
                AgentConfig(
                    complexity=AgentComplexity.SIMPLE,
                    domain=AgentDomain.GENERAL,
                ),
            ],
            protection_levels=[ProtectionLevel.NONE, ProtectionLevel.FULL],
            attack_suites=["injection"],
            max_attacks_per_suite=10,
            parallel=True,
            max_workers=3,
        )

    elif template == "airline":
        # Airline: 3 CS agents (simple, tools, multi), all protections, 2 suites
        return TestbedConfig(
            agents=[
                AgentConfig(
                    complexity=AgentComplexity.SIMPLE,
                    domain=AgentDomain.CUSTOMER_SERVICE,
                ),
                AgentConfig(
                    complexity=AgentComplexity.TOOLS,
                    domain=AgentDomain.CUSTOMER_SERVICE,
                ),
                AgentConfig(
                    complexity=AgentComplexity.MULTI_AGENT,
                    domain=AgentDomain.CUSTOMER_SERVICE,
                ),
            ],
            protection_levels=[
                ProtectionLevel.NONE,
                ProtectionLevel.PARTIAL,
                ProtectionLevel.FULL,
            ],
            attack_suites=["injection", "jailbreak"],
            max_attacks_per_suite=None,  # Use all payloads
            parallel=True,
            max_workers=5,
        )

    else:  # default
        # Default: 2 agents (simple-cs, simple-general), all protections, 2 suites
        return TestbedConfig(
            agents=[
                AgentConfig(
                    complexity=AgentComplexity.SIMPLE,
                    domain=AgentDomain.CUSTOMER_SERVICE,
                ),
                AgentConfig(
                    complexity=AgentComplexity.SIMPLE,
                    domain=AgentDomain.GENERAL,
                ),
            ],
            protection_levels=[
                ProtectionLevel.NONE,
                ProtectionLevel.PARTIAL,
                ProtectionLevel.FULL,
            ],
            attack_suites=["injection", "jailbreak"],
            max_attacks_per_suite=None,
            parallel=True,
            max_workers=5,
        )


def _print_summary(results: TestbedResults) -> None:
    """
    Print a formatted summary of testbed results.

    Args:
        results: TestbedResults to summarize.
    """
    click.echo("\n" + "=" * 60)
    click.echo("TESTBED EVALUATION SUMMARY")
    click.echo("=" * 60)

    # Overall metrics
    click.echo(f"\nOverall Score: {results.overall_score:.1f}%")
    click.echo(f"Total Tests: {results.total_tests}")
    click.echo(f"Total Attacks: {results.total_attacks}")
    click.echo(f"Total Blocked: {results.total_blocked}")
    click.echo(f"Execution Time: {results.execution_time_seconds:.2f}s")

    # Comparison matrix
    if results.comparison_matrix:
        click.echo("\n" + "-" * 60)
        click.echo("PROTECTION LEVEL COMPARISON")
        click.echo("-" * 60)
        click.echo(
            f"{'Agent':<20} {'None':<10} {'Partial':<10} {'Full':<10} {'Improvement':<12}"
        )
        click.echo("-" * 60)

        for row in results.comparison_matrix:
            click.echo(
                f"{row.agent_name:<20} "
                f"{row.none_block_rate:>8.1%}  "
                f"{row.partial_block_rate:>8.1%}  "
                f"{row.full_block_rate:>8.1%}  "
                f"{row.improvement:>+10.1%}"
            )

    # Critical vulnerabilities
    if results.critical_vulnerabilities:
        click.echo("\n" + "-" * 60)
        click.secho("CRITICAL VULNERABILITIES", fg="red", bold=True)
        click.echo("-" * 60)

        for vuln in results.critical_vulnerabilities:
            click.secho(
                f"  - {vuln['agent_name']}: {vuln['block_rate']:.1%} "
                f"(threshold: {vuln['threshold']:.0%}, bypassed: {vuln['bypassed_count']})",
                fg="red",
            )
    else:
        click.echo("\n" + "-" * 60)
        click.secho("No critical vulnerabilities found.", fg="green")

    click.echo("=" * 60 + "\n")


def _save_results(results: TestbedResults, output_dir: str) -> str:
    """
    Save testbed results to a JSON file.

    Args:
        results: TestbedResults to save.
        output_dir: Directory to save results to.

    Returns:
        Path to the saved results file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"testbed_results_{timestamp}.json"
    filepath = output_path / filename

    # Convert results to dict
    results_dict = {
        "timestamp": results.timestamp,
        "execution_time_seconds": results.execution_time_seconds,
        "overall_score": results.overall_score,
        "total_tests": results.total_tests,
        "total_attacks": results.total_attacks,
        "total_blocked": results.total_blocked,
        "results": [
            {
                "agent_name": r.agent_name,
                "complexity": r.complexity,
                "domain": r.domain,
                "protection_level": r.protection_level,
                "total_attacks": r.total_attacks,
                "blocked_attacks": r.blocked_attacks,
                "bypassed_attacks": r.bypassed_attacks,
                "block_rate": r.block_rate,
                "by_category": r.by_category,
                "bypassed_payloads": r.bypassed_payloads,
                "avg_latency_ms": r.avg_latency_ms,
                "total_time_seconds": r.total_time_seconds,
            }
            for r in results.results
        ],
        "comparison_matrix": [
            {
                "agent_name": row.agent_name,
                "complexity": row.complexity,
                "domain": row.domain,
                "none_block_rate": row.none_block_rate,
                "partial_block_rate": row.partial_block_rate,
                "full_block_rate": row.full_block_rate,
                "improvement": row.improvement,
            }
            for row in results.comparison_matrix
        ],
        "critical_vulnerabilities": results.critical_vulnerabilities,
    }

    with open(filepath, "w") as f:
        json.dump(results_dict, f, indent=2)

    return str(filepath)


def _make_progress_callback(verbose: bool):
    """
    Create a progress callback function.

    Args:
        verbose: Whether to show verbose output.

    Returns:
        Callback function for progress events.
    """

    def callback(event: ProgressEvent) -> None:
        if event.type == "test_start":
            click.echo(
                f"\n[{event.current}/{event.total}] Testing {event.agent_name}..."
            )
        elif event.type == "test_complete":
            click.echo(
                f"  Completed: block rate = {event.block_rate:.1%}"
            )
        elif event.type == "attack_result" and verbose:
            status = click.style("BLOCKED", fg="green") if event.blocked else click.style("BYPASSED", fg="red")
            click.echo(
                f"    [{event.current}/{event.total}] {status}: {event.payload[:50]}..."
            )
        elif event.type == "all_complete":
            click.secho(
                f"\nAll tests complete! Overall block rate: {event.block_rate:.1%}",
                fg="cyan",
                bold=True,
            )

    return callback


@click.group()
@click.version_option(version="0.1.0", prog_name="dspy-testbed")
def cli():
    """dspyGuardrails Agent Testbed - Security evaluation for AI agents."""
    pass


@cli.command()
@click.option(
    "--template",
    type=click.Choice(["default", "airline", "minimal"]),
    default="default",
    help="Configuration template to use.",
)
@click.option(
    "--output",
    "-o",
    default="testbed.yml",
    help="Output path for the configuration file.",
)
def init(template: str, output: str):
    """Initialize a testbed configuration file."""
    click.echo(f"Creating testbed configuration from '{template}' template...")

    config = _create_template_config(template)
    config.save_yaml(output)

    click.secho(f"Configuration saved to: {output}", fg="green")
    click.echo("\nConfiguration summary:")
    click.echo(f"  - Agents: {len(config.agents)}")
    for agent in config.agents:
        click.echo(f"      {agent.name} ({agent.complexity.value}/{agent.domain.value})")
    click.echo(f"  - Protection levels: {[p.value for p in config.protection_levels]}")
    click.echo(f"  - Attack suites: {config.attack_suites}")
    click.echo(f"  - Max attacks per suite: {config.max_attacks_per_suite or 'unlimited'}")
    click.echo(f"\nEdit {output} to customize, then run: dspy-testbed run -c {output}")


@cli.command()
@click.option(
    "--config",
    "-c",
    default="testbed.yml",
    help="Path to the testbed configuration file.",
)
@click.option(
    "--parallel/--no-parallel",
    default=True,
    help="Run tests in parallel.",
)
@click.option(
    "--workers",
    "-w",
    default=5,
    type=int,
    help="Number of parallel workers.",
)
@click.option(
    "--max-attacks",
    "-m",
    type=int,
    default=None,
    help="Override max attacks per suite.",
)
@click.option(
    "--suite",
    "-s",
    multiple=True,
    help="Attack suites to run (can specify multiple).",
)
@click.option(
    "--output-dir",
    "-o",
    default="./testbed_reports",
    help="Directory for output reports.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show verbose output including individual attack results.",
)
def run(
    config: str,
    parallel: bool,
    workers: int,
    max_attacks: int | None,
    suite: tuple,
    output_dir: str,
    verbose: bool,
):
    """Run the testbed evaluation."""
    # Check if config file exists
    config_path = Path(config)
    if not config_path.exists():
        click.secho(f"Error: Configuration file not found: {config}", fg="red")
        click.echo(f"Run 'dspy-testbed init -o {config}' to create one.")
        sys.exit(1)

    click.echo(f"Loading configuration from: {config}")

    # Load config
    testbed_config = TestbedConfig.from_yaml(config)

    # Apply CLI overrides
    testbed_config.parallel = parallel
    testbed_config.max_workers = workers
    testbed_config.output_dir = output_dir

    if max_attacks is not None:
        testbed_config.max_attacks_per_suite = max_attacks

    if suite:
        testbed_config.attack_suites = list(suite)

    # Display configuration
    click.echo("\nTestbed Configuration:")
    click.echo(f"  - Agents: {len(testbed_config.agents)}")
    click.echo(f"  - Protection levels: {len(testbed_config.protection_levels)}")
    click.echo(f"  - Attack suites: {testbed_config.attack_suites}")
    click.echo(f"  - Max attacks/suite: {testbed_config.max_attacks_per_suite or 'unlimited'}")
    click.echo(f"  - Parallel: {testbed_config.parallel} ({testbed_config.max_workers} workers)")

    # Create orchestrator
    orchestrator = TestbedOrchestrator(testbed_config)

    # Create progress callback
    progress_callback = _make_progress_callback(verbose)

    # Run evaluation
    click.echo("\nStarting testbed evaluation...")
    click.echo("-" * 40)

    try:
        results = orchestrator.run(progress_callback=progress_callback)
    except Exception as e:
        click.secho(f"\nError during evaluation: {e}", fg="red")
        sys.exit(1)

    # Print summary
    _print_summary(results)

    # Save results
    results_path = _save_results(results, output_dir)
    click.secho(f"Results saved to: {results_path}", fg="green")


@cli.command("list")
@click.argument(
    "item",
    type=click.Choice(["suites", "agents", "payloads"]),
)
@click.option(
    "--category",
    "-c",
    help="Filter by category (for payloads).",
)
def list_resources(item: str, category: str | None):
    """List available resources."""
    if item == "suites":
        click.echo("\nAvailable Attack Suites:")
        click.echo("-" * 40)
        suites = [
            ("injection", "Prompt injection attacks (50+ payloads)"),
            ("jailbreak", "Jailbreak attacks (30+ payloads)"),
            ("mcp", "MCP protocol attacks (20+ payloads)"),
            ("bypass", "Detection bypass techniques (30+ payloads)"),
        ]
        for suite_name, description in suites:
            click.echo(f"  {suite_name:<12} - {description}")

    elif item == "agents":
        click.echo("\nAgent Complexity Levels:")
        click.echo("-" * 40)
        for complexity in AgentComplexity:
            click.echo(f"  {complexity.value:<12} - {complexity.name}")

        click.echo("\nAgent Domains:")
        click.echo("-" * 40)
        for domain in AgentDomain:
            click.echo(f"  {domain.value:<12} - {domain.name}")

        click.echo("\nProtection Levels:")
        click.echo("-" * 40)
        for level in ProtectionLevel:
            click.echo(f"  {level.value:<12} - {level.name}")

    elif item == "payloads":
        from dspy_guardrails.redteam.payloads import (
            BypassPayloads,
            InjectionPayloads,
            JailbreakPayloads,
            MCPPayloads,
        )
        from dspy_guardrails.redteam.payloads.base import PayloadCategory

        if category:
            # Filter by category
            category_upper = category.upper()
            try:
                cat_enum = PayloadCategory[category_upper]
            except KeyError:
                click.secho(f"Unknown category: {category}", fg="red")
                click.echo(f"Available categories: {[c.value for c in PayloadCategory]}")
                sys.exit(1)

            category_mapping = {
                PayloadCategory.INJECTION: InjectionPayloads,
                PayloadCategory.JAILBREAK: JailbreakPayloads,
                PayloadCategory.MCP: MCPPayloads,
                PayloadCategory.BYPASS: BypassPayloads,
            }

            provider = category_mapping.get(cat_enum)
            if provider:
                payloads = provider.get_all()
                click.echo(f"\n{category.upper()} Payloads ({len(payloads)} total):")
                click.echo("-" * 60)

                # Group by technique
                techniques = {}
                for p in payloads:
                    if p.technique not in techniques:
                        techniques[p.technique] = []
                    techniques[p.technique].append(p)

                for tech, tech_payloads in sorted(techniques.items()):
                    click.echo(f"  {tech}: {len(tech_payloads)} payloads")
            else:
                click.secho(f"No payloads available for category: {category}", fg="yellow")
        else:
            # Show all categories summary
            click.echo("\nPayload Categories:")
            click.echo("-" * 40)

            categories = [
                ("injection", InjectionPayloads.get_all()),
                ("jailbreak", JailbreakPayloads.get_all()),
                ("mcp", MCPPayloads.get_all()),
                ("bypass", BypassPayloads.get_all()),
            ]

            total = 0
            for cat_name, payloads in categories:
                count = len(payloads)
                total += count
                click.echo(f"  {cat_name:<12}: {count:>4} payloads")

            click.echo("-" * 40)
            click.echo(f"  {'Total':<12}: {total:>4} payloads")

            click.echo("\nUse --category/-c to filter by category:")
            click.echo("  dspy-testbed list payloads -c injection")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
