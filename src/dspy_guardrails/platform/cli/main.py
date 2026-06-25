"""CLI entry point for dspyGuardrails Security Platform.

This module provides the command-line interface for running security tests,
attacks, and evaluations against AI systems using dspyGuardrails.

Example:
    # Quick scan a guardrail function
    $ dspy-guardrails scan --target guardrail:no_injection

    # Attack an HTTP agent
    $ dspy-guardrails attack --target http://localhost:8000/chat

    # Run full evaluation from config
    $ dspy-guardrails run -c security.yaml
"""

import sys

import click

__version__ = "0.3.0"


class Context:
    """CLI context object for sharing state between commands.

    This context is passed through the command hierarchy using click's
    make_pass_decorator pattern.

    Attributes:
        verbose: Enable verbose output
        config_path: Path to YAML configuration file
        output_format: Output format (console, json, html)
    """

    def __init__(self):
        self.verbose: bool = False
        self.config_path: str | None = None
        self.output_format: str = "console"


pass_context = click.make_pass_decorator(Context, ensure=True)


@click.group()
@click.version_option(version=__version__, prog_name="dspy-guardrails")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(exists=False),
    help="Path to YAML configuration file",
)
@click.option(
    "-o",
    "--output",
    "output_format",
    type=click.Choice(["console", "json", "html"]),
    default="console",
    help="Output format",
)
@pass_context
def cli(ctx: Context, verbose: bool, config_path: str | None, output_format: str):
    """dspyGuardrails Security Platform - AI Security Testing CLI

    Run security scans, attacks, and evaluations against AI systems.

    \b
    Examples:
        # Quick scan a guardrail function
        dspy-guardrails scan --target guardrail:no_injection

        # Attack an HTTP agent
        dspy-guardrails attack --target http://localhost:8000/chat

        # Run full evaluation from config
        dspy-guardrails run -c security.yaml
    """
    ctx.verbose = verbose
    ctx.config_path = config_path
    ctx.output_format = output_format


@cli.command()
@click.option(
    "--target",
    "-t",
    required=True,
    help="Target to scan (guardrail:name or URL)",
)
@click.option(
    "--max-payloads",
    "-n",
    default=20,
    type=int,
    help="Maximum payloads to test",
)
@click.option(
    "--categories",
    "-cat",
    multiple=True,
    help="Attack categories (injection, jailbreak, bypass)",
)
@click.option(
    "--severity",
    "-s",
    default="medium",
    type=click.Choice(["low", "medium", "high", "critical"]),
    help="Minimum severity level",
)
@pass_context
def scan(ctx: Context, target: str, max_payloads: int, categories: tuple, severity: str):
    """Run a quick security scan against a target.

    \b
    Examples:
        dspy-guardrails scan -t guardrail:no_injection
        dspy-guardrails scan -t http://localhost:8000 -n 50
        dspy-guardrails scan -t guardrail:safe -cat injection -cat jailbreak

    \b
    Target Formats:
        - guardrail:name - Test a built-in guardrail function
        - http://url     - Test an HTTP endpoint

    \b
    Available Guardrails:
        - no_injection   - Prompt injection detection
        - no_pii         - PII detection
        - no_toxicity    - Toxicity detection
        - safe           - Combined safety check
        - safe_input     - Input safety check
        - safe_output    - Output safety check
        - no_mcp_attack  - MCP attack detection
        - safe_mcp       - MCP safety check
    """
    from ..plugins import PluginConfig
    from ..scanners import QuickScanner
    from .utils import format_scan_result, parse_target

    if ctx.verbose:
        click.echo(f"[VERBOSE] Output format: {ctx.output_format}")
        if ctx.config_path:
            click.echo(f"[VERBOSE] Config file: {ctx.config_path}")
        click.echo(f"[VERBOSE] Scanning target: {target}")
        click.echo(f"[VERBOSE] Max payloads: {max_payloads}")
        click.echo(f"[VERBOSE] Severity filter: {severity}")
        if categories:
            click.echo(f"[VERBOSE] Categories: {', '.join(categories)}")

    try:
        # Parse target
        unified_target = parse_target(target)

        # Configure scanner
        scanner = QuickScanner()
        scanner.configure(PluginConfig(options={
            "max_payloads": max_payloads,
            "categories": list(categories) if categories else ["injection", "jailbreak"],
            "severity_filter": severity,
        }))

        # Execute scan
        click.echo(f"Scanning {target}...")
        result = scanner.execute({"target": unified_target})

        # Output results
        output = format_scan_result(result, ctx.output_format)
        click.echo(output)

        # Exit code based on vulnerabilities
        if result.data.get("vulnerabilities"):
            sys.exit(1)  # Vulnerabilities found
        sys.exit(0)

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)
    except Exception as e:
        click.echo(f"Scan failed: {e}", err=True)
        if ctx.verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(2)


@cli.command()
@click.option(
    "--target",
    "-t",
    required=True,
    help="Target to attack",
)
@click.option(
    "--attack-type",
    "-a",
    multiple=True,
    help="Attack types (injection, jailbreak, bypass)",
)
@click.option(
    "--budget",
    "-b",
    default=100,
    type=int,
    help="Attack budget (number of attacks)",
)
@click.option(
    "--use-llm/--no-llm",
    default=False,
    help="Use LLM for attack generation",
)
@click.option(
    "--stop-on-success",
    is_flag=True,
    help="Stop after first successful attack",
)
@pass_context
def attack(ctx: Context, target: str, attack_type: tuple, budget: int,
           use_llm: bool, stop_on_success: bool):
    """Execute attacks against a target.

    \b
    Examples:
        dspy-guardrails attack -t http://localhost:8000 -a injection
        dspy-guardrails attack -t guardrail:safe --use-llm -b 50
        dspy-guardrails attack -t guardrail:no_injection --stop-on-success

    \b
    Attack Types:
        - injection  - Prompt injection attacks
        - jailbreak  - Jailbreak attacks (roleplay, hypothetical, etc.)
        - bypass     - Guardrail bypass techniques
        - mcp        - MCP protocol attacks
    """
    from ..attackers import LLMAttacker, StaticAttacker
    from ..plugins import PluginConfig
    from .utils import format_attack_result, parse_target

    if ctx.verbose:
        click.echo(f"[VERBOSE] Output format: {ctx.output_format}")
        click.echo(f"[VERBOSE] Attacking target: {target}")
        click.echo(f"[VERBOSE] Attack types: {', '.join(attack_type) if attack_type else 'all'}")
        click.echo(f"[VERBOSE] Budget: {budget}")
        click.echo(f"[VERBOSE] Use LLM: {use_llm}")
        click.echo(f"[VERBOSE] Stop on success: {stop_on_success}")

    try:
        # Parse target
        unified_target = parse_target(target)

        # Choose attacker based on --use-llm flag
        if use_llm:
            attacker = LLMAttacker()
            attacker.configure(PluginConfig(options={
                "attack_types": list(attack_type) if attack_type else ["injection", "jailbreak"],
                "num_attacks": budget,
                "fallback_to_static": True,
            }))
        else:
            attacker = StaticAttacker()
            attacker.configure(PluginConfig(options={
                "categories": list(attack_type) if attack_type else ["injection", "jailbreak", "bypass"],
                "attack_budget": budget,
                "stop_on_success": stop_on_success,
            }))

        # Execute attacks
        click.echo(f"Attacking {target}...")
        result = attacker.execute({"target": unified_target})

        # Output results
        output = format_attack_result(result, ctx.output_format)
        click.echo(output)

        # Exit code based on success rate (vulnerabilities found = exit 1)
        success_rate = result.metrics.get("success_rate", result.metrics.get("attack_success_rate", 0))
        if success_rate > 0:
            sys.exit(1)  # Successful attacks found (vulnerabilities)
        sys.exit(0)

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)
    except Exception as e:
        click.echo(f"Attack failed: {e}", err=True)
        if ctx.verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(2)


@cli.command()
@click.option(
    "--config",
    "-c",
    "config_file",
    type=click.Path(exists=False),
    help="YAML configuration file",
)
@click.option(
    "--target",
    "-t",
    help="Target override (guardrail:name or URL)",
)
@click.option(
    "--init-config",
    is_flag=True,
    help="Generate sample configuration file",
)
@click.option(
    "--validate",
    "validate_only",
    is_flag=True,
    help="Validate configuration file without running",
)
@pass_context
def run(ctx: Context, config_file: str | None, target: str | None,
        init_config: bool, validate_only: bool):
    """Run full security evaluation from configuration.

    \b
    Examples:
        dspy-guardrails run -c security.yaml
        dspy-guardrails run -c config.yaml -t http://localhost:8000
        dspy-guardrails run --init-config > security.yaml
        dspy-guardrails run -c security.yaml --validate

    \b
    Configuration File Format (YAML):
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
    from ..attackers import LLMAttacker, StaticAttacker
    from ..plugins import PluginConfig
    from ..scanners import QuickScanner
    from .utils import format_attack_result, format_scan_result, parse_target
    from .yaml_config import SecurityConfig, create_sample_config, validate_config_file

    # Generate sample config if requested
    if init_config:
        click.echo(create_sample_config())
        return

    # Get config file path
    config_path = config_file or ctx.config_path
    if not config_path:
        click.echo("Error: No configuration file specified", err=True)
        click.echo("Use -c/--config to specify a YAML config file", err=True)
        click.echo("Or use --init-config to generate a sample config", err=True)
        sys.exit(2)

    # Validate only mode
    if validate_only:
        is_valid, error = validate_config_file(config_path)
        if is_valid:
            click.echo(f"Configuration file is valid: {config_path}")
            sys.exit(0)
        else:
            click.echo(f"Configuration file is invalid: {error}", err=True)
            sys.exit(2)

    try:
        # Load configuration
        config = SecurityConfig.from_yaml(config_path)

        if ctx.verbose:
            click.echo(f"[VERBOSE] Loaded config from: {config_path}")
            click.echo(f"[VERBOSE] Target type: {config.target.type}")
            click.echo(f"[VERBOSE] Target value: {config.target.value}")
            click.echo(f"[VERBOSE] Scan enabled: {config.scan.enabled}")
            click.echo(f"[VERBOSE] Attack enabled: {config.attack.enabled}")

        # Determine target string
        if target:
            # Override from command line
            target_str = target
        else:
            # Use config target
            target_str = config.target.to_target_string()

        # Parse unified target
        unified_target = parse_target(target_str)

        # Display header
        display_name = config.target.name or target_str
        click.echo(f"Running security evaluation: {display_name}")
        click.echo("=" * 60)

        results = {"scan": None, "attack": None}
        has_vulnerabilities = False
        phase_count = int(config.scan.enabled) + int(config.attack.enabled)
        current_phase = 0

        # Run scan if enabled
        if config.scan.enabled:
            current_phase += 1
            click.echo(f"\n[{current_phase}/{phase_count}] Running security scan...")

            scanner = QuickScanner()
            scanner.configure(PluginConfig(options={
                "max_payloads": config.scan.max_payloads,
                "categories": config.scan.categories,
                "severity_filter": config.scan.severity,
            }))

            results["scan"] = scanner.execute({"target": unified_target})

            if results["scan"].data.get("vulnerabilities"):
                has_vulnerabilities = True

            # Output scan results
            output = format_scan_result(results["scan"], ctx.output_format)
            click.echo(output)

        # Run attack if enabled
        if config.attack.enabled:
            current_phase += 1
            click.echo(f"\n[{current_phase}/{phase_count}] Running attack evaluation...")

            if config.attack.use_llm:
                attacker = LLMAttacker()
                attacker.configure(PluginConfig(options={
                    "attack_types": config.attack.types,
                    "num_attacks": config.attack.budget,
                    "fallback_to_static": True,
                }))
            else:
                attacker = StaticAttacker()
                attacker.configure(PluginConfig(options={
                    "categories": config.attack.types,
                    "attack_budget": config.attack.budget,
                    "stop_on_success": config.attack.stop_on_success,
                }))

            results["attack"] = attacker.execute({"target": unified_target})

            # Check for successful attacks
            success_rate = results["attack"].metrics.get(
                "success_rate",
                results["attack"].metrics.get("attack_success_rate", 0)
            )
            if success_rate > 0:
                has_vulnerabilities = True

            # Output attack results
            output = format_attack_result(results["attack"], ctx.output_format)
            click.echo(output)

        # Final summary
        click.echo("\n" + "=" * 60)
        click.echo("EVALUATION COMPLETE")
        click.echo("=" * 60)

        # Summary statistics
        if results["scan"]:
            scan_vulns = results["scan"].data.get("vulnerabilities", [])
            click.echo(f"\nScan: {len(scan_vulns)} vulnerabilities found")

        if results["attack"]:
            attack_success = results["attack"].metrics.get(
                "successful_attacks",
                results["attack"].metrics.get("success_rate", 0) * results["attack"].metrics.get("total_attacks", 0)
            )
            total_attacks = results["attack"].metrics.get("total_attacks", 0)
            click.echo(f"Attack: {int(attack_success)}/{int(total_attacks)} successful attacks")

        if has_vulnerabilities:
            click.echo("\n[!] Vulnerabilities detected!")
            sys.exit(1)
        else:
            click.echo("\n[OK] No vulnerabilities found.")
            sys.exit(0)

    except FileNotFoundError:
        click.echo(f"Error: Config file not found: {config_path}", err=True)
        sys.exit(2)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)
    except Exception as e:
        click.echo(f"Evaluation failed: {e}", err=True)
        if ctx.verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(2)


@cli.command()
@click.option("--target", "-t", required=True, help="Target (guardrail:name or URL)")
@click.option("--budget", "-b", default=30, type=int, help="Max attack attempts")
@click.option(
    "--mode", default="state_machine",
    type=click.Choice(["state_machine", "react"]),
    help="Agent orchestration mode",
)
@click.option(
    "--strategy", default="balanced",
    type=click.Choice(["aggressive", "balanced", "stealth"]),
    help="Attack intensity profile",
)
@click.option("--categories", "-cat", multiple=True, help="Attack categories")
@click.option("--output-dir", "-o", default="./pentest_reports", help="Report output directory")
@pass_context
def adaptive(ctx: Context, target: str, budget: int, mode: str, strategy: str,
             categories: tuple, output_dir: str):
    """Run adaptive LLM-driven penetration test.

    \b
    Examples:
        dspy-guardrails adaptive -t guardrail:no_injection --budget 20
        dspy-guardrails adaptive -t http://localhost:9000 --mode react
        dspy-guardrails adaptive -t guardrail:safe --strategy aggressive --budget 50

    \b
    Strategies:
        - aggressive: High budget, all categories, fast adaptation
        - balanced:   Default settings
        - stealth:    Lower budget, more evolution, less direct attack
    """
    from ...redteam.pentest.agent import PentestAgent
    from ...redteam.pentest.config import PentestAgentConfig
    from .utils import parse_target

    if ctx.verbose:
        click.echo(f"[VERBOSE] Target: {target}")
        click.echo(f"[VERBOSE] Mode: {mode}")
        click.echo(f"[VERBOSE] Strategy: {strategy}")
        click.echo(f"[VERBOSE] Budget: {budget}")

    try:
        unified_target = parse_target(target)

        # Strategy presets
        cats = list(categories) if categories else ["injection", "jailbreak", "bypass"]
        config_kwargs = {
            "max_attempts": budget,
            "categories": cats,
            "enable_recon": True,
            "enable_adaptation": True,
            "enable_multi_turn": True,
            "use_llm_evaluation": True,
            "verbose": ctx.verbose,
            "output_dir": output_dir,
        }

        if strategy == "aggressive":
            config_kwargs["max_attempts"] = budget * 2
            config_kwargs["categories"] = ["injection", "jailbreak", "bypass", "mcp"]
            config_kwargs["adaptation_threshold"] = 3
            config_kwargs["multi_turn_threshold"] = 5
        elif strategy == "stealth":
            config_kwargs["max_attempts"] = max(budget // 2, 5)
            config_kwargs["adaptation_threshold"] = 8

        agent_config = PentestAgentConfig(**config_kwargs)
        agent = PentestAgent(unified_target, agent_config, mode=mode)

        click.echo(f"Running adaptive pentest ({mode} mode, {strategy} strategy)...")
        click.echo(f"Target: {target} | Budget: {agent_config.max_attempts}")
        click.echo("=" * 60)

        report = agent.run()
        paths = agent.save_report(report, output_dir)

        # Print summary
        click.echo("\n" + "=" * 60)
        click.echo("ADAPTIVE PENTEST COMPLETE")
        click.echo("=" * 60)
        report.print_summary()

        click.echo("\nReports saved to:")
        for fmt, path in paths.items():
            click.echo(f"  {fmt}: {path}")

        if report.vulnerabilities:
            sys.exit(1)
        sys.exit(0)

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)
    except Exception as e:
        click.echo(f"Adaptive pentest failed: {e}", err=True)
        if ctx.verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(2)


@cli.command()
@click.option(
    "--report-dir",
    "-d",
    default=".",
    type=click.Path(exists=True),
    help="Directory containing JSON report files",
)
@click.option(
    "--port",
    "-p",
    default=8501,
    type=int,
    help="Port for the Streamlit server",
)
def viz(report_dir: str, port: int):
    """Launch the interactive security dashboard.

    \b
    Examples:
        dspy-guardrails viz --report-dir ./reports
        dspy-guardrails viz -d tests/security/reports/output/ -p 8502
    """
    import subprocess
    from pathlib import Path

    try:
        import streamlit  # noqa: F401
    except ImportError:
        click.echo("Error: streamlit is not installed.", err=True)
        click.echo("Install with: pip install dspy-guardrails[viz]", err=True)
        sys.exit(2)

    app_path = Path(__file__).resolve().parent.parent.parent / "dspy_guardrails" / "viz" / "app.py"
    # Resolve from package
    from dspy_guardrails.viz import __file__ as viz_init
    app_path = str(Path(viz_init).parent / "app.py")

    click.echo(f"Starting dashboard on port {port}...")
    click.echo(f"Report directory: {report_dir}")

    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.port", str(port),
            "--server.headless", "true",
            "--", report_dir,
        ],
        check=False,
    )


@cli.command()
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind")  # noqa: S104
@click.option("--port", "-p", default=8000, type=int, help="Port to bind")
@click.option("--workers", "-w", default=1, type=int, help="Number of workers")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve(host: str, port: int, workers: int, reload: bool):
    """Start the guardrail HTTP server.

    \b
    Examples:
        dspy-guardrails serve
        dspy-guardrails serve --port 9000 --workers 4
        dspy-guardrails serve --reload

    \b
    Endpoints:
        POST /v1/check       - Single text check
        POST /v1/check/batch - Batch check
        POST /v1/score       - Risk scores
        GET  /v1/health      - Health check
        GET  /v1/config      - Configuration
        GET  /docs           - OpenAPI docs
    """
    try:
        import uvicorn
    except ImportError:
        click.echo("Error: uvicorn is not installed.", err=True)
        click.echo("Install with: pip install dspy-guardrails[server]", err=True)
        sys.exit(2)

    click.echo(f"Starting guardrail server on {host}:{port}...")
    click.echo(f"API docs: http://{host}:{port}/docs")

    uvicorn.run(
        "dspy_guardrails.server.app:create_app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        factory=True,
    )


def main():
    """Main entry point for the CLI."""
    # Register promptfoo commands
    try:
        from dspy_guardrails.promptfoo.cli import register_commands
        register_commands(cli)
    except ImportError:
        pass  # promptfoo module not available

    cli()


if __name__ == "__main__":
    main()
