"""
Security Testing CLI

Command-line interface for running security tests.

Usage:
    python -m tests.security.cli run --config config.yaml
    python -m tests.security.cli run --redteam-only
    python -m tests.security.cli run --url http://localhost:8000
"""

import argparse
import sys
from pathlib import Path

from .runner import SecurityTestRunner, SecurityTestConfig


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Security Testing Framework for OpenAI CS Agents Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests with default config
  python -m tests.security.cli run

  # Run with custom config
  python -m tests.security.cli run --config my_config.yaml

  # Run only red team tests
  python -m tests.security.cli run --redteam-only

  # Run against specific URL
  python -m tests.security.cli run --url http://localhost:8000

  # Verbose output
  python -m tests.security.cli run -v
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run security tests")
    run_parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to YAML config file",
    )
    run_parser.add_argument(
        "--url",
        "-u",
        type=str,
        default="http://localhost:8000",
        help="Target URL (default: http://localhost:8000)",
    )
    run_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./reports/output",
        help="Output directory for reports",
    )
    run_parser.add_argument(
        "--format",
        "-f",
        type=str,
        nargs="+",
        default=["console", "json", "html"],
        choices=["console", "json", "html"],
        help="Report formats to generate",
    )
    run_parser.add_argument(
        "--redteam-only",
        action="store_true",
        help="Run only red team tests",
    )
    run_parser.add_argument(
        "--blueteam-only",
        action="store_true",
        help="Run only blue team tests",
    )
    run_parser.add_argument(
        "--hallucination-only",
        action="store_true",
        help="Run only hallucination tests",
    )
    run_parser.add_argument(
        "--benchmark",
        "-b",
        type=str,
        nargs="+",
        default=["jailbreakbench", "advbench"],
        help="Benchmarks to run (jailbreakbench, advbench, harmbench)",
    )
    run_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate report from results")
    report_parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input JSON results file",
    )
    report_parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="html",
        choices=["console", "json", "html"],
        help="Output format",
    )

    args = parser.parse_args()

    if args.command == "run":
        run_tests(args)
    elif args.command == "report":
        generate_report(args)
    else:
        parser.print_help()
        sys.exit(1)


def run_tests(args):
    """Run security tests."""
    print("=" * 60)
    print("   Security Testing Framework")
    print("   OpenAI CS Agents Demo Evaluation")
    print("=" * 60)
    print()

    # Create config
    if args.config:
        config = SecurityTestConfig.from_yaml(args.config)
    else:
        config = SecurityTestConfig(
            target_base_url=args.url,
            output_dir=args.output,
            report_formats=args.format,
            redteam_benchmarks=args.benchmark,
            verbose=args.verbose,
        )

    # Handle test type flags
    if args.redteam_only:
        config.run_redteam = True
        config.run_blueteam = False
        config.run_hallucination = False
    elif args.blueteam_only:
        config.run_redteam = False
        config.run_blueteam = True
        config.run_hallucination = False
    elif args.hallucination_only:
        config.run_redteam = False
        config.run_blueteam = False
        config.run_hallucination = True

    # Create runner
    runner = SecurityTestRunner(config)

    print(f"Target: {config.target_base_url}")
    print(f"Tests: ", end="")
    tests = []
    if config.run_redteam:
        tests.append("RedTeam")
    if config.run_blueteam:
        tests.append("BlueTeam")
    if config.run_hallucination:
        tests.append("Hallucination")
    print(", ".join(tests))
    print()

    # Run tests
    try:
        results = runner.run_all()
    except Exception as e:
        print(f"Error running tests: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Generate reports
    outputs = runner.generate_report(results)

    print()
    print("-" * 60)
    print("Reports generated:")
    for fmt, path in outputs.items():
        if fmt != "console":
            print(f"  {fmt}: {path}")

    # Exit with appropriate code
    if results.overall_score < 60:
        print()
        print("⚠ Security score below threshold (60). Consider fixing vulnerabilities.")
        sys.exit(1)


def generate_report(args):
    """Generate report from existing results."""
    import json

    with open(args.input) as f:
        data = json.load(f)

    # This would need to reconstruct results from JSON
    # For now, just print a message
    print(f"Report generation from {args.input} not yet implemented.")
    print("Run tests directly to generate reports.")


if __name__ == "__main__":
    main()
