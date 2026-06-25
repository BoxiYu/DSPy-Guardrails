"""Autoresearch CLI — command-line interface for the autoresearch system."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Subcommand: eval
# ---------------------------------------------------------------------------


def cmd_eval(args: argparse.Namespace) -> None:
    """Evaluate a single algorithm file using ResearchHarness."""
    sys.path.insert(0, ".")

    # Load .env so API keys are available even when not exported in the shell
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from dspy_guardrails.autoresearch.registry import load_algorithm
    from dspy_guardrails.autoresearch.harness import ResearchHarness

    algo_path = Path(args.algorithm)
    if not algo_path.exists():
        print(f"[ERROR] Algorithm file not found: {algo_path}", file=sys.stderr)
        sys.exit(1)

    n_behaviors = args.n_behaviors
    query_budget = args.query_budget
    if args.quick:
        n_behaviors = 3
        query_budget = 5
        print("[eval] --quick mode: n_behaviors=3, query_budget=5")

    # Configure attacker LM (local vLLM)
    attacker_lm = None
    try:
        import dspy
        import os
        vllm_base = os.getenv("ATTACK_MODEL_BASE_URL", "http://localhost:18921/v1")
        vllm_key = os.getenv("ATTACK_MODEL_API_KEY", "EMPTY")
        vllm_model = os.getenv("ATTACK_MODEL", "openai/Huihui-Qwen3.5-27B-abliterated")
        attacker_lm = dspy.LM(
            vllm_model, api_key=vllm_key, api_base=vllm_base,
            cache=False, timeout=60.0, max_tokens=4096,
        )
        dspy.configure(lm=attacker_lm)
        print(f"[eval] Attacker LM: {vllm_model} @ {vllm_base}")
    except Exception as exc:
        print(f"[eval] WARN: Could not configure attacker LM: {exc}")

    # Configure judge LM (OpenRouter)
    judge_lm = None
    try:
        judge_model = os.getenv("AUTORESEARCH_JUDGE_MODEL", "openrouter/qwen/qwen3-235b-a22b-2507")
        or_key = os.getenv("OPENROUTER_API_KEY", "")
        if or_key:
            judge_lm = dspy.LM(
                judge_model, api_key=or_key,
                api_base="https://openrouter.ai/api/v1",
                cache=False, timeout=60.0, max_tokens=1024,
            )
            print(f"[eval] Judge LM: {judge_model}")
        else:
            print("[eval] WARN: No OPENROUTER_API_KEY, using keyword heuristic judge")
    except Exception as exc:
        print(f"[eval] WARN: Could not configure judge LM: {exc}")

    print(f"[eval] Loading algorithm from {algo_path} …")
    algo_cls = load_algorithm(algo_path)
    algorithm = algo_cls()

    # Configure target model (victim to be attacked)
    target_model = os.getenv(
        "AUTORESEARCH_TARGET_MODEL",
        "meta-llama/llama-3.3-70b-instruct:free",
    )
    target_provider = os.getenv("AUTORESEARCH_TARGET_PROVIDER", "openrouter")
    guard_mode = getattr(args, "guard_mode", None) or os.getenv(
        "AUTORESEARCH_GUARD_MODE", "none"
    )
    print(f"[eval] Target: {target_model} ({target_provider}), guard_mode={guard_mode}")

    harness = ResearchHarness(
        target_model=target_model,
        target_provider=target_provider,
        guard_mode=guard_mode,
        query_budget=query_budget,
        n_behaviors=n_behaviors,
        seed=args.seed,
        attacker_lm=attacker_lm,
        judge_lm=judge_lm,
    )

    from dspy_guardrails.autoresearch.registry import AttackAlgorithm, DefenseAlgorithm

    if isinstance(algorithm, AttackAlgorithm):
        result = harness.evaluate_attack(algorithm)
        print("\n" + result.summary())
        if args.output:
            result.save(Path(args.output))
            print(f"[eval] Result saved to {args.output}")
    elif isinstance(algorithm, DefenseAlgorithm):
        # For a lone defense eval, run with no attacks — just report structure
        result = harness.evaluate_defense(algorithm, attack_algorithms=[])
        print("\n" + result.summary())
        if args.output:
            result.save(Path(args.output))
            print(f"[eval] Result saved to {args.output}")
    else:
        print(f"[ERROR] Loaded class {algo_cls!r} is not an AttackAlgorithm or DefenseAlgorithm.",
              file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand: status
# ---------------------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> None:
    """Show current research state from AGENT_LOG.md."""
    sys.path.insert(0, ".")

    from dspy_guardrails.autoresearch.memory import AgentMemory

    log_path = Path(args.log_path)
    memory = AgentMemory(log_path=log_path)

    if not log_path.exists():
        print(f"[status] Log not found at {log_path}. Run 'init' first.")
        return

    iteration_count = memory.get_iteration_count()
    best_attacks = memory.get_best_algorithms(kind="attack", top_k=3)
    best_defenses = memory.get_best_algorithms(kind="defense", top_k=3)
    history = memory.load_history()
    recent = history[-5:] if history else []

    print(f"=== Autoresearch Status ===")
    print(f"Log path     : {log_path}")
    print(f"Iterations   : {iteration_count}")
    print()

    print("--- Best Attack Algorithms ---")
    if best_attacks:
        for r in best_attacks:
            asr = r.results.get("asr", 0.0)
            print(f"  {r.algorithm_name}  ASR={asr:.3f}  [{r.status}]")
    else:
        print("  (none recorded)")

    print()
    print("--- Best Defense Algorithms ---")
    if best_defenses:
        for r in best_defenses:
            f1 = r.results.get("f1", 0.0)
            print(f"  {r.algorithm_name}  F1={f1:.3f}  [{r.status}]")
    else:
        print("  (none recorded)")

    print()
    print("--- Recent History (last 5) ---")
    if recent:
        for r in recent:
            print(f"  [{r.iteration:3d}] {r.kind:8s} {r.status:8s}  {r.algorithm_name}")
    else:
        print("  (no history)")


# ---------------------------------------------------------------------------
# Subcommand: report
# ---------------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> None:
    """Generate a results summary from results.tsv."""
    sys.path.insert(0, ".")

    import csv

    tsv_path = Path(args.tsv_path)
    if not tsv_path.exists():
        print(f"[report] TSV not found at {tsv_path}.", file=sys.stderr)
        sys.exit(1)

    with tsv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        rows = list(reader)

    if not rows:
        print("[report] No data in TSV.")
        return

    if args.format == "json":
        print(json.dumps(rows, indent=2))
        return

    if args.format == "latex":
        _print_latex(rows)
        return

    # Default: table
    _print_table(rows)


def _print_table(rows: list[dict]) -> None:
    """Print rows as a plain-text table."""
    if not rows:
        return
    headers = list(rows[0].keys())
    # Compute column widths
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))

    sep = "+-" + "-+-".join("-" * widths[h] for h in headers) + "-+"
    header_line = "| " + " | ".join(h.ljust(widths[h]) for h in headers) + " |"

    print(sep)
    print(header_line)
    print(sep)
    for row in rows:
        line = "| " + " | ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers) + " |"
        print(line)
    print(sep)


def _print_latex(rows: list[dict]) -> None:
    """Print rows as a LaTeX tabular."""
    if not rows:
        return
    headers = list(rows[0].keys())
    col_spec = "l" * len(headers)
    print(r"\begin{tabular}{" + col_spec + "}")
    print(r"\hline")
    print(" & ".join(f"\\textbf{{{h}}}" for h in headers) + r" \\")
    print(r"\hline")
    for row in rows:
        print(" & ".join(str(row.get(h, "")) for h in headers) + r" \\")
    print(r"\hline")
    print(r"\end{tabular}")


# ---------------------------------------------------------------------------
# Subcommand: list
# ---------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> None:
    """List registered algorithms."""
    sys.path.insert(0, ".")

    from dspy_guardrails.autoresearch.registry import list_algorithms

    # Attempt to load seed algorithms from the standard location
    seed_dir = Path("autoresearch/methods")
    if seed_dir.exists():
        from dspy_guardrails.autoresearch.registry import load_algorithm
        for algo_file in sorted(seed_dir.rglob("*.py")):
            try:
                load_algorithm(algo_file)
            except Exception:  # noqa: BLE001
                pass  # Skip files that fail to load

    kind = args.kind
    algorithms = list_algorithms(kind=kind)

    if not algorithms:
        print(f"No {kind} algorithms registered.")
        return

    print(f"=== Registered Algorithms [{kind}] ===")
    for info in sorted(algorithms, key=lambda i: (i.kind, i.name)):
        print(f"  {info.kind:8s}  v{info.version}  {info.name}")
        if info.description:
            print(f"             {info.description}")
        print(f"             {info.module_path}")


# ---------------------------------------------------------------------------
# Subcommand: init
# ---------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> None:
    """Initialize a new autoresearch run."""
    sys.path.insert(0, ".")

    from dspy_guardrails.autoresearch.memory import init_log

    base_dir = Path("autoresearch")
    dirs_to_create = [
        base_dir,
        base_dir / "methods" / "v0",
        base_dir / "results",
    ]
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
        print(f"[init] Created directory: {d}")

    log_path = base_dir / "AGENT_LOG.md"
    if log_path.exists() and not args.force:
        print(f"[init] {log_path} already exists. Use --force to overwrite.")
    else:
        init_log(log_path, goal=args.goal)
        print(f"[init] Wrote {log_path}")

    tsv_path = base_dir / "results.tsv"
    if not tsv_path.exists():
        tsv_path.touch()
        print(f"[init] Created empty {tsv_path}")

    print(f"[init] Autoresearch run initialised. Goal: {args.goal!r}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dspy_guardrails.autoresearch.cli",
        description="Autoresearch CLI — autonomous algorithm discovery for LLM security.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # --- eval ---
    p_eval = subparsers.add_parser("eval", help="Evaluate a single algorithm file.")
    p_eval.add_argument("algorithm", help="Path to the algorithm Python file.")
    p_eval.add_argument(
        "--query-budget", type=int, default=20, metavar="N",
        help="Maximum queries (attack iterations) per behavior (default: 20).",
    )
    p_eval.add_argument(
        "--n-behaviors", type=int, default=20, metavar="N",
        help="Number of behaviors to sample from JBB100 (default: 20).",
    )
    p_eval.add_argument(
        "--seed", type=int, default=42, metavar="N",
        help="Random seed for reproducible behavior sampling (default: 42).",
    )
    p_eval.add_argument(
        "--output", metavar="PATH", default=None,
        help="Save the JSON result to this path.",
    )
    p_eval.add_argument(
        "--quick", action="store_true",
        help="Fast test mode: sets n_behaviors=3, query_budget=5.",
    )
    p_eval.add_argument(
        "--guard-mode", default=None,
        choices=["none", "input", "output", "both"],
        help="Guardrail placement: none, input, output, or both (default: env or none).",
    )
    p_eval.set_defaults(func=cmd_eval)

    # --- status ---
    p_status = subparsers.add_parser("status", help="Show current research state.")
    p_status.add_argument(
        "--log-path", default="autoresearch/AGENT_LOG.md", metavar="PATH",
        help="Path to AGENT_LOG.md (default: autoresearch/AGENT_LOG.md).",
    )
    p_status.set_defaults(func=cmd_status)

    # --- report ---
    p_report = subparsers.add_parser("report", help="Generate results summary.")
    p_report.add_argument(
        "--format", choices=["table", "json", "latex"], default="table",
        help="Output format (default: table).",
    )
    p_report.add_argument(
        "--tsv-path", default="autoresearch/results.tsv", metavar="PATH",
        help="Path to results.tsv (default: autoresearch/results.tsv).",
    )
    p_report.set_defaults(func=cmd_report)

    # --- list ---
    p_list = subparsers.add_parser("list", help="List registered algorithms.")
    p_list.add_argument(
        "--kind", choices=["all", "attack", "defense"], default="all",
        help="Filter by algorithm kind (default: all).",
    )
    p_list.set_defaults(func=cmd_list)

    # --- init ---
    p_init = subparsers.add_parser("init", help="Initialize a new autoresearch run.")
    p_init.add_argument(
        "--goal", default="Discover novel LLM security algorithms", metavar="GOAL",
        help="Research goal to embed in AGENT_LOG.md.",
    )
    p_init.add_argument(
        "--force", action="store_true",
        help="Overwrite existing AGENT_LOG.md if present.",
    )
    p_init.set_defaults(func=cmd_init)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    sys.path.insert(0, ".")
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
