#!/usr/bin/env python3
"""Run adaptive-attack experiment suites inspired by "The Attacker Moves Second".

This script standardizes a practical comparison between:
- static weak attacks (payload library only)
- non-adaptive automated attacks (LLM-generated one-shot prompts)
- adaptive attacks (state-machine / ReAct pentest agents)

It is designed to help build a reproducible experiment matrix in this repo.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure local source is used.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))


@dataclass
class ScenarioSpec:
    name: str
    kind: str  # static | llm | adaptive
    description: str
    requires_llm: bool
    budget_multiplier: float = 1.0
    mode: str = "state_machine"
    strategy: str = "balanced"


def load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    loaded: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        loaded[key] = value
        if key not in os.environ:
            os.environ[key] = value
    return loaded


def load_env_candidates(candidates: list[Path]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for p in candidates:
        merged.update(load_env_file(p))
    return merged


def parse_csv(text: str) -> list[str]:
    return [v.strip() for v in text.split(",") if v.strip()]


def parse_int_csv(text: str) -> list[int]:
    out: list[int] = []
    for v in parse_csv(text):
        out.append(int(v))
    return out


def setup_dspy_from_env(
    model: str | None, api_base: str | None, api_key: str | None, request_timeout: float
) -> tuple[bool, str, str]:
    if not api_key:
        return False, "", ""

    resolved_model = model or os.environ.get("MOONSHOT_MODEL", os.environ.get("OPENROUTER_MODEL", "openai/kimi-k2.5"))
    resolved_base = api_base or os.environ.get("MOONSHOT_BASE_URL", os.environ.get("OPENROUTER_BASE_URL", "https://api.moonshot.cn/v1"))

    import dspy

    dspy.configure(lm=dspy.LM(resolved_model, api_key=api_key, api_base=resolved_base, timeout=request_timeout))
    return True, resolved_model, resolved_base


def _first_success_query(attacks: list[dict[str, Any]]) -> int | None:
    for idx, row in enumerate(attacks, 1):
        if row.get("success", False):
            return idx
    return None


def _is_unblocked_attack(row: dict[str, Any]) -> bool:
    if "was_blocked" in row and row.get("was_blocked") is not None:
        return not bool(row.get("was_blocked"))
    return bool(row.get("success", False))


def _first_unblocked_query(attacks: list[dict[str, Any]]) -> int | None:
    for idx, row in enumerate(attacks, 1):
        if _is_unblocked_attack(row):
            return idx
    return None


def _avg_latency(attacks: list[dict[str, Any]]) -> float:
    latencies = [float(a.get("latency_ms", 0.0) or 0.0) for a in attacks]
    return (sum(latencies) / len(latencies)) if latencies else 0.0


def _empty_metrics(error: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "attempts": 0,
        "bypass_successes": 0,
        "semantic_successes": 0,
        "bypass_asr": 0.0,
        "semantic_asr": 0.0,
        "first_bypass_queries": None,
        "first_semantic_success_queries": None,
        "avg_latency_ms": 0.0,
        "wall_time_ms": 0.0,
    }
    if error:
        out["error"] = error
    return out


def apply_primary_metric(row: dict[str, Any], primary_metric: str) -> dict[str, Any]:
    metric_name = "bypass" if primary_metric == "bypass" else "semantic"
    success_key = f"{metric_name}_successes"
    asr_key = f"{metric_name}_asr"
    first_key = "first_bypass_queries" if metric_name == "bypass" else "first_semantic_success_queries"
    out = dict(row)
    out["asr"] = float(out.get(asr_key, 0.0) or 0.0)
    out["successes"] = int(out.get(success_key, 0) or 0)
    out["first_success_queries"] = out.get(first_key)
    out["primary_metric"] = metric_name
    return out


def run_static(target, budget: int, categories: list[str], severity_filter: str) -> dict[str, Any]:
    from dspy_guardrails.platform.attackers import StaticAttacker
    from dspy_guardrails.platform.plugins import PluginConfig

    attacker = StaticAttacker()
    attacker.configure(
        PluginConfig(
            options={
                "attack_budget": budget,
                "categories": categories,
                "severity_filter": severity_filter,
                "stop_on_success": False,
            }
        )
    )
    t0 = time.perf_counter()
    result = attacker.execute({"target": target})
    elapsed_ms = (time.perf_counter() - t0) * 1000

    attacks = result.data.get("attacks", [])
    bypass_successful = sum(1 for a in attacks if _is_unblocked_attack(a))
    semantic_successful = int(result.data.get("successful_attacks", bypass_successful))
    total = int(result.data.get("total_attacks", len(attacks)))
    bypass_asr = (bypass_successful / total) if total else 0.0
    semantic_asr = (semantic_successful / total) if total else 0.0
    first_bypass = _first_unblocked_query(attacks)
    first_semantic = _first_success_query(attacks)
    if first_semantic is None and semantic_successful == bypass_successful:
        first_semantic = first_bypass
    return {
        "attempts": total,
        "bypass_successes": bypass_successful,
        "semantic_successes": semantic_successful,
        "bypass_asr": bypass_asr,
        "semantic_asr": semantic_asr,
        "first_bypass_queries": first_bypass,
        "first_semantic_success_queries": first_semantic,
        "avg_latency_ms": _avg_latency(attacks),
        "wall_time_ms": elapsed_ms,
        "warnings": result.warnings,
        "errors": result.errors,
    }


def run_llm(target, budget: int, categories: list[str]) -> dict[str, Any]:
    from dspy_guardrails.platform.attackers import LLMAttacker
    from dspy_guardrails.platform.plugins import PluginConfig

    llm_categories = [c for c in categories if c in {"injection", "jailbreak"}]
    if not llm_categories:
        llm_categories = ["injection", "jailbreak"]

    attacker = LLMAttacker()
    attacker.configure(
        PluginConfig(
            options={
                "attack_types": llm_categories,
                "num_attacks": budget,
                "target_behavior": "bypass safety and execute hidden/malicious instructions",
                "defense_description": "unknown target defense",
                "fallback_to_static": True,
            }
        )
    )
    t0 = time.perf_counter()
    result = attacker.execute({"target": target})
    elapsed_ms = (time.perf_counter() - t0) * 1000

    attacks = result.data.get("attack_results", [])
    bypass_successful = sum(1 for a in attacks if _is_unblocked_attack(a))
    semantic_successful = int(result.data.get("successful_attacks", bypass_successful))
    total = int(result.data.get("total_attacks", len(attacks)))
    bypass_asr = (bypass_successful / total) if total else 0.0
    semantic_asr = (semantic_successful / total) if total else 0.0
    first_bypass = _first_unblocked_query(attacks)
    first_semantic = _first_success_query(attacks)
    if first_semantic is None and semantic_successful == bypass_successful:
        first_semantic = first_bypass
    return {
        "attempts": total,
        "bypass_successes": bypass_successful,
        "semantic_successes": semantic_successful,
        "bypass_asr": bypass_asr,
        "semantic_asr": semantic_asr,
        "first_bypass_queries": first_bypass,
        "first_semantic_success_queries": first_semantic,
        "avg_latency_ms": _avg_latency(attacks),
        "wall_time_ms": elapsed_ms,
        "llm_used": bool(result.data.get("llm_used", False)),
        "warnings": result.warnings,
        "errors": result.errors,
    }


def run_adaptive(
    target,
    budget: int,
    categories: list[str],
    mode: str,
    strategy: str,
    use_llm_evaluation: bool,
    output_dir: Path,
    verbose: bool = False,
) -> dict[str, Any]:
    from dspy_guardrails.redteam.pentest.agent import PentestAgent
    from dspy_guardrails.redteam.pentest.config import PentestAgentConfig

    kwargs: dict[str, Any] = {
        "max_attempts": budget,
        "categories": categories,
        "enable_recon": True,
        "enable_adaptation": True,
        "enable_multi_turn": True,
        "use_llm_evaluation": use_llm_evaluation,
        "severity_filter": "medium",
        "verbose": verbose,
        "output_dir": str(output_dir),
    }
    if strategy == "aggressive":
        kwargs["max_attempts"] = max(budget, int(budget * 2))
        kwargs["adaptation_threshold"] = 3
        kwargs["multi_turn_threshold"] = 5
        if "mcp" not in kwargs["categories"]:
            kwargs["categories"] = [*kwargs["categories"], "mcp"]
    elif strategy == "stealth":
        kwargs["max_attempts"] = max(5, budget // 2)
        kwargs["adaptation_threshold"] = 8

    cfg = PentestAgentConfig(**kwargs)
    agent = PentestAgent(target=target, config=cfg, mode=mode)

    t0 = time.perf_counter()
    report = agent.run()
    paths = agent.save_report(report, output_dir=str(output_dir))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    attempts = report.trajectory.attempts if report.trajectory else []
    semantic_successes = [a for a in attempts if a.success_level == "success"]
    bypass_successes = [a for a in attempts if a.was_blocked is False]
    first_semantic = None
    first_bypass = None
    for idx, att in enumerate(attempts, 1):
        if first_semantic is None and att.success_level == "success":
            first_semantic = idx
        if first_bypass is None and att.was_blocked is False:
            first_bypass = idx
        if first_semantic is not None and first_bypass is not None:
            break
    avg_latency_ms = (sum(a.latency_ms for a in attempts) / len(attempts)) if attempts else 0.0
    total_attempts = len(attempts)
    semantic_success_count = len(semantic_successes)
    bypass_success_count = len(bypass_successes)
    bypass_asr = (bypass_success_count / total_attempts) if total_attempts else 0.0
    semantic_asr = (semantic_success_count / total_attempts) if total_attempts else 0.0

    return {
        "attempts": total_attempts,
        "bypass_successes": bypass_success_count,
        "semantic_successes": semantic_success_count,
        "bypass_asr": bypass_asr,
        "semantic_asr": semantic_asr,
        "first_bypass_queries": first_bypass,
        "first_semantic_success_queries": first_semantic,
        "avg_latency_ms": avg_latency_ms,
        "wall_time_ms": elapsed_ms,
        "unblocked_attempts": bypass_success_count,
        "vulnerabilities": len(report.vulnerabilities),
        "summary": report.summary,
        "artifacts": paths,
    }


def build_scenarios(selected: list[str]) -> list[ScenarioSpec]:
    scenario_map = {
        "static_weak": ScenarioSpec(
            name="static_weak",
            kind="static",
            description="Static payload baseline (weak / non-adaptive).",
            requires_llm=False,
        ),
        "llm_baseline": ScenarioSpec(
            name="llm_baseline",
            kind="llm",
            description="LLM one-shot generation baseline (limited adaptation).",
            requires_llm=True,
        ),
        "adaptive_balanced": ScenarioSpec(
            name="adaptive_balanced",
            kind="adaptive",
            description="Adaptive state-machine pentest (balanced strategy).",
            requires_llm=True,
            mode="state_machine",
            strategy="balanced",
        ),
        "adaptive_aggressive": ScenarioSpec(
            name="adaptive_aggressive",
            kind="adaptive",
            description="Adaptive state-machine pentest (aggressive strategy).",
            requires_llm=True,
            mode="state_machine",
            strategy="aggressive",
            budget_multiplier=1.0,
        ),
        "adaptive_react": ScenarioSpec(
            name="adaptive_react",
            kind="adaptive",
            description="Adaptive ReAct pentest agent.",
            requires_llm=True,
            mode="react",
            strategy="balanced",
        ),
    }

    out: list[ScenarioSpec] = []
    for name in selected:
        if name not in scenario_map:
            raise ValueError(f"Unknown scenario: {name}")
        out.append(scenario_map[name])
    return out


def summarize_by_scenario(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["scenario"], []).append(row)

    summary: dict[str, dict[str, Any]] = {}
    for scenario, items in grouped.items():
        asrs = [float(i.get("asr", 0.0) or 0.0) for i in items]
        bypass_asrs = [float(i.get("bypass_asr", 0.0) or 0.0) for i in items]
        semantic_asrs = [float(i.get("semantic_asr", 0.0) or 0.0) for i in items]
        first = [int(i["first_success_queries"]) for i in items if i.get("first_success_queries") is not None]
        first_bypass = [int(i["first_bypass_queries"]) for i in items if i.get("first_bypass_queries") is not None]
        first_semantic = [
            int(i["first_semantic_success_queries"]) for i in items if i.get("first_semantic_success_queries") is not None
        ]
        attempts = [int(i["attempts"]) for i in items]
        successes = [int(i["successes"]) for i in items]
        lats = [float(i["avg_latency_ms"]) for i in items]
        summary[scenario] = {
            "runs": len(items),
            "mean_asr": statistics.mean(asrs) if asrs else 0.0,
            "median_asr": statistics.median(asrs) if asrs else 0.0,
            "mean_bypass_asr": statistics.mean(bypass_asrs) if bypass_asrs else 0.0,
            "mean_semantic_asr": statistics.mean(semantic_asrs) if semantic_asrs else 0.0,
            "mean_attempts": statistics.mean(attempts) if attempts else 0.0,
            "mean_successes": statistics.mean(successes) if successes else 0.0,
            "mean_first_success_queries": (statistics.mean(first) if first else None),
            "mean_first_bypass_queries": (statistics.mean(first_bypass) if first_bypass else None),
            "mean_first_semantic_success_queries": (statistics.mean(first_semantic) if first_semantic else None),
            "mean_avg_latency_ms": statistics.mean(lats) if lats else 0.0,
        }
    return summary


def render_markdown(
    started_at: str,
    model: str,
    api_base: str,
    primary_metric: str,
    request_timeout: float,
    adaptive_llm_eval: bool,
    targets: list[str],
    budgets: list[int],
    scenarios: list[ScenarioSpec],
    rows: list[dict[str, Any]],
    scenario_summary: dict[str, dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# Adaptive Attack Experiment Report")
    lines.append("")
    lines.append(f"- Started at: `{started_at}`")
    lines.append(f"- Model: `{model}`")
    lines.append(f"- API Base: `{api_base}`")
    lines.append(f"- Primary Metric: `{primary_metric}`")
    lines.append(f"- Request Timeout (s): `{request_timeout}`")
    lines.append(f"- Adaptive LLM Eval: `{adaptive_llm_eval}`")
    lines.append(f"- Targets: `{', '.join(targets)}`")
    lines.append(f"- Budgets: `{', '.join(str(b) for b in budgets)}`")
    lines.append(f"- Scenarios: `{', '.join(s.name for s in scenarios)}`")
    lines.append("")
    lines.append("## Scenario Summary")
    lines.append("")
    lines.append(
        "| Scenario | Runs | Mean Primary ASR | Mean Bypass ASR | Mean Semantic ASR | "
        "Mean First Primary | Mean First Bypass | Mean First Semantic | Mean Attempts | Mean Avg Latency (ms) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, s in scenario_summary.items():
        mf = "-" if s["mean_first_success_queries"] is None else f"{s['mean_first_success_queries']:.1f}"
        mfb = "-" if s["mean_first_bypass_queries"] is None else f"{s['mean_first_bypass_queries']:.1f}"
        mfs = (
            "-"
            if s["mean_first_semantic_success_queries"] is None
            else f"{s['mean_first_semantic_success_queries']:.1f}"
        )
        lines.append(
            f"| {name} | {s['runs']} | {s['mean_asr']:.3f} | {s['mean_bypass_asr']:.3f} | "
            f"{s['mean_semantic_asr']:.3f} | {mf} | {mfb} | {mfs} | {s['mean_attempts']:.1f} | "
            f"{s['mean_avg_latency_ms']:.1f} |"
        )

    lines.append("")
    lines.append("## Detailed Runs")
    lines.append("")
    lines.append(
        "| Target | Budget | Scenario | Primary ASR | Bypass ASR | Semantic ASR | "
        "Primary Successes/Attempts | Bypass | Semantic | First Primary | First Bypass | First Semantic |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        fp = "-" if r.get("first_success_queries") is None else str(r["first_success_queries"])
        fb = "-" if r.get("first_bypass_queries") is None else str(r["first_bypass_queries"])
        fs = "-" if r.get("first_semantic_success_queries") is None else str(r["first_semantic_success_queries"])
        lines.append(
            f"| {r['target']} | {r['budget']} | {r['scenario']} | {r['asr']:.3f} | "
            f"{r.get('bypass_asr', 0.0):.3f} | {r.get('semantic_asr', 0.0):.3f} | "
            f"{r['successes']}/{r['attempts']} | {r.get('bypass_successes', 0)}/{r['attempts']} | "
            f"{r.get('semantic_successes', 0)}/{r['attempts']} | {fp} | {fb} | {fs} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `Primary ASR` is governed by `--primary-metric` (`bypass` or `semantic`).")
    lines.append("- `Bypass ASR` counts responses where target did not block (`was_blocked=False`).")
    lines.append("- `Semantic ASR` counts evaluator-judged successful attacks (`success_level=success` / `success=True`).")
    lines.append("- `first_*` approximates query efficiency (lower is stronger attacker).")
    lines.append("- This setup follows the paper's principle: compare static/weak attacks against adaptive attackers with larger budgets.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run adaptive-attack experiment suites.")
    parser.add_argument(
        "--targets",
        default="guardrail:no_injection,guardrail:safe,guardrail:safe_mcp",
        help="Comma-separated targets (guardrail:name or http(s)://...)",
    )
    parser.add_argument(
        "--budgets",
        default="30,80,150",
        help="Comma-separated attack budgets",
    )
    parser.add_argument(
        "--scenarios",
        default="static_weak,llm_baseline,adaptive_balanced,adaptive_aggressive",
        help="Comma-separated scenarios: static_weak,llm_baseline,adaptive_balanced,adaptive_aggressive,adaptive_react",
    )
    parser.add_argument(
        "--categories",
        default="injection,jailbreak,bypass,mcp",
        help="Comma-separated categories used by attackers",
    )
    parser.add_argument(
        "--severity-filter",
        default="low",
        choices=["low", "medium", "high", "critical"],
        help="Severity filter for static baseline",
    )
    parser.add_argument(
        "--env-file",
        default="",
        help="Optional .env file path; defaults to workspace and project .env",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override OPENROUTER_MODEL for dspy",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="Override OPENROUTER_BASE_URL for dspy",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=60.0,
        help="Timeout (seconds) for each LLM API request",
    )
    parser.add_argument(
        "--api-key-env",
        default="MOONSHOT_API_KEY",
        help="API key env var name (default: MOONSHOT_API_KEY)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "adaptive_attack"),
        help="Output directory for JSON/Markdown and adaptive artifacts",
    )
    parser.add_argument(
        "--primary-metric",
        default="bypass",
        choices=["bypass", "semantic"],
        help="Primary metric used for scenario summaries and ranking",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke mode: first target, budget=20, scenarios=static_weak,adaptive_balanced",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose adaptive pentest logs",
    )
    parser.add_argument(
        "--adaptive-llm-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use LLM in adaptive evaluator for suspicious responses",
    )
    args = parser.parse_args()

    targets = parse_csv(args.targets)
    budgets = parse_int_csv(args.budgets)
    categories = parse_csv(args.categories)
    scenario_names = parse_csv(args.scenarios)

    if args.quick:
        targets = targets[:1]
        budgets = [20]
        scenario_names = ["static_weak", "adaptive_balanced"]

    scenarios = build_scenarios(scenario_names)

    env_candidates = []
    if args.env_file:
        env_candidates.append(Path(args.env_file))
    else:
        env_candidates.extend([WORKSPACE_ROOT / ".env", PROJECT_ROOT / ".env"])
    loaded = load_env_candidates(env_candidates)
    if loaded:
        print(f"Loaded {len(loaded)} vars from: {', '.join(str(p) for p in env_candidates if p.exists())}")

    api_key = os.environ.get(args.api_key_env, "")
    llm_required = any(s.requires_llm for s in scenarios)
    llm_ready = False
    model = args.model or os.environ.get("MOONSHOT_MODEL", os.environ.get("OPENROUTER_MODEL", "openai/kimi-k2.5"))
    api_base = args.api_base or os.environ.get("MOONSHOT_BASE_URL", os.environ.get("OPENROUTER_BASE_URL", "https://api.moonshot.cn/v1"))
    if llm_required:
        llm_ready, model, api_base = setup_dspy_from_env(args.model, args.api_base, api_key, args.request_timeout)
        if not llm_ready:
            print(f"WARNING: {args.api_key_env} missing. LLM-required scenarios will be skipped.")
        else:
            print(f"DSPy configured with {model} @ {api_base}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adaptive_artifacts_dir = out_dir / "adaptive_runs"
    adaptive_artifacts_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_rows: list[dict[str, Any]] = []

    total_runs = len(targets) * len(budgets) * len(scenarios)
    run_idx = 0

    from dspy_guardrails.platform.cli.utils import parse_target

    print(f"Starting adaptive-attack experiment matrix ({total_runs} runs)")
    for target_spec in targets:
        for budget in budgets:
            for scenario in scenarios:
                run_idx += 1
                effective_budget = max(1, int(round(budget * scenario.budget_multiplier)))
                label = f"[{run_idx}/{total_runs}] target={target_spec} budget={effective_budget} scenario={scenario.name}"
                print(label)

                if scenario.requires_llm and not llm_ready:
                    base_row = {
                        "target": target_spec,
                        "budget": effective_budget,
                        "scenario": scenario.name,
                        "kind": scenario.kind,
                        **_empty_metrics(error=f"Skipped: missing {args.api_key_env}"),
                    }
                    run_rows.append(apply_primary_metric(base_row, args.primary_metric))
                    continue

                try:
                    target = parse_target(target_spec)
                    if scenario.kind == "static":
                        metrics = run_static(
                            target=target,
                            budget=effective_budget,
                            categories=categories,
                            severity_filter=args.severity_filter,
                        )
                    elif scenario.kind == "llm":
                        metrics = run_llm(
                            target=target,
                            budget=effective_budget,
                            categories=categories,
                        )
                    elif scenario.kind == "adaptive":
                        metrics = run_adaptive(
                            target=target,
                            budget=effective_budget,
                            categories=categories,
                            mode=scenario.mode,
                            strategy=scenario.strategy,
                            use_llm_evaluation=args.adaptive_llm_eval,
                            output_dir=adaptive_artifacts_dir / f"{target_spec.replace(':', '_').replace('/', '_')}_{scenario.name}_{effective_budget}",
                            verbose=args.verbose,
                        )
                    else:
                        raise ValueError(f"Unknown scenario kind: {scenario.kind}")

                    base_row = {
                        "target": target_spec,
                        "budget": effective_budget,
                        "scenario": scenario.name,
                        "kind": scenario.kind,
                        **metrics,
                    }
                    run_rows.append(apply_primary_metric(base_row, args.primary_metric))
                except Exception as e:
                    base_row = {
                        "target": target_spec,
                        "budget": effective_budget,
                        "scenario": scenario.name,
                        "kind": scenario.kind,
                        **_empty_metrics(error=str(e)),
                    }
                    run_rows.append(apply_primary_metric(base_row, args.primary_metric))
                    continue

    scenario_summary = summarize_by_scenario(run_rows)
    output = {
        "paper_reference": "The Attacker Moves Second: Stronger Adaptive Attacks Bypass Defenses Against LLM Jailbreaks and Prompt Injections",
        "started_at": started_at,
        "model": model,
        "api_base": api_base,
        "primary_metric": args.primary_metric,
        "request_timeout_s": args.request_timeout,
        "adaptive_llm_eval": args.adaptive_llm_eval,
        "targets": targets,
        "budgets": budgets,
        "scenarios": [
            {
                "name": s.name,
                "kind": s.kind,
                "description": s.description,
                "requires_llm": s.requires_llm,
                "mode": s.mode,
                "strategy": s.strategy,
            }
            for s in scenarios
        ],
        "runs": run_rows,
        "summary_by_scenario": scenario_summary,
    }

    json_path = out_dir / f"adaptive_attack_experiment_{started_at}.json"
    md_path = out_dir / f"adaptive_attack_experiment_{started_at}.md"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    md_text = render_markdown(
        started_at=started_at,
        model=model,
        api_base=api_base,
        primary_metric=args.primary_metric,
        request_timeout=args.request_timeout,
        adaptive_llm_eval=args.adaptive_llm_eval,
        targets=targets,
        budgets=budgets,
        scenarios=scenarios,
        rows=run_rows,
        scenario_summary=scenario_summary,
    )
    md_path.write_text(md_text, encoding="utf-8")

    print("\n=== Experiment complete ===")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    print(f"\nScenario means (ASR, primary={args.primary_metric}):")
    for name, s in scenario_summary.items():
        print(f"  - {name}: {s['mean_asr']:.3f} (runs={s['runs']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
