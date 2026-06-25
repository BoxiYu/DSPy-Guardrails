from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class IterationRecord:
    iteration: int
    timestamp: str  # ISO format
    kind: str  # "attack" or "defense"
    algorithm_name: str
    algorithm_path: str
    hypothesis: str
    results: dict = field(default_factory=dict)
    status: str = "keep"  # "keep" | "discard" | "crash"
    analysis: str = ""
    next_ideas: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_TSV_HEADER = ["iteration", "kind", "algorithm", "version", "status",
               "asr", "f1", "fpr", "strongreject", "description"]


def _extract_version(algorithm_name: str) -> str:
    """Extract numeric version from names like 'pair_v3' or 'v3_pair'."""
    import re
    m = re.search(r"v(\d+)", algorithm_name, re.IGNORECASE)
    return m.group(1) if m else "0"


def _format_results(results: dict) -> str:
    """Format results dict into readable inline string."""
    parts = []
    for k, v in results.items():
        if isinstance(v, float):
            parts.append(f"{k.upper()}={v:.3f}")
        else:
            parts.append(f"{k.upper()}={v}")
    return ", ".join(parts) if parts else "N/A"


# ---------------------------------------------------------------------------
# Public init helper
# ---------------------------------------------------------------------------

def init_log(log_path: Path, goal: str) -> None:
    """Create initial AGENT_LOG.md with header section."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()
    content = (
        "# Autoresearch Agent Log\n\n"
        "## Run Info\n"
        f"- Started: {started}\n"
        f"- Goal: {goal}\n"
        "- Best attack: N/A\n"
        "- Best defense: N/A\n"
        "- Total iterations: 0\n\n"
        "---\n"
    )
    log_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# AgentMemory
# ---------------------------------------------------------------------------

class AgentMemory:
    """Cross-iteration memory system backed by AGENT_LOG.md and results.tsv."""

    def __init__(
        self,
        log_path: Path | None = None,
        results_tsv_path: Path | None = None,
    ) -> None:
        self.log_path = log_path or (_PROJECT_ROOT / "autoresearch" / "AGENT_LOG.md")
        self.results_tsv_path = (
            results_tsv_path or (_PROJECT_ROOT / "autoresearch" / "results.tsv")
        )

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def append_iteration(self, record: IterationRecord) -> None:
        """Append record to AGENT_LOG.md and a row to results.tsv."""
        self._ensure_log_exists()
        self._append_md_entry(record)
        self._update_run_info()
        self._append_tsv_row(record)

    def _ensure_log_exists(self) -> None:
        if not self.log_path.exists():
            init_log(self.log_path, "Discover novel algorithms")

    def _append_md_entry(self, record: IterationRecord) -> None:
        entry = (
            f"\n## Iteration {record.iteration} [{record.kind}] — {record.status}\n"
            f"**Algorithm**: {_extract_version(record.algorithm_name)} ({record.algorithm_name})\n"
            f"**Path**: {record.algorithm_path}\n"
            f"**Timestamp**: {record.timestamp}\n"
            f"**Hypothesis**: {record.hypothesis}\n"
            f"**Results**: {_format_results(record.results)}\n"
            f"**Analysis**: {record.analysis}\n"
            f"**Next ideas**: {record.next_ideas}\n\n"
            "---\n"
        )
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(entry)

    def _update_run_info(self) -> None:
        """Rewrite the Run Info block with updated totals and best algorithms."""
        history = self.load_history()
        total = len(history)

        best_attack_str = "N/A"
        attacks = self.get_best_algorithms(kind="attack", top_k=1)
        if attacks:
            a = attacks[0]
            asr = a.results.get("asr", 0)
            best_attack_str = f"{a.algorithm_name} (ASR: {asr:.2f})"

        best_defense_str = "N/A"
        defenses = self.get_best_algorithms(kind="defense", top_k=1)
        if defenses:
            d = defenses[0]
            f1 = d.results.get("f1", 0)
            best_defense_str = f"{d.algorithm_name} (F1: {f1:.2f})"

        text = self.log_path.read_text(encoding="utf-8")
        import re
        run_info_pattern = re.compile(
            r"(## Run Info\n)(.*?)(\n---)", re.DOTALL
        )

        def replacement(m: re.Match) -> str:
            # Keep the Started and Goal lines, replace Best/Total
            block = m.group(2)
            started_m = re.search(r"- Started: (.+)", block)
            goal_m = re.search(r"- Goal: (.+)", block)
            started = started_m.group(1) if started_m else "unknown"
            goal = goal_m.group(1) if goal_m else "unknown"
            new_block = (
                f"- Started: {started}\n"
                f"- Goal: {goal}\n"
                f"- Best attack: {best_attack_str}\n"
                f"- Best defense: {best_defense_str}\n"
                f"- Total iterations: {total}"
            )
            return f"{m.group(1)}{new_block}{m.group(3)}"

        new_text = run_info_pattern.sub(replacement, text, count=1)
        self.log_path.write_text(new_text, encoding="utf-8")

    def _append_tsv_row(self, record: IterationRecord) -> None:
        self.results_tsv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = (
            not self.results_tsv_path.exists()
            or self.results_tsv_path.stat().st_size == 0
        )
        with self.results_tsv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_TSV_HEADER, delimiter="\t")
            if write_header:
                writer.writeheader()
            row = {
                "iteration": record.iteration,
                "kind": record.kind,
                "algorithm": record.algorithm_name,
                "version": _extract_version(record.algorithm_name),
                "status": record.status,
                "asr": f"{record.results.get('asr', 0.0):.3f}",
                "f1": f"{record.results.get('f1', 0.0):.3f}",
                "fpr": f"{record.results.get('fpr', 0.0):.3f}",
                "strongreject": f"{record.results.get('strongreject', 0.0):.3f}",
                "description": record.hypothesis[:120],
            }
            writer.writerow(row)

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def load_history(self) -> list[IterationRecord]:
        """Parse AGENT_LOG.md back into IterationRecord objects."""
        if not self.log_path.exists():
            return []
        text = self.log_path.read_text(encoding="utf-8")
        return _parse_md(text)

    def get_iteration_count(self) -> int:
        return len(self.load_history())

    def get_next_version(self) -> int:
        history = self.load_history()
        if not history:
            return 1
        return max(r.iteration for r in history) + 1

    def get_context(self, last_n: int = 5) -> str:
        """Return formatted context string with last N iteration summaries."""
        history = self.load_history()
        recent = history[-last_n:] if last_n > 0 else history
        if not recent:
            return "No previous iterations recorded.\n"
        lines = [f"## Agent Context (last {len(recent)} iteration(s))\n"]
        for r in recent:
            lines.append(
                f"- Iteration {r.iteration} [{r.kind}] {r.status}: "
                f"{r.algorithm_name} | {_format_results(r.results)}"
            )
            lines.append(f"  Hypothesis: {r.hypothesis}")
            lines.append(f"  Analysis: {r.analysis}")
            lines.append(f"  Next ideas: {r.next_ideas}")
        return "\n".join(lines) + "\n"

    def get_best_algorithms(
        self, kind: str = "all", top_k: int = 3
    ) -> list[IterationRecord]:
        """Return top-K kept iterations ranked by ASR (attacks) or F1 (defenses)."""
        history = self.load_history()
        kept = [r for r in history if r.status == "keep"]
        if kind == "attack":
            candidates = [r for r in kept if r.kind == "attack"]
            candidates.sort(key=lambda r: r.results.get("asr", 0.0), reverse=True)
        elif kind == "defense":
            candidates = [r for r in kept if r.kind == "defense"]
            candidates.sort(key=lambda r: r.results.get("f1", 0.0), reverse=True)
        else:
            # Mixed: sort attacks by ASR, defenses by F1, interleave top results
            attacks = sorted(
                [r for r in kept if r.kind == "attack"],
                key=lambda r: r.results.get("asr", 0.0),
                reverse=True,
            )
            defenses = sorted(
                [r for r in kept if r.kind == "defense"],
                key=lambda r: r.results.get("f1", 0.0),
                reverse=True,
            )
            candidates = []
            for a, d in zip(attacks, defenses):
                candidates.extend([a, d])
            # Append remaining
            longer = attacks if len(attacks) > len(defenses) else defenses
            candidates.extend(longer[min(len(attacks), len(defenses)):])
        return candidates[:top_k]


# ---------------------------------------------------------------------------
# Markdown parser
# ---------------------------------------------------------------------------

def _parse_md(text: str) -> list[IterationRecord]:
    """Parse AGENT_LOG.md content into IterationRecord list."""
    import re
    records: list[IterationRecord] = []

    # Split on iteration headers: ## Iteration N [kind] — status
    header_re = re.compile(
        r"^## Iteration (\d+) \[(\w+)\] — (\w+)\s*$", re.MULTILINE
    )
    field_re = re.compile(r"^\*\*(\w[\w ]*)\*\*: (.*)$")

    headers = list(header_re.finditer(text))
    for i, hm in enumerate(headers):
        iteration = int(hm.group(1))
        kind = hm.group(2)
        status = hm.group(3)

        # Extract block until next header or end
        block_start = hm.end()
        block_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        block = text[block_start:block_end]

        fields: dict[str, Any] = {
            "iteration": iteration,
            "kind": kind,
            "status": status,
            "timestamp": "",
            "algorithm_name": "",
            "algorithm_path": "",
            "hypothesis": "",
            "results": {},
            "analysis": "",
            "next_ideas": "",
        }

        for line in block.splitlines():
            fm = field_re.match(line.strip())
            if not fm:
                continue
            key_raw, value = fm.group(1).strip(), fm.group(2).strip()
            key = key_raw.lower().replace(" ", "_")
            if key == "algorithm":
                # Format: "vN (name)"
                name_m = re.match(r".*\((.+)\)", value)
                fields["algorithm_name"] = name_m.group(1) if name_m else value
            elif key == "path":
                fields["algorithm_path"] = value
            elif key == "timestamp":
                fields["timestamp"] = value
            elif key == "hypothesis":
                fields["hypothesis"] = value
            elif key == "results":
                fields["results"] = _parse_results_str(value)
            elif key == "analysis":
                fields["analysis"] = value
            elif key == "next_ideas":
                fields["next_ideas"] = value

        records.append(IterationRecord(**fields))
    return records


def _parse_results_str(s: str) -> dict:
    """Parse 'ASR=0.35, F1=0.82' style string into dict with lowercase keys."""
    import re
    result: dict = {}
    for m in re.finditer(r"(\w+)=([^\s,]+)", s):
        key = m.group(1).lower()
        raw = m.group(2)
        try:
            result[key] = float(raw)
        except ValueError:
            result[key] = raw
    return result
