"""Regression tests for adversarial artifact persistence."""

import os
import sys
from pathlib import Path

import dspy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dspy_guardrails.adversarial.metrics import AdversarialConfig
from dspy_guardrails.adversarial.trainer import AdversarialTrainer


class _TinyGuardrail(dspy.Module):
    """Minimal dspy.Module guardrail used to validate save() is invoked."""

    def forward(self, **kwargs):
        return dspy.Prediction(is_unsafe=False)


class _ArtifactTarget:
    def __init__(self):
        self.guardrail = _TinyGuardrail()
        self.few_shot_examples = [{"input": "x", "label": "UNSAFE"}]

    def invoke(self, prompt: str):
        class _Resp:
            def __init__(self, blocked: bool):
                self.was_blocked = blocked
                self.response = ""
                self.metadata = {}

        return _Resp(blocked=True)

    def update_defense(self, update):
        return None

    def reset_session(self):
        return None

    def get_defense_stats(self):
        return {}


def test_final_artifacts_are_written(tmp_path):
    out_dir = tmp_path / "runs"
    config = AdversarialConfig(
        output_dir=str(out_dir),
        verbose=False,
        save_every_round=False,
        max_rounds=1,
        attacks_per_round=1,
        attack_categories=["injection"],
        attack_optimizer_enabled=False,
        attack_use_llm_bypass=False,
    )
    trainer = AdversarialTrainer(target=_ArtifactTarget(), config=config)

    result = trainer.run()
    assert result.total_rounds == 1

    run_dirs = sorted(out_dir.glob("run_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    # Core artifacts from trainer.
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "evolved_attacks.json").exists()

    # New artifacts for downstream reuse.
    assert (run_dir / "evolved_attack_payloads.txt").exists()
    assert (run_dir / "final_guardrail_module.json").exists()
    assert (run_dir / "final_guardrail_few_shot_examples.json").exists()
    assert (run_dir / "artifacts.json").exists()

