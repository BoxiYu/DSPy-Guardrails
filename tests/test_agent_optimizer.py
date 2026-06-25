"""D1: Unit tests for agent_optimizer/ module"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dspy_guardrails.agent_optimizer.trajectory import (
    Step,
    StepType,
    Trajectory,
    Outcome,
    TrajectoryRecorder,
)
from dspy_guardrails.agent_optimizer.credit import (
    CreditAssigner,
    UniformCredit,
    DecayCredit,
    AttentionCredit,
    CreditSummary,
    assign_credit,
)


class TestStep:
    """Test Step dataclass."""

    def test_create(self):
        s = Step(step_id=0, step_type=StepType.TOOL_CALL)
        assert s.step_id == 0
        assert s.step_type == StepType.TOOL_CALL
        assert s.reward == 0.0
        assert s.credit == 0.0

    def test_to_dict(self):
        s = Step(
            step_id=1,
            step_type=StepType.TOOL_CALL,
            agent_name="test_agent",
            action_name="search",
        )
        d = s.to_dict()
        assert d["step_id"] == 1
        assert d["step_type"] == "tool_call"
        assert d["agent_name"] == "test_agent"
        assert d["action"]["name"] == "search"


class TestTrajectory:
    """Test Trajectory recording and queries."""

    def test_add_step(self):
        t = Trajectory(trajectory_id="t1", task="test")
        s = Step(step_id=0, step_type=StepType.TOOL_CALL, action_name="search")
        t.add_step(s)
        assert t.total_steps == 1
        assert t.total_tool_calls == 1

    def test_add_handoff(self):
        t = Trajectory()
        t.add_step(Step(step_id=0, step_type=StepType.HANDOFF, agent_name="a", action_name="b"))
        assert t.total_handoffs == 1

    def test_finish(self):
        t = Trajectory()
        t.finish("done", Outcome.SUCCESS, reward=1.0)
        assert t.final_response == "done"
        assert t.outcome == Outcome.SUCCESS
        assert t.final_reward == 1.0
        assert t.end_time > 0

    def test_get_tools_used(self):
        t = Trajectory()
        t.add_step(Step(step_id=0, step_type=StepType.TOOL_CALL, action_name="search"))
        t.add_step(Step(step_id=1, step_type=StepType.TOOL_CALL, action_name="calc"))
        t.add_step(Step(step_id=2, step_type=StepType.LLM_CALL))
        assert t.get_tools_used() == ["search", "calc"]

    def test_get_agents_visited(self):
        t = Trajectory()
        t.add_step(Step(step_id=0, step_type=StepType.AGENT_CALL, agent_name="a"))
        t.add_step(Step(step_id=1, step_type=StepType.AGENT_CALL, agent_name="a"))
        t.add_step(Step(step_id=2, step_type=StepType.HANDOFF, agent_name="b"))
        assert t.get_agents_visited() == ["a", "b"]

    def test_summary(self):
        t = Trajectory(trajectory_id="abcdefgh")
        t.finish("done", Outcome.SUCCESS, reward=0.8)
        s = t.summary()
        assert "abcdefg" in s  # first 8 chars
        assert "success" in s

    def test_to_dict(self):
        t = Trajectory(trajectory_id="test", task="test task")
        t.add_step(Step(step_id=0, step_type=StepType.TOOL_CALL, action_name="s"))
        t.finish("done", Outcome.SUCCESS)
        d = t.to_dict()
        assert d["trajectory_id"] == "test"
        assert d["outcome"] == "success"
        assert len(d["steps"]) == 1

    def test_to_json(self):
        t = Trajectory(trajectory_id="test")
        t.finish("ok", Outcome.SUCCESS)
        j = t.to_json()
        import json
        parsed = json.loads(j)
        assert parsed["trajectory_id"] == "test"


class TestTrajectoryRecorder:
    """Test TrajectoryRecorder convenience methods."""

    def test_record_tool_call(self):
        rec = TrajectoryRecorder(task="test")
        step = rec.record_tool_call("search", {"q": "test"}, "results", latency_ms=50)
        assert step.step_type == StepType.TOOL_CALL
        assert step.action_name == "search"

    def test_record_handoff(self):
        rec = TrajectoryRecorder(task="test")
        rec.set_agent("agent_a")
        step = rec.record_handoff("agent_a", "agent_b", "escalation")
        assert step.step_type == StepType.HANDOFF

    def test_record_llm_call(self):
        rec = TrajectoryRecorder()
        step = rec.record_llm_call("prompt", "thought", "reasoning", "output", 100, 500)
        assert step.step_type == StepType.LLM_CALL
        assert step.token_count == 500

    def test_finish(self):
        rec = TrajectoryRecorder(task="t")
        rec.record_tool_call("s", {}, "r")
        traj = rec.finish("done", Outcome.SUCCESS, reward=1.0)
        assert traj.final_response == "done"
        assert traj.total_steps == 1

    def test_generated_id(self):
        rec = TrajectoryRecorder()
        assert rec.trajectory.trajectory_id.startswith("traj_")


class TestCreditAssignment:
    """Test credit assignment strategies."""

    def _make_trajectory(self, reward=1.0, n_tools=3):
        t = Trajectory()
        for i in range(n_tools):
            t.add_step(Step(step_id=i, step_type=StepType.TOOL_CALL, action_name=f"tool_{i}"))
        t.finish("done", Outcome.SUCCESS, reward=reward)
        return t

    def test_uniform_credit(self):
        t = self._make_trajectory(reward=1.0, n_tools=4)
        UniformCredit().assign(t)
        credits = [s.credit for s in t.steps]
        assert all(abs(c - 0.25) < 0.01 for c in credits)

    def test_decay_credit(self):
        t = self._make_trajectory(reward=1.0, n_tools=3)
        DecayCredit(decay_factor=0.5).assign(t)
        credits = [s.credit for s in t.steps]
        # Later steps should get more credit (default: not reversed)
        assert credits[-1] > credits[0]
        assert abs(sum(credits) - 1.0) < 0.01

    def test_decay_reverse(self):
        t = self._make_trajectory(reward=1.0, n_tools=3)
        DecayCredit(decay_factor=0.5, reverse=True).assign(t)
        credits = [s.credit for s in t.steps]
        assert credits[0] > credits[-1]

    def test_attention_credit(self):
        t = self._make_trajectory(reward=1.0, n_tools=3)
        AttentionCredit().assign(t)
        total = sum(s.credit for s in t.steps)
        assert abs(total - 1.0) < 0.01

    def test_assign_credit_function(self):
        t = self._make_trajectory(reward=1.0, n_tools=3)
        assign_credit(t, method="uniform")
        assert all(s.credit > 0 for s in t.steps)

    def test_assign_credit_decay(self):
        t = self._make_trajectory(reward=1.0, n_tools=3)
        assign_credit(t, method="decay")
        assert all(s.credit > 0 for s in t.steps)

    def test_empty_trajectory(self):
        t = Trajectory()
        t.finish("done", Outcome.SUCCESS, reward=1.0)
        UniformCredit().assign(t)
        # Should not crash

    def test_zero_reward(self):
        t = self._make_trajectory(reward=0.0, n_tools=2)
        UniformCredit().assign(t)
        assert all(s.credit == 0.0 for s in t.steps)


class TestCreditSummary:
    """Test CreditSummary generation."""

    def test_from_trajectory(self):
        t = Trajectory()
        s1 = Step(step_id=0, step_type=StepType.TOOL_CALL, action_name="a")
        s1.credit = 0.5
        s2 = Step(step_id=1, step_type=StepType.TOOL_CALL, action_name="b")
        s2.credit = -0.2
        t.steps = [s1, s2]

        summary = CreditSummary.from_trajectory(t)
        assert summary.total_positive_credit == 0.5
        assert summary.total_negative_credit == -0.2
        assert len(summary.top_contributors) > 0
