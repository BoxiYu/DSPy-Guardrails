"""Crescendo attack trajectory page."""

from __future__ import annotations

import streamlit as st

from dspy_guardrails.viz.components.charts import progress_line


def render(files: dict[str, dict]) -> None:
    """Render trajectory visualization page."""
    st.header("Attack Trajectory")

    trajectories = {k: v for k, v in files.items() if v["type"] == "trajectory"}
    if not trajectories:
        st.info("No trajectory files found.")
        return

    selected = st.selectbox("Select Trajectory", list(trajectories.keys()), format_func=lambda k: trajectories[k]["name"])
    data = trajectories[selected]["data"]

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Phase Reached", data.get("phase_reached", "N/A"))
    col2.metric("Total Turns", len(data.get("progress_scores", data.get("conversation", []))))
    col3.metric("Final Score", f"{(data.get('progress_scores', [0])[-1] if data.get('progress_scores') else 0):.1f}")

    # Progress line chart
    scores = data.get("progress_scores", [])
    phases = data.get("phases", data.get("phase_labels", None))
    if scores:
        st.subheader("Progress Over Turns")
        st.plotly_chart(progress_line(scores, phases, "Attack Progress"), use_container_width=True)

    # Conversation viewer
    conversation = data.get("conversation", data.get("turns", []))
    if conversation:
        st.subheader("Conversation")
        for i, turn in enumerate(conversation):
            if isinstance(turn, dict):
                role = turn.get("role", "unknown")
                content = turn.get("content", turn.get("message", ""))
                is_retreat = turn.get("retreat", turn.get("backoff", False))

                if role in ("user", "attacker"):
                    st.chat_message("user").write(content)
                else:
                    st.chat_message("assistant").write(content)

                if is_retreat:
                    st.warning(f"Turn {i}: Retreat/backoff event")
            elif isinstance(turn, str):
                st.text(turn)
