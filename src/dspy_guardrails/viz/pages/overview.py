"""Overview dashboard page."""

from __future__ import annotations

import streamlit as st

from dspy_guardrails.viz.components.charts import radar_chart, score_gauge


def render(files: dict[str, dict]) -> None:
    """Render the overview page."""
    st.header("Overview")

    # Find security reports
    reports = {k: v for k, v in files.items() if v["type"] == "security_report"}
    if not reports:
        st.info("No security reports found.")
        return

    # Report selector
    selected = st.selectbox("Select Report", list(reports.keys()), format_func=lambda k: reports[k]["name"])
    data = reports[selected]["data"]

    # Overall score gauge
    overall = data.get("overall_score", 0)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.plotly_chart(score_gauge(overall, "Overall Score"), use_container_width=True)

    # Metric cards
    with col2:
        metrics = data.get("metrics", data.get("blueteam", {}).get("metrics", {}))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Block Rate", f"{metrics.get('block_rate', 0):.1%}")
        m2.metric("Precision", f"{metrics.get('precision', 0):.1%}")
        m3.metric("Recall", f"{metrics.get('recall', 0):.1%}")
        m4.metric("F1 Score", f"{metrics.get('f1', metrics.get('f1_score', 0)):.1%}")

    # Radar chart: multi-dimensional scores
    dimensions = []
    values = []
    for key, label in [("redteam", "Red Team"), ("blueteam", "Blue Team"), ("hallucination", "Hallucination")]:
        section = data.get(key, {})
        score = section.get("score", section.get("overall_score", None))
        if score is not None:
            dimensions.append(label)
            values.append(float(score) if score <= 100 else float(score) * 100)

    if dimensions:
        st.plotly_chart(radar_chart(dimensions, values, "Multi-dimensional Scores"), use_container_width=True)

    # Critical vulnerabilities
    vulns = data.get("critical_vulnerabilities", data.get("vulnerabilities", []))
    if vulns:
        st.subheader("Critical Vulnerabilities")
        for v in vulns:
            if isinstance(v, dict):
                st.error(f"**{v.get('name', v.get('category', 'Unknown'))}**: {v.get('description', v.get('detail', ''))}")
            else:
                st.error(str(v))
