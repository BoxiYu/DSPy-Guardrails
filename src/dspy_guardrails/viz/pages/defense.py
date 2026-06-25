"""Defense analysis (Blue Team) page."""

from __future__ import annotations

import streamlit as st

from dspy_guardrails.viz.components.charts import bar_comparison, confusion_matrix


def render(files: dict[str, dict]) -> None:
    """Render defense analysis page."""
    st.header("Defense Analysis (Blue Team)")

    reports = {k: v for k, v in files.items() if v["type"] == "security_report"}
    if not reports:
        st.info("No security reports found.")
        return

    selected = st.selectbox("Select Report", list(reports.keys()), format_func=lambda k: reports[k]["name"], key="defense_report")
    data = reports[selected]["data"]
    blueteam = data.get("blueteam", {})

    if not blueteam:
        st.info("No blue team data in this report.")
        return

    # Confusion matrix
    metrics = blueteam.get("metrics", {})
    tp = metrics.get("true_positives", metrics.get("tp", 0))
    tn = metrics.get("true_negatives", metrics.get("tn", 0))
    fp = metrics.get("false_positives", metrics.get("fp", 0))
    fn = metrics.get("false_negatives", metrics.get("fn", 0))

    if any([tp, tn, fp, fn]):
        st.subheader("Confusion Matrix")
        st.plotly_chart(confusion_matrix(tp, tn, fp, fn), use_container_width=True)

    # FPR vs FNR
    col1, col2 = st.columns(2)
    fpr = metrics.get("fpr", metrics.get("false_positive_rate", 0))
    fnr = metrics.get("fnr", metrics.get("false_negative_rate", 0))
    col1.metric("False Positive Rate", f"{fpr:.1%}")
    col2.metric("False Negative Rate", f"{fnr:.1%}")

    # Per-guardrail breakdown
    by_guardrail = blueteam.get("by_guardrail", {})
    if by_guardrail:
        st.subheader("Per-Guardrail Performance")
        chart_data = []
        for name, gdata in by_guardrail.items():
            if isinstance(gdata, dict):
                for metric_name in ["precision", "recall", "f1"]:
                    val = gdata.get(metric_name, 0)
                    chart_data.append({"guardrail": name, "value": val, "metric": metric_name})

        if chart_data:
            st.plotly_chart(
                bar_comparison(chart_data, "guardrail", "value", "metric", "Guardrail Metrics"),
                use_container_width=True,
            )

    # Failed cases
    failures = blueteam.get("failures", blueteam.get("failed_cases", []))
    if failures:
        st.subheader("Failed Cases")
        for i, case in enumerate(failures):
            if isinstance(case, dict):
                with st.expander(f"Case {i+1}: {case.get('type', 'Unknown')}"):
                    st.json(case)
