"""Experiments comparison page."""

from __future__ import annotations

import streamlit as st

from dspy_guardrails.viz.components.charts import bar_comparison


def render(files: dict[str, dict]) -> None:
    """Render experiments comparison page."""
    st.header("Experiment Comparison")

    experiments = {k: v for k, v in files.items() if v["type"] == "experiment"}
    if not experiments:
        st.info("No experiment files found.")
        return

    # Multi-select experiments
    selected_keys = st.multiselect(
        "Select Experiments to Compare",
        list(experiments.keys()),
        default=list(experiments.keys())[:3],
        format_func=lambda k: experiments[k]["name"],
    )

    if not selected_keys:
        return

    # Collect metrics across experiments
    core_metrics = ["f1", "accuracy", "bypass_rate", "latency", "precision", "recall"]
    chart_data = []
    table_rows = []

    for key in selected_keys:
        exp_data = experiments[key]["data"]
        exp_name = exp_data.get("experiment", experiments[key]["name"])

        conditions = exp_data.get("conditions", [])
        if not conditions:
            # Treat entire file as a single condition
            conditions = [exp_data]

        for cond in conditions:
            cond_name = cond.get("name", cond.get("condition", exp_name))
            cond_metrics = cond.get("metrics", cond)
            row = {"experiment": exp_name, "condition": cond_name}

            for m in core_metrics:
                val = cond_metrics.get(m)
                if val is not None:
                    chart_data.append({"condition": cond_name, "value": float(val), "metric": m})
                    row[m] = val

            table_rows.append(row)

    # Grouped bar chart
    if chart_data:
        st.subheader("Metrics Comparison")
        st.plotly_chart(
            bar_comparison(chart_data, "condition", "value", "metric", "Experiment Metrics"),
            use_container_width=True,
        )

    # Table view
    if table_rows:
        st.subheader("Detailed View")
        st.dataframe(table_rows, use_container_width=True)
