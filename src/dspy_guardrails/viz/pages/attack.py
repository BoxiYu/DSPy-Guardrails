"""Attack analysis (Red Team) page."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dspy_guardrails.viz.components.charts import bar_comparison


def render(files: dict[str, dict]) -> None:
    """Render attack analysis page."""
    st.header("Attack Analysis (Red Team)")

    reports = {k: v for k, v in files.items() if v["type"] == "security_report"}
    if not reports:
        st.info("No security reports found.")
        return

    selected = st.selectbox("Select Report", list(reports.keys()), format_func=lambda k: reports[k]["name"], key="attack_report")
    data = reports[selected]["data"]
    redteam = data.get("redteam", {})

    if not redteam:
        st.info("No red team data in this report.")
        return

    # Category breakdown
    by_category = redteam.get("by_category", {})
    if by_category:
        st.subheader("Results by Category")

        # Category filter
        categories = list(by_category.keys())
        selected_cats = st.multiselect("Filter Categories", categories, default=categories)

        chart_data = []
        for cat in selected_cats:
            cat_data = by_category[cat]
            if isinstance(cat_data, dict):
                chart_data.append({"category": cat, "count": cat_data.get("blocked", 0), "result": "Blocked"})
                chart_data.append({"category": cat, "count": cat_data.get("bypassed", cat_data.get("success", 0)), "result": "Bypassed"})

        if chart_data:
            st.plotly_chart(
                bar_comparison(chart_data, "category", "count", "result", "Blocked vs Bypassed by Category"),
                use_container_width=True,
            )

    # Severity distribution
    by_severity = redteam.get("by_severity", {})
    if by_severity:
        st.subheader("Severity Distribution")
        labels = list(by_severity.keys())
        values = [by_severity[s] if isinstance(by_severity[s], (int, float)) else by_severity[s].get("count", 0) for s in labels]
        fig = go.Figure(go.Pie(labels=labels, values=values))
        fig.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    # Bypass details
    bypassed = redteam.get("bypassed_attacks", redteam.get("successful_attacks", []))
    if bypassed:
        st.subheader("Bypassed Attacks")
        for i, attack in enumerate(bypassed):
            if isinstance(attack, dict):
                with st.expander(f"{attack.get('category', 'Unknown')} - {attack.get('name', f'Attack {i+1}')}"):
                    if "prompt" in attack:
                        st.text_area("Prompt", attack["prompt"], height=100, key=f"atk_prompt_{i}", disabled=True)
                    if "response" in attack:
                        st.text_area("Response", attack["response"], height=100, key=f"atk_resp_{i}", disabled=True)
                    if "severity" in attack:
                        st.caption(f"Severity: {attack['severity']}")
