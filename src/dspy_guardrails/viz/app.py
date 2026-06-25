"""Streamlit application entry point for dspyGuardrails Security Dashboard."""

from __future__ import annotations

import sys

import streamlit as st

from dspy_guardrails.viz.data_loader import scan_directory

# Accept report directory from CLI args (passed after --)
_default_dir = sys.argv[-1] if len(sys.argv) > 1 and not sys.argv[-1].startswith("-") else "."

st.set_page_config(
    page_title="dspyGuardrails Security Dashboard",
    layout="wide",
)

st.title("dspyGuardrails Security Dashboard")

# Sidebar: directory selector + page navigation
with st.sidebar:
    st.header("Settings")
    report_dir = st.text_input("Report Directory", value=_default_dir, help="Path to directory containing JSON reports")

    page = st.radio("Page", [
        "Overview",
        "Attack Analysis",
        "Defense Analysis",
        "Experiments",
        "Trajectory",
    ])

# Load data
files = scan_directory(report_dir)

if not files:
    st.warning(f"No JSON files found in `{report_dir}`. Please specify a valid report directory.")
    st.stop()

# Show file summary in sidebar
with st.sidebar:
    st.divider()
    st.caption(f"Found {len(files)} JSON files")
    type_counts: dict[str, int] = {}
    for info in files.values():
        t = info["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items()):
        st.caption(f"  {t}: {c}")

# Route to pages
if page == "Overview":
    from dspy_guardrails.viz.pages.overview import render
    render(files)
elif page == "Attack Analysis":
    from dspy_guardrails.viz.pages.attack import render
    render(files)
elif page == "Defense Analysis":
    from dspy_guardrails.viz.pages.defense import render
    render(files)
elif page == "Experiments":
    from dspy_guardrails.viz.pages.experiments import render
    render(files)
elif page == "Trajectory":
    from dspy_guardrails.viz.pages.trajectory import render
    render(files)
