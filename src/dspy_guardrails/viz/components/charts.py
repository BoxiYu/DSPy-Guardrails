"""Reusable Plotly chart components."""

from __future__ import annotations

import plotly.graph_objects as go


def score_gauge(score: float, title: str = "Score") -> go.Figure:
    """Create a 0-100 gauge indicator."""
    color = "green" if score >= 70 else "orange" if score >= 40 else "red"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "#ffcccc"},
                {"range": [40, 70], "color": "#fff3cd"},
                {"range": [70, 100], "color": "#d4edda"},
            ],
        },
    ))
    fig.update_layout(height=250, margin=dict(t=40, b=0, l=20, r=20))
    return fig


def radar_chart(categories: list[str], values: list[float], title: str = "") -> go.Figure:
    """Create a radar chart."""
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name=title,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=350,
        margin=dict(t=30, b=30, l=60, r=60),
        title=title,
    )
    return fig


def confusion_matrix(tp: int, tn: int, fp: int, fn: int) -> go.Figure:
    """Create a confusion matrix heatmap."""
    z = [[tn, fp], [fn, tp]]
    labels = [["TN", "FP"], ["FN", "TP"]]
    text = [[f"{labels[i][j]}<br>{z[i][j]}" for j in range(2)] for i in range(2)]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=["Predicted Negative", "Predicted Positive"],
        y=["Actual Negative", "Actual Positive"],
        text=text,
        texttemplate="%{text}",
        colorscale="RdYlGn",
        showscale=False,
    ))
    fig.update_layout(
        title="Confusion Matrix",
        height=350,
        margin=dict(t=40, b=40, l=40, r=40),
    )
    return fig


def bar_comparison(
    data: list[dict],
    x: str,
    y: str,
    color: str | None = None,
    title: str = "",
) -> go.Figure:
    """Create a grouped bar chart from a list of dicts."""
    if color:
        groups = sorted(set(d[color] for d in data))
        fig = go.Figure()
        for group in groups:
            subset = [d for d in data if d[color] == group]
            fig.add_trace(go.Bar(
                x=[d[x] for d in subset],
                y=[d[y] for d in subset],
                name=str(group),
            ))
        fig.update_layout(barmode="group")
    else:
        fig = go.Figure(go.Bar(
            x=[d[x] for d in data],
            y=[d[y] for d in data],
        ))
    fig.update_layout(title=title, height=400, margin=dict(t=40, b=40, l=40, r=40))
    return fig


def progress_line(
    scores: list[float],
    phases: list[str] | None = None,
    title: str = "Progress",
) -> go.Figure:
    """Create a progress line chart with optional phase annotations."""
    fig = go.Figure(go.Scatter(
        x=list(range(len(scores))),
        y=scores,
        mode="lines+markers",
        name="Progress Score",
    ))
    if phases:
        for i, phase in enumerate(phases):
            if i < len(scores):
                fig.add_annotation(
                    x=i, y=scores[i],
                    text=phase,
                    showarrow=True,
                    arrowhead=2,
                    yshift=15,
                )
    fig.update_layout(
        title=title,
        xaxis_title="Turn",
        yaxis_title="Score",
        yaxis=dict(range=[0, 100]),
        height=400,
        margin=dict(t=40, b=40, l=40, r=40),
    )
    return fig
