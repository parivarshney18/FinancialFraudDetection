"""
Top contributing features horizontal bar chart component.
"""

from typing import List, Dict, Any
import plotly.graph_objects as go


def render_contributors_chart(
    top_contributors: List[Dict[str, Any]],
) -> go.Figure:
    """Render horizontal bar chart ranking the top features driving model prediction."""
    if not top_contributors:
        return go.Figure()

    # Sort descending for display
    sorted_items = list(reversed(top_contributors))
    features = [c["feature"] for c in sorted_items]
    percentages = [c["contribution_pct"] for c in sorted_items]
    hover_texts = [
        f"Feature: {c['feature']}<br>"
        f"Contribution: {c['contribution_pct']:.1f}%<br>"
        f"Attention Weight: {c['importance']:.4f}<br>"
        f"Primary Step: Step {c.get('primary_step', 0)}<br>"
        f"Raw Value: {c.get('raw_value', 'N/A')}"
        for c in sorted_items
    ]

    fig = go.Figure(
        go.Bar(
            x=percentages,
            y=features,
            orientation="h",
            text=[f"{p:.1f}%" for p in percentages],
            textposition="outside",
            hovertext=hover_texts,
            hoverinfo="text",
            marker=dict(
                color=percentages,
                colorscale="Blues",
                line=dict(color="#1a365d", width=1),
            ),
        )
    )

    fig.update_layout(
        title="Top Contributing Risk Drivers (TabNet Sparsemax Attribution)",
        xaxis_title="Relative Attribution Share (%)",
        yaxis_title="Feature",
        height=320,
        margin=dict(l=10, r=40, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(showgrid=False),
    )
    return fig
