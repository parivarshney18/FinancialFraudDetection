"""
TabNet sequential decision step attention heatmap component.
"""

from typing import Dict, Any
import plotly.graph_objects as go
import pandas as pd


def render_step_attention_heatmap(
    step_attentions: Dict[str, Dict[str, float]],
) -> go.Figure:
    """Render interactive heatmap showing attention weight allocation across sequential decision steps."""
    if not step_attentions:
        return go.Figure()

    # Convert to DataFrame
    df = pd.DataFrame(step_attentions).T  # rows: steps, columns: features

    fig = go.Figure(
        data=go.Heatmap(
            z=df.values,
            x=list(df.columns),
            y=list(df.index),
            colorscale="Viridis",
            colorbar=dict(title="Attention"),
            text=df.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 11},
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title="Sequential Decision Step Attention Heatmap (M[0] → M[1] → M[2])",
        xaxis_title="Engineered Transaction Features",
        yaxis_title="Decision Step",
        height=260,
        margin=dict(l=10, r=10, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
