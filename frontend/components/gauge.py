"""
Plotly risk gauge indicator component for financial fraud probability.
"""

import plotly.graph_objects as go


def render_risk_gauge(
    fraud_probability: float,
    threshold: float = 0.50,
) -> go.Figure:
    """Render high-contrast radial gauge indicating fraud risk tier."""
    pct_val = fraud_probability * 100.0

    if fraud_probability >= 0.80:
        bar_color = "#e53e3e"  # Critical red
    elif fraud_probability >= 0.50:
        bar_color = "#dd6b20"  # High orange
    elif fraud_probability >= 0.20:
        bar_color = "#d69e2e"  # Medium yellow
    else:
        bar_color = "#38a169"  # Low green

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct_val,
            number={"suffix": "%", "font": {"size": 38, "family": "Arial", "color": bar_color}},
            title={"text": "Fraud Risk Score", "font": {"size": 20, "family": "Arial"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#718096"},
                "bar": {"color": bar_color, "thickness": 0.3},
                "bgcolor": "#edf2f7",
                "borderwidth": 1,
                "bordercolor": "#cbd5e0",
                "steps": [
                    {"range": [0, 20], "color": "rgba(56, 161, 105, 0.15)"},
                    {"range": [20, 50], "color": "rgba(214, 158, 46, 0.15)"},
                    {"range": [50, 80], "color": "rgba(221, 107, 32, 0.15)"},
                    {"range": [80, 100], "color": "rgba(229, 62, 62, 0.2)"},
                ],
                "threshold": {
                    "line": {"color": "#1a202c", "width": 4},
                    "thickness": 0.85,
                    "value": threshold * 100.0,
                },
            },
        )
    )

    fig.update_layout(
        height=280,
        margin=dict(l=25, r=25, t=40, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#2d3748"},
    )
    return fig
