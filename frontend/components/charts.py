"""
AwaazDesk - Plotly Data Visualizations & Analytics Charts
Configured with High-Visibility White Typography for maximum readability across all themes.
"""
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

# Professional High-Contrast Theme (Pure white text for crystal-clear readability)
CHART_THEME = {
    "background": "rgba(0,0,0,0)",
    "font_family": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    "text_color": "#ffffff",
    "grid_color": "rgba(255, 255, 255, 0.15)",
    "axis_title_color": "#ffffff",
    "axis_tick_color": "#ffffff"
}

def render_category_donut(category_data: Dict[str, int]) -> go.Figure:
    """Renders high-contrast donut chart for grievance category distribution with white typography."""
    if not category_data:
        category_data = {"No Data": 1}

    labels = list(category_data.keys())
    values = list(category_data.values())

    colors = [
        "#3b82f6", "#8b5cf6", "#10b981", "#f59e0b",
        "#ef4444", "#06b6d4", "#6366f1", "#64748b"
    ]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#1e293b", width=2)),
        textinfo="percent+label",
        hoverinfo="label+value+percent",
        textposition="inside",
        insidetextorientation="radial",
        textfont=dict(color="#ffffff", size=12)
    )])

    fig.update_layout(
        title=dict(
            text="<b>Grievance Category Distribution</b>",
            font=dict(size=18, color="#ffffff")
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color="#ffffff")
        ),
        margin=dict(t=50, b=50, l=20, r=20),
        paper_bgcolor=CHART_THEME["background"],
        plot_bgcolor=CHART_THEME["background"],
        font=dict(family=CHART_THEME["font_family"], color="#ffffff")
    )
    return fig

def render_urgency_bar(urgency_data: Dict[str, int]) -> go.Figure:
    """Renders urgency breakdown with color-coded severity bars and white text."""
    order = ["Critical", "High", "Medium", "Low"]
    color_map = {
        "Critical": "#ef4444",  # Bright Red
        "High": "#f97316",      # Bright Orange
        "Medium": "#eab308",    # Bright Yellow/Amber
        "Low": "#64748b"        # Slate Gray
    }

    x_vals = []
    y_vals = []
    bar_colors = []

    for level in order:
        x_vals.append(level)
        val = urgency_data.get(level, 0)
        y_vals.append(val)
        bar_colors.append(color_map[level])

    fig = go.Figure(data=[go.Bar(
        x=x_vals,
        y=y_vals,
        marker=dict(color=bar_colors, line=dict(color="#1e293b", width=1.5)),
        text=y_vals,
        textposition="auto",
        textfont=dict(color="#ffffff", size=13, family=CHART_THEME["font_family"])
    )])

    fig.update_layout(
        title=dict(
            text="<b>Urgency & Severity Breakdown</b>",
            font=dict(size=18, color="#ffffff")
        ),
        xaxis=dict(
            title=dict(text="<b>Urgency Tier</b>", font=dict(size=14, color="#ffffff")),
            tickfont=dict(size=13, color="#ffffff"),
            showgrid=False
        ),
        yaxis=dict(
            title=dict(text="<b>Tickets</b>", font=dict(size=14, color="#ffffff")),
            tickfont=dict(size=13, color="#ffffff"),
            showgrid=True,
            gridcolor=CHART_THEME["grid_color"]
        ),
        margin=dict(t=50, b=50, l=20, r=20),
        paper_bgcolor=CHART_THEME["background"],
        plot_bgcolor=CHART_THEME["background"],
        font=dict(family=CHART_THEME["font_family"], color="#ffffff")
    )
    return fig

def render_language_donut(lang_data: Dict[str, int]) -> go.Figure:
    """Renders multilingual submission channels (Hindi, English, Hinglish) with white text."""
    display_names = {
        "hi": "हिन्दी (Hindi)",
        "en": "English",
        "hinglish": "Hinglish (Code-Mixed)"
    }
    colors_map = {
        "hi": "#f97316",
        "en": "#3b82f6",
        "hinglish": "#10b981"
    }

    labels = [display_names.get(k, k) for k in lang_data.keys()]
    values = list(lang_data.values())
    slice_colors = [colors_map.get(k, "#64748b") for k in lang_data.keys()]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker=dict(colors=slice_colors, line=dict(color="#1e293b", width=2)),
        textinfo="percent+label",
        hoverinfo="label+value",
        textfont=dict(color="#ffffff", size=12)
    )])

    fig.update_layout(
        title=dict(
            text="<b>Language Breakdown</b>",
            font=dict(size=18, color="#ffffff")
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color="#ffffff")
        ),
        margin=dict(t=50, b=40, l=20, r=20),
        paper_bgcolor=CHART_THEME["background"],
        plot_bgcolor=CHART_THEME["background"],
        font=dict(family=CHART_THEME["font_family"], color="#ffffff")
    )
    return fig

def render_status_pipeline(status_data: Dict[str, int]) -> go.Figure:
    """Renders resolution lifecycle pipeline bar chart with white labels."""
    lifecycle = ["Open", "In Progress", "Under Investigation", "Resolved", "Closed"]
    colors = ["#3b82f6", "#f59e0b", "#8b5cf6", "#10b981", "#64748b"]

    counts = [status_data.get(s, 0) for s in lifecycle]

    fig = go.Figure(data=[go.Bar(
        x=lifecycle,
        y=counts,
        marker=dict(color=colors, line=dict(color="#1e293b", width=1.5)),
        text=counts,
        textposition="auto",
        textfont=dict(color="#ffffff", size=13, family=CHART_THEME["font_family"])
    )])

    fig.update_layout(
        title=dict(
            text="<b>Grievance Resolution Lifecycle</b>",
            font=dict(size=18, color="#ffffff")
        ),
        xaxis=dict(
            title=dict(text="<b>Status</b>", font=dict(size=14, color="#ffffff")),
            tickfont=dict(size=13, color="#ffffff"),
            showgrid=False
        ),
        yaxis=dict(
            title=dict(text="<b>Count</b>", font=dict(size=14, color="#ffffff")),
            tickfont=dict(size=13, color="#ffffff"),
            showgrid=True,
            gridcolor=CHART_THEME["grid_color"]
        ),
        margin=dict(t=50, b=50, l=20, r=20),
        paper_bgcolor=CHART_THEME["background"],
        plot_bgcolor=CHART_THEME["background"],
        font=dict(family=CHART_THEME["font_family"], color="#ffffff")
    )
    return fig

def render_sentiment_bar(sentiment_data: Dict[str, int]) -> go.Figure:
    """Renders sentiment distribution chart with white labels."""
    labels = list(sentiment_data.keys())
    values = list(sentiment_data.values())

    color_map = {
        "Strongly Negative": "#dc2626",
        "Negative": "#f43f5e",
        "Neutral": "#64748b",
        "Constructive / Inquisitive": "#10b981"
    }
    bar_colors = [color_map.get(lbl, "#64748b") for lbl in labels]

    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=values,
        marker=dict(color=bar_colors),
        text=values,
        textposition="auto",
        textfont=dict(color="#ffffff", size=13, family=CHART_THEME["font_family"])
    )])

    fig.update_layout(
        title=dict(
            text="<b>Citizen Frustration & Sentiment Polarity</b>",
            font=dict(size=18, color="#ffffff")
        ),
        xaxis=dict(
            title=dict(text="<b>Sentiment Tone</b>", font=dict(size=14, color="#ffffff")),
            tickfont=dict(size=13, color="#ffffff"),
            showgrid=False
        ),
        yaxis=dict(
            title=dict(text="<b>Count</b>", font=dict(size=14, color="#ffffff")),
            tickfont=dict(size=13, color="#ffffff"),
            showgrid=True,
            gridcolor=CHART_THEME["grid_color"]
        ),
        margin=dict(t=50, b=50, l=20, r=20),
        paper_bgcolor=CHART_THEME["background"],
        plot_bgcolor=CHART_THEME["background"],
        font=dict(family=CHART_THEME["font_family"], color="#ffffff")
    )
    return fig

def render_confusion_matrix_heatmap(cm_matrix: list, labels: list) -> go.Figure:
    """Renders interactive multi-class Confusion Matrix Heatmap with white labels."""
    short_labels = [
        lbl.replace(" & ", " &\n") if len(lbl) > 20 else lbl
        for lbl in labels
    ]

    fig = go.Figure(data=go.Heatmap(
        z=cm_matrix,
        x=short_labels,
        y=short_labels,
        colorscale="Blues",
        hoverongaps=False,
        text=cm_matrix,
        texttemplate="%{text}",
        textfont=dict(size=13, color="#ffffff"),
        colorbar=dict(
            title=dict(text="Cases", font=dict(color="#ffffff")),
            tickfont=dict(color="#ffffff")
        )
    ))

    fig.update_layout(
        title=dict(
            text="<b>Model Confusion Matrix (Actual vs Predicted)</b>",
            font=dict(size=18, color="#ffffff")
        ),
        xaxis=dict(
            title=dict(text="<b>Predicted Category</b>", font=dict(size=14, color="#ffffff")),
            tickfont=dict(size=12, color="#ffffff"),
            tickangle=-30
        ),
        yaxis=dict(
            title=dict(text="<b>Actual True Category</b>", font=dict(size=14, color="#ffffff")),
            tickfont=dict(size=12, color="#ffffff"),
            autorange="reversed"
        ),
        margin=dict(t=60, b=90, l=120, r=40),
        paper_bgcolor=CHART_THEME["background"],
        plot_bgcolor=CHART_THEME["background"],
        font=dict(family=CHART_THEME["font_family"], color="#ffffff")
    )
    return fig
