"""
visualizations.py
=================
All Plotly chart functions used across the dashboard pages.
Each function returns a Plotly Figure object ready for st.plotly_chart().
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Colour palette ──────────────────────────────────────────────────────────
PALETTE = {
    "apprehended":  "#2563EB",  # blue
    "cbp":          "#7C3AED",  # violet
    "transferred":  "#059669",  # emerald
    "hhs":          "#DC2626",  # red
    "discharged":   "#D97706",  # amber
    "background":   "#F8FAFC",
    "grid":         "#E2E8F0",
}

COL_COLORS = {
    "Children_Apprehended": PALETTE["apprehended"],
    "Children_in_CBP":      PALETTE["cbp"],
    "Children_Transferred": PALETTE["transferred"],
    "Children_in_HHS":      PALETTE["hhs"],
    "Children_Discharged":  PALETTE["discharged"],
}

FRIENDLY = {
    "Children_Apprehended": "Apprehended",
    "Children_in_CBP":      "In CBP Custody",
    "Children_Transferred": "Transferred",
    "Children_in_HHS":      "In HHS Care",
    "Children_Discharged":  "Discharged",
}

# ─── Layout template ─────────────────────────────────────────────────────────
def _base_layout(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=16, color="#1E293B"), x=0.01),
        paper_bgcolor="white",
        plot_bgcolor=PALETTE["background"],
        font=dict(family="Inter, sans-serif", color="#475569"),
        xaxis=dict(showgrid=False, linecolor="#CBD5E1"),
        yaxis=dict(gridcolor=PALETTE["grid"], linecolor="#CBD5E1"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=11)
        ),
        margin=dict(l=50, r=30, t=60, b=50),
        hovermode="x unified",
    )


# ─── 1. Multi-line trend chart ────────────────────────────────────────────────
def line_chart_multi(df: pd.DataFrame, columns: list, title: str = "Trend Over Time") -> go.Figure:
    """Plots multiple columns as lines over the Date axis."""
    fig = go.Figure()
    for col in columns:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["Date"], y=df[col],
                mode="lines",
                name=FRIENDLY.get(col, col),
                line=dict(color=COL_COLORS.get(col, "#64748B"), width=2),
                hovertemplate="%{y:,.0f}",
            ))
    fig.update_layout(**_base_layout(title))
    return fig


# ─── 2. Single line with 7-day moving average ────────────────────────────────
def line_with_ma(df: pd.DataFrame, column: str, window: int = 7) -> go.Figure:
    """Plots a column with a moving average overlay."""
    ma = df[column].rolling(window=window, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df[column],
        mode="lines", name=FRIENDLY.get(column, column),
        line=dict(color=COL_COLORS.get(column, "#2563EB"), width=1.5, dash="dot"),
        opacity=0.55,
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=ma,
        mode="lines",
        name=f"{window}-Day Moving Avg",
        line=dict(color="#0F172A", width=2.5),
    ))
    title = f"{FRIENDLY.get(column, column)} – {window}-Day Moving Average"
    fig.update_layout(**_base_layout(title))
    return fig


# ─── 3. Bar chart – monthly totals ───────────────────────────────────────────
def bar_monthly(monthly_df: pd.DataFrame, column: str) -> go.Figure:
    """Grouped bar chart of monthly totals/averages."""
    fig = px.bar(
        monthly_df, x="Period", y=column,
        color="Year", barmode="group",
        color_discrete_sequence=px.colors.qualitative.Bold,
        labels={"Period": "Month", column: FRIENDLY.get(column, column)},
        title=f"Monthly Summary – {FRIENDLY.get(column, column)}",
    )
    fig.update_layout(**_base_layout(f"Monthly Summary – {FRIENDLY.get(column, column)}"))
    fig.update_xaxes(tickangle=-45, nticks=24)
    return fig


# ─── 4. Yearly bar chart ─────────────────────────────────────────────────────
def bar_yearly(yearly_df: pd.DataFrame, columns: list) -> go.Figure:
    """Side-by-side bar chart of selected columns across years."""
    fig = go.Figure()
    for col in columns:
        if col in yearly_df.columns:
            fig.add_trace(go.Bar(
                x=yearly_df["Year"].astype(str),
                y=yearly_df[col],
                name=FRIENDLY.get(col, col),
                marker_color=COL_COLORS.get(col, "#64748B"),
            ))
    fig.update_layout(barmode="group", **_base_layout("Year-over-Year Comparison"))
    return fig


# ─── 5. Pie / Donut – composition for a date range ───────────────────────────
def pie_chart(df: pd.DataFrame, columns: list, title: str = "Composition") -> go.Figure:
    """Donut chart showing relative magnitudes of selected averages."""
    labels, values, colors = [], [], []
    for col in columns:
        if col in df.columns:
            labels.append(FRIENDLY.get(col, col))
            values.append(df[col].mean())
            colors.append(COL_COLORS.get(col, "#94A3B8"))

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.42,
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:,.1f} avg<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1E293B"), x=0.01),
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color="#475569"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# ─── 6. Correlation heatmap ──────────────────────────────────────────────────
def correlation_heatmap(df: pd.DataFrame, columns: list) -> go.Figure:
    """Annotated heatmap of Pearson correlations between selected columns."""
    valid = [c for c in columns if c in df.columns]
    corr = df[valid].corr().round(2)
    labels = [FRIENDLY.get(c, c) for c in valid]

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels, y=labels,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=corr.values,
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title="Pearson r"),
        hoverongaps=False,
    ))
    fig.update_layout(
        title=dict(text="Correlation Heatmap", font=dict(size=16, color="#1E293B"), x=0.01),
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=140, r=30, t=60, b=140),
        xaxis=dict(tickangle=-30),
    )
    return fig


# ─── 7. Histogram ────────────────────────────────────────────────────────────
def histogram(df: pd.DataFrame, column: str, bins: int = 30) -> go.Figure:
    """Distribution histogram with KDE overlay."""
    fig = px.histogram(
        df, x=column, nbins=bins,
        color_discrete_sequence=[COL_COLORS.get(column, "#2563EB")],
        labels={column: FRIENDLY.get(column, column)},
        title=f"Distribution – {FRIENDLY.get(column, column)}",
        opacity=0.8,
    )
    fig.update_layout(**_base_layout(f"Distribution – {FRIENDLY.get(column, column)}"))
    return fig


# ─── 8. Box plot by year ──────────────────────────────────────────────────────
def boxplot_by_year(df: pd.DataFrame, column: str) -> go.Figure:
    """Box plots of a column grouped by year, showing spread and outliers."""
    fig = px.box(
        df, x="Year", y=column,
        color="Year",
        color_discrete_sequence=px.colors.qualitative.Bold,
        labels={"Year": "Year", column: FRIENDLY.get(column, column)},
        title=f"Year-wise Distribution – {FRIENDLY.get(column, column)}",
    )
    fig.update_layout(**_base_layout(f"Year-wise Distribution – {FRIENDLY.get(column, column)}"))
    fig.update_xaxes(type="category")
    return fig


# ─── 9. Scatter – two columns ────────────────────────────────────────────────
def scatter_two_cols(df: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    """Scatter plot between two numeric columns, coloured by year."""
    fig = px.scatter(
        df, x=x_col, y=y_col, color="Year",
        color_continuous_scale="Viridis",
        labels={
            x_col: FRIENDLY.get(x_col, x_col),
            y_col: FRIENDLY.get(y_col, y_col),
        },
        title=f"{FRIENDLY.get(x_col, x_col)} vs {FRIENDLY.get(y_col, y_col)}",
        opacity=0.7,
        hover_data={"Date": True},
    )
    fig.update_layout(**_base_layout(
        f"{FRIENDLY.get(x_col, x_col)} vs {FRIENDLY.get(y_col, y_col)}"
    ))
    return fig


# ─── 10. Area chart – stacked ────────────────────────────────────────────────
def area_chart(df: pd.DataFrame, columns: list) -> go.Figure:
    """Stacked area chart showing composition over time."""
    fig = go.Figure()
    for col in columns:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["Date"], y=df[col],
                stackgroup="one",
                name=FRIENDLY.get(col, col),
                fill="tonexty",
                line=dict(color=COL_COLORS.get(col, "#64748B"), width=0.5),
            ))
    fig.update_layout(**_base_layout("Children in System – Stacked Area View"))
    return fig


# ─── 11. Monthly heatmap calendar ────────────────────────────────────────────
def monthly_heatmap(df: pd.DataFrame, column: str) -> go.Figure:
    """Pivot heatmap with Month on x-axis, Year on y-axis."""
    pivot = df.pivot_table(values=column, index="Year", columns="Month", aggfunc="mean")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[month_names[m-1] for m in pivot.columns],
        y=pivot.index.astype(str),
        colorscale="YlOrRd",
        colorbar=dict(title="Avg"),
        hoverongaps=False,
        hovertemplate="Year %{y} | %{x}: %{z:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"Monthly Heatmap – {FRIENDLY.get(column, column)}",
            font=dict(size=16, color="#1E293B"), x=0.01,
        ),
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=60, r=30, t=60, b=50),
        xaxis=dict(side="bottom"),
    )
    return fig
