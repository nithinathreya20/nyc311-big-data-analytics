"""Reusable, accessible Plotly chart builders for NYC 311 Dashboard."""

from __future__ import annotations

from typing import List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Centralized design system constants
FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

TIER_COLORS = {
    "FAST": "#10B981",      # Emerald Green (<24h)
    "MODERATE": "#F59E0B",  # Warm Amber (24-72h)
    "SLOW": "#EF4444",      # Crimson Red (>72h)
}

THEME_COLORS = {
    "primary": "#2563EB",     # Royal Blue
    "primary_light": "#93C5FD",
    "partial_year": "#F97316", # Orange accent for partial year 2026
    "grid": "#F1F5F9",
    "text": "#1E293B",
    "subtext": "#64748B",
    "background": "#FFFFFF",
}


def _apply_standard_layout(
    fig: go.Figure,
    title: Optional[str] = None,
    height: int = 400,
    showlegend: bool = True,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
) -> go.Figure:
    """Apply consistent styling, margins, typography, and hover behavior."""
    fig.update_layout(
        title={
            "text": title or "",
            "font": {"family": FONT_FAMILY, "size": 15, "color": THEME_COLORS["text"], "weight": "bold"},
            "x": 0.01,
            "xanchor": "left",
        } if title else None,
        height=height,
        font={"family": FONT_FAMILY, "color": THEME_COLORS["text"]},
        margin={"l": 40, "r": 30, "t": 50 if title else 25, "b": 40},
        paper_bgcolor=THEME_COLORS["background"],
        plot_bgcolor=THEME_COLORS["background"],
        hovermode="x unified" if xaxis_title else "closest",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 12, "color": THEME_COLORS["subtext"]},
        } if showlegend else {"visible": False},
        xaxis={
            "title": {"text": xaxis_title or "", "font": {"size": 13, "color": "#0F172A", "weight": "bold"}},
            "gridcolor": THEME_COLORS["grid"],
            "showline": True,
            "linecolor": "#CBD5E1",
            "tickfont": {"size": 11, "color": "#334155"},
        },
        yaxis={
            "title": {"text": yaxis_title or "", "font": {"size": 13, "color": "#0F172A", "weight": "bold"}},
            "gridcolor": THEME_COLORS["grid"],
            "showline": True,
            "linecolor": "#CBD5E1",
            "tickfont": {"size": 11, "color": "#334155"},
        },
    )
    return fig


def create_yearly_requests_bar(df: pd.DataFrame) -> go.Figure:
    """Create interactive bar chart of annual request volume (2020-2026)."""
    fig = go.Figure()

    colors = [
        THEME_COLORS["partial_year"] if row.get("is_partial_year", False) else THEME_COLORS["primary"]
        for _, row in df.iterrows()
    ]

    customdata = [
        "Partial Year (Jan-Sep)" if row.get("is_partial_year", False) else "Full Calendar Year"
        for _, row in df.iterrows()
    ]

    fig.add_trace(
        go.Bar(
            x=df["request_year"],
            y=df["total_requests"],
            marker_color=colors,
            marker_line={"color": "#CBD5E1", "width": 1},
            text=df["total_requests"].apply(lambda v: f"{v/1_000_000:.2f}M"),
            textposition="outside",
            customdata=customdata,
            hovertemplate=(
                "<b>Year %{x}</b><br>"
                "Total Requests: %{y:,.0f}<br>"
                "Status: %{customdata}<extra></extra>"
            ),
        )
    )

    fig = _apply_standard_layout(
        fig,
        title="Annual Service Request Volume (2020–2026)",
        height=380,
        showlegend=False,
        xaxis_title="Request Year",
        yaxis_title="Total Service Requests",
    )
    fig.update_xaxes(type="category")
    # Leave room for text labels
    max_y = df["total_requests"].max() * 1.15
    fig.update_yaxes(range=[0, max_y])
    return fig


def create_resolution_donut(df: pd.DataFrame) -> go.Figure:
    """Create donut chart showing turnaround velocity tier distribution."""
    fig = go.Figure()

    colors = [TIER_COLORS.get(cat, "#94A3B8") for cat in df["resolution_category"]]

    fig.add_trace(
        go.Pie(
            labels=df["resolution_category"],
            values=df["total_requests"],
            hole=0.62,
            marker={"colors": colors, "line": {"color": "#FFFFFF", "width": 2}},
            textinfo="label+percent",
            textfont={"size": 13, "family": FONT_FAMILY},
            hovertemplate=(
                "<b>Tier: %{label}</b><br>"
                "Requests: %{value:,.0f}<br>"
                "Share: %{percent}<extra></extra>"
            ),
        )
    )

    fast_row = df[df["resolution_category"] == "FAST"]
    fast_pct = fast_row["percentage"].iloc[0] if not fast_row.empty else 0.0

    fig.update_layout(
        annotations=[{
            "text": f"<b>{fast_pct:.1f}%</b><br><span style='font-size:11px;color:#64748B'>FAST (&lt;24h)</span>",
            "x": 0.5,
            "y": 0.5,
            "font_size": 22,
            "font_family": FONT_FAMILY,
            "showarrow": False,
        }],
        title={
            "text": "Resolution Tier Distribution",
            "font": {"family": FONT_FAMILY, "size": 15, "color": THEME_COLORS["text"], "weight": "bold"},
            "x": 0.01,
            "xanchor": "left",
        },
        height=380,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        paper_bgcolor=THEME_COLORS["background"],
        legend={"orientation": "h", "y": -0.05, "x": 0.5, "xanchor": "center"},
    )
    return fig


def create_yearly_resolution_trend(df: pd.DataFrame) -> go.Figure:
    """Create line/bar chart for yearly average turnaround hours."""
    fig = go.Figure()

    colors = [
        THEME_COLORS["partial_year"] if row.get("is_partial_year", False) else "#0EA5E9"
        for _, row in df.iterrows()
    ]

    fig.add_trace(
        go.Bar(
            x=df["request_year"],
            y=df["avg_resolution_hours"],
            marker_color=colors,
            text=df["avg_resolution_hours"].apply(lambda v: f"{v:.1f}h"),
            textposition="outside",
            hovertemplate="<b>Year %{x}</b><br>Average Resolution: %{y:.2f} hours<extra></extra>",
        )
    )

    fig = _apply_standard_layout(
        fig,
        title="Yearly Average Resolution Time (Hours)",
        height=380,
        showlegend=False,
        xaxis_title="Request Year",
        yaxis_title="Average Hours",
    )
    fig.update_xaxes(type="category")
    max_y = df["avg_resolution_hours"].max() * 1.15
    fig.update_yaxes(range=[0, max_y])
    return fig


def create_top_complaints_bar(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Create horizontal bar chart of top municipal complaint types."""
    sub_df = df.head(top_n).copy()
    # Sort ascending for horizontal bar (highest on top)
    sub_df = sub_df.sort_values("total_requests", ascending=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=sub_df["total_requests"],
            y=sub_df["complaint_type"],
            orientation="h",
            marker_color="#3B82F6",
            text=sub_df["total_requests"].apply(lambda v: f"{v/1_000_000:.2f}M" if v >= 1e6 else f"{v/1_000:.0f}K"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Total Requests: %{x:,.0f}<extra></extra>",
        )
    )

    fig = _apply_standard_layout(
        fig,
        title=f"Top {top_n} Complaint Types by Volume",
        height=max(420, top_n * 26),
        showlegend=False,
        xaxis_title="Total Requests",
        yaxis_title=None,
    )
    fig.update_layout(margin={"l": 180, "r": 50, "t": 50, "b": 40})
    max_x = sub_df["total_requests"].max() * 1.18
    fig.update_xaxes(range=[0, max_x])
    return fig


def create_monthly_trend_line(df: pd.DataFrame, selected_year: Optional[str] = "All") -> go.Figure:
    """Create chronological monthly time series line chart."""
    fig = go.Figure()

    if selected_year and selected_year != "All":
        filtered_df = df[df["request_year"] == int(selected_year)].copy()
        fig.add_trace(
            go.Scatter(
                x=filtered_df["month_label"],
                y=filtered_df["total_requests"],
                mode="lines+markers",
                line={"color": THEME_COLORS["primary"], "width": 3},
                marker={"size": 7, "color": THEME_COLORS["primary"]},
                hovertemplate="<b>%{x}</b><br>Requests: %{y:,.0f}<extra></extra>",
            )
        )
        title = f"Monthly Service Requests in {selected_year}"
        xaxis_title = "Month"
    else:
        fig.add_trace(
            go.Scatter(
                x=df["year_month"],
                y=df["total_requests"],
                mode="lines",
                line={"color": THEME_COLORS["primary"], "width": 2},
                hovertemplate="<b>%{x}</b><br>Requests: %{y:,.0f}<extra></extra>",
            )
        )
        title = "Monthly Service Request Volume (Jan 2020 – Sep 2026)"
        xaxis_title = "Year-Month"

    fig = _apply_standard_layout(
        fig,
        title=title,
        height=380,
        showlegend=False,
        xaxis_title=xaxis_title,
        yaxis_title="Service Requests",
    )
    return fig


def create_hourly_profile_chart(df: pd.DataFrame) -> go.Figure:
    """Create 24-hour diurnal profile bar chart with morning/evening rush highlights."""
    fig = go.Figure()

    # Color code peak hours: 9-14 and 21-22
    colors = []
    for h in df["request_hour"]:
        if 9 <= h <= 14:
            colors.append("#2563EB")  # Primary daytime peak
        elif 21 <= h <= 22:
            colors.append("#7C3AED")  # Secondary late evening peak
        elif 1 <= h <= 5:
            colors.append("#94A3B8")  # Nighttime low
        else:
            colors.append("#60A5FA")

    fig.add_trace(
        go.Bar(
            x=df["hour_label"],
            y=df["total_requests"],
            marker_color=colors,
            hovertemplate="<b>Hour %{x}</b><br>Total Volume: %{y:,.0f}<extra></extra>",
        )
    )

    fig = _apply_standard_layout(
        fig,
        title="Diurnal Hourly Request Pattern (24-Hour Cycle)",
        height=380,
        showlegend=False,
        xaxis_title="Hour of Day (EST)",
        yaxis_title="Total Requests",
    )
    return fig


def create_borough_breakdown_bars(df: pd.DataFrame) -> go.Figure:
    """Create 100% stacked bar chart of turnaround tiers by borough."""
    # Filter out empty or unrecorded rows with very low volumes for cleaner presentation
    plot_df = df[df["total_requests"] > 1000].copy()

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="FAST (<24h)",
            x=plot_df["borough"],
            y=plot_df["fast_percentage"],
            marker_color=TIER_COLORS["FAST"],
            hovertemplate="<b>%{x}</b><br>FAST: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="MODERATE (24-72h)",
            x=plot_df["borough"],
            y=plot_df["moderate_percentage"],
            marker_color=TIER_COLORS["MODERATE"],
            hovertemplate="<b>%{x}</b><br>MODERATE: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="SLOW (>72h)",
            x=plot_df["borough"],
            y=plot_df["slow_percentage"],
            marker_color=TIER_COLORS["SLOW"],
            hovertemplate="<b>%{x}</b><br>SLOW: %{y:.2f}%<extra></extra>",
        )
    )

    fig = _apply_standard_layout(
        fig,
        title="Resolution Tier Share by Borough (%)",
        height=380,
        showlegend=True,
        xaxis_title="Borough",
        yaxis_title="Percentage (%)",
    )
    fig.update_layout(barmode="stack")
    fig.update_yaxes(range=[0, 100])
    return fig


def create_agency_comparison_chart(df: pd.DataFrame, metric: str = "avg_resolution_hours") -> go.Figure:
    """Create agency comparison bar chart."""
    plot_df = df.copy()

    if metric == "avg_resolution_hours":
        plot_df = plot_df.sort_values("avg_resolution_hours", ascending=True)
        fig = go.Figure(
            go.Bar(
                x=plot_df["avg_resolution_hours"],
                y=plot_df["agency"],
                orientation="h",
                marker_color="#0284C7",
                text=plot_df["avg_resolution_hours"].apply(lambda v: f"{v:.1f}h"),
                textposition="outside",
                hovertemplate="<b>Agency: %{y}</b><br>Avg Resolution: %{x:.2f} hrs<extra></extra>",
            )
        )
        title = "Average Resolution Time by Agency (Hours)"
        xaxis_title = "Hours"
        max_x = plot_df["avg_resolution_hours"].max() * 1.15
    else:
        plot_df = plot_df.sort_values("total_requests", ascending=True)
        fig = go.Figure(
            go.Bar(
                x=plot_df["total_requests"],
                y=plot_df["agency"],
                orientation="h",
                marker_color="#2563EB",
                text=plot_df["total_requests"].apply(lambda v: f"{v/1e6:.2f}M" if v >= 1e6 else f"{v/1e3:.0f}K"),
                textposition="outside",
                hovertemplate="<b>Agency: %{y}</b><br>Total Requests: %{x:,.0f}<extra></extra>",
            )
        )
        title = "Total Request Volume by Agency"
        xaxis_title = "Total Requests"
        max_x = plot_df["total_requests"].max() * 1.18

    fig = _apply_standard_layout(
        fig,
        title=title,
        height=max(420, len(plot_df) * 22),
        showlegend=False,
        xaxis_title=xaxis_title,
        yaxis_title=None,
    )
    fig.update_layout(margin={"l": 90, "r": 50, "t": 50, "b": 40})
    fig.update_xaxes(range=[0, max_x])
    return fig


def create_complaint_tier_stacked(
    comp_res_df: pd.DataFrame,
    top_complaints: List[str],
) -> go.Figure:
    """Create stacked bar chart showing resolution tier distribution for top complaints."""
    filtered = comp_res_df[comp_res_df["complaint_type"].isin(top_complaints)].copy()

    # Pivot table: complaint_type x resolution_category -> total_requests
    pivot = filtered.pivot(
        index="complaint_type", columns="resolution_category", values="total_requests"
    ).fillna(0)

    for tier in ["FAST", "MODERATE", "SLOW"]:
        if tier not in pivot.columns:
            pivot[tier] = 0

    # Calculate percentages
    totals = pivot.sum(axis=1)
    pivot_pct = pivot.div(totals, axis=0) * 100

    # Reorder index by FAST percentage descending
    pivot_pct = pivot_pct.sort_values("FAST", ascending=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="FAST (<24h)",
            y=pivot_pct.index,
            x=pivot_pct["FAST"],
            orientation="h",
            marker_color=TIER_COLORS["FAST"],
            hovertemplate="<b>%{y}</b><br>FAST: %{x:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="MODERATE (24-72h)",
            y=pivot_pct.index,
            x=pivot_pct["MODERATE"],
            orientation="h",
            marker_color=TIER_COLORS["MODERATE"],
            hovertemplate="<b>%{y}</b><br>MODERATE: %{x:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="SLOW (>72h)",
            y=pivot_pct.index,
            x=pivot_pct["SLOW"],
            orientation="h",
            marker_color=TIER_COLORS["SLOW"],
            hovertemplate="<b>%{y}</b><br>SLOW: %{x:.2f}%<extra></extra>",
        )
    )

    fig = _apply_standard_layout(
        fig,
        title="Resolution Tiers Across Top Complaints (%)",
        height=max(420, len(top_complaints) * 24),
        showlegend=True,
        xaxis_title="Percentage (%)",
        yaxis_title=None,
    )
    fig.update_layout(barmode="stack", margin={"l": 180, "r": 30, "t": 50, "b": 40})
    fig.update_xaxes(range=[0, 100])
    return fig


def create_resolution_skew_comparison(stats_df: pd.DataFrame) -> go.Figure:
    """Create chart contrasting Median vs Mean vs P90 vs P95 to illustrate severe positive skew."""
    metrics = ["Median (P50)", "Mean (Average)", "90th Percentile", "95th Percentile"]
    values = [
        stats_df["median_hours"].iloc[0],
        stats_df["average_hours"].iloc[0],
        stats_df["p90_hours"].iloc[0],
        stats_df["p95_hours"].iloc[0],
    ]
    colors = ["#10B981", "#3B82F6", "#F59E0B", "#EF4444"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=metrics,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f} hrs ({v/24:.1f} d)" if v >= 24 else f"{v:.2f} hrs" for v in values],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Duration: %{y:.2f} hours<extra></extra>",
        )
    )

    fig = _apply_standard_layout(
        fig,
        title="Overall Turnaround Duration Metrics: Tail Skew Comparison",
        height=380,
        showlegend=False,
        xaxis_title=None,
        yaxis_title="Resolution Time (Hours)",
    )
    max_y = max(values) * 1.15
    fig.update_yaxes(range=[0, max_y])
    return fig


def create_bottlenecks_bubble_chart(b_df: pd.DataFrame) -> go.Figure:
    """Create high-contrast bubble chart for operational bottlenecks with explicit axis annotations."""
    df = b_df.copy()
    if "breach_pct" not in df.columns and "breach_rate" in df.columns:
        df["breach_pct"] = df["breach_rate"].apply(lambda v: v if v > 1.0 else v * 100.0)

    # Scale bubble sizes nicely: min 14px, max 42px
    min_vol = df["volume"].min()
    max_vol = df["volume"].max()
    size_scaled = 14 + (df["volume"] - min_vol) / (max_vol - min_vol) * 28

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["avg_hours"],
            y=df["breach_pct"],
            mode="markers",
            marker={
                "size": size_scaled,
                "color": df["breach_pct"],
                "colorscale": [
                    [0.0, "#FED7AA"],   # Warm light orange
                    [0.3, "#F97316"],   # Bright orange
                    [0.6, "#EF4444"],   # Crimson red
                    [1.0, "#7F1D1D"],   # Deep dark red
                ],
                "showscale": True,
                "colorbar": {
                    "title": {"text": "<b>SLA Breach %</b>", "font": {"size": 12, "color": "#0F172A", "family": FONT_FAMILY}},
                    "tickfont": {"size": 11, "color": "#1E293B", "family": FONT_FAMILY},
                    "ticksuffix": "%",
                    "len": 0.85,
                    "thickness": 16,
                },
                "line": {"color": "#450A0A", "width": 1.5},
            },
            customdata=list(
                zip(
                    df["complaint_type"],
                    df["avg_hours"] / 24.0,
                    df["volume"],
                )
            ),
            hovertemplate=(
                "<b style='font-size:14px;'>%{customdata[0]}</b><br><br>"
                "• <b>SLA Breach Rate:</b> %{y:.2f}% (Taking &gt;72h)<br>"
                "• <b>Average Resolution:</b> %{x:.1f} hours (~%{customdata[1]:.1f} days)<br>"
                "• <b>Total Requests:</b> %{customdata[2]:,}<extra></extra>"
            ),
        )
    )

    # Add text annotations for key severe bottleneck complaints
    top_severe = df[df["breach_pct"] > 75].head(5)
    for _, row in top_severe.iterrows():
        fig.add_annotation(
            x=row["avg_hours"],
            y=row["breach_pct"],
            text=f"<b>{row['complaint_type']}</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.2,
            arrowcolor="#991B1B",
            ax=30,
            ay=-22,
            font={"size": 10.5, "color": "#7F1D1D", "family": FONT_FAMILY},
            bgcolor="rgba(255, 255, 255, 0.92)",
            bordercolor="#FCA5A5",
            borderwidth=1,
            borderpad=3,
        )

    # Annotate high-volume low-breach benchmark (Illegal Parking)
    parking_row = df[df["complaint_type"].str.contains("Parking", case=False, na=False)]
    if not parking_row.empty:
        p_r = parking_row.iloc[0]
        fig.add_annotation(
            x=p_r["avg_hours"],
            y=p_r["breach_pct"],
            text=f"<b>{p_r['complaint_type']} (High Vol, Rapid)</b>",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#15803D",
            ax=60,
            ay=-18,
            font={"size": 10.5, "color": "#15803D", "family": FONT_FAMILY},
            bgcolor="rgba(255, 255, 255, 0.92)",
            bordercolor="#86EFAC",
            borderwidth=1,
            borderpad=3,
        )

    fig.update_layout(
        title={
            "text": "<b>Operational Bottlenecks: Resolution Duration vs SLA Breach Rate</b>",
            "font": {"family": FONT_FAMILY, "size": 15, "color": "#0F172A"},
            "x": 0.01,
            "xanchor": "left",
        },
        height=480,
        font={"family": FONT_FAMILY, "color": "#0F172A"},
        margin={"l": 65, "r": 30, "t": 60, "b": 60},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis={
            "title": {
                "text": "<b>Horizontal Axis (X): Average Resolution Duration (Hours)</b>  <i>[Further Right = Slower Turnaround]</i>",
                "font": {"size": 12.5, "color": "#0F172A", "family": FONT_FAMILY},
            },
            "gridcolor": "#F1F5F9",
            "showline": True,
            "linecolor": "#94A3B8",
            "linewidth": 1.5,
            "tickfont": {"size": 11, "color": "#1E293B", "family": FONT_FAMILY},
        },
        yaxis={
            "title": {
                "text": "<b>Vertical Axis (Y): SLA Breach Rate (%)</b>  <i>[Higher Up = Worse Failure Rate (>72h)]</i>",
                "font": {"size": 12.5, "color": "#0F172A", "family": FONT_FAMILY},
            },
            "gridcolor": "#F1F5F9",
            "showline": True,
            "linecolor": "#94A3B8",
            "linewidth": 1.5,
            "tickfont": {"size": 11, "color": "#1E293B", "family": FONT_FAMILY},
            "range": [-6, 105],
        },
    )
    return fig


def create_confusion_matrix_heatmap(cm_data: dict) -> go.Figure:
    """Create an interactive heatmap for the Binary SLA Breach Classifier confusion matrix."""
    tn = cm_data.get("true_negatives", 142104)
    fp = cm_data.get("false_positives", 8512)
    fn = cm_data.get("false_negatives", 19478)
    tp = cm_data.get("true_positives", 36180)
    total = max(1, tn + fp + fn + tp)

    z_vals = [[tn, fp], [fn, tp]]
    text_vals = [
        [
            f"<b>True Negative (TN)</b><br>{tn:,}<br>({tn/total*100:.1f}%)",
            f"<b>False Positive (FP)</b><br>{fp:,}<br>({fp/total*100:.1f}%)",
        ],
        [
            f"<b>False Negative (FN)</b><br>{fn:,}<br>({fn/total*100:.1f}%)",
            f"<b>True Positive (TP)</b><br>{tp:,}<br>({tp/total*100:.1f}%)",
        ],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals,
            x=["Predicted: Met SLA (≤72h)", "Predicted: Breach (>72h)"],
            y=["Actual: Met SLA (≤72h)", "Actual: Breach (>72h)"],
            text=text_vals,
            texttemplate="%{text}",
            textfont={"size": 13, "family": FONT_FAMILY, "color": "#0F172A"},
            colorscale=[[0, "#E2E8F0"], [0.2, "#BAE6FD"], [0.6, "#7DD3FC"], [1.0, "#0284C7"]],
            showscale=False,
            hoverongaps=False,
        )
    )

    fig = _apply_standard_layout(
        fig,
        title=f"Person A Binary Confusion Matrix (Test Split: N={total:,})",
        height=320,
        showlegend=False,
    )
    fig.update_xaxes(side="top", tickfont={"size": 12, "family": FONT_FAMILY})
    fig.update_yaxes(autorange="reversed", tickfont={"size": 12, "family": FONT_FAMILY})
    return fig


def create_multiclass_confusion_matrix_heatmap(cm_data: dict) -> go.Figure:
    """Create an interactive 3x3 heatmap for Person B Multi-Class Velocity Tier Classifier."""
    labels = cm_data.get("labels", ["FAST", "MODERATE", "SLOW"])
    matrix = cm_data.get("matrix", [[109844, 4276, 7718], [6920, 9481, 12847], [8580, 3730, 43166]])
    total = sum(sum(row) for row in matrix) or 1

    axis_labels = ["FAST (<24h)", "MODERATE (24–72h)", "SLOW (>72h)"]

    text_vals = []
    for i, row in enumerate(matrix):
        row_text = []
        row_sum = sum(row) or 1
        for j, val in enumerate(row):
            pct_row = val / row_sum * 100
            match_mark = " ✓" if i == j else ""
            row_text.append(f"<b>{val:,}</b>{match_mark}<br>({pct_row:.1f}% of actual)")
        text_vals.append(row_text)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[f"Pred: {lbl}" for lbl in axis_labels],
            y=[f"Actual: {lbl}" for lbl in axis_labels],
            text=text_vals,
            texttemplate="%{text}",
            textfont={"size": 12, "family": FONT_FAMILY, "color": "#0F172A"},
            colorscale=[[0, "#F8FAFC"], [0.15, "#E0F2FE"], [0.5, "#7DD3FC"], [1.0, "#0284C7"]],
            showscale=False,
            hoverongaps=False,
        )
    )

    fig = _apply_standard_layout(
        fig,
        title=f"Person B 3×3 Resolution Tier Confusion Matrix (Test Split: N={total:,})",
        height=350,
        showlegend=False,
    )
    fig.update_xaxes(side="top", tickfont={"size": 11, "family": FONT_FAMILY})
    fig.update_yaxes(autorange="reversed", tickfont={"size": 11, "family": FONT_FAMILY})
    return fig



def create_tier_probability_bar(tier_probs: dict) -> go.Figure:
    """Create a horizontal bar chart displaying predicted velocity tier probabilities."""
    tiers = ["FAST (<24h)", "MODERATE (24–72h)", "SLOW (>72h)"]
    probs = [
        tier_probs.get("FAST", 0.0) * 100,
        tier_probs.get("MODERATE", 0.0) * 100,
        tier_probs.get("SLOW", 0.0) * 100,
    ]
    colors = ["#10B981", "#F59E0B", "#EF4444"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=tiers,
            x=probs,
            orientation="h",
            marker={"color": colors, "line": {"color": "#CBD5E1", "width": 1}},
            text=[f"{p:.1f}%" for p in probs],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Predicted Probability: %{x:.2f}%<extra></extra>",
        )
    )

    fig = _apply_standard_layout(
        fig,
        title="Person B: Multi-Class Probability Distribution",
        height=240,
        showlegend=False,
        xaxis_title="Confidence Probability (%)",
        yaxis_title="Resolution Velocity Tier",
    )
    fig.update_xaxes(range=[0, 115])
    fig.update_yaxes(autorange="reversed")
    return fig


