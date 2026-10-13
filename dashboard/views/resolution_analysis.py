"""Stage 5: Resolution Analysis Page View for NYC 311 Dashboard."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from dashboard.components.charts import (
    create_resolution_donut,
    create_yearly_resolution_trend,
    create_borough_breakdown_bars,
    create_agency_comparison_chart,
    create_complaint_tier_stacked,
    create_resolution_skew_comparison,
)
from dashboard.components.kpis import render_kpi_card
from dashboard.utils.data_loader import (
    load_resolution_distribution,
    load_yearly_resolution,
    load_borough_performance,
    load_agency_performance,
    load_complaint_resolution,
    load_top_complaints,
    load_resolution_stats,
)
from dashboard.utils.validation import format_count, format_hours, format_percentage


def render_resolution_analysis_view() -> None:
    """Render the Resolution Analysis dashboard view."""
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="color: #FFFFFF; font-size: 2.1rem; font-weight: 800; margin-bottom: 6px; letter-spacing: -0.02em;">
                Resolution Analysis & Operational Velocity
            </h1>
            <p style="color: #94A3B8; font-size: 1rem; margin-top: 0;">
                Comprehensive evaluation of municipal turnaround velocity across resolution tiers, administrative boroughs, responsible agencies, and complaint classifications.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        res_dist_df = load_resolution_distribution()
        yearly_res_df = load_yearly_resolution()
        borough_df = load_borough_performance()
        agency_df = load_agency_performance()
        comp_res_df = load_complaint_resolution()
        top_complaints_df = load_top_complaints()
        stats_df = load_resolution_stats()
    except Exception as e:
        st.error(f"Error loading resolution analysis data: {e}")
        return

    # Section 1: Turnaround Statistics & Tail Skewness
    st.markdown(
        """
        <h3 style="color: #38BDF8; font-size: 1.25rem; font-weight: 700; margin-bottom: 4px;">
            1. Resolution Velocity Benchmarks & Distribution Tail
        </h3>
        <p style="color: #CBD5E1; font-size: 0.9rem; margin-top: 0; margin-bottom: 12px;">
            Overall citywide resolution benchmarks derived from 20.6 million service records.
        </p>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card(
            title="Median Turnaround",
            value=f"{stats_df['median_hours'].iloc[0]:.2f} hrs",
            subtitle="P50: 50% resolved under ~7.8 hours",
            badge="Typical Case",
            badge_color="#10B981",
        )
    with col2:
        render_kpi_card(
            title="Mean (Average)",
            value=format_hours(stats_df["average_hours"].iloc[0], include_days=True),
            subtitle="Arithmetic mean pulled up by heavy tail",
            badge="Tail-Skewed",
            badge_color="#3B82F6",
        )
    with col3:
        render_kpi_card(
            title="90th Percentile",
            value=format_hours(stats_df["p90_hours"].iloc[0], include_days=True),
            subtitle="10% of requests take longer than ~16.3 days",
            badge="P90 Threshold",
            badge_color="#F59E0B",
        )
    with col4:
        render_kpi_card(
            title="95th Percentile",
            value=format_hours(stats_df["p95_hours"].iloc[0], include_days=True),
            subtitle="5% of requests take longer than ~37.0 days",
            badge="P95 Outlier",
            badge_color="#EF4444",
        )

    # Resolution Donut & Tail Skew Chart
    row1_left, row1_right = st.columns([1, 1])
    with row1_left:
        st.plotly_chart(
            create_resolution_donut(res_dist_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with row1_right:
        st.plotly_chart(
            create_resolution_skew_comparison(stats_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 24px 0;'>", unsafe_allow_html=True)

    # Section 2: Yearly Average Resolution Trend
    st.markdown(
        """
        <h3 style="color: #38BDF8; font-size: 1.25rem; font-weight: 700; margin-bottom: 4px;">
            2. Multi-Year Turnaround Trends (2020–2026)
        </h3>
        <p style="color: #CBD5E1; font-size: 0.9rem; margin-top: 0; margin-bottom: 12px;">
            Average resolution hours per calendar year. Peak operational backlog occurred in 2023 (188.01h average).
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        create_yearly_resolution_trend(yearly_res_df),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 24px 0;'>", unsafe_allow_html=True)

    # Section 3: Borough Performance Breakdown
    st.markdown(
        """
        <h3 style="color: #38BDF8; font-size: 1.25rem; font-weight: 700; margin-bottom: 4px;">
            3. Borough-Level Performance & Tier Distribution
        </h3>
        <p style="color: #CBD5E1; font-size: 0.9rem; margin-top: 0; margin-bottom: 12px;">
            Analysis of request volume and resolution tier efficiency across New York City's five boroughs.
        </p>
        """,
        unsafe_allow_html=True,
    )

    b_col1, b_col2 = st.columns([3, 2])
    with b_col1:
        st.plotly_chart(
            create_borough_breakdown_bars(borough_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with b_col2:
        st.markdown("**Borough Performance Summary Table:**")
        # Display clean formatted table
        display_borough_df = borough_df[borough_df["total_requests"] > 1000].copy()
        display_borough_df["Volume"] = display_borough_df["total_requests"].apply(lambda v: f"{v:,}")
        display_borough_df["Avg Hours"] = display_borough_df["avg_resolution_hours"].apply(lambda v: f"{v:.1f}h")
        display_borough_df["FAST %"] = display_borough_df["fast_percentage"].apply(lambda v: f"{v:.1f}%")
        display_borough_df["MODERATE %"] = display_borough_df["moderate_percentage"].apply(lambda v: f"{v:.1f}%")
        display_borough_df["SLOW %"] = display_borough_df["slow_percentage"].apply(lambda v: f"{v:.1f}%")

        table_cols = ["borough", "Volume", "Avg Hours", "FAST %", "SLOW %"]
        st.dataframe(
            display_borough_df[table_cols].rename(columns={"borough": "Borough"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Queens achieved the highest FAST share (64.40%), while Manhattan experienced the highest average turnaround (189.42 hours) due to complex tenant housing complaints.")

    st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 24px 0;'>", unsafe_allow_html=True)

    # Section 4: Agency Performance with Contextual Interpretation
    st.markdown(
        """
        <h3 style="color: #38BDF8; font-size: 1.25rem; font-weight: 700; margin-bottom: 4px;">
            4. Agency Performance & Operational Mandate Context
        </h3>
        <p style="color: #CBD5E1; font-size: 0.9rem; margin-top: 0; margin-bottom: 12px;">
            Turnaround times vary dramatically based on the agency's operational mandate and field protocols.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; font-size: 0.88rem; color: #1E40AF;">
            <b>Operational Context Notice:</b> Turnaround comparisons across municipal agencies must not be viewed as a simple 'efficiency rank'. 
            <b>NYPD</b> handles immediate field interventions (parking, noise) with rapid on-scene closure (99.0% FAST, 3.4h avg). In contrast, <b>HPD</b> (Housing Preservation & Development) and <b>DOB</b> (Buildings) investigate building structural integrity, tenant heating violations, and permit infractions requiring physical inspections, owner notices, re-inspections, and legal cures lasting weeks to months.
        </div>
        """,
        unsafe_allow_html=True,
    )

    agency_metric = st.radio(
        "Sort agencies by:",
        options=["avg_resolution_hours", "total_requests"],
        format_func=lambda x: "Average Resolution Time (Hours)" if x == "avg_resolution_hours" else "Total Request Volume",
        horizontal=True,
    )

    st.plotly_chart(
        create_agency_comparison_chart(agency_df, metric=agency_metric),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 24px 0;'>", unsafe_allow_html=True)

    # Section 5: Complaint Type vs. Resolution Category
    st.markdown(
        """
        <h3 style="color: #38BDF8; font-size: 1.25rem; font-weight: 700; margin-bottom: 4px;">
            5. Turnaround Tier Share Across Top Complaint Types
        </h3>
        <p style="color: #CBD5E1; font-size: 0.9rem; margin-top: 0; margin-bottom: 12px;">
            Examining how the nature of the citizen complaint dictates operational velocity.
        </p>
        """,
        unsafe_allow_html=True,
    )

    top_complaint_names = top_complaints_df["complaint_type"].head(12).tolist()
    st.plotly_chart(
        create_complaint_tier_stacked(comp_res_df, top_complaint_names),
        use_container_width=True,
        config={"displayModeBar": False},
    )
