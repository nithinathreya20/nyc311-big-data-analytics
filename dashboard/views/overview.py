"""Stage 3: Executive Overview Page View for NYC 311 Dashboard."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from dashboard.components.charts import (
    create_yearly_requests_bar,
    create_resolution_donut,
    create_top_complaints_bar,
    create_resolution_skew_comparison,
)
from dashboard.components.kpis import render_kpi_card
from dashboard.utils.data_loader import (
    load_yearly_requests,
    load_resolution_distribution,
    load_top_complaints,
    load_resolution_stats,
)
from dashboard.utils.validation import format_count, format_hours, format_percentage, calculate_skew_ratio


def render_overview_view() -> None:
    """Render the Executive Overview dashboard view."""
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <h1 style="color: #FFFFFF; font-size: 2.1rem; font-weight: 800; margin-bottom: 6px; letter-spacing: -0.02em;">
                NYC 311 Executive Overview
            </h1>
            <p style="color: #94A3B8; font-size: 1rem; margin-top: 0;">
                Macro performance indicators and operational health metrics across <b>20.6 million</b> municipal service requests (2020–2026).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Load precomputed aggregates safely
    try:
        yearly_df = load_yearly_requests()
        res_dist_df = load_resolution_distribution()
        top_complaints_df = load_top_complaints()
        stats_df = load_resolution_stats()
    except Exception as e:
        st.error(f"Error loading overview analytics datasets: {e}")
        return

    # Derive macro KPI values dynamically from verified files
    total_requests = yearly_df["total_requests"].sum()
    avg_hours = stats_df["average_hours"].iloc[0]
    median_hours = stats_df["median_hours"].iloc[0]

    fast_row = res_dist_df[res_dist_df["resolution_category"] == "FAST"]
    fast_pct = fast_row["percentage"].iloc[0] if not fast_row.empty else 0.0

    skew_ratio = calculate_skew_ratio(avg_hours, median_hours)

    # 2. Render Macro KPI Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_kpi_card(
            title="Total Requests",
            value=format_count(total_requests, compact=True),
            subtitle=f"{format_count(total_requests)} total records (2020–2026)",
            badge="2026 Partial",
            badge_color="#F97316",
        )

    with col2:
        render_kpi_card(
            title="Median Resolution",
            value=f"{median_hours:.1f} hrs",
            subtitle="50% of complaints resolved under 8 hours",
            badge="P50 Benchmark",
            badge_color="#10B981",
        )

    with col3:
        render_kpi_card(
            title="Average Resolution",
            value=format_hours(avg_hours, include_days=True),
            subtitle=f"{skew_ratio}x higher than median due to heavy tail",
            badge="Tail-Skewed",
            badge_color="#EF4444",
        )

    with col4:
        render_kpi_card(
            title="Fast-Turnaround Share",
            value=format_percentage(fast_pct),
            subtitle="Resolved within 24 hours of submission",
            badge="< 24 Hours",
            badge_color="#2563EB",
        )

    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

    # 3. Macro Visuals Row 1: Annual Volume & Turnaround Tiers
    row1_left, row1_right = st.columns([3, 2])

    with row1_left:
        st.plotly_chart(
            create_yearly_requests_bar(yearly_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption(
            "**Note:** 2026 data reflects partial-year collection (January through September 2026: 2,655,034 requests). Full years cover 2020–2025."
        )

    with row1_right:
        st.plotly_chart(
            create_resolution_donut(res_dist_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption(
            "**Turnaround Tiers:** FAST (<24h: 58.91%), MODERATE (24–72h: 14.13%), SLOW (>72h: 26.97%)."
        )

    st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 24px 0;'>", unsafe_allow_html=True)

    # 4. Macro Visuals Row 2: Top Complaints & Tail Skewness Spotlight
    row2_left, row2_right = st.columns([3, 2])

    with row2_left:
        st.plotly_chart(
            create_top_complaints_bar(top_complaints_df, top_n=10),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with row2_right:
        st.plotly_chart(
            create_resolution_skew_comparison(stats_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown(
            f"""
            <div style="background: #F8FAFC; border-left: 4px solid #3B82F6; padding: 12px 14px; border-radius: 4px; font-size: 0.85rem; color: #334155;">
                <b>Distribution Interpretation:</b> The arithmetic mean (<b>{avg_hours:.1f}h / 6.8d</b>) is drastically higher than the median (<b>{median_hours:.1f}h</b>) because municipal operations feature a heavy right tail: the 95th percentile reaches <b>{stats_df['p95_hours'].iloc[0]:.1f}h (37.0 days)</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 24px 0;'>", unsafe_allow_html=True)

    # 5. Data-Supported Key Insights
    st.markdown(
        """
        <h3 style="color: #38BDF8; font-size: 1.25rem; font-weight: 700; margin-bottom: 12px;">
            Key Analytical Observations
        </h3>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; height: 100%;">
                <div style="font-weight: 700; color: #1E293B; margin-bottom: 6px;">Dominance of Parking & Noise</div>
                <div style="color: #64748B; font-size: 0.88rem; line-height: 1.45;">
                    <i>Illegal Parking</i> (2.90M) and <i>Noise - Residential</i> (2.50M) constitute over <b>26.1%</b> of all 311 service requests citywide. These complaints are dispatched rapidly to the NYPD, resolving in hours.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; height: 100%;">
                <div style="font-weight: 700; color: #1E293B; margin-bottom: 6px;">Bimodal Operational Velocity</div>
                <div style="color: #64748B; font-size: 0.88rem; line-height: 1.45;">
                    Nearly 6 out of 10 requests (58.9%) are closed within 24 hours. However, complex residential issues (e.g. <i>UNSANITARY CONDITION</i> and <i>PLUMBING</i>) handled by HPD routinely exceed 72 hours.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_c:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; height: 100%;">
                <div style="font-weight: 700; color: #1E293B; margin-bottom: 6px;">Growing Municipal Volume</div>
                <div style="color: #64748B; font-size: 0.88rem; line-height: 1.45;">
                    Annual volume grew from 2.64M in 2020 to a record 3.42M in 2025 (+29.7%). With 2.66M requests already logged through September 2026, citizen engagement continues at historic highs.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
