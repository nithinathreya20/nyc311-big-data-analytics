"""Stage 4: Request Analytics Page View for NYC 311 Dashboard."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from dashboard.components.charts import (
    create_yearly_requests_bar,
    create_top_complaints_bar,
    create_monthly_trend_line,
    create_hourly_profile_chart,
)
from dashboard.components.kpis import render_kpi_card
from dashboard.utils.data_loader import (
    load_yearly_requests,
    load_top_complaints,
    load_monthly_requests,
    load_hourly_requests,
)
from dashboard.utils.validation import format_count


def render_request_analytics_view() -> None:
    """Render the Request Analytics dashboard view."""
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="color: #FFFFFF; font-size: 2.1rem; font-weight: 800; margin-bottom: 6px; letter-spacing: -0.02em;">
                Request Analytics & Inflow Trends
            </h1>
            <p style="color: #94A3B8; font-size: 1rem; margin-top: 0;">
                Detailed analysis of municipal service request inflow volume across annual, monthly, and hourly temporal scales.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        yearly_df = load_yearly_requests()
        top_complaints_df = load_top_complaints()
        monthly_df = load_monthly_requests()
        hourly_df = load_hourly_requests()
    except Exception as e:
        st.error(f"Error loading request analytics data: {e}")
        return

    # Section 1: Annual Request Volume
    st.markdown(
        """
        <h3 style="color: #38BDF8; font-size: 1.25rem; font-weight: 700; margin-bottom: 4px;">
            1. Annual Inflow Trends (2020–2026)
        </h3>
        <p style="color: #CBD5E1; font-size: 0.9rem; margin-top: 0; margin-bottom: 12px;">
            Tracking the steady growth in municipal service engagement over seven consecutive years.
        </p>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(
            create_yearly_requests_bar(yearly_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with col2:
        max_year_row = yearly_df[yearly_df["request_year"] < 2026].sort_values("total_requests", ascending=False).iloc[0]
        render_kpi_card(
            title="Peak Full Year",
            value=f"{int(max_year_row['request_year'])}",
            subtitle=f"{format_count(max_year_row['total_requests'])} requests",
            badge="Record High",
            badge_color="#2563EB",
        )
        y2026_row = yearly_df[yearly_df["request_year"] == 2026].iloc[0]
        render_kpi_card(
            title="2026 YTD Volume",
            value=format_count(y2026_row["total_requests"], compact=True),
            subtitle=f"{format_count(y2026_row['total_requests'])} through Sep 2026",
            badge="Partial Year",
            badge_color="#F97316",
        )

    st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 24px 0;'>", unsafe_allow_html=True)

    # Section 2: Most Common Municipal Complaint Types
    st.markdown(
        """
        <h3 style="color: #38BDF8; font-size: 1.25rem; font-weight: 700; margin-bottom: 4px;">
            2. Most Common Municipal Complaint Types
        </h3>
        """,
        unsafe_allow_html=True,
    )

    filter_col, _ = st.columns([2, 4])
    with filter_col:
        top_n = st.selectbox(
            "Select ranking depth:",
            options=[10, 15],
            index=1,
            format_func=lambda x: f"Top {x} Complaint Types",
        )

    st.plotly_chart(
        create_top_complaints_bar(top_complaints_df, top_n=top_n),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 24px 0;'>", unsafe_allow_html=True)

    # Section 3: Chronological Monthly Request Trends
    st.markdown(
        """
        <h3 style="color: #38BDF8; font-size: 1.25rem; font-weight: 700; margin-bottom: 4px;">
            3. Monthly Seasonality & Time-Series Patterns
        </h3>
        <p style="color: #CBD5E1; font-size: 0.9rem; margin-top: 0; margin-bottom: 12px;">
            Chronological volume across 81 distinct calendar months. Note distinct seasonal surges in summer months.
        </p>
        """,
        unsafe_allow_html=True,
    )

    m_col1, _ = st.columns([2, 4])
    with m_col1:
        available_years = ["All"] + [str(y) for y in sorted(monthly_df["request_year"].unique())]
        selected_year = st.selectbox(
            "Filter timeline by year:",
            options=available_years,
            index=0,
            help="Select 'All' to view the full continuous 2020-2026 timeline, or a specific calendar year.",
        )

    st.plotly_chart(
        create_monthly_trend_line(monthly_df, selected_year=selected_year),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    if selected_year == "2026":
        st.info("ℹ️ **2026 Data Coverage Notice:** 2026 monthly data encompasses January through September 2026 (Month 9). Q4 (October–December) is not included in the dataset.")

    st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 24px 0;'>", unsafe_allow_html=True)

    # Section 4: Diurnal Hourly Request Cycles
    st.markdown(
        """
        <h3 style="color: #38BDF8; font-size: 1.25rem; font-weight: 700; margin-bottom: 4px;">
            4. Diurnal 24-Hour Cycle Request Distribution
        </h3>
        <p style="color: #CBD5E1; font-size: 0.9rem; margin-top: 0; margin-bottom: 12px;">
            Aggregation of all 20.6 million requests by hour of submission (00:00 to 23:00 EST).
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        create_hourly_profile_chart(hourly_df),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # Operational callouts
    st.markdown(
        """
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px 18px; margin-top: 8px; font-size: 0.88rem; color: #475569;">
            <b>Hourly Peak Highlights:</b>
            <ul style="margin-top: 6px; margin-bottom: 0; padding-left: 20px;">
                <li><b>Primary Daytime Peak:</b> 10:00 AM (1,201,438 requests) and 11:00 AM (1,195,963 requests) represent the municipal operational peak for commercial, parking, and sanitation reporting.</li>
                <li><b>Secondary Evening Peak:</b> A notable surge re-emerges between 21:00 (1.05M) and 22:00 (1.11M), primarily driven by residential noise and parking obstruction complaints filed after work hours.</li>
                <li><b>Overnight Low:</b> Lowest submission activity occurs at 04:00 AM (233,391 requests).</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
