"""Stage 7: Data Provenance, Quality Audit, and Metadata View for NYC 311 Dashboard."""

from __future__ import annotations

import streamlit as st
import pandas as pd


def render_data_info_view() -> None:
    """Render the Data Information & Pipeline Provenance view."""
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="color: #FFFFFF; font-size: 2.1rem; font-weight: 800; margin-bottom: 6px; letter-spacing: -0.02em;">
                Data Provenance, Quality Audit & Definitions
            </h1>
            <p style="color: #94A3B8; font-size: 1rem; margin-top: 0;">
                Comprehensive documentation of source origins, Big Data pipeline cleaning stages, verified schemas, and data quality audits.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Dataset Overview & Scope
    st.markdown(
        """
        <h3 style="color: #38BDF8; font-size: 1.25rem; font-weight: 700; margin-bottom: 8px;">
            1. Source Origin & Coverage Scope
        </h3>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        - **Source:** NYC Open Data Portal (API Endpoint: `https://data.cityofnewyork.us/resource/erm2-nwe9`)
        - **Dataset Name:** NYC 311 Service Requests from 2020 to Present
        - **Total Records Analyzed:** **20,631,296** service request records
        - **Coverage Period:** Calendar years **2020 through 2026**
        - **Partial Year Note:** The **2026** dataset represents a partial year (January through September 2026: 2,655,034 records). Full calendar years cover 2020–2025.
        """
    )

    st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 20px 0;'>", unsafe_allow_html=True)

    # 2. Resolution Tier Definitions
    st.markdown(
        """
        <h3 style="color: #38BDF8; font-size: 1.25rem; font-weight: 700; margin-bottom: 8px;">
            2. Resolution-Time Category Definitions
        </h3>
        """,
        unsafe_allow_html=True,
    )

    tier_data = [
        {"Tier": "FAST", "Operational Window": "< 24 hours", "Overall Share": "58.91% (12,153,125 requests)", "Description": "Same-day municipal response; typical for NYPD parking & noise dispatches."},
        {"Tier": "MODERATE", "Operational Window": "24 to 72 hours", "Overall Share": "14.13% (2,914,222 requests)", "Description": "Standard turnaround; typical for sanitation and street repairs."},
        {"Tier": "SLOW", "Operational Window": "> 72 hours (>3 days)", "Overall Share": "26.97% (5,563,949 requests)", "Description": "Operational backlog or complex multi-week inspections (HPD housing, DOB)."},
    ]
    st.dataframe(pd.DataFrame(tier_data), use_container_width=True, hide_index=True)

    st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 20px 0;'>", unsafe_allow_html=True)

    # 3. Pipeline Data Engineering Stages
    st.markdown(
        """
        <h3 style="color: #38BDF8; font-size: 1.25rem; font-weight: 700; margin-bottom: 8px;">
            3. Distributed Pipeline Stages
        </h3>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ```text
        Raw CSVs (NYC Open Data)
               │
               ▼
        [1. HDFS Storage] ────────► Stored year-wise: /nyc311/raw/full/$YEAR/*.csv
               │
               ▼
        [2. Apache Pig] ──────────► Script: pig/clean_311_final.pig
                                    • Header removal & required field filtering (unique_key, created, closed, agency, complaint)
                                    • Deduplication using unique_key
                                    • Temporal outlier filtering: 2 minutes ≤ duration ≤ 180 days (0.033h ≤ h ≤ 4320h)
                                    • Spatial outlier filtering: NYC bounding box (40.47 ≤ lat ≤ 40.92, -74.26 ≤ lon ≤ -73.70)
                                    • Resolution tier assignment (FAST, MODERATE, SLOW)
                                    • Projected into 29 validated columns
               │
               ▼
        [3. Cleaned HDFS] ────────► Stored year-partitioned: /nyc311/cleaned_final/$YEAR
               │
        ┌──────┴──────┐
        ▼             ▼
  [Apache Hive]   [Apache Spark]
  • Analytics HQL • Feature Engineering (StringIndexer, OneHotEncoder, VectorAssembler)
  • Aggregate     • Spark MLlib: Person A (Logistic Regression) & Person B (Random Forest)
    Dashboard CSVs• Exported evaluation & equity metrics to models/dashboard_metrics.json
        │             │
        └──────┬──────┘
               ▼
      [Streamlit Dashboard]
      • Visualizes precomputed Hive extracts & Spark ML metrics with zero JVM overhead
        ```
        """
    )

    st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 20px 0;'>", unsafe_allow_html=True)

    # 4. Known Issue Investigation Report
    st.markdown(
        """
        <h3 style="color: #38BDF8; font-size: 1.25rem; font-weight: 700; margin-bottom: 8px;">
            4. Investigation Report: 2020–2021 Monthly Aggregation
        </h3>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 16px; font-size: 0.9rem; color: #166534;">
            <b>Audit Verification: RESOLVED in current codebase</b><br>
            • <b>Prior Issue Description:</b> Earlier iterations noted missing monthly values for 2020 and 2021 due to disparate timestamp formatting in NYC Open Data dumps (some records used ISO <code>yyyy-MM-dd'T'HH:mm:ss.SSS</code>, while others used <code>MM/dd/yyyy hh:mm:ss a</code>).<br>
            • <b>Query Resolution:</b> The production Hive query in <code>hive/create_dashboard_data.hql</code> incorporates a <code>COALESCE</code> clause with <code>REGEXP_REPLACE</code> that robustly parses both timestamp variants.<br>
            • <b>Empirical Audit:</b> Verification of the generated <code>dashboard/data/monthly_requests/000000_0</code> proves that all 12 months are fully present for both 2020 (sum = 2,638,197) and 2021 (sum = 2,915,272), exactly matching the yearly totals.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 20px 0;'>", unsafe_allow_html=True)

    # 5. Cleaned Schema Documentation (29 Fields)
    st.markdown(
        """
        <h3 style="color: #38BDF8; font-size: 1.25rem; font-weight: 700; margin-bottom: 8px;">
            5. Cleaned Dataset Schema Reference (29 Fields)
        </h3>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📋 View Complete 29-Column Schema Table", expanded=True):
        schema_records = [
            {"#": 1, "Field": "unique_key", "Type": "STRING", "Role": "Unique identifier, deduplication primary key"},
            {"#": 2, "Field": "created_date", "Type": "STRING", "Role": "Timestamp of service request creation"},
            {"#": 3, "Field": "closed_date", "Type": "STRING", "Role": "Timestamp of closure [Excluded from ML features: Leakage]"},
            {"#": 4, "Field": "agency", "Type": "STRING", "Role": "Responsible municipal agency acronym"},
            {"#": 5, "Field": "agency_name", "Type": "STRING", "Role": "Full administrative agency title"},
            {"#": 6, "Field": "complaint_type", "Type": "STRING", "Role": "Primary classification of municipal incident"},
            {"#": 7, "Field": "descriptor", "Type": "STRING", "Role": "Specific complaint detail"},
            {"#": 8, "Field": "descriptor_2", "Type": "STRING", "Role": "Secondary complaint detail"},
            {"#": 9, "Field": "location_type", "Type": "STRING", "Role": "Facility or property context"},
            {"#": 10, "Field": "incident_zip", "Type": "STRING", "Role": "5-digit NYC postal code"},
            {"#": 11, "Field": "incident_address", "Type": "STRING", "Role": "Street address of incident"},
            {"#": 12, "Field": "street_name", "Type": "STRING", "Role": "Name of street"},
            {"#": 13, "Field": "cross_street_1", "Type": "STRING", "Role": "First cross street boundary"},
            {"#": 14, "Field": "cross_street_2", "Type": "STRING", "Role": "Second cross street boundary"},
            {"#": 15, "Field": "intersection_street_1", "Type": "STRING", "Role": "Intersection street 1"},
            {"#": 16, "Field": "intersection_street_2", "Type": "STRING", "Role": "Intersection street 2"},
            {"#": 17, "Field": "address_type", "Type": "STRING", "Role": "Address type code (ADDRESS, INTERSECTION, BLOCKFACE)"},
            {"#": 18, "Field": "city", "Type": "STRING", "Role": "City name"},
            {"#": 19, "Field": "landmark", "Type": "STRING", "Role": "Named landmark if applicable"},
            {"#": 20, "Field": "facility_type", "Type": "STRING", "Role": "Facility category"},
            {"#": 21, "Field": "status", "Type": "STRING", "Role": "Closure status (Closed, In Progress)"},
            {"#": 22, "Field": "community_board", "Type": "STRING", "Role": "Administrative community board district"},
            {"#": 23, "Field": "council_district", "Type": "STRING", "Role": "NYC city council district"},
            {"#": 24, "Field": "police_precinct", "Type": "STRING", "Role": "NYPD police precinct division"},
            {"#": 25, "Field": "borough", "Type": "STRING", "Role": "NYC Borough name (5 boroughs)"},
            {"#": 26, "Field": "latitude", "Type": "DOUBLE", "Role": "Geographic latitude (40.47 to 40.92)"},
            {"#": 27, "Field": "longitude", "Type": "DOUBLE", "Role": "Geographic longitude (-74.26 to -73.70)"},
            {"#": 28, "Field": "resolution_hours", "Type": "DOUBLE", "Role": "Derived duration in hours [Excluded from ML features: Leakage]"},
            {"#": 29, "Field": "resolution_category", "Type": "STRING", "Role": "Derived target class: FAST, MODERATE, SLOW"},
        ]
        st.dataframe(pd.DataFrame(schema_records), use_container_width=True, hide_index=True)
