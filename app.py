"""NYC 311 Service Requests Analytics Dashboard.

Canonical frontend entry point.
Built with Streamlit, Pandas, and Plotly.
Visualizes 20.6M records across Pig -> Hive -> Spark MLlib pipelines.
"""

from __future__ import annotations

import streamlit as st

from dashboard.views.overview import render_overview_view
from dashboard.views.request_analytics import render_request_analytics_view
from dashboard.views.resolution_analysis import render_resolution_analysis_view
from dashboard.views.ml_insights import render_ml_insights_view
from dashboard.views.data_info import render_data_info_view


# Configure Streamlit application
st.set_page_config(
    page_title="NYC 311 Big Data Analytics",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern analytics aesthetics
CUSTOM_CSS = """
<style>
    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Main container padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 1400px;
    }
    
    /* Sidebar aesthetic */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        color: #F8FAFC;
        border-right: 1px solid #1E293B;
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #94A3B8;
    }
    
    /* Navigation radio button custom styling */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 6px;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 8px 12px;
        color: #E2E8F0 !important;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.15s ease;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: #334155;
        border-color: #64748B;
    }
    
    /* High contrast heading styling */
    h1, [data-testid="stMarkdownContainer"] h1 {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }
    
    h2, [data-testid="stMarkdownContainer"] h2,
    h3, [data-testid="stMarkdownContainer"] h3 {
        color: #38BDF8 !important; /* Vibrant Sky Blue */
        font-weight: 700 !important;
        letter-spacing: -0.01em;
    }
    
    h4, [data-testid="stMarkdownContainer"] h4 {
        color: #F1F5F9 !important;
        font-weight: 600 !important;
    }
    
    /* Ensure markdown bold **text** in captions and notes renders crisply */
    [data-testid="stCaptionContainer"] p, [data-testid="stCaptionContainer"] {
        color: #CBD5E1 !important;
        font-size: 0.85rem !important;
    }
    
    [data-testid="stCaptionContainer"] strong {
        color: #F8FAFC !important;
        font-weight: 700 !important;
    }

    /* Metric Card styling */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 16px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    
    /* Hide default Streamlit footer */
    footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_sidebar() -> str:
    """Render the sidebar branding and navigation router."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 8px 0 16px 0; border-bottom: 1px solid #334155; margin-bottom: 18px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.8rem;">🏙️</span>
                    <div>
                        <div style="color: #FFFFFF; font-size: 1.15rem; font-weight: 800; letter-spacing: -0.01em;">NYC 311 ANALYTICS</div>
                        <div style="color: #38BDF8; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;">Big Data Platform</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<p style='font-size: 0.75rem; text-transform: uppercase; font-weight: 700; color: #64748B; letter-spacing: 0.05em; margin-bottom: 8px;'>NAVIGATION</p>", unsafe_allow_html=True)

        nav_options = [
            "📊 Executive Overview",
            "📈 Request Analytics",
            "⏱️ Resolution Analysis",
            "🤖 Machine Learning Insights",
            "ℹ️ Data Information & Quality",
        ]

        selected_page = st.radio(
            label="Navigation Menu",
            options=nav_options,
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("<div style='margin-top: 24px; border-top: 1px solid #334155; padding-top: 16px;'></div>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 0.75rem; text-transform: uppercase; font-weight: 700; color: #64748B; letter-spacing: 0.05em; margin-bottom: 8px;'>PIPELINE METADATA</p>", unsafe_allow_html=True)

        st.markdown(
            """
            <div style="background: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 12px; font-size: 0.78rem; color: #94A3B8; line-height: 1.5;">
                <div>• <b>Engineered:</b> 20.6M Records</div>
                <div>• <b>Coverage:</b> 2020–2026 (Partial)</div>
                <div>• <b>Storage:</b> HDFS Cluster</div>
                <div>• <b>ETL:</b> Apache Pig</div>
                <div>• <b>Analytics:</b> Apache Hive</div>
                <div>• <b>ML:</b> Apache Spark MLlib</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

        if st.button("🔄 Refresh Data Cache", use_container_width=True, help="Clear Streamlit cached data extracts and reload from disk"):
            st.cache_data.clear()
            st.success("Cache refreshed!")
            st.rerun()

    return selected_page


def main() -> None:
    """Main dashboard router."""
    selected_page = render_sidebar()

    try:
        if selected_page == "📊 Executive Overview":
            render_overview_view()
        elif selected_page == "📈 Request Analytics":
            render_request_analytics_view()
        elif selected_page == "⏱️ Resolution Analysis":
            render_resolution_analysis_view()
        elif selected_page == "🤖 Machine Learning Insights":
            render_ml_insights_view()
        elif selected_page == "ℹ️ Data Information & Quality":
            render_data_info_view()
    except Exception as e:
        st.error(
            f"**Error rendering {selected_page}:** {e}\n\n"
            "Please verify that precomputed Hive datasets are present in `dashboard/data/`."
        )

    # Footer
    st.markdown(
        """
        <div style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #E2E8F0; text-align: center; color: #94A3B8; font-size: 0.8rem;">
            NYC 311 Big Data Analytics Platform • Distributed Systems Final Project • Powered by HDFS, Apache Pig, Apache Hive, Apache Spark & Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
