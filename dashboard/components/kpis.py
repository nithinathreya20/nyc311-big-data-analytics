"""KPI presentation components for NYC 311 Dashboard."""

from __future__ import annotations

from typing import Optional
import streamlit as st


def render_kpi_card(
    title: str,
    value: str,
    subtitle: Optional[str] = None,
    badge: Optional[str] = None,
    badge_color: str = "#2563EB",
    border_color: str = "#E2E8F0",
) -> None:
    """Render a clean, modern KPI card."""
    badge_html = (
        f'<span style="background-color: {badge_color}15; color: {badge_color}; '
        f'padding: 3px 8px; border-radius: 9999px; font-size: 0.72rem; font-weight: 600; '
        f'letter-spacing: 0.03em; border: 1px solid {badge_color}30;">{badge}</span>'
        if badge
        else ""
    )

    subtitle_html = (
        f'<div style="color: #64748B; font-size: 0.8rem; margin-top: 6px; line-height: 1.3;">{subtitle}</div>'
        if subtitle
        else ""
    )

    card_html = f"""
    <div style="
        background: #FFFFFF;
        border: 1px solid {border_color};
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    ">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
            <div style="color: #475569; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
                {title}
            </div>
            {badge_html}
        </div>
        <div style="color: #0F172A; font-size: 1.85rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.1;">
            {value}
        </div>
        {subtitle_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
