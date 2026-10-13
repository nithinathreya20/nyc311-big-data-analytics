"""Validation and formatting utilities for NYC 311 Dashboard."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


def validate_dataframe(
    df: Optional[pd.DataFrame],
    required_columns: List[str],
    dataset_name: str,
) -> Tuple[bool, Optional[str]]:
    """Validate that a DataFrame is present, non-empty, and possesses required columns.

    Returns:
        (is_valid, error_message)
    """
    if df is None:
        return False, f"Dataset '{dataset_name}' could not be loaded."
    if df.empty:
        return False, f"Dataset '{dataset_name}' contains zero records."
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        return False, f"Dataset '{dataset_name}' is missing expected columns: {missing}"
    return True, None


def format_count(val: int | float, compact: bool = False) -> str:
    """Format integers cleanly with commas or compact abbreviation (M/K)."""
    if pd.isna(val):
        return "N/A"
    if compact:
        if abs(val) >= 1_000_000:
            return f"{val / 1_000_000:.2f}M"
        if abs(val) >= 1_000:
            return f"{val / 1_000:.1f}K"
        return f"{int(val):,}"
    return f"{int(val):,}"


def format_hours(hours: float, include_days: bool = True) -> str:
    """Format duration in hours with optional day equivalent."""
    if pd.isna(hours):
        return "N/A"
    if hours < 1.0:
        minutes = round(hours * 60)
        return f"{hours:.2f} hrs ({minutes} mins)"
    if include_days and hours >= 24.0:
        days = hours / 24.0
        return f"{hours:.2f} hrs ({days:.1f} days)"
    return f"{hours:.2f} hrs"


def format_percentage(pct: float) -> str:
    """Format float percentage with two decimal places."""
    if pd.isna(pct):
        return "N/A"
    return f"{pct:.2f}%"


def calculate_skew_ratio(mean_val: float, median_val: float) -> Optional[float]:
    """Calculate ratio of mean to median to illustrate positive distribution skew."""
    if median_val and median_val > 0:
        return round(mean_val / median_val, 1)
    return None
