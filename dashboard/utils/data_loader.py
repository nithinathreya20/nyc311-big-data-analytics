"""Data loading utilities for NYC 311 Analytics Dashboard.

Reads and caches precomputed Hive aggregate extracts and Spark ML evaluation metrics.
All paths are resolved relative to the project root.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

# Support running in both Streamlit and pure Python (e.g. CLI tests)
try:
    import streamlit as st
    cache_data = st.cache_data
except ImportError:
    def cache_data(func):  # type: ignore
        """Fallback decorator when Streamlit is not imported."""
        return func


# Resolve project root regardless of terminal working directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "dashboard" / "data"
MODELS_DIR = PROJECT_ROOT / "models"


class DataLoadError(Exception):
    """Raised when an aggregate data file cannot be found or parsed."""


def _find_part_file(directory_path: Path) -> Path:
    """Find the first data part file in a Hive output directory."""
    if not directory_path.exists():
        raise DataLoadError(f"Directory not found: {directory_path}")

    # Hive directory output files typically named 000000_0 or part-*
    candidates = list(directory_path.glob("000000*")) or list(directory_path.glob("part*"))
    if not candidates:
        # Fall back to any non-hidden file
        candidates = [p for p in directory_path.iterdir() if p.is_file() and not p.name.startswith(".")]

    if not candidates:
        raise DataLoadError(f"No data part files found in {directory_path}")

    target_file = candidates[0]
    if target_file.stat().st_size == 0:
        raise DataLoadError(f"Data file is empty (0 bytes): {target_file}")

    return target_file


@cache_data
def load_yearly_requests() -> pd.DataFrame:
    """Load annual request counts (2020-2026)."""
    file_path = _find_part_file(DATA_DIR / "yearly_requests")
    df = pd.read_csv(
        file_path,
        header=None,
        names=["request_year", "total_requests"],
        dtype={"request_year": int, "total_requests": int},
    )
    df = df.sort_values("request_year").reset_index(drop=True)
    df["is_partial_year"] = df["request_year"] == 2026
    return df


@cache_data
def load_resolution_distribution() -> pd.DataFrame:
    """Load overall turnaround tier distribution (FAST, MODERATE, SLOW)."""
    file_path = _find_part_file(DATA_DIR / "resolution_distribution")
    df = pd.read_csv(
        file_path,
        header=None,
        names=["resolution_category", "total_requests", "percentage"],
        dtype={"resolution_category": str, "total_requests": int, "percentage": float},
    )
    # Ensure ordered categories: FAST, MODERATE, SLOW
    order = {"FAST": 0, "MODERATE": 1, "SLOW": 2}
    df["order"] = df["resolution_category"].map(order).fillna(99)
    df = df.sort_values("order").drop(columns=["order"]).reset_index(drop=True)
    return df


@cache_data
def load_yearly_resolution() -> pd.DataFrame:
    """Load yearly average resolution duration in hours."""
    file_path = _find_part_file(DATA_DIR / "yearly_resolution")
    df = pd.read_csv(
        file_path,
        header=None,
        names=["request_year", "avg_resolution_hours"],
        dtype={"request_year": int, "avg_resolution_hours": float},
    )
    df = df.sort_values("request_year").reset_index(drop=True)
    df["is_partial_year"] = df["request_year"] == 2026
    return df


@cache_data
def load_borough_performance() -> pd.DataFrame:
    """Load borough-level volume, average hours, and tier shares."""
    file_path = _find_part_file(DATA_DIR / "borough_performance")
    df = pd.read_csv(
        file_path,
        header=None,
        names=[
            "borough",
            "total_requests",
            "avg_resolution_hours",
            "fast_percentage",
            "moderate_percentage",
            "slow_percentage",
        ],
        dtype={
            "borough": str,
            "total_requests": int,
            "avg_resolution_hours": float,
            "fast_percentage": float,
            "moderate_percentage": float,
            "slow_percentage": float,
        },
    )
    # Clean missing / empty borough labels
    df["borough"] = df["borough"].fillna("Unrecorded").replace("", "Unrecorded")
    df = df.sort_values("total_requests", ascending=False).reset_index(drop=True)
    return df


@cache_data
def load_top_complaints() -> pd.DataFrame:
    """Load top 15 most frequent municipal complaint types."""
    file_path = _find_part_file(DATA_DIR / "top_complaints")
    df = pd.read_csv(
        file_path,
        header=None,
        names=["complaint_type", "total_requests"],
        dtype={"complaint_type": str, "total_requests": int},
    )
    df = df.sort_values("total_requests", ascending=False).reset_index(drop=True)
    return df


@cache_data
def load_complaint_resolution() -> pd.DataFrame:
    """Load cross-tabulation of complaint type by resolution category."""
    file_path = _find_part_file(DATA_DIR / "complaint_resolution")
    df = pd.read_csv(
        file_path,
        header=None,
        names=["complaint_type", "resolution_category", "total_requests"],
        dtype={"complaint_type": str, "resolution_category": str, "total_requests": int},
    )
    return df


@cache_data
def load_agency_performance() -> pd.DataFrame:
    """Load agency performance volume, turnaround time, and tier shares."""
    file_path = _find_part_file(DATA_DIR / "agency_performance")
    df = pd.read_csv(
        file_path,
        header=None,
        names=[
            "agency",
            "total_requests",
            "avg_resolution_hours",
            "fast_percentage",
            "moderate_percentage",
            "slow_percentage",
        ],
        dtype={
            "agency": str,
            "total_requests": int,
            "avg_resolution_hours": float,
            "fast_percentage": float,
            "moderate_percentage": float,
            "slow_percentage": float,
        },
    )
    df = df.sort_values("total_requests", ascending=False).reset_index(drop=True)
    return df


@cache_data
def load_monthly_requests() -> pd.DataFrame:
    """Load chronological monthly request counts."""
    file_path = _find_part_file(DATA_DIR / "monthly_requests")
    df = pd.read_csv(
        file_path,
        header=None,
        names=["request_year", "request_month", "total_requests"],
        dtype={"request_year": int, "request_month": int, "total_requests": int},
    )
    # Sort chronologically by year then month
    df = df.sort_values(["request_year", "request_month"]).reset_index(drop=True)
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    df["month_label"] = df["request_month"].map(month_names)
    df["year_month"] = df["request_year"].astype(str) + "-" + df["request_month"].apply(lambda m: f"{m:02d}")
    df["year_month_label"] = df["month_label"] + " " + df["request_year"].astype(str)
    return df


@cache_data
def load_hourly_requests() -> pd.DataFrame:
    """Load diurnal 24-hour cycle request counts."""
    file_path = _find_part_file(DATA_DIR / "hourly_requests")
    df = pd.read_csv(
        file_path,
        header=None,
        names=["request_hour", "total_requests"],
        dtype={"request_hour": int, "total_requests": int},
    )
    df = df.sort_values("request_hour").reset_index(drop=True)
    df["hour_label"] = df["request_hour"].apply(lambda h: f"{h:02d}:00")
    return df


@cache_data
def load_resolution_stats() -> pd.DataFrame:
    """Load overall summary statistics: mean, median, P90, P95 resolution hours."""
    file_path = _find_part_file(DATA_DIR / "resolution_stats")
    df = pd.read_csv(
        file_path,
        header=None,
        names=["average_hours", "median_hours", "p90_hours", "p95_hours"],
        dtype={
            "average_hours": float,
            "median_hours": float,
            "p90_hours": float,
            "p95_hours": float,
        },
    )
    return df


@cache_data
def load_dashboard_metrics() -> Dict[str, Any]:
    """Load Spark ML evaluation metrics, equity records, and bottlenecks."""
    metrics_path = MODELS_DIR / "dashboard_metrics.json"
    if not metrics_path.exists():
        raise DataLoadError(f"Metrics file not found: {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_datasets() -> Dict[str, Any]:
    """Convenience helper to safely load all datasets and return status mapping."""
    loaders = {
        "yearly_requests": load_yearly_requests,
        "resolution_distribution": load_resolution_distribution,
        "yearly_resolution": load_yearly_resolution,
        "borough_performance": load_borough_performance,
        "top_complaints": load_top_complaints,
        "complaint_resolution": load_complaint_resolution,
        "agency_performance": load_agency_performance,
        "monthly_requests": load_monthly_requests,
        "hourly_requests": load_hourly_requests,
        "resolution_stats": load_resolution_stats,
        "dashboard_metrics": load_dashboard_metrics,
    }
    results = {}
    for name, loader in loaders.items():
        try:
            results[name] = loader()
        except Exception as e:
            results[name] = None
            results[f"{name}_error"] = str(e)
    return results
