"""Comprehensive unit and integration tests for NYC 311 Dashboard data loaders and components."""

from __future__ import annotations

import unittest
import pandas as pd

from dashboard.utils.data_loader import (
    load_yearly_requests,
    load_resolution_distribution,
    load_yearly_resolution,
    load_borough_performance,
    load_top_complaints,
    load_complaint_resolution,
    load_agency_performance,
    load_monthly_requests,
    load_hourly_requests,
    load_resolution_stats,
    load_dashboard_metrics,
)
from dashboard.utils.validation import (
    format_count,
    format_hours,
    format_percentage,
    calculate_skew_ratio,
    validate_dataframe,
)
from dashboard.components.charts import (
    create_yearly_requests_bar,
    create_resolution_donut,
    create_yearly_resolution_trend,
    create_top_complaints_bar,
    create_monthly_trend_line,
    create_hourly_profile_chart,
    create_borough_breakdown_bars,
    create_agency_comparison_chart,
    create_complaint_tier_stacked,
    create_resolution_skew_comparison,
    create_bottlenecks_bubble_chart,
)


class TestDashboardData(unittest.TestCase):
    """Test suite verifying schemas, counts, math consistency, and chart generation."""

    def test_yearly_requests(self):
        df = load_yearly_requests()
        self.assertEqual(len(df), 7)
        self.assertListEqual(list(df["request_year"]), [2020, 2021, 2022, 2023, 2024, 2025, 2026])
        self.assertEqual(df["total_requests"].sum(), 20_631_296)
        self.assertTrue(df.loc[df["request_year"] == 2026, "is_partial_year"].iloc[0])
        self.assertFalse(df.loc[df["request_year"] == 2025, "is_partial_year"].iloc[0])

    def test_resolution_distribution(self):
        df = load_resolution_distribution()
        self.assertEqual(len(df), 3)
        self.assertSetEqual(set(df["resolution_category"]), {"FAST", "MODERATE", "SLOW"})
        self.assertEqual(df["total_requests"].sum(), 20_631_296)
        pct_sum = df["percentage"].sum()
        self.assertAlmostEqual(pct_sum, 100.0, delta=0.1)

    def test_yearly_resolution(self):
        df = load_yearly_resolution()
        self.assertEqual(len(df), 7)
        self.assertTrue((df["avg_resolution_hours"] > 0).all())

    def test_borough_performance(self):
        df = load_borough_performance()
        self.assertEqual(len(df), 7)
        self.assertEqual(df["total_requests"].sum(), 20_631_296)
        # Ensure cleaned borough names do not have empty strings or nulls
        self.assertFalse(df["borough"].isnull().any())
        self.assertFalse((df["borough"] == "").any())

    def test_top_complaints(self):
        df = load_top_complaints()
        self.assertEqual(len(df), 15)
        self.assertEqual(df["complaint_type"].iloc[0], "Illegal Parking")
        self.assertEqual(df["total_requests"].iloc[0], 2_897_951)
        # Ensure descending order
        self.assertTrue(df["total_requests"].is_monotonic_decreasing)

    def test_complaint_resolution(self):
        df = load_complaint_resolution()
        self.assertEqual(len(df), 715)
        self.assertTrue({"FAST", "MODERATE", "SLOW"}.issuperset(set(df["resolution_category"].unique())))

    def test_agency_performance(self):
        df = load_agency_performance()
        self.assertEqual(len(df), 19)
        self.assertEqual(df["total_requests"].sum(), 20_631_296)
        self.assertEqual(df["agency"].iloc[0], "NYPD")
        self.assertEqual(df["total_requests"].iloc[0], 9_682_287)

    def test_monthly_requests_chronology(self):
        df = load_monthly_requests()
        self.assertEqual(len(df), 81)
        self.assertEqual(df["total_requests"].sum(), 20_631_296)
        # Check that 2020 and 2021 have all 12 months
        self.assertEqual(len(df[df["request_year"] == 2020]), 12)
        self.assertEqual(len(df[df["request_year"] == 2021]), 12)
        # Check that 2026 has 9 months
        self.assertEqual(len(df[df["request_year"] == 2026]), 9)

    def test_hourly_requests(self):
        df = load_hourly_requests()
        self.assertEqual(len(df), 24)
        self.assertListEqual(list(df["request_hour"]), list(range(24)))
        self.assertEqual(df["total_requests"].sum(), 20_631_296)

    def test_resolution_stats_skewness(self):
        df = load_resolution_stats()
        self.assertEqual(len(df), 1)
        mean_h = df["average_hours"].iloc[0]
        median_h = df["median_hours"].iloc[0]
        self.assertEqual(median_h, 7.78)
        self.assertEqual(mean_h, 162.17)
        # Confirm severe positive skew
        self.assertGreater(mean_h, median_h * 20)

    def test_dashboard_metrics_json(self):
        payload = load_dashboard_metrics()
        self.assertIn("model_eval", payload)
        self.assertIn("person_a_roc_auc", payload["model_eval"])
        self.assertIn("person_b_weighted_f1", payload["model_eval"])
        self.assertEqual(payload["model_eval"]["person_a_roc_auc"], 0.9161)
        self.assertEqual(payload["model_eval"]["person_b_weighted_f1"], 0.7766)
        self.assertGreater(len(payload["equity"]), 0)
        self.assertGreater(len(payload["bottlenecks"]), 0)

    def test_formatting_helpers(self):
        self.assertEqual(format_count(1_234_567), "1,234,567")
        self.assertEqual(format_count(20_631_296, compact=True), "20.63M")
        self.assertEqual(format_percentage(58.91), "58.91%")
        self.assertEqual(format_hours(0.5), "0.50 hrs (30 mins)")
        self.assertEqual(format_hours(48.0, include_days=True), "48.00 hrs (2.0 days)")
        self.assertEqual(calculate_skew_ratio(162.17, 7.78), 20.8)

    def test_chart_generation(self):
        yearly_df = load_yearly_requests()
        fig_bar = create_yearly_requests_bar(yearly_df)
        self.assertIsNotNone(fig_bar)

        dist_df = load_resolution_distribution()
        fig_donut = create_resolution_donut(dist_df)
        self.assertIsNotNone(fig_donut)

        monthly_df = load_monthly_requests()
        fig_month = create_monthly_trend_line(monthly_df)
        self.assertIsNotNone(fig_month)

        hourly_df = load_hourly_requests()
        fig_hour = create_hourly_profile_chart(hourly_df)
        self.assertIsNotNone(fig_hour)

        borough_df = load_borough_performance()
        fig_borough = create_borough_breakdown_bars(borough_df)
        self.assertIsNotNone(fig_borough)

        agency_df = load_agency_performance()
        fig_agency = create_agency_comparison_chart(agency_df)
        self.assertIsNotNone(fig_agency)

        metrics = load_dashboard_metrics()
        b_df = pd.DataFrame(metrics["bottlenecks"])
        fig_bubble = create_bottlenecks_bubble_chart(b_df)
        self.assertIsNotNone(fig_bubble)


if __name__ == "__main__":
    unittest.main()
