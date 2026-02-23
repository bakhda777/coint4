from __future__ import annotations

import math

import pandas as pd

from coint2.core.performance import compute_coverage_metrics


def test_compute_coverage_metrics_basic_missing_and_zero_days() -> None:
    pnl = pd.Series(
        [1.0, 0.0, -2.0],
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"]),
    )
    metrics = compute_coverage_metrics(pnl, start_date="2024-01-01", end_date="2024-01-05")

    assert metrics["expected_test_days"] == 5.0
    assert metrics["observed_test_days"] == 3.0
    assert math.isclose(float(metrics["coverage_ratio"]), 3.0 / 5.0, rel_tol=1e-12)
    assert metrics["missing_test_days"] == 2.0
    assert metrics["zero_pnl_days"] == 1.0
    assert math.isclose(float(metrics["zero_pnl_days_pct"]), 1.0 / 5.0, rel_tol=1e-12)


def test_compute_coverage_metrics_groups_intraday_rows_by_day() -> None:
    pnl = pd.Series(
        [1.0, -1.0, 2.0],
        index=pd.to_datetime(["2024-01-01T00:00:00", "2024-01-01T12:00:00", "2024-01-03T00:00:00"]),
    )
    metrics = compute_coverage_metrics(pnl, start_date="2024-01-01", end_date="2024-01-03")

    assert metrics["expected_test_days"] == 3.0
    assert metrics["observed_test_days"] == 2.0
    # 2024-01-01 sums to zero after grouping.
    assert metrics["zero_pnl_days"] == 1.0
    assert math.isclose(float(metrics["coverage_ratio"]), 2.0 / 3.0, rel_tol=1e-12)


def test_compute_coverage_metrics_clips_observed_days_to_config_window() -> None:
    pnl = pd.Series(
        [5.0, 1.0, 0.0, -1.0, 7.0],
        index=pd.to_datetime(
            [
                "2024-01-01",  # outside
                "2024-01-02",  # inside
                "2024-01-03",  # inside (zero)
                "2024-01-04",  # inside
                "2024-01-05",  # outside
            ]
        ),
    )
    metrics = compute_coverage_metrics(pnl, start_date="2024-01-02", end_date="2024-01-04")

    assert metrics["expected_test_days"] == 3.0
    assert metrics["observed_test_days"] == 3.0
    assert metrics["coverage_ratio"] == 1.0
    assert metrics["missing_test_days"] == 0.0
    assert metrics["zero_pnl_days"] == 1.0
    assert math.isclose(float(metrics["zero_pnl_days_pct"]), 1.0 / 3.0, rel_tol=1e-12)
