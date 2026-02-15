import math
from pathlib import Path

import numpy as np
import pytest

from coint2.core.sharpe import (
    annualized_sharpe_ratio,
    compute_equity_sharpe_from_equity_curve_csv,
    compute_sharpe_ratio_abs_from_equity_curve_csv,
)


def _write_equity_curve_csv(path: Path, timestamps: list[str], equities: list[float]) -> None:
    # Matches typical pandas.Series.to_csv: empty header for index + column name.
    lines = [",Equity"]
    lines.extend(f"{ts},{eq}" for ts, eq in zip(timestamps, equities, strict=True))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_annualized_sharpe_ratio_matches_sample_std_definition() -> None:
    returns = [0.01, -0.02, 0.03, -0.01, 0.02]
    periods_per_year = 365.0

    expected = math.sqrt(periods_per_year) * np.mean(returns) / np.std(returns, ddof=1)
    actual = annualized_sharpe_ratio(returns, periods_per_year)
    assert actual == pytest.approx(expected, rel=1e-12)


def test_annualized_sharpe_ratio_constant_returns_is_zero() -> None:
    returns = [0.001] * 100
    assert annualized_sharpe_ratio(returns, 365.0) == 0.0


def test_equity_curve_csv_sharpe_matches_rollup_formula(tmp_path: Path) -> None:
    equity_path = tmp_path / "equity_curve.csv"
    _write_equity_curve_csv(
        equity_path,
        timestamps=[
            "2026-01-01 00:00:00",
            "2026-01-01 00:15:00",
            "2026-01-01 00:30:00",
            "2026-01-01 00:45:00",
        ],
        equities=[100.0, 101.0, 102.0, 101.0],
    )

    sharpe = compute_sharpe_ratio_abs_from_equity_curve_csv(equity_path, days_per_year=365.0)
    assert sharpe is not None

    returns = [
        (101.0 - 100.0) / 100.0,
        (102.0 - 101.0) / 101.0,
        (101.0 - 102.0) / 102.0,
    ]
    periods_per_year = 365.0 * (86400.0 / 900.0)  # 15m bars -> 96/day
    expected = math.sqrt(periods_per_year) * np.mean(returns) / np.std(returns, ddof=1)
    assert sharpe == pytest.approx(expected, rel=1e-12)

    stats = compute_equity_sharpe_from_equity_curve_csv(equity_path, days_per_year=365.0)
    assert stats is not None
    assert stats.n_returns == 3
    assert stats.period_seconds_median == pytest.approx(900.0, rel=0, abs=1e-12)


def test_equity_curve_csv_returns_zero_when_std_is_zero(tmp_path: Path) -> None:
    equity_path = tmp_path / "equity_curve.csv"
    _write_equity_curve_csv(
        equity_path,
        timestamps=[
            "2026-01-01 00:00:00",
            "2026-01-01 00:15:00",
            "2026-01-01 00:30:00",
        ],
        equities=[100.0, 100.0, 100.0],
    )

    stats = compute_equity_sharpe_from_equity_curve_csv(equity_path)
    assert stats is not None
    assert stats.sharpe_full == 0.0
    assert compute_sharpe_ratio_abs_from_equity_curve_csv(equity_path) == 0.0


def test_equity_curve_csv_returns_none_when_too_short(tmp_path: Path) -> None:
    equity_path = tmp_path / "equity_curve.csv"
    _write_equity_curve_csv(
        equity_path,
        timestamps=[
            "2026-01-01 00:00:00",
            "2026-01-01 00:15:00",
        ],
        equities=[100.0, 101.0],
    )

    assert compute_equity_sharpe_from_equity_curve_csv(equity_path) is None
    assert compute_sharpe_ratio_abs_from_equity_curve_csv(equity_path) is None

