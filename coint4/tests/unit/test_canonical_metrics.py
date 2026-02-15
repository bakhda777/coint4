import math
from pathlib import Path

import numpy as np
import pytest

from coint2.core.canonical_metrics import compute_canonical_metrics_from_equity_curve_csv


def _write_equity_curve_csv(path: Path, timestamps: list[str], equities: list[float]) -> None:
    # Matches typical pandas.Series.to_csv: empty header for index + column name.
    lines = [",Equity"]
    lines.extend(f"{ts},{eq}" for ts, eq in zip(timestamps, equities, strict=True))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_compute_canonical_metrics_from_equity_curve_csv(tmp_path: Path) -> None:
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

    periods_per_year = 365.0 * 96.0  # 15m bars -> 96/day
    metrics = compute_canonical_metrics_from_equity_curve_csv(
        equity_path,
        periods_per_year=periods_per_year,
        risk_free_rate=0.0,
    )
    assert metrics is not None

    returns = [
        (101.0 - 100.0) / 100.0,
        (102.0 - 101.0) / 101.0,
        (101.0 - 102.0) / 102.0,
    ]
    expected_sharpe = math.sqrt(periods_per_year) * np.mean(returns) / np.std(returns, ddof=1)

    assert metrics.canonical_sharpe == pytest.approx(expected_sharpe, rel=1e-12)
    assert metrics.canonical_pnl_abs == pytest.approx(1.0, rel=0, abs=1e-12)
    assert metrics.canonical_max_drawdown_abs == pytest.approx(-1.0, rel=0, abs=1e-12)
    assert metrics.period_seconds_median == pytest.approx(900.0, rel=0, abs=1e-12)


def test_compute_canonical_metrics_returns_none_when_missing(tmp_path: Path) -> None:
    assert (
        compute_canonical_metrics_from_equity_curve_csv(
            tmp_path / "missing.csv",
            periods_per_year=365.0,
        )
        is None
    )

