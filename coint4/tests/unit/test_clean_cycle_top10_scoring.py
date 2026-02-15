import importlib.util
from pathlib import Path

import pytest


def _load_scoring_module():
    app_root = Path(__file__).resolve().parents[2]
    scoring_path = app_root / "scripts/optimization/clean_cycle_top10/scoring.py"
    assert scoring_path.exists()

    spec = importlib.util.spec_from_file_location("clean_cycle_top10_scoring", scoring_path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_compute_score_formula_and_defaults() -> None:
    scoring = _load_scoring_module()

    assert scoring.DEFAULT_LAMBDA_DD == pytest.approx(0.02)

    score = scoring.compute_score(
        canonical_sharpe=1.0,
        canonical_max_drawdown_abs=-10.0,
        lambda_dd=0.02,
    )
    assert score == pytest.approx(1.0 - 0.02 * 10.0)

    # Abs(dd) is used, so dd sign should not matter.
    score_pos = scoring.compute_score(
        canonical_sharpe=1.0,
        canonical_max_drawdown_abs=10.0,
        lambda_dd=0.02,
    )
    assert score_pos == score

    assert scoring.compute_score(canonical_sharpe=None, canonical_max_drawdown_abs=-1.0, lambda_dd=0.02) is None
    assert scoring.compute_score(canonical_sharpe=1.0, canonical_max_drawdown_abs=None, lambda_dd=0.02) is None

