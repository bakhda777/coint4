from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_OPT = REPO_ROOT / "scripts" / "optimization"
if str(SCRIPTS_OPT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_OPT))

from hypothesis_factor_dsl import DEFAULT_SCHEMA_PATH, load_schema, validate_hypothesis_factor_payload  # noqa: E402


def _valid_payload() -> dict:
    return {
        "dsl_version": "pair_crypto_hypothesis_factor.v1",
        "strategy_scope": "pair_crypto",
        "metadata": {
            "generated_by": "unit_test",
            "generated_at": "2026-02-26T00:00:00Z",
            "source": "tests",
        },
        "hypotheses": [
            {
                "hypothesis_id": "HYP-REGIME_01",
                "title": "Regime-aware stop and var tuning",
                "thesis": "Risk caps and stop tuning can reduce worst-window drawdown without degrading OOS Sharpe.",
                "priority": 2,
                "confidence": 0.68,
                "expected_effect": {
                    "sharpe": "up",
                    "drawdown": "down",
                    "turnover": "neutral",
                    "notes": "Primary goal is DD control.",
                },
                "factors": [
                    {
                        "factor_id": "F-RISK_STOP_01",
                        "category": "risk",
                        "target_key": "backtest.pair_stop_loss_usd",
                        "op": "set",
                        "value": 12.0,
                        "bounds": {"lower": 8.0, "upper": 20.0},
                        "rationale": "Bound extreme losses while preserving trade frequency.",
                    }
                ],
                "wfa_checks": [
                    "oos_sharpe",
                    "max_drawdown_on_equity",
                    "worst_window_dd",
                ],
                "linked_papers": ["c50273f8c577"],
            }
        ],
    }


def test_hypothesis_factor_dsl_valid_payload() -> None:
    schema = load_schema(DEFAULT_SCHEMA_PATH)
    payload = _valid_payload()

    ok, errors = validate_hypothesis_factor_payload(payload, schema)

    assert ok, errors
    assert not errors


def test_hypothesis_factor_dsl_invalid_payload() -> None:
    schema = load_schema(DEFAULT_SCHEMA_PATH)
    payload = _valid_payload()
    payload["hypotheses"][0]["expected_effect"]["sharpe"] = "increase"
    del payload["hypotheses"][0]["factors"][0]["target_key"]

    ok, errors = validate_hypothesis_factor_payload(payload, schema)

    assert not ok
    assert errors
    assert any("value not in enum" in item for item in errors)
    assert any("missing required field 'target_key'" in item for item in errors)
