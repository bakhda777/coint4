from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_OPT = REPO_ROOT / "coint4" / "scripts" / "optimization"
if not SCRIPTS_OPT.exists():
    SCRIPTS_OPT = REPO_ROOT / "scripts" / "optimization"
if str(SCRIPTS_OPT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_OPT))

from hypothesis_factor_dsl import (  # noqa: E402
    DEFAULT_SCHEMA_PATH,
    load_schema,
    materialize_hypothesis_factor_patch,
    validate_hypothesis_factor_payload,
)


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


def test_materialize_hypothesis_patch_with_base_config() -> None:
    payload = _valid_payload()
    payload["hypotheses"][0]["factors"].append(
        {
            "factor_id": "F-RISK_VAR_01",
            "category": "risk",
            "target_key": "backtest.max_var_multiplier",
            "op": "scale",
            "value": 1.05,
            "bounds": {"lower": 1.0, "upper": 2.0},
            "rationale": "Slightly increase cap while staying in bounds.",
        }
    )
    base_config = {
        "backtest": {
            "pair_stop_loss_usd": 10.0,
            "max_var_multiplier": 1.2,
        }
    }

    patch = materialize_hypothesis_factor_patch(
        payload,
        hypothesis_id="HYP-REGIME_01",
        base_config=base_config,
        base_config_ref="configs/base.yaml",
    )

    assert patch["base_config"] == "configs/base.yaml"
    assert patch["backtest"]["pair_stop_loss_usd"] == 12.0
    assert patch["backtest"]["max_var_multiplier"] == pytest.approx(1.26)


def test_materialize_rejects_unsafe_target_root() -> None:
    payload = _valid_payload()
    payload["hypotheses"][0]["factors"][0]["target_key"] = "logging.debug_level"

    with pytest.raises(ValueError, match="Unsafe target_key root"):
        materialize_hypothesis_factor_patch(payload)


def test_materialize_requires_base_config_for_scale() -> None:
    payload = _valid_payload()
    payload["hypotheses"][0]["factors"][0]["op"] = "scale"
    payload["hypotheses"][0]["factors"][0]["value"] = 0.9

    with pytest.raises(ValueError, match="requires --base-config"):
        materialize_hypothesis_factor_patch(payload)


def test_materialize_rejects_duplicate_target_key() -> None:
    payload = _valid_payload()
    payload["hypotheses"][0]["factors"].append(
        {
            "factor_id": "F-RISK_STOP_02",
            "category": "risk",
            "target_key": "backtest.pair_stop_loss_usd",
            "op": "set",
            "value": 11.0,
            "bounds": {"lower": 8.0, "upper": 20.0},
            "rationale": "Intentional duplicate for safety check.",
        }
    )

    with pytest.raises(ValueError, match="Duplicate factor target_key"):
        materialize_hypothesis_factor_patch(payload)
