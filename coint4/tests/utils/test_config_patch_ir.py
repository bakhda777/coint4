from __future__ import annotations

from coint2.ops.config_patch_ir import validate_patch_ir_payload


def _base_payload() -> dict[str, object]:
    return {
        "ir_version": "config_patch_ast.v1",
        "source": "deterministic_patch_v1",
        "parents": ["cand_prev"],
        "hypothesis": {
            "hypothesis_id": "HYP-01",
            "title": "Test hypothesis",
            "thesis": "Снижаем tail-risk через более строгие risk/guard настройки.",
        },
        "factors": [
            {
                "factor_id": "F-01",
                "category": "risk",
                "target_key": "risk.daily_stop_pct",
                "op": "set",
                "value": 0.02,
                "rationale": "risk clamp",
            },
            {
                "factor_id": "F-02",
                "category": "guard",
                "target_key": "guards.enable_tail_guard",
                "op": "enable",
                "value": None,
                "rationale": "enable tail guard",
            },
        ],
        "materialized_patch": {
            "risk": {"daily_stop_pct": 0.02},
            "guards": {"enable_tail_guard": True},
        },
        "gates": {
            "complexity": {
                "score": 12.5,
                "symbolic_length": 8,
                "parameter_count": 1,
                "feature_count": 2,
            },
            "redundancy": {
                "nearest_id": "cand_old",
                "nearest_similarity": 0.4,
                "nearest_common_subtree": 3,
            },
            "limits": {
                "max_complexity_score": 60.0,
                "max_redundancy_similarity": 0.85,
                "alpha_sl": 1.0,
                "alpha_pc": 1.0,
                "alpha_feat": 1.0,
            },
        },
        "semantic_gate": {
            "ok": True,
            "source": "deterministic",
            "reasons": [],
            "model": None,
            "effort": None,
            "error": None,
        },
    }


def test_validate_patch_ir_accepts_valid_payload() -> None:
    payload = _base_payload()
    issues = validate_patch_ir_payload(payload)
    assert issues == []


def test_validate_patch_ir_rejects_undeclared_patch_keys() -> None:
    payload = _base_payload()
    patch = dict(payload["materialized_patch"])  # type: ignore[index]
    patch["portfolio"] = {"max_active_positions": 12}
    payload["materialized_patch"] = patch
    issues = validate_patch_ir_payload(payload)
    assert any("undeclared keys" in item for item in issues)


def test_validate_patch_ir_requires_reasons_for_semantic_failure() -> None:
    payload = _base_payload()
    payload["semantic_gate"] = {
        "ok": False,
        "source": "deterministic",
        "reasons": [],
    }
    issues = validate_patch_ir_payload(payload)
    assert any("reasons must be non-empty" in item for item in issues)

