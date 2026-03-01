from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_OPT = REPO_ROOT / "coint4" / "scripts" / "optimization"
if not SCRIPTS_OPT.exists():
    SCRIPTS_OPT = REPO_ROOT / "scripts" / "optimization"
if str(SCRIPTS_OPT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_OPT))

from semantic_consistency_gate import (  # noqa: E402
    SemanticGateResult,
    deterministic_semantic_check,
    run_semantic_gate,
)


def _factor(target_key: str, op: str, value: object, rationale: str = "ok rationale") -> dict[str, object]:
    return {"target_key": target_key, "op": op, "value": value, "rationale": rationale}


def test_deterministic_semantic_check_accepts_basic_payload() -> None:
    issues = deterministic_semantic_check(
        hypothesis_thesis="Уменьшаем риск хвостовых потерь через более строгий стоп и guards.",
        factors=[
            _factor("risk.daily_stop_pct", "set", 0.02),
            _factor("guards.enable_tail_guard", "enable", None),
        ],
    )
    assert issues == []


def test_deterministic_semantic_check_rejects_duplicates_and_invalid_op() -> None:
    issues = deterministic_semantic_check(
        hypothesis_thesis="Достаточно длинная гипотеза про механику стратегии.",
        factors=[
            _factor("risk.daily_stop_pct", "set", 0.02),
            _factor("risk.daily_stop_pct", "set", 0.03),
            _factor("risk.deleverage_factor", "oops", 0.9),
        ],
    )
    assert any("duplicate target_key" in issue for issue in issues)
    assert any(".op is invalid" in issue for issue in issues)


def test_deterministic_semantic_check_rejects_patch_mismatch_and_extra_keys() -> None:
    issues = deterministic_semantic_check(
        hypothesis_thesis="Достаточно длинная гипотеза про механику стратегии.",
        factors=[_factor("risk.daily_stop_pct", "set", 0.02)],
        materialized_patch={
            "risk": {
                "daily_stop_pct": 0.03,
                "unexpected_key": 1.0,
            }
        },
    )
    assert any("mismatch" in issue for issue in issues)
    assert any("undeclared keys" in issue for issue in issues)


def test_run_semantic_gate_llm_error_is_fail_closed_by_default(monkeypatch) -> None:
    def _fake_llm_semantic_verdict(**_kwargs: object) -> SemanticGateResult:
        return SemanticGateResult(
            ok=False,
            source="llm_codex_exec",
            reasons=(),
            model="gpt-5.2",
            effort="xhigh",
            error="codex exec failed",
        )

    monkeypatch.setattr("semantic_consistency_gate.llm_semantic_verdict", _fake_llm_semantic_verdict)
    result = run_semantic_gate(
        hypothesis={"thesis": "Достаточно длинная гипотеза про механику стратегии."},
        factors=[_factor("risk.daily_stop_pct", "set", 0.02)],
        materialized_patch={"risk": {"daily_stop_pct": 0.02}},
        use_llm=True,
        model="gpt-5.2",
        effort="xhigh",
        codex_bin="codex",
        timeout_sec=30,
        repo_root=REPO_ROOT,
    )
    assert result.ok is False
    assert result.source == "llm_codex_exec_error"
    assert result.error == "codex exec failed"


def test_run_semantic_gate_rejects_patch_without_required_key() -> None:
    result = run_semantic_gate(
        hypothesis={"thesis": "Достаточно длинная гипотеза про механику стратегии."},
        factors=[_factor("risk.daily_stop_pct", "set", 0.02)],
        materialized_patch={"risk": {"other_key": 0.01}},
        use_llm=False,
        model="gpt-5.2",
        effort="xhigh",
        codex_bin="codex",
        timeout_sec=30,
        repo_root=REPO_ROOT,
    )
    assert result.ok is False
    assert result.source == "deterministic"
    assert any("missing target_key" in item for item in result.reasons)


def test_run_semantic_gate_can_be_fail_open_on_llm_error(monkeypatch) -> None:
    def _fake_llm_semantic_verdict(**_kwargs: object) -> SemanticGateResult:
        return SemanticGateResult(
            ok=False,
            source="llm_codex_exec",
            reasons=(),
            model="gpt-5.2",
            effort="xhigh",
            error="codex exec failed",
        )

    monkeypatch.setattr("semantic_consistency_gate.llm_semantic_verdict", _fake_llm_semantic_verdict)
    result = run_semantic_gate(
        hypothesis={"thesis": "Достаточно длинная гипотеза про механику стратегии."},
        factors=[_factor("risk.daily_stop_pct", "set", 0.02)],
        materialized_patch={"risk": {"daily_stop_pct": 0.02}},
        use_llm=True,
        model="gpt-5.2",
        effort="xhigh",
        codex_bin="codex",
        timeout_sec=30,
        repo_root=REPO_ROOT,
        fail_open_on_llm_error=True,
    )
    assert result.ok is True
    assert result.source == "deterministic_fallback"
