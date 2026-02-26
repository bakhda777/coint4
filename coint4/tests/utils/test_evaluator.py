from __future__ import annotations

import pytest

from coint2.ops.evaluator import (
    CandidateEvaluationInput,
    ObjectiveSpec,
    format_decomposition,
    rank_candidates_v2,
)


def test_rank_candidates_v2_pareto_front_and_utility_order() -> None:
    specs = [
        ObjectiveSpec(name="worst_robust_sh", direction="maximize", weight=0.60),
        ObjectiveSpec(name="worst_dd_pct", direction="minimize", weight=0.30),
        ObjectiveSpec(name="worst_pnl", direction="maximize", weight=0.10),
    ]
    rows = [
        CandidateEvaluationInput(
            candidate_id="a",
            objectives={"worst_robust_sh": 1.20, "worst_dd_pct": 0.15, "worst_pnl": 20.0},
        ),
        CandidateEvaluationInput(
            candidate_id="b",
            objectives={"worst_robust_sh": 1.00, "worst_dd_pct": 0.10, "worst_pnl": 50.0},
        ),
        CandidateEvaluationInput(
            candidate_id="c",
            objectives={"worst_robust_sh": 0.80, "worst_dd_pct": 0.25, "worst_pnl": 10.0},
        ),
    ]

    ranked = rank_candidates_v2(rows, objective_specs=specs)

    assert [item.candidate_id for item in ranked] == ["a", "b", "c"]
    assert ranked[0].pareto_front == 1
    assert ranked[1].pareto_front == 1
    assert ranked[2].pareto_front == 2
    assert ranked[0].utility_score > ranked[1].utility_score > ranked[2].utility_score


def test_rank_candidates_v2_missing_value_is_worst_case() -> None:
    specs = [ObjectiveSpec(name="worst_pnl", direction="maximize", weight=1.0)]
    rows = [
        CandidateEvaluationInput(candidate_id="missing", objectives={"worst_pnl": None}),
        CandidateEvaluationInput(candidate_id="present", objectives={"worst_pnl": 12.0}),
    ]

    ranked = rank_candidates_v2(rows, objective_specs=specs)
    ranked_by_id = {item.candidate_id: item for item in ranked}

    assert ranked_by_id["present"].pareto_front == 1
    assert ranked_by_id["missing"].pareto_front == 2
    assert ranked_by_id["missing"].utility_score == pytest.approx(0.0, abs=1e-12)
    assert ranked_by_id["present"].utility_score == pytest.approx(1.0, abs=1e-12)


def test_format_decomposition_marks_missing_terms() -> None:
    specs = [
        ObjectiveSpec(name="worst_robust_sh", direction="maximize", weight=0.8),
        ObjectiveSpec(name="worst_pnl", direction="maximize", weight=0.2),
    ]
    ranked = rank_candidates_v2(
        [
            CandidateEvaluationInput(
                candidate_id="x",
                objectives={"worst_robust_sh": 1.10, "worst_pnl": None},
            )
        ],
        objective_specs=specs,
    )

    payload = format_decomposition(ranked[0].decomposition, top_n=2)
    assert "worst_robust_sh:" in payload
    assert "worst_pnl:" in payload
    assert "~missing" in payload


def test_rank_candidates_v2_rejects_duplicate_objective_names() -> None:
    specs = [
        ObjectiveSpec(name="worst_robust_sh", direction="maximize"),
        ObjectiveSpec(name="worst_robust_sh", direction="maximize"),
    ]
    rows = [CandidateEvaluationInput(candidate_id="x", objectives={"worst_robust_sh": 1.0})]
    with pytest.raises(ValueError, match="duplicate objective name"):
        rank_candidates_v2(rows, objective_specs=specs)
