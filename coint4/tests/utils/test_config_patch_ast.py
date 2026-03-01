from __future__ import annotations

from coint2.ops.config_patch_ast import (
    ast_from_factors,
    complexity_score,
    feature_count,
    max_common_subtree_size,
    parameter_count,
    redundancy_similarity,
    symbolic_length,
)


def _factor(target_key: str, op: str, value: object = None) -> dict[str, object]:
    return {"target_key": target_key, "op": op, "value": value}


def test_ast_from_factors_builds_stable_tree() -> None:
    ast = ast_from_factors(
        [
            _factor("risk.daily_stop_pct", "set", 0.02),
            _factor("pair_selection.max_pairs", "set", 20),
        ]
    )
    assert ast.label == "PATCH"
    assert symbolic_length(ast) >= 1


def test_feature_and_parameter_count() -> None:
    factors = [
        _factor("risk.daily_stop_pct", "set", 0.02),
        _factor("risk.deleverage_factor", "scale", 0.8),
        _factor("guards.enable_tail_guard", "enable", None),
        _factor("pair_selection.rank_mode", "set", "spread_std"),
        _factor("risk.enable_something", "set", True),
    ]
    assert feature_count(factors) == 5
    # numeric params: set(0.02), scale(0.8) => 2
    assert parameter_count(factors) == 2


def test_complexity_score_increases_with_more_ops() -> None:
    factors_small = [_factor("risk.daily_stop_pct", "set", 0.02)]
    factors_big = [
        _factor("risk.daily_stop_pct", "set", 0.02),
        _factor("pair_selection.max_pairs", "set", 20),
        _factor("guards.enable_tail_guard", "enable", None),
    ]
    ast_small = ast_from_factors(factors_small)
    ast_big = ast_from_factors(factors_big)
    assert complexity_score(ast=ast_big, factors=factors_big) > complexity_score(ast=ast_small, factors=factors_small)


def test_redundancy_similarity_identical_is_one() -> None:
    factors = [
        _factor("risk.daily_stop_pct", "set", 0.02),
        _factor("pair_selection.max_pairs", "set", 20),
    ]
    a = ast_from_factors(factors)
    b = ast_from_factors(factors)
    assert redundancy_similarity(a, b) == 1.0


def test_max_common_subtree_size_detects_shared_structure() -> None:
    a = ast_from_factors(
        [
            _factor("risk.daily_stop_pct", "set", 0.02),
            _factor("pair_selection.max_pairs", "set", 20),
        ]
    )
    b = ast_from_factors(
        [
            _factor("risk.daily_stop_pct", "offset", -0.01),
            _factor("guards.enable_tail_guard", "enable", None),
        ]
    )
    common = max_common_subtree_size(a, b)
    assert common >= 3  # PATCH -> risk -> daily_stop_pct at least

