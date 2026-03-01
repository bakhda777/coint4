from __future__ import annotations

from coint2.ops.config_patch_gates import PatchGateLimits, evaluate_patch_gates


def _factor(target_key: str, op: str, value: object = None) -> dict[str, object]:
    return {"target_key": target_key, "op": op, "value": value}


def test_complexity_gate_blocks_when_over_threshold() -> None:
    factors = [_factor("risk.daily_stop_pct", "set", 0.02)]
    result = evaluate_patch_gates(factors, limits=PatchGateLimits(max_complexity_score=0.0))
    assert result.ok is False
    assert result.complexity_ok is False


def test_redundancy_gate_blocks_when_too_similar_to_zoo() -> None:
    factors = [
        _factor("risk.daily_stop_pct", "set", 0.02),
        _factor("pair_selection.max_pairs", "set", 20),
    ]
    zoo_result = evaluate_patch_gates(factors)
    result = evaluate_patch_gates(
        factors,
        zoo=[("zoo", zoo_result.ast)],
        limits=PatchGateLimits(max_redundancy_similarity=0.5),
    )
    assert result.ok is False
    assert result.redundancy_ok is False

