"""Gates for ConfigPatch AST candidates (pair-crypto QuantaAlpha parity adaptation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from coint2.ops.config_patch_ast import (
    PatchAstNode,
    ast_from_factors,
    complexity_score,
    feature_count,
    max_common_subtree_size,
    parameter_count,
    redundancy_similarity,
    symbolic_length,
)


@dataclass(frozen=True, slots=True)
class PatchGateLimits:
    max_complexity_score: float = 60.0
    max_redundancy_similarity: float = 0.85
    alpha_sl: float = 1.0
    alpha_pc: float = 1.0
    alpha_feat: float = 1.0


@dataclass(frozen=True, slots=True)
class PatchGateResult:
    ok: bool
    complexity_ok: bool
    redundancy_ok: bool
    complexity: dict[str, float | int]
    redundancy: dict[str, float | int | str | None]
    ast: PatchAstNode


def evaluate_patch_gates(
    factors: Sequence[Mapping[str, Any]],
    *,
    zoo: Sequence[tuple[str, PatchAstNode]] = (),
    limits: PatchGateLimits = PatchGateLimits(),
) -> PatchGateResult:
    ast = ast_from_factors(factors)
    sl = symbolic_length(ast)
    pc = parameter_count(factors)
    feat = feature_count(factors)
    score = complexity_score(
        ast=ast,
        factors=factors,
        alpha_sl=float(limits.alpha_sl),
        alpha_pc=float(limits.alpha_pc),
        alpha_feat=float(limits.alpha_feat),
    )
    complexity_ok = score <= float(limits.max_complexity_score)

    nearest_id: str | None = None
    nearest_similarity: float | None = None
    nearest_common: int | None = None
    for candidate_id, other in zoo:
        sim = redundancy_similarity(ast, other)
        if nearest_similarity is None or sim > nearest_similarity:
            nearest_similarity = float(sim)
            nearest_id = str(candidate_id)
            nearest_common = int(max_common_subtree_size(ast, other))
    redundancy_ok = True
    if nearest_similarity is not None:
        redundancy_ok = float(nearest_similarity) <= float(limits.max_redundancy_similarity)

    ok = bool(complexity_ok and redundancy_ok)
    return PatchGateResult(
        ok=ok,
        complexity_ok=complexity_ok,
        redundancy_ok=redundancy_ok,
        complexity={
            "score": float(score),
            "symbolic_length": int(sl),
            "parameter_count": int(pc),
            "feature_count": int(feat),
        },
        redundancy={
            "nearest_id": nearest_id,
            "nearest_similarity": None if nearest_similarity is None else float(nearest_similarity),
            "nearest_common_subtree": None if nearest_common is None else int(nearest_common),
        },
        ast=ast,
    )

