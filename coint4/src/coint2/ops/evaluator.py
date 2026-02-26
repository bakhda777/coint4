"""Evaluator protocol v2 (multi-objective) with deterministic decomposition.

The protocol is intentionally strict and fail-closed:
- missing/non-finite objective values are treated as worst-case for dominance;
- decomposition still remains deterministic via normalized contributions;
- Pareto front is primary rank, utility score is secondary tie-break.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence


ObjectiveDirection = Literal["maximize", "minimize"]


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    """Objective definition for protocol v2."""

    name: str
    direction: ObjectiveDirection
    weight: float = 1.0
    epsilon: float = 0.0

    def __post_init__(self) -> None:
        token = str(self.name).strip()
        if not token:
            raise ValueError("ObjectiveSpec.name must be non-empty")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError(f"ObjectiveSpec.direction must be maximize|minimize (got: {self.direction!r})")
        if not math.isfinite(float(self.weight)) or float(self.weight) < 0.0:
            raise ValueError(f"ObjectiveSpec.weight must be finite and >= 0 (got: {self.weight!r})")
        if not math.isfinite(float(self.epsilon)) or float(self.epsilon) < 0.0:
            raise ValueError(f"ObjectiveSpec.epsilon must be finite and >= 0 (got: {self.epsilon!r})")


@dataclass(frozen=True, slots=True)
class CandidateEvaluationInput:
    """Raw objective payload for one candidate."""

    candidate_id: str
    objectives: Mapping[str, float | int | None]


@dataclass(frozen=True, slots=True)
class ObjectiveComponent:
    """One objective term inside decomposition."""

    name: str
    direction: ObjectiveDirection
    value: float | None
    missing: bool
    normalized: float
    weight: float
    contribution: float


@dataclass(frozen=True, slots=True)
class CandidateEvaluationResult:
    """Protocol v2 result for one candidate."""

    candidate_id: str
    pareto_front: int
    dominated_by: int
    dominates: int
    utility_score: float
    decomposition: tuple[ObjectiveComponent, ...]


@dataclass(frozen=True, slots=True)
class _DominanceState:
    dominated_by: list[int]
    dominates: list[int]
    dominates_edges: list[list[int]]


def rank_candidates_v2(
    candidates: Sequence[CandidateEvaluationInput],
    *,
    objective_specs: Sequence[ObjectiveSpec],
) -> list[CandidateEvaluationResult]:
    """Rank candidates by Pareto dominance + weighted utility.

    The returned list is sorted by:
    1) pareto_front asc,
    2) utility_score desc,
    3) candidate_id asc.
    """

    specs = _validate_specs(objective_specs)
    if not candidates:
        return []

    values = _collect_values(candidates, specs=specs)
    lows, highs = _objective_bounds(values=values, specs=specs)
    effective = _effective_matrix(values=values, specs=specs, lows=lows, highs=highs)
    dominance = _compute_dominance(effective=effective, specs=specs)
    fronts = _pareto_fronts(dominance.dominated_by, dominance.dominates_edges)

    total_weight = float(sum(max(0.0, float(spec.weight)) for spec in specs))
    if total_weight <= 0.0:
        total_weight = 1.0

    out: list[CandidateEvaluationResult] = []
    for idx, candidate in enumerate(candidates):
        components: list[ObjectiveComponent] = []
        utility_sum = 0.0
        for spec in specs:
            value = values[idx][spec.name]
            normalized = _normalize_value(
                value=value,
                missing=value is None,
                direction=spec.direction,
                lo=lows[spec.name],
                hi=highs[spec.name],
            )
            weight = max(0.0, float(spec.weight))
            contribution = weight * normalized
            utility_sum += contribution
            components.append(
                ObjectiveComponent(
                    name=spec.name,
                    direction=spec.direction,
                    value=value,
                    missing=value is None,
                    normalized=normalized,
                    weight=weight,
                    contribution=contribution,
                )
            )

        out.append(
            CandidateEvaluationResult(
                candidate_id=str(candidate.candidate_id),
                pareto_front=int(fronts[idx]),
                dominated_by=int(dominance.dominated_by[idx]),
                dominates=int(dominance.dominates[idx]),
                utility_score=float(utility_sum / total_weight),
                decomposition=tuple(components),
            )
        )

    out.sort(key=lambda row: (row.pareto_front, -row.utility_score, row.candidate_id))
    return out


def format_decomposition(
    components: Sequence[ObjectiveComponent],
    *,
    top_n: int = 3,
) -> str:
    """Format the strongest objective contributions as a compact text payload."""

    if top_n <= 0:
        return ""
    ordered = sorted(
        components,
        key=lambda part: (-abs(float(part.contribution)), part.name),
    )[:top_n]
    chunks: list[str] = []
    for item in ordered:
        suffix = "~missing" if item.missing else ""
        chunks.append(
            "{name}:{norm:.3f}*{weight:.2f}={contrib:.3f}{suffix}".format(
                name=item.name,
                norm=float(item.normalized),
                weight=float(item.weight),
                contrib=float(item.contribution),
                suffix=suffix,
            )
        )
    return ";".join(chunks)


def _validate_specs(specs: Sequence[ObjectiveSpec]) -> list[ObjectiveSpec]:
    if not specs:
        raise ValueError("objective_specs must be non-empty")
    seen: set[str] = set()
    out: list[ObjectiveSpec] = []
    for spec in specs:
        if spec.name in seen:
            raise ValueError(f"duplicate objective name: {spec.name}")
        seen.add(spec.name)
        out.append(spec)
    return out


def _collect_values(
    candidates: Sequence[CandidateEvaluationInput],
    *,
    specs: Sequence[ObjectiveSpec],
) -> list[dict[str, float | None]]:
    rows: list[dict[str, float | None]] = []
    for candidate in candidates:
        payload: dict[str, float | None] = {}
        for spec in specs:
            raw = candidate.objectives.get(spec.name)
            payload[spec.name] = _to_finite_float_or_none(raw)
        rows.append(payload)
    return rows


def _objective_bounds(
    *,
    values: Sequence[Mapping[str, float | None]],
    specs: Sequence[ObjectiveSpec],
) -> tuple[dict[str, float], dict[str, float]]:
    lows: dict[str, float] = {}
    highs: dict[str, float] = {}
    for spec in specs:
        finite = [float(row[spec.name]) for row in values if row.get(spec.name) is not None]
        if not finite:
            lows[spec.name] = 0.0
            highs[spec.name] = 0.0
            continue
        lows[spec.name] = min(finite)
        highs[spec.name] = max(finite)
    return lows, highs


def _effective_matrix(
    *,
    values: Sequence[Mapping[str, float | None]],
    specs: Sequence[ObjectiveSpec],
    lows: Mapping[str, float],
    highs: Mapping[str, float],
) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for row in values:
        effective_row: dict[str, float] = {}
        for spec in specs:
            lo = float(lows[spec.name])
            hi = float(highs[spec.name])
            span = hi - lo
            fallback = span if span > 0.0 else 1.0
            value = row.get(spec.name)
            if value is None:
                if spec.direction == "maximize":
                    effective_row[spec.name] = lo - fallback
                else:
                    effective_row[spec.name] = hi + fallback
                continue
            effective_row[spec.name] = float(value)
        out.append(effective_row)
    return out


def _compute_dominance(
    *,
    effective: Sequence[Mapping[str, float]],
    specs: Sequence[ObjectiveSpec],
) -> _DominanceState:
    n = len(effective)
    dominated_by = [0 for _ in range(n)]
    dominates = [0 for _ in range(n)]
    dominates_edges: list[list[int]] = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            i_dom_j = _dominates(effective[i], effective[j], specs=specs)
            j_dom_i = _dominates(effective[j], effective[i], specs=specs)
            if i_dom_j and not j_dom_i:
                dominates[i] += 1
                dominated_by[j] += 1
                dominates_edges[i].append(j)
            elif j_dom_i and not i_dom_j:
                dominates[j] += 1
                dominated_by[i] += 1
                dominates_edges[j].append(i)

    return _DominanceState(
        dominated_by=dominated_by,
        dominates=dominates,
        dominates_edges=dominates_edges,
    )


def _pareto_fronts(dominated_by: Sequence[int], dominates_edges: Sequence[Sequence[int]]) -> list[int]:
    remaining = set(range(len(dominated_by)))
    dominated_work = list(dominated_by)
    fronts = [0 for _ in dominated_by]
    front = 1

    while remaining:
        current = sorted(idx for idx in remaining if dominated_work[idx] == 0)
        if not current:
            # Fallback for pathological cycles due to numeric noise.
            current = [min(remaining)]
        for idx in current:
            fronts[idx] = front
        remaining.difference_update(current)
        for parent in current:
            for child in dominates_edges[parent]:
                if child not in remaining:
                    continue
                dominated_work[child] = max(0, dominated_work[child] - 1)
        front += 1

    return fronts


def _dominates(a: Mapping[str, float], b: Mapping[str, float], *, specs: Sequence[ObjectiveSpec]) -> bool:
    strictly_better = False
    for spec in specs:
        av = float(a[spec.name])
        bv = float(b[spec.name])
        eps = max(0.0, float(spec.epsilon))
        if spec.direction == "maximize":
            if av + eps < bv:
                return False
            if av > bv + eps:
                strictly_better = True
            continue

        if av - eps > bv:
            return False
        if av < bv - eps:
            strictly_better = True
    return strictly_better


def _normalize_value(
    *,
    value: float | None,
    missing: bool,
    direction: ObjectiveDirection,
    lo: float,
    hi: float,
) -> float:
    if missing:
        return 0.0
    span = float(hi) - float(lo)
    if span <= 0.0:
        return 1.0
    val = float(value)
    if direction == "maximize":
        normalized = (val - float(lo)) / span
    else:
        normalized = (float(hi) - val) / span
    return float(min(1.0, max(0.0, normalized)))


def _to_finite_float_or_none(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out
