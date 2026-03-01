"""Safe mutation operator registry (MVP, v1).

This module implements the MVP operator set described in `docs/evolution_engine.md`:
- mutate_step_v1
- crossover_uniform_v1
- random_restart_v1
- coordinate_sweep_v1

Design goals:
- Fail-closed: only knob-space keys can be produced/changed.
- Deterministic given `numpy.random.Generator(PCG64)` state.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np

from coint2.ops.genome import Genome, GenomeValue, KnobSpec, coerce_genome_value

OperatorKind = Literal[
    "mutate_step_v1",
    "crossover_uniform_v1",
    "random_restart_v1",
    "coordinate_sweep_v1",
]


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    kind: OperatorKind
    parents_required: int
    produces_many: bool
    description: str


OPERATOR_REGISTRY_V1: dict[OperatorKind, OperatorSpec] = {
    "mutate_step_v1": OperatorSpec(
        kind="mutate_step_v1",
        parents_required=1,
        produces_many=False,
        description="Mutate a single knob by +/- step (or flip / change choice).",
    ),
    "crossover_uniform_v1": OperatorSpec(
        kind="crossover_uniform_v1",
        parents_required=2,
        produces_many=False,
        description="Uniform crossover: per-key pick from parent A/B with p=0.5.",
    ),
    "random_restart_v1": OperatorSpec(
        kind="random_restart_v1",
        parents_required=0,
        produces_many=False,
        description="Sample a fresh genome from knob-space priors (uniform).",
    ),
    "coordinate_sweep_v1": OperatorSpec(
        kind="coordinate_sweep_v1",
        parents_required=1,
        produces_many=True,
        description="Deterministic micro-sweep around parent: {v-step, v, v+step} cartesian product (budget-limited).",
    ),
}


def list_operator_specs_v1() -> list[OperatorSpec]:
    return [OPERATOR_REGISTRY_V1[k] for k in sorted(OPERATOR_REGISTRY_V1)]


def apply_operator_v1(
    kind: OperatorKind,
    *,
    parents: Sequence[Mapping[str, GenomeValue]] | None = None,
    knob_space: Sequence[KnobSpec],
    rng: np.random.Generator,
    budget: int = 1,
    params: Mapping[str, Any] | None = None,
) -> list[Genome]:
    spec = OPERATOR_REGISTRY_V1.get(kind)
    if spec is None:
        raise KeyError(f"Unknown operator kind: {kind!r}")
    parents_list = list(parents or [])
    if len(parents_list) != spec.parents_required:
        raise ValueError(f"{kind} requires {spec.parents_required} parents (got {len(parents_list)})")
    params_map = dict(params or {})

    if kind == "mutate_step_v1":
        child = mutate_step_v1(parents_list[0], knob_space=knob_space, rng=rng, params=params_map)
        return [child]
    if kind == "crossover_uniform_v1":
        child = crossover_uniform_v1(
            parents_list[0],
            parents_list[1],
            knob_space=knob_space,
            rng=rng,
            params=params_map,
        )
        return [child]
    if kind == "random_restart_v1":
        child = random_restart_v1(knob_space=knob_space, rng=rng, params=params_map)
        return [child]
    if kind == "coordinate_sweep_v1":
        return coordinate_sweep_v1(
            parents_list[0],
            knob_space=knob_space,
            rng=rng,
            budget=budget,
            params=params_map,
        )
    raise AssertionError(f"Unhandled operator kind: {kind}")


def mutate_step_v1(
    parent: Mapping[str, GenomeValue],
    *,
    knob_space: Sequence[KnobSpec],
    rng: np.random.Generator,
    params: Mapping[str, Any] | None = None,
) -> Genome:
    params_map = dict(params or {})
    sanitized = sanitize_genome(parent, knob_space=knob_space)
    knob_by_key = {spec.key: spec for spec in knob_space}

    forced_key = str(params_map.get("key") or "").strip() or None
    keys_filter = _to_str_list(params_map.get("keys"))
    if forced_key and keys_filter and forced_key not in keys_filter:
        raise ValueError("mutate_step_v1 params conflict: key is not in keys[] filter")

    candidate_specs = _select_mutable_specs(
        knob_space,
        keys_filter=keys_filter,
        forced_key=forced_key,
        numeric_requires_step=True,
    )
    if not candidate_specs:
        raise ValueError("mutate_step_v1: no mutable knobs available")

    chosen = _weighted_choice(candidate_specs, rng=rng)
    spec = knob_by_key[chosen.key]
    current = sanitized.get(spec.key)

    mutated_value = _mutate_value_step(current, spec=spec, rng=rng)
    out = dict(sanitized)
    out[spec.key] = mutated_value
    return sanitize_genome(out, knob_space=knob_space)


def crossover_uniform_v1(
    parent_a: Mapping[str, GenomeValue],
    parent_b: Mapping[str, GenomeValue],
    *,
    knob_space: Sequence[KnobSpec],
    rng: np.random.Generator,
    params: Mapping[str, Any] | None = None,
) -> Genome:
    _ = params  # reserved for future schema; keep stable signature
    a = sanitize_genome(parent_a, knob_space=knob_space)
    b = sanitize_genome(parent_b, knob_space=knob_space)
    out: Genome = {}
    for spec in knob_space:
        av = a.get(spec.key)
        bv = b.get(spec.key)
        if av is None and bv is None:
            out[spec.key] = None
            continue
        if av is None:
            out[spec.key] = bv
            continue
        if bv is None:
            out[spec.key] = av
            continue
        pick_b = bool(rng.integers(0, 2))
        out[spec.key] = bv if pick_b else av
    return sanitize_genome(out, knob_space=knob_space)


def random_restart_v1(
    *,
    knob_space: Sequence[KnobSpec],
    rng: np.random.Generator,
    params: Mapping[str, Any] | None = None,
) -> Genome:
    params_map = dict(params or {})
    keys_filter = _to_str_list(params_map.get("keys"))
    if keys_filter:
        unknown = sorted(set(keys_filter) - {spec.key for spec in knob_space})
        if unknown:
            raise ValueError(f"random_restart_v1: unknown keys in params.keys: {unknown}")

    out: Genome = {}
    for spec in knob_space:
        if keys_filter and spec.key not in keys_filter:
            out[spec.key] = None
            continue
        out[spec.key] = _sample_value(spec, rng=rng)
    return sanitize_genome(out, knob_space=knob_space)


def coordinate_sweep_v1(
    parent: Mapping[str, GenomeValue],
    *,
    knob_space: Sequence[KnobSpec],
    rng: np.random.Generator,
    budget: int,
    params: Mapping[str, Any] | None = None,
) -> list[Genome]:
    params_map = dict(params or {})
    sanitized = sanitize_genome(parent, knob_space=knob_space)
    if budget <= 0:
        return []

    knob_by_key = {spec.key: spec for spec in knob_space}
    requested_keys = _to_str_list(params_map.get("keys"))
    max_keys_raw = params_map.get("max_keys")
    try:
        max_keys = int(max_keys_raw) if max_keys_raw is not None else 2
    except (TypeError, ValueError):
        max_keys = 2
    max_keys = max(1, min(5, max_keys))

    keys: list[str]
    if requested_keys:
        # Preserve order but drop duplicates early to avoid repeated axes.
        deduped: list[str] = []
        seen: set[str] = set()
        for item in requested_keys:
            token = str(item or "").strip()
            if not token or token in seen:
                continue
            seen.add(token)
            deduped.append(token)
        unknown = sorted(set(deduped) - set(knob_by_key))
        if unknown:
            raise ValueError(f"coordinate_sweep_v1: unknown keys in params.keys: {unknown}")
        keys = list(deduped)
        # Respect max_keys even when keys are explicitly provided.
        if len(keys) > max_keys:
            weights = np.asarray(
                [max(0.0, float(knob_by_key[key].weight or 1.0)) for key in keys],
                dtype=float,
            )
            if float(weights.sum()) > 0.0 and np.isfinite(weights).all():
                p = weights / float(weights.sum())
                chosen = rng.choice(len(keys), size=int(max_keys), replace=False, p=p)
            else:
                chosen = rng.choice(len(keys), size=int(max_keys), replace=False)
            idxs = sorted(int(idx) for idx in chosen)
            keys = [keys[idx] for idx in idxs]
    else:
        keys = _default_coordinate_keys(knob_space, max_keys=max_keys)

    sweep_axes: list[tuple[str, list[GenomeValue]]] = []
    for key in keys:
        spec = knob_by_key[key]
        if spec.type == "bool":
            current = sanitized.get(key)
            if current is None:
                values = [False, True]
            else:
                values = _unique([bool(current), not bool(current)])
            sweep_axes.append((key, values))
            continue

        if spec.type not in {"int", "float"}:
            raise ValueError(
                f"coordinate_sweep_v1 only supports numeric/bool knobs (key={key}, type={spec.type})"
            )
        if spec.step is None or float(spec.step) <= 0.0:
            raise ValueError(f"coordinate_sweep_v1 requires step for numeric knob (key={key})")

        current = sanitized.get(key)
        base_value = current
        if base_value is None:
            base_value = _numeric_midpoint(spec)
        values = _numeric_triplet(base_value, spec=spec)
        sweep_axes.append((key, values))

    axes_values = [vals for _, vals in sweep_axes]
    axes_keys = [k for k, _ in sweep_axes]

    out: list[Genome] = []
    seen: set[str] = set()
    for combo in itertools.product(*axes_values):
        candidate = dict(sanitized)
        for key, value in zip(axes_keys, combo, strict=True):
            candidate[key] = value
        normalized = sanitize_genome(candidate, knob_space=knob_space)
        fp = _fingerprint_genome_simple(normalized)
        if fp in seen:
            continue
        seen.add(fp)
        out.append(normalized)
        if len(out) >= budget:
            break

    return out


def sanitize_genome(genome: Mapping[str, GenomeValue], *, knob_space: Sequence[KnobSpec]) -> Genome:
    """Fail-closed normalization: output contains ONLY knob-space keys."""
    out: Genome = {}
    for spec in knob_space:
        raw = genome.get(spec.key)
        coerced = coerce_genome_value(raw, value_type=spec.type, round_to=spec.round_to)
        out[spec.key] = _normalize_value(coerced, spec=spec)
    return out


def _select_mutable_specs(
    knob_space: Sequence[KnobSpec],
    *,
    keys_filter: list[str] | None,
    forced_key: str | None,
    numeric_requires_step: bool,
) -> list[KnobSpec]:
    allowed = set(keys_filter or [])
    out: list[KnobSpec] = []
    for spec in knob_space:
        if forced_key is not None and spec.key != forced_key:
            continue
        if keys_filter and spec.key not in allowed:
            continue
        if spec.type in {"int", "float"}:
            if spec.min is None or spec.max is None:
                continue
            if numeric_requires_step and (spec.step is None or float(spec.step) <= 0.0):
                continue
            out.append(spec)
            continue
        if spec.type == "bool":
            out.append(spec)
            continue
        if spec.type in {"categorical", "string"}:
            if spec.choices and len(spec.choices) >= 2:
                out.append(spec)
            continue
    return out


def _weighted_choice(specs: Sequence[KnobSpec], *, rng: np.random.Generator) -> KnobSpec:
    if not specs:
        raise ValueError("weighted_choice: empty specs")
    weights: list[float] = []
    for spec in specs:
        w = float(spec.weight) if spec.weight is not None else 1.0
        weights.append(max(0.0, w))
    total = float(sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        idx = int(rng.integers(0, len(specs)))
        return specs[idx]
    p = np.asarray([w / total for w in weights], dtype=float)
    idx = int(rng.choice(len(specs), p=p))
    return specs[idx]


def _mutate_value_step(value: GenomeValue, *, spec: KnobSpec, rng: np.random.Generator) -> GenomeValue:
    if spec.type == "bool":
        if value is None:
            return bool(rng.integers(0, 2))
        return not bool(value)

    if spec.type in {"categorical", "string"}:
        choices = list(spec.choices or [])
        if not choices:
            return value
        if value not in choices:
            return choices[int(rng.integers(0, len(choices)))]
        if len(choices) == 1:
            return value
        # Choose a different option.
        idx = int(rng.integers(0, len(choices) - 1))
        options = [c for c in choices if c != value]
        return options[idx % len(options)]

    if spec.type == "int":
        step = _require_int_step(spec)
        cur: int
        if value is None:
            cur = int(_sample_value(spec, rng=rng) or 0)
        else:
            cur = int(value)
        plus = _clip_int(cur + step, spec=spec)
        minus = _clip_int(cur - step, spec=spec)
        pick_plus = bool(rng.integers(0, 2))
        chosen = plus if pick_plus else minus
        if chosen == cur:
            other = minus if pick_plus else plus
            chosen = other
        return chosen

    if spec.type == "float":
        if spec.step is None or float(spec.step) <= 0.0:
            raise ValueError(f"float knob requires step for mutate_step_v1: {spec.key}")
        step = float(spec.step)
        cur: float
        if value is None:
            sampled = _sample_value(spec, rng=rng)
            cur = float(sampled) if sampled is not None else 0.0
        else:
            cur = float(value)
        plus = _clip_float(cur + step, spec=spec)
        minus = _clip_float(cur - step, spec=spec)
        pick_plus = bool(rng.integers(0, 2))
        chosen = plus if pick_plus else minus
        if chosen == cur:
            other = minus if pick_plus else plus
            chosen = other
        return _round_float(chosen, spec=spec)

    return value


def _sample_value(spec: KnobSpec, *, rng: np.random.Generator) -> GenomeValue:
    if spec.type == "bool":
        return bool(rng.integers(0, 2))

    if spec.type in {"categorical", "string"}:
        choices = list(spec.choices or [])
        if not choices:
            return None
        return choices[int(rng.integers(0, len(choices)))]

    if spec.type == "int":
        step = _require_int_step(spec, default=1)
        lo, hi = _require_int_bounds(spec)
        if hi < lo:
            return lo
        n = ((hi - lo) // step) + 1
        idx = int(rng.integers(0, max(1, n)))
        return _clip_int(lo + idx * step, spec=spec)

    if spec.type == "float":
        lo, hi = _require_float_bounds(spec)
        if hi < lo:
            return _round_float(lo, spec=spec)
        step = float(spec.step) if spec.step is not None and float(spec.step) > 0.0 else None
        if step is None:
            value = float(rng.uniform(lo, hi))
            return _round_float(_clip_float(value, spec=spec), spec=spec)
        n = int(math.floor((hi - lo) / step)) + 1
        idx = int(rng.integers(0, max(1, n)))
        value = lo + float(idx) * step
        return _round_float(_clip_float(value, spec=spec), spec=spec)

    raise ValueError(f"Unsupported knob type: {spec.type}")


def _normalize_value(value: GenomeValue, *, spec: KnobSpec) -> GenomeValue:
    if value is None:
        return None

    if spec.type == "bool":
        return bool(value)

    if spec.type in {"categorical", "string"}:
        if spec.choices is None:
            return None
        return value if value in spec.choices else None

    if spec.type == "int":
        if spec.min is None or spec.max is None:
            return None
        lo, hi = _require_int_bounds(spec)
        out = _clip_int(int(value), spec=spec)
        step = _require_int_step(spec, default=1)
        origin = lo
        n = int(round((out - origin) / step))
        snapped = origin + n * step
        return int(min(max(snapped, lo), hi))

    if spec.type == "float":
        if spec.min is None or spec.max is None:
            return None
        lo, hi = _require_float_bounds(spec)
        out = _clip_float(float(value), spec=spec)
        if spec.step is not None and float(spec.step) > 0.0:
            step = float(spec.step)
            origin = lo
            n = round((out - origin) / step)
            out = origin + float(n) * step
            out = min(max(out, lo), hi)
        return _round_float(out, spec=spec)

    return None


def _numeric_midpoint(spec: KnobSpec) -> GenomeValue:
    if spec.type == "int":
        lo, hi = _require_int_bounds(spec)
        mid = int(round((lo + hi) / 2))
        return _clip_int(mid, spec=spec)
    lo, hi = _require_float_bounds(spec)
    return _round_float((lo + hi) / 2.0, spec=spec)


def _numeric_triplet(base: GenomeValue, *, spec: KnobSpec) -> list[GenomeValue]:
    if spec.type == "int":
        step = _require_int_step(spec)
        cur = int(base) if base is not None else int(_numeric_midpoint(spec) or 0)
        vals = [_clip_int(cur - step, spec=spec), _clip_int(cur, spec=spec), _clip_int(cur + step, spec=spec)]
        return _unique(vals)
    if spec.type != "float":
        raise ValueError("numeric_triplet expects numeric spec")
    if spec.step is None or float(spec.step) <= 0.0:
        raise ValueError("numeric_triplet requires step for float")
    step = float(spec.step)
    cur = float(base) if base is not None else float(_numeric_midpoint(spec) or 0.0)
    vals = [
        _round_float(_clip_float(cur - step, spec=spec), spec=spec),
        _round_float(_clip_float(cur, spec=spec), spec=spec),
        _round_float(_clip_float(cur + step, spec=spec), spec=spec),
    ]
    return _unique(vals)


def _default_coordinate_keys(knob_space: Sequence[KnobSpec], *, max_keys: int) -> list[str]:
    numeric = [
        spec
        for spec in knob_space
        if spec.type in {"int", "float"} and spec.step is not None and float(spec.step) > 0.0
    ]
    numeric.sort(key=lambda s: (-(float(s.weight) if s.weight is not None else 1.0), s.key))
    return [spec.key for spec in numeric[:max_keys]]


def _require_int_bounds(spec: KnobSpec) -> tuple[int, int]:
    if spec.min is None or spec.max is None:
        raise ValueError(f"int knob requires min/max bounds: {spec.key}")
    lo = float(spec.min)
    hi = float(spec.max)
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError(f"int knob bounds must be finite: {spec.key}")
    lo_i = int(round(lo))
    hi_i = int(round(hi))
    if not math.isclose(lo, lo_i) or not math.isclose(hi, hi_i):
        raise ValueError(f"int knob bounds must be integer-like: {spec.key}")
    return lo_i, hi_i


def _require_float_bounds(spec: KnobSpec) -> tuple[float, float]:
    if spec.min is None or spec.max is None:
        raise ValueError(f"float knob requires min/max bounds: {spec.key}")
    lo = float(spec.min)
    hi = float(spec.max)
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError(f"float knob bounds must be finite: {spec.key}")
    return lo, hi


def _require_int_step(spec: KnobSpec, *, default: int | None = None) -> int:
    if spec.step is None:
        if default is None:
            raise ValueError(f"int knob requires step: {spec.key}")
        return int(default)
    step = float(spec.step)
    if not math.isfinite(step) or step <= 0.0:
        if default is None:
            raise ValueError(f"int knob step must be positive: {spec.key}")
        return int(default)
    step_i = int(round(step))
    if step_i <= 0 or not math.isclose(step, step_i):
        raise ValueError(f"int knob step must be integer-like: {spec.key}")
    return step_i


def _clip_int(value: int, *, spec: KnobSpec) -> int:
    out = int(value)
    if spec.min is not None:
        out = max(out, int(round(float(spec.min))))
    if spec.max is not None:
        out = min(out, int(round(float(spec.max))))
    return out


def _clip_float(value: float, *, spec: KnobSpec) -> float:
    out = float(value)
    if spec.min is not None:
        out = max(out, float(spec.min))
    if spec.max is not None:
        out = min(out, float(spec.max))
    if out == 0.0:
        out = 0.0
    return out


def _round_float(value: float, *, spec: KnobSpec) -> float:
    out = float(value)
    decimals = spec.round_to
    if decimals is None and spec.step is not None and float(spec.step) > 0.0:
        decimals = _decimals_from_step(float(spec.step))
    if decimals is not None:
        out = round(out, int(decimals))
        if out == 0.0:
            out = 0.0
    return out


def _decimals_from_step(step: float) -> int:
    if not math.isfinite(step) or step <= 0.0:
        return 10
    try:
        d = Decimal(str(step)).normalize()
    except (InvalidOperation, ValueError):
        return 10
    exp = d.as_tuple().exponent
    return int(-exp) if exp < 0 else 0


def _to_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            token = str(item or "").strip()
            if token:
                out.append(token)
        return out
    token = str(value).strip()
    return [token] if token else []


def _unique(items: Iterable[GenomeValue]) -> list[GenomeValue]:
    out: list[GenomeValue] = []
    for item in items:
        if item in out:
            continue
        out.append(item)
    return out


def _fingerprint_genome_simple(genome: Mapping[str, GenomeValue]) -> str:
    # Lightweight stable fingerprint for internal dedupe (operator-level).
    # Engine-level fingerprint uses sha256 canonicalization (see docs).
    parts: list[str] = []
    for key in sorted(genome):
        parts.append(f"{key}={genome.get(key)!r}")
    return "|".join(parts)
