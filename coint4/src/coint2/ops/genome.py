"""Genome/fingerprint/similarity helpers for YAML configs.

This module is intended for optimization tooling:
- Stable fingerprints for YAML configs (with optional base_config materialization).
- A compact "genome" view (dotted keys -> scalar values).
- A similarity/distance metric for genomes to de-dupe near-duplicates and limit knob jitter.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, MutableMapping, Sequence

import yaml

GenomeValue = float | int | str | bool | None
Genome = dict[str, GenomeValue]

ValueType = Literal["int", "float", "bool", "categorical", "string"]
NormMode = Literal["range", "step", "none"]

DEFAULT_FLOAT_ROUND = 10


@dataclass(frozen=True, slots=True)
class KnobSpec:
    key: str
    type: ValueType
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[GenomeValue] | None = None
    round_to: int | None = None
    weight: float | None = None
    norm: NormMode | None = None

    def resolve_weight(self, *, overrides: Mapping[str, float] | None = None) -> float:
        if overrides is not None and self.key in overrides:
            return float(overrides[self.key])
        if self.weight is None:
            return 1.0
        return float(self.weight)

    def resolve_norm(self, *, default: NormMode) -> NormMode:
        if self.norm is None:
            return default
        return self.norm


def knob_specs_from_dicts(items: Sequence[Mapping[str, Any]]) -> list[KnobSpec]:
    out: list[KnobSpec] = []
    for idx, raw in enumerate(items):
        if not isinstance(raw, Mapping):
            raise TypeError(f"knob_specs[{idx}] must be a mapping")
        key = str(raw.get("key") or "").strip()
        if not key:
            raise ValueError(f"knob_specs[{idx}] missing 'key'")
        raw_type = str(raw.get("type") or "").strip()
        if raw_type not in {"int", "float", "bool", "categorical", "string"}:
            raise ValueError(f"knob_specs[{idx}] invalid type={raw_type!r}")
        norm_raw = raw.get("norm")
        norm: NormMode | None
        if norm_raw is None:
            norm = None
        else:
            norm = str(norm_raw)
            if norm not in {"range", "step", "none"}:
                raise ValueError(f"knob_specs[{idx}] invalid norm={norm_raw!r}")
        out.append(
            KnobSpec(
                key=key,
                type=raw_type,  # type: ignore[arg-type]
                min=_to_optional_float(raw.get("min")),
                max=_to_optional_float(raw.get("max")),
                step=_to_optional_float(raw.get("step")),
                choices=_to_optional_choices(raw.get("choices")),
                round_to=_to_optional_int(raw.get("round_to")),
                weight=_to_optional_float(raw.get("weight")),
                norm=norm,
            )
        )
    return out


def load_effective_yaml_config(path: Path | str) -> dict[str, Any]:
    """Load YAML config and materialize base_config chain (if present).

    Rules follow project conventions:
    - base_config can be absolute, relative to current file, or relative to CWD.
    - In override configs, `null` means "no override" for keys present in base.
    - Returned config never includes `base_config`.
    """

    root = Path(path) if isinstance(path, str) else path
    return _load_raw_config(root)


def fingerprint_yaml_config(
    path: Path | str,
    *,
    float_round: int | None = DEFAULT_FLOAT_ROUND,
) -> str:
    """Stable sha256 fingerprint for an effective YAML config (base_config materialized)."""

    cfg = load_effective_yaml_config(path)
    return fingerprint_object(cfg, float_round=float_round)


def fingerprint_object(obj: Any, *, float_round: int | None = DEFAULT_FLOAT_ROUND) -> str:
    canonical = canonicalize_object(obj, float_round=float_round)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def genome_from_config(
    cfg: Mapping[str, Any],
    *,
    knob_space: Sequence[KnobSpec],
    float_round: int | None = DEFAULT_FLOAT_ROUND,
    include_missing: bool = True,
) -> Genome:
    genome: Genome = {}
    for spec in knob_space:
        try:
            raw_value = get_dotted(cfg, spec.key)
            present = True
        except KeyError:
            raw_value = None
            present = False
        if not present and not include_missing:
            continue
        genome[spec.key] = coerce_genome_value(
            raw_value,
            value_type=spec.type,
            float_round=float_round,
            round_to=spec.round_to,
        )
    return genome


def fingerprint_genome(genome: Mapping[str, GenomeValue], *, float_round: int | None = DEFAULT_FLOAT_ROUND) -> str:
    return fingerprint_object(dict(genome), float_round=float_round)


def genome_from_yaml_config(
    path: Path | str,
    *,
    knob_space: Sequence[KnobSpec],
    float_round: int | None = DEFAULT_FLOAT_ROUND,
    include_missing: bool = True,
) -> Genome:
    cfg = load_effective_yaml_config(path)
    return genome_from_config(cfg, knob_space=knob_space, float_round=float_round, include_missing=include_missing)


def fingerprint_yaml_genome(
    path: Path | str,
    *,
    knob_space: Sequence[KnobSpec],
    float_round: int | None = DEFAULT_FLOAT_ROUND,
    include_missing: bool = True,
) -> str:
    return fingerprint_genome(
        genome_from_yaml_config(path, knob_space=knob_space, float_round=float_round, include_missing=include_missing),
        float_round=float_round,
    )


def genome_distance_weighted_l1_v1(
    a: Mapping[str, GenomeValue],
    b: Mapping[str, GenomeValue],
    *,
    knob_space: Sequence[KnobSpec],
    normalization: NormMode = "range",
    key_weights: Mapping[str, float] | None = None,
) -> float:
    """Weighted L1 distance over the provided knob_space.

    Smaller is more similar. 0.0 means identical on all knob_space keys.
    """

    total = 0.0
    for spec in knob_space:
        key = spec.key
        w = spec.resolve_weight(overrides=key_weights)
        if w <= 0.0:
            continue

        av = a.get(key)
        bv = b.get(key)
        if av is None and bv is None:
            continue
        if av is None or bv is None:
            total += w * 1.0
            continue

        if spec.type in {"int", "float"}:
            ad = _to_float_value(av)
            bd = _to_float_value(bv)
            if ad is None or bd is None:
                total += w * 1.0
                continue
            denom = _distance_denom(spec, spec.resolve_norm(default=normalization))
            total += w * abs(ad - bd) / denom
            continue

        # bool/string/categorical: exact match only
        total += w * (0.0 if av == bv else 1.0)

    return float(total)


def genome_similarity_v1(distance: float) -> float:
    """A bounded similarity score derived from distance (0..1, higher=more similar)."""

    d = float(distance)
    if d <= 0.0:
        return 1.0
    return 1.0 / (1.0 + d)


def genome_distance_yaml_configs(
    a_path: Path | str,
    b_path: Path | str,
    *,
    knob_space: Sequence[KnobSpec],
    normalization: NormMode = "range",
    key_weights: Mapping[str, float] | None = None,
    float_round: int | None = DEFAULT_FLOAT_ROUND,
    include_missing: bool = True,
) -> float:
    a_cfg = load_effective_yaml_config(a_path)
    b_cfg = load_effective_yaml_config(b_path)
    a_genome = genome_from_config(a_cfg, knob_space=knob_space, float_round=float_round, include_missing=include_missing)
    b_genome = genome_from_config(b_cfg, knob_space=knob_space, float_round=float_round, include_missing=include_missing)
    return genome_distance_weighted_l1_v1(
        a_genome,
        b_genome,
        knob_space=knob_space,
        normalization=normalization,
        key_weights=key_weights,
    )


def get_dotted(cfg: Mapping[str, Any], dotted_key: str) -> Any:
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise KeyError(dotted_key)
        node = node[part]
    return node


def canonicalize_object(obj: Any, *, float_round: int | None = DEFAULT_FLOAT_ROUND) -> Any:
    """Convert arbitrary object tree to a JSON-stable payload."""

    if obj is None or isinstance(obj, (str, bool, int)):
        return obj

    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"Non-finite float in payload: {obj!r}")
        val = 0.0 if obj == 0.0 else float(obj)
        if float_round is not None:
            val = round(val, int(float_round))
            if val == 0.0:
                val = 0.0
        return val

    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()

    if isinstance(obj, Path):
        return obj.as_posix()

    if isinstance(obj, Mapping):
        items = []
        for k, v in obj.items():
            key = str(k)
            items.append((key, canonicalize_object(v, float_round=float_round)))
        # stable order
        return {k: v for k, v in sorted(items, key=lambda kv: kv[0])}

    if isinstance(obj, (list, tuple)):
        return [canonicalize_object(v, float_round=float_round) for v in obj]

    if isinstance(obj, set):
        normalized = [canonicalize_object(v, float_round=float_round) for v in obj]
        # Stable order via canonical JSON encoding.
        normalized.sort(key=lambda v: json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
        return normalized

    # Fallback for unexpected scalars (e.g. Decimal, numpy types).
    return str(obj)


def coerce_genome_value(
    value: Any,
    *,
    value_type: ValueType,
    float_round: int | None = DEFAULT_FLOAT_ROUND,
    round_to: int | None = None,
) -> GenomeValue:
    if value is None:
        return None

    if value_type == "bool":
        coerced = _to_optional_bool(value)
        return None if coerced is None else bool(coerced)

    if value_type == "int":
        coerced = _to_optional_int(value)
        return None if coerced is None else int(coerced)

    if value_type == "float":
        coerced = _to_optional_float(value)
        if coerced is None:
            return None
        decimals = round_to if round_to is not None else float_round
        out = float(coerced)
        if decimals is not None:
            out = round(out, int(decimals))
        if out == 0.0:
            out = 0.0
        return out

    if value_type == "string":
        token = str(value).strip()
        return token or None

    # categorical: keep JSON scalar types, otherwise stringify
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        out = 0.0 if value == 0.0 else float(value)
        if float_round is not None:
            out = round(out, int(float_round))
            if out == 0.0:
                out = 0.0
        return out
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    return str(value)


def _distance_denom(spec: KnobSpec, mode: NormMode) -> float:
    if mode == "none":
        return 1.0
    if mode == "step":
        if spec.step is not None and float(spec.step) > 0.0:
            return float(spec.step)
        return 1.0
    # mode == "range"
    if spec.min is None or spec.max is None:
        return 1.0
    span = float(spec.max) - float(spec.min)
    if not math.isfinite(span) or span <= 0.0:
        return 1.0
    return span


def _to_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _to_float_value(value: GenomeValue) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _to_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    token = str(value).strip().lower()
    if token in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "f", "no", "n", "off"}:
        return False
    return None


def _to_optional_choices(value: Any) -> list[GenomeValue] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    out: list[GenomeValue] = []
    for item in value:
        if item is None or isinstance(item, (str, bool, int)):
            out.append(item)
        elif isinstance(item, float):
            out.append(item if math.isfinite(item) else None)
        else:
            out.append(str(item))
    return out


def _read_raw_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML config (expected mapping): {path}")
    return payload


def _deep_merge_config(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged: MutableMapping[str, Any] = copy.deepcopy(base)
        for key, value in override.items():
            # In generated batch configs `null` is used as "no override".
            if key in merged and value is None:
                continue
            if key in merged:
                merged[key] = _deep_merge_config(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return dict(merged)
    if override is None and base is not None:
        return copy.deepcopy(base)
    return copy.deepcopy(override)


def _resolve_base_config_path(base_config: str, current_path: Path) -> Path:
    candidate = Path(base_config)
    if candidate.is_absolute():
        return candidate.resolve()

    local_candidate = (current_path.parent / candidate).resolve()
    if local_candidate.exists():
        return local_candidate

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return local_candidate


def _load_raw_config(path: Path, visited: set[Path] | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    visited_paths = set(visited or ())
    if resolved in visited_paths:
        chain = " -> ".join(str(p) for p in sorted(visited_paths | {resolved}))
        raise ValueError(f"Circular base_config chain detected: {chain}")
    visited_paths.add(resolved)

    raw_cfg = _read_raw_yaml(resolved)
    base_ref = str(raw_cfg.get("base_config") or "").strip()
    if not base_ref:
        raw_cfg.pop("base_config", None)
        return raw_cfg

    base_path = _resolve_base_config_path(base_ref, resolved)
    base_cfg = _load_raw_config(base_path, visited=visited_paths)
    override_cfg = copy.deepcopy(raw_cfg)
    override_cfg.pop("base_config", None)
    return _deep_merge_config(base_cfg, override_cfg)
