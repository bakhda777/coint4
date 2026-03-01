"""Validation helpers for ConfigPatch IR payloads.

ConfigPatch IR is a pair-crypto adaptation of QuantaAlpha symbolic factor IR.
This module provides deterministic, fail-closed checks for IR artifacts.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

_ALLOWED_OPS = {"set", "scale", "offset", "enable", "disable"}
_REQUIRED_TOP_KEYS = (
    "ir_version",
    "source",
    "parents",
    "hypothesis",
    "factors",
    "materialized_patch",
    "gates",
    "semantic_gate",
)


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _flatten_patch_leaves(patch: Mapping[str, Any], *, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in patch.items():
        token = str(key or "").strip()
        if not token or token == "base_config":
            continue
        dotted = f"{prefix}.{token}" if prefix else token
        if isinstance(value, Mapping):
            out.update(_flatten_patch_leaves(value, prefix=dotted))
            continue
        out[dotted] = value
    return out


def validate_patch_ir_payload(payload: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    if not isinstance(payload, Mapping):
        return ["patch_ir must be an object"]

    for key in _REQUIRED_TOP_KEYS:
        if key not in payload:
            issues.append(f"missing required key: {key}")

    ir_version = str(payload.get("ir_version") or "").strip()
    if ir_version != "config_patch_ast.v1":
        issues.append("ir_version must be 'config_patch_ast.v1'")

    if not _is_nonempty_string(payload.get("source")):
        issues.append("source must be non-empty string")

    parents_raw = payload.get("parents")
    if not isinstance(parents_raw, list) or not parents_raw:
        issues.append("parents must be non-empty list")
    else:
        for idx, parent in enumerate(parents_raw):
            if not _is_nonempty_string(parent):
                issues.append(f"parents[{idx}] must be non-empty string")

    hypothesis = payload.get("hypothesis")
    if not isinstance(hypothesis, Mapping):
        issues.append("hypothesis must be object")
    else:
        thesis = str(hypothesis.get("thesis") or "").strip()
        if len(thesis) < 16:
            issues.append("hypothesis.thesis is too short")

    factor_keys: list[str] = []
    factors = payload.get("factors")
    if not isinstance(factors, list) or not factors:
        issues.append("factors must be non-empty list")
    else:
        for idx, factor in enumerate(factors):
            if not isinstance(factor, Mapping):
                issues.append(f"factors[{idx}] must be object")
                continue
            target_key = str(factor.get("target_key") or "").strip()
            op = str(factor.get("op") or "").strip()
            rationale = str(factor.get("rationale") or "").strip()
            if not target_key or "." not in target_key:
                issues.append(f"factors[{idx}].target_key is invalid")
            else:
                factor_keys.append(target_key)
            if op not in _ALLOWED_OPS:
                issues.append(f"factors[{idx}].op is invalid: {op!r}")
            if len(rationale) < 8:
                issues.append(f"factors[{idx}].rationale is too short")
            if op in {"scale", "offset"} and not _is_finite_number(factor.get("value")):
                issues.append(f"factors[{idx}].value must be finite number for op={op}")

    unique_factor_keys = {key for key in factor_keys if key}
    if len(unique_factor_keys) != len(factor_keys):
        issues.append("factors contain duplicate target_key")

    materialized_patch = payload.get("materialized_patch")
    leaves: dict[str, Any] = {}
    if not isinstance(materialized_patch, Mapping):
        issues.append("materialized_patch must be object")
    else:
        leaves = _flatten_patch_leaves(materialized_patch)
        if not leaves:
            issues.append("materialized_patch has no editable leaves")
        extra_keys = sorted(set(leaves) - unique_factor_keys)
        if extra_keys:
            preview = ", ".join(extra_keys[:5])
            suffix = " ..." if len(extra_keys) > 5 else ""
            issues.append(f"materialized_patch touches undeclared keys: {preview}{suffix}")

    if isinstance(factors, list) and leaves:
        for idx, factor in enumerate(factors):
            if not isinstance(factor, Mapping):
                continue
            target_key = str(factor.get("target_key") or "").strip()
            op = str(factor.get("op") or "").strip()
            factor_value = factor.get("value")
            if not target_key:
                continue
            if target_key not in leaves:
                issues.append(f"materialized_patch missing target_key from factors[{idx}]: {target_key}")
                continue
            patch_value = leaves[target_key]
            if op == "set" and patch_value != factor_value:
                issues.append(f"materialized_patch mismatch for factors[{idx}] op=set at {target_key}")
            elif op == "enable" and patch_value is not True:
                issues.append(f"materialized_patch mismatch for factors[{idx}] op=enable at {target_key}")
            elif op == "disable" and patch_value is not False:
                issues.append(f"materialized_patch mismatch for factors[{idx}] op=disable at {target_key}")
            elif op in {"scale", "offset"} and not _is_finite_number(patch_value):
                issues.append(f"materialized_patch value must be finite number for factors[{idx}] op={op} at {target_key}")

    gates = payload.get("gates")
    if not isinstance(gates, Mapping):
        issues.append("gates must be object")
    else:
        complexity = gates.get("complexity")
        redundancy = gates.get("redundancy")
        limits = gates.get("limits")
        if not isinstance(complexity, Mapping):
            issues.append("gates.complexity must be object")
        else:
            for name in ("score", "symbolic_length", "parameter_count", "feature_count"):
                if name not in complexity:
                    issues.append(f"gates.complexity.{name} is required")
            if "score" in complexity and not _is_finite_number(complexity.get("score")):
                issues.append("gates.complexity.score must be finite number")
        if not isinstance(redundancy, Mapping):
            issues.append("gates.redundancy must be object")
        else:
            nearest_similarity = redundancy.get("nearest_similarity")
            if nearest_similarity is not None and not _is_finite_number(nearest_similarity):
                issues.append("gates.redundancy.nearest_similarity must be finite number or null")
            if _is_finite_number(nearest_similarity):
                value = float(nearest_similarity)
                if value < 0.0 or value > 1.0:
                    issues.append("gates.redundancy.nearest_similarity must be in [0,1]")
        if not isinstance(limits, Mapping):
            issues.append("gates.limits must be object")
        else:
            for name in ("max_complexity_score", "max_redundancy_similarity", "alpha_sl", "alpha_pc", "alpha_feat"):
                if name not in limits:
                    issues.append(f"gates.limits.{name} is required")
            max_redundancy_similarity = limits.get("max_redundancy_similarity")
            if not _is_finite_number(max_redundancy_similarity):
                issues.append("gates.limits.max_redundancy_similarity must be finite number")
            else:
                value = float(max_redundancy_similarity)
                if value < 0.0 or value > 1.0:
                    issues.append("gates.limits.max_redundancy_similarity must be in [0,1]")

    semantic_gate = payload.get("semantic_gate")
    if not isinstance(semantic_gate, Mapping):
        issues.append("semantic_gate must be object")
    else:
        if not isinstance(semantic_gate.get("ok"), bool):
            issues.append("semantic_gate.ok must be boolean")
        if not _is_nonempty_string(semantic_gate.get("source")):
            issues.append("semantic_gate.source must be non-empty string")
        reasons = semantic_gate.get("reasons")
        if not isinstance(reasons, list):
            issues.append("semantic_gate.reasons must be list")
        elif not all(_is_nonempty_string(item) for item in reasons):
            issues.append("semantic_gate.reasons must contain only non-empty strings")
        if semantic_gate.get("ok") is False and isinstance(reasons, list) and len(reasons) == 0:
            issues.append("semantic_gate.reasons must be non-empty when semantic_gate.ok is false")

    return issues


def assert_valid_patch_ir_payload(payload: Mapping[str, Any]) -> None:
    issues = validate_patch_ir_payload(payload)
    if issues:
        raise ValueError("; ".join(issues))

