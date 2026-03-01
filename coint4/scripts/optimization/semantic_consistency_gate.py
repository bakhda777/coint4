#!/usr/bin/env python3
"""Semantic consistency gate for (hypothesis + ConfigPatch IR) candidates.

QuantaAlpha-style check adapted to pair-crypto config patches:
- deterministic structural checks (fail-closed)
- optional LLM verifier (codex exec) for semantic alignment
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class SemanticGateResult:
    ok: bool
    source: str
    reasons: tuple[str, ...]
    model: str | None = None
    effort: str | None = None
    error: str | None = None


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


def deterministic_semantic_check(
    *,
    hypothesis_thesis: str,
    factors: Sequence[Mapping[str, Any]],
    materialized_patch: Mapping[str, Any] | None = None,
) -> list[str]:
    issues: list[str] = []

    thesis = str(hypothesis_thesis or "").strip()
    if len(thesis) < 16:
        issues.append("hypothesis.thesis is too short")

    if not factors:
        issues.append("factors is empty")
        return issues

    seen: set[str] = set()
    normalized_factors: list[dict[str, Any]] = []
    for idx, factor in enumerate(factors):
        target_key = str(factor.get("target_key") or "").strip()
        op = str(factor.get("op") or "").strip()
        if not target_key or "." not in target_key:
            issues.append(f"factors[{idx}].target_key is invalid")
            continue
        if target_key in seen:
            issues.append(f"duplicate target_key: {target_key}")
        seen.add(target_key)
        if op not in {"set", "scale", "offset", "enable", "disable"}:
            issues.append(f"factors[{idx}].op is invalid: {op!r}")

        value = factor.get("value")
        if op in {"enable", "disable"}:
            continue
        if op in {"scale", "offset"}:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                issues.append(f"factors[{idx}].value must be finite number for op={op}")
        if op == "set" and isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(float(value)):
            issues.append(f"factors[{idx}].value must be finite number for op=set")

        rationale = str(factor.get("rationale") or "").strip()
        if len(rationale) < 8:
            issues.append(f"factors[{idx}].rationale is too short")
        normalized_factors.append({"target_key": target_key, "op": op, "value": value})

    if materialized_patch is None:
        return issues
    if not isinstance(materialized_patch, Mapping):
        issues.append("materialized_patch must be an object")
        return issues

    leaves = _flatten_patch_leaves(materialized_patch)
    if not leaves:
        issues.append("materialized_patch has no editable leaves")
        return issues

    factor_keys = {str(item["target_key"]) for item in normalized_factors if str(item["target_key"])}
    extra_patch_keys = sorted(set(leaves.keys()) - factor_keys)
    if extra_patch_keys:
        preview = ", ".join(extra_patch_keys[:5])
        suffix = " ..." if len(extra_patch_keys) > 5 else ""
        issues.append(f"materialized_patch touches undeclared keys: {preview}{suffix}")

    for idx, factor in enumerate(normalized_factors):
        target_key = str(factor["target_key"])
        op = str(factor["op"])
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

    return issues


def llm_semantic_verdict(
    *,
    hypothesis: Mapping[str, Any],
    factors: Sequence[Mapping[str, Any]],
    materialized_patch: Mapping[str, Any],
    model: str,
    effort: str,
    codex_bin: str,
    timeout_sec: int,
    repo_root: Path,
) -> SemanticGateResult:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["ok", "reasons", "summary"],
        "properties": {
            "ok": {"type": "boolean"},
            "summary": {"type": "string", "minLength": 1, "maxLength": 800},
            "reasons": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
        },
    }

    context = {
        "hypothesis": dict(hypothesis),
        "factors": [dict(item) for item in factors],
        "materialized_patch": dict(materialized_patch),
        "criteria": [
            "Hypothesis thesis must match the intent of factors (no unrelated keys).",
            "Factors must be internally consistent and implementable by patch.",
            "Materialized patch must faithfully reflect factor operations on target_key.",
            "Reject if patch touches keys not justified by hypothesis or if semantics drift.",
        ],
        "effort_hint": str(effort),
    }
    prompt = (
        "You are a strict semantic consistency verifier for a pair-crypto config patch candidate.\n"
        "Return strictly one JSON object matching the output schema.\n"
        "Be fail-closed: if anything is ambiguous, set ok=false and list concrete reasons.\n"
        f"Context JSON:\n{json.dumps(context, ensure_ascii=False, sort_keys=True)}"
    )

    with tempfile.TemporaryDirectory(prefix="semantic_gate_") as temp_dir:
        temp = Path(temp_dir)
        schema_path = temp / "schema.json"
        out_path = temp / "out.json"
        schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

        command = [
            str(codex_bin),
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(out_path),
            "--model",
            str(model),
            prompt,
        ]
        try:
            proc = subprocess.run(
                command,
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=max(10, int(timeout_sec)),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            waited = max(10, int(timeout_sec))
            return SemanticGateResult(
                ok=False,
                source="llm_codex_exec",
                reasons=(),
                model=str(model),
                effort=str(effort),
                error=f"codex exec timeout after {waited}s: {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            return SemanticGateResult(
                ok=False,
                source="llm_codex_exec",
                reasons=(),
                model=str(model),
                effort=str(effort),
                error=f"codex exec exception: {type(exc).__name__}: {exc}",
            )
        if proc.returncode != 0 or not out_path.exists():
            return SemanticGateResult(
                ok=False,
                source="llm_codex_exec",
                reasons=(),
                model=str(model),
                effort=str(effort),
                error=f"codex exec failed rc={proc.returncode}: {proc.stderr.strip()}",
            )
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return SemanticGateResult(
                ok=False,
                source="llm_codex_exec",
                reasons=(),
                model=str(model),
                effort=str(effort),
                error=f"invalid json: {exc}",
            )
        ok = bool(payload.get("ok"))
        reasons = payload.get("reasons") or []
        if not isinstance(reasons, list):
            reasons = []
        reasons_out = tuple(str(item) for item in reasons if str(item).strip())
        if not reasons_out:
            summary = str(payload.get("summary") or "").strip()
            if summary:
                reasons_out = (summary,)
        return SemanticGateResult(
            ok=ok,
            source="llm_codex_exec",
            reasons=reasons_out,
            model=str(model),
            effort=str(effort),
        )


def run_semantic_gate(
    *,
    hypothesis: Mapping[str, Any],
    factors: Sequence[Mapping[str, Any]],
    materialized_patch: Mapping[str, Any],
    use_llm: bool,
    model: str,
    effort: str,
    codex_bin: str,
    timeout_sec: int,
    repo_root: Path,
    fail_open_on_llm_error: bool = False,
) -> SemanticGateResult:
    thesis = str(hypothesis.get("thesis") or "").strip()
    issues = deterministic_semantic_check(
        hypothesis_thesis=thesis,
        factors=factors,
        materialized_patch=materialized_patch,
    )
    if issues:
        return SemanticGateResult(ok=False, source="deterministic", reasons=tuple(issues))

    if not use_llm:
        return SemanticGateResult(ok=True, source="deterministic", reasons=())

    llm = llm_semantic_verdict(
        hypothesis=hypothesis,
        factors=factors,
        materialized_patch=materialized_patch,
        model=model,
        effort=effort,
        codex_bin=codex_bin,
        timeout_sec=timeout_sec,
        repo_root=repo_root,
    )
    if llm.error:
        if not bool(fail_open_on_llm_error):
            return SemanticGateResult(
                ok=False,
                source="llm_codex_exec_error",
                reasons=(str(llm.error),),
                model=str(model),
                effort=str(effort),
                error=str(llm.error),
            )
        return SemanticGateResult(
            ok=True,
            source="deterministic_fallback",
            reasons=(),
            model=str(model),
            effort=str(effort),
            error=str(llm.error),
        )
    return llm


def _main() -> int:
    raise SystemExit(
        "This module is intended to be imported and used by evolution scripts "
        "(e.g., evolve_next_batch.py)."
    )


if __name__ == "__main__":
    _main()
