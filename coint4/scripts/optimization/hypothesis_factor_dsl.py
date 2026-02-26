#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SCHEMA_PATH = (
    Path(__file__).resolve().parent
    / "schemas"
    / "pair_crypto_hypothesis_factor.dsl.v1.schema.json"
)

SAFE_PATCH_ROOTS: tuple[str, ...] = (
    "backtest",
    "pair_selection",
    "portfolio",
    "filter_params",
    "walk_forward",
    "data_processing",
    "data_filters",
    "time",
    "risk",
    "guards",
)


def _match_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    return True


def _validate_by_schema(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    errors: list[str] = []

    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        if not _match_type(value, schema_type):
            errors.append(f"{path}: expected {schema_type}")
            return errors
    elif isinstance(schema_type, list):
        if not any(_match_type(value, entry) for entry in schema_type):
            errors.append(f"{path}: expected one of {schema_type}")
            return errors

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        errors.append(f"{path}: value not in enum")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            errors.append(f"{path}: minLength={schema['minLength']}")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            errors.append(f"{path}: maxLength={schema['maxLength']}")
        if "pattern" in schema:
            pattern = str(schema["pattern"])
            if re.fullmatch(pattern, value) is None:
                errors.append(f"{path}: pattern mismatch")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < float(schema["minimum"]):
            errors.append(f"{path}: minimum={schema['minimum']}")
        if "maximum" in schema and value > float(schema["maximum"]):
            errors.append(f"{path}: maximum={schema['maximum']}")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            errors.append(f"{path}: minItems={schema['minItems']}")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            errors.append(f"{path}: maxItems={schema['maxItems']}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(value):
                errors.extend(_validate_by_schema(item, item_schema, path=f"{path}[{idx}]"))

    if isinstance(value, dict):
        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                if key not in value:
                    errors.append(f"{path}: missing required field '{key}'")

        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, child_schema in properties.items():
                if key in value and isinstance(child_schema, dict):
                    errors.extend(_validate_by_schema(value[key], child_schema, path=f"{path}.{key}"))

            if schema.get("additionalProperties") is False:
                for key in value:
                    if key not in properties:
                        errors.append(f"{path}: additionalProperties not allowed ('{key}')")

    return errors


def load_schema(schema_path: Path) -> dict[str, Any]:
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("schema must be a JSON object")
    return payload


def validate_hypothesis_factor_payload(payload: dict[str, Any], schema: dict[str, Any]) -> tuple[bool, list[str]]:
    if not isinstance(payload, dict):
        return False, ["$ must be object"]
    errors = _validate_by_schema(payload, schema)
    return (len(errors) == 0), errors


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_finite_float(value: Any, *, label: str) -> float:
    if not _is_number(value):
        raise ValueError(f"{label} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{label} must be a finite number")
    return out


def _get_nested(cfg: dict[str, Any], dotted_key: str) -> Any:
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted_key)
        node = node[part]
    return node


def _set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        child = node.get(part)
        if child is None:
            child = {}
            node[part] = child
        if not isinstance(child, dict):
            raise ValueError(
                f"Cannot materialize nested key '{dotted_key}': '{part}' is not an object in patch"
            )
        node = child
    node[parts[-1]] = value


def _select_hypothesis(payload: dict[str, Any], hypothesis_id: str | None) -> dict[str, Any]:
    hypotheses = payload.get("hypotheses")
    if not isinstance(hypotheses, list) or not hypotheses:
        raise ValueError("payload.hypotheses must be a non-empty array")
    if hypothesis_id is None:
        first = hypotheses[0]
        if not isinstance(first, dict):
            raise ValueError("payload.hypotheses[0] must be an object")
        return first
    for item in hypotheses:
        if isinstance(item, dict) and str(item.get("hypothesis_id") or "").strip() == hypothesis_id:
            return item
    raise ValueError(f"Unknown hypothesis_id: {hypothesis_id}")


def _validate_target_key_for_safe_patch(target_key: str) -> None:
    root = target_key.split(".", 1)[0]
    if root not in SAFE_PATCH_ROOTS:
        raise ValueError(
            f"Unsafe target_key root '{root}' for '{target_key}'. "
            f"Allowed roots: {', '.join(SAFE_PATCH_ROOTS)}"
        )


def _coerce_set_value_with_base_type(value: Any, current: Any, *, target_key: str) -> Any:
    if current is None:
        return value
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise ValueError(f"target_key '{target_key}' expects boolean value")
        return value
    if _is_number(current):
        if not _is_number(value):
            raise ValueError(f"target_key '{target_key}' expects numeric value")
        return value
    if isinstance(current, str):
        if not isinstance(value, str):
            raise ValueError(f"target_key '{target_key}' expects string value")
        return value
    raise ValueError(f"target_key '{target_key}' points to non-scalar value; patching is not allowed")


def _enforce_bounds(*, target_key: str, candidate: Any, factor: dict[str, Any]) -> None:
    bounds = factor.get("bounds")
    if bounds is None:
        return
    if not isinstance(bounds, dict):
        raise ValueError(f"factor '{target_key}' has invalid bounds (expected object)")
    lower = _as_finite_float(bounds.get("lower"), label=f"{target_key}.bounds.lower")
    upper = _as_finite_float(bounds.get("upper"), label=f"{target_key}.bounds.upper")
    if lower > upper:
        raise ValueError(f"factor '{target_key}' has invalid bounds: lower > upper")
    if not _is_number(candidate):
        raise ValueError(f"factor '{target_key}' has bounds but resolved value is not numeric")
    numeric_candidate = float(candidate)
    if numeric_candidate < lower or numeric_candidate > upper:
        raise ValueError(
            f"factor '{target_key}' resolved value {numeric_candidate} is outside bounds [{lower}, {upper}]"
        )


def _materialize_factor_value(factor: dict[str, Any], base_config: dict[str, Any] | None) -> Any:
    target_key = str(factor.get("target_key") or "").strip()
    op = str(factor.get("op") or "").strip()
    raw_value = factor.get("value")

    has_base_value = False
    base_value: Any = None
    if base_config is not None:
        try:
            base_value = _get_nested(base_config, target_key)
            has_base_value = True
        except KeyError as exc:
            raise ValueError(f"target_key '{target_key}' is missing in base config") from exc

    if op == "set":
        candidate = raw_value
        if has_base_value:
            candidate = _coerce_set_value_with_base_type(raw_value, base_value, target_key=target_key)
    elif op == "enable":
        if has_base_value and not isinstance(base_value, (bool, type(None))):
            raise ValueError(f"target_key '{target_key}' is not boolean in base config")
        candidate = True
    elif op == "disable":
        if has_base_value and not isinstance(base_value, (bool, type(None))):
            raise ValueError(f"target_key '{target_key}' is not boolean in base config")
        candidate = False
    elif op in {"scale", "offset"}:
        if not has_base_value:
            raise ValueError(f"op '{op}' for target_key '{target_key}' requires --base-config")
        base_numeric = _as_finite_float(base_value, label=f"{target_key} base value")
        operand = _as_finite_float(raw_value, label=f"{target_key} factor value")
        candidate = base_numeric * operand if op == "scale" else base_numeric + operand
    else:
        raise ValueError(f"Unsupported operation '{op}' for target_key '{target_key}'")

    _enforce_bounds(target_key=target_key, candidate=candidate, factor=factor)
    return candidate


def materialize_hypothesis_factor_patch(
    payload: dict[str, Any],
    *,
    hypothesis_id: str | None = None,
    base_config: dict[str, Any] | None = None,
    base_config_ref: str | None = None,
) -> dict[str, Any]:
    hypothesis = _select_hypothesis(payload, hypothesis_id=hypothesis_id)
    factors = hypothesis.get("factors")
    if not isinstance(factors, list) or not factors:
        raise ValueError("hypothesis.factors must be a non-empty array")

    patch: dict[str, Any] = {}
    if base_config_ref:
        patch["base_config"] = base_config_ref

    seen_target_keys: set[str] = set()
    for idx, factor in enumerate(factors):
        if not isinstance(factor, dict):
            raise ValueError(f"hypothesis.factors[{idx}] must be an object")
        target_key = str(factor.get("target_key") or "").strip()
        if not target_key:
            raise ValueError(f"hypothesis.factors[{idx}] has empty target_key")
        _validate_target_key_for_safe_patch(target_key)
        if target_key in seen_target_keys:
            raise ValueError(f"Duplicate factor target_key is not allowed: {target_key}")
        seen_target_keys.add(target_key)

        value = _materialize_factor_value(factor, base_config=base_config)
        _set_nested(patch, target_key, value)

    return patch


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"YAML file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML must be an object: {path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate pair-crypto hypothesis/factor DSL payload")
    parser.add_argument("--input-path", required=True, help="Path to JSON payload")
    parser.add_argument(
        "--schema-path",
        default=str(DEFAULT_SCHEMA_PATH),
        help="Path to DSL schema JSON",
    )
    parser.add_argument(
        "--materialize-output-path",
        help="If set, materialize selected hypothesis into executable YAML patch at this path",
    )
    parser.add_argument(
        "--hypothesis-id",
        help="Hypothesis ID to materialize (default: first hypothesis)",
    )
    parser.add_argument(
        "--base-config-path",
        help="Optional base YAML config used for safety checks and scale/offset operations",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path)
    schema_path = Path(args.schema_path)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        print(json.dumps({"ok": False, "error": "input must be JSON object"}, ensure_ascii=False, sort_keys=True))
        return 1

    schema = load_schema(schema_path)
    ok, errors = validate_hypothesis_factor_payload(payload, schema)
    result: dict[str, Any] = {
        "ok": ok,
        "input_path": str(input_path),
        "schema_path": str(schema_path),
        "error_count": len(errors),
    }

    if not ok:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        for item in errors:
            print(item, file=sys.stderr)
        return 1

    if args.materialize_output_path:
        try:
            base_config_path = str(args.base_config_path or "").strip()
            base_config = load_yaml_mapping(Path(base_config_path)) if base_config_path else None
            patch = materialize_hypothesis_factor_patch(
                payload,
                hypothesis_id=args.hypothesis_id,
                base_config=base_config,
                base_config_ref=base_config_path or None,
            )
            output_path = Path(args.materialize_output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                yaml.safe_dump(patch, sort_keys=False, allow_unicode=False),
                encoding="utf-8",
            )
            selected = _select_hypothesis(payload, hypothesis_id=args.hypothesis_id)
            result["materialized"] = True
            result["materialize_output_path"] = str(output_path)
            result["materialized_hypothesis_id"] = str(selected.get("hypothesis_id") or "")
            result["materialized_factor_count"] = len(selected.get("factors") or [])
        except ValueError as exc:
            result["ok"] = False
            result["materialized"] = False
            result["materialize_error"] = str(exc)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            print(str(exc), file=sys.stderr)
            return 1

    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
