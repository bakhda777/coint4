#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

DEFAULT_SCHEMA_PATH = (
    Path(__file__).resolve().parent
    / "schemas"
    / "pair_crypto_hypothesis_factor.dsl.v1.schema.json"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate pair-crypto hypothesis/factor DSL payload")
    parser.add_argument("--input-path", required=True, help="Path to JSON payload")
    parser.add_argument(
        "--schema-path",
        default=str(DEFAULT_SCHEMA_PATH),
        help="Path to DSL schema JSON",
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

    print(
        json.dumps(
            {
                "ok": ok,
                "input_path": str(input_path),
                "schema_path": str(schema_path),
                "error_count": len(errors),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    if not ok:
        for item in errors:
            print(item, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
