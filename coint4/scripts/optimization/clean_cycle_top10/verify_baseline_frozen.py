#!/usr/bin/env python3
"""Verify baseline freeze sentinel integrity (manifest + configs + FIXED_WINDOWS).

Checks:
- BASELINE_FROZEN.txt exists in BASELINE_DIR.
- baseline_manifest_sha256 matches current manifest file.
- All config files listed in sentinel exist and match their recorded sha256.
- All configs have walk_forward matching FIXED_WINDOWS.walk_forward recorded in sentinel,
  and that FIXED_WINDOWS.walk_forward matches current definitions.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from baseline_guard import SENTINEL_FILENAME, read_sentinel

from definitions import BASELINE_DIR as DEFAULT_BASELINE_DIR
from definitions import CLEAN_AGG_DIR as DEFAULT_CLEAN_AGG_DIR
from definitions import FIXED_WINDOWS


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[3]


def _normalize_repo_relative_path(raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    if value.startswith("coint4/"):
        value = value[len("coint4/") :]
    while value.startswith("./"):
        value = value[2:]
    return value


def _resolve_under_project(path_str: str, project_root: Path) -> Optional[Path]:
    raw = str(path_str or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    normalized = _normalize_repo_relative_path(raw)
    return project_root / normalized


def _ensure_under_project(path: Path, project_root: Path) -> Path:
    resolved = path.resolve()
    root = project_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"refusing to operate outside project root: {resolved}") from exc
    return resolved


def _try_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _require_dict(value: Any, *, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be an object/dict")
    return value


def _require_str(value: Any, *, label: str) -> str:
    s = str(value or "").strip()
    if not s:
        raise SystemExit(f"{label} must be a non-empty string")
    return s


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(f"manifest must be a JSON array: {path}")
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(payload, 1):
        if not isinstance(item, dict):
            raise SystemExit(f"manifest entry #{idx} must be an object: {path}")
        out.append(item)
    return out


def _extract_walk_forward(config_path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"config must be a YAML mapping: {config_path}")
    wf = payload.get("walk_forward")
    if not isinstance(wf, dict):
        raise SystemExit(f"config missing walk_forward mapping: {config_path}")
    return dict(wf)


def _compare_walk_forward(*, actual: Dict[str, Any], expected: Dict[str, Any], config_rel: str) -> None:
    missing = sorted(set(expected.keys()) - set(actual.keys()))
    extra = sorted(set(actual.keys()) - set(expected.keys()))
    mismatched: List[str] = []
    for key, exp in expected.items():
        if key not in actual:
            continue
        if actual.get(key) != exp:
            mismatched.append(f"{key}: expected={exp!r} actual={actual.get(key)!r}")

    if missing or mismatched:
        parts: List[str] = [f"walk_forward mismatch in {config_rel}"]
        if missing:
            parts.append("missing keys: " + ", ".join(missing))
        if mismatched:
            parts.append("value diffs: " + "; ".join(mismatched))
        raise SystemExit(" | ".join(parts))

    # Extra keys are allowed, but we still want deterministic fixed fields.
    if extra:
        # Only warn via stdout (do not fail), since configs may grow new fields.
        print(f"WARN: {config_rel} has extra walk_forward keys not in FIXED_WINDOWS: {', '.join(extra)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify baseline freeze sentinel integrity")
    parser.add_argument(
        "--baseline-dir",
        default=DEFAULT_BASELINE_DIR,
        help="Baseline results directory (relative to coint4/ unless absolute).",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional: override baseline_manifest.json path (relative to coint4/ unless absolute).",
    )
    args = parser.parse_args()

    project_root = _resolve_project_root()
    baseline_dir = _resolve_under_project(args.baseline_dir, project_root) or (project_root / args.baseline_dir)
    baseline_dir = _ensure_under_project(baseline_dir, project_root)

    sentinel = read_sentinel(baseline_dir)
    # Required fields.
    datestamp = _require_str(sentinel.get("DATESTAMP"), label="DATESTAMP")
    created_at = _require_str(sentinel.get("created_at_utc"), label="created_at_utc")
    manifest_sha_expected = _require_str(sentinel.get("baseline_manifest_sha256"), label="baseline_manifest_sha256")
    fixed_windows = _require_dict(sentinel.get("FIXED_WINDOWS"), label="FIXED_WINDOWS")
    fixed_wf_expected = _require_dict(fixed_windows.get("walk_forward"), label="FIXED_WINDOWS.walk_forward")

    manifest_path_str = args.manifest or sentinel.get("baseline_manifest_path") or f"{DEFAULT_CLEAN_AGG_DIR}/baseline_manifest.json"
    manifest_path = _resolve_under_project(str(manifest_path_str), project_root) or (project_root / str(manifest_path_str))
    manifest_path = _ensure_under_project(manifest_path, project_root)
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")

    manifest_sha_actual = _sha256_file(manifest_path)
    if manifest_sha_actual != manifest_sha_expected:
        raise SystemExit(
            "baseline_manifest_sha256 mismatch: expected={exp} actual={act} path={path}".format(
                exp=manifest_sha_expected,
                act=manifest_sha_actual,
                path=_try_relpath(manifest_path, project_root),
            )
        )

    # FIXED_WINDOWS in sentinel must match current definitions.py (safety net).
    current_wf = dict(FIXED_WINDOWS.get("walk_forward") or {})
    if fixed_wf_expected != current_wf:
        raise SystemExit(
            "FIXED_WINDOWS.walk_forward mismatch between sentinel and definitions.py: "
            f"sentinel={fixed_wf_expected!r} definitions={current_wf!r}"
        )

    configs = sentinel.get("baseline_configs")
    if not isinstance(configs, list) or not configs:
        raise SystemExit("baseline_configs must be a non-empty list in sentinel")

    for idx, cfg in enumerate(configs, 1):
        if not isinstance(cfg, dict):
            raise SystemExit(f"baseline_configs[{idx}] must be an object")
        cfg_path = _normalize_repo_relative_path(str(cfg.get("config_path") or ""))
        cfg_sha_expected = _require_str(cfg.get("sha256"), label=f"baseline_configs[{idx}].sha256")
        if not cfg_path:
            raise SystemExit(f"baseline_configs[{idx}].config_path is empty")

        resolved = _resolve_under_project(cfg_path, project_root)
        if resolved is None or not resolved.exists():
            raise SystemExit(f"config not found: {cfg_path}")
        if not resolved.is_file():
            raise SystemExit(f"config is not a file: {cfg_path}")

        cfg_sha_actual = _sha256_file(resolved)
        if cfg_sha_actual != cfg_sha_expected:
            raise SystemExit(
                "config sha256 mismatch: path={path} expected={exp} actual={act}".format(
                    path=cfg_path,
                    exp=cfg_sha_expected,
                    act=cfg_sha_actual,
                )
            )

        wf_actual = _extract_walk_forward(resolved)
        _compare_walk_forward(actual=wf_actual, expected=fixed_wf_expected, config_rel=cfg_path)

    # Optional: validate manifest config_sha256 fields still match files.
    manifest = _load_manifest(manifest_path)
    for idx, entry in enumerate(manifest, 1):
        config_path = _normalize_repo_relative_path(str(entry.get("config_path") or ""))
        if not config_path:
            raise SystemExit(f"manifest entry #{idx} missing config_path (unexpected for frozen baseline)")
        expected_sha = _require_str(entry.get("config_sha256"), label=f"manifest[{idx}].config_sha256")
        resolved = _resolve_under_project(config_path, project_root)
        if resolved is None or not resolved.exists():
            raise SystemExit(f"manifest config not found: {config_path}")
        actual_sha = _sha256_file(resolved)
        if actual_sha != expected_sha:
            raise SystemExit(
                "manifest config_sha256 mismatch: path={path} expected={exp} actual={act}".format(
                    path=config_path,
                    exp=expected_sha,
                    act=actual_sha,
                )
            )

    print("OK: baseline is frozen and consistent")
    print(f"- DATESTAMP: {datestamp}")
    print(f"- created_at_utc: {created_at}")
    print(f"- sentinel: {_try_relpath(baseline_dir / SENTINEL_FILENAME, project_root)}")
    print(f"- manifest: {_try_relpath(manifest_path, project_root)} (sha256={manifest_sha_actual})")
    print(f"- configs: {len(configs)} verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

