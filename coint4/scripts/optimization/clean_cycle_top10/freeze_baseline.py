#!/usr/bin/env python3
"""Freeze clean-cycle baseline by writing BASELINE_DIR/BASELINE_FROZEN.txt.

Sentinel format: JSON (stored in a .txt file).

Required fields (per PRD BASELINE_FREEZE.required_in_sentinel):
- DATESTAMP
- baseline_manifest_sha256
- FIXED_WINDOWS.walk_forward
- created_at_utc

This script also records per-config sha256 (derived from the manifest) so
verification can detect config edits after baseline is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from baseline_guard import SENTINEL_FILENAME

from definitions import BASELINE_DIR as DEFAULT_BASELINE_DIR
from definitions import CLEAN_AGG_DIR as DEFAULT_CLEAN_AGG_DIR
from definitions import DATESTAMP as DEFAULT_DATESTAMP
from definitions import FIXED_WINDOWS


_CYCLE_RE = re.compile(r"(?P<datestamp>[0-9]{8})_clean_top10")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_project_root() -> Path:
    """Return app-root (directory containing pyproject.toml)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: assume we are under coint4/scripts/**.
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


def _infer_datestamp_from_path(path_rel: str) -> Optional[str]:
    match = _CYCLE_RE.search(path_rel)
    if not match:
        return None
    return match.group("datestamp")


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


def _collect_config_hashes(
    manifest: List[Dict[str, Any]], project_root: Path
) -> Tuple[List[Dict[str, str]], str]:
    configs: List[Dict[str, str]] = []
    for idx, entry in enumerate(manifest, 1):
        config_path = _normalize_repo_relative_path(str(entry.get("config_path") or ""))
        if not config_path:
            raise SystemExit(f"manifest entry #{idx} missing config_path (cannot freeze configs)")
        expected_sha = str(entry.get("config_sha256") or "").strip()
        if not expected_sha:
            raise SystemExit(f"manifest entry #{idx} missing config_sha256 (cannot freeze configs)")

        resolved = _resolve_under_project(config_path, project_root)
        if resolved is None or not resolved.exists():
            raise SystemExit(f"config path not found: {config_path}")
        if not resolved.is_file():
            raise SystemExit(f"config path is not a file: {config_path}")

        actual_sha = _sha256_file(resolved)
        if actual_sha != expected_sha:
            raise SystemExit(
                "config_sha256 mismatch for {path}: manifest={expected} actual={actual}".format(
                    path=config_path,
                    expected=expected_sha,
                    actual=actual_sha,
                )
            )

        configs.append(
            {
                "rank": str(entry.get("rank") or idx),
                "run_group": str(entry.get("run_group") or ""),
                "run_id": str(entry.get("run_id") or ""),
                "config_path": config_path,
                "sha256": actual_sha,
            }
        )

    rollup_payload = json.dumps(configs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    rollup_sha = hashlib.sha256(rollup_payload).hexdigest()
    return configs, rollup_sha


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze baseline by writing BASELINE_FROZEN.txt sentinel")
    parser.add_argument(
        "--manifest",
        default=f"{DEFAULT_CLEAN_AGG_DIR}/baseline_manifest.json",
        help="Path to baseline_manifest.json (relative to coint4/ unless absolute).",
    )
    parser.add_argument(
        "--baseline-dir",
        default=DEFAULT_BASELINE_DIR,
        help="Baseline results directory (relative to coint4/ unless absolute).",
    )
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument(
        "--refuse-overwrite",
        action="store_true",
        help="Refuse overwriting an existing sentinel file (default).",
    )
    overwrite_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing sentinel file.",
    )
    args = parser.parse_args()

    project_root = _resolve_project_root()
    manifest_path = _resolve_under_project(args.manifest, project_root) or (project_root / args.manifest)
    baseline_dir = _resolve_under_project(args.baseline_dir, project_root) or (project_root / args.baseline_dir)

    manifest_path = _ensure_under_project(manifest_path, project_root)
    baseline_dir = _ensure_under_project(baseline_dir, project_root)

    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")

    baseline_rel = _try_relpath(baseline_dir, project_root)
    if not baseline_rel.startswith("artifacts/wfa/runs_clean/"):
        raise SystemExit(
            f"refusing to freeze: baseline_dir must be under artifacts/wfa/runs_clean/: got {baseline_rel}"
        )

    baseline_dir.mkdir(parents=True, exist_ok=True)

    sentinel_path = baseline_dir / SENTINEL_FILENAME
    if sentinel_path.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing sentinel: {sentinel_path} (use --overwrite)")

    manifest_sha = _sha256_file(manifest_path)
    manifest = _load_manifest(manifest_path)
    configs, configs_rollup_sha = _collect_config_hashes(manifest, project_root)

    datestamp = _infer_datestamp_from_path(baseline_rel) or DEFAULT_DATESTAMP
    created_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fixed_windows_walk_forward: Dict[str, Any] = dict(FIXED_WINDOWS.get("walk_forward") or {})

    payload: Dict[str, Any] = {
        "DATESTAMP": str(datestamp),
        "created_at_utc": created_at_utc,
        "baseline_dir": baseline_rel,
        "baseline_manifest_path": _try_relpath(manifest_path, project_root),
        "baseline_manifest_sha256": manifest_sha,
        "baseline_manifest_entries": len(manifest),
        "FIXED_WINDOWS": {"walk_forward": fixed_windows_walk_forward},
        "baseline_configs": configs,
        "baseline_configs_rollup_sha256": configs_rollup_sha,
        "frozen_by": "freeze_baseline.py",
    }

    tmp_path = sentinel_path.with_suffix(sentinel_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(sentinel_path)

    print(f"Wrote baseline sentinel: {_try_relpath(sentinel_path, project_root)}")
    print(f"Manifest sha256: {manifest_sha} ({_try_relpath(manifest_path, project_root)})")
    print(f"Configs rollup sha256: {configs_rollup_sha} (entries={len(configs)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

