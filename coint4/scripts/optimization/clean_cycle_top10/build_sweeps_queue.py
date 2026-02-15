#!/usr/bin/env python3
"""Build a sweeps run queue CSV for clean-cycle TOP-10 (C15).

Creates a CSV with columns: config_path, results_dir, status.

Inputs:
- --sweeps-manifest: JSON list produced by make_sweep_configs.py (preferred), or
- --configs-dir: directory with sweep YAML configs.

Safety/guardrails:
- results_dir is always placed under OPT_DIR and must be unique.
- each config must match FIXED_WINDOWS.walk_forward and have explicit max_steps <= 5.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from definitions import CLEAN_AGG_DIR, FIXED_WINDOWS
from definitions import OPT_DIR as DEFAULT_OPT_DIR


WF_KEYS_ORDER = [
    "start_date",
    "end_date",
    "training_period_days",
    "testing_period_days",
    "step_size_days",
    "max_steps",
    "gap_minutes",
    "refit_frequency",
]


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
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    normalized = _normalize_repo_relative_path(raw)
    return project_root / normalized


def _try_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if int(value) != value:
            return None
        return int(value)
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _norm_date_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    s = str(value).strip()
    return s or None


def _normalize_walk_forward_subset(wf: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in keys:
        if key in {"start_date", "end_date"}:
            out[key] = _norm_date_str(wf.get(key))
        elif key in {"training_period_days", "testing_period_days", "step_size_days", "max_steps", "gap_minutes"}:
            out[key] = _to_int(wf.get(key))
        elif key in {"refit_frequency"}:
            value = wf.get(key)
            out[key] = None if value is None else str(value).strip()
        else:
            out[key] = wf.get(key)
    return out


def _load_expected_walk_forward() -> Dict[str, Any]:
    if not isinstance(FIXED_WINDOWS, dict):
        raise SystemExit("FIXED_WINDOWS must be a JSON object/dict")
    wf_node = FIXED_WINDOWS.get("walk_forward", FIXED_WINDOWS)
    if not isinstance(wf_node, dict):
        raise SystemExit("FIXED_WINDOWS.walk_forward must be a JSON object/dict")
    return _normalize_walk_forward_subset(wf_node, WF_KEYS_ORDER)


def _load_config_walk_forward(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"config must parse into a mapping/object: {path}")
    wf = payload.get("walk_forward")
    if not isinstance(wf, dict):
        raise SystemExit(f"config missing walk_forward mapping: {path}")
    return _normalize_walk_forward_subset(wf, WF_KEYS_ORDER)


def _validate_config(path: Path, *, expected_wf: Dict[str, Any]) -> None:
    wf = _load_config_walk_forward(path)
    mismatches: List[str] = []
    for key in WF_KEYS_ORDER:
        if wf.get(key) != expected_wf.get(key):
            mismatches.append(f"{key}={wf.get(key)!r} expected={expected_wf.get(key)!r}")

    max_steps = wf.get("max_steps")
    if max_steps is None:
        mismatches.append("max_steps is missing/null (must be an explicit int <= 5)")
    elif int(max_steps) > 5:
        mismatches.append(f"unsafe max_steps={max_steps} (must be <= 5)")

    if mismatches:
        msg = "walk_forward mismatches for config: {cfg}\n- ".format(cfg=path) + "\n- ".join(mismatches)
        raise SystemExit(msg)


def _load_manifest_configs(path: Path) -> List[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(f"sweeps manifest must be a JSON list: {path}")

    out: List[str] = []
    for i, item in enumerate(payload, 1):
        if not isinstance(item, dict):
            raise SystemExit(f"sweeps manifest entry #{i} must be an object: {path}")
        cfg = str(item.get("config_path") or "").strip()
        if not cfg:
            raise SystemExit(f"sweeps manifest entry #{i} missing config_path: {path}")
        out.append(cfg)
    return out


def _discover_configs(dir_path: Path) -> List[Path]:
    if not dir_path.exists() or not dir_path.is_dir():
        raise SystemExit(f"--configs-dir must be an existing directory: {dir_path}")
    configs = sorted([p for p in dir_path.rglob("*.yaml") if p.is_file()])
    configs += sorted([p for p in dir_path.rglob("*.yml") if p.is_file()])
    # Filter out obvious non-config placeholders.
    configs = [p for p in configs if not p.name.startswith(".")]
    # Deduplicate while keeping deterministic order.
    seen: set[Path] = set()
    out: List[Path] = []
    for p in configs:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _normalize_opt_dir(raw: str, *, project_root: Path) -> str:
    value = str(raw or "").strip()
    if not value:
        raise SystemExit("--opt-dir is empty")
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return _normalize_repo_relative_path(value) or value


def _queue_row_for_config(config_path: str, *, project_root: Path, opt_dir: str) -> Tuple[str, str]:
    cfg_path = _resolve_under_project(config_path, project_root)
    if cfg_path is None:
        raise SystemExit("invalid empty config_path")
    if not cfg_path.exists():
        raise SystemExit(f"config not found: {cfg_path}")

    # Prefer writing queue paths relative to app-root when possible.
    cfg_rel = _try_relpath(cfg_path, project_root)
    run_id = cfg_path.stem
    results_dir = str(Path(opt_dir) / run_id)
    return cfg_rel, results_dir


def _write_queue(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["config_path", "results_dir", "status"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build sweep run_queue CSV (clean_cycle_top10)")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--sweeps-manifest",
        default=None,
        help="Path to sweeps_manifest.json (relative to app-root `coint4/` unless absolute).",
    )
    input_group.add_argument(
        "--configs-dir",
        default=None,
        help="Directory containing sweep YAML configs (relative to app-root `coint4/` unless absolute).",
    )

    parser.add_argument(
        "--opt-dir",
        default=DEFAULT_OPT_DIR,
        help="OPT_DIR for sweeps results (relative to app-root `coint4/` unless absolute).",
    )
    parser.add_argument(
        "--output",
        default=f"{CLEAN_AGG_DIR}/sweeps_run_queue.csv",
        help="Output CSV path (relative to app-root `coint4/` unless absolute).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan without writing CSV.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output CSV.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    project_root = _resolve_project_root()
    expected_wf = _load_expected_walk_forward()

    opt_dir = _normalize_opt_dir(args.opt_dir, project_root=project_root)
    output_path = _resolve_under_project(args.output, project_root) or (project_root / args.output)

    config_paths: List[str] = []
    if args.sweeps_manifest:
        manifest_path = _resolve_under_project(args.sweeps_manifest, project_root)
        if manifest_path is None:
            raise SystemExit("--sweeps-manifest is empty")
        if not manifest_path.exists():
            raise SystemExit(f"sweeps manifest not found: {manifest_path}")
        config_paths = _load_manifest_configs(manifest_path)
    else:
        configs_dir = _resolve_under_project(args.configs_dir, project_root)
        if configs_dir is None:
            raise SystemExit("--configs-dir is empty")
        configs = _discover_configs(configs_dir)
        if not configs:
            raise SystemExit(f"no YAML configs found under --configs-dir: {configs_dir}")
        config_paths = [str(p) for p in configs]

    rows: List[Dict[str, str]] = []
    results_dirs: set[str] = set()
    for cfg in config_paths:
        cfg_rel, results_dir = _queue_row_for_config(cfg, project_root=project_root, opt_dir=opt_dir)
        # Validate the config file itself.
        cfg_path = _resolve_under_project(cfg, project_root) or Path(cfg)
        _validate_config(cfg_path, expected_wf=expected_wf)

        if results_dir in results_dirs:
            raise SystemExit(f"results_dir collision (must be unique): {results_dir}")
        results_dirs.add(results_dir)

        rows.append({"config_path": cfg_rel, "results_dir": results_dir, "status": "planned"})

    if args.dry_run:
        print(f"DRY: opt_dir={opt_dir}")
        print(f"DRY: output={output_path}")
        for row in rows:
            print(f"DRY: {row['status']}: {row['config_path']} -> {row['results_dir']}")
        return 0

    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing output (use --overwrite): {output_path}")

    _write_queue(output_path, rows)
    print(f"Wrote sweeps queue: {output_path} ({len(rows)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

