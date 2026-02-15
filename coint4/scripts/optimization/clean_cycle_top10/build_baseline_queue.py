#!/usr/bin/env python3
"""Build a baseline run queue CSV for clean-cycle TOP-10.

Creates a CSV with columns: config_path, results_dir, status.

Rules:
- Queue is built strictly from baseline YAML configs (default: configs/clean_cycle_top10/baseline).
- results_dir is always placed under BASELINE_DIR (runs_clean) and must be unique.
- Each baseline config must match FIXED_WINDOWS.walk_forward and have explicit max_steps <= 5.

Example (run from app-root `coint4/`):

  PYTHONPATH=src ./.venv/bin/python scripts/optimization/clean_cycle_top10/build_baseline_queue.py \
    --dry-run
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from definitions import BASELINE_DIR as DEFAULT_BASELINE_DIR
from definitions import CLEAN_AGG_DIR, FIXED_WINDOWS


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

_BASELINE_PREFIX_RE = re.compile(r"^(?P<prefix>b[0-9]{2})_")


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
    return project_root / (_normalize_repo_relative_path(raw) or raw)


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


def _to_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
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


def _discover_baseline_configs(dir_path: Path) -> List[Path]:
    if not dir_path.exists() or not dir_path.is_dir():
        raise SystemExit(f"--configs-dir must be an existing directory: {dir_path}")
    configs = sorted([p for p in dir_path.glob("b*.yaml") if p.is_file()])
    configs += sorted([p for p in dir_path.glob("b*.yml") if p.is_file()])
    # Deduplicate while keeping deterministic order.
    seen: set[Path] = set()
    out: List[Path] = []
    for p in configs:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _baseline_sort_key(path: Path) -> Tuple[int, str]:
    match = _BASELINE_PREFIX_RE.match(path.stem)
    if not match:
        return (10_000, path.name)
    prefix = match.group("prefix")
    try:
        idx = int(prefix[1:])
    except ValueError:
        idx = 10_000
    return (idx, path.name)


def _queue_row_for_config(cfg_path: Path, *, project_root: Path, baseline_dir: str) -> Tuple[str, str]:
    cfg_rel = _try_relpath(cfg_path, project_root)
    run_id = cfg_path.stem
    results_dir = str(Path(baseline_dir) / run_id)
    return cfg_rel, results_dir


def _write_queue(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["config_path", "results_dir", "status"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build baseline run_queue CSV (clean_cycle_top10)")
    parser.add_argument(
        "--configs-dir",
        default="configs/clean_cycle_top10/baseline",
        help="Directory containing baseline YAML configs (relative to app-root `coint4/` unless absolute).",
    )
    parser.add_argument(
        "--baseline-dir",
        default=DEFAULT_BASELINE_DIR,
        help="BASELINE_DIR for baseline results (relative to app-root `coint4/` unless absolute).",
    )
    parser.add_argument(
        "--output",
        default=f"{CLEAN_AGG_DIR}/baseline_run_queue.csv",
        help="Output CSV path (relative to app-root `coint4/` unless absolute).",
    )
    parser.add_argument(
        "--status",
        default="planned",
        help="Initial status to write for each queue row (default: planned).",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=10,
        help="Expected number of baseline configs (default: 10).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan without writing CSV.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output CSV.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    project_root = _resolve_project_root()

    configs_dir = _resolve_under_project(args.configs_dir, project_root) or (project_root / args.configs_dir)
    configs_dir = _ensure_under_project(configs_dir, project_root)

    baseline_dir_raw = str(args.baseline_dir or "").strip()
    if not baseline_dir_raw:
        raise SystemExit("--baseline-dir is empty")
    baseline_dir_path = _resolve_under_project(baseline_dir_raw, project_root) or (project_root / baseline_dir_raw)
    baseline_dir_path = _ensure_under_project(baseline_dir_path, project_root)
    baseline_dir_rel = _try_relpath(baseline_dir_path, project_root)

    output_path = _resolve_under_project(args.output, project_root) or (project_root / args.output)
    output_path = _ensure_under_project(output_path, project_root)

    if output_path.exists() and not args.overwrite and not args.dry_run:
        raise SystemExit(f"refusing to overwrite existing queue: {output_path} (use --overwrite)")

    expected_wf = _load_expected_walk_forward()
    configs = _discover_baseline_configs(configs_dir)
    configs = sorted(configs, key=_baseline_sort_key)

    expected_count = max(1, int(args.expected_count))
    if len(configs) != expected_count:
        cfgs_rel = "\n".join(f"- {_try_relpath(p, project_root)}" for p in configs)
        raise SystemExit(
            f"expected {expected_count} baseline configs, got {len(configs)} in {configs_dir}\n{cfgs_rel}"
        )

    seen_run_ids: set[str] = set()
    seen_results: set[str] = set()
    rows: List[Dict[str, str]] = []
    for cfg_path in configs:
        _validate_config(cfg_path, expected_wf=expected_wf)
        cfg_rel, results_dir = _queue_row_for_config(cfg_path, project_root=project_root, baseline_dir=baseline_dir_rel)
        run_id = cfg_path.stem
        if run_id in seen_run_ids:
            raise SystemExit(f"duplicate run_id from config stem: {run_id}")
        if results_dir in seen_results:
            raise SystemExit(f"duplicate results_dir: {results_dir}")
        seen_run_ids.add(run_id)
        seen_results.add(results_dir)
        rows.append({"config_path": cfg_rel, "results_dir": results_dir, "status": str(args.status).strip() or "planned"})

    if args.dry_run:
        print("Would write baseline queue:")
        print(f"- output: {_try_relpath(output_path, project_root)}")
        print(f"- configs_dir: {_try_relpath(configs_dir, project_root)}")
        print(f"- baseline_dir: {baseline_dir_rel}")
        print(f"- rows: {len(rows)}")
        for row in rows:
            print(f"  - {row['config_path']} -> {row['results_dir']} ({row['status']})")
        return 0

    _write_queue(output_path, rows)
    print(f"Wrote baseline queue: {_try_relpath(output_path, project_root)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
