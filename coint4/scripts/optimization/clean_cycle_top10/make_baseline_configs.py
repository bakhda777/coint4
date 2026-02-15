#!/usr/bin/env python3
"""Generate baseline config YAMLs (TOP-10) for clean_cycle_top10.

Reads `baseline_manifest.json` (from `select_top10.py`) and writes 10 baseline
YAML configs into `configs/clean_cycle_top10/baseline/`.

For each entry, this script:
- loads the source config from manifest["config_path"]
- patches walk_forward window fields to match FIXED_WINDOWS.walk_forward
- writes `b01_<run_slug>.yaml` ... `b10_<run_slug>.yaml`

This script never modifies the original configs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from definitions import FIXED_WINDOWS


_SAFE_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")
_UNDERSCORES_RE = re.compile(r"_+")

_WF_REQUIRED_KEYS = [
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


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise SystemExit(f"manifest must be a JSON array: {path}")
        out: List[Dict[str, Any]] = []
        for idx, item in enumerate(payload, 1):
            if not isinstance(item, dict):
                raise SystemExit(f"manifest entry #{idx} must be an object: {path}")
            out.append(item)
        return out

    if suffix == ".csv":
        out_rows: List[Dict[str, Any]] = []
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                out_rows.append(dict(row))
        return out_rows

    raise SystemExit(f"unsupported manifest format (expected .json or .csv): {path}")


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


def _require_str(value: Any, *, label: str) -> str:
    s = str(value or "").strip()
    if not s:
        raise SystemExit(f"{label} must be a non-empty string")
    return s


def _slugify(value: str) -> str:
    raw = str(value or "").strip()
    slug = _SAFE_SLUG_RE.sub("_", raw)
    slug = _UNDERSCORES_RE.sub("_", slug).strip("_")
    if not slug:
        slug = "run"

    # Keep filenames within typical filesystem limits and deterministic.
    if len(slug) > 200:
        digest8 = hashlib.sha256(slug.encode("utf-8")).hexdigest()[:8]
        slug = f"{slug[:191]}_{digest8}"
    return slug


def _load_fixed_windows(path: Optional[Path]) -> Dict[str, Any]:
    """Return normalized FIXED_WINDOWS.walk_forward dict."""
    node: Any
    if path is None:
        node = FIXED_WINDOWS
    else:
        node = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(node, dict):
        raise SystemExit("fixed windows must be a JSON object/dict")

    wf = node.get("walk_forward", node)
    if not isinstance(wf, dict):
        raise SystemExit("FIXED_WINDOWS.walk_forward must be a JSON object/dict")

    missing = [k for k in _WF_REQUIRED_KEYS if k not in wf]
    if missing:
        raise SystemExit("FIXED_WINDOWS.walk_forward missing keys: " + ", ".join(missing))

    out: Dict[str, Any] = {}
    out["start_date"] = _require_str(wf.get("start_date"), label="FIXED_WINDOWS.walk_forward.start_date")
    out["end_date"] = _require_str(wf.get("end_date"), label="FIXED_WINDOWS.walk_forward.end_date")

    for key in ["training_period_days", "testing_period_days", "step_size_days", "max_steps", "gap_minutes"]:
        value = _to_int(wf.get(key))
        if value is None:
            raise SystemExit(f"FIXED_WINDOWS.walk_forward.{key} must be an int")
        out[key] = value

    refit = wf.get("refit_frequency")
    out["refit_frequency"] = None if refit is None else str(refit).strip()

    max_steps = int(out.get("max_steps") or 0)
    if max_steps <= 0:
        raise SystemExit("FIXED_WINDOWS.walk_forward.max_steps must be a positive int")
    if max_steps > 5:
        raise SystemExit("unsafe FIXED_WINDOWS.walk_forward.max_steps (must be <= 5 for queue-compatible runs)")

    return out


def _patch_walk_forward(config: Dict[str, Any], fixed_wf: Dict[str, Any], *, src_rel: str) -> None:
    wf = config.get("walk_forward")
    if not isinstance(wf, dict):
        raise SystemExit(f"source config missing walk_forward mapping: {src_rel}")
    for key, value in fixed_wf.items():
        wf[key] = value


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate baseline configs (10 YAML) with normalized walk_forward windows")
    parser.add_argument(
        "--manifest",
        default="artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/baseline_manifest.json",
        help="Path to baseline_manifest.(json|csv) (relative to coint4/ unless absolute).",
    )
    parser.add_argument(
        "--out-dir",
        default="configs/clean_cycle_top10/baseline",
        help="Where to write baseline configs (relative to coint4/ unless absolute).",
    )
    parser.add_argument(
        "--fixed-windows",
        default=None,
        help="Optional: JSON file containing FIXED_WINDOWS (with walk_forward) or a raw walk_forward dict.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned writes but do not write files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output YAML files.")
    args = parser.parse_args()

    project_root = _resolve_project_root()
    manifest_path = _resolve_under_project(args.manifest, project_root) or (project_root / args.manifest)
    out_dir = _resolve_under_project(args.out_dir, project_root) or (project_root / args.out_dir)
    fixed_path = _resolve_under_project(args.fixed_windows, project_root) if args.fixed_windows else None

    manifest_path = _ensure_under_project(manifest_path, project_root)
    out_dir = _ensure_under_project(out_dir, project_root)
    if fixed_path is not None:
        fixed_path = _ensure_under_project(fixed_path, project_root)

    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")

    entries = _load_manifest(manifest_path)
    if len(entries) != 10:
        raise SystemExit(f"expected exactly 10 manifest entries, got {len(entries)}: {manifest_path}")

    fixed_wf = _load_fixed_windows(fixed_path)

    # Keep ordering deterministic even if ranks are missing/duplicated:
    # primary: rank (if any), secondary: original manifest order.
    indexed = list(enumerate(entries))
    indexed_sorted = sorted(
        indexed,
        key=lambda t: (
            _to_int(t[1].get("rank")) or (t[0] + 1),
            t[0],
        ),
    )
    entries_sorted = [e for _, e in indexed_sorted]

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for idx, entry in enumerate(entries_sorted, 1):
        rank = _to_int(entry.get("rank")) or idx
        run_group = str(entry.get("run_group") or "").strip()
        run_id = str(entry.get("run_id") or "").strip()
        config_path_str = str(entry.get("config_path") or "").strip()
        if not config_path_str:
            raise SystemExit(f"manifest entry rank={rank} is missing config_path")

        src_cfg = _resolve_under_project(config_path_str, project_root)
        if src_cfg is None:
            raise SystemExit(f"invalid config_path in manifest (rank={rank}): {config_path_str!r}")
        src_cfg = _ensure_under_project(src_cfg, project_root)
        if not src_cfg.exists() or not src_cfg.is_file():
            raise SystemExit(f"source config not found (rank={rank}): {_try_relpath(src_cfg, project_root)}")

        slug = _slugify(f"{run_group}__{run_id}" if (run_group or run_id) else src_cfg.stem)
        out_name = f"b{rank:02d}_{slug}.yaml"
        out_path = out_dir / out_name

        if out_path.exists() and not args.overwrite:
            raise SystemExit(f"refusing to overwrite existing output: {_try_relpath(out_path, project_root)} (use --overwrite)")

        raw_cfg = yaml.safe_load(src_cfg.read_text(encoding="utf-8")) or {}
        if not isinstance(raw_cfg, dict):
            raise SystemExit(f"source config must parse into a mapping/object: {_try_relpath(src_cfg, project_root)}")

        _patch_walk_forward(raw_cfg, fixed_wf, src_rel=_try_relpath(src_cfg, project_root))

        if args.dry_run:
            print(
                "DRY: {out} <= {src}".format(
                    out=_try_relpath(out_path, project_root),
                    src=_try_relpath(src_cfg, project_root),
                )
            )
            continue

        dumped = yaml.safe_dump(raw_cfg, sort_keys=False, allow_unicode=True, default_flow_style=False)
        if not dumped.endswith("\n"):
            dumped += "\n"
        out_path.write_text(dumped, encoding="utf-8")
        print(f"Wrote: {_try_relpath(out_path, project_root)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
