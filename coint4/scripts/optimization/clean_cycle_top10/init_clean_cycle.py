#!/usr/bin/env python3
"""Initialize the clean-cycle directory structure (C04).

Creates directories (no heavy work):
- CLEAN_ROOT
- BASELINE_DIR
- OPT_DIR
- CLEAN_AGG_DIR

Also writes a short README into CLEAN_AGG_DIR describing FIXED_WINDOWS and
overwrite rules.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional, Tuple

from definitions import BASELINE_DIR as DEFAULT_BASELINE_DIR
from definitions import CLEAN_AGG_DIR as DEFAULT_CLEAN_AGG_DIR
from definitions import CLEAN_ROOT as DEFAULT_CLEAN_ROOT
from definitions import DATESTAMP as DEFAULT_DATESTAMP
from definitions import FIXED_WINDOWS
from definitions import OPT_DIR as DEFAULT_OPT_DIR


_DATESTAMP_RE = re.compile(r"^[0-9]{8}$")
_CYCLE_NAME_RE = re.compile(r"^(?P<datestamp>[0-9]{8})_clean_top10$")


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


def _cycle_name_from_root(root_rel: Path) -> str:
    name = root_rel.name
    match = _CYCLE_NAME_RE.match(name)
    if not match:
        raise SystemExit(f"invalid clean root name (expected YYYYMMDD_clean_top10): {name}")
    return name


def _datestamp_from_cycle_name(cycle_name: str) -> str:
    match = _CYCLE_NAME_RE.match(cycle_name)
    if not match:
        raise SystemExit(f"invalid cycle name (expected YYYYMMDD_clean_top10): {cycle_name}")
    return match.group("datestamp")


def _default_paths_from_datestamp(datestamp: str) -> Tuple[str, str, str, str]:
    if not _DATESTAMP_RE.match(datestamp):
        raise SystemExit(f"--datestamp must be YYYYMMDD (digits only): {datestamp!r}")
    cycle_name = f"{datestamp}_clean_top10"
    clean_root = f"artifacts/wfa/runs_clean/{cycle_name}"
    baseline_dir = f"{clean_root}/baseline_top10"
    opt_dir = f"{clean_root}/opt_sweeps"
    clean_agg_dir = f"artifacts/wfa/aggregate/clean_cycle_top10/{cycle_name}"
    return clean_root, baseline_dir, opt_dir, clean_agg_dir


def _readme_text(*, datestamp: str, cycle_name: str, clean_root: str, baseline_dir: str, opt_dir: str, clean_agg_dir: str) -> str:
    wf = FIXED_WINDOWS.get("walk_forward", {})
    wf_json = json.dumps(wf, indent=2, sort_keys=True, ensure_ascii=False)

    # Keep it short and copy-pastable. Paths below are app-root relative.
    return (
        "# Clean cycle TOP-10 (clean_cycle_top10)\n"
        "\n"
        f"- DATESTAMP: {datestamp}\n"
        f"- cycle: {cycle_name}\n"
        "\n"
        "## Paths (relative to app-root `coint4/`)\n"
        f"- CLEAN_ROOT: {clean_root}\n"
        f"- BASELINE_DIR: {baseline_dir}\n"
        f"- OPT_DIR: {opt_dir}\n"
        f"- CLEAN_AGG_DIR: {clean_agg_dir}\n"
        "\n"
        "## FIXED_WINDOWS.walk_forward\n"
        "All baseline runs and clean sweeps must use exactly these walk_forward params.\n"
        "\n"
        "```json\n"
        f"{wf_json}\n"
        "```\n"
        "\n"
        "## Overwrite rules\n"
        "- This init script is idempotent: it only creates missing directories.\n"
        "- README is created once; existing README is not overwritten unless `--overwrite` is used.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize clean-cycle directories (CLEAN_ROOT + aggregate)")
    parser.add_argument(
        "--datestamp",
        default=DEFAULT_DATESTAMP,
        help="Cycle datestamp in YYYYMMDD (default from definitions.py).",
    )
    parser.add_argument(
        "--root",
        default=None,
        help=(
            "Explicit CLEAN_ROOT path (relative to app-root `coint4/` unless absolute). "
            "If set, cycle name is derived from the last path component."
        ),
    )
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument(
        "--refuse-overwrite",
        action="store_true",
        help="Refuse overwriting an existing README file (default).",
    )
    overwrite_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite README if it already exists.",
    )
    args = parser.parse_args()

    project_root = _resolve_project_root()

    if args.root:
        root_path = _resolve_under_project(args.root, project_root)
        if root_path is None:
            raise SystemExit("--root is empty")
        root_path = _ensure_under_project(root_path, project_root)
        root_rel = root_path.relative_to(project_root.resolve())
        if not str(root_rel).startswith("artifacts/wfa/runs_clean/"):
            raise SystemExit(
                f"refusing to init: --root must be under artifacts/wfa/runs_clean/: got {root_rel}"
            )
        cycle_name = _cycle_name_from_root(root_rel)
        datestamp = _datestamp_from_cycle_name(cycle_name)
        clean_root, baseline_dir, opt_dir, clean_agg_dir = _default_paths_from_datestamp(datestamp)
    else:
        datestamp = str(args.datestamp or "").strip()
        clean_root, baseline_dir, opt_dir, clean_agg_dir = _default_paths_from_datestamp(datestamp)
        cycle_name = f"{datestamp}_clean_top10"

    # Allow future changes in definitions while keeping defaults stable.
    if datestamp == DEFAULT_DATESTAMP and not args.root:
        clean_root = DEFAULT_CLEAN_ROOT
        baseline_dir = DEFAULT_BASELINE_DIR
        opt_dir = DEFAULT_OPT_DIR
        clean_agg_dir = DEFAULT_CLEAN_AGG_DIR

    # Resolve all paths under project root and ensure they stay inside it.
    clean_root_path = _ensure_under_project(_resolve_under_project(clean_root, project_root) or (project_root / clean_root), project_root)
    baseline_dir_path = _ensure_under_project(
        _resolve_under_project(baseline_dir, project_root) or (project_root / baseline_dir), project_root
    )
    opt_dir_path = _ensure_under_project(_resolve_under_project(opt_dir, project_root) or (project_root / opt_dir), project_root)
    clean_agg_dir_path = _ensure_under_project(
        _resolve_under_project(clean_agg_dir, project_root) or (project_root / clean_agg_dir), project_root
    )

    # Create missing directories (never delete).
    for path in [clean_root_path, baseline_dir_path, opt_dir_path, clean_agg_dir_path]:
        path.mkdir(parents=True, exist_ok=True)

    readme_path = clean_agg_dir_path / "README_clean_cycle_top10.md"
    if readme_path.exists() and not args.overwrite:
        print(f"README exists, skipping: {_try_relpath(readme_path, project_root)}")
    else:
        content = _readme_text(
            datestamp=datestamp,
            cycle_name=cycle_name,
            clean_root=_try_relpath(clean_root_path, project_root),
            baseline_dir=_try_relpath(baseline_dir_path, project_root),
            opt_dir=_try_relpath(opt_dir_path, project_root),
            clean_agg_dir=_try_relpath(clean_agg_dir_path, project_root),
        )
        tmp_path = readme_path.with_suffix(readme_path.suffix + ".tmp")
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(readme_path)
        print(f"Wrote README: {_try_relpath(readme_path, project_root)}")

    print("Initialized clean-cycle structure:")
    print(f"- CLEAN_ROOT: {_try_relpath(clean_root_path, project_root)}")
    print(f"- BASELINE_DIR: {_try_relpath(baseline_dir_path, project_root)}")
    print(f"- OPT_DIR: {_try_relpath(opt_dir_path, project_root)}")
    print(f"- CLEAN_AGG_DIR: {_try_relpath(clean_agg_dir_path, project_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

