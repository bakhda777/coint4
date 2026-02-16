#!/usr/bin/env python3
"""Post-process a WFA queue after execution (status sync -> canonical metrics -> rollup).

Run from app-root (coint4/) or repo root; paths are resolved relative to app-root.

Typical usage:
  PYTHONPATH=src ./.venv/bin/python scripts/optimization/postprocess_queue.py \
    --queue artifacts/wfa/aggregate/<run_group>/run_queue.csv \
    --bar-minutes 15 \
    --overwrite-canonical \
    --build-rollup
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional


def _resolve_app_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_under_root(path: str, *, root: Path) -> Path:
    raw = str(path or "").strip()
    if not raw:
        raise ValueError("empty path")
    p = Path(raw)
    if p.is_absolute():
        return p
    return root / raw


def _venv_python(app_root: Path) -> Path:
    candidate = app_root / ".venv" / "bin" / "python"
    if candidate.exists():
        return candidate
    candidate = app_root / ".venv" / "bin" / "python3"
    if candidate.exists():
        return candidate
    return Path(sys.executable)


def _run(cmd: List[str], *, cwd: Path, env: dict, check: bool = True) -> int:
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return int(proc.returncode)


def _unique(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _load_completed_results_dirs(queue_path: Path, *, app_root: Path) -> List[str]:
    sys.path.insert(0, str(app_root / "src"))
    from coint2.ops.run_queue import load_run_queue, select_by_status  # type: ignore

    entries = load_run_queue(queue_path)
    completed = select_by_status(entries, ["completed"])
    return _unique([e.results_dir for e in completed if (e.results_dir or "").strip()])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Post-process a WFA run_queue.csv (sync statuses, recompute canonical metrics, rebuild rollup)."
    )
    parser.add_argument(
        "--queue",
        action="append",
        default=[],
        required=True,
        help="Path to run_queue.csv (repeatable; relative to app-root coint4/ unless absolute).",
    )
    parser.add_argument(
        "--bar-minutes",
        type=float,
        default=15.0,
        help="Fixed bar timeframe used for canonical recompute (default: 15).",
    )
    parser.add_argument(
        "--overwrite-canonical",
        action="store_true",
        help="Overwrite existing canonical_metrics.json files (default: compute only missing).",
    )
    parser.add_argument(
        "--build-rollup",
        action="store_true",
        help="Rebuild global rollup index artifacts/wfa/aggregate/rollup after recompute.",
    )
    parser.add_argument(
        "--rollup-scan-runs",
        action="store_true",
        help=(
            "Also scan runs_dir for stray strategy_metrics.csv (slower). "
            "By default rollup is built from run_queue.csv references only."
        ),
    )
    parser.add_argument(
        "--rollup-output-dir",
        default="artifacts/wfa/aggregate/rollup",
        help="Output dir for rollup (relative to app-root).",
    )
    parser.add_argument(
        "--print-rank-multiwindow",
        action="store_true",
        help="Print multi-window robust ranking table for the detected run_group (best-effort).",
    )
    parser.add_argument(
        "--rank-contains",
        action="append",
        default=[],
        help="Substring filters passed to rank script (repeatable).",
    )
    args = parser.parse_args(argv)

    app_root = _resolve_app_root()
    py = _venv_python(app_root)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(app_root / "src")

    queue_paths = [_resolve_under_root(q, root=app_root) for q in args.queue]
    for qp in queue_paths:
        if not qp.exists():
            raise SystemExit(f"Queue not found: {qp}")

    # 1) Sync statuses (planned/running/stalled -> completed when metrics present).
    for qp in queue_paths:
        _run(
            [
                str(py),
                "scripts/optimization/sync_queue_status.py",
                "--queue",
                str(qp.relative_to(app_root)),
            ],
            cwd=app_root,
            env=env,
            check=False,
        )

    # 2) Recompute canonical metrics for completed runs.
    all_results_dirs: List[str] = []
    for qp in queue_paths:
        all_results_dirs.extend(_load_completed_results_dirs(qp, app_root=app_root))
    all_results_dirs = _unique(all_results_dirs)

    # Only compute missing unless overwrite requested.
    to_recompute: List[str] = []
    for results_dir in all_results_dirs:
        results_path = _resolve_under_root(results_dir, root=app_root)
        out_path = results_path / "canonical_metrics.json"
        if args.overwrite_canonical or not out_path.exists():
            to_recompute.append(results_dir)

    if to_recompute:
        cmd = [
            str(py),
            "scripts/optimization/recompute_canonical_metrics.py",
            "--bar-minutes",
            str(float(args.bar_minutes)),
        ]
        if args.overwrite_canonical:
            cmd.append("--overwrite")
        for rd in to_recompute:
            cmd.extend(["--run-dir", rd])

        _run(cmd, cwd=app_root, env=env, check=False)

    # 3) Rebuild rollup index (optional).
    if args.build_rollup:
        cmd = [
            str(py),
            "scripts/optimization/build_run_index.py",
            "--output-dir",
            str(args.rollup_output_dir),
        ]
        if not args.rollup_scan_runs:
            # Fast mode: avoid scanning the entire artifacts/wfa/runs tree.
            cmd.extend(["--runs-dir", "__noop__"])
        _run(cmd, cwd=app_root, env=env, check=True)

    # 4) Print ranking table (best-effort helper for humans).
    if args.print_rank_multiwindow:
        cmd = [
            str(py),
            "scripts/optimization/rank_multiwindow_robust_runs.py",
            "--run-index",
            "artifacts/wfa/aggregate/rollup/run_index.csv",
        ]
        for needle in args.rank_contains:
            cmd.extend(["--contains", str(needle)])
        _run(cmd, cwd=app_root, env=env, check=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
