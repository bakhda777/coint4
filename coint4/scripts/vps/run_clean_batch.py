#!/usr/bin/env python3
"""Run clean-cycle baseline/sweep batches from a run_queue.csv with safety guards.

This script is designed for running on the VPS (heavy execution host). It runs
queue entries sequentially, writes a per-run log into results_dir, and updates
queue statuses (planned -> running -> completed/stalled).

Safety defaults:
- Refuse overwriting existing results directories unless --allow-overwrite.
- If any target results_dir is under a frozen baseline directory (BASELINE_FROZEN.txt),
  refuse writes unless --allow-overwrite.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from coint2.ops.run_queue import RunQueueEntry, load_run_queue, select_by_status, write_run_queue


DEFAULT_RUNNER = "run_wfa_fullcpu.sh"
DEFAULT_STATUSES = "planned,stalled"


def _resolve_project_root(explicit: Optional[str]) -> Path:
    """Return app-root (directory containing pyproject.toml)."""
    if explicit:
        root = Path(str(explicit)).expanduser().resolve()
        if not (root / "pyproject.toml").exists():
            raise SystemExit(f"--project-root must contain pyproject.toml: {root}")
        return root

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: assume we are under <app_root>/scripts/vps/**.
    return here.parents[2]


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


def _ensure_under_project(path: Path, project_root: Path) -> Path:
    resolved = path.resolve()
    root = project_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"refusing to operate outside project root: {resolved}") from exc
    return resolved


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rotate_file(path: Path, *, suffix: str) -> None:
    if not path.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rotated = path.with_name(f"{path.name}.{suffix}_{stamp}")
    path.rename(rotated)


def _is_dir_empty(path: Path) -> bool:
    if not path.exists():
        return True
    if not path.is_dir():
        return False
    try:
        next(path.iterdir())
    except StopIteration:
        return True
    return False


def _baseline_dir_for_results_dir(results_dir: Path) -> Optional[Path]:
    """Return baseline_top10 dir if results_dir is inside it."""
    for candidate in [results_dir, results_dir.parent, *results_dir.parents]:
        if candidate.name == "baseline_top10":
            return candidate
    return None


def _load_baseline_guard(project_root: Path) -> Any:
    """Load scripts/optimization/clean_cycle_top10/baseline_guard.py as a module."""
    guard_path = project_root / "scripts" / "optimization" / "clean_cycle_top10" / "baseline_guard.py"
    if not guard_path.exists():
        raise SystemExit(f"baseline_guard not found: {guard_path}")

    spec = importlib.util.spec_from_file_location("baseline_guard", guard_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"failed to load baseline_guard module: {guard_path}")

    mod = importlib.util.module_from_spec(spec)
    # Some libs look up sys.modules during import; register before exec.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _parse_statuses(raw: str) -> List[str]:
    items = [s.strip() for s in str(raw or "").split(",")]
    return [s for s in items if s]


def _validate_runner(runner: Path) -> None:
    if not runner.exists():
        raise SystemExit(f"runner not found: {runner}")
    if runner.is_dir():
        raise SystemExit(f"runner must be a file, got directory: {runner}")


def _validate_selected_entries(
    *,
    selected: Sequence[RunQueueEntry],
    project_root: Path,
    allow_overwrite: bool,
    refuse_overwrite: bool,
) -> None:
    baseline_dirs: List[Path] = []
    for entry in selected:
        results_dir = _resolve_under_project(entry.results_dir, project_root)
        if results_dir is None:
            raise SystemExit(f"invalid empty results_dir in queue: {entry}")
        results_dir = _ensure_under_project(results_dir, project_root)
        baseline_dir = _baseline_dir_for_results_dir(results_dir)
        if baseline_dir is not None and baseline_dir not in baseline_dirs:
            baseline_dirs.append(baseline_dir)

        if refuse_overwrite and not allow_overwrite:
            if results_dir.exists() and not _is_dir_empty(results_dir):
                raise SystemExit(
                    "refusing to overwrite existing results_dir: {path} "
                    "(use --allow-overwrite or clean the directory / fix queue status)".format(
                        path=results_dir
                    )
                )

    if baseline_dirs:
        guard = _load_baseline_guard(project_root)
        for baseline_dir in baseline_dirs:
            guard.refuse_if_frozen(
                baseline_dir=baseline_dir,
                allow_overwrite=allow_overwrite,
                action="run clean batch into baseline dir",
            )


def _print_plan(
    *,
    header: str,
    selected: Sequence[RunQueueEntry],
    runner: Path,
) -> None:
    print(header)
    for entry in selected:
        cmd = [str(runner), entry.config_path, entry.results_dir]
        print(f"- {entry.status}: {entry.config_path} -> {entry.results_dir}")
        print(f"  cmd: {shlex.join(cmd)}")


def _run_single_entry(
    *,
    entry: RunQueueEntry,
    runner: Path,
    project_root: Path,
    allow_overwrite: bool,
) -> str:
    results_dir = _ensure_under_project(
        _resolve_under_project(entry.results_dir, project_root) or (project_root / entry.results_dir),
        project_root,
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    log_path = results_dir / "run.log"
    if log_path.exists() and allow_overwrite:
        _rotate_file(log_path, suffix="prev")

    cmd = [str(runner), entry.config_path, entry.results_dir]
    with log_path.open("w") as handle:
        handle.write(f"[run_clean_batch] {_utc_stamp()} cmd: {shlex.join(cmd)}\n")
        handle.flush()
        proc = subprocess.run(
            cmd,
            cwd=project_root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    metrics_path = results_dir / "strategy_metrics.csv"
    if proc.returncode == 0 and metrics_path.exists():
        return "completed"
    return "stalled"


def _run_queue_file(
    *,
    queue_path: Path,
    runner: Path,
    statuses: Iterable[str],
    project_root: Path,
    dry_run: bool,
    refuse_overwrite: bool,
    allow_overwrite: bool,
) -> int:
    entries = load_run_queue(queue_path)
    selected = select_by_status(entries, statuses)
    if not selected:
        print(f"{queue_path}: nothing to run for {', '.join(statuses)}")
        return 0

    _validate_runner(runner)
    _validate_selected_entries(
        selected=selected,
        project_root=project_root,
        allow_overwrite=allow_overwrite,
        refuse_overwrite=refuse_overwrite,
    )

    _print_plan(
        header=f"{queue_path}: queued {len(selected)} run(s)",
        selected=selected,
        runner=runner,
    )

    if dry_run:
        return 0

    any_stalled = False
    for entry in selected:
        entry.status = "running"
        write_run_queue(queue_path, entries)

        status = _run_single_entry(
            entry=entry,
            runner=runner,
            project_root=project_root,
            allow_overwrite=allow_overwrite,
        )
        entry.status = status
        write_run_queue(queue_path, entries)
        if status != "completed":
            any_stalled = True

    return 1 if any_stalled else 0


def _load_manifest_entries(path: Path) -> List[Dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        raw_entries = payload.get("entries") or []
    elif isinstance(payload, list):
        raw_entries = payload
    else:
        raise SystemExit(f"unsupported manifest format (expected dict or list): {path}")

    out: List[Dict[str, str]] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        queue = str(item.get("queue") or "").strip()
        config_path = str(item.get("config_path") or "").strip()
        results_dir = str(item.get("results_dir") or "").strip()
        status = str(item.get("status") or "").strip()
        if not queue or not config_path or not results_dir or not status:
            continue
        out.append(
            {
                "queue": queue,
                "config_path": config_path,
                "results_dir": results_dir,
                "status": status,
            }
        )
    return out


def _group_manifest_entries(entries: Sequence[Dict[str, str]]) -> List[Tuple[str, List[Dict[str, str]]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    order: List[str] = []
    for e in entries:
        q = e["queue"]
        if q not in grouped:
            grouped[q] = []
            order.append(q)
        grouped[q].append(e)
    return [(q, grouped[q]) for q in order]


def _run_manifest(
    *,
    manifest_path: Path,
    runner: Path,
    statuses: Iterable[str],
    project_root: Path,
    dry_run: bool,
    refuse_overwrite: bool,
    allow_overwrite: bool,
) -> int:
    entries = _load_manifest_entries(manifest_path)
    if not entries:
        print(f"{manifest_path}: no runnable entries found")
        return 0

    statuses_set = {s.lower() for s in statuses}
    grouped = _group_manifest_entries(entries)
    rc = 0
    for queue_str, group_entries in grouped:
        queue_path = _resolve_under_project(queue_str, project_root)
        if queue_path is None:
            raise SystemExit(f"invalid queue path in manifest: {queue_str!r}")
        queue_path = _ensure_under_project(queue_path, project_root)
        if not queue_path.exists():
            raise SystemExit(f"queue not found (from manifest): {queue_path}")

        queue_rows = load_run_queue(queue_path)
        # Index rows for quick lookup and in-place mutation.
        idx: Dict[Tuple[str, str], RunQueueEntry] = {(r.config_path, r.results_dir): r for r in queue_rows}

        selected: List[RunQueueEntry] = []
        for e in group_entries:
            status = e.get("status", "").strip().lower()
            if status not in statuses_set:
                continue
            key = (e["config_path"], e["results_dir"])
            row = idx.get(key)
            if row is None:
                raise SystemExit(f"manifest entry not found in queue: queue={queue_path} key={key}")
            selected.append(row)

        if not selected:
            print(f"{queue_path}: nothing to run for {', '.join(statuses)} (manifest subset)")
            continue

        _validate_runner(runner)
        _validate_selected_entries(
            selected=selected,
            project_root=project_root,
            allow_overwrite=allow_overwrite,
            refuse_overwrite=refuse_overwrite,
        )

        _print_plan(
            header=f"{queue_path}: queued {len(selected)} run(s) (manifest subset)",
            selected=selected,
            runner=runner,
        )

        if dry_run:
            continue

        any_stalled = False
        for row in selected:
            row.status = "running"
            write_run_queue(queue_path, queue_rows)

            status = _run_single_entry(
                entry=row,
                runner=runner,
                project_root=project_root,
                allow_overwrite=allow_overwrite,
            )
            row.status = status
            write_run_queue(queue_path, queue_rows)
            if status != "completed":
                any_stalled = True

        if any_stalled:
            rc = 1

    return rc


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--queue", help="Path to run_queue.csv (relative to app-root unless absolute).")
    input_group.add_argument(
        "--manifest",
        help=(
            "Path to JSON manifest with entries (e.g. outputs/wfa_manifest_*.json). "
            "Each entry must include queue, config_path, results_dir, status."
        ),
    )

    parser.add_argument("--project-root", default=None, help="App-root containing pyproject.toml (optional).")
    parser.add_argument(
        "--runner",
        default=DEFAULT_RUNNER,
        help=f"Runner script path (relative to app-root; default: {DEFAULT_RUNNER}).",
    )
    parser.add_argument(
        "--statuses",
        default=DEFAULT_STATUSES,
        help=f"Comma-separated statuses to run (default: {DEFAULT_STATUSES}).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands only (no execution).")

    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument(
        "--refuse-overwrite",
        action="store_true",
        help="Refuse overwriting existing results dirs (default).",
    )
    overwrite_group.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow overwriting existing results dirs and bypass frozen-baseline guard.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    project_root = _resolve_project_root(args.project_root)
    statuses = _parse_statuses(args.statuses)
    if not statuses:
        raise SystemExit("--statuses produced empty set")

    runner = _resolve_under_project(args.runner, project_root) or (project_root / args.runner)
    runner = _ensure_under_project(runner, project_root)

    allow_overwrite = bool(args.allow_overwrite)
    refuse_overwrite = not allow_overwrite
    # Keep explicit flag for readability/CLI parity.
    if args.refuse_overwrite:
        refuse_overwrite = True

    if args.queue:
        queue_path = _resolve_under_project(args.queue, project_root)
        if queue_path is None:
            raise SystemExit("--queue is empty")
        queue_path = _ensure_under_project(queue_path, project_root)
        if not queue_path.exists():
            raise SystemExit(f"queue not found: {queue_path}")
        return _run_queue_file(
            queue_path=queue_path,
            runner=runner,
            statuses=statuses,
            project_root=project_root,
            dry_run=bool(args.dry_run),
            refuse_overwrite=refuse_overwrite,
            allow_overwrite=allow_overwrite,
        )

    manifest_path = _resolve_under_project(args.manifest, project_root)
    if manifest_path is None:
        raise SystemExit("--manifest is empty")
    manifest_path = _ensure_under_project(manifest_path, project_root)
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")
    return _run_manifest(
        manifest_path=manifest_path,
        runner=runner,
        statuses=statuses,
        project_root=project_root,
        dry_run=bool(args.dry_run),
        refuse_overwrite=refuse_overwrite,
        allow_overwrite=allow_overwrite,
    )


if __name__ == "__main__":
    raise SystemExit(main())

