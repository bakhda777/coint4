#!/usr/bin/env python3
"""Run queued WFA configs with resume-friendly logging."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence

from coint2.ops.run_queue import (
    RunQueueEntry,
    load_run_queue,
    select_by_status,
    write_run_queue,
)
from coint2.ops.heavy_guardrails import (
    DEFAULT_ALLOW_ENV,
    DEFAULT_HOST_ALLOWLIST,
    DEFAULT_MIN_CPU,
    DEFAULT_MIN_RAM_GB,
    HeavyGuardrailConfig,
    ensure_heavy_run_allowed,
    parse_host_allowlist,
)


def _rotate_log(log_path: Path) -> None:
    if not log_path.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rotated = log_path.with_name(f"{log_path.name}.stalled_{stamp}")
    log_path.rename(rotated)


def _run_entry(
    entry: RunQueueEntry, runner: Path, project_root: Path, rotate_logs: bool
) -> str:
    results_dir = project_root / entry.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "run.log"
    if rotate_logs:
        _rotate_log(log_path)
    with log_path.open("w") as handle:
        cmd = [str(runner), entry.config_path, entry.results_dir]
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        handle.write(f"[run_wfa_queue] {timestamp} cmd: {shlex.join(cmd)}\n")
        handle.flush()
        process = subprocess.run(
            cmd,
            cwd=project_root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    metrics_path = results_dir / "strategy_metrics.csv"
    if process.returncode == 0 and metrics_path.exists():
        return "completed"
    return "stalled"


def _update_status(
    queue_path: Path,
    entries: List[RunQueueEntry],
    entry: RunQueueEntry,
    status: str,
    lock: threading.Lock,
) -> None:
    with lock:
        entry.status = status
        write_run_queue(queue_path, entries)


def _run_queue(
    queue_path: Path,
    runner: Path,
    statuses: Iterable[str],
    parallel: int,
    rotate_logs: bool,
    project_root: Path,
    dry_run: bool,
) -> None:
    entries = load_run_queue(queue_path)
    selected = select_by_status(entries, statuses)
    if not selected:
        print(f"{queue_path}: nothing to run for {', '.join(statuses)}")
        return

    print(f"{queue_path}: queued {len(selected)} run(s)")
    for entry in selected:
        print(f" - {entry.status}: {entry.config_path} -> {entry.results_dir}")

    if dry_run:
        return

    lock = threading.Lock()
    for entry in selected:
        _update_status(queue_path, entries, entry, "running", lock)

    if parallel <= 1:
        for entry in selected:
            status = _run_entry(entry, runner, project_root, rotate_logs)
            _update_status(queue_path, entries, entry, status, lock)
        return

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(_run_entry, entry, runner, project_root, rotate_logs): entry
            for entry in selected
        }
        for future in as_completed(futures):
            entry = futures[future]
            status = future.result()
            _update_status(queue_path, entries, entry, status, lock)


def _parse_bool_flag(value: str) -> bool:
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def _resolve_under_root(raw_path: str, *, project_root: Path) -> Path:
    path = Path(str(raw_path or "").strip())
    if path.is_absolute():
        return path
    return project_root / path


def _pythonpath_env(project_root: Path) -> dict:
    env = os.environ.copy()
    src_path = str(project_root / "src")
    existing = str(env.get("PYTHONPATH") or "").strip()
    env["PYTHONPATH"] = f"{src_path}:{existing}" if existing else src_path
    return env


def _run_cmd(cmd: Sequence[str], *, cwd: Path, env: dict) -> None:
    proc = subprocess.run(list(cmd), cwd=str(cwd), env=env, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _sync_queue_statuses(
    *,
    queue_paths: Sequence[Path],
    project_root: Path,
    env: dict,
) -> None:
    if not queue_paths:
        return
    cmd: List[str] = [
        str(sys.executable),
        "scripts/optimization/sync_queue_status.py",
    ]
    for queue_path in queue_paths:
        cmd.extend(["--queue", str(queue_path)])
    _run_cmd(cmd, cwd=project_root, env=env)


def _git_commit(project_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(project_root),
            check=True,
            capture_output=True,
            text=True,
        )
        commit = str(proc.stdout).strip()
        return commit or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _load_completed_entries(queue_paths: Sequence[Path]) -> List[RunQueueEntry]:
    completed: List[RunQueueEntry] = []
    for queue_path in queue_paths:
        entries = load_run_queue(queue_path)
        completed.extend(select_by_status(entries, ["completed"]))
    return completed


def _write_snapshots_for_completed(
    *,
    completed_entries: Sequence[RunQueueEntry],
    project_root: Path,
) -> None:
    commit = _git_commit(project_root)
    for entry in completed_entries:
        run_dir = _resolve_under_root(entry.results_dir, project_root=project_root)
        run_dir.mkdir(parents=True, exist_ok=True)

        config_path = _resolve_under_root(entry.config_path, project_root=project_root)
        if config_path.exists():
            snapshot_path = run_dir / "config_snapshot.yaml"
            snapshot_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

        (run_dir / "git_commit.txt").write_text(f"{commit}\n", encoding="utf-8")


def _recompute_canonical_metrics(
    *,
    completed_entries: Sequence[RunQueueEntry],
    project_root: Path,
    env: dict,
) -> None:
    run_dirs: List[str] = []
    seen: set[str] = set()
    for entry in completed_entries:
        run_dir = str(_resolve_under_root(entry.results_dir, project_root=project_root))
        if run_dir in seen:
            continue
        seen.add(run_dir)
        run_dirs.append(run_dir)

    if not run_dirs:
        return

    cmd: List[str] = [
        str(sys.executable),
        "scripts/optimization/recompute_canonical_metrics.py",
        "--bar-minutes",
        "15",
        "--overwrite",
    ]
    for run_dir in run_dirs:
        cmd.extend(["--run-dir", run_dir])
    _run_cmd(cmd, cwd=project_root, env=env)


def _rebuild_rollup(
    *,
    project_root: Path,
    rollup_output_dir: str,
    rollup_queue_dir: str,
    rollup_runs_dir: str,
    env: dict,
) -> None:
    cmd: List[str] = [
        str(sys.executable),
        "scripts/optimization/build_run_index.py",
        "--output-dir",
        str(rollup_output_dir),
        "--queue-dir",
        str(rollup_queue_dir),
        "--runs-dir",
        str(rollup_runs_dir),
    ]
    _run_cmd(cmd, cwd=project_root, env=env)


def _postprocess_queue_results(
    *,
    queue_paths: Sequence[Path],
    project_root: Path,
    rollup_output_dir: str,
    rollup_queue_dir: str,
    rollup_runs_dir: str,
) -> None:
    env = _pythonpath_env(project_root)
    _sync_queue_statuses(queue_paths=queue_paths, project_root=project_root, env=env)
    completed_entries = _load_completed_entries(queue_paths)
    _write_snapshots_for_completed(completed_entries=completed_entries, project_root=project_root)
    _recompute_canonical_metrics(completed_entries=completed_entries, project_root=project_root, env=env)
    _rebuild_rollup(
        project_root=project_root,
        rollup_output_dir=rollup_output_dir,
        rollup_queue_dir=rollup_queue_dir,
        rollup_runs_dir=rollup_runs_dir,
        env=env,
    )


def _resolve_queue_paths(queue_dir: Path, queue_paths: List[Path]) -> List[Path]:
    if queue_paths:
        return queue_paths
    if not queue_dir.exists():
        return []
    return sorted(queue_dir.rglob("run_queue.csv"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run WFA configs from queue")
    parser.add_argument(
        "--queue-dir",
        default="artifacts/wfa/aggregate",
        help="Directory to search for run_queue.csv files.",
    )
    parser.add_argument(
        "--queue",
        action="append",
        default=[],
        help="Explicit run_queue.csv path (repeatable).",
    )
    parser.add_argument(
        "--statuses",
        default="planned,stalled",
        help="Comma-separated statuses to run.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of concurrent runs (default: CPU count).",
    )
    parser.add_argument(
        "--runner",
        default="run_wfa_fullcpu.sh",
        help="Runner script path (relative to project root).",
    )
    parser.add_argument(
        "--enforce-heavy-guardrails",
        type=_parse_bool_flag,
        default=True,
        help="Require ALLOW_HEAVY_RUN + hostname/resources guardrails before compute.",
    )
    parser.add_argument(
        "--heavy-allow-env",
        default=DEFAULT_ALLOW_ENV,
        help="Env var required for heavy execution (must equal '1').",
    )
    parser.add_argument(
        "--heavy-host-allowlist",
        default=os.environ.get("HEAVY_HOSTNAME_ALLOWLIST", ",".join(DEFAULT_HOST_ALLOWLIST)),
        help="Comma-separated hostname/IP allowlist for heavy execution.",
    )
    parser.add_argument(
        "--heavy-min-ram-gb",
        type=float,
        default=float(os.environ.get("HEAVY_MIN_RAM_GB", DEFAULT_MIN_RAM_GB)),
        help="Minimum RAM requirement for heavy execution.",
    )
    parser.add_argument(
        "--heavy-min-cpu",
        type=int,
        default=int(os.environ.get("HEAVY_MIN_CPU", DEFAULT_MIN_CPU)),
        help="Minimum CPU core requirement for heavy execution.",
    )
    parser.add_argument(
        "--postprocess",
        type=_parse_bool_flag,
        default=False,
        help="Run queue postprocess (status sync, snapshots, canonical metrics, rollup rebuild).",
    )
    parser.add_argument(
        "--rollup-output-dir",
        default="artifacts/wfa/aggregate/rollup",
        help="Rollup output dir for postprocess mode.",
    )
    parser.add_argument(
        "--rollup-queue-dir",
        default="artifacts/wfa/aggregate",
        help="Queue dir scanned by rollup rebuild in postprocess mode.",
    )
    parser.add_argument(
        "--rollup-runs-dir",
        default="artifacts/wfa/runs",
        help="Runs dir scanned by rollup rebuild in postprocess mode.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print queue only.")
    parser.add_argument(
        "--no-rotate-logs",
        action="store_true",
        help="Disable run.log rotation before relaunch.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    queue_dir = project_root / args.queue_dir
    runner = project_root / args.runner
    queue_paths = [project_root / path for path in args.queue]
    queue_paths = _resolve_queue_paths(queue_dir, queue_paths)
    if not queue_paths:
        print("No run_queue.csv files found.")
        return 1

    if args.enforce_heavy_guardrails and not args.dry_run:
        ensure_heavy_run_allowed(
            HeavyGuardrailConfig(
                entrypoint="scripts/optimization/run_wfa_queue.py",
                allow_env=str(args.heavy_allow_env),
                host_allowlist=parse_host_allowlist(str(args.heavy_host_allowlist)),
                min_ram_gb=float(args.heavy_min_ram_gb),
                min_cpu=int(args.heavy_min_cpu),
            )
        )

    statuses = [status.strip() for status in args.statuses.split(",") if status.strip()]
    for queue_path in queue_paths:
        _run_queue(
            queue_path=queue_path,
            runner=runner,
            statuses=statuses,
            parallel=max(1, args.parallel),
            rotate_logs=not args.no_rotate_logs,
            project_root=project_root,
            dry_run=args.dry_run,
        )

    if args.postprocess and not args.dry_run:
        _postprocess_queue_results(
            queue_paths=queue_paths,
            project_root=project_root,
            rollup_output_dir=str(args.rollup_output_dir),
            rollup_queue_dir=str(args.rollup_queue_dir),
            rollup_runs_dir=str(args.rollup_runs_dir),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
