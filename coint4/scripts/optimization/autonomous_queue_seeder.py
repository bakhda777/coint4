#!/usr/bin/env python3
"""Autonomous queue seeder for WFA aggregate queues.

This agent watches all aggregate run queues and triggers a new evolution batch
when work is low or missing. It avoids changing promote policy by delegating
entirely to evolve_next_batch.py for queue generation.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _parse_bool(value: str | bool, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_args() -> argparse.Namespace:
    root = _repo_root()
    defaults = {
        "base_config": os.getenv("AUTONOMOUS_QUEUE_SEEDER_BASE_CONFIG", "configs/prod_final_budget1000.yaml"),
        "controller_group": os.getenv(
            "AUTONOMOUS_QUEUE_SEEDER_CONTROLLER_GROUP", os.getenv("CONTROLLER_GROUP", "autonomous_queue_seeder")
        ),
        "run_group_prefix": os.getenv("AUTONOMOUS_QUEUE_SEEDER_RUN_GROUP_PREFIX", "autonomous_seed"),
        "pending_threshold": int(os.getenv("AUTONOMOUS_QUEUE_SEEDER_PENDING_THRESHOLD", "4")),
        "num_variants": int(os.getenv("AUTONOMOUS_QUEUE_SEEDER_NUM_VARIANTS", "8")),
        "max_changed_keys": int(os.getenv("AUTONOMOUS_QUEUE_SEEDER_MAX_CHANGED_KEYS", "4")),
        "dedupe_distance": float(os.getenv("AUTONOMOUS_QUEUE_SEEDER_DEDUPE_DISTANCE", "0.06")),
        "policy_scale": os.getenv("AUTONOMOUS_QUEUE_SEEDER_POLICY_SCALE", "micro"),
        "ir_mode": os.getenv("AUTONOMOUS_QUEUE_SEEDER_IR_MODE", "patch_ast"),
        "llm_propose": _parse_bool(os.getenv("AUTONOMOUS_QUEUE_SEEDER_LLM_PROPOSE", "false"), default=False),
        "llm_model": os.getenv("AUTONOMOUS_QUEUE_SEEDER_LLM_MODEL", "gpt-5.2"),
        "llm_effort": os.getenv("AUTONOMOUS_QUEUE_SEEDER_LLM_EFFORT", "high"),
        "llm_timeout": int(os.getenv("AUTONOMOUS_QUEUE_SEEDER_LLM_TIMEOUT", "300")),
        "run_index": os.getenv("AUTONOMOUS_QUEUE_SEEDER_RUN_INDEX", "artifacts/wfa/aggregate/rollup/run_index.csv"),
        "contains": [token.strip() for token in os.getenv("AUTONOMOUS_QUEUE_SEEDER_CONTAINS", "").split(",") if token.strip()],
        "windows": [token.strip() for token in os.getenv("AUTONOMOUS_QUEUE_SEEDER_WINDOWS", "").split(";") if token.strip()],
        "aggregate_dir": os.getenv("AUTONOMOUS_QUEUE_SEEDER_AGGREGATE_DIR", "artifacts/wfa/aggregate"),
        "python_bin": os.getenv("AUTONOMOUS_QUEUE_SEEDER_PYTHON_BIN", str(root / ".venv" / "bin" / "python")),
    }

    parser = argparse.ArgumentParser(description="Autonomously seed run queue when backlog is low.")
    parser.add_argument("--base-config", default=defaults["base_config"], help="Base config for evolution planner")
    parser.add_argument("--controller-group", default=defaults["controller_group"], help="Controller group for evolution state/decisions")
    parser.add_argument("--run-group-prefix", default=defaults["run_group_prefix"], help="Prefix for generated run groups")
    parser.add_argument("--pending-threshold", type=int, default=defaults["pending_threshold"], help="Trigger when total pending < threshold")
    parser.add_argument("--num-variants", type=int, default=defaults["num_variants"], help="Variants to request from evolve_next_batch")
    parser.add_argument("--max-changed-keys", type=int, default=defaults["max_changed_keys"], help="Max changed knobs per candidate")
    parser.add_argument("--dedupe-distance", type=float, default=defaults["dedupe_distance"], help="Minimum inter-candidate distance")
    parser.add_argument("--policy-scale", default=defaults["policy_scale"], choices=["auto", "micro", "macro"], help="Planner policy scale")
    parser.add_argument("--ir-mode", default=defaults["ir_mode"], choices=["knob", "patch_ast"], help="Evolution IR mode")
    parser.add_argument("--llm-propose", action="store_true", default=defaults["llm_propose"], help="Enable LLM policy override")
    parser.add_argument("--llm-model", default=defaults["llm_model"], help="LLM model for planner")
    parser.add_argument("--llm-effort", default=defaults["llm_effort"], help="LLM effort for planner")
    parser.add_argument("--llm-timeout-sec", type=int, default=defaults["llm_timeout"], help="LLM timeout seconds")
    parser.add_argument("--contains", action="append", default=defaults["contains"], help="run_index contain token(s)")
    parser.add_argument("--window", action="append", default=defaults["windows"], help="walk-forward windows as YYYY-MM-DD,YYYY-MM-DD")
    parser.add_argument("--aggregate-dir", default=defaults["aggregate_dir"], help="Run queue root folder")
    parser.add_argument("--run-index", default=defaults["run_index"], help="run_index CSV path")
    parser.add_argument("--python-bin", default=defaults["python_bin"], help="Python binary for evolve_next_batch")
    parser.add_argument("--state-prefix", default="queue_seeder", help="State artifact prefix")
    return parser.parse_args()


def _resolve_under_root(value: str, root: Path) -> Path:
    path = Path(str(value).strip())
    return path if path.is_absolute() else root / path


def _load_queue_rows(queue_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not queue_path.exists():
        return rows
    with queue_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows


def _count_pending(rows: Iterable[dict[str, str]]) -> int:
    wanted = {"planned", "running", "stalled", "failed"}
    count = 0
    for row in rows:
        if (row.get("status") or "").strip().lower() in wanted:
            count += 1
    return count


def _summarize_queues(aggregate_dir: Path) -> tuple[int, int, int, list[Path]]:
    total_pending = 0
    runnable_queue_count = 0
    scanned = 0
    queue_paths: list[Path] = []
    if not aggregate_dir.exists():
        return (total_pending, runnable_queue_count, scanned, queue_paths)

    for queue_file in sorted(aggregate_dir.glob("*/run_queue.csv")):
        if queue_file.parent.name.startswith("."):
            continue
        scanned += 1
        queue_paths.append(queue_file)
        rows = _load_queue_rows(queue_file)
        pending = _count_pending(rows)
        if pending > 0:
            runnable_queue_count += 1
            total_pending += pending
    return total_pending, runnable_queue_count, scanned, queue_paths


def _emit_state(state_path: Path, payload: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _append_log(log_path: Path, payload: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _load_decision_path(
    *,
    decision_dir: Path,
    before_mtime: float,
    app_root: Path,
) -> dict[str, Any] | None:
    if not decision_dir.exists():
        return None
    candidates = sorted(
        (path for path in decision_dir.glob("*.json") if path.stat().st_mtime >= before_mtime),
        key=lambda p: p.stat().st_mtime,
    )
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        queue_rel = str(payload.get("queue_path", "")).strip()
        if not queue_rel:
            continue
        queue_path = (app_root / queue_rel).resolve()
        if queue_path.exists():
            return {
                "decision_path": path,
                "queue_path": queue_path,
                "decision_payload": payload,
            }
    return None


def main() -> int:
    args = _load_args()

    app_root = _repo_root()
    aggregate_dir = _resolve_under_root(args.aggregate_dir, app_root)
    state_dir = aggregate_dir / ".autonomous"
    lock_path = state_dir / f"{args.state_prefix}.lock"
    state_path = state_dir / f"{args.state_prefix}.state.json"
    log_path = state_dir / f"{args.state_prefix}.log.jsonl"
    planner_script = app_root / "scripts" / "optimization" / "evolve_next_batch.py"

    # Strict fail-closed defaults: do not act when required inputs are missing.
    if not planner_script.exists():
        payload = {
            "ts": _utc_now_iso(),
            "status": "failed",
            "reason": "planner_missing",
            "planner_script": str(planner_script),
        }
        _emit_state(state_path, payload)
        _append_log(log_path, payload)
        return 1

    # Basic locking to prevent concurrent seeders.
    state_dir.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            payload = {
                "ts": _utc_now_iso(),
                "status": "skipped",
                "reason": "lock_busy",
            }
            _emit_state(state_path, payload)
            _append_log(log_path, payload)
            return 0

        try:
            before_mtime = datetime.now().timestamp()
            total_pending, runnable_queue_count, scanned, queue_files = _summarize_queues(aggregate_dir)
            seed_needed = (total_pending < int(args.pending_threshold)) or (runnable_queue_count == 0)
            snapshot = {
                "ts": _utc_now_iso(),
                "status": "skipped" if not seed_needed else "active",
                "total_pending": total_pending,
                "runnable_queue_count": runnable_queue_count,
                "scanned_queue_groups": scanned,
                "queue_files": [str(path) for path in queue_files],
                "pending_threshold": int(args.pending_threshold),
                "trigger": None,
            }
            if not seed_needed:
                _emit_state(state_path, snapshot)
                _append_log(log_path, snapshot)
                return 0

            snapshot["trigger"] = "below_threshold_or_no_runnable"

            controller_group = str(args.controller_group).strip() or "autonomous_queue_seeder"
            base_config = str(args.base_config).strip() or "configs/prod_final_budget1000.yaml"
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_group = f"{str(args.run_group_prefix).strip() or 'autonomous_seed'}_{timestamp}"
            contains = [token.strip() for token in list(args.contains or []) if token.strip()]
            if not contains:
                contains = [controller_group]

            decision_dir = aggregate_dir / controller_group / "decisions"

            cmd: list[str] = [
                str(Path(str(args.python_bin)).expanduser()),
                "scripts/optimization/evolve_next_batch.py",
                "--base-config",
                base_config,
                "--controller-group",
                controller_group,
                "--run-group",
                run_group,
                "--run-index",
                str(_resolve_under_root(args.run_index, app_root)),
                "--num-variants",
                str(int(args.num_variants)),
                "--ir-mode",
                str(args.ir_mode),
                "--dedupe-distance",
                str(float(args.dedupe_distance)),
                "--max-changed-keys",
                str(int(args.max_changed_keys)),
                "--policy-scale",
                str(args.policy_scale),
            ]
            for token in contains:
                cmd.extend(["--contains", token])
            for raw in args.window:
                if not raw:
                    continue
                cmd.extend(["--window", raw])
            if args.llm_propose:
                cmd.extend(
                    [
                        "--llm-propose",
                        "--llm-model",
                        str(args.llm_model),
                        "--llm-effort",
                        str(args.llm_effort),
                        "--llm-timeout-sec",
                        str(int(args.llm_timeout_sec)),
                    ]
                )

            env = os.environ.copy()
            env["PYTHONPATH"] = f"{str((app_root / 'src'))}:{env.get('PYTHONPATH', '')}".rstrip(":")

            result = subprocess.run(
                cmd,
                cwd=str(app_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            snapshot["planner_cmd"] = cmd
            snapshot["returncode"] = int(result.returncode)
            snapshot["planner_stdout_tail"] = (result.stdout or "")[-2000:]
            snapshot["planner_stderr_tail"] = (result.stderr or "")[-2000:]

            if result.returncode != 0:
                snapshot.update({"status": "failed", "reason": "evolve_failed"})
                _emit_state(state_path, snapshot)
                _append_log(log_path, snapshot)
                return int(result.returncode)

            decision = _load_decision_path(
                decision_dir=decision_dir,
                before_mtime=before_mtime,
                app_root=app_root,
            )
            if not decision:
                snapshot.update({"status": "failed", "reason": "no_new_decision"})
                _emit_state(state_path, snapshot)
                _append_log(log_path, snapshot)
                return 1

            queue_path = decision["queue_path"]
            queue_payload = _load_queue_rows(queue_path)
            snapshot.update(
                {
                    "status": "seeded",
                    "status_detail": "queued",
                    "run_group": run_group,
                    "controller_group": controller_group,
                    "decision_path": _safe_rel(decision["decision_path"], app_root),
                    "queue_path": _safe_rel(queue_path, app_root),
                    "queue_rows_generated": len(queue_payload),
                    "reason": None,
                }
            )
            _emit_state(state_path, snapshot)
            _append_log(log_path, snapshot)
            return 0

        except Exception as exc:
            err = {
                "ts": _utc_now_iso(),
                "status": "failed",
                "reason": f"runtime_error:{type(exc).__name__}",
                "detail": str(exc),
            }
            _emit_state(state_path, err)
            _append_log(log_path, err)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
