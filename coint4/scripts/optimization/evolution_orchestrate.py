#!/usr/bin/env python3
"""One-command orchestration for evolution loop.

Plan -> (optional run) -> postprocess -> rank -> reflect.
Heavy run step is opt-in via --run-command.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _python_exec(app_root: Path) -> Path:
    for candidate in (app_root / ".venv" / "bin" / "python", app_root / ".venv" / "bin" / "python3"):
        if candidate.exists():
            return candidate
    return Path(sys.executable)


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=str(cwd), check=False, text=True, capture_output=True)


def _latest_decision(decision_dir: Path) -> Path:
    candidates = sorted(decision_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no decision files in {decision_dir}")
    return candidates[0]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-command evolution orchestrator (safe by default).")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--controller-group", required=True)
    parser.add_argument("--run-group", required=True)
    parser.add_argument("--contains", action="append", default=[])
    parser.add_argument("--window", action="append", default=[])
    parser.add_argument("--num-variants", type=int, default=12)
    parser.add_argument("--ir-mode", choices=["knob", "patch_ast"], default="knob")
    parser.add_argument("--policy-scale", choices=["auto", "micro", "macro"], default="auto")
    parser.add_argument("--llm-propose", action="store_true")
    parser.add_argument("--llm-model", default="gpt-5.2")
    parser.add_argument("--llm-effort", default="xhigh")
    parser.add_argument("--llm-verify-semantic", action="store_true")
    parser.add_argument("--ast-max-complexity-score", type=float, default=60.0)
    parser.add_argument("--ast-max-redundancy-similarity", type=float, default=0.85)
    parser.add_argument("--patch-max-attempts", type=int, default=8)
    parser.add_argument(
        "--run-command",
        help=(
            "Optional command to execute queue. Use {queue} placeholder for queue path. "
            "If omitted, heavy run step is skipped."
        ),
    )
    parser.add_argument("--skip-postprocess", action="store_true")
    parser.add_argument("--skip-rank", action="store_true")
    parser.add_argument("--skip-reflect", action="store_true")
    parser.add_argument("--output-json", help="Optional orchestration summary JSON path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    app_root = Path(__file__).resolve().parents[2]
    repo_root = app_root.parent
    python_exec = _python_exec(app_root)
    run_index = app_root / "artifacts" / "wfa" / "aggregate" / "rollup" / "run_index.csv"
    decision_dir = app_root / "artifacts" / "wfa" / "aggregate" / args.controller_group / "decisions"
    reflections_dir = app_root / "artifacts" / "wfa" / "aggregate" / args.controller_group / "reflections"

    summary: dict[str, object] = {
        "started_at": _utc_now(),
        "base_config": args.base_config,
        "controller_group": args.controller_group,
        "run_group": args.run_group,
        "steps": [],
    }

    evolve_cmd = [
        str(python_exec),
        "scripts/optimization/evolve_next_batch.py",
        "--base-config",
        args.base_config,
        "--controller-group",
        args.controller_group,
        "--run-group",
        args.run_group,
        "--num-variants",
        str(args.num_variants),
        "--ir-mode",
        str(args.ir_mode),
        "--policy-scale",
        args.policy_scale,
        "--ast-max-complexity-score",
        str(args.ast_max_complexity_score),
        "--ast-max-redundancy-similarity",
        str(args.ast_max_redundancy_similarity),
        "--patch-max-attempts",
        str(args.patch_max_attempts),
    ]
    for token in args.contains:
        evolve_cmd.extend(["--contains", str(token)])
    for window in args.window:
        evolve_cmd.extend(["--window", str(window)])
    if args.llm_propose:
        evolve_cmd.extend(["--llm-propose", "--llm-model", args.llm_model, "--llm-effort", args.llm_effort])
    if args.llm_verify_semantic:
        evolve_cmd.append("--llm-verify-semantic")

    evolve_proc = _run(evolve_cmd, cwd=app_root)
    summary["steps"].append(
        {
            "name": "plan",
            "ok": evolve_proc.returncode == 0,
            "returncode": evolve_proc.returncode,
            "stdout_tail": evolve_proc.stdout[-4000:],
            "stderr_tail": evolve_proc.stderr[-4000:],
        }
    )
    if evolve_proc.returncode != 0:
        _finalize_summary(summary=summary, output_json=args.output_json, repo_root=repo_root)
        return evolve_proc.returncode

    decision_path = _latest_decision(decision_dir)
    decision_payload = json.loads(decision_path.read_text(encoding="utf-8"))
    queue_rel = str(decision_payload.get("queue_path") or "").strip()
    if not queue_rel:
        raise SystemExit("decision has empty queue_path")
    queue_path = app_root / queue_rel

    if args.run_command:
        run_shell = args.run_command.replace("{queue}", shlex.quote(str(queue_path)))
        run_proc = subprocess.run(run_shell, cwd=str(app_root), shell=True, check=False, text=True, capture_output=True)
        summary["steps"].append(
            {
                "name": "run",
                "ok": run_proc.returncode == 0,
                "returncode": run_proc.returncode,
                "command": args.run_command,
                "stdout_tail": run_proc.stdout[-4000:],
                "stderr_tail": run_proc.stderr[-4000:],
            }
        )
        if run_proc.returncode != 0:
            _finalize_summary(summary=summary, output_json=args.output_json, repo_root=repo_root)
            return run_proc.returncode
    else:
        summary["steps"].append(
            {
                "name": "run",
                "ok": True,
                "skipped": True,
                "reason": "no --run-command (safe default, heavy run skipped)",
            }
        )

    if not args.skip_postprocess:
        sync_proc = _run(
            [
                str(python_exec),
                "scripts/optimization/sync_queue_status.py",
                "--queue",
                str(queue_path),
            ],
            cwd=app_root,
        )
        build_proc = _run(
            [
                str(python_exec),
                "scripts/optimization/build_run_index.py",
                "--output-dir",
                "artifacts/wfa/aggregate/rollup",
            ],
            cwd=app_root,
        )
        ok = sync_proc.returncode == 0 and build_proc.returncode == 0
        summary["steps"].append(
            {
                "name": "postprocess",
                "ok": ok,
                "sync_rc": sync_proc.returncode,
                "build_rc": build_proc.returncode,
                "sync_stderr_tail": sync_proc.stderr[-3000:],
                "build_stderr_tail": build_proc.stderr[-3000:],
            }
        )
        if not ok:
            _finalize_summary(summary=summary, output_json=args.output_json, repo_root=repo_root)
            return 1

    if not args.skip_rank:
        rank_out = app_root / "artifacts" / "wfa" / "aggregate" / args.controller_group / f"rank_{args.run_group}.txt"
        rank_proc = _run(
            [
                str(python_exec),
                "scripts/optimization/rank_multiwindow_robust_runs.py",
                "--run-index",
                str(run_index),
                "--contains",
                args.run_group,
                "--top",
                "20",
            ],
            cwd=app_root,
        )
        rank_out.parent.mkdir(parents=True, exist_ok=True)
        rank_out.write_text((rank_proc.stdout or "") + ("\n" + rank_proc.stderr if rank_proc.stderr else ""), encoding="utf-8")
        summary["steps"].append(
            {
                "name": "rank",
                "ok": rank_proc.returncode == 0,
                "returncode": rank_proc.returncode,
                "rank_output": str(rank_out),
            }
        )
        if rank_proc.returncode != 0:
            _finalize_summary(summary=summary, output_json=args.output_json, repo_root=repo_root)
            return rank_proc.returncode

    if not args.skip_reflect:
        reflection_path = reflections_dir / f"reflection_{_utc_now().replace(':', '').replace('-', '')}.json"
        reflect_proc = _run(
            [
                str(python_exec),
                "scripts/optimization/reflect_next_action.py",
                "--decision",
                str(decision_path),
                "--run-index",
                str(run_index),
                "--contains",
                args.run_group,
                "--output-json",
                str(reflection_path),
            ],
            cwd=app_root,
        )
        summary["steps"].append(
            {
                "name": "reflect",
                "ok": reflect_proc.returncode == 0,
                "returncode": reflect_proc.returncode,
                "reflection_path": str(reflection_path),
            }
        )
        if reflect_proc.returncode != 0:
            _finalize_summary(summary=summary, output_json=args.output_json, repo_root=repo_root)
            return reflect_proc.returncode

    summary["finished_at"] = _utc_now()
    summary["ok"] = True
    _finalize_summary(summary=summary, output_json=args.output_json, repo_root=repo_root)
    print("Orchestration completed.")
    return 0


def _finalize_summary(*, summary: dict[str, object], output_json: str | None, repo_root: Path) -> None:
    if output_json:
        target = Path(output_json)
    else:
        target = repo_root / "docs" / "optimization_orchestration_latest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote summary: {target}")


if __name__ == "__main__":
    raise SystemExit(main())
