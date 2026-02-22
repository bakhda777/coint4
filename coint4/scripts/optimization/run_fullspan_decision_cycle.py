#!/usr/bin/env python3
"""Canonical fullspan decision cycle: sync -> rollup -> strict rank -> diagnostic rank.

Usage (from coint4/):
  PYTHONPATH=src ./.venv/bin/python scripts/optimization/run_fullspan_decision_cycle.py \
    --queue artifacts/wfa/aggregate/20260220_top3_fullspan_wfa/run_queue.csv \
    --contains 20260220_top3_fullspan_wfa
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def _run(cmd: List[str], *, cwd: Path, env: dict, allow_no_matches: bool = False) -> int:
    print(f"[cycle] $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
    if proc.returncode == 0:
        return 0
    if allow_no_matches and proc.returncode == 1:
        return proc.returncode
    raise SystemExit(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run canonical fullspan decision cycle (sync queue status, rebuild rollup, rank strict/diagnostic)."
    )
    parser.add_argument(
        "--queue",
        action="append",
        required=True,
        default=[],
        help="run_queue.csv path (repeatable, relative to coint4/ unless absolute).",
    )
    parser.add_argument(
        "--contains",
        action="append",
        required=True,
        default=[],
        help="Substring filter(s) passed to ranker (repeatable).",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/wfa/aggregate/rollup",
        help="Rollup output directory.",
    )
    parser.add_argument(
        "--run-index",
        default="artifacts/wfa/aggregate/rollup/run_index.csv",
        help="run_index.csv path used by ranker.",
    )
    parser.add_argument(
        "--strict-tail-worst-gate-pct",
        type=float,
        default=0.20,
        help="Promote profile tail worst-step hard gate.",
    )
    parser.add_argument(
        "--diagnostic-tail-worst-gate-pct",
        type=float,
        default=0.21,
        help="Research profile tail worst-step gate (diagnostic only).",
    )
    parser.add_argument(
        "--research-top",
        type=int,
        default=10,
        help="Top N rows for research (diagnostic) profile output.",
    )
    parser.add_argument("--min-windows", type=int, default=1)
    parser.add_argument("--min-trades", type=int, default=200)
    parser.add_argument("--min-pairs", type=int, default=20)
    parser.add_argument("--max-dd-pct", type=float, default=0.50)
    parser.add_argument("--min-pnl", type=float, default=0.0)
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--tail-quantile", type=float, default=0.20)
    parser.add_argument("--tail-q-soft-loss-pct", type=float, default=0.03)
    parser.add_argument("--tail-worst-soft-loss-pct", type=float, default=0.10)
    parser.add_argument("--tail-q-penalty", type=float, default=2.0)
    parser.add_argument("--tail-worst-penalty", type=float, default=1.0)
    args = parser.parse_args()

    if args.strict_tail_worst_gate_pct > args.diagnostic_tail_worst_gate_pct:
        raise SystemExit(
            "strict-tail-worst-gate-pct must be <= diagnostic-tail-worst-gate-pct "
            "(promote profile cannot be softer than research profile)."
        )

    app_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    src = str(app_root / "src")
    existing = str(env.get("PYTHONPATH") or "").strip()
    env["PYTHONPATH"] = f"{src}:{existing}" if existing else src

    # 1) Sync queue statuses.
    sync_cmd = [str(sys.executable), "scripts/optimization/sync_queue_status.py"]
    for queue_path in args.queue:
        sync_cmd.extend(["--queue", str(queue_path)])
    _run(sync_cmd, cwd=app_root, env=env)

    # 2) Rebuild rollup without implicit auto-sync (already done in step 1).
    build_cmd = [
        str(sys.executable),
        "scripts/optimization/build_run_index.py",
        "--output-dir",
        str(args.output_dir),
        "--no-auto-sync-status",
    ]
    _run(build_cmd, cwd=app_root, env=env)

    # 3) Strict promote profile.
    print(
        "[cycle] profile=strict "
        f"tail_worst_gate_pct={args.strict_tail_worst_gate_pct:.6f} "
        f"contains={','.join(args.contains)}"
    )
    strict_cmd = [
        str(sys.executable),
        "scripts/optimization/rank_multiwindow_robust_runs.py",
        "--run-index",
        str(args.run_index),
        "--fullspan-policy-v1",
        "--min-windows",
        str(args.min_windows),
        "--min-trades",
        str(args.min_trades),
        "--min-pairs",
        str(args.min_pairs),
        "--max-dd-pct",
        str(args.max_dd_pct),
        "--min-pnl",
        str(args.min_pnl),
        "--initial-capital",
        str(args.initial_capital),
        "--tail-quantile",
        str(args.tail_quantile),
        "--tail-q-soft-loss-pct",
        str(args.tail_q_soft_loss_pct),
        "--tail-worst-soft-loss-pct",
        str(args.tail_worst_soft_loss_pct),
        "--tail-q-penalty",
        str(args.tail_q_penalty),
        "--tail-worst-penalty",
        str(args.tail_worst_penalty),
        "--tail-worst-gate-pct",
        str(args.strict_tail_worst_gate_pct),
    ]
    for needle in args.contains:
        strict_cmd.extend(["--contains", str(needle)])
    strict_rc = _run(strict_cmd, cwd=app_root, env=env, allow_no_matches=True)
    print(f"[cycle] strict_profile_rc={strict_rc}")

    # 4) Diagnostic research profile.
    print(
        "[cycle] profile=research "
        f"tail_worst_gate_pct={args.diagnostic_tail_worst_gate_pct:.6f} "
        f"top={args.research_top}"
    )
    diag_cmd = strict_cmd.copy()
    gate_idx = diag_cmd.index("--tail-worst-gate-pct") + 1
    diag_cmd[gate_idx] = str(args.diagnostic_tail_worst_gate_pct)
    diag_cmd.extend(["--top", str(args.research_top)])
    diag_rc = _run(diag_cmd, cwd=app_root, env=env, allow_no_matches=True)
    print(f"[cycle] research_profile_rc={diag_rc}")

    print("[cycle] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
