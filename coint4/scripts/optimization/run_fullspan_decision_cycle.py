#!/usr/bin/env python3
"""Canonical fullspan decision cycle: sync -> rollup -> strict rank -> diagnostic rank.

The script keeps a strict, fail-closed promote pass (`promote_profile`) and an optional
diagnostic/research pass for observability. Diagnostic ranking is skipped on clear strict
hard-gate failures to save compute.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def _run(cmd, *, cwd, env, allow_no_matches=False):
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode not in (0, 1) and not (allow_no_matches and proc.returncode == 1):
        raise SystemExit(proc.returncode)
    return proc.returncode, proc.stdout


def _parse_rank_output(output: str) -> Dict[str, object]:
    header: List[str] = []
    rows: List[Dict[str, str]] = []
    for raw in output.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        if re.match(r"^\|\s*rank\s*\|", line, re.IGNORECASE):
            header = [part.strip() for part in line.strip("|").split("|")]
            continue
        if line.startswith("|---") or line.startswith("| -"):
            continue
        if header and re.match(r"^\|\s*\d+\s*\|", line):
            cells = [part.strip() for part in line.strip("|").split("|")]
            if len(cells) < len(header):
                cells.extend([""] * (len(header) - len(cells)))
            row = {k: (cells[i] if i < len(cells) else "") for i, k in enumerate(header)}
            rows.append(row)

    pass_run_groups = [row.get("run_group", "") for row in rows if row.get("run_group", "")]
    uniq = []
    seen = set()
    for g in pass_run_groups:
        if g not in seen:
            uniq.append(g)
            seen.add(g)

    top = rows[0] if rows else {}
    top_run_group = top.get("run_group", "")
    top_variant = top.get("variant_id", "")
    top_score = top.get("score", "")
    top_cfg = top.get("sample_config", "")

    return {
        "pass_count": len(rows),
        "run_group_count": len(uniq),
        "run_groups": uniq,
        "rows": rows,
        "top_run_group": top_run_group,
        "top_variant": top_variant,
        "top_config": top_cfg,
        "top_score": top_score,
    }


def _extract_strict_reason(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"No variants matched fullspan policy v1\s*\((.*?)\)", text)
    if m:
        return m.group(1).strip()
    return ""


def _empty_profile_summary(*, pass_count: int = 0, run_groups: List[str] | None = None, reason: str = "") -> Dict[str, object]:
    return {
        "pass_count": int(pass_count),
        "run_group_count": len(run_groups or []),
        "run_groups": list(run_groups or []),
        "rows": [],
        "top_run_group": "",
        "top_variant": "",
        "top_config": "",
        "top_score": "",
        "rejection_reason_line": reason,
        "exit_code": 1,
        "status": "reject" if pass_count <= 0 else "pass",
    }


def _build_summary(
    args,
    *,
    strict_rc: int,
    strict_out: str,
    diag_rc: int,
    diag_out: str,
    diag_skipped_reason: str,
) -> Dict[str, object]:
    strict_summary = _parse_rank_output(strict_out)
    strict_summary.update(
        {
            "exit_code": strict_rc,
            "status": "pass" if strict_rc == 0 and strict_summary["pass_count"] > 0 else "reject",
            "rejection_reason_line": _extract_strict_reason(strict_out) if strict_summary["pass_count"] == 0 else "",
        }
    )

    if diag_skipped_reason:
        diagnostic_summary = _empty_profile_summary(reason=diag_skipped_reason)
    else:
        diag_summary = _parse_rank_output(diag_out)
        diag_summary.update(
            {
                "exit_code": diag_rc,
                "status": "pass" if diag_rc == 0 and diag_summary["pass_count"] > 0 else "reject",
            }
        )
        diagnostic_summary = diag_summary

    result = {
        "version": 1,
        "queue": args.queue[0],
        "contains": args.contains,
        "strict": strict_summary,
        "diagnostic": diagnostic_summary,
        "policy": "fullspan_v1",
        "mode": "fullspan",
        "selection_profile": "promote_profile",
        "selection_mode": "fullspan",
        "strict_profile": "promote_profile",
        "diagnostic_profile": "research_profile",
        "diagnostic_skipped": bool(diag_skipped_reason),
    }
    return result


def _safe_json_dump(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    parser.add_argument(
        "--min-windows",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--max-dd-pct",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--min-pnl",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=1000.0,
    )
    parser.add_argument(
        "--tail-quantile",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--tail-q-soft-loss-pct",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--tail-worst-soft-loss-pct",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--tail-q-penalty",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--tail-worst-penalty",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--min-coverage-ratio",
        type=float,
        default=0.95,
        help="Coverage floor for ranker (0..1). Default from optimization contract.",
    )
    parser.add_argument(
        "--summary-json",
        default="",
        help="Optional path to JSON summary output (for autonomous drivers/state machines).",
    )

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
        "[cycle] profile=strict selection_profile=promote_profile "
        f"tail_worst_gate_pct={args.strict_tail_worst_gate_pct:.6f} "
        f"contains={','.join(args.contains)}"
    )
    strict_cmd = [
        str(sys.executable),
        "scripts/optimization/rank_multiwindow_robust_runs.py",
        "--run-index",
        str(args.run_index),
        "--fullspan-policy-v1",
        "--selection-mode",
        "fullspan",
        "--selection-profile",
        "promote_profile",
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
        "--min-coverage-ratio",
        str(args.min_coverage_ratio),
    ]
    for needle in args.contains:
        strict_cmd.extend(["--contains", str(needle)])
    strict_rc, strict_out = _run(strict_cmd, cwd=app_root, env=env, allow_no_matches=True)
    strict_hard_reason = _extract_strict_reason(strict_out)
    print(f"[cycle] strict_profile_rc={strict_rc}")

    # 4) Optional diagnostic research profile.
    # If strict profile has explicit hard-gate hard-fail (or zero-tail), diagnostic is skipped.
    diag_out = ""
    diag_rc = 0
    diag_skip_reason = ""
    if strict_rc == 1 and strict_hard_reason:
        diag_skip_reason = f"strict_hard_fail: {strict_hard_reason}"
    else:
        print(
            "[cycle] profile=research selection_profile=research_profile "
            f"tail_worst_gate_pct={args.diagnostic_tail_worst_gate_pct:.6f} "
            f"top={args.research_top}"
        )
        diag_cmd = strict_cmd.copy()
        gate_idx = diag_cmd.index("--tail-worst-gate-pct") + 1
        diag_cmd[gate_idx] = str(args.diagnostic_tail_worst_gate_pct)
        # diagnostic should stay conservative diagnostics-only; keep strict top-K
        # as in default reporting flow while allowing weaker diagnostics gate.
        diag_cmd.extend(["--top", str(args.research_top)])
        diag_cmd[diag_cmd.index("--selection-profile") + 1] = "research_profile"
        diag_rc, diag_out = _run(diag_cmd, cwd=app_root, env=env, allow_no_matches=True)
        print(f"[cycle] research_profile_rc={diag_rc}")

    # 5) Build machine-readable summary for downstream orchestration.
    summary = _build_summary(
        args,
        strict_rc=strict_rc,
        strict_out=strict_out,
        diag_rc=diag_rc,
        diag_out=diag_out,
        diag_skipped_reason=diag_skip_reason,
    )
    print(f"[cycle] strict_pass_count={summary['strict'].get('pass_count', 0)}")
    print(f"[cycle] strict_run_groups={summary['strict'].get('run_group_count', 0)}")
    print(
        f"[cycle] strict_top={summary['strict'].get('top_run_group', '')}/{summary['strict'].get('top_variant', '')}"
    )
    print(f"[cycle] strict_rejection={summary['strict'].get('rejection_reason_line', '')}")

    if args.summary_json:
        _safe_json_dump(Path(args.summary_json), summary)

    print("[cycle] done")

    # Only hard command failures (beyond expected ranker no-match code 1) should fail.
    if strict_rc > 1:
        return strict_rc
    if diag_rc > 1:
        return diag_rc

    # Keep run status for orchestration stable: this is expected fail-closed flow.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
