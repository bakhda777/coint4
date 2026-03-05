#!/usr/bin/env python3
"""Canonical fullspan decision cycle: sync -> rollup -> strict rank -> diagnostic rank.

The script keeps a strict, fail-closed promote pass (`promote_profile`) and an optional
diagnostic/research pass for observability. Diagnostic ranking is skipped on clear strict
hard-gate failures to save compute.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from fullspan_contract import (
    CONTRACT_NAME,
    DIAGNOSTIC_RANKING_KEY,
    FullspanThresholds,
    PRIMARY_RANKING_KEY,
    evaluate_variant_contract,
)


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


def _to_float(value: object, default: float | None = None) -> float | None:
    try:
        text = str(value if value is not None else "").strip()
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def _replace_arg_value(cmd: List[str], flag: str, value: str) -> None:
    for idx in range(len(cmd) - 1):
        if cmd[idx] == flag:
            cmd[idx + 1] = value
            return
    cmd.extend([flag, value])


def _strict_contract_summary(
    args: argparse.Namespace,
    *,
    strict_preview: Dict[str, object],
    run_index_path: Path,
    strict_rc: int,
    strict_hard_reason: str,
) -> Dict[str, object]:
    rows = list(strict_preview.get("rows", []) or [])
    thresholds = FullspanThresholds(
        min_trades=float(args.min_trades),
        min_pairs=float(args.min_pairs),
        max_dd_pct=float(args.max_dd_pct),
        min_pnl=float(args.min_pnl),
        initial_capital=float(args.initial_capital),
        max_worst_step_loss_pct=float(args.strict_tail_worst_gate_pct),
    )

    accepted: list[dict[str, object]] = []
    rejects: Counter[str] = Counter()
    for row in rows:
        if not isinstance(row, dict):
            continue
        run_group = str(row.get("run_group") or "").strip()
        variant_id = str(row.get("variant_id") or "").strip()
        if not run_group or not variant_id:
            continue
        contract = evaluate_variant_contract(
            run_index_path=run_index_path,
            run_group=run_group,
            variant_id=variant_id,
            thresholds=thresholds,
            min_windows=int(args.min_windows),
            tail_quantile=float(args.tail_quantile),
            tail_q_soft_loss_pct=float(args.tail_q_soft_loss_pct),
            tail_worst_soft_loss_pct=float(args.tail_worst_soft_loss_pct),
            tail_q_penalty=float(args.tail_q_penalty),
            tail_worst_penalty=float(args.tail_worst_penalty),
        )
        if not contract.passed or contract.score_fullspan_v1 is None:
            rejects.update([str(contract.reason or "UNKNOWN")])
            continue

        accepted.append(
            {
                "score_mode": "fullspan_v1_contract",
                "score": f"{float(contract.score_fullspan_v1):.6f}",
                PRIMARY_RANKING_KEY: float(contract.score_fullspan_v1),
                "avg_robust_sharpe": float(contract.avg_robust_sharpe or 0.0),
                "worst_robust_sharpe": float(contract.worst_robust_sharpe or 0.0),
                "worst_dd_pct": float(contract.worst_dd_pct or 0.0),
                "worst_robust_pnl": float(contract.worst_robust_pnl or 0.0),
                "worst_step_pnl": float(contract.worst_step_pnl or 0.0),
                "q_step_pnl": float(contract.q_step_pnl or contract.worst_step_pnl or 0.0),
                "min_total_trades": float(contract.min_total_trades or 0.0),
                "min_total_pairs_traded": float(contract.min_total_pairs_traded or 0.0),
                "windows": int(contract.windows_passed),
                "run_group": contract.run_group,
                "variant_id": contract.variant_id,
                "sample_config": contract.sample_config or str(row.get("sample_config") or ""),
                "contract_pass": True,
                "contract_reason": "PASS",
            }
        )

    accepted.sort(
        key=lambda rec: (
            -float(_to_float(rec.get(PRIMARY_RANKING_KEY), -10**9) or -10**9),
            str(rec.get("run_group") or ""),
            str(rec.get("variant_id") or ""),
        ),
    )

    top = accepted[0] if accepted else {}
    groups: list[str] = []
    seen = set()
    for item in accepted:
        group = str(item.get("run_group") or "").strip()
        if group and group not in seen:
            seen.add(group)
            groups.append(group)

    if accepted:
        rejection_reason_line = ""
        status = "pass"
        exit_code = 0
    else:
        if rejects:
            reason_parts = [f"{name}:{count}" for name, count in sorted(rejects.items(), key=lambda x: (-x[1], x[0]))]
            rejection_reason_line = "strict_contract_fail(" + ",".join(reason_parts) + ")"
        else:
            rejection_reason_line = strict_hard_reason
            if not rejection_reason_line and strict_rc == 1:
                rejection_reason_line = "strict_no_match"
        status = "reject"
        exit_code = 1

    return {
        "pass_count": len(accepted),
        "run_group_count": len(groups),
        "run_groups": groups,
        "rows": accepted,
        "top_run_group": str(top.get("run_group") or ""),
        "top_variant": str(top.get("variant_id") or ""),
        "top_config": str(top.get("sample_config") or ""),
        "top_score": str(top.get("score") or ""),
        "rejection_reason_line": rejection_reason_line,
        "exit_code": exit_code,
        "status": status,
        "ranking_primary_key": PRIMARY_RANKING_KEY,
        "diagnostic_key": DIAGNOSTIC_RANKING_KEY,
        "contract_name": CONTRACT_NAME,
        "contract_reject_reasons": dict(rejects),
    }


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
    strict_summary_override: Dict[str, object] | None = None,
) -> Dict[str, object]:
    if strict_summary_override is None:
        strict_summary = _parse_rank_output(strict_out)
        strict_summary.update(
            {
                "exit_code": strict_rc,
                "status": "pass" if strict_rc == 0 and strict_summary["pass_count"] > 0 else "reject",
                "rejection_reason_line": _extract_strict_reason(strict_out) if strict_summary["pass_count"] == 0 else "",
            }
        )
    else:
        strict_summary = dict(strict_summary_override)
    strict_summary.setdefault("ranking_primary_key", PRIMARY_RANKING_KEY)
    strict_summary.setdefault("diagnostic_key", DIAGNOSTIC_RANKING_KEY)
    strict_summary.setdefault("contract_name", CONTRACT_NAME)

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
        "winner_contract": CONTRACT_NAME,
        "primary_ranking_key": PRIMARY_RANKING_KEY,
        "diagnostic_ranking_key": DIAGNOSTIC_RANKING_KEY,
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
        "--strict-top",
        type=int,
        default=200,
        help="Top N strict candidates passed to fullspan contract validation.",
    )
    parser.add_argument(
        "--run-diagnostic-on-strict-pass",
        action="store_true",
        help="Run diagnostic profile even when strict profile already passed (default: skip for fast-lane).",
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
        default=0.20,
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
    run_index_path = Path(args.run_index)
    if not run_index_path.is_absolute():
        run_index_path = app_root / run_index_path
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
        "--top",
        str(args.strict_top),
    ]
    for needle in args.contains:
        strict_cmd.extend(["--contains", str(needle)])
    strict_rc, strict_out = _run(strict_cmd, cwd=app_root, env=env, allow_no_matches=True)
    strict_hard_reason = _extract_strict_reason(strict_out)
    strict_preview = _parse_rank_output(strict_out)
    strict_contract = _strict_contract_summary(
        args,
        strict_preview=strict_preview,
        run_index_path=run_index_path,
        strict_rc=strict_rc,
        strict_hard_reason=strict_hard_reason,
    )
    strict_pass_detected = bool(int(strict_contract.get("pass_count", 0) or 0) > 0)
    print(f"[cycle] strict_profile_rc={strict_rc}")
    print(
        "[cycle] strict_contract pass_count={pass_count} run_groups={run_groups} primary_key={primary_key}".format(
            pass_count=strict_contract.get("pass_count", 0),
            run_groups=strict_contract.get("run_group_count", 0),
            primary_key=PRIMARY_RANKING_KEY,
        )
    )

    # 4) Optional diagnostic research profile.
    # If strict profile has explicit hard-gate hard-fail (or strict-pass fast-lane), diagnostic is skipped.
    diag_out = ""
    diag_rc = 0
    diag_skip_reason = ""
    if strict_rc == 1 and strict_hard_reason:
        diag_skip_reason = f"strict_hard_fail: {strict_hard_reason}"
    elif not strict_pass_detected:
        diag_skip_reason = f"strict_contract_fail: {strict_contract.get('rejection_reason_line', 'no_pass')}"
    elif strict_pass_detected and not args.run_diagnostic_on_strict_pass:
        diag_skip_reason = "strict_pass_fastlane"
    else:
        print(
            "[cycle] profile=research selection_profile=research_profile "
            f"tail_worst_gate_pct={args.diagnostic_tail_worst_gate_pct:.6f} "
            f"top={args.research_top}"
        )
        diag_cmd = strict_cmd.copy()
        _replace_arg_value(diag_cmd, "--tail-worst-gate-pct", str(args.diagnostic_tail_worst_gate_pct))
        _replace_arg_value(diag_cmd, "--top", str(args.research_top))
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
        strict_summary_override=strict_contract,
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
