#!/usr/bin/env python3
"""Independent strict fullspan contract auditor.

Fail-closed behavior:
- re-evaluates contract from run_index for queue top candidate
- if state is inconsistent with contract, forces cutover to FAIL_CLOSED
- if PROMOTE_ELIGIBLE violates contract/confirm minima, demotes decision
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fullspan_contract import FullspanThresholds, evaluate_variant_contract


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value or 0))
    except Exception:
        return default


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return default


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict fullspan contract auditor (fail-closed).")
    parser.add_argument("--root", default="", help="App root (`coint4/`).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve() if str(args.root or "").strip() else Path(__file__).resolve().parents[2]
    state_dir = root / "artifacts" / "wfa" / "aggregate" / ".autonomous"
    state_path = state_dir / "fullspan_decision_state.json"
    run_index_path = root / "artifacts" / "wfa" / "aggregate" / "rollup" / "run_index.csv"
    lock_path = state_dir / "contract_auditor.lock"
    log_path = state_dir / "contract_auditor.log"
    events_path = state_dir / "contract_auditor_events.jsonl"

    thresholds = FullspanThresholds(
        min_trades=parse_float(os.environ.get("FULLSPAN_MIN_TRADES", "200"), 200.0),
        min_pairs=parse_float(os.environ.get("FULLSPAN_MIN_PAIRS", "20"), 20.0),
        max_dd_pct=parse_float(os.environ.get("FULLSPAN_MAX_DD_PCT", "0.20"), 0.20),
        min_pnl=parse_float(os.environ.get("FULLSPAN_MIN_PNL", "0"), 0.0),
        initial_capital=parse_float(os.environ.get("FULLSPAN_INITIAL_CAPITAL", "1000"), 1000.0),
        max_worst_step_loss_pct=parse_float(os.environ.get("FULLSPAN_MAX_WORST_STEP_LOSS_PCT", "0.20"), 0.20),
    )
    min_groups = parse_int(os.environ.get("FULLSPAN_CONFIRM_MIN_GROUPS", "2"), 2)
    min_replies = parse_int(os.environ.get("FULLSPAN_CONFIRM_MIN_REPLIES", "2"), 2)
    min_windows = parse_int(os.environ.get("FULLSPAN_MIN_WINDOWS", "1"), 1)
    tail_quantile = parse_float(os.environ.get("FULLSPAN_TAIL_QUANTILE", "0.20"), 0.20)
    tail_q_soft_loss_pct = parse_float(os.environ.get("FULLSPAN_TAIL_Q_SOFT_LOSS_PCT", "0.03"), 0.03)
    tail_worst_soft_loss_pct = parse_float(os.environ.get("FULLSPAN_TAIL_WORST_SOFT_LOSS_PCT", "0.10"), 0.10)
    tail_q_penalty = parse_float(os.environ.get("FULLSPAN_TAIL_Q_PENALTY", "2.0"), 2.0)
    tail_worst_penalty = parse_float(os.environ.get("FULLSPAN_TAIL_WORST_PENALTY", "1.0"), 1.0)

    state_dir.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return 0

        state = load_json(state_path, {})
        if not isinstance(state, dict):
            state = {}
        queues = state.get("queues", {})
        if not isinstance(queues, dict):
            queues = {}

        changed = 0
        violations = 0
        for queue, entry in list(queues.items()):
            if not isinstance(entry, dict):
                continue

            top_run_group = str(entry.get("top_run_group") or "").strip()
            top_variant = str(entry.get("top_variant") or "").strip()
            verdict_prev = str(entry.get("promotion_verdict") or "ANALYZE").strip().upper() or "ANALYZE"
            strict_pass_count = parse_int(entry.get("strict_pass_count"), 0)
            strict_run_group_count = parse_int(entry.get("strict_run_group_count"), 0)
            confirm_count = parse_int(entry.get("confirm_count"), 0)
            confirm_lineage_count = parse_int(entry.get("confirm_verified_lineage_count"), 0)

            contract_pass_expected = False
            contract_reason_expected = "METRICS_MISSING"
            score_expected = None
            avg_sharpe_expected = None
            contract_windows_total = 0
            contract_windows_passed = 0

            if top_run_group and top_variant and run_index_path.exists():
                contract = evaluate_variant_contract(
                    run_index_path=run_index_path,
                    run_group=top_run_group,
                    variant_id=top_variant,
                    thresholds=thresholds,
                    min_windows=min_windows,
                    tail_quantile=tail_quantile,
                    tail_q_soft_loss_pct=tail_q_soft_loss_pct,
                    tail_worst_soft_loss_pct=tail_worst_soft_loss_pct,
                    tail_q_penalty=tail_q_penalty,
                    tail_worst_penalty=tail_worst_penalty,
                )
                contract_pass_expected = bool(contract.passed and contract.score_fullspan_v1 is not None)
                contract_reason_expected = str(contract.reason or "METRICS_MISSING")
                score_expected = float(contract.score_fullspan_v1) if contract.score_fullspan_v1 is not None else None
                avg_sharpe_expected = float(contract.avg_robust_sharpe) if contract.avg_robust_sharpe is not None else None
                contract_windows_total = int(contract.windows_total)
                contract_windows_passed = int(contract.windows_passed)

            stored_contract_pass = bool(entry.get("contract_hard_pass"))
            stored_contract_reason = str(entry.get("contract_reason") or "").strip() or "METRICS_MISSING"

            eligible_expected = bool(
                strict_pass_count > 0
                and strict_run_group_count >= min_groups
                and confirm_count >= min_replies
                and confirm_lineage_count >= min_replies
                and contract_pass_expected
            )

            queue_changed = False

            if stored_contract_pass != contract_pass_expected or stored_contract_reason != contract_reason_expected:
                entry["contract_name"] = "strict_fullspan_holdout_stress_v1"
                entry["contract_hard_pass"] = bool(contract_pass_expected)
                entry["contract_reason"] = contract_reason_expected
                entry["score_fullspan_v1"] = score_expected
                entry["avg_robust_sharpe"] = avg_sharpe_expected
                entry["contract_windows_total"] = contract_windows_total
                entry["contract_windows_passed"] = contract_windows_passed
                queue_changed = True
                violations += 1
                append_jsonl(
                    events_path,
                    {
                        "ts": utc_now_iso(),
                        "queue": queue,
                        "event": "CONTRACT_STATE_MISMATCH",
                        "stored_contract_pass": stored_contract_pass,
                        "expected_contract_pass": contract_pass_expected,
                        "stored_contract_reason": stored_contract_reason,
                        "expected_contract_reason": contract_reason_expected,
                    },
                )

            verdict_new = verdict_prev
            if verdict_prev == "PROMOTE_ELIGIBLE" and not eligible_expected:
                if strict_pass_count > 0 and strict_run_group_count >= min_groups and contract_pass_expected:
                    verdict_new = "PROMOTE_PENDING_CONFIRM"
                else:
                    verdict_new = "REJECT"
                entry["promotion_verdict"] = verdict_new
                entry["cutover_permission"] = "FAIL_CLOSED"
                entry["cutover_ready"] = False
                entry["rejection_reason"] = contract_reason_expected
                queue_changed = True
                violations += 1
                append_jsonl(
                    events_path,
                    {
                        "ts": utc_now_iso(),
                        "queue": queue,
                        "event": "AUDIT_FAIL_CLOSED_OVERRIDE",
                        "from_verdict": verdict_prev,
                        "to_verdict": verdict_new,
                        "contract_reason": contract_reason_expected,
                        "strict_pass_count": strict_pass_count,
                        "strict_run_group_count": strict_run_group_count,
                        "confirm_count": confirm_count,
                        "confirm_lineage_count": confirm_lineage_count,
                    },
                )

            if not eligible_expected and str(entry.get("cutover_permission") or "") == "ALLOW_PROMOTE":
                entry["cutover_permission"] = "FAIL_CLOSED"
                entry["cutover_ready"] = False
                queue_changed = True
                violations += 1
                append_jsonl(
                    events_path,
                    {
                        "ts": utc_now_iso(),
                        "queue": queue,
                        "event": "AUDIT_CUTOVER_FAIL_CLOSED",
                        "reason": contract_reason_expected,
                    },
                )

            if queue_changed:
                entry["contract_auditor_last_update"] = utc_now_iso()
                queues[queue] = entry
                changed += 1

        state["queues"] = queues
        metrics = state.get("runtime_metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        metrics["contract_auditor_cycle_count"] = parse_int(metrics.get("contract_auditor_cycle_count"), 0) + 1
        metrics["contract_auditor_last_epoch"] = int(datetime.now(timezone.utc).timestamp())
        metrics["contract_auditor_last_violations"] = int(violations)
        state["runtime_metrics"] = metrics

        if changed > 0 and not args.dry_run:
            dump_json(state_path, state)

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{utc_now_iso()} | cycle queues={len(queues)} changed={changed} violations={violations} dry_run={int(bool(args.dry_run))}\n"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
