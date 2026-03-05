#!/usr/bin/env python3
"""Fail-closed promotion gatekeeper for strict fullspan.

Single source of truth for promotion transition:
- PROMOTE_ELIGIBLE is assigned only by this agent.
- cutover_permission is ALLOW_PROMOTE only after full contract pass.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fullspan_contract import FullspanThresholds, evaluate_variant_contract
from fullspan_lineage import count_confirms_by_lineage, derive_candidate_uid


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


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


@dataclass(frozen=True)
class GateConfig:
    min_groups: int
    min_replays: int
    min_windows: int
    thresholds: FullspanThresholds
    tail_quantile: float
    tail_q_soft_loss_pct: float
    tail_worst_soft_loss_pct: float
    tail_q_penalty: float
    tail_worst_penalty: float


def evaluate_queue(
    *,
    queue: str,
    entry: dict,
    cfg: GateConfig,
    run_index_path: Path,
    registry_path: Path,
) -> tuple[dict, list[dict], list[dict]]:
    events: list[dict] = []
    ledger: list[dict] = []
    updated = dict(entry)

    strict_pass_count = parse_int(updated.get("strict_pass_count"), 0)
    strict_run_groups = parse_int(updated.get("strict_run_group_count"), 0)
    strict_gate_status = str(updated.get("strict_gate_status") or "").strip()
    rejection_reason = str(updated.get("rejection_reason") or "").strip()
    verdict = str(updated.get("promotion_verdict") or "ANALYZE").strip().upper() or "ANALYZE"

    top_run_group = str(updated.get("top_run_group") or "").strip()
    top_variant = str(updated.get("top_variant") or "").strip()
    top_score = str(updated.get("top_score") or "").strip()

    candidate_uid = str(updated.get("candidate_uid") or "").strip()
    if not candidate_uid:
        candidate_uid = derive_candidate_uid(
            top_run_group=top_run_group,
            top_variant=top_variant,
            top_score=top_score,
            top_config="",
        )
        if candidate_uid:
            updated["candidate_uid"] = candidate_uid

    confirm_count = parse_int(updated.get("confirm_count"), 0)
    confirmed_groups: list[str] = []
    observed_groups: list[str] = []
    confirmed_lineage_keys: list[str] = []
    observed_lineage_keys: list[str] = []
    confirmed_group_lineage_keys: dict[str, list[str]] = {}
    if candidate_uid:
        stats = count_confirms_by_lineage(
            run_index_path=run_index_path,
            registry_path=registry_path,
            candidate_uid=candidate_uid,
        )
        confirm_count = int(stats.confirmed_count)
        confirmed_groups = list(stats.confirmed_run_groups)
        observed_groups = list(stats.observed_run_groups)
        confirmed_lineage_keys = list(stats.confirmed_lineage_keys)
        observed_lineage_keys = list(stats.observed_lineage_keys)
        confirmed_group_lineage_keys = dict(stats.confirmed_group_lineage_keys)

    updated["confirm_count"] = confirm_count
    updated["confirm_verified_run_groups"] = confirmed_groups
    updated["confirm_observed_run_groups"] = observed_groups
    updated["confirm_verified_lineage_keys"] = confirmed_lineage_keys
    updated["confirm_observed_lineage_keys"] = observed_lineage_keys
    updated["confirm_verified_group_lineage_keys"] = confirmed_group_lineage_keys
    updated["confirm_verified_lineage_count"] = len(confirmed_lineage_keys)
    lineage_confirm_count = len(confirmed_lineage_keys)

    contract = evaluate_variant_contract(
        run_index_path=run_index_path,
        run_group=top_run_group,
        variant_id=top_variant,
        thresholds=cfg.thresholds,
        min_windows=cfg.min_windows,
        tail_quantile=cfg.tail_quantile,
        tail_q_soft_loss_pct=cfg.tail_q_soft_loss_pct,
        tail_worst_soft_loss_pct=cfg.tail_worst_soft_loss_pct,
        tail_q_penalty=cfg.tail_q_penalty,
        tail_worst_penalty=cfg.tail_worst_penalty,
    )
    contract_pass = bool(contract.passed and contract.score_fullspan_v1 is not None)
    contract_reason = str(contract.reason or "METRICS_MISSING")
    updated["contract_name"] = "strict_fullspan_holdout_stress_v1"
    updated["contract_hard_pass"] = contract_pass
    updated["contract_reason"] = contract_reason
    updated["score_fullspan_v1"] = (
        float(contract.score_fullspan_v1) if contract.score_fullspan_v1 is not None else None
    )
    updated["avg_robust_sharpe"] = float(contract.avg_robust_sharpe) if contract.avg_robust_sharpe is not None else None
    updated["contract_windows_total"] = int(contract.windows_total)
    updated["contract_windows_passed"] = int(contract.windows_passed)
    updated["ranking_primary_key"] = "score_fullspan_v1"
    updated["ranking_diagnostic_key"] = "avg_robust_sharpe"

    hard_reject = bool(
        verdict == "REJECT"
        or strict_gate_status == "FULLSPAN_PREFILTER_REJECT"
        or (strict_pass_count > 0 and not contract_pass)
    )

    eligible = bool(
        strict_pass_count > 0
        and strict_run_groups >= cfg.min_groups
        and confirm_count >= cfg.min_replays
        and lineage_confirm_count >= cfg.min_replays
        and strict_gate_status != "FULLSPAN_PREFILTER_REJECT"
        and contract_pass
        and not hard_reject
    )

    prev_verdict = verdict
    if eligible:
        verdict = "PROMOTE_ELIGIBLE"
        updated["rejection_reason"] = ""
        updated["cutover_permission"] = "ALLOW_PROMOTE"
        updated["cutover_ready"] = True
    else:
        updated["cutover_permission"] = "FAIL_CLOSED"
        updated["cutover_ready"] = False
        if hard_reject:
            verdict = "REJECT"
            if strict_pass_count > 0 and not contract_pass:
                updated["rejection_reason"] = contract_reason
            if not updated.get("rejection_reason") and rejection_reason:
                updated["rejection_reason"] = rejection_reason
        elif strict_pass_count > 0:
            if strict_run_groups >= cfg.min_groups:
                verdict = "PROMOTE_PENDING_CONFIRM"
                if confirm_count >= cfg.min_replays and lineage_confirm_count < cfg.min_replays:
                    updated["rejection_reason"] = "pending_confirm_lineage"
                elif not updated.get("rejection_reason"):
                    updated["rejection_reason"] = "pending_confirm"
            else:
                verdict = "PROMOTE_DEFER_CONFIRM"
                updated["rejection_reason"] = "insufficient_run_groups"
        else:
            verdict = "ANALYZE"

    updated["promotion_verdict"] = verdict
    updated["gatekeeper_last_update"] = utc_now_iso()

    if verdict != prev_verdict:
        events.append(
            {
                "ts": utc_now_iso(),
                "queue": queue,
                "event": "PROMOTION_VERDICT_CHANGED",
                "from": prev_verdict,
                "to": verdict,
                "candidate_uid": candidate_uid,
                "strict_pass_count": strict_pass_count,
                "strict_run_group_count": strict_run_groups,
                "confirm_count": confirm_count,
                "confirm_lineage_count": len(confirmed_lineage_keys),
                "contract_hard_pass": bool(contract_pass),
                "contract_reason": contract_reason,
            }
        )

    if verdict != prev_verdict or verdict == "PROMOTE_ELIGIBLE":
        ledger.append(
            {
                "ts": utc_now_iso(),
                "queue": queue,
                "candidate_uid": candidate_uid,
                "verdict_before": prev_verdict,
                "verdict_after": verdict,
                "strict_pass_count": strict_pass_count,
                "strict_run_group_count": strict_run_groups,
                "confirm_count": confirm_count,
                "confirm_lineage_count": len(confirmed_lineage_keys),
                "confirm_min_replays": cfg.min_replays,
                "confirm_min_groups": cfg.min_groups,
                "contract_name": "strict_fullspan_holdout_stress_v1",
                "contract_hard_pass": bool(contract_pass),
                "contract_reason": contract_reason,
                "contract_windows_total": int(contract.windows_total),
                "contract_windows_passed": int(contract.windows_passed),
                "score_fullspan_v1": (
                    float(contract.score_fullspan_v1) if contract.score_fullspan_v1 is not None else None
                ),
                "avg_robust_sharpe": (
                    float(contract.avg_robust_sharpe) if contract.avg_robust_sharpe is not None else None
                ),
                "cutover_permission": updated.get("cutover_permission"),
                "cutover_ready": bool(updated.get("cutover_ready")),
            }
        )

    return updated, events, ledger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promotion gatekeeper agent (strict fullspan, fail-closed).")
    parser.add_argument("--root", default="", help="coint4 app root (defaults to script parents[2]).")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve() if str(args.root or "").strip() else Path(__file__).resolve().parents[2]

    state_dir = root / "artifacts" / "wfa" / "aggregate" / ".autonomous"
    state_path = state_dir / "fullspan_decision_state.json"
    registry_path = state_dir / "confirm_lineage_registry.json"
    run_index_path = root / "artifacts" / "wfa" / "aggregate" / "rollup" / "run_index.csv"
    lock_path = state_dir / "promotion_gatekeeper.lock"
    log_path = state_dir / "promotion_gatekeeper.log"
    events_path = state_dir / "promotion_gatekeeper_events.jsonl"
    ledger_path = state_dir / "promotion_ledger.jsonl"

    cfg = GateConfig(
        min_groups=parse_int(os.environ.get("FULLSPAN_CONFIRM_MIN_GROUPS", "2"), 2),
        min_replays=parse_int(os.environ.get("FULLSPAN_CONFIRM_MIN_REPLIES", "2"), 2),
        min_windows=parse_int(os.environ.get("FULLSPAN_MIN_WINDOWS", "1"), 1),
        thresholds=FullspanThresholds(
            min_trades=parse_float(os.environ.get("FULLSPAN_MIN_TRADES", "200"), 200.0),
            min_pairs=parse_float(os.environ.get("FULLSPAN_MIN_PAIRS", "20"), 20.0),
            max_dd_pct=parse_float(os.environ.get("FULLSPAN_MAX_DD_PCT", "0.20"), 0.20),
            min_pnl=parse_float(os.environ.get("FULLSPAN_MIN_PNL", "0"), 0.0),
            initial_capital=parse_float(os.environ.get("FULLSPAN_INITIAL_CAPITAL", "1000"), 1000.0),
            max_worst_step_loss_pct=parse_float(os.environ.get("FULLSPAN_MAX_WORST_STEP_LOSS_PCT", "0.20"), 0.20),
        ),
        tail_quantile=parse_float(os.environ.get("FULLSPAN_TAIL_QUANTILE", "0.20"), 0.20),
        tail_q_soft_loss_pct=parse_float(os.environ.get("FULLSPAN_TAIL_Q_SOFT_LOSS_PCT", "0.03"), 0.03),
        tail_worst_soft_loss_pct=parse_float(os.environ.get("FULLSPAN_TAIL_WORST_SOFT_LOSS_PCT", "0.10"), 0.10),
        tail_q_penalty=parse_float(os.environ.get("FULLSPAN_TAIL_Q_PENALTY", "2.0"), 2.0),
        tail_worst_penalty=parse_float(os.environ.get("FULLSPAN_TAIL_WORST_PENALTY", "1.0"), 1.0),
    )

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

        emitted = 0
        ledger_emitted = 0
        eligible_count = 0
        for queue, entry in list(queues.items()):
            if not isinstance(entry, dict):
                continue
            updated, events, ledger_items = evaluate_queue(
                queue=queue,
                entry=entry,
                cfg=cfg,
                run_index_path=run_index_path,
                registry_path=registry_path,
            )
            queues[queue] = updated
            if str(updated.get("promotion_verdict") or "") == "PROMOTE_ELIGIBLE":
                eligible_count += 1
            for ev in events:
                append_jsonl(events_path, ev)
                emitted += 1
            for item in ledger_items:
                append_jsonl(ledger_path, item)
                ledger_emitted += 1

        state["queues"] = queues
        metrics = state.get("runtime_metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        metrics["promotion_gatekeeper_cycle_count"] = parse_int(metrics.get("promotion_gatekeeper_cycle_count"), 0) + 1
        metrics["promotion_gatekeeper_last_epoch"] = int(datetime.now(timezone.utc).timestamp())
        metrics["promotion_gatekeeper_eligible_queues"] = int(eligible_count)
        metrics["promotion_gatekeeper_ledger_events"] = parse_int(
            metrics.get("promotion_gatekeeper_ledger_events"), 0
        ) + int(ledger_emitted)
        state["runtime_metrics"] = metrics
        state["state_version"] = max(parse_int(state.get("state_version"), 2), 2)

        if not args.dry_run:
            dump_json(state_path, state)

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{utc_now_iso()} | cycle queues={len(queues)} eligible={eligible_count} events={emitted} ledger={ledger_emitted} dry_run={int(bool(args.dry_run))}\n"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
