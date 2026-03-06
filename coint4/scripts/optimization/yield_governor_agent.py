#!/usr/bin/env python3
"""Build fail-safe yield governor state for autonomous queue seeding."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def parse_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    try:
        return float(value)
    except Exception:
        return default


def parse_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def safe_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def canonical_hard_fail_reason(row: Mapping[str, Any] | None) -> str:
    if not isinstance(row, Mapping):
        return "METRICS_MISSING"
    if not parse_bool(row.get("metrics_present")):
        return "METRICS_MISSING"
    if parse_float(row.get("total_trades"), 0.0) < 200.0:
        return "TRADES_FAIL"
    if parse_float(row.get("total_pairs_traded"), 0.0) < 20.0:
        return "PAIRS_FAIL"
    if abs(parse_float(row.get("max_drawdown_on_equity"), 0.0)) > 0.20:
        return "DD_FAIL"
    if parse_float(row.get("total_pnl"), 0.0) < 0.0:
        return "ECONOMIC_FAIL"
    worst_step = row.get("tail_loss_worst_period_pnl")
    if worst_step in {None, ""}:
        worst_step = row.get("tail_loss_worst_pair_pnl")
    if parse_float(worst_step, 0.0) < -200.0:
        return "STEP_FAIL"
    return ""


def _decode_metadata_json(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = str(row.get("metadata_json") or "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _row_lineage_uid(row: Mapping[str, Any], queue_group: str) -> str:
    explicit = str(row.get("lineage_uid") or "").strip()
    if explicit:
        return explicit
    meta = _decode_metadata_json(row)
    explicit = str(meta.get("lineage_uid") or "").strip()
    if explicit:
        return explicit
    return queue_group


def _row_operator_id(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("operator_id") or "").strip()
    if explicit:
        return explicit
    meta = _decode_metadata_json(row)
    explicit = str(meta.get("operator_id") or "").strip()
    if explicit:
        return explicit
    return "unknown"


def _row_run_group_token(queue_group: str, run_index_row: Mapping[str, Any] | None) -> str:
    if isinstance(run_index_row, Mapping):
        token = str(run_index_row.get("run_group") or "").strip()
        if token:
            return token
    return queue_group


def _classify_yield(*, completed: int, metrics_present: int, zero_activity: int, hard_fail: int) -> tuple[str, float]:
    if completed <= 0:
        return "neutral", 0.0
    metrics_rate = float(metrics_present) / float(completed)
    zero_rate = float(zero_activity) / float(completed)
    hard_fail_rate = float(hard_fail) / float(completed)
    sample_bonus = min(float(completed), 24.0) / 24.0
    score = (metrics_rate * 2.0) - zero_rate - (hard_fail_rate * 0.75) + (sample_bonus * 0.25)
    if completed >= 8 and (zero_rate >= 0.75 or metrics_rate <= 0.20 or hard_fail_rate >= 0.80):
        return "cooldown", round(score, 6)
    if completed >= 8 and metrics_rate >= 0.60 and zero_rate <= 0.25 and hard_fail_rate <= 0.40:
        return "boost", round(score, 6)
    return "neutral", round(score, 6)


def build_yield_governor_state(
    *,
    root: Path,
    aggregate_dir: Path,
    run_index_path: Path,
    fullspan_state_path: Path,
    recent_queue_limit: int = 200,
) -> dict[str, Any]:
    run_index_map: dict[str, dict[str, str]] = {}
    if run_index_path.exists():
        try:
            with run_index_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    run_id = str(row.get("run_id") or "").strip()
                    if run_id:
                        run_index_map[run_id] = {k: (v or "").strip() for k, v in row.items()}
        except Exception:
            run_index_map = {}

    fullspan_state = load_json(fullspan_state_path, {})
    queues_state = fullspan_state.get("queues", {}) if isinstance(fullspan_state, dict) else {}
    if not isinstance(queues_state, dict):
        queues_state = {}

    queue_paths = sorted(
        (path for path in aggregate_dir.glob("*/run_queue.csv") if not path.parent.name.startswith(".")),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )[: max(1, int(recent_queue_limit))]

    lineage_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"completed": 0, "metrics_present": 0, "zero_activity": 0, "hard_fail": 0, "tokens": set()}
    )
    operator_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"completed": 0, "metrics_present": 0, "zero_activity": 0, "hard_fail": 0}
    )

    completed_rows_total = 0
    for queue_path in queue_paths:
        queue_group = queue_path.parent.name
        try:
            rows = list(csv.DictReader(queue_path.open("r", encoding="utf-8", newline="")))
        except Exception:
            continue
        for row in rows:
            if str(row.get("status") or "").strip().lower() != "completed":
                continue
            results_dir = str(row.get("results_dir") or "").strip()
            run_id = Path(results_dir).name.strip() if results_dir else ""
            run_index_row = run_index_map.get(run_id)
            lineage_uid = _row_lineage_uid(row, queue_group)
            operator_id = _row_operator_id(row)
            token = _row_run_group_token(queue_group, run_index_row)
            metrics_present = int(parse_bool((run_index_row or {}).get("metrics_present")))
            zero_activity = 0
            if run_index_row:
                trades = parse_float(run_index_row.get("total_trades"), 0.0)
                pairs = parse_float(run_index_row.get("total_pairs_traded"), 0.0)
                zero_activity = int((not metrics_present) or trades <= 0.0 or pairs <= 0.0)
            else:
                zero_activity = 1
            hard_fail = int(bool(canonical_hard_fail_reason(run_index_row)))

            lineage_entry = lineage_stats[lineage_uid]
            lineage_entry["completed"] += 1
            lineage_entry["metrics_present"] += metrics_present
            lineage_entry["zero_activity"] += zero_activity
            lineage_entry["hard_fail"] += hard_fail
            lineage_entry["tokens"].add(token)

            operator_entry = operator_stats[operator_id]
            operator_entry["completed"] += 1
            operator_entry["metrics_present"] += metrics_present
            operator_entry["zero_activity"] += zero_activity
            operator_entry["hard_fail"] += hard_fail

            completed_rows_total += 1

    lineages: list[dict[str, Any]] = []
    for lineage_uid, stats in lineage_stats.items():
        status, yield_score = _classify_yield(
            completed=int(stats["completed"]),
            metrics_present=int(stats["metrics_present"]),
            zero_activity=int(stats["zero_activity"]),
            hard_fail=int(stats["hard_fail"]),
        )
        lineages.append(
            {
                "lineage_uid": lineage_uid,
                "status": status,
                "yield_score": yield_score,
                "completed": int(stats["completed"]),
                "metrics_present": int(stats["metrics_present"]),
                "zero_activity": int(stats["zero_activity"]),
                "hard_fail": int(stats["hard_fail"]),
                "tokens": sorted(str(token) for token in stats["tokens"] if str(token).strip()),
            }
        )
    lineages.sort(key=lambda item: (float(item["yield_score"]), int(item["completed"])), reverse=True)

    operators: list[dict[str, Any]] = []
    for operator_id, stats in operator_stats.items():
        status, yield_score = _classify_yield(
            completed=int(stats["completed"]),
            metrics_present=int(stats["metrics_present"]),
            zero_activity=int(stats["zero_activity"]),
            hard_fail=int(stats["hard_fail"]),
        )
        operators.append(
            {
                "operator_id": operator_id,
                "status": status,
                "yield_score": yield_score,
                "completed": int(stats["completed"]),
                "metrics_present": int(stats["metrics_present"]),
                "zero_activity": int(stats["zero_activity"]),
                "hard_fail": int(stats["hard_fail"]),
            }
        )
    operators.sort(key=lambda item: (float(item["yield_score"]), int(item["completed"])), reverse=True)

    winner_tokens: list[str] = []
    for queue_rel, entry in queues_state.items():
        if not isinstance(entry, Mapping):
            continue
        verdict = str(entry.get("promotion_verdict") or "").strip().upper()
        strict_pass = int(parse_float(entry.get("strict_pass_count"), 0.0))
        if verdict not in {"PROMOTE_PENDING_CONFIRM", "PROMOTE_DEFER_CONFIRM", "PROMOTE_ELIGIBLE"} and strict_pass <= 0:
            continue
        token = str(entry.get("top_run_group") or "").strip()
        if not token:
            token = Path(str(queue_rel)).parent.name
        if token and token not in winner_tokens:
            winner_tokens.append(token)
        if len(winner_tokens) >= 6:
            break

    preferred_contains: list[str] = list(winner_tokens)
    if len(preferred_contains) < 6:
        for entry in lineages:
            if str(entry.get("status")) != "boost":
                continue
            for token in list(entry.get("tokens") or []):
                if token not in preferred_contains:
                    preferred_contains.append(token)
                if len(preferred_contains) >= 6:
                    break
            if len(preferred_contains) >= 6:
                break

    cooldown_contains: list[str] = []
    for entry in lineages:
        if str(entry.get("status")) != "cooldown":
            continue
        for token in list(entry.get("tokens") or []):
            if token not in cooldown_contains:
                cooldown_contains.append(token)
            if len(cooldown_contains) >= 12:
                break
        if len(cooldown_contains) >= 12:
            break

    preferred_operator_ids = [str(entry["operator_id"]) for entry in operators if str(entry.get("status")) == "boost"][:6]
    cooldown_operator_ids = [str(entry["operator_id"]) for entry in operators if str(entry.get("status")) == "cooldown"][:6]

    return {
        "ts": utc_now_iso(),
        "schema_version": "v1",
        "active": True,
        "sample_size": {
            "queue_files": len(queue_paths),
            "completed_rows": completed_rows_total,
            "lineages": len(lineages),
            "operators": len(operators),
        },
        "preferred_contains": preferred_contains,
        "cooldown_contains": cooldown_contains,
        "preferred_operator_ids": preferred_operator_ids,
        "cooldown_operator_ids": cooldown_operator_ids,
        "winner_proximate": {
            "enabled": bool(preferred_contains),
            "contains": preferred_contains,
            "reason": "strict_pass_or_high_yield_lineage",
        },
        "lane_weights": {
            "winner_proximate": 40,
            "broad_search": 45,
            "confirm_replay": 15,
        },
        "policy_overrides": {
            "num_variants_cap": 64,
            "policy_scale": "micro" if cooldown_operator_ids else "auto",
        },
        "lineages": lineages[:48],
        "operators": operators[:24],
        "run_index_path": safe_rel(run_index_path, root),
        "fullspan_state_path": safe_rel(fullspan_state_path, root),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fail-safe yield governor state for autonomous queue seeding.")
    parser.add_argument("--root", default="", help="App root (`coint4/`).")
    parser.add_argument("--aggregate-dir", default="artifacts/wfa/aggregate")
    parser.add_argument("--run-index", default="artifacts/wfa/aggregate/rollup/run_index.csv")
    parser.add_argument("--fullspan-state", default="artifacts/wfa/aggregate/.autonomous/fullspan_decision_state.json")
    parser.add_argument("--output", default="artifacts/wfa/aggregate/.autonomous/yield_governor_state.json")
    parser.add_argument("--recent-queue-limit", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve() if str(args.root or "").strip() else Path(__file__).resolve().parents[2]
    aggregate_dir = Path(args.aggregate_dir) if Path(args.aggregate_dir).is_absolute() else root / str(args.aggregate_dir)
    run_index_path = Path(args.run_index) if Path(args.run_index).is_absolute() else root / str(args.run_index)
    fullspan_state_path = Path(args.fullspan_state) if Path(args.fullspan_state).is_absolute() else root / str(args.fullspan_state)
    output_path = Path(args.output) if Path(args.output).is_absolute() else root / str(args.output)

    payload = build_yield_governor_state(
        root=root,
        aggregate_dir=aggregate_dir,
        run_index_path=run_index_path,
        fullspan_state_path=fullspan_state_path,
        recent_queue_limit=int(args.recent_queue_limit),
    )
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        dump_json(output_path, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
