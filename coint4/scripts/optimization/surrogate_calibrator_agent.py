#!/usr/bin/env python3
"""Calibrate gate-surrogate reject/refine thresholds from observed outcomes."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MIN_SAMPLE_SIZE = 100
MAX_DELTA_PER_UPDATE = 0.05
MIN_APPLY_INTERVAL_SEC = 86400
DEFAULT_REJECT_THRESHOLD = 0.75
DEFAULT_REFINE_THRESHOLD = 0.45


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value or 0))
    except Exception:
        return default


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def parse_utc_epoch(raw: Any) -> int:
    text = str(raw or "").strip()
    if not text:
        return 0
    try:
        dt = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return 0
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


def percentile(values: list[float], q: float, default: float) -> float:
    if not values:
        return default
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * q))))
    return float(ordered[idx])


def canonical_outcome(entry: dict[str, Any], queue_notes: list[dict[str, Any]], run_index_group: dict[str, Any]) -> tuple[str, str]:
    verdict = str(entry.get("promotion_verdict") or "").strip().upper()
    rejection_reason = str(entry.get("rejection_reason") or "").strip().upper()
    strict_reason = str(entry.get("strict_gate_reason") or "").strip().upper()
    contract_reason = str(entry.get("contract_reason") or "").strip().upper()
    cutover = str(entry.get("cutover_permission") or "").strip().upper()
    strict_pass = parse_int(entry.get("strict_pass_count"), 0)
    confirm_count = parse_int(entry.get("confirm_count"), 0)

    if verdict in {"PROMOTE_ELIGIBLE", "PROMOTE_PENDING_CONFIRM", "PROMOTE_DEFER_CONFIRM"} or strict_pass > 0 or confirm_count > 0:
        return "positive", verdict or "STRICT_PASS"

    merged_reason = " ".join(item for item in [rejection_reason, strict_reason, contract_reason] if item)
    hard_fail_markers = ("DD_FAIL", "STEP_FAIL", "TRADES_FAIL", "PAIRS_FAIL", "ECONOMIC_FAIL", "NO_PROGRESS")
    if verdict == "REJECT" and any(marker in merged_reason for marker in hard_fail_markers):
        return "negative", merged_reason or verdict

    for note in reversed(queue_notes):
        action = str(note.get("action") or "").strip().upper()
        reason = str(note.get("reason") or "").strip().upper()
        if any(marker in f"{action} {reason}" for marker in hard_fail_markers):
            return "negative", reason or action

    metrics_missing = parse_int(run_index_group.get("metrics_missing"), 0)
    rows = parse_int(run_index_group.get("rows"), 0)
    completed = parse_int(run_index_group.get("completed"), 0)
    zero_activity = parse_int(run_index_group.get("completed_zero_activity"), 0)
    if rows > 0 and metrics_missing >= rows and completed <= 0:
        return "negative", "RUN_INDEX_METRICS_MISSING"
    if completed >= 4 and zero_activity >= completed and verdict == "REJECT":
        return "negative", "RUN_INDEX_ZERO_ACTIVITY"

    if cutover == "FAIL_CLOSED" and merged_reason and "METRICS_MISSING" not in merged_reason:
        return "negative", merged_reason
    return "unknown", merged_reason or verdict or "UNKNOWN"


def estimate_thresholds(positive_risks: list[float], negative_risks: list[float]) -> tuple[float, float, float, float]:
    if not positive_risks or not negative_risks:
        return DEFAULT_REJECT_THRESHOLD, DEFAULT_REFINE_THRESHOLD, 0.0, 0.0

    grid = [round(step / 100.0, 2) for step in range(35, 96, 5)]
    best_reject = DEFAULT_REJECT_THRESHOLD
    best_reject_score: tuple[float, float, float] | None = None
    for threshold in grid:
        false_reject_rate = sum(1 for value in positive_risks if value >= threshold) / float(len(positive_risks))
        negative_capture_rate = sum(1 for value in negative_risks if value >= threshold) / float(len(negative_risks))
        score = (false_reject_rate * 100.0) - negative_capture_rate
        tuple_score = (score, false_reject_rate, -threshold)
        if best_reject_score is None or tuple_score < best_reject_score:
            best_reject_score = tuple_score
            best_reject = threshold

    best_refine = DEFAULT_REFINE_THRESHOLD
    best_refine_score: tuple[float, float, float] | None = None
    for threshold in grid:
        if threshold >= best_reject:
            continue
        pos_band = sum(1 for value in positive_risks if threshold <= value < best_reject) / float(len(positive_risks))
        neg_band = sum(1 for value in negative_risks if threshold <= value < best_reject) / float(len(negative_risks))
        score = (pos_band * 20.0) - neg_band
        tuple_score = (score, pos_band, -threshold)
        if best_refine_score is None or tuple_score < best_refine_score:
            best_refine_score = tuple_score
            best_refine = threshold

    false_reject_rate = sum(1 for value in positive_risks if value >= best_reject) / float(len(positive_risks))
    refine_candidates = [value for value in positive_risks + negative_risks if best_refine <= value < best_reject]
    if refine_candidates:
        negative_refine = sum(1 for value in negative_risks if best_refine <= value < best_reject)
        refine_yield = negative_refine / float(len(refine_candidates))
    else:
        refine_yield = 0.0
    return best_reject, best_refine, false_reject_rate, refine_yield


def clamp_threshold(previous: float, recommended: float) -> float:
    lower = previous - MAX_DELTA_PER_UPDATE
    upper = previous + MAX_DELTA_PER_UPDATE
    return max(0.0, min(1.0, max(lower, min(upper, recommended))))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate gate surrogate thresholds from fullspan outcomes.")
    parser.add_argument("--root", default="", help="App root (`coint4/`).")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root).resolve() if str(args.root or "").strip() else Path(__file__).resolve().parents[2]
    state_dir = root / "artifacts" / "wfa" / "aggregate" / ".autonomous"
    run_index_path = root / "artifacts" / "wfa" / "aggregate" / "rollup" / "run_index.csv"
    gate_state_path = state_dir / "gate_surrogate_state.json"
    fullspan_path = state_dir / "fullspan_decision_state.json"
    notes_path = state_dir / "decision_notes.jsonl"
    output_path = state_dir / "surrogate_calibration_state.json"
    lock_path = state_dir / "surrogate_calibrator.lock"
    log_path = state_dir / "surrogate_calibrator.log"

    state_dir.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return 0

        gate_state = load_json(gate_state_path, {})
        fullspan = load_json(fullspan_path, {})
        previous = load_json(output_path, {})
        if not isinstance(gate_state, dict):
            gate_state = {}
        if not isinstance(fullspan, dict):
            fullspan = {}
        if not isinstance(previous, dict):
            previous = {}

        queues = gate_state.get("queues", {})
        fullspan_queues = fullspan.get("queues", {}) if isinstance(fullspan.get("queues"), dict) else {}
        runtime = fullspan.get("runtime_metrics", {}) if isinstance(fullspan.get("runtime_metrics"), dict) else {}

        run_index_groups: dict[str, dict[str, Any]] = defaultdict(dict)
        if run_index_path.exists():
            try:
                with run_index_path.open("r", encoding="utf-8", newline="") as csv_handle:
                    reader = csv.DictReader(csv_handle)
                    counters: dict[str, Counter[str]] = defaultdict(Counter)
                    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"rows": 0, "metrics_missing": 0, "completed": 0, "completed_zero_activity": 0})
                    for row in reader:
                        if not isinstance(row, dict):
                            continue
                        run_group = str(row.get("run_group") or "").strip()
                        if not run_group:
                            continue
                        stats[run_group]["rows"] += 1
                        status = str(row.get("status") or "").strip().lower()
                        if status:
                            counters[run_group][status] += 1
                        if str(row.get("metrics_present") or "").strip().lower() not in {"1", "true", "yes", "on"}:
                            stats[run_group]["metrics_missing"] += 1
                        if status == "completed":
                            stats[run_group]["completed"] += 1
                            total_trades = parse_float(row.get("total_trades"), 0.0)
                            total_pairs = parse_float(row.get("total_pairs_traded"), 0.0)
                            total_pnl = parse_float(row.get("total_pnl"), 0.0)
                            if total_trades <= 0 and total_pairs <= 0 and abs(total_pnl) <= 1e-12:
                                stats[run_group]["completed_zero_activity"] += 1
                    for run_group, item in stats.items():
                        item["status"] = dict(counters.get(run_group, {}))
                        run_index_groups[run_group] = item
            except Exception:
                run_index_groups = defaultdict(dict)

        notes_by_queue: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if notes_path.exists():
            try:
                for line in notes_path.read_text(encoding="utf-8").splitlines():
                    text = line.strip()
                    if not text:
                        continue
                    parsed = json.loads(text)
                    if not isinstance(parsed, dict):
                        continue
                    queue_key = str(parsed.get("queue") or "").strip()
                    if queue_key:
                        notes_by_queue[queue_key].append(parsed)
            except Exception:
                notes_by_queue = defaultdict(list)

        observations: list[dict[str, Any]] = []
        if isinstance(queues, dict):
            for queue_key, item in queues.items():
                if not isinstance(item, dict):
                    continue
                risk_score = max(0.0, min(1.0, parse_float(item.get("risk_score"), 0.0)))
                run_group = Path(str(queue_key)).parent.name
                outcome, reason = canonical_outcome(
                    fullspan_queues.get(queue_key, {}) if isinstance(fullspan_queues, dict) else {},
                    notes_by_queue.get(str(queue_key), []),
                    run_index_groups.get(run_group, {}),
                )
                observations.append(
                    {
                        "queue": str(queue_key),
                        "run_group": run_group,
                        "risk_score": round(risk_score, 4),
                        "surrogate_decision": str(item.get("decision") or ""),
                        "outcome": outcome,
                        "outcome_reason": reason,
                    }
                )

        positive_risks = [float(item["risk_score"]) for item in observations if item["outcome"] == "positive"]
        negative_risks = [float(item["risk_score"]) for item in observations if item["outcome"] == "negative"]
        known_observations = [item for item in observations if item["outcome"] in {"positive", "negative"}]
        sample_size = len(known_observations)

        base_reject = parse_float(
            previous.get("applied_reject_threshold"),
            parse_float((gate_state.get("hard_fail_risk_policy") or {}).get("reject_threshold"), DEFAULT_REJECT_THRESHOLD),
        )
        base_refine = parse_float(
            previous.get("applied_refine_threshold"),
            parse_float((gate_state.get("hard_fail_risk_policy") or {}).get("refine_threshold"), DEFAULT_REFINE_THRESHOLD),
        )
        if base_reject < base_refine:
            base_reject, base_refine = base_refine, base_reject

        recommended_reject, recommended_refine, false_reject_rate, refine_yield = estimate_thresholds(
            positive_risks,
            negative_risks,
        )

        now_epoch = parse_utc_epoch(utc_now_iso())
        prev_apply_epoch = parse_utc_epoch(previous.get("last_applied_ts"))
        apply_allowed = prev_apply_epoch <= 0 or (now_epoch - prev_apply_epoch) >= MIN_APPLY_INTERVAL_SEC
        applied = bool(sample_size >= MIN_SAMPLE_SIZE and apply_allowed)
        applied_reason = "calibration_applied"
        if sample_size < MIN_SAMPLE_SIZE:
            applied_reason = "insufficient_sample"
        elif not apply_allowed:
            applied_reason = "apply_interval_guard"

        applied_reject = base_reject
        applied_refine = base_refine
        if applied:
            applied_reject = clamp_threshold(base_reject, recommended_reject)
            applied_refine = clamp_threshold(base_refine, recommended_refine)
            if applied_reject < applied_refine:
                applied_reject, applied_refine = applied_refine, applied_reject

        payload = {
            "version": 1,
            "ts": utc_now_iso(),
            "source": "surrogate_calibrator_agent",
            "enabled": bool(sample_size >= MIN_SAMPLE_SIZE),
            "applied": bool(applied),
            "reason": applied_reason,
            "sample_size": int(sample_size),
            "min_sample_size": MIN_SAMPLE_SIZE,
            "known_positive_count": int(len(positive_risks)),
            "known_negative_count": int(len(negative_risks)),
            "recommended_reject_threshold": round(recommended_reject, 4),
            "recommended_refine_threshold": round(recommended_refine, 4),
            "applied_reject_threshold": round(applied_reject, 4),
            "applied_refine_threshold": round(applied_refine, 4),
            "estimated_false_reject_rate": round(false_reject_rate, 6),
            "estimated_refine_yield": round(refine_yield, 6),
            "hysteresis": {
                "max_delta_per_update": MAX_DELTA_PER_UPDATE,
                "min_apply_interval_sec": MIN_APPLY_INTERVAL_SEC,
                "apply_allowed": bool(apply_allowed),
                "previous_reject_threshold": round(base_reject, 4),
                "previous_refine_threshold": round(base_refine, 4),
            },
            "runtime_snapshot": {
                "strict_fullspan_pass_count": parse_int(runtime.get("strict_fullspan_pass_count"), 0),
                "promotion_eligible_count": parse_int(runtime.get("promotion_eligible_count"), 0),
            },
            "observations_preview": known_observations[:20],
            "distribution": {
                "positive_p95": round(percentile(positive_risks, 0.95, 0.0), 4),
                "negative_p50": round(percentile(negative_risks, 0.50, 0.0), 4),
                "negative_p75": round(percentile(negative_risks, 0.75, 0.0), 4),
            },
        }

        if args.dry_run:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            if applied:
                payload["last_applied_ts"] = payload["ts"]
            else:
                payload["last_applied_ts"] = str(previous.get("last_applied_ts") or "")
            dump_json(output_path, payload)

        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(
                f"{payload['ts']} | sample={sample_size} applied={int(bool(applied))} "
                f"reject={payload['applied_reject_threshold']:.4f} refine={payload['applied_refine_threshold']:.4f} "
                f"reason={applied_reason} false_reject={payload['estimated_false_reject_rate']:.4f} "
                f"refine_yield={payload['estimated_refine_yield']:.4f}\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
