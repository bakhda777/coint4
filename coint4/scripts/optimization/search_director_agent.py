#!/usr/bin/env python3
"""Reason-aware search director for autonomous queue seeding.

Reads strict fullspan reject reasons and emits a directive consumed by
autonomous_queue_seeder.py to bias next batch generation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def parse_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    try:
        return float(value)
    except Exception:
        return default


def parse_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def canonical_reason(entry: dict[str, Any]) -> str:
    strict_reason = str(entry.get("strict_gate_reason") or "").strip().upper()
    reject_reason = str(entry.get("rejection_reason") or "").strip().upper()
    merged = strict_reason or reject_reason
    if "DD_FAIL" in merged:
        return "DD_FAIL"
    if "STEP_FAIL" in merged:
        return "STEP_FAIL"
    if "TRADES_FAIL" in merged:
        return "TRADES_FAIL"
    if "PAIRS_FAIL" in merged:
        return "PAIRS_FAIL"
    if "ECONOMIC_FAIL" in merged:
        return "ECONOMIC_FAIL"
    if "METRICS_MISSING" in merged:
        return "METRICS_MISSING"
    return merged or "UNKNOWN"


def build_directive(queues: dict[str, Any]) -> dict[str, Any]:
    reasons = Counter()
    run_group_hints: list[str] = []

    for queue, entry in queues.items():
        if not isinstance(entry, dict):
            continue
        verdict = str(entry.get("promotion_verdict") or "").strip().upper()
        if verdict != "REJECT":
            continue
        reason = canonical_reason(entry)
        reasons[reason] += 1
        top_group = str(entry.get("top_run_group") or "").strip()
        if top_group:
            run_group_hints.append(top_group)
        else:
            queue_group = Path(str(queue)).parent.name
            if queue_group:
                run_group_hints.append(queue_group)

    dominant_reason = "UNKNOWN"
    if reasons:
        dominant_reason = reasons.most_common(1)[0][0]

    contains = []
    seen = set()
    for token in run_group_hints:
        if token and token not in seen:
            seen.add(token)
            contains.append(token)
        if len(contains) >= 3:
            break

    directive: dict[str, Any] = {
        "version": 1,
        "mode": "neutral",
        "dominant_reason": dominant_reason,
        "contains": contains,
        "policy_scale": "auto",
        "num_variants": 24,
        "max_changed_keys": 4,
        "dedupe_distance": 0.03,
        "extra_cli_args": {},
        "impossibility_pruner": {
            "enabled": False,
            "reason": "",
            "max_changed_keys_cap": 4,
            "dedupe_distance_floor": 0.03,
            "num_variants_cap": 24,
            "policy_scale": "auto",
        },
    }

    if dominant_reason == "DD_FAIL":
        directive.update(
            {
                "mode": "dd_focus",
                "policy_scale": "micro",
                "max_changed_keys": 2,
                "dedupe_distance": 0.05,
                "extra_cli_args": {
                    "--max-dd-pct": "0.12",
                    "--max-tail-pair-share": "0.40",
                    "--max-tail-period-share": "0.55",
                },
                "impossibility_pruner": {
                    "enabled": True,
                    "reason": "DD_FAIL",
                    "max_changed_keys_cap": 2,
                    "dedupe_distance_floor": 0.05,
                    "num_variants_cap": 24,
                    "policy_scale": "micro",
                },
            }
        )
    elif dominant_reason in {"TRADES_FAIL", "PAIRS_FAIL"}:
        directive.update(
            {
                "mode": "breadth_focus",
                "policy_scale": "macro",
                "max_changed_keys": 5,
                "num_variants": 32,
                "dedupe_distance": 0.04,
                "extra_cli_args": {
                    "--max-dd-pct": "0.20",
                },
                "impossibility_pruner": {
                    "enabled": True,
                    "reason": dominant_reason,
                    "max_changed_keys_cap": 5,
                    "dedupe_distance_floor": 0.04,
                    "num_variants_cap": 32,
                    "policy_scale": "macro",
                },
            }
        )
    elif dominant_reason in {"STEP_FAIL", "ECONOMIC_FAIL"}:
        directive.update(
            {
                "mode": "tail_economic_focus",
                "policy_scale": "micro",
                "max_changed_keys": 3,
                "dedupe_distance": 0.05,
                "extra_cli_args": {
                    "--max-dd-pct": "0.18",
                    "--max-tail-pair-share": "0.35",
                    "--max-tail-period-share": "0.50",
                },
                "impossibility_pruner": {
                    "enabled": True,
                    "reason": dominant_reason,
                    "max_changed_keys_cap": 3,
                    "dedupe_distance_floor": 0.05,
                    "num_variants_cap": 24,
                    "policy_scale": "micro",
                },
            }
        )
    elif dominant_reason == "METRICS_MISSING":
        directive.update(
            {
                "mode": "stability_focus",
                "policy_scale": "micro",
                "max_changed_keys": 2,
                "num_variants": 20,
                "impossibility_pruner": {
                    "enabled": True,
                    "reason": "METRICS_MISSING",
                    "max_changed_keys_cap": 2,
                    "dedupe_distance_floor": 0.05,
                    "num_variants_cap": 20,
                    "policy_scale": "micro",
                },
            }
        )

    return directive


def _neutral_hard_fail_risk_policy() -> dict[str, Any]:
    return {
        "enabled": False,
        "reject_threshold": 0.75,
        "refine_threshold": 0.45,
    }


def _neutral_repair_mode() -> dict[str, Any]:
    return {
        "enabled": False,
        "validation_neighbor": 0,
        "max_neighbor_attempts": 0,
    }


def _sanitize_lineage_priority(raw: Any, limit: int = 12) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen = set()
    for token in raw:
        value = str(token or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
        if len(out) >= limit:
            break
    return out


def build_gate_surrogate_overlay(gate_state: Any) -> dict[str, Any]:
    overlay: dict[str, Any] = {
        "hard_fail_risk_policy": _neutral_hard_fail_risk_policy(),
        "lineage_priority": [],
        "repair_mode": _neutral_repair_mode(),
        "gate_surrogate_mode": "neutral",
        "gate_surrogate_ts": "",
    }
    if not isinstance(gate_state, dict):
        return overlay

    raw_policy = gate_state.get("hard_fail_risk_policy", {})
    policy = raw_policy if isinstance(raw_policy, dict) else {}
    enabled = bool(policy.get("enabled"))
    reject_threshold = max(0.0, min(1.0, parse_float(policy.get("reject_threshold"), 0.75)))
    refine_threshold = max(0.0, min(1.0, parse_float(policy.get("refine_threshold"), 0.45)))
    if reject_threshold < refine_threshold:
        reject_threshold, refine_threshold = refine_threshold, reject_threshold

    overlay["hard_fail_risk_policy"] = {
        "enabled": enabled,
        "reject_threshold": round(reject_threshold, 4),
        "refine_threshold": round(refine_threshold, 4),
    }

    overlay["lineage_priority"] = _sanitize_lineage_priority(gate_state.get("lineage_priority", []), limit=12)

    raw_repair = gate_state.get("repair_mode", {})
    repair = raw_repair if isinstance(raw_repair, dict) else {}
    repair_enabled = bool(enabled and repair.get("enabled"))
    overlay["repair_mode"] = {
        "enabled": repair_enabled,
        "validation_neighbor": max(0, parse_int(repair.get("validation_neighbor"), 0)) if repair_enabled else 0,
        "max_neighbor_attempts": max(0, parse_int(repair.get("max_neighbor_attempts"), 0)) if repair_enabled else 0,
    }

    summary = gate_state.get("summary", {})
    summary_mode = ""
    if isinstance(summary, dict):
        summary_mode = str(summary.get("mode") or "").strip().lower()
    overlay["gate_surrogate_mode"] = summary_mode or ("active" if enabled else "neutral")
    overlay["gate_surrogate_ts"] = str(gate_state.get("ts") or "")
    return overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reason-aware search directive from fullspan decision state.")
    parser.add_argument("--root", default="", help="App root (`coint4/`).")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve() if str(args.root or "").strip() else Path(__file__).resolve().parents[2]
    state_dir = root / "artifacts" / "wfa" / "aggregate" / ".autonomous"
    fullspan_state_path = state_dir / "fullspan_decision_state.json"
    gate_state_path = state_dir / "gate_surrogate_state.json"
    directive_path = state_dir / "search_director_directive.json"
    log_path = state_dir / "search_director.log"

    fullspan_state = load_json(fullspan_state_path, {})
    queues = fullspan_state.get("queues", {}) if isinstance(fullspan_state, dict) else {}
    if not isinstance(queues, dict):
        queues = {}
    gate_state = load_json(gate_state_path, {})

    directive = build_directive(queues)
    directive.update(build_gate_surrogate_overlay(gate_state))
    directive["ts"] = utc_now_iso()
    directive["source"] = "search_director_agent"
    directive["queue_count"] = len(queues)

    if not args.dry_run:
        dump_json(directive_path, directive)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{utc_now_iso()} | mode={directive.get('mode')} dominant_reason={directive.get('dominant_reason')} "
                f"contains={directive.get('contains')} risk_policy={int(bool((directive.get('hard_fail_risk_policy') or {}).get('enabled')))} "
                f"lineage={len(list(directive.get('lineage_priority') or []))} "
                f"repair={int(bool((directive.get('repair_mode') or {}).get('enabled')))}\n"
            )
    else:
        print(json.dumps(directive, ensure_ascii=False, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
