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
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _queue_status_contract import (
    PENDING_LIKE_STATUSES,
    load_fullspan_queue_state,
    load_orphan_queue_cooldowns,
    normalize_queue_status,
    queue_dispatch_block_reason,
    queue_rel_path,
    row_counts_dispatchable_pending,
    row_counts_executable,
)
from _search_quality_contract import (
    CANONICAL_ZERO_EVIDENCE_REASONS,
    CONTROLLED_BROAD_REARM_DELAY_SEC,
    CONTROLLED_RECOVERY_REASON,
    CONTROLLED_RECOVERY_VARIANTS_CAP,
    DETERMINISTIC_QUARANTINE_CODES,
    build_controlled_recovery_state,
    canonical_zero_evidence_reason,
    controlled_recovery_backlog_ready,
    has_positive_coverage_trade_evidence,
    micro_broad_search_caps,
    normalize_search_quality_state,
    summarize_recent_zero_evidence,
)


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


CONTROLLED_BROAD_NUM_VARIANTS = max(
    4,
    int(float(os.getenv("COINT4_CONTROLLED_BROAD_NUM_VARIANTS", "24"))),
)
CONTROLLED_BROAD_NUM_VARIANTS_FLOOR = max(
    1,
    min(
        CONTROLLED_BROAD_NUM_VARIANTS,
        int(float(os.getenv("COINT4_CONTROLLED_BROAD_NUM_VARIANTS_FLOOR", "16"))),
    ),
)
CONTROLLED_BROAD_MAX_CHANGED_KEYS = max(
    1,
    int(float(os.getenv("COINT4_CONTROLLED_BROAD_MAX_CHANGED_KEYS", "3"))),
)
CONTROLLED_BROAD_DEDUPE_DISTANCE = max(
    0.0,
    float(os.getenv("COINT4_CONTROLLED_BROAD_DEDUPE_DISTANCE", "0.04")),
)
SEEDER_SKIP_RETRY_SEC = max(
    60,
    int(float(os.getenv("COINT4_QUEUE_SEEDER_SKIP_RETRY_SEC", "300"))),
)


def _load_args() -> argparse.Namespace:
    root = _repo_root()
    defaults = {
        "base_config": os.getenv("AUTONOMOUS_QUEUE_SEEDER_BASE_CONFIG", "configs/prod_final_budget1000.yaml"),
        "controller_group": os.getenv(
            "AUTONOMOUS_QUEUE_SEEDER_CONTROLLER_GROUP", os.getenv("CONTROLLER_GROUP", "autonomous_queue_seeder")
        ),
        "run_group_prefix": os.getenv("AUTONOMOUS_QUEUE_SEEDER_RUN_GROUP_PREFIX", "autonomous_seed"),
        "pending_threshold": int(os.getenv("AUTONOMOUS_QUEUE_SEEDER_PENDING_THRESHOLD", "48")),
        "num_variants": int(os.getenv("AUTONOMOUS_QUEUE_SEEDER_NUM_VARIANTS", "32")),
        "num_variants_floor": int(os.getenv("AUTONOMOUS_QUEUE_SEEDER_NUM_VARIANTS_FLOOR", "24")),
        "max_changed_keys": int(os.getenv("AUTONOMOUS_QUEUE_SEEDER_MAX_CHANGED_KEYS", "4")),
        "dedupe_distance": float(os.getenv("AUTONOMOUS_QUEUE_SEEDER_DEDUPE_DISTANCE", "0.06")),
        "policy_scale": os.getenv("AUTONOMOUS_QUEUE_SEEDER_POLICY_SCALE", "micro"),
        "ir_mode": os.getenv("AUTONOMOUS_QUEUE_SEEDER_IR_MODE", "patch_ast"),
        "include_stress": _parse_bool(os.getenv("AUTONOMOUS_QUEUE_SEEDER_INCLUDE_STRESS", "true"), default=True),
        "llm_propose": _parse_bool(os.getenv("AUTONOMOUS_QUEUE_SEEDER_LLM_PROPOSE", "false"), default=False),
        "llm_model": os.getenv("AUTONOMOUS_QUEUE_SEEDER_LLM_MODEL", "gpt-5.2"),
        "llm_effort": os.getenv("AUTONOMOUS_QUEUE_SEEDER_LLM_EFFORT", "high"),
        "llm_timeout": int(os.getenv("AUTONOMOUS_QUEUE_SEEDER_LLM_TIMEOUT", "300")),
        "run_index": os.getenv("AUTONOMOUS_QUEUE_SEEDER_RUN_INDEX", "artifacts/wfa/aggregate/rollup/run_index.csv"),
        "contains": [token.strip() for token in os.getenv("AUTONOMOUS_QUEUE_SEEDER_CONTAINS", "").split(",") if token.strip()],
        "windows": [token.strip() for token in os.getenv("AUTONOMOUS_QUEUE_SEEDER_WINDOWS", "").split(";") if token.strip()],
        "aggregate_dir": os.getenv("AUTONOMOUS_QUEUE_SEEDER_AGGREGATE_DIR", "artifacts/wfa/aggregate"),
        "python_bin": os.getenv("AUTONOMOUS_QUEUE_SEEDER_PYTHON_BIN", str(root / ".venv" / "bin" / "python")),
        "directive_path": os.getenv(
            "AUTONOMOUS_QUEUE_SEEDER_DIRECTIVE_PATH",
            "artifacts/wfa/aggregate/.autonomous/search_director_directive.json",
        ),
        "blacklist_path": os.getenv(
            "AUTONOMOUS_QUEUE_SEEDER_BLACKLIST_PATH",
            "artifacts/wfa/aggregate/.autonomous/search_policy_blacklist.json",
        ),
        "gate_surrogate_state_path": os.getenv(
            "AUTONOMOUS_QUEUE_SEEDER_GATE_SURROGATE_STATE_PATH",
            "artifacts/wfa/aggregate/.autonomous/gate_surrogate_state.json",
        ),
        "yield_governor_state_path": os.getenv(
            "AUTONOMOUS_QUEUE_SEEDER_YIELD_GOVERNOR_STATE_PATH",
            "artifacts/wfa/aggregate/.autonomous/yield_governor_state.json",
        ),
        "ready_buffer_state_path": os.getenv(
            "AUTONOMOUS_QUEUE_SEEDER_READY_BUFFER_STATE_PATH",
            "artifacts/wfa/aggregate/.autonomous/ready_queue_buffer.json",
        ),
        "repair_mode": _parse_bool(os.getenv("AUTONOMOUS_QUEUE_SEEDER_REPAIR_MODE", "false"), default=False),
        "repair_max_neighbors": int(os.getenv("AUTONOMOUS_QUEUE_SEEDER_REPAIR_MAX_NEIGHBORS", "8")),
        "exclude_knobs": [
            token.strip()
            for token in os.getenv("AUTONOMOUS_QUEUE_SEEDER_EXCLUDE_KNOBS", "").split(",")
            if token.strip()
        ],
        "fallback_base_configs": [
            token.strip()
            for token in os.getenv("AUTONOMOUS_QUEUE_SEEDER_FALLBACK_BASE_CONFIGS", "").split(",")
            if token.strip()
        ],
    }

    parser = argparse.ArgumentParser(description="Autonomously seed run queue when backlog is low.")
    parser.add_argument("--base-config", default=defaults["base_config"], help="Base config for evolution planner")
    parser.add_argument("--controller-group", default=defaults["controller_group"], help="Controller group for evolution state/decisions")
    parser.add_argument("--run-group-prefix", default=defaults["run_group_prefix"], help="Prefix for generated run groups")
    parser.add_argument(
        "--pending-threshold",
        type=int,
        default=defaults["pending_threshold"],
        help="Trigger when executable pending (planned/queued/running with valid config) < threshold",
    )
    parser.add_argument("--num-variants", type=int, default=defaults["num_variants"], help="Variants to request from evolve_next_batch")
    parser.add_argument(
        "--num-variants-floor",
        type=int,
        default=defaults["num_variants_floor"],
        help="Floor for variant count after directive/blacklist/pruner caps.",
    )
    parser.add_argument("--max-changed-keys", type=int, default=defaults["max_changed_keys"], help="Max changed knobs per candidate")
    parser.add_argument("--dedupe-distance", type=float, default=defaults["dedupe_distance"], help="Minimum inter-candidate distance")
    parser.add_argument("--policy-scale", default=defaults["policy_scale"], choices=["auto", "micro", "macro"], help="Planner policy scale")
    parser.add_argument("--ir-mode", default=defaults["ir_mode"], choices=["knob", "patch_ast"], help="Evolution IR mode")
    parser.add_argument("--include-stress", dest="include_stress", action="store_true", help="Generate holdout+stress paired queue entries")
    parser.add_argument("--no-include-stress", dest="include_stress", action="store_false", help="Generate holdout-only queue entries")
    parser.set_defaults(include_stress=defaults["include_stress"])
    parser.add_argument("--llm-propose", action="store_true", default=defaults["llm_propose"], help="Enable LLM policy override")
    parser.add_argument("--llm-model", default=defaults["llm_model"], help="LLM model for planner")
    parser.add_argument("--llm-effort", default=defaults["llm_effort"], help="LLM effort for planner")
    parser.add_argument("--llm-timeout-sec", type=int, default=defaults["llm_timeout"], help="LLM timeout seconds")
    parser.add_argument("--contains", action="append", default=defaults["contains"], help="run_index contain token(s)")
    parser.add_argument("--window", action="append", default=defaults["windows"], help="walk-forward windows as YYYY-MM-DD,YYYY-MM-DD")
    parser.add_argument("--aggregate-dir", default=defaults["aggregate_dir"], help="Run queue root folder")
    parser.add_argument("--run-index", default=defaults["run_index"], help="run_index CSV path")
    parser.add_argument("--python-bin", default=defaults["python_bin"], help="Python binary for evolve_next_batch")
    parser.add_argument(
        "--directive-path",
        default=defaults["directive_path"],
        help="Optional reason-aware directive JSON (search_director output).",
    )
    parser.add_argument(
        "--blacklist-path",
        default=defaults["blacklist_path"],
        help="Optional deterministic-error blacklist JSON.",
    )
    parser.add_argument(
        "--gate-surrogate-state-path",
        default=defaults["gate_surrogate_state_path"],
        help="Optional gate surrogate state JSON with hard_fail_risk_policy/queue decisions.",
    )
    parser.add_argument(
        "--yield-governor-state-path",
        default=defaults["yield_governor_state_path"],
        help="Optional yield governor state JSON with preferred/cooldown search lanes.",
    )
    parser.add_argument(
        "--ready-buffer-state-path",
        default=defaults["ready_buffer_state_path"],
        help="Optional ready-buffer state JSON for depth-aware seed triggers.",
    )
    parser.add_argument(
        "--repair-mode",
        dest="repair_mode",
        action="store_true",
        help="Forward repair mode into evolve_next_batch planner.",
    )
    parser.add_argument(
        "--no-repair-mode",
        dest="repair_mode",
        action="store_false",
        help="Disable repair mode forwarding.",
    )
    parser.set_defaults(repair_mode=defaults["repair_mode"])
    parser.add_argument(
        "--repair-max-neighbors",
        type=int,
        default=defaults["repair_max_neighbors"],
        help="Forward repair neighborhood limit into evolve_next_batch planner.",
    )
    parser.add_argument(
        "--exclude-knob",
        action="append",
        default=defaults["exclude_knobs"],
        help="Exclude knob(s) for evolve_next_batch planner (repeatable).",
    )
    parser.add_argument(
        "--fallback-base-config",
        action="append",
        default=defaults["fallback_base_configs"],
        help="Fallback base config(s) when primary planner input fails",
    )
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
            normalized = {k: (v or "").strip() for k, v in row.items()}
            if any(str(value or "").strip() for value in normalized.values()):
                rows.append(normalized)
    return rows


def _write_queue_rows(queue_path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    if not fieldnames and queue_path.exists():
        try:
            with queue_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
            for key in header:
                key_text = str(key).strip()
                if key_text and key_text not in fieldnames:
                    fieldnames.append(key_text)
        except Exception:
            fieldnames = []
    if not fieldnames:
        fieldnames = ["config_path", "results_dir", "status"]
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with queue_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _legacy_header_only_queue_reason(queue_path: Path) -> str:
    if not queue_path.exists():
        return ""
    try:
        with queue_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            if not header:
                return ""
            for row in reader:
                if any(str(value or "").strip() for value in row):
                    return ""
    except Exception:
        return ""
    return "queue_pruned_empty_legacy"


def _load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _walk_forward_window(config_payload: dict[str, Any]) -> tuple[str, str]:
    walk_forward = config_payload.get("walk_forward", {}) if isinstance(config_payload, dict) else {}
    if not isinstance(walk_forward, dict):
        walk_forward = {}
    start = str(walk_forward.get("start_date") or "").strip()
    end = str(walk_forward.get("end_date") or "").strip()
    return start, end


def _month_sequence(start_date: str, end_date: str) -> list[tuple[int, int]]:
    try:
        start = datetime.strptime(str(start_date).strip(), "%Y-%m-%d")
        end = datetime.strptime(str(end_date).strip(), "%Y-%m-%d")
    except Exception:
        return []
    if end < start:
        return []
    months: list[tuple[int, int]] = []
    year = int(start.year)
    month = int(start.month)
    end_key = (int(end.year), int(end.month))
    while (year, month) <= end_key:
        months.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def _coverage_gate_for_config(*, config_path: Path, app_root: Path) -> dict[str, Any]:
    payload = _load_yaml_config(config_path)
    start_date, end_date = _walk_forward_window(payload)
    if not start_date or not end_date:
        return {"ok": True, "reason": "window_missing", "missing_months": []}
    data_root = app_root / "data_downloaded"
    missing_months: list[str] = []
    for year, month in _month_sequence(start_date, end_date):
        month_dir = data_root / f"year={year:04d}" / f"month={month:02d}"
        if not month_dir.exists():
            missing_months.append(f"{year:04d}-{month:02d}")
    return {
        "ok": len(missing_months) == 0,
        "reason": "ok" if len(missing_months) == 0 else "missing_data_coverage",
        "window": {"start_date": start_date, "end_date": end_date},
        "missing_months": missing_months,
    }


def _assess_window_data_coverage(data_root: Path, windows: list[str]) -> dict[str, Any]:
    requested_windows = [str(raw or "").strip() for raw in list(windows or []) if str(raw or "").strip()]
    if not requested_windows:
        return {
            "ok": True,
            "reason": "no_windows_requested",
            "windows": [],
            "covered_window_count": 0,
            "uncovered_window_count": 0,
            "missing_months": [],
        }

    details: list[dict[str, Any]] = []
    missing_months: set[str] = set()
    covered_window_count = 0
    uncovered_window_count = 0
    invalid_window_count = 0
    for raw in requested_windows:
        start_date, sep, end_date = raw.partition(",")
        start_date = start_date.strip()
        end_date = end_date.strip()
        if not sep or not start_date or not end_date:
            invalid_window_count += 1
            details.append(
                {
                    "window": raw,
                    "ok": False,
                    "reason": "invalid_window",
                    "start_date": start_date,
                    "end_date": end_date,
                    "missing_months": [],
                }
            )
            continue
        window_missing: list[str] = []
        for year, month in _month_sequence(start_date, end_date):
            month_token = f"{year:04d}-{month:02d}"
            month_dir = data_root / f"year={year:04d}" / f"month={month:02d}"
            if not month_dir.exists():
                missing_months.add(month_token)
                window_missing.append(month_token)
        covered = len(window_missing) == 0
        if covered:
            covered_window_count += 1
        else:
            uncovered_window_count += 1
        details.append(
            {
                "window": raw,
                "ok": covered,
                "reason": "ok" if covered else "missing_data_coverage",
                "start_date": start_date,
                "end_date": end_date,
                "missing_months": window_missing,
            }
        )

    ok = invalid_window_count == 0 and uncovered_window_count == 0
    reason = "ok"
    if invalid_window_count > 0:
        reason = "invalid_window"
    elif uncovered_window_count > 0:
        reason = "missing_data_coverage"
    return {
        "ok": ok,
        "reason": reason,
        "windows": details,
        "covered_window_count": int(covered_window_count),
        "uncovered_window_count": int(uncovered_window_count + invalid_window_count),
        "missing_months": sorted(missing_months),
    }


def _decision_lineage_uid(decision_payload: dict[str, Any]) -> str:
    if not isinstance(decision_payload, dict):
        return ""
    parent_resolution = decision_payload.get("parent_resolution", {})
    if isinstance(parent_resolution, dict):
        lineage = str(parent_resolution.get("lineage_uid") or "").strip()
        if lineage:
            return lineage
    parent_diversification = decision_payload.get("parent_diversification", {})
    if not isinstance(parent_diversification, dict):
        parent_diversification = {}
    pool = list(parent_diversification.get("pool") or [])
    try:
        primary_idx = int(parent_diversification.get("primary_parent_index") or 0)
    except Exception:
        primary_idx = 0
    if 0 <= primary_idx < len(pool):
        primary = pool[primary_idx]
        if isinstance(primary, dict):
            return str(primary.get("lineage_uid") or "").strip()
    primary_parent = decision_payload.get("primary_parent", {})
    if isinstance(primary_parent, dict):
        return str(primary_parent.get("lineage_uid") or "").strip()
    return ""


def _config_identity_signature(
    *,
    row: dict[str, str],
    config_payload: dict[str, Any],
    config_path: Path,
    fallback_lineage_uid: str = "",
) -> str:
    meta_raw = str(row.get("metadata_json") or "").strip()
    meta: dict[str, Any] = {}
    if meta_raw:
        try:
            parsed = json.loads(meta_raw)
        except Exception:
            parsed = {}
        if isinstance(parsed, dict):
            meta = dict(parsed)
    evolution = config_payload.get("evolution", {}) if isinstance(config_payload, dict) else {}
    if not isinstance(evolution, dict):
        evolution = {}
    metadata = config_payload.get("metadata", {}) if isinstance(config_payload, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    start_date, end_date = _walk_forward_window(config_payload)
    evo_hash = (
        str(meta.get("evo_hash") or metadata.get("evo_hash") or evolution.get("evo_hash") or "")
        .strip()
    )
    lineage = (
        str(meta.get("lineage_uid") or metadata.get("lineage_uid") or evolution.get("lineage_uid") or "")
        .strip()
    )
    if not lineage:
        lineage = str(fallback_lineage_uid or "").strip()
    run_name = str(row.get("run_name") or "").strip().lower()
    config_text = str(config_path).lower()
    role = "default"
    if "stress" in run_name or "stress" in config_text:
        role = "stress"
    elif "holdout" in run_name or "holdout" in config_text:
        role = "holdout"
    return "|".join([evo_hash, start_date, end_date, lineage, role])


def _prune_seed_queue(
    *,
    queue_path: Path,
    app_root: Path,
    lineage_uid: str = "",
    decision_payload: dict[str, Any] | None = None,
    run_index_groups: dict[str, list[dict[str, str]]] | None = None,
    quarantine_by_group: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    rows = _load_queue_rows(queue_path)
    if not rows:
        blocked_rows_written = 0
        if queue_path.exists():
            _write_queue_rows(
                queue_path,
                [
                    {
                        "config_path": "",
                        "results_dir": "",
                        "status": "blocked",
                        "note": "queue_pruned_empty",
                    }
                ],
            )
            blocked_rows_written = 1
        return {
            "rows_before": 0,
            "rows_after": 0,
            "coverage_rejected": 0,
            "dedupe_rejected": 0,
            "missing_months": [],
            "blocked_rows_written": blocked_rows_written,
            "block_reason": "queue_pruned_empty",
        }

    filtered_rows: list[dict[str, str]] = []
    seen_signatures: set[str] = set()
    coverage_rejected = 0
    dedupe_rejected = 0
    missing_months: set[str] = set()

    for row in rows:
        config_rel = str(row.get("config_path") or "").strip()
        config_path = _resolve_under_root(config_rel, app_root)
        config_payload = _load_yaml_config(config_path)
        coverage = _coverage_gate_for_config(config_path=config_path, app_root=app_root)
        if not bool(coverage.get("ok")):
            coverage_rejected += 1
            for item in list(coverage.get("missing_months") or []):
                token = str(item).strip()
                if token:
                    missing_months.add(token)
            continue
        signature = _config_identity_signature(
            row=row,
            config_payload=config_payload,
            config_path=config_path,
            fallback_lineage_uid=lineage_uid,
        )
        if signature and signature in seen_signatures:
            dedupe_rejected += 1
            continue
        if signature:
            seen_signatures.add(signature)
        filtered_rows.append(row)

    blocked_rows_written = 0
    block_reason = ""
    if not filtered_rows:
        if coverage_rejected > 0:
            block_reason = "coverage_fail_closed"
        elif dedupe_rejected > 0:
            block_reason = "dedupe_fail_closed"
        else:
            block_reason = "queue_pruned_empty"
    lineage_feasibility = _assess_lineage_preseed_feasibility(
        decision_payload=decision_payload,
        run_index_groups=run_index_groups,
        quarantine_by_group=quarantine_by_group,
    )
    if filtered_rows and not bool(lineage_feasibility.get("ok")):
        block_reason = str(lineage_feasibility.get("reason") or "NO_RECENT_POSITIVE_COVERAGE_TRADE_EVIDENCE").strip()
        filtered_rows = []
    if filtered_rows:
        _write_queue_rows(queue_path, filtered_rows)
    else:
        blocked_rows: list[dict[str, str]] = []
        for row in rows:
            blocked = dict(row)
            blocked["status"] = "blocked"
            blocked["note"] = block_reason or "queue_pruned_empty"
            blocked_rows.append(blocked)
        blocked_rows_written = len(blocked_rows)
        _write_queue_rows(queue_path, blocked_rows)
    return {
        "rows_before": len(rows),
        "rows_after": len(filtered_rows),
        "coverage_rejected": coverage_rejected,
        "dedupe_rejected": dedupe_rejected,
        "missing_months": sorted(missing_months),
        "blocked_rows_written": int(blocked_rows_written),
        "block_reason": block_reason,
        "lineage_hints": list(lineage_feasibility.get("lineage_hints", []) or []),
        "matched_run_groups": list(lineage_feasibility.get("matched_run_groups", []) or []),
        "lineage_recent_summary": dict(lineage_feasibility.get("recent_summary") or {}),
        "quarantine_counts": dict(lineage_feasibility.get("quarantine_counts") or {}),
    }


def _recent_zero_yield_signal(rank_results_dir: Path, *, limit: int = 24) -> dict[str, Any]:
    files = sorted(
        rank_results_dir.glob("autonomous_seed*_latest.json"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )[: max(1, int(limit))]
    analyzed = 0
    zeroish = 0
    strict_binding = 0
    for path in files:
        payload = _load_directive(path)
        if not payload:
            continue
        analyzed += 1
        details = str(payload.get("details") or "").strip()
        zero_reason = canonical_zero_evidence_reason(payload)
        if zero_reason in CANONICAL_ZERO_EVIDENCE_REASONS:
            zeroish += 1
        if details.startswith("RANK_OK_FALLBACK_STRICT_BINDING") or any(
            token in details for token in ("min_windows", "min_trades", "min_pairs", "coverage_below")
        ):
            strict_binding += 1
    zeroish_ratio = (float(zeroish) / float(analyzed)) if analyzed > 0 else 0.0
    active = bool(analyzed >= 4 and (zeroish_ratio >= 0.75 or strict_binding >= 3))
    return {
        "active": active,
        "analyzed": analyzed,
        "zeroish": zeroish,
        "strict_binding": strict_binding,
        "zeroish_ratio": zeroish_ratio,
    }


def _load_rank_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _dominant_zero_reason(rows: list[dict[str, str]], *, limit: int = 12) -> str:
    summary = summarize_recent_zero_evidence(rows, limit=limit)
    return str(summary.get("dominant_zero_reason") or "").strip()


def _load_run_index_groups(run_index_path: Path, *, run_group_prefix: str = "") -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    if not run_index_path.exists():
        return groups
    try:
        with run_index_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                run_group = str(row.get("run_group") or "").strip()
                if not run_group:
                    continue
                if run_group_prefix and not run_group.startswith(run_group_prefix):
                    continue
                groups.setdefault(run_group, []).append({k: (v or "").strip() for k, v in row.items()})
    except Exception:
        return {}
    return groups


def _decision_lineage_hints(decision_payload: dict[str, Any]) -> list[str]:
    hints: list[str] = []

    def _push(value: Any) -> None:
        token = str(value or "").strip()
        if token and token not in hints:
            hints.append(token)

    if not isinstance(decision_payload, dict):
        return hints

    for key in ("contains", "lineage_priority", "winner_proximate_tokens"):
        for item in list(decision_payload.get(key, []) or []):
            _push(item)

    primary_parent = decision_payload.get("primary_parent", {})
    if isinstance(primary_parent, dict):
        _push(primary_parent.get("run_group"))
        _push(primary_parent.get("lineage_uid"))

    parent_resolution = decision_payload.get("parent_resolution", {})
    if isinstance(parent_resolution, dict):
        for key in ("winner_proximate_tokens", "preferred_any_contains", "contains", "lineage_priority"):
            for item in list(parent_resolution.get(key, []) or []):
                _push(item)
        _push(parent_resolution.get("lineage_uid"))

    parent_diversification = decision_payload.get("parent_diversification", {})
    if isinstance(parent_diversification, dict):
        for parent in list(parent_diversification.get("pool", []) or []):
            if not isinstance(parent, dict):
                continue
            _push(parent.get("run_group"))
            _push(parent.get("lineage_uid"))

    return hints


def _select_covered_recovery_windows(
    *,
    candidate_groups: Sequence[str],
    run_index_groups: dict[str, list[dict[str, str]]] | None,
    app_root: Path,
    limit: int = 1,
) -> list[str]:
    windows: dict[str, tuple[tuple[str, float, float, float], str]] = {}
    data_root = app_root / "data_downloaded"
    seen_groups: set[str] = set()
    ordered_groups: list[str] = []
    for raw in list(candidate_groups or []):
        token = str(raw or "").strip()
        if not token or token in seen_groups:
            continue
        seen_groups.add(token)
        ordered_groups.append(token)

    for group in ordered_groups:
        rows = list((run_index_groups or {}).get(group) or [])
        for row in rows:
            if not has_positive_coverage_trade_evidence(row):
                continue
            config_rel = str(row.get("config_path") or "").strip()
            if not config_rel:
                continue
            try:
                config_path = _resolve_under_root(config_rel, app_root)
                payload = _load_yaml_config(config_path)
            except Exception:
                continue
            start_date, end_date = _walk_forward_window(payload)
            if not start_date or not end_date:
                continue
            window = f"{start_date},{end_date}"
            coverage = _assess_window_data_coverage(data_root, [window])
            if not bool(coverage.get("ok")):
                continue
            sort_key = (
                end_date,
                _as_float(row.get("coverage_ratio"), default=0.0, min_value=0.0) or 0.0,
                _as_float(row.get("total_trades"), default=0.0, min_value=0.0) or 0.0,
                _as_float(row.get("observed_test_days"), default=0.0, min_value=0.0) or 0.0,
            )
            existing = windows.get(window)
            if existing is None or sort_key > existing[0]:
                windows[window] = (sort_key, group)

    selected = sorted(windows.items(), key=lambda item: item[1][0], reverse=True)
    return [window for window, _meta in selected[: max(1, int(limit))]]


def _resolve_effective_seed_windows(
    *,
    explicit_windows: Sequence[str],
    candidate_groups: Sequence[str],
    run_index_groups: dict[str, list[dict[str, str]]] | None,
    app_root: Path,
    limit: int = 1,
) -> list[str]:
    windows = [str(raw or "").strip() for raw in list(explicit_windows or []) if str(raw or "").strip()]
    if windows:
        return windows
    return _select_covered_recovery_windows(
        candidate_groups=candidate_groups,
        run_index_groups=run_index_groups,
        app_root=app_root,
        limit=limit,
    )


def _load_recent_quarantine_by_group(path: Path, *, limit: int = 200) -> dict[str, dict[str, int]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    if not isinstance(entries, list):
        return {}
    grouped: dict[str, dict[str, int]] = {}
    for entry in entries[-max(1, int(limit)) :]:
        if not isinstance(entry, dict):
            continue
        queue = str(entry.get("queue") or "").strip()
        code = str(entry.get("code") or "").strip().upper()
        if not queue or not code:
            continue
        group = Path(queue).parent.name
        counters = grouped.setdefault(group, {})
        counters[code] = int(counters.get(code, 0) or 0) + 1
    return grouped


def _assess_lineage_preseed_feasibility(
    *,
    decision_payload: dict[str, Any] | None,
    run_index_groups: dict[str, list[dict[str, str]]] | None,
    quarantine_by_group: dict[str, dict[str, int]] | None,
) -> dict[str, Any]:
    hints = _decision_lineage_hints(decision_payload or {})
    candidate_groups = set(run_index_groups or {})
    candidate_groups.update(set(quarantine_by_group or {}))
    matched_groups = [hint for hint in hints if hint in candidate_groups]
    recent_rows: list[dict[str, str]] = []
    for group in matched_groups:
        recent_rows.extend(list((run_index_groups or {}).get(group) or []))

    zero_summary = summarize_recent_zero_evidence(recent_rows, limit=12)
    quarantine_counts: dict[str, int] = {}
    for group in matched_groups:
        for code, count in dict((quarantine_by_group or {}).get(group) or {}).items():
            quarantine_counts[code] = int(quarantine_counts.get(code, 0) or 0) + int(count or 0)

    quarantine_reason = ""
    for code in DETERMINISTIC_QUARANTINE_CODES:
        if int(quarantine_counts.get(code, 0) or 0) > 0:
            quarantine_reason = code
            break

    if quarantine_reason:
        return {
            "ok": False,
            "reason": quarantine_reason,
            "lineage_hints": hints,
            "matched_run_groups": matched_groups,
            "recent_summary": zero_summary,
            "quarantine_counts": quarantine_counts,
        }

    if matched_groups and not bool(zero_summary.get("has_positive_coverage_trade_evidence")):
        return {
            "ok": False,
            "reason": str(zero_summary.get("dominant_zero_reason") or "NO_RECENT_POSITIVE_COVERAGE_TRADE_EVIDENCE"),
            "lineage_hints": hints,
            "matched_run_groups": matched_groups,
            "recent_summary": zero_summary,
            "quarantine_counts": quarantine_counts,
        }

    return {
        "ok": True,
        "reason": "",
        "lineage_hints": hints,
        "matched_run_groups": matched_groups,
        "recent_summary": zero_summary,
        "quarantine_counts": quarantine_counts,
    }


def _is_zero_coverage_seed_group(*, rows: list[dict[str, str]], rank_result: dict[str, Any]) -> bool:
    if not rows:
        return False
    positive_coverage_trade = False
    for row in rows:
        positive_coverage_trade = positive_coverage_trade or has_positive_coverage_trade_evidence(row)
    strict_diag = rank_result.get("strict_diag", {}) if isinstance(rank_result, dict) else {}
    if not isinstance(strict_diag, dict):
        strict_diag = {}
    try:
        variants_passing_all = int(float(strict_diag.get("variants_passing_all") or 0))
    except Exception:
        variants_passing_all = 0
    return not positive_coverage_trade and variants_passing_all <= 0


def _rank_result_has_zero_coverage_binding(rank_result: dict[str, Any]) -> bool:
    if not isinstance(rank_result, dict):
        return False
    details = str(rank_result.get("details") or "").strip()
    strict_diag = rank_result.get("strict_diag", {})
    if not isinstance(strict_diag, dict):
        strict_diag = {}
    binding_gates = {str(item).strip() for item in list(strict_diag.get("binding_gates", []) or []) if str(item).strip()}
    rejects = strict_diag.get("rejects", {})
    if not isinstance(rejects, dict):
        rejects = {}
    variants_passing_all = _as_int(strict_diag.get("variants_passing_all"), default=0, min_value=0) or 0
    coverage_below_rejects = _as_int(rejects.get("coverage_below"), default=0, min_value=0) or 0
    min_trades_rejects = _as_int(rejects.get("min_trades"), default=0, min_value=0) or 0
    return bool(
        variants_passing_all <= 0
        and (
            "coverage_below" in binding_gates
            or coverage_below_rejects > 0
            or "coverage_below" in details
        )
        and (
            "min_trades" in binding_gates
            or min_trades_rejects > 0
            or "min_trades" in details
        )
    )


def _assess_recent_seed_quality(
    *,
    app_root: Path,
    run_group_prefix: str,
    run_index_path: Path,
    limit_groups: int = 8,
) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, str]]] = {}
    if run_index_path.exists():
        try:
            with run_index_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    group = str(row.get("run_group") or "").strip()
                    if not group.startswith(f"{run_group_prefix}_"):
                        continue
                    by_group.setdefault(group, []).append({k: (v or "").strip() for k, v in row.items()})
        except Exception:
            by_group = {}

    recent_groups = sorted(by_group.keys())[-max(1, int(limit_groups)) :]
    groups: list[dict[str, Any]] = []
    zero_coverage_seed_streak = 0
    covered_window_count = 0
    for group in reversed(recent_groups):
        rows = list(by_group.get(group) or [])
        rank_result = _load_rank_result(app_root / "artifacts" / "wfa" / "aggregate" / group / "rank_result.json")
        zero_coverage = _is_zero_coverage_seed_group(rows=rows, rank_result=rank_result)
        if zero_coverage and zero_coverage_seed_streak == len(groups):
            zero_coverage_seed_streak += 1
        if not zero_coverage:
            covered_window_count += 1
        groups.append(
            {
                "run_group": group,
                "rows": len(rows),
                "zero_coverage": bool(zero_coverage),
            }
        )

    return {
        "groups_analyzed": len(groups),
        "recent_groups": list(groups),
        "zero_coverage_seed_streak": int(zero_coverage_seed_streak),
        "covered_window_count": int(covered_window_count),
        "repair_mode": bool(zero_coverage_seed_streak >= 1),
        "backlog_suppress": bool(zero_coverage_seed_streak >= 2),
        "hard_block_recommended": bool(zero_coverage_seed_streak >= 2),
        "hard_block_reason": "zero_coverage_seed_streak" if zero_coverage_seed_streak >= 2 else "",
    }


def _upsert_orphan_queue(*, orphan_path: Path, queue_rel: str, reason: str, cooldown_sec: int = 21600) -> None:
    queue = str(queue_rel or "").strip()
    if not queue:
        return
    rows: list[dict[str, str]] = []
    if orphan_path.exists():
        try:
            with orphan_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if str(row.get("queue") or "").strip() == queue:
                        continue
                    rows.append(
                        {
                            "queue": str(row.get("queue") or "").strip(),
                            "until_ts": str(row.get("until_ts") or "").strip(),
                            "reason": str(row.get("reason") or "").strip(),
                        }
                    )
        except Exception:
            rows = []
    until_ts = int(datetime.now(timezone.utc).timestamp()) + max(60, int(cooldown_sec))
    rows.append({"queue": queue, "until_ts": str(until_ts), "reason": str(reason or "").strip()})
    orphan_path.parent.mkdir(parents=True, exist_ok=True)
    with orphan_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["queue", "until_ts", "reason"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _hygiene_seed_queues(
    *,
    aggregate_dir: Path,
    app_root: Path,
    run_group_prefix: str,
    orphan_path: Path,
    incident_path: Path | None = None,
    cooldown_sec: int = 21600,
) -> dict[str, Any]:
    reviewed = 0
    pruned = 0
    orphaned = 0
    coverage_rejected = 0
    dedupe_rejected = 0
    zero_coverage_rejected = 0
    covered_window_count = 0
    queue_results: list[dict[str, Any]] = []
    run_index_path = aggregate_dir / "rollup" / "run_index.csv"
    run_index_groups = _load_run_index_groups(run_index_path, run_group_prefix=f"{run_group_prefix}_")
    for queue_path in sorted(aggregate_dir.glob(f"{run_group_prefix}_*/run_queue.csv")):
        rows = _load_queue_rows(queue_path)
        legacy_reason = ""
        if not rows:
            legacy_reason = _legacy_header_only_queue_reason(queue_path)
            if legacy_reason:
                prune_stats = _prune_seed_queue(queue_path=queue_path, app_root=app_root)
                queue_rel = _safe_rel(queue_path, app_root)
                queue_policy_path = _write_queue_policy_sidecar(
                    queue_path=queue_path,
                    app_root=app_root,
                    planner_policy_hash="coverage_hygiene",
                    selected_lane="hygiene",
                    selected_lane_index=0,
                    token_rotation=0,
                    parent_rotation_offset=0,
                    parent_diversity_depth=0,
                    confirm_replay_hints=[],
                    decision_payload={},
                    coverage_verified=False,
                    coverage_reason=legacy_reason,
                    ready_buffer_excluded=True,
                    seed_feasibility_status="blocked",
                    seed_feasibility_reason=legacy_reason,
                    lineage_positive_evidence=False,
                )
                _decorate_queue_metadata(
                    queue_path=queue_path,
                    planner_policy_hash="coverage_hygiene",
                    queue_policy_path=queue_policy_path,
                    app_root=app_root,
                    coverage_verified=False,
                    coverage_reason=legacy_reason,
                    ready_buffer_excluded=True,
                    seed_feasibility_status="blocked",
                    seed_feasibility_reason=legacy_reason,
                    lineage_positive_evidence=False,
                )
                _upsert_orphan_queue(
                    orphan_path=orphan_path,
                    queue_rel=queue_rel,
                    reason=legacy_reason,
                    cooldown_sec=cooldown_sec,
                )
                orphaned += 1
                queue_results.append(
                    {
                        "queue": queue_rel,
                        "rows_after": int(prune_stats.get("rows_after", 0) or 0),
                        "coverage_rejected": int(prune_stats.get("coverage_rejected", 0) or 0),
                        "dedupe_rejected": int(prune_stats.get("dedupe_rejected", 0) or 0),
                        "zero_coverage_history": False,
                        "orphan_reason": legacy_reason,
                        "blocked_rows_written": int(prune_stats.get("blocked_rows_written", 0) or 0),
                        "promotion_potential": 0.0,
                        "pre_rank_score": 0.0,
                        "gate_status": "UNKNOWN",
                        "gate_reason": legacy_reason,
                        "strict_gate_status": "UNKNOWN",
                        "strict_gate_reason": legacy_reason,
                    }
                )
            continue
        statuses = {normalize_queue_status(row.get("status")) for row in rows}
        if "completed" in statuses or "error" in statuses:
            continue
        if not statuses.intersection(PENDING_LIKE_STATUSES):
            continue
        reviewed += 1
        prune_stats = _prune_seed_queue(queue_path=queue_path, app_root=app_root)
        coverage_rejected += int(prune_stats.get("coverage_rejected", 0) or 0)
        dedupe_rejected += int(prune_stats.get("dedupe_rejected", 0) or 0)
        if int(prune_stats.get("coverage_rejected", 0) or 0) > 0 or int(prune_stats.get("dedupe_rejected", 0) or 0) > 0:
            pruned += 1
        rows_after = int(prune_stats.get("rows_after", 0) or 0)
        queue_rel = _safe_rel(queue_path, app_root)
        run_group = queue_path.parent.name
        group_rows = list(run_index_groups.get(run_group) or [])
        rank_result = _load_rank_result(queue_path.parent / "rank_result.json")
        zero_coverage_history = _is_zero_coverage_seed_group(
            rows=group_rows,
            rank_result=rank_result,
        ) and _rank_result_has_zero_coverage_binding(rank_result)
        reason = ""
        if rows_after <= 0:
            reason = str(prune_stats.get("block_reason") or "queue_pruned_empty")
        elif zero_coverage_history:
            dominant_zero_reason = _dominant_zero_reason(group_rows)
            reason = dominant_zero_reason or "NO_RECENT_POSITIVE_COVERAGE_TRADE_EVIDENCE"
            zero_coverage_rejected += 1
        else:
            covered_window_count += 1
        if reason:
            _upsert_orphan_queue(orphan_path=orphan_path, queue_rel=queue_rel, reason=reason, cooldown_sec=cooldown_sec)
            orphaned += 1
            if incident_path is not None:
                _append_log(
                    incident_path,
                    {
                        "ts": _utc_now_iso(),
                        "agent": "queue_seeder",
                        "kind": "queue_hygiene_blocked",
                        "human": "Queue hygiene fail-closed: queue excluded from dispatch due to coverage checks.",
                        "payload": {
                            "queue": queue_rel,
                            "reason": reason,
                            "rows_before": int(prune_stats.get("rows_before", 0) or 0),
                            "rows_after": rows_after,
                            "blocked_rows_written": int(prune_stats.get("blocked_rows_written", 0) or 0),
                            "coverage_rejected": int(prune_stats.get("coverage_rejected", 0) or 0),
                            "dedupe_rejected": int(prune_stats.get("dedupe_rejected", 0) or 0),
                            "missing_months": list(prune_stats.get("missing_months", []) or []),
                        },
                    },
                )
        coverage_verified = bool(rows_after > 0) and not bool(reason)
        lineage_recent_summary = dict(prune_stats.get("lineage_recent_summary") or {})
        lineage_positive_evidence = bool(lineage_recent_summary.get("has_positive_coverage_trade_evidence"))
        seed_feasibility_status = "blocked" if reason else "ok"
        seed_feasibility_reason = str(reason or "").strip()
        existing_policy = _existing_queue_policy_inputs(queue_path)
        preserve_recovery_identity = bool(
            coverage_verified
            and not reason
            and str(existing_policy.get("recovery_mode") or "").strip().lower() == "controlled"
        )
        planner_policy_hash = "coverage_hygiene"
        selected_lane = "hygiene"
        selected_lane_index = 0
        token_rotation = 0
        parent_rotation_offset = 0
        parent_diversity_depth = 0
        confirm_replay_hints: list[str] = []
        decision_payload: dict[str, Any] = {}
        recovery_mode = ""
        recovery_reason = ""
        recovery_lineage_anchor = ""
        if preserve_recovery_identity:
            planner_policy_hash = str(existing_policy.get("planner_policy_hash") or "").strip() or "coverage_hygiene"
            selected_lane = str(existing_policy.get("selected_lane") or "").strip() or "winner_proximate"
            selected_lane_index = int(existing_policy.get("selected_lane_index") or 0)
            token_rotation = int(existing_policy.get("token_rotation") or 0)
            parent_rotation_offset = int(existing_policy.get("parent_rotation_offset") or 0)
            parent_diversity_depth = int(existing_policy.get("parent_diversity_depth") or 0)
            confirm_replay_hints = list(existing_policy.get("confirm_replay_hints") or [])
            raw_decision_payload = existing_policy.get("decision_payload")
            if isinstance(raw_decision_payload, dict):
                decision_payload = raw_decision_payload
            recovery_mode = str(existing_policy.get("recovery_mode") or "").strip()
            recovery_reason = str(existing_policy.get("recovery_reason") or "").strip()
            recovery_lineage_anchor = str(existing_policy.get("recovery_lineage_anchor") or "").strip()
        queue_policy_path = _write_queue_policy_sidecar(
            queue_path=queue_path,
            app_root=app_root,
            planner_policy_hash=planner_policy_hash,
            selected_lane=selected_lane,
            selected_lane_index=selected_lane_index,
            token_rotation=token_rotation,
            parent_rotation_offset=parent_rotation_offset,
            parent_diversity_depth=parent_diversity_depth,
            confirm_replay_hints=confirm_replay_hints,
            decision_payload=decision_payload,
            coverage_verified=coverage_verified,
            coverage_reason=str(reason or "coverage_verified"),
            ready_buffer_excluded=bool(reason),
            seed_feasibility_status=seed_feasibility_status,
            seed_feasibility_reason=seed_feasibility_reason,
            lineage_positive_evidence=lineage_positive_evidence,
            recovery_mode=recovery_mode,
            recovery_reason=recovery_reason,
            recovery_lineage_anchor=recovery_lineage_anchor,
        )
        _decorate_queue_metadata(
            queue_path=queue_path,
            planner_policy_hash=planner_policy_hash,
            queue_policy_path=queue_policy_path,
            app_root=app_root,
            coverage_verified=coverage_verified,
            coverage_reason=str(reason or "coverage_verified"),
            ready_buffer_excluded=bool(reason),
            seed_feasibility_status=seed_feasibility_status,
            seed_feasibility_reason=seed_feasibility_reason,
            lineage_positive_evidence=lineage_positive_evidence,
            recovery_mode=recovery_mode,
            recovery_reason=recovery_reason,
            recovery_lineage_anchor=recovery_lineage_anchor,
        )
        queue_results.append(
            {
                "queue": queue_rel,
                "rows_after": rows_after,
                "coverage_rejected": int(prune_stats.get("coverage_rejected", 0) or 0),
                "dedupe_rejected": int(prune_stats.get("dedupe_rejected", 0) or 0),
                "zero_coverage_history": bool(zero_coverage_history),
                "orphan_reason": reason,
                "blocked_rows_written": int(prune_stats.get("blocked_rows_written", 0) or 0),
                "promotion_potential": 0.0,
                "pre_rank_score": 0.0,
                "gate_status": str(existing_policy.get("decision_payload", {}).get("gate_status") or "UNKNOWN"),
                "gate_reason": str(reason or existing_policy.get("decision_payload", {}).get("gate_reason") or "coverage_verified"),
                "strict_gate_status": str(
                    existing_policy.get("decision_payload", {}).get("strict_gate_status") or "UNKNOWN"
                ),
                "strict_gate_reason": str(
                    reason or existing_policy.get("decision_payload", {}).get("strict_gate_reason") or "coverage_verified"
                ),
            }
        )
    return {
        "reviewed": reviewed,
        "pruned": pruned,
        "orphaned": orphaned,
        "coverage_rejected": coverage_rejected,
        "dedupe_rejected": dedupe_rejected,
        "zero_coverage_rejected": zero_coverage_rejected,
        "covered_window_count": covered_window_count,
        "queues": queue_results[:32],
    }


def _summarize_queues(aggregate_dir: Path) -> tuple[int, int, int, int, int, list[Path]]:
    total_pending_like = 0
    dispatchable_pending = 0
    executable_pending = 0
    runnable_queue_count = 0
    scanned = 0
    queue_paths: list[Path] = []
    app_root = aggregate_dir.parents[2] if len(aggregate_dir.parents) >= 3 else aggregate_dir
    state_dir = aggregate_dir / ".autonomous"
    fullspan_queues = load_fullspan_queue_state(state_dir / "fullspan_decision_state.json")
    orphan_queues = load_orphan_queue_cooldowns(state_dir / "orphan_queues.csv")
    current_epoch = datetime.now(timezone.utc).timestamp()
    if not aggregate_dir.exists():
        return (total_pending_like, dispatchable_pending, executable_pending, runnable_queue_count, scanned, queue_paths)

    for queue_file in sorted(aggregate_dir.glob("*/run_queue.csv")):
        if queue_file.parent.name.startswith("."):
            continue
        scanned += 1
        queue_paths.append(queue_file)
        rows = _load_queue_rows(queue_file)
        legacy_reason = _legacy_header_only_queue_reason(queue_file)
        if legacy_reason:
            rows = []
        queue_rel = queue_rel_path(queue_file, app_root)
        blocked_reason = queue_dispatch_block_reason(
            queue_rel=queue_rel,
            fullspan_entry=fullspan_queues.get(queue_rel),
            orphan_entry=orphan_queues.get(queue_rel),
            now_epoch=current_epoch,
        )
        queue_executable = 0
        for row in rows:
            status = normalize_queue_status(row.get("status"))
            if status in PENDING_LIKE_STATUSES:
                total_pending_like += 1
            if row_counts_executable(status, row.get("config_path"), app_root):
                executable_pending += 1
                queue_executable += 1
            if not blocked_reason and row_counts_dispatchable_pending(status, row.get("config_path"), app_root):
                dispatchable_pending += 1
        if queue_executable > 0:
            runnable_queue_count += 1
    return total_pending_like, dispatchable_pending, executable_pending, runnable_queue_count, scanned, queue_paths


def _emit_state(state_path: Path, payload: dict[str, Any]) -> None:
    normalized = _normalize_seeder_state_payload(payload)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(normalized, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _append_log(log_path: Path, payload: dict[str, Any]) -> None:
    normalized = _normalize_seeder_state_payload(payload)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        json.dump(normalized, handle, ensure_ascii=False)
        handle.write("\n")


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _collect_empty_queue_refs(payload: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()

    def _push(value: Any) -> None:
        ref = str(value or "").strip()
        if not ref or ref in seen:
            return
        seen.add(ref)
        refs.append(ref)

    hygiene = payload.get("hygiene", {})
    if isinstance(hygiene, dict):
        for entry in list(hygiene.get("queues") or [])[:32]:
            if not isinstance(entry, dict):
                continue
            queue_ref = str(entry.get("queue") or "").strip()
            rows_after = _as_int(entry.get("rows_after"), default=None, min_value=0)
            orphan_reason = str(entry.get("orphan_reason") or "").strip()
            if queue_ref and ((rows_after is not None and rows_after <= 0) or orphan_reason):
                _push(queue_ref)

    queue_ref = str(payload.get("queue_path") or "").strip()
    reason = str(payload.get("reason") or "").strip().lower()
    if queue_ref and any(
        token in reason
        for token in (
            "queue_pruned_empty",
            "coverage",
            "dedupe",
            "empty",
            "blocked",
        )
    ):
        _push(queue_ref)

    return refs[:16]


def _skip_next_retry_epoch(payload: dict[str, Any], now_epoch: int) -> int:
    hard_block_reason = str(payload.get("hard_block_reason") or "").strip()
    reason = str(payload.get("reason") or "").strip()
    if not hard_block_reason and reason.startswith("hard_block:"):
        hard_block_reason = reason.split(":", 1)[1].strip()
    yield_governor = payload.get("yield_governor", {})
    if not hard_block_reason and isinstance(yield_governor, dict):
        hard_block_reason = str(yield_governor.get("hard_block_reason") or "").strip()
    controlled_recovery_attempts_remaining = _as_int(
        payload.get("controlled_recovery_attempts_remaining"),
        default=None,
        min_value=0,
    )
    if controlled_recovery_attempts_remaining is None and isinstance(yield_governor, dict):
        controlled_recovery_attempts_remaining = _as_int(
            yield_governor.get("controlled_recovery_attempts_remaining"),
            default=None,
            min_value=0,
        )
    search_quality = payload.get("search_quality", {})
    if controlled_recovery_attempts_remaining is None and isinstance(search_quality, dict):
        controlled_recovery_attempts_remaining = _as_int(
            search_quality.get("controlled_recovery_attempts_remaining"),
            default=None,
            min_value=0,
        )
    controlled_broad_rearm_after_epoch = _as_int(
        payload.get("controlled_broad_rearm_after_epoch"),
        default=0,
        min_value=0,
    ) or 0
    if isinstance(yield_governor, dict):
        controlled_broad_rearm_after_epoch = max(
            int(controlled_broad_rearm_after_epoch),
            int(_as_int(yield_governor.get("controlled_broad_rearm_after_epoch"), default=0, min_value=0) or 0),
        )
    if isinstance(search_quality, dict):
        controlled_broad_rearm_after_epoch = max(
            int(controlled_broad_rearm_after_epoch),
            int(_as_int(search_quality.get("controlled_broad_rearm_after_epoch"), default=0, min_value=0) or 0),
        )
    zero_coverage_exhausted = (
        hard_block_reason == "zero_coverage_seed_streak"
        and controlled_recovery_attempts_remaining is not None
        and int(controlled_recovery_attempts_remaining) <= 0
    )
    if zero_coverage_exhausted:
        if controlled_broad_rearm_after_epoch > now_epoch:
            return int(controlled_broad_rearm_after_epoch)
        explicit = _as_int(payload.get("next_retry_epoch"), default=0, min_value=0) or 0
        if explicit > 0:
            return int(explicit)
        return int(now_epoch + CONTROLLED_BROAD_REARM_DELAY_SEC)
    explicit = _as_int(payload.get("next_retry_epoch"), default=0, min_value=0) or 0
    if explicit > 0:
        return int(explicit)
    hard_block_until_epoch = _as_int(payload.get("hard_block_until_epoch"), default=0, min_value=0) or 0
    if isinstance(yield_governor, dict):
        hard_block_until_epoch = max(
            int(hard_block_until_epoch),
            int(_as_int(yield_governor.get("hard_block_until_epoch"), default=0, min_value=0) or 0),
        )
    if hard_block_until_epoch > now_epoch:
        return int(hard_block_until_epoch)
    return int(now_epoch + SEEDER_SKIP_RETRY_SEC)


def _normalize_seeder_state_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload or {})
    now_epoch = int(datetime.now(timezone.utc).timestamp())
    status = str(normalized.get("status") or "").strip().lower()
    result_kind = status if status in {"seeded", "skipped", "failed"} else ("skipped" if status == "active" else status)
    empty_queue_refs = _collect_empty_queue_refs(normalized)
    next_retry_epoch = 0
    if result_kind == "skipped":
        next_retry_epoch = _skip_next_retry_epoch(normalized, now_epoch)
    normalized["result_kind"] = str(result_kind or status or "")
    normalized["next_retry_epoch"] = int(next_retry_epoch)
    normalized["empty_queue_refs"] = list(empty_queue_refs)
    return normalized


def _normalized_selector_evidence(decision_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = decision_payload if isinstance(decision_payload, dict) else {}

    def _as_score(key: str) -> float:
        try:
            return float(payload.get(key) or 0.0)
        except Exception:
            return 0.0

    return {
        "promotion_potential": _as_score("promotion_potential"),
        "pre_rank_score": _as_score("pre_rank_score"),
        "gate_status": str(payload.get("gate_status") or "UNKNOWN").strip() or "UNKNOWN",
        "gate_reason": str(payload.get("gate_reason") or "UNKNOWN").strip() or "UNKNOWN",
        "strict_gate_status": str(payload.get("strict_gate_status") or "UNKNOWN").strip() or "UNKNOWN",
        "strict_gate_reason": str(payload.get("strict_gate_reason") or "UNKNOWN").strip() or "UNKNOWN",
    }


def _write_queue_policy_sidecar(
    *,
    queue_path: Path,
    app_root: Path,
    planner_policy_hash: str,
    selected_lane: str,
    selected_lane_index: int,
    token_rotation: int,
    parent_rotation_offset: int,
    parent_diversity_depth: int,
    confirm_replay_hints: list[str],
    decision_payload: dict[str, Any],
    coverage_verified: bool = True,
    coverage_reason: str = "",
    ready_buffer_excluded: bool = False,
    seed_feasibility_status: str = "unknown",
    seed_feasibility_reason: str = "",
    lineage_positive_evidence: bool = False,
    recovery_mode: str = "",
    recovery_reason: str = "",
    recovery_lineage_anchor: str = "",
) -> Path:
    queue_dir = queue_path.parent
    sidecar_path = queue_dir / "queue_policy.json"
    planner_hashes = decision_payload.get("planner_hashes", {}) if isinstance(decision_payload, dict) else {}
    if not isinstance(planner_hashes, dict):
        planner_hashes = {}
    lane_selection = decision_payload.get("lane_selection", {}) if isinstance(decision_payload, dict) else {}
    if not isinstance(lane_selection, dict):
        lane_selection = {}
    parent_diversification = decision_payload.get("parent_diversification", {}) if isinstance(decision_payload, dict) else {}
    if not isinstance(parent_diversification, dict):
        parent_diversification = {}
    parent_resolution = decision_payload.get("parent_resolution", {}) if isinstance(decision_payload, dict) else {}
    if not isinstance(parent_resolution, dict):
        parent_resolution = {}

    parent_pool = list(parent_diversification.get("pool") or [])
    primary_parent_index = int(parent_diversification.get("primary_parent_index") or 0)
    primary_parent: dict[str, Any] = {}
    if 0 <= primary_parent_index < len(parent_pool):
        raw_parent = parent_pool[primary_parent_index]
        if isinstance(raw_parent, dict):
            primary_parent = dict(raw_parent)

    selector_evidence = _normalized_selector_evidence(decision_payload)

    payload = {
        "ts": _utc_now_iso(),
        "queue_path": _safe_rel(queue_path, app_root),
        "planner_policy_hash": planner_policy_hash,
        "buffer_policy_version": planner_policy_hash,
        "planner_hash": str(planner_hashes.get("planner_hash") or "").strip(),
        "seed_lane": str(lane_selection.get("seed_lane") or selected_lane).strip(),
        "seed_lane_index": int(lane_selection.get("seed_lane_index") or selected_lane_index),
        "token_rotation": int(token_rotation),
        "parent_rotation_offset": int(parent_diversification.get("rotation_offset") or parent_rotation_offset),
        "parent_diversity_depth": int(parent_diversification.get("depth") or parent_diversity_depth),
        "winner_proximate_tokens": [
            str(token).strip()
            for token in list(parent_resolution.get("winner_proximate_tokens", []) or [])
            if str(token).strip()
        ][:8],
        "confirm_replay_hints": [
            str(token).strip()
            for token in list(lane_selection.get("confirm_replay_hints", confirm_replay_hints) or [])
            if str(token).strip()
        ][:8],
        "parent_resolution": parent_resolution,
        "parent_diversification": parent_diversification,
        "primary_parent": {
            "parent_id": str(primary_parent.get("parent_id") or parent_diversification.get("primary_parent_id") or "").strip(),
            "parent_config_path": str(primary_parent.get("parent_config_path") or "").strip(),
            "run_group": str(primary_parent.get("run_group") or "").strip(),
            "lineage_uid": str(primary_parent.get("lineage_uid") or "").strip(),
        },
        "coverage_verified": bool(coverage_verified),
        "coverage_reason": str(coverage_reason or ("coverage_verified" if coverage_verified else "coverage_unverified")).strip(),
        "ready_buffer_excluded": bool(ready_buffer_excluded),
        "seed_feasibility_status": str(seed_feasibility_status or "unknown").strip(),
        "seed_feasibility_reason": str(seed_feasibility_reason or "").strip(),
        "lineage_positive_evidence": bool(lineage_positive_evidence),
        "recovery_mode": str(recovery_mode or "").strip(),
        "recovery_reason": str(recovery_reason or "").strip(),
        "recovery_lineage_anchor": str(recovery_lineage_anchor or "").strip(),
        **selector_evidence,
    }
    sidecar_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return sidecar_path


def _decorate_queue_metadata(
    *,
    queue_path: Path,
    planner_policy_hash: str,
    queue_policy_path: Path,
    app_root: Path,
    coverage_verified: bool = True,
    coverage_reason: str = "",
    ready_buffer_excluded: bool = False,
    seed_feasibility_status: str = "unknown",
    seed_feasibility_reason: str = "",
    lineage_positive_evidence: bool = False,
    recovery_mode: str = "",
    recovery_reason: str = "",
    recovery_lineage_anchor: str = "",
) -> None:
    if not queue_path.exists():
        return
    with queue_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        return
    if "metadata_json" not in fieldnames:
        fieldnames.append("metadata_json")
    queue_policy_rel = _safe_rel(queue_policy_path, app_root)
    queue_rel = _safe_rel(queue_path, app_root)
    queue_policy_payload = {}
    if queue_policy_path.exists():
        try:
            parsed_policy = json.loads(queue_policy_path.read_text(encoding="utf-8"))
        except Exception:
            parsed_policy = {}
        if isinstance(parsed_policy, dict):
            queue_policy_payload = parsed_policy
    selector_evidence = _normalized_selector_evidence(queue_policy_payload)
    for row in rows:
        meta: dict[str, Any] = {}
        raw_meta = str(row.get("metadata_json") or "").strip()
        if raw_meta:
            try:
                parsed = json.loads(raw_meta)
            except Exception:
                parsed = {}
            if isinstance(parsed, dict):
                meta = dict(parsed)
        meta["planner_policy_hash"] = planner_policy_hash
        meta["buffer_policy_version"] = planner_policy_hash
        meta["queue_policy_path"] = queue_policy_rel
        meta["queue_path"] = queue_rel
        meta["coverage_verified"] = bool(coverage_verified)
        meta["coverage_reason"] = str(
            coverage_reason or ("coverage_verified" if coverage_verified else "coverage_unverified")
        ).strip()
        meta["ready_buffer_excluded"] = bool(ready_buffer_excluded)
        meta["seed_feasibility_status"] = str(seed_feasibility_status or "unknown").strip()
        meta["seed_feasibility_reason"] = str(seed_feasibility_reason or "").strip()
        meta["lineage_positive_evidence"] = bool(lineage_positive_evidence)
        meta["recovery_mode"] = str(recovery_mode or "").strip()
        meta["recovery_reason"] = str(recovery_reason or "").strip()
        meta["recovery_lineage_anchor"] = str(recovery_lineage_anchor or "").strip()
        for key, default_value in selector_evidence.items():
            if key in {"promotion_potential", "pre_rank_score"}:
                try:
                    meta[key] = float(meta.get(key, default_value) or default_value)
                except Exception:
                    meta[key] = float(default_value)
            else:
                meta[key] = str(meta.get(key) or default_value).strip() or str(default_value)
        row["metadata_json"] = json.dumps(meta, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with queue_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_queue_policy_payload(queue_path: Path) -> dict[str, Any]:
    policy_path = queue_path.parent / "queue_policy.json"
    if not policy_path.exists():
        return {}
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_first_row_metadata(queue_path: Path) -> dict[str, Any]:
    rows = _load_queue_rows(queue_path)
    if not rows:
        return {}
    raw_meta = str(rows[0].get("metadata_json") or "").strip()
    if not raw_meta:
        return {}
    try:
        payload = json.loads(raw_meta)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _existing_queue_policy_inputs(queue_path: Path) -> dict[str, Any]:
    sidecar = _load_queue_policy_payload(queue_path)
    metadata = _load_first_row_metadata(queue_path)
    parent_diversification = sidecar.get("parent_diversification", {})
    if not isinstance(parent_diversification, dict):
        parent_diversification = {}
    parent_resolution = sidecar.get("parent_resolution", {})
    if not isinstance(parent_resolution, dict):
        parent_resolution = {}
    confirm_replay_hints = sidecar.get("confirm_replay_hints", [])
    if not isinstance(confirm_replay_hints, list):
        confirm_replay_hints = []
    selected_lane = str(sidecar.get("seed_lane") or metadata.get("seed_lane") or "").strip()
    selected_lane_index = _as_int(
        sidecar.get("seed_lane_index", metadata.get("seed_lane_index")),
        default=0,
        min_value=0,
    )
    token_rotation = _as_int(sidecar.get("token_rotation"), default=0, min_value=0)
    parent_rotation_offset = _as_int(
        sidecar.get("parent_rotation_offset", parent_diversification.get("rotation_offset")),
        default=0,
        min_value=0,
    )
    parent_diversity_depth = _as_int(
        sidecar.get("parent_diversity_depth", parent_diversification.get("depth")),
        default=0,
        min_value=0,
    )
    planner_policy_hash = str(
        sidecar.get("planner_policy_hash")
        or metadata.get("planner_policy_hash")
        or metadata.get("buffer_policy_version")
        or ""
    ).strip()
    planner_hash = str(sidecar.get("planner_hash") or "").strip()
    recovery_mode = str(sidecar.get("recovery_mode") or metadata.get("recovery_mode") or "").strip()
    recovery_reason = str(sidecar.get("recovery_reason") or metadata.get("recovery_reason") or "").strip()
    recovery_lineage_anchor = str(
        sidecar.get("recovery_lineage_anchor") or metadata.get("recovery_lineage_anchor") or ""
    ).strip()
    return {
        "planner_policy_hash": planner_policy_hash,
        "selected_lane": selected_lane,
        "selected_lane_index": int(selected_lane_index or 0),
        "token_rotation": int(token_rotation or 0),
        "parent_rotation_offset": int(parent_rotation_offset or 0),
        "parent_diversity_depth": int(parent_diversity_depth or 0),
        "confirm_replay_hints": [str(token).strip() for token in confirm_replay_hints if str(token).strip()][:8],
        "decision_payload": {
            "planner_hashes": {"planner_hash": planner_hash},
            "lane_selection": {
                "seed_lane": selected_lane,
                "seed_lane_index": int(selected_lane_index or 0),
                "confirm_replay_hints": [str(token).strip() for token in confirm_replay_hints if str(token).strip()][:8],
            },
            "parent_diversification": parent_diversification,
            "parent_resolution": parent_resolution,
        },
        "recovery_mode": recovery_mode,
        "recovery_reason": recovery_reason,
        "recovery_lineage_anchor": recovery_lineage_anchor,
    }


def _load_directive(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _controlled_broad_recovery_from_directive(directive: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(directive, dict):
        return {"enabled": False}
    payload = directive.get("controlled_broad_recovery", {})
    if not isinstance(payload, dict):
        payload = {}
    search_quality = directive.get("search_quality", {})
    if not isinstance(search_quality, dict):
        search_quality = {}

    enabled = bool(
        payload.get("enabled")
        or search_quality.get("controlled_broad_active")
        or str(directive.get("mode") or "").strip() == "controlled_broad_recovery"
    )
    contains = _to_tokens(payload.get("contains", []))
    if not contains:
        contains = _to_tokens(search_quality.get("controlled_broad_contains", []))
    if not contains:
        contains = _to_tokens(directive.get("contains", []))
    return {
        "enabled": enabled,
        "reason": str(
            payload.get("reason")
            or search_quality.get("controlled_broad_reason")
            or "stagnant_empty_backlog_no_strict_progress"
        ).strip(),
        "contains": contains,
        "num_variants": int(
            _as_int(
                payload.get("num_variants"),
                default=CONTROLLED_BROAD_NUM_VARIANTS,
                min_value=1,
            )
            or CONTROLLED_BROAD_NUM_VARIANTS
        ),
        "num_variants_floor": int(
            _as_int(
                payload.get("num_variants_floor"),
                default=CONTROLLED_BROAD_NUM_VARIANTS_FLOOR,
                min_value=1,
            )
            or CONTROLLED_BROAD_NUM_VARIANTS_FLOOR
        ),
        "max_changed_keys": int(
            _as_int(
                payload.get("max_changed_keys"),
                default=CONTROLLED_BROAD_MAX_CHANGED_KEYS,
                min_value=1,
            )
            or CONTROLLED_BROAD_MAX_CHANGED_KEYS
        ),
        "dedupe_distance": float(
            _as_float(
                payload.get("dedupe_distance"),
                default=CONTROLLED_BROAD_DEDUPE_DISTANCE,
                min_value=0.0,
            )
            or CONTROLLED_BROAD_DEDUPE_DISTANCE
        ),
        "cooldown_sec": int(
            _as_int(
                payload.get("cooldown_sec"),
                default=_as_int(
                    search_quality.get("controlled_broad_cooldown_sec"),
                    default=CONTROLLED_BROAD_REARM_DELAY_SEC,
                    min_value=0,
                ),
                min_value=0,
            )
            or CONTROLLED_BROAD_REARM_DELAY_SEC
        ),
    }


def _load_impossibility_pruner(path: Path) -> dict[str, Any]:
    micro_caps = micro_broad_search_caps()
    if not path.exists():
        return {"active": False, "reason": "missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"active": False, "reason": "invalid_json"}
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    if not isinstance(entries, list) or not entries:
        return {"active": False, "reason": "empty"}

    counts: dict[str, int] = {}
    for entry in entries[-200:]:
        if not isinstance(entry, dict):
            continue
        code = str(entry.get("code") or "").strip().upper()
        if not code:
            continue
        counts[code] = counts.get(code, 0) + 1

    if not counts:
        return {"active": False, "reason": "no_codes"}

    dominant_code, dominant_count = max(counts.items(), key=lambda item: item[1])
    total = sum(counts.values())
    dominant_ratio = (float(dominant_count) / float(total)) if total > 0 else 0.0
    active = bool(
        total >= 6
        and dominant_ratio >= 0.45
        and dominant_code
        in {
            "CONFIG_VALIDATION_ERROR",
            "MAX_VAR_MULTIPLIER_INVALID",
            "MAX_CORRELATION_INVALID",
            "NON_POSITIVE_THRESHOLD",
            "INVALID_PARAM",
        }
    )
    return {
        "active": active,
        "reason": dominant_code if active else "no_dominant_impossible_pattern",
        "dominant_code": dominant_code,
        "dominant_count": dominant_count,
        "total": total,
        "dominant_ratio": dominant_ratio,
        **micro_caps,
    }


def _load_search_blacklist(path: Path) -> dict[str, Any]:
    micro_caps = micro_broad_search_caps()
    if not path.exists():
        return {"active": False, "reason": "missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"active": False, "reason": "invalid_json"}
    if not isinstance(payload, dict):
        return {"active": False, "reason": "invalid_payload"}

    active = bool(payload.get("active"))
    caps = payload.get("recommended_caps", {})
    if not isinstance(caps, dict):
        caps = {}
    stats = payload.get("stats", {})
    if not isinstance(stats, dict):
        stats = {}

    return {
        "active": active,
        "reason": str(stats.get("dominant_code") or payload.get("reason") or ""),
        "dominant_code": str(stats.get("dominant_code") or ""),
        "dominant_ratio": float(stats.get("dominant_ratio") or 0.0),
        "total": int(stats.get("total_coded") or 0),
        "max_changed_keys_cap": int(caps.get("max_changed_keys_cap") or micro_caps["max_changed_keys_cap"]),
        "dedupe_distance_floor": float(caps.get("dedupe_distance_floor") or micro_caps["dedupe_distance_floor"]),
        "num_variants_cap": int(caps.get("num_variants_cap") or micro_caps["num_variants_cap"]),
        "policy_scale": str(caps.get("policy_scale") or micro_caps["policy_scale"]),
        "blocked_evo_uids_count": int(len(list(payload.get("blocked_evo_uids", []) or []))),
    }


def _merge_unique(left: list[str], right: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for token in [*left, *right]:
        norm = str(token or "").strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _stable_hash(payload: Any, *, prefix: str, size: int = 16) -> str:
    """Compact deterministic hash for state/metadata provenance fields."""
    try:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        encoded = str(payload)
    digest = hashlib.sha1(encoded.encode("utf-8")).hexdigest()[: max(6, int(size))]
    return f"{prefix}_{digest}"


def _rotate_tokens(tokens: list[str], offset: int) -> list[str]:
    if not tokens:
        return []
    n = len(tokens)
    if n <= 1:
        return list(tokens)
    idx = int(offset) % n
    if idx == 0:
        return list(tokens)
    return list(tokens[idx:]) + list(tokens[:idx])


def _load_previous_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _select_seed_lane(
    *,
    winner_proximate_tokens: list[str],
    preferred_any_contains: list[str],
    generic_contains: list[str],
    directive_replay_fastlane_tokens: list[str] | None = None,
    yield_governor: dict[str, Any],
    previous_state: dict[str, Any],
) -> dict[str, Any]:
    lane_weights_defaults = {
        "winner_proximate": 65,
        "broad_search": 15,
        "confirm_replay": 20,
    }
    lane_weights = dict(lane_weights_defaults)
    raw_weights = yield_governor.get("lane_weights", {}) if isinstance(yield_governor, dict) else {}
    if isinstance(raw_weights, dict):
        for key in lane_weights_defaults:
            try:
                lane_weights[key] = max(0, int(float(raw_weights.get(key, lane_weights[key]))))
            except Exception:
                continue

    winner_tokens = [str(token).strip() for token in list(winner_proximate_tokens or []) if str(token).strip()]
    preferred_tokens = [str(token).strip() for token in list(preferred_any_contains or []) if str(token).strip()]
    generic_tokens = [str(token).strip() for token in list(generic_contains or []) if str(token).strip()]
    search_quality = normalize_search_quality_state(yield_governor, winner_proximate_contains=winner_tokens)
    raw_search_quality = yield_governor.get("search_quality", {}) if isinstance(yield_governor, dict) else {}
    if isinstance(raw_search_quality, dict) and bool(raw_search_quality.get("controlled_broad_active")):
        search_quality = {
            **dict(search_quality),
            "broad_search_allowed": True,
            "seed_generation_mode": str(raw_search_quality.get("seed_generation_mode") or "controlled_broad_micro").strip()
            or "controlled_broad_micro",
            "controlled_broad_active": True,
            "controlled_broad_reason": str(raw_search_quality.get("controlled_broad_reason") or "").strip(),
            "controlled_broad_contains": _to_tokens(raw_search_quality.get("controlled_broad_contains", [])),
            "controlled_broad_cooldown_sec": int(
                _as_int(raw_search_quality.get("controlled_broad_cooldown_sec"), default=0, min_value=0) or 0
            ),
        }
    broad_search_allowed = bool(search_quality.get("broad_search_allowed"))
    confirm_replay_source = "winner_fallback"
    confirm_replay_tokens = _to_tokens(directive_replay_fastlane_tokens or [])
    if confirm_replay_tokens:
        confirm_replay_source = "directive_replay_fastlane"
    if not confirm_replay_tokens:
        confirm_replay_tokens = _to_tokens((yield_governor.get("replay_fastlane") or {}).get("contains", []))
        if confirm_replay_tokens:
            confirm_replay_source = "yield_replay_fastlane"
    if not confirm_replay_tokens:
        confirm_replay_tokens = _to_tokens((yield_governor.get("confirm_replay") or {}).get("contains", []))
        if confirm_replay_tokens:
            confirm_replay_source = "legacy_confirm_replay"
    if not confirm_replay_tokens:
        confirm_replay_tokens = _to_tokens(yield_governor.get("confirm_replay_contains", []))
        if confirm_replay_tokens:
            confirm_replay_source = "legacy_confirm_replay_contains"
    if not confirm_replay_tokens:
        # Compatibility fallback only when replay fastlane is genuinely empty.
        confirm_replay_tokens = list(winner_tokens)
    if not broad_search_allowed and confirm_replay_source == "winner_fallback":
        confirm_replay_tokens = []

    lane_tokens = {
        "winner_proximate": winner_tokens,
        "broad_search": (preferred_tokens or generic_tokens) if broad_search_allowed else [],
        "confirm_replay": confirm_replay_tokens,
    }
    available_lanes = [lane for lane in ("winner_proximate", "broad_search", "confirm_replay") if lane_tokens.get(lane)]
    if not available_lanes:
        fallback = winner_tokens or confirm_replay_tokens or generic_tokens or ["autonomous_queue_seeder"]
        lane_tokens["winner_proximate"] = fallback
        available_lanes = ["winner_proximate"]

    ranked = sorted(
        available_lanes,
        key=lambda lane: (-int(lane_weights.get(lane, 0)), 0 if lane == "winner_proximate" else 1, lane),
    )
    selected_lane = ranked[0]

    prev_lane = str(
        previous_state.get("selected_lane")
        or (previous_state.get("lane_selection") or {}).get("selected_lane")
        or ""
    ).strip()
    try:
        prev_streak = int(
            previous_state.get("lane_streak")
            or (previous_state.get("lane_selection") or {}).get("lane_streak")
            or 0
        )
    except Exception:
        prev_streak = 0
    if len(ranked) > 1 and prev_lane == selected_lane and prev_streak >= 2:
        selected_lane = ranked[1]

    selected_index = ranked.index(selected_lane) if selected_lane in ranked else 0
    try:
        prev_token_rotation = int(
            previous_state.get("token_rotation")
            or (previous_state.get("lane_selection") or {}).get("token_rotation")
            or 0
        )
    except Exception:
        prev_token_rotation = 0
    selected_tokens = list(lane_tokens.get(selected_lane) or [])
    token_rotation = prev_token_rotation + 1 if prev_lane == selected_lane else 0
    selected_tokens = _rotate_tokens(selected_tokens, token_rotation)

    selected_anchor = selected_tokens[0] if selected_tokens else ""
    contains_out = [selected_anchor] if selected_anchor else []
    if not contains_out:
        contains_out = list(generic_tokens or preferred_tokens or winner_tokens or ["autonomous_queue_seeder"])

    lane_streak = prev_streak + 1 if prev_lane == selected_lane else 1
    exploit_first = bool(winner_tokens)
    parent_rotation_offset = int(token_rotation + selected_index)
    confirm_replay_hints = selected_tokens if selected_lane == "confirm_replay" else []

    return {
        "selected_lane": selected_lane,
        "available_lanes": available_lanes,
        "ranked_lanes": ranked,
        "selected_index": int(selected_index),
        "lane_streak": int(lane_streak),
        "token_rotation": int(token_rotation),
        "exploit_first": bool(exploit_first),
        "lane_weights": lane_weights,
        "contains": contains_out,
        "winner_proximate_tokens": list(winner_tokens),
        "confirm_replay_hints": list(confirm_replay_hints),
        "confirm_replay_source": str(confirm_replay_source),
        "parent_rotation_offset": int(parent_rotation_offset),
        "search_quality": search_quality,
    }


def _to_tokens(value: Any) -> list[str]:
    items: list[Any]
    if isinstance(value, str):
        text = str(value).strip()
        if not text:
            return []
        items = [part.strip() for part in text.split(",")] if "," in text else [text]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        return []
    out: list[str] = []
    seen = set()
    for item in items:
        token = str(item or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _as_int(value: Any, *, default: int | None = None, min_value: int | None = None) -> int | None:
    try:
        parsed = int(float(value))
    except Exception:
        return default
    if min_value is not None and parsed < min_value:
        return min_value
    return parsed


def _as_float(value: Any, *, default: float | None = None, min_value: float | None = None) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return default
    if min_value is not None and parsed < min_value:
        return min_value
    return parsed


def _load_ready_queue_buffer(path: Path) -> dict[str, Any]:
    state: dict[str, Any] = {
        "exists": False,
        "status": "missing",
        "reason": "missing",
        "target_depth": None,
        "refill_threshold": None,
        "effective_refill_threshold": None,
        "ready_depth": 0,
        "ready_depth_total": 0,
        "coverage_verified_ready_count": 0,
        "seed_needed": False,
    }
    if not path.exists():
        return state

    state["exists"] = True
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        state.update({"status": "invalid_json", "reason": "invalid_json"})
        return state

    if not isinstance(payload, dict):
        state.update({"status": "invalid_payload", "reason": "invalid_payload"})
        return state

    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        state.update({"status": "invalid_entries", "reason": "invalid_entries"})
        return state

    target_depth = _as_int(payload.get("target_depth"), default=None, min_value=0)
    refill_threshold = _as_int(payload.get("refill_threshold"), default=None, min_value=0)
    effective_refill_threshold = refill_threshold if refill_threshold is not None else target_depth
    ready_depth_total = len(entries)
    coverage_verified_ready_count = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if bool(entry.get("coverage_verified", True)):
            coverage_verified_ready_count += 1
    ready_depth = int(coverage_verified_ready_count)

    state.update(
        {
            "status": "ok",
            "reason": "threshold_missing",
            "target_depth": int(target_depth) if target_depth is not None else None,
            "refill_threshold": int(refill_threshold) if refill_threshold is not None else None,
            "effective_refill_threshold": int(effective_refill_threshold) if effective_refill_threshold is not None else None,
            "ready_depth": int(ready_depth),
            "ready_depth_total": int(ready_depth_total),
            "coverage_verified_ready_count": int(coverage_verified_ready_count),
        }
    )
    if effective_refill_threshold is None:
        return state

    seed_needed = int(ready_depth) < int(effective_refill_threshold)
    state["seed_needed"] = bool(seed_needed)
    state["reason"] = "below_refill_threshold" if seed_needed else "buffer_ok"
    return state


def _evaluate_seed_needed(
    *,
    executable_pending: int,
    pending_threshold: int,
    runnable_queue_count: int,
    ready_buffer_state: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if (int(executable_pending) < int(pending_threshold)) or (int(runnable_queue_count) == 0):
        reasons.append("below_executable_threshold_or_no_runnable")
    if bool(ready_buffer_state.get("seed_needed")):
        reasons.append("ready_buffer_below_refill_threshold")
    return bool(reasons), reasons


def _normalize_decision_entries(raw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                out.append(dict(item))
        return out
    if not isinstance(raw, dict):
        return out

    if any(
        key in raw
        for key in {
            "decision",
            "action",
            "reason",
            "contains",
            "exclude_knob",
            "exclude_knobs",
            "repair_mode",
            "repair_max_neighbors",
        }
    ):
        out.append(dict(raw))
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        entry = dict(value)
        entry.setdefault("queue", str(key))
        out.append(entry)
    return out


def _planner_repair_mode_args(*, enabled: bool, supported_planner_args: set[str]) -> list[str]:
    if not enabled or "--repair-mode" not in supported_planner_args:
        return []
    return ["--repair-mode", "validation_neighbor"]


def _load_gate_surrogate_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "status": "missing",
            "hard_fail_risk_policy": {},
            "queue_decisions": [],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "exists": True,
            "status": "invalid_json",
            "hard_fail_risk_policy": {},
            "queue_decisions": [],
        }
    if not isinstance(payload, dict):
        return {
            "exists": True,
            "status": "invalid_payload",
            "hard_fail_risk_policy": {},
            "queue_decisions": [],
        }

    hard_policy = payload.get("hard_fail_risk_policy", {})
    if not isinstance(hard_policy, dict):
        hard_policy = {}

    queue_decisions = _normalize_decision_entries(payload.get("queue_decisions"))
    if not queue_decisions:
        queue_decisions = _normalize_decision_entries(hard_policy.get("queue_decisions") or hard_policy.get("decisions"))

    top_decision = str(hard_policy.get("queue_decision") or hard_policy.get("decision") or hard_policy.get("action") or "").strip()
    if top_decision:
        synthetic = dict(hard_policy)
        synthetic["decision"] = top_decision
        synthetic.setdefault("source", "hard_fail_risk_policy")
        queue_decisions = [synthetic, *queue_decisions]

    return {
        "exists": True,
        "status": "ok",
        "hard_fail_risk_policy": hard_policy,
        "queue_decisions": queue_decisions,
    }


def _load_yield_governor_state(path: Path) -> dict[str, Any]:
    state: dict[str, Any] = {
        "exists": False,
        "status": "missing",
        "reason": "missing",
        "active": False,
        "hard_block_active": False,
        "hard_block_reason": "",
        "hard_block_until_epoch": 0,
        "zero_coverage_seed_streak": 0,
        "zero_coverage_seed_streak_reason": "",
        "preferred_contains": [],
        "cooldown_contains": [],
        "preferred_operator_ids": [],
        "cooldown_operator_ids": [],
        "winner_proximate": {"enabled": False, "contains": [], "reason": ""},
        "replay_fastlane": {"enabled": False, "contains": [], "replay_ready_count": 0, "source": "missing"},
        "confirm_replay": {"enabled": False, "contains": [], "replay_ready_count": 0, "source": "missing"},
        "confirm_replay_contains": [],
        "search_quality": {
            "positive_lineage_count": 0,
            "zero_evidence_lineage_count": 0,
            "winner_proximate_positive_lineage_count": 0,
            "winner_proximate_positive_contains": [],
            "controlled_recovery_contains": [],
            "broad_search_allowed": True,
            "seed_generation_mode": "broad_search_micro",
            "controlled_recovery_active": False,
            "controlled_recovery_reason": "",
            "controlled_recovery_attempts_remaining": 0,
            "controlled_recovery_variants_cap": CONTROLLED_RECOVERY_VARIANTS_CAP,
            "controlled_broad_rearm_after_epoch": 0,
        },
        "positive_lineage_count": 0,
        "zero_evidence_lineage_count": 0,
        "winner_proximate_positive_lineage_count": 0,
        "winner_proximate_positive_contains": [],
        "controlled_recovery_contains": [],
        "broad_search_allowed": True,
        "seed_generation_mode": "broad_search_micro",
        "controlled_recovery_active": False,
        "controlled_recovery_reason": "",
        "controlled_recovery_attempts_remaining": 0,
        "controlled_recovery_variants_cap": CONTROLLED_RECOVERY_VARIANTS_CAP,
        "controlled_broad_rearm_after_epoch": 0,
        "lane_weights": {},
        "policy_overrides": {},
    }
    if not path.exists():
        return state
    state["exists"] = True
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        state.update({"status": "invalid_json", "reason": "invalid_json"})
        return state
    if not isinstance(payload, dict):
        state.update({"status": "invalid_payload", "reason": "invalid_payload"})
        return state

    winner = payload.get("winner_proximate", {})
    if not isinstance(winner, dict):
        winner = {}
    replay_fastlane = payload.get("replay_fastlane", {})
    if not isinstance(replay_fastlane, dict):
        replay_fastlane = {}
    legacy_confirm_replay = payload.get("confirm_replay", {})
    if not isinstance(legacy_confirm_replay, dict):
        legacy_confirm_replay = {}
    lane_weights = payload.get("lane_weights", {})
    if not isinstance(lane_weights, dict):
        lane_weights = {}
    policy_overrides = payload.get("policy_overrides", {})
    if not isinstance(policy_overrides, dict):
        policy_overrides = {}

    replay_contains = _to_tokens(replay_fastlane.get("contains", []))
    replay_source = "replay_fastlane"
    if not replay_contains:
        replay_contains = _to_tokens(legacy_confirm_replay.get("contains", []))
        if replay_contains:
            replay_source = "legacy_confirm_replay"
    if not replay_contains:
        replay_contains = _to_tokens(payload.get("confirm_replay_contains", []))
        if replay_contains:
            replay_source = "legacy_confirm_replay_contains"
    replay_ready_count = _as_int(replay_fastlane.get("replay_ready_count"), default=None, min_value=0)
    if replay_ready_count is None:
        replay_ready_count = _as_int(legacy_confirm_replay.get("replay_ready_count"), default=0, min_value=0) or 0
    replay_enabled = bool(
        replay_fastlane.get("enabled")
        or legacy_confirm_replay.get("enabled")
        or replay_contains
        or int(replay_ready_count) > 0
    )
    winner_contains = _to_tokens(winner.get("contains", []))
    search_quality = normalize_search_quality_state(payload, winner_proximate_contains=winner_contains)

    state.update(
        {
            "status": "ok",
            "reason": "ok",
            "active": bool(payload.get("active")),
            "hard_block_active": bool(payload.get("hard_block_active")),
            "hard_block_reason": str(payload.get("hard_block_reason") or "").strip(),
            "hard_block_until_epoch": int(_as_int(payload.get("hard_block_until_epoch"), default=0, min_value=0) or 0),
            "zero_coverage_seed_streak": int(_as_int(payload.get("zero_coverage_seed_streak"), default=0, min_value=0) or 0),
            "zero_coverage_seed_streak_reason": str(payload.get("zero_coverage_seed_streak_reason") or "").strip(),
            "preferred_contains": _to_tokens(payload.get("preferred_contains", [])),
            "cooldown_contains": _to_tokens(payload.get("cooldown_contains", [])),
            "preferred_operator_ids": _to_tokens(payload.get("preferred_operator_ids", [])),
            "cooldown_operator_ids": _to_tokens(payload.get("cooldown_operator_ids", [])),
            "winner_proximate": {
                "enabled": bool(winner.get("enabled")),
                "contains": list(winner_contains),
                "reason": str(winner.get("reason") or "").strip(),
            },
            "replay_fastlane": {
                "enabled": bool(replay_enabled),
                "contains": list(replay_contains),
                "replay_ready_count": int(replay_ready_count),
                "source": str(replay_source),
            },
            "confirm_replay": {
                "enabled": bool(replay_enabled),
                "contains": list(replay_contains),
                "replay_ready_count": int(replay_ready_count),
                "source": str(replay_source),
            },
            "confirm_replay_contains": list(replay_contains),
            "search_quality": dict(search_quality),
            "positive_lineage_count": int(search_quality.get("positive_lineage_count", 0) or 0),
            "zero_evidence_lineage_count": int(search_quality.get("zero_evidence_lineage_count", 0) or 0),
            "winner_proximate_positive_lineage_count": int(
                search_quality.get("winner_proximate_positive_lineage_count", 0) or 0
            ),
            "winner_proximate_positive_contains": _to_tokens(
                search_quality.get("winner_proximate_positive_contains", payload.get("winner_proximate_positive_contains", []))
            ),
            "controlled_recovery_contains": _to_tokens(
                search_quality.get("controlled_recovery_contains", payload.get("controlled_recovery_contains", []))
            ),
            "broad_search_allowed": bool(search_quality.get("broad_search_allowed")),
            "seed_generation_mode": str(search_quality.get("seed_generation_mode") or "broad_search_micro").strip(),
            "controlled_recovery_active": bool(search_quality.get("controlled_recovery_active")),
            "controlled_recovery_reason": str(search_quality.get("controlled_recovery_reason") or ""),
            "controlled_recovery_attempts_remaining": int(
                _as_int(search_quality.get("controlled_recovery_attempts_remaining"), default=0, min_value=0) or 0
            ),
            "controlled_recovery_variants_cap": int(
                _as_int(
                    search_quality.get("controlled_recovery_variants_cap"),
                    default=CONTROLLED_RECOVERY_VARIANTS_CAP,
                    min_value=1,
                )
                or CONTROLLED_RECOVERY_VARIANTS_CAP
            ),
            "controlled_broad_rearm_after_epoch": int(
                _as_int(search_quality.get("controlled_broad_rearm_after_epoch"), default=0, min_value=0) or 0
            ),
            "lane_weights": {str(k): (_as_int(v, default=0, min_value=0) or 0) for k, v in lane_weights.items()},
            "policy_overrides": dict(policy_overrides),
        }
    )
    return state


def _persist_yield_governor_state(path: Path, updates: dict[str, Any]) -> None:
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload.update(dict(updates))
    payload["ts"] = _utc_now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _entry_matches_controller(entry: dict[str, Any], controller_group: str) -> bool:
    wanted = str(controller_group or "").strip()
    if not wanted:
        return True
    group = str(entry.get("controller_group") or entry.get("group") or entry.get("target_group") or "").strip()
    if group:
        return group == wanted
    return True


def _derive_planner_focus(
    *,
    user_contains: list[str],
    directive_contains: list[str],
    directive_winner_contains: list[str],
    yield_governor: dict[str, Any],
    controller_group: str,
) -> dict[str, Any]:
    winner_contains = _merge_unique(
        list(directive_winner_contains),
        _to_tokens((yield_governor.get("winner_proximate") or {}).get("contains", [])),
    )
    preferred_any = _merge_unique(
        list(winner_contains),
        _merge_unique(
            list(directive_contains),
            list(yield_governor.get("preferred_contains", []) or []),
        ),
    )
    generic_contains = list(user_contains)
    anchor_source = "user_contains"
    if not generic_contains:
        if winner_contains:
            generic_contains = [winner_contains[0]]
            anchor_source = "winner_proximate_anchor"
        elif preferred_any:
            generic_contains = [preferred_any[0]]
            anchor_source = "preferred_anchor"
        elif str(controller_group or "").strip():
            generic_contains = [str(controller_group).strip()]
            anchor_source = "controller_group_fallback"
        else:
            anchor_source = "empty"
    return {
        "generic_contains": generic_contains,
        "winner_proximate_tokens": winner_contains,
        "preferred_any_contains": preferred_any,
        "anchor_source": anchor_source,
    }


def _extract_lineage_tokens(entry: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key in (
        "contains",
        "reject_contains",
        "lineage",
        "lineage_key",
        "lineage_keys",
        "reject_lineage",
        "reject_lineages",
        "focus_contains",
        "queue",
        "queue_path",
        "run_group",
    ):
        out = _merge_unique(out, _to_tokens(entry.get(key)))
    return out


def _detect_supported_planner_args(*, python_bin: str, app_root: Path) -> set[str]:
    wanted = {
        "--repair-mode",
        "--repair-max-neighbors",
        "--exclude-knob",
        "--winner-proximate-token",
        "--planner-policy-hash",
        "--planner-hash",
        "--seed-lane",
        "--seed-lane-index",
        "--parent-diversity-depth",
        "--parent-rotation-offset",
        "--confirm-replay-hint",
    }
    cmd = [
        str(Path(str(python_bin)).expanduser()),
        "scripts/optimization/evolve_next_batch.py",
        "--help",
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(app_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception:
        return set()
    text = "\n".join([str(result.stdout or ""), str(result.stderr or "")])
    return {flag for flag in wanted if flag in text}


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
    incident_path = state_dir / "incidents.jsonl"
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
        _append_log(
            incident_path,
            {
                "ts": _utc_now_iso(),
                "agent": "queue_seeder",
                "kind": "seed_failed",
                "human": "Queue seeder: planner script missing; seeding skipped.",
                "payload": payload,
            },
        )
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
            previous_state = _load_previous_state(state_path)
            controller_group = str(args.controller_group).strip() or "autonomous_queue_seeder"
            run_group_prefix = str(args.run_group_prefix).strip() or "autonomous_seed"
            orphan_path = aggregate_dir / ".autonomous" / "orphan_queues.csv"
            hygiene = _hygiene_seed_queues(
                aggregate_dir=aggregate_dir,
                app_root=app_root,
                run_group_prefix=run_group_prefix,
                orphan_path=orphan_path,
                incident_path=incident_path,
            )
            (
                total_pending_like,
                dispatchable_pending,
                executable_pending,
                runnable_queue_count,
                scanned,
                queue_files,
            ) = _summarize_queues(aggregate_dir)
            ready_buffer_state_path = _resolve_under_root(str(args.ready_buffer_state_path), app_root)
            ready_buffer_state = _load_ready_queue_buffer(ready_buffer_state_path)
            seed_needed, trigger_reasons = _evaluate_seed_needed(
                executable_pending=executable_pending,
                pending_threshold=int(args.pending_threshold),
                runnable_queue_count=runnable_queue_count,
                ready_buffer_state=ready_buffer_state,
            )
            candidate_pool_status = "ready"
            if int(ready_buffer_state.get("ready_depth", 0) or 0) <= 0:
                candidate_pool_status = "empty_error" if dispatchable_pending > 0 else "empty_expected"
            snapshot = {
                "ts": _utc_now_iso(),
                "status": "skipped" if not seed_needed else "active",
                "total_pending": total_pending_like,
                "dispatchable_pending": dispatchable_pending,
                "executable_pending": executable_pending,
                "runnable_queue_count": runnable_queue_count,
                "scanned_queue_groups": scanned,
                "queue_files_total": len(queue_files),
                "queue_files_sample": [str(path) for path in queue_files[:120]],
                "pending_threshold": int(args.pending_threshold),
                "hygiene": dict(hygiene),
                "ready_buffer_state_path": _safe_rel(ready_buffer_state_path, app_root),
                "candidate_pool_status": str(candidate_pool_status),
                "ready_queue_buffer": {
                    "exists": bool(ready_buffer_state.get("exists")),
                    "status": str(ready_buffer_state.get("status") or ""),
                    "reason": str(ready_buffer_state.get("reason") or ""),
                    "target_depth": ready_buffer_state.get("target_depth"),
                    "refill_threshold": ready_buffer_state.get("refill_threshold"),
                    "effective_refill_threshold": ready_buffer_state.get("effective_refill_threshold"),
                    "ready_depth": ready_buffer_state.get("ready_depth"),
                    "ready_depth_total": ready_buffer_state.get("ready_depth_total"),
                    "coverage_verified_ready_count": ready_buffer_state.get("coverage_verified_ready_count"),
                    "seed_needed": bool(ready_buffer_state.get("seed_needed")),
                },
                "yield_governor_state_path": "",
                "yield_governor": {},
                "trigger": None,
                "trigger_reasons": [],
            }
            if not seed_needed:
                _emit_state(state_path, snapshot)
                _append_log(log_path, snapshot)
                return 0

            snapshot["trigger_reasons"] = list(trigger_reasons)
            snapshot["trigger"] = "+".join(trigger_reasons)

            base_config = str(args.base_config).strip() or "configs/prod_final_budget1000.yaml"
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_group = f"{run_group_prefix}_{timestamp}"
            run_index_path = _resolve_under_root(str(args.run_index), app_root)
            run_index_groups_all = _load_run_index_groups(run_index_path)
            quarantine_by_group = _load_recent_quarantine_by_group(
                aggregate_dir / ".autonomous" / "deterministic_quarantine.json"
            )
            window_coverage = _assess_window_data_coverage(app_root / "data_downloaded", list(args.window or []))
            snapshot["window_coverage"] = window_coverage
            if not bool(window_coverage.get("ok")):
                snapshot.update(
                    {
                        "status": "skipped",
                        "status_detail": "preflight_data_coverage",
                        "reason": f"window_coverage:{window_coverage.get('reason')}",
                        "human": "Queue seeder skipped: requested OOS window is not fully covered by local data.",
                    }
                )
                _emit_state(state_path, snapshot)
                _append_log(log_path, snapshot)
                return 0
            recent_seed_quality = _assess_recent_seed_quality(
                app_root=app_root,
                run_group_prefix=str(args.run_group_prefix).strip() or "autonomous_seed",
                run_index_path=run_index_path,
            )
            snapshot["recent_seed_quality"] = recent_seed_quality

            directive_path = _resolve_under_root(str(args.directive_path), app_root)
            directive = _load_directive(directive_path)
            blacklist_path = _resolve_under_root(str(args.blacklist_path), app_root)
            search_blacklist = _load_search_blacklist(blacklist_path)
            impossibility_pruner_path = aggregate_dir / ".autonomous" / "deterministic_quarantine.json"
            impossibility_pruner = _load_impossibility_pruner(impossibility_pruner_path)
            gate_surrogate_path = _resolve_under_root(str(args.gate_surrogate_state_path), app_root)
            gate_surrogate = _load_gate_surrogate_state(gate_surrogate_path)
            yield_governor_path = _resolve_under_root(str(args.yield_governor_state_path), app_root)
            yield_governor = _load_yield_governor_state(yield_governor_path)
            current_epoch = int(datetime.now(timezone.utc).timestamp())
            yield_updates: dict[str, Any] = {
                "zero_coverage_seed_streak": int(recent_seed_quality.get("zero_coverage_seed_streak", 0) or 0),
                "zero_coverage_seed_streak_reason": str(recent_seed_quality.get("hard_block_reason") or "").strip(),
            }
            existing_hard_block_reason = str(yield_governor.get("hard_block_reason") or "").strip()
            if bool(recent_seed_quality.get("hard_block_recommended")) and (
                not bool(yield_governor.get("hard_block_active"))
                or existing_hard_block_reason in {"", "zero_coverage_seed_streak"}
            ):
                yield_updates.update(
                    {
                        "hard_block_active": True,
                        "hard_block_reason": str(recent_seed_quality.get("hard_block_reason") or "zero_coverage_seed_streak"),
                        "hard_block_until_epoch": current_epoch + 21600,
                    }
                )
            elif existing_hard_block_reason == "zero_coverage_seed_streak":
                yield_updates.update(
                    {
                        "hard_block_active": False,
                        "hard_block_reason": "",
                        "hard_block_until_epoch": 0,
                        "controlled_recovery_active": False,
                        "controlled_recovery_reason": "",
                        "controlled_recovery_attempts_remaining": 0,
                        "controlled_recovery_variants_cap": CONTROLLED_RECOVERY_VARIANTS_CAP,
                        "controlled_broad_rearm_after_epoch": 0,
                    }
                )
            _persist_yield_governor_state(yield_governor_path, yield_updates)
            yield_governor = _load_yield_governor_state(yield_governor_path)
            hard_block_until_epoch = int(yield_governor.get("hard_block_until_epoch") or 0)
            hard_block_active = bool(yield_governor.get("hard_block_active")) and (
                hard_block_until_epoch <= 0 or hard_block_until_epoch > current_epoch
            )
            controlled_recovery_state = build_controlled_recovery_state(
                hard_block_active=hard_block_active,
                hard_block_reason=str(yield_governor.get("hard_block_reason") or ""),
                winner_proximate_positive_contains=yield_governor.get("winner_proximate_positive_contains", []),
                controlled_recovery_contains=yield_governor.get("controlled_recovery_contains", []),
                existing_state=yield_governor,
            )
            controlled_recovery_updates = {
                "winner_proximate_positive_contains": list(
                    controlled_recovery_state.get("winner_proximate_positive_contains", []) or []
                ),
                "controlled_recovery_contains": list(
                    controlled_recovery_state.get("controlled_recovery_contains", []) or []
                ),
                "controlled_recovery_active": bool(controlled_recovery_state.get("controlled_recovery_active")),
                "controlled_recovery_reason": str(controlled_recovery_state.get("controlled_recovery_reason") or ""),
                "controlled_recovery_attempts_remaining": int(
                    controlled_recovery_state.get("controlled_recovery_attempts_remaining", 0) or 0
                ),
                "controlled_recovery_variants_cap": int(
                    controlled_recovery_state.get("controlled_recovery_variants_cap", CONTROLLED_RECOVERY_VARIANTS_CAP)
                    or CONTROLLED_RECOVERY_VARIANTS_CAP
                ),
                "controlled_broad_rearm_after_epoch": int(
                    controlled_recovery_state.get("controlled_broad_rearm_after_epoch", 0) or 0
                ),
            }
            if any(yield_governor.get(key) != value for key, value in controlled_recovery_updates.items()):
                _persist_yield_governor_state(yield_governor_path, controlled_recovery_updates)
                yield_governor = _load_yield_governor_state(yield_governor_path)
                controlled_recovery_state = dict(controlled_recovery_updates)
            controlled_recovery_active = bool(controlled_recovery_state.get("controlled_recovery_active"))
            controlled_recovery_attempts_remaining = int(
                controlled_recovery_state.get("controlled_recovery_attempts_remaining", 0) or 0
            )
            controlled_recovery_variants_cap = int(
                controlled_recovery_state.get("controlled_recovery_variants_cap", CONTROLLED_RECOVERY_VARIANTS_CAP)
                or CONTROLLED_RECOVERY_VARIANTS_CAP
            )
            controlled_recovery_reason = str(controlled_recovery_state.get("controlled_recovery_reason") or "")
            controlled_recovery_tokens = _to_tokens(
                controlled_recovery_state.get("controlled_recovery_contains")
                or controlled_recovery_state.get("winner_proximate_positive_contains", [])
            )
            controlled_recovery_backlog_ok = controlled_recovery_backlog_ready(
                dispatchable_pending=dispatchable_pending,
                candidate_pool_status=candidate_pool_status,
            )
            controlled_recovery_triggered = False
            controlled_recovery_lineage_anchor = ""
            controlled_recovery_queue_reason = ""
            zero_yield_signal = _recent_zero_yield_signal(app_root / "artifacts" / "optimization_state" / "rank_results")
            hard_fail_risk_policy = gate_surrogate.get("hard_fail_risk_policy", {})
            if not isinstance(hard_fail_risk_policy, dict):
                hard_fail_risk_policy = {}
            controlled_broad = _controlled_broad_recovery_from_directive(directive)
            controlled_broad_active = bool(controlled_broad.get("enabled"))
            controlled_broad_reason = str(controlled_broad.get("reason") or "").strip()
            controlled_broad_contains = _to_tokens(controlled_broad.get("contains", []))
            controlled_broad_num_variants = int(
                _as_int(controlled_broad.get("num_variants"), default=CONTROLLED_BROAD_NUM_VARIANTS, min_value=1)
                or CONTROLLED_BROAD_NUM_VARIANTS
            )
            controlled_broad_num_variants_floor = int(
                _as_int(
                    controlled_broad.get("num_variants_floor"),
                    default=CONTROLLED_BROAD_NUM_VARIANTS_FLOOR,
                    min_value=1,
                )
                or CONTROLLED_BROAD_NUM_VARIANTS_FLOOR
            )
            controlled_broad_max_changed_keys = int(
                _as_int(
                    controlled_broad.get("max_changed_keys"),
                    default=CONTROLLED_BROAD_MAX_CHANGED_KEYS,
                    min_value=1,
                )
                or CONTROLLED_BROAD_MAX_CHANGED_KEYS
            )
            controlled_broad_dedupe_distance = float(
                _as_float(
                    controlled_broad.get("dedupe_distance"),
                    default=CONTROLLED_BROAD_DEDUPE_DISTANCE,
                    min_value=0.0,
                )
                or CONTROLLED_BROAD_DEDUPE_DISTANCE
            )

            directive_contains = [str(token).strip() for token in list(directive.get("contains", []) or []) if str(token).strip()]
            directive_winner = directive.get("winner_proximate", {})
            if not isinstance(directive_winner, dict):
                directive_winner = {}
            planner_focus = _derive_planner_focus(
                user_contains=[token.strip() for token in list(args.contains or []) if token.strip()],
                directive_contains=directive_contains,
                directive_winner_contains=_to_tokens(directive_winner.get("contains", [])),
                yield_governor=yield_governor,
                controller_group=controller_group,
            )
            contains = list(planner_focus["generic_contains"])
            winner_proximate_tokens = list(planner_focus["winner_proximate_tokens"])
            preferred_any_contains = list(planner_focus["preferred_any_contains"])
            if controlled_broad_active and controlled_broad_contains:
                contains = list(controlled_broad_contains)
                preferred_any_contains = _merge_unique(list(controlled_broad_contains), list(preferred_any_contains))
                winner_proximate_tokens = []

            effective_num_variants = int(args.num_variants)
            effective_num_variants_floor = max(1, int(args.num_variants_floor))
            effective_max_changed_keys = int(args.max_changed_keys)
            effective_dedupe_distance = float(args.dedupe_distance)
            effective_policy_scale = str(args.policy_scale)
            micro_caps = micro_broad_search_caps()
            effective_repair_mode = bool(args.repair_mode)
            effective_repair_max_neighbors = max(1, int(args.repair_max_neighbors))
            effective_include_stress = bool(args.include_stress)
            effective_exclude_knobs = [token.strip() for token in list(args.exclude_knob or []) if token and token.strip()]
            extra_cli_args = directive.get("extra_cli_args", {})
            if not isinstance(extra_cli_args, dict):
                extra_cli_args = {}
            quality_variant_cap: int | None = None
            quality_actions: list[str] = []
            selected_lane = "broad_search"
            selected_lane_index = 0
            lane_streak = 1
            token_rotation = 0
            parent_rotation_offset = 0
            confirm_replay_hints: list[str] = []
            lane_selection_payload: dict[str, Any] = {}
            parent_diversity_depth = 4

            try:
                if "num_variants" in directive:
                    effective_num_variants = max(
                        int(effective_num_variants),
                        max(1, int(float(directive.get("num_variants")))),
                    )
            except Exception:
                pass
            try:
                if "max_changed_keys" in directive:
                    effective_max_changed_keys = max(1, int(float(directive.get("max_changed_keys"))))
            except Exception:
                pass
            try:
                if "dedupe_distance" in directive:
                    effective_dedupe_distance = max(0.0, float(directive.get("dedupe_distance")))
            except Exception:
                pass
            directive_policy = str(directive.get("policy_scale", "")).strip()
            if directive_policy in {"auto", "micro", "macro"}:
                effective_policy_scale = directive_policy
            yield_policy_overrides = yield_governor.get("policy_overrides", {})
            if not isinstance(yield_policy_overrides, dict):
                yield_policy_overrides = {}
            y_policy_scale = str(yield_policy_overrides.get("policy_scale", "")).strip()
            if y_policy_scale in {"auto", "micro", "macro"}:
                effective_policy_scale = y_policy_scale
            try:
                if "num_variants_cap" in yield_policy_overrides:
                    effective_num_variants = min(
                        int(effective_num_variants),
                        max(1, int(float(yield_policy_overrides.get("num_variants_cap")))),
                    )
            except Exception:
                pass
            if bool(zero_yield_signal.get("active")) and not controlled_broad_active:
                repair_variant_cap = max(4, min(int(effective_num_variants_floor), 8))
                effective_repair_mode = True
                effective_policy_scale = "micro"
                effective_dedupe_distance = max(float(effective_dedupe_distance), 0.08)
                effective_num_variants_floor = min(int(effective_num_variants_floor), int(repair_variant_cap))
                effective_num_variants = min(int(effective_num_variants), int(repair_variant_cap))
                effective_repair_max_neighbors = min(int(effective_repair_max_neighbors), 8)
                quality_actions.append("recent_zero_yield_signal")
            if bool(recent_seed_quality.get("repair_mode")) and not controlled_broad_active:
                effective_repair_mode = True
                effective_policy_scale = "micro"
                effective_dedupe_distance = max(float(effective_dedupe_distance), 0.08)
                effective_max_changed_keys = min(int(effective_max_changed_keys), 3)
                effective_repair_max_neighbors = min(int(effective_repair_max_neighbors), 8)
                quality_variant_cap = max(
                    4,
                    min(int(effective_num_variants_floor), max(4, int(effective_num_variants) // 2)),
                )
                effective_num_variants = min(int(effective_num_variants), int(quality_variant_cap))
                quality_actions.append("recent_seed_quality_repair")
            if bool(recent_seed_quality.get("backlog_suppress")) and not controlled_broad_active:
                effective_include_stress = False
                suppress_cap = max(4, min(int(effective_num_variants_floor), 12))
                quality_variant_cap = suppress_cap if quality_variant_cap is None else min(int(quality_variant_cap), suppress_cap)
                effective_num_variants = min(int(effective_num_variants), int(quality_variant_cap))
                quality_actions.append("recent_seed_quality_backlog_suppress")

            if controlled_broad_active:
                effective_num_variants = max(int(effective_num_variants), int(controlled_broad_num_variants))
                effective_num_variants_floor = max(
                    int(effective_num_variants_floor),
                    int(controlled_broad_num_variants_floor),
                )
                effective_max_changed_keys = max(
                    int(effective_max_changed_keys),
                    int(controlled_broad_max_changed_keys),
                )
                effective_dedupe_distance = max(
                    float(effective_dedupe_distance),
                    float(controlled_broad_dedupe_distance),
                )
                quality_actions.append("controlled_broad_min_batch")

            directive_pruner = directive.get("impossibility_pruner", {})
            if isinstance(directive_pruner, dict) and bool(directive_pruner.get("enabled")):
                try:
                    effective_max_changed_keys = min(
                        int(effective_max_changed_keys),
                        int(directive_pruner.get("max_changed_keys_cap", micro_caps["max_changed_keys_cap"])),
                    )
                except Exception:
                    pass
                try:
                    effective_dedupe_distance = max(
                        float(effective_dedupe_distance),
                        float(directive_pruner.get("dedupe_distance_floor", micro_caps["dedupe_distance_floor"])),
                    )
                except Exception:
                    pass
                try:
                    effective_num_variants = min(
                        int(effective_num_variants),
                        int(directive_pruner.get("num_variants_cap", micro_caps["num_variants_cap"])),
                    )
                except Exception:
                    pass
                pscale = str(directive_pruner.get("policy_scale", micro_caps["policy_scale"])).strip()
                if pscale in {"auto", "micro", "macro"}:
                    effective_policy_scale = pscale

            if bool(search_blacklist.get("active")):
                try:
                    effective_max_changed_keys = min(
                        int(effective_max_changed_keys),
                        int(search_blacklist.get("max_changed_keys_cap", effective_max_changed_keys)),
                    )
                except Exception:
                    pass
                try:
                    effective_dedupe_distance = max(
                        float(effective_dedupe_distance),
                        float(search_blacklist.get("dedupe_distance_floor", effective_dedupe_distance)),
                    )
                except Exception:
                    pass
                try:
                    effective_num_variants = min(
                        int(effective_num_variants),
                        int(search_blacklist.get("num_variants_cap", effective_num_variants)),
                    )
                except Exception:
                    pass
                pscale = str(search_blacklist.get("policy_scale", "")).strip()
                if pscale in {"auto", "micro", "macro"}:
                    effective_policy_scale = pscale

            if bool(impossibility_pruner.get("active")):
                try:
                    effective_max_changed_keys = min(
                        int(effective_max_changed_keys),
                        int(impossibility_pruner.get("max_changed_keys_cap", 2)),
                    )
                except Exception:
                    pass
                try:
                    effective_dedupe_distance = max(
                        float(effective_dedupe_distance),
                        float(impossibility_pruner.get("dedupe_distance_floor", 0.08)),
                    )
                except Exception:
                    pass
                try:
                    effective_num_variants = min(
                        int(effective_num_variants),
                        int(impossibility_pruner.get("num_variants_cap", 8)),
                    )
                except Exception:
                    pass
                pscale = str(impossibility_pruner.get("policy_scale", "")).strip()
                if pscale in {"auto", "micro", "macro"}:
                    effective_policy_scale = pscale

            surrogate_queue_decisions = list(gate_surrogate.get("queue_decisions", []) or [])
            surrogate_reject_lineages: list[str] = []
            surrogate_reject_reasons: list[str] = []
            surrogate_refine_reasons: list[str] = []
            surrogate_matched_decisions: list[dict[str, Any]] = []
            surrogate_refine_active = False
            surrogate_max_changed_keys_cap = 2
            surrogate_dedupe_distance_floor = 0.08
            surrogate_num_variants_cap: int | None = None
            surrogate_policy_scale: str | None = "micro"
            surrogate_repair_max_neighbors: int | None = None
            surrogate_exclude_knobs: list[str] = []

            for entry in surrogate_queue_decisions:
                if not isinstance(entry, dict) or not _entry_matches_controller(entry, controller_group):
                    continue
                decision = str(entry.get("queue_decision") or entry.get("decision") or entry.get("action") or "").strip().lower()
                if decision not in {"reject", "refine"}:
                    continue
                reason = str(entry.get("reason") or entry.get("decision_reason") or "").strip()
                source = str(entry.get("source") or "").strip() or "queue_decisions"
                surrogate_matched_decisions.append(
                    {
                        "decision": decision,
                        "reason": reason,
                        "source": source,
                    }
                )

                if decision == "reject":
                    surrogate_reject_lineages = _merge_unique(surrogate_reject_lineages, _extract_lineage_tokens(entry))
                    if reason:
                        surrogate_reject_reasons.append(reason)
                    continue

                surrogate_refine_active = True
                if reason:
                    surrogate_refine_reasons.append(reason)

                conservative_knobs = entry.get("conservative_knobs", {})
                if not isinstance(conservative_knobs, dict):
                    conservative_knobs = {}

                max_keys_cap = _as_int(
                    entry.get("max_changed_keys_cap"),
                    default=_as_int(conservative_knobs.get("max_changed_keys_cap"), default=None, min_value=1),
                    min_value=1,
                )
                if max_keys_cap is not None:
                    surrogate_max_changed_keys_cap = min(int(surrogate_max_changed_keys_cap), int(max_keys_cap))

                dedupe_floor = _as_float(
                    entry.get("dedupe_distance_floor"),
                    default=_as_float(conservative_knobs.get("dedupe_distance_floor"), default=None, min_value=0.0),
                    min_value=0.0,
                )
                if dedupe_floor is not None:
                    surrogate_dedupe_distance_floor = max(float(surrogate_dedupe_distance_floor), float(dedupe_floor))

                num_variants_cap = _as_int(
                    entry.get("num_variants_cap"),
                    default=_as_int(conservative_knobs.get("num_variants_cap"), default=None, min_value=1),
                    min_value=1,
                )
                if num_variants_cap is not None:
                    if surrogate_num_variants_cap is None:
                        surrogate_num_variants_cap = int(num_variants_cap)
                    else:
                        surrogate_num_variants_cap = min(int(surrogate_num_variants_cap), int(num_variants_cap))

                pscale = str(entry.get("policy_scale") or conservative_knobs.get("policy_scale") or "").strip()
                if pscale in {"auto", "micro", "macro"}:
                    surrogate_policy_scale = pscale

                neighbors_cap = _as_int(
                    entry.get("repair_max_neighbors"),
                    default=_as_int(conservative_knobs.get("repair_max_neighbors"), default=None, min_value=1),
                    min_value=1,
                )
                if neighbors_cap is not None:
                    if surrogate_repair_max_neighbors is None:
                        surrogate_repair_max_neighbors = int(neighbors_cap)
                    else:
                        surrogate_repair_max_neighbors = min(int(surrogate_repair_max_neighbors), int(neighbors_cap))

                for key in ("exclude_knob", "exclude_knobs"):
                    surrogate_exclude_knobs = _merge_unique(surrogate_exclude_knobs, _to_tokens(entry.get(key)))
                    surrogate_exclude_knobs = _merge_unique(surrogate_exclude_knobs, _to_tokens(conservative_knobs.get(key)))

            contains_before_surrogate = list(contains)
            winner_tokens_before_surrogate = list(winner_proximate_tokens)
            preferred_any_before_surrogate = list(preferred_any_contains)
            if surrogate_reject_lineages:
                contains = [token for token in contains if token not in set(surrogate_reject_lineages)]
                winner_proximate_tokens = [
                    token for token in winner_proximate_tokens if token not in set(surrogate_reject_lineages)
                ]
                preferred_any_contains = [
                    token for token in preferred_any_contains if token not in set(surrogate_reject_lineages)
                ]
                if not contains:
                    if controller_group and controller_group not in set(surrogate_reject_lineages):
                        contains = [controller_group]
                    else:
                        contains = ["__surrogate_unfocused__"]

            if surrogate_refine_active:
                effective_repair_mode = True
                effective_max_changed_keys = min(int(effective_max_changed_keys), int(surrogate_max_changed_keys_cap))
                effective_dedupe_distance = max(float(effective_dedupe_distance), float(surrogate_dedupe_distance_floor))
                if surrogate_num_variants_cap is not None:
                    effective_num_variants = min(int(effective_num_variants), int(surrogate_num_variants_cap))
                if surrogate_policy_scale in {"auto", "micro", "macro"}:
                    effective_policy_scale = str(surrogate_policy_scale)
                if surrogate_repair_max_neighbors is None:
                    effective_repair_max_neighbors = min(int(effective_repair_max_neighbors), 8)
                else:
                    effective_repair_max_neighbors = min(int(effective_repair_max_neighbors), int(surrogate_repair_max_neighbors))

            effective_exclude_knobs = _merge_unique(effective_exclude_knobs, surrogate_exclude_knobs)
            if controlled_broad_active:
                effective_policy_scale = "micro"
                effective_repair_mode = False
                effective_num_variants_floor = min(
                    int(controlled_broad_num_variants),
                    max(1, int(controlled_broad_num_variants_floor)),
                )
                effective_num_variants = min(int(effective_num_variants), int(controlled_broad_num_variants))
                effective_max_changed_keys = min(
                    int(effective_max_changed_keys),
                    int(controlled_broad_max_changed_keys),
                )
                effective_dedupe_distance = max(
                    float(effective_dedupe_distance),
                    float(controlled_broad_dedupe_distance),
                )
                effective_include_stress = True
            effective_num_variants = max(int(effective_num_variants), int(effective_num_variants_floor))
            if quality_variant_cap is not None:
                effective_num_variants = min(int(effective_num_variants), int(quality_variant_cap))

            lane_selection_yield_governor = dict(yield_governor)
            lane_selection_winner_tokens = list(winner_proximate_tokens)
            lane_selection_preferred_tokens = list(preferred_any_contains)
            lane_selection_generic_tokens = list(contains)
            lane_selection_replay_tokens = _to_tokens((directive.get("replay_fastlane") or {}).get("contains", []))
            if controlled_broad_active:
                lane_selection_winner_tokens = []
                lane_selection_replay_tokens = []
                lane_selection_preferred_tokens = list(controlled_broad_contains or preferred_any_contains)
                lane_selection_generic_tokens = list(controlled_broad_contains or contains)
                lane_selection_yield_governor["replay_fastlane"] = {"enabled": False, "contains": [], "replay_ready_count": 0}
                lane_selection_yield_governor["confirm_replay"] = {"enabled": False, "contains": [], "replay_ready_count": 0}
                lane_selection_yield_governor["confirm_replay_contains"] = []
                lane_selection_yield_governor["lane_weights"] = {
                    "winner_proximate": 0,
                    "confirm_replay": 0,
                    "broad_search": 100,
                }
                lane_selection_yield_governor["search_quality"] = {
                    **dict(yield_governor.get("search_quality") or {}),
                    "broad_search_allowed": True,
                    "seed_generation_mode": "controlled_broad_micro",
                    "controlled_broad_active": True,
                    "controlled_broad_reason": str(controlled_broad_reason or ""),
                    "controlled_broad_contains": list(controlled_broad_contains),
                }
            if controlled_recovery_active and controlled_recovery_attempts_remaining > 0 and controlled_recovery_backlog_ok:
                lane_selection_winner_tokens = list(controlled_recovery_tokens)
                lane_selection_preferred_tokens = list(controlled_recovery_tokens)
                lane_selection_generic_tokens = list(controlled_recovery_tokens or contains)
                lane_selection_replay_tokens = []
                lane_selection_yield_governor["replay_fastlane"] = {"enabled": False, "contains": [], "replay_ready_count": 0}
                lane_selection_yield_governor["confirm_replay"] = {"enabled": False, "contains": [], "replay_ready_count": 0}
                lane_selection_yield_governor["confirm_replay_contains"] = []
                lane_selection_yield_governor["lane_weights"] = {
                    "winner_proximate": 100,
                    "confirm_replay": 0,
                    "broad_search": 0,
                }
                lane_selection_yield_governor["search_quality"] = {
                    **dict(yield_governor.get("search_quality") or {}),
                    "broad_search_allowed": False,
                    "seed_generation_mode": "winner_proximate_only",
                }
            lane_selection_payload = _select_seed_lane(
                winner_proximate_tokens=list(lane_selection_winner_tokens),
                preferred_any_contains=list(lane_selection_preferred_tokens),
                generic_contains=list(lane_selection_generic_tokens),
                directive_replay_fastlane_tokens=lane_selection_replay_tokens,
                yield_governor=lane_selection_yield_governor,
                previous_state=previous_state,
            )
            selected_lane = str(lane_selection_payload.get("selected_lane") or "broad_search").strip() or "broad_search"
            selected_lane_index = int(lane_selection_payload.get("selected_index") or 0)
            lane_streak = int(lane_selection_payload.get("lane_streak") or 1)
            token_rotation = int(lane_selection_payload.get("token_rotation") or 0)
            parent_rotation_offset = int(lane_selection_payload.get("parent_rotation_offset") or 0)
            contains = list(lane_selection_payload.get("contains") or contains)
            winner_proximate_tokens = list(
                lane_selection_payload.get("winner_proximate_tokens") or winner_proximate_tokens
            )
            confirm_replay_hints = list(lane_selection_payload.get("confirm_replay_hints") or [])
            confirm_replay_source = str(lane_selection_payload.get("confirm_replay_source") or "").strip()
            search_quality_state = dict(lane_selection_payload.get("search_quality") or {})
            if controlled_recovery_active and controlled_recovery_attempts_remaining > 0 and controlled_recovery_backlog_ok:
                selected_lane = "winner_proximate"
                selected_lane_index = 0
                contains = list(contains[:1] or lane_selection_winner_tokens[:1] or ["autonomous_queue_seeder"])
                controlled_recovery_lineage_anchor = str((contains or [""])[0] or "").strip()
                controlled_recovery_queue_reason = str(yield_governor.get("hard_block_reason") or "zero_coverage_seed_streak")
                winner_proximate_tokens = [controlled_recovery_lineage_anchor] if controlled_recovery_lineage_anchor else []
                confirm_replay_hints = []
                confirm_replay_source = ""
                controlled_recovery_triggered = bool(controlled_recovery_lineage_anchor)
                effective_policy_scale = "micro"
                effective_num_variants_floor = int(controlled_recovery_variants_cap)
                effective_num_variants = int(controlled_recovery_variants_cap)
                effective_max_changed_keys = min(int(effective_max_changed_keys), 3)
                effective_dedupe_distance = max(float(effective_dedupe_distance), 0.08)
                effective_repair_mode = True
                effective_repair_max_neighbors = min(int(effective_repair_max_neighbors), 8)
                effective_include_stress = False
                quality_variant_cap = int(controlled_recovery_variants_cap)
                quality_actions.append("controlled_recovery")
                search_quality_state = {
                    **dict(search_quality_state),
                    "controlled_recovery_active": True,
                    "controlled_recovery_reason": str(controlled_recovery_reason or CONTROLLED_RECOVERY_REASON),
                    "controlled_recovery_attempts_remaining": int(controlled_recovery_attempts_remaining),
                    "controlled_recovery_variants_cap": int(controlled_recovery_variants_cap),
                    "winner_proximate_positive_contains": list(controlled_recovery_tokens),
                    "controlled_recovery_contains": list(controlled_recovery_tokens),
                    "broad_search_allowed": False,
                    "seed_generation_mode": "winner_proximate_only",
                }
            if selected_lane == "broad_search":
                effective_policy_scale = "micro"
                effective_max_changed_keys = min(
                    int(effective_max_changed_keys),
                    int(micro_caps["max_changed_keys_cap"]),
                )
                effective_dedupe_distance = max(
                    float(effective_dedupe_distance),
                    float(micro_caps["dedupe_distance_floor"]),
                )
                effective_num_variants = min(
                    int(effective_num_variants),
                    int(micro_caps["num_variants_cap"]),
                )
            if selected_lane == "winner_proximate":
                parent_diversity_depth = 5
            elif selected_lane == "confirm_replay":
                parent_diversity_depth = 6
            if confirm_replay_hints and selected_lane != "confirm_replay":
                confirm_replay_hints = confirm_replay_hints[:2]
            snapshot["controlled_recovery_triggered"] = bool(controlled_recovery_triggered)
            snapshot["controlled_recovery_lineage_anchor"] = str(controlled_recovery_lineage_anchor or "")

            policy_fingerprint_payload = {
                "controller_group": controller_group,
                "seed_lane": selected_lane,
                "seed_lane_index": int(selected_lane_index),
                "contains": list(contains),
                "winner_proximate_tokens": list(winner_proximate_tokens),
                "confirm_replay_hints": list(confirm_replay_hints),
                "confirm_replay_source": str(confirm_replay_source),
                "policy_scale": str(effective_policy_scale),
                "num_variants": int(effective_num_variants),
                "num_variants_floor": int(effective_num_variants_floor),
                "max_changed_keys": int(effective_max_changed_keys),
                "dedupe_distance": float(effective_dedupe_distance),
                "include_stress": bool(effective_include_stress),
                "repair_mode": bool(effective_repair_mode),
                "repair_max_neighbors": int(effective_repair_max_neighbors),
                "exclude_knobs": list(effective_exclude_knobs),
                "parent_rotation_offset": int(parent_rotation_offset),
                "parent_diversity_depth": int(parent_diversity_depth),
                "search_quality": dict(search_quality_state),
                "broad_search_allowed": bool(search_quality_state.get("broad_search_allowed")),
                "seed_generation_mode": str(search_quality_state.get("seed_generation_mode") or ""),
                "controlled_recovery_active": bool(controlled_recovery_triggered),
                "controlled_recovery_reason": str(controlled_recovery_reason or ""),
                "controlled_recovery_lineage_anchor": str(controlled_recovery_lineage_anchor or ""),
                "controlled_broad_active": bool(controlled_broad_active),
                "controlled_broad_reason": str(controlled_broad_reason or ""),
                "controlled_broad_contains": list(controlled_broad_contains),
            }
            planner_policy_hash = _stable_hash(policy_fingerprint_payload, prefix="policy")

            snapshot["directive"] = {
                "path": _safe_rel(directive_path, app_root),
                "exists": directive_path.exists(),
                "mode": str(directive.get("mode", "")),
                "dominant_reason": str(directive.get("dominant_reason", "")),
                "contains": contains,
                "winner_proximate_tokens": winner_proximate_tokens,
                "preferred_any_contains": preferred_any_contains,
                "focus_anchor_source": str(planner_focus.get("anchor_source") or ""),
                "policy_scale": effective_policy_scale,
                "num_variants": effective_num_variants,
                "num_variants_floor": effective_num_variants_floor,
                "max_changed_keys": effective_max_changed_keys,
                "dedupe_distance": effective_dedupe_distance,
                "include_stress": bool(effective_include_stress),
                "repair_mode": bool(effective_repair_mode),
                "repair_max_neighbors": int(effective_repair_max_neighbors),
                "exclude_knobs": list(effective_exclude_knobs),
                "extra_cli_args": extra_cli_args,
                "controlled_recovery_active": bool(controlled_recovery_triggered),
                "controlled_recovery_reason": str(controlled_recovery_reason or ""),
                "controlled_recovery_lineage_anchor": str(controlled_recovery_lineage_anchor or ""),
                "controlled_broad_active": bool(controlled_broad_active),
                "controlled_broad_reason": str(controlled_broad_reason or ""),
                "controlled_broad_contains": list(controlled_broad_contains),
            }
            snapshot["impossibility_pruner"] = {
                "path": _safe_rel(impossibility_pruner_path, app_root),
                "directive_enabled": bool(isinstance(directive_pruner, dict) and directive_pruner.get("enabled")),
                "directive_reason": str(directive_pruner.get("reason", "")) if isinstance(directive_pruner, dict) else "",
                "active": bool(impossibility_pruner.get("active")),
                "reason": str(impossibility_pruner.get("reason", "")),
                "dominant_code": str(impossibility_pruner.get("dominant_code", "")),
                "dominant_ratio": float(impossibility_pruner.get("dominant_ratio", 0.0) or 0.0),
                "total": int(impossibility_pruner.get("total", 0) or 0),
            }
            snapshot["search_blacklist"] = {
                "path": _safe_rel(blacklist_path, app_root),
                "active": bool(search_blacklist.get("active")),
                "reason": str(search_blacklist.get("reason", "")),
                "dominant_code": str(search_blacklist.get("dominant_code", "")),
                "dominant_ratio": float(search_blacklist.get("dominant_ratio", 0.0) or 0.0),
                "total": int(search_blacklist.get("total", 0) or 0),
                "blocked_evo_uids_count": int(search_blacklist.get("blocked_evo_uids_count", 0) or 0),
            }
            snapshot["gate_surrogate"] = {
                "path": _safe_rel(gate_surrogate_path, app_root),
                "exists": bool(gate_surrogate.get("exists")),
                "status": str(gate_surrogate.get("status") or ""),
                "hard_fail_risk_policy": {
                    "decision": str(
                        hard_fail_risk_policy.get("queue_decision")
                        or hard_fail_risk_policy.get("decision")
                        or hard_fail_risk_policy.get("action")
                        or ""
                    ).strip(),
                    "reason": str(hard_fail_risk_policy.get("reason") or hard_fail_risk_policy.get("decision_reason") or "").strip(),
                },
                "queue_decisions_total": int(len(surrogate_queue_decisions)),
                "matched_decisions": surrogate_matched_decisions[:32],
                "reject_lineages": surrogate_reject_lineages[:64],
                "reject_reasons": surrogate_reject_reasons[:16],
                "refine_reasons": surrogate_refine_reasons[:16],
                "winner_tokens_before": winner_tokens_before_surrogate,
                "winner_tokens_after": winner_proximate_tokens,
                "preferred_any_before": preferred_any_before_surrogate,
                "preferred_any_after": preferred_any_contains,
                "contains_before": contains_before_surrogate,
                "contains_after": contains,
                "repair_mode_effective": bool(effective_repair_mode),
                "repair_max_neighbors_effective": int(effective_repair_max_neighbors),
                "exclude_knobs_effective": list(effective_exclude_knobs),
            }
            snapshot["yield_governor_state_path"] = _safe_rel(yield_governor_path, app_root)
            snapshot["yield_governor"] = {
                "exists": bool(yield_governor.get("exists")),
                "status": str(yield_governor.get("status") or ""),
                "active": bool(yield_governor.get("active")),
                "hard_block_active": bool(yield_governor.get("hard_block_active")),
                "hard_block_reason": str(yield_governor.get("hard_block_reason") or ""),
                "hard_block_until_epoch": int(yield_governor.get("hard_block_until_epoch") or 0),
                "zero_coverage_seed_streak": int(yield_governor.get("zero_coverage_seed_streak") or 0),
                "zero_coverage_seed_streak_reason": str(yield_governor.get("zero_coverage_seed_streak_reason") or ""),
                "preferred_contains": list(yield_governor.get("preferred_contains", []) or [])[:8],
                "cooldown_contains": list(yield_governor.get("cooldown_contains", []) or [])[:8],
                "winner_proximate": dict(yield_governor.get("winner_proximate") or {}),
                "winner_proximate_positive_contains": list(
                    yield_governor.get("winner_proximate_positive_contains", []) or []
                )[:8],
                "controlled_recovery_contains": list(
                    yield_governor.get("controlled_recovery_contains", []) or []
                )[:8],
                "controlled_recovery_active": bool(controlled_recovery_active),
                "controlled_recovery_reason": str(controlled_recovery_reason or ""),
                "controlled_recovery_attempts_remaining": int(controlled_recovery_attempts_remaining),
                "controlled_recovery_variants_cap": int(controlled_recovery_variants_cap),
                "replay_fastlane": dict(yield_governor.get("replay_fastlane") or {}),
                "lane_weights": dict(yield_governor.get("lane_weights") or {}),
                "policy_overrides": dict(yield_governor.get("policy_overrides") or {}),
            }
            snapshot["covered_window_count"] = int(hygiene.get("covered_window_count", 0) or 0) + int(
                recent_seed_quality.get("covered_window_count", 0) or 0
            )
            snapshot["coverage_verified_ready_count"] = int(
                ready_buffer_state.get("coverage_verified_ready_count", ready_buffer_state.get("ready_depth", 0)) or 0
            )
            snapshot["recent_zero_yield_signal"] = {
                "active": bool(zero_yield_signal.get("active")),
                "analyzed": int(zero_yield_signal.get("analyzed", 0) or 0),
                "zeroish": int(zero_yield_signal.get("zeroish", 0) or 0),
                "strict_binding": int(zero_yield_signal.get("strict_binding", 0) or 0),
                "zeroish_ratio": float(zero_yield_signal.get("zeroish_ratio", 0.0) or 0.0),
            }
            snapshot["quality_governor"] = {
                "actions": list(quality_actions),
                "variant_cap": quality_variant_cap,
                "include_stress_effective": bool(effective_include_stress),
                "repair_mode_effective": bool(effective_repair_mode),
                "search_quality": dict(search_quality_state),
                "positive_lineage_count": int(search_quality_state.get("positive_lineage_count", 0) or 0),
                "zero_evidence_lineage_count": int(search_quality_state.get("zero_evidence_lineage_count", 0) or 0),
                "broad_search_allowed": bool(search_quality_state.get("broad_search_allowed")),
                "seed_generation_mode": str(search_quality_state.get("seed_generation_mode") or "broad_search_micro"),
                "controlled_recovery_active": bool(controlled_recovery_active),
                "controlled_recovery_reason": str(controlled_recovery_reason or ""),
                "controlled_recovery_attempts_remaining": int(controlled_recovery_attempts_remaining),
                "controlled_recovery_contains": list(controlled_recovery_tokens),
                "winner_proximate_positive_contains": list(controlled_recovery_tokens),
                "controlled_broad_rearm_after_epoch": int(
                    yield_governor.get("controlled_broad_rearm_after_epoch", 0) or 0
                ),
                "controlled_broad_active": bool(controlled_broad_active),
                "controlled_broad_reason": str(controlled_broad_reason or ""),
                "controlled_broad_contains": list(controlled_broad_contains),
            }
            snapshot["controlled_recovery_active"] = bool(controlled_recovery_active)
            snapshot["controlled_recovery_reason"] = str(controlled_recovery_reason or "")
            snapshot["controlled_recovery_attempts_remaining"] = int(controlled_recovery_attempts_remaining)
            snapshot["controlled_recovery_variants_cap"] = int(controlled_recovery_variants_cap)
            snapshot["controlled_recovery_triggered"] = bool(controlled_recovery_triggered)
            snapshot["controlled_recovery_lineage_anchor"] = str(controlled_recovery_lineage_anchor or "")
            snapshot["controlled_broad_rearm_after_epoch"] = int(
                yield_governor.get("controlled_broad_rearm_after_epoch", 0) or 0
            )
            snapshot["controlled_broad_active"] = bool(controlled_broad_active)
            snapshot["controlled_broad_reason"] = str(controlled_broad_reason or "")
            snapshot["controlled_broad_contains"] = list(controlled_broad_contains)
            if hard_block_active and not (
                controlled_recovery_active and controlled_recovery_attempts_remaining > 0 and controlled_recovery_backlog_ok
            ) and not controlled_broad_active:
                snapshot.update(
                    {
                        "status": "skipped",
                        "status_detail": "hard_block",
                        "reason": f"hard_block:{yield_governor.get('hard_block_reason') or 'unknown'}",
                        "human": "Queue seeder skipped: hard block is active due to repeated zero-coverage seed batches.",
                    }
                )
                _emit_state(state_path, snapshot)
                _append_log(log_path, snapshot)
                return 0
            snapshot["lane_selection"] = {
                "selected_lane": selected_lane,
                "selected_index": int(selected_lane_index),
                "lane_streak": int(lane_streak),
                "token_rotation": int(token_rotation),
                "available_lanes": list(lane_selection_payload.get("available_lanes") or []),
                "ranked_lanes": list(lane_selection_payload.get("ranked_lanes") or []),
                "exploit_first": bool(lane_selection_payload.get("exploit_first")),
                "lane_weights": dict(lane_selection_payload.get("lane_weights") or {}),
                "confirm_replay_hints": list(confirm_replay_hints),
                "confirm_replay_source": str(confirm_replay_source),
            }
            snapshot["confirm_replay_materialization"] = {
                "enabled": bool(confirm_replay_hints),
                "status": "prepared" if confirm_replay_hints else "idle",
                "hints": list(confirm_replay_hints)[:8],
                "source": str(confirm_replay_source),
                "mode": "planner_only_groundwork",
            }
            snapshot["policy_hash"] = planner_policy_hash
            snapshot["policy_fingerprint"] = policy_fingerprint_payload

            decision_dir = aggregate_dir / controller_group / "decisions"

            fallback_configs = [
                token.strip() for token in list(getattr(args, "fallback_base_config", []) or []) if token.strip()
            ]
            planner_bases: list[str] = []
            for cfg in [base_config, *fallback_configs]:
                if cfg not in planner_bases:
                    planner_bases.append(cfg)

            recovery_window_candidates = [
                str(controlled_recovery_lineage_anchor or "").strip(),
                *[str(token).strip() for token in list(winner_proximate_tokens or [])],
                *[str(token).strip() for token in list(contains or [])],
                *[str(token).strip() for token in list(preferred_any_contains or [])],
            ]
            effective_windows = _resolve_effective_seed_windows(
                explicit_windows=list(args.window or []),
                candidate_groups=recovery_window_candidates,
                run_index_groups=run_index_groups_all,
                app_root=app_root,
                limit=1,
            )
            snapshot["planner_windows"] = list(effective_windows)

            env = os.environ.copy()
            env["PYTHONPATH"] = f"{str((app_root / 'src'))}:{env.get('PYTHONPATH', '')}".rstrip(":")
            supported_planner_args = _detect_supported_planner_args(
                python_bin=str(args.python_bin),
                app_root=app_root,
            )
            requested_planner_args: list[str] = []
            if bool(effective_repair_mode):
                requested_planner_args.extend(["--repair-mode", "--repair-max-neighbors"])
            if effective_exclude_knobs:
                requested_planner_args.append("--exclude-knob")
            if winner_proximate_tokens:
                requested_planner_args.append("--winner-proximate-token")
            requested_planner_args.extend(
                [
                    "--planner-policy-hash",
                    "--planner-hash",
                    "--seed-lane",
                    "--seed-lane-index",
                    "--parent-diversity-depth",
                    "--parent-rotation-offset",
                ]
            )
            if confirm_replay_hints:
                requested_planner_args.append("--confirm-replay-hint")
            snapshot["planner_arg_support"] = {
                "supported": sorted(supported_planner_args),
                "requested": requested_planner_args,
                "missing_requested": sorted(
                    {flag for flag in requested_planner_args if flag not in supported_planner_args}
                ),
            }

            attempts: list[dict[str, Any]] = []
            decision = None
            final_result = None
            final_cmd: list[str] | None = None
            selected_base = None

            for idx, cfg in enumerate(planner_bases):
                run_group_try = run_group if idx == 0 else f"{run_group}_fb{idx}"
                cmd: list[str] = [
                    str(Path(str(args.python_bin)).expanduser()),
                    "scripts/optimization/evolve_next_batch.py",
                    "--base-config",
                    cfg,
                    "--controller-group",
                    controller_group,
                    "--run-group",
                    run_group_try,
                    "--run-index",
                    str(_resolve_under_root(args.run_index, app_root)),
                    "--num-variants",
                    str(int(effective_num_variants)),
                    "--ir-mode",
                    str(args.ir_mode),
                    "--dedupe-distance",
                    str(float(effective_dedupe_distance)),
                    "--max-changed-keys",
                    str(int(effective_max_changed_keys)),
                    "--policy-scale",
                    str(effective_policy_scale),
                ]
                planner_hash = _stable_hash({"cmd_base": cmd, "run_group": run_group_try}, prefix="planner")
                for token in contains:
                    cmd.extend(["--contains", token])
                for token in winner_proximate_tokens:
                    cmd.extend(["--winner-proximate-token", token])
                if "--planner-policy-hash" in supported_planner_args:
                    cmd.extend(["--planner-policy-hash", planner_policy_hash])
                if "--planner-hash" in supported_planner_args:
                    cmd.extend(["--planner-hash", planner_hash])
                if "--seed-lane" in supported_planner_args:
                    cmd.extend(["--seed-lane", selected_lane])
                if "--seed-lane-index" in supported_planner_args:
                    cmd.extend(["--seed-lane-index", str(int(selected_lane_index))])
                if "--parent-diversity-depth" in supported_planner_args:
                    cmd.extend(["--parent-diversity-depth", str(int(parent_diversity_depth))])
                if "--parent-rotation-offset" in supported_planner_args:
                    cmd.extend(["--parent-rotation-offset", str(int(parent_rotation_offset))])
                if "--confirm-replay-hint" in supported_planner_args:
                    for hint in confirm_replay_hints[:8]:
                        cmd.extend(["--confirm-replay-hint", str(hint)])
                for raw in effective_windows:
                    if not raw:
                        continue
                    cmd.extend(["--window", raw])
                cmd.append("--include-stress" if bool(effective_include_stress) else "--no-include-stress")
                cmd.extend(
                    _planner_repair_mode_args(
                        enabled=bool(effective_repair_mode),
                        supported_planner_args=supported_planner_args,
                    )
                )
                if bool(effective_repair_mode) and "--repair-max-neighbors" in supported_planner_args:
                    cmd.extend(["--repair-max-neighbors", str(int(effective_repair_max_neighbors))])
                if "--exclude-knob" in supported_planner_args:
                    for knob in effective_exclude_knobs:
                        cmd.extend(["--exclude-knob", knob])
                for key, value in extra_cli_args.items():
                    opt = str(key).strip()
                    if not opt.startswith("--"):
                        continue
                    text = str(value).strip()
                    if not text:
                        continue
                    cmd.extend([opt, text])
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

                result = subprocess.run(
                    cmd,
                    cwd=str(app_root),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                attempts.append(
                    {
                        "base_config": cfg,
                        "run_group": run_group_try,
                        "returncode": int(result.returncode),
                        "policy_hash": planner_policy_hash,
                        "planner_hash": planner_hash,
                        "seed_lane": selected_lane,
                        "parent_rotation_offset": int(parent_rotation_offset),
                        "stdout_tail": (result.stdout or "")[-800:],
                        "stderr_tail": (result.stderr or "")[-800:],
                    }
                )

                if result.returncode != 0:
                    continue

                decision_try = _load_decision_path(
                    decision_dir=decision_dir,
                    before_mtime=before_mtime,
                    app_root=app_root,
                )
                if not decision_try:
                    continue

                decision = decision_try
                final_result = result
                final_cmd = cmd
                selected_base = cfg
                run_group = run_group_try
                break

            if decision is None and not controlled_recovery_triggered:
                emergency_run_group = f"{run_group}_fallback_emergency"
                emergency_cmd: list[str] = [
                    str(Path(str(args.python_bin)).expanduser()),
                    "scripts/optimization/evolve_next_batch.py",
                    "--base-config",
                    base_config,
                    "--controller-group",
                    controller_group,
                    "--run-group",
                    emergency_run_group,
                    "--run-index",
                    str(_resolve_under_root(args.run_index, app_root)),
                    "--num-variants",
                    "1",
                    "--ir-mode",
                    "knob",
                    "--dedupe-distance",
                    "0.0",
                    "--max-changed-keys",
                    "6",
                    "--policy-scale",
                    "macro",
                ]
                emergency_planner_hash = _stable_hash(
                    {"cmd_base": emergency_cmd, "run_group": emergency_run_group},
                    prefix="planner",
                )
                emergency_cmd.extend(["--contains", controller_group])
                if "--planner-policy-hash" in supported_planner_args:
                    emergency_cmd.extend(["--planner-policy-hash", planner_policy_hash])
                if "--planner-hash" in supported_planner_args:
                    emergency_cmd.extend(["--planner-hash", emergency_planner_hash])
                if "--seed-lane" in supported_planner_args:
                    emergency_cmd.extend(["--seed-lane", selected_lane])
                if "--seed-lane-index" in supported_planner_args:
                    emergency_cmd.extend(["--seed-lane-index", str(int(selected_lane_index))])
                if "--parent-diversity-depth" in supported_planner_args:
                    emergency_cmd.extend(["--parent-diversity-depth", str(int(parent_diversity_depth))])
                if "--parent-rotation-offset" in supported_planner_args:
                    emergency_cmd.extend(["--parent-rotation-offset", str(int(parent_rotation_offset))])
                if "--confirm-replay-hint" in supported_planner_args:
                    for hint in confirm_replay_hints[:8]:
                        emergency_cmd.extend(["--confirm-replay-hint", str(hint)])
                for raw in effective_windows:
                    if not raw:
                        continue
                    emergency_cmd.extend(["--window", raw])
                emergency_cmd.append("--include-stress" if bool(effective_include_stress) else "--no-include-stress")
                emergency_cmd.extend(
                    _planner_repair_mode_args(
                        enabled=bool(effective_repair_mode),
                        supported_planner_args=supported_planner_args,
                    )
                )
                if bool(effective_repair_mode) and "--repair-max-neighbors" in supported_planner_args:
                    emergency_cmd.extend(["--repair-max-neighbors", str(int(effective_repair_max_neighbors))])
                if "--exclude-knob" in supported_planner_args:
                    for knob in effective_exclude_knobs:
                        emergency_cmd.extend(["--exclude-knob", knob])

                emergency_result = subprocess.run(
                    emergency_cmd,
                    cwd=str(app_root),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                attempts.append(
                    {
                        "base_config": base_config,
                        "run_group": emergency_run_group,
                        "returncode": int(emergency_result.returncode),
                        "policy_hash": planner_policy_hash,
                        "planner_hash": emergency_planner_hash,
                        "seed_lane": selected_lane,
                        "parent_rotation_offset": int(parent_rotation_offset),
                        "stdout_tail": (emergency_result.stdout or "")[-800:],
                        "stderr_tail": (emergency_result.stderr or "")[-800:],
                        "fallback_mode": "emergency_knob_relaxed",
                    }
                )
                if emergency_result.returncode == 0:
                    emergency_decision = _load_decision_path(
                        decision_dir=decision_dir,
                        before_mtime=before_mtime,
                        app_root=app_root,
                    )
                    if emergency_decision is not None:
                        decision = emergency_decision
                        final_result = emergency_result
                        final_cmd = emergency_cmd
                        selected_base = base_config
                        run_group = emergency_run_group
                        snapshot["fallback_mode"] = "emergency_knob_relaxed"

            snapshot["attempts"] = attempts
            if final_result is not None:
                snapshot["planner_cmd"] = final_cmd
                snapshot["returncode"] = int(final_result.returncode)
                snapshot["planner_stdout_tail"] = (final_result.stdout or "")[-2000:]
                snapshot["planner_stderr_tail"] = (final_result.stderr or "")[-2000:]
                snapshot["selected_base_config"] = selected_base
                snapshot["selected_lane"] = selected_lane
                snapshot["lane_streak"] = int(lane_streak)
                snapshot["token_rotation"] = int(token_rotation)
                snapshot["parent_rotation_offset"] = int(parent_rotation_offset)
                snapshot["parent_diversity_depth"] = int(parent_diversity_depth)

            if decision is None:
                snapshot.update({"status": "failed", "reason": "all_planners_failed", "human": "Queue seeder failed: all planner inputs invalid/failed; waiting for fallback fix."})
                _emit_state(state_path, snapshot)
                _append_log(log_path, snapshot)
                _append_log(
                    incident_path,
                    {
                        "ts": _utc_now_iso(),
                        "agent": "queue_seeder",
                        "kind": "seed_failed",
                        "human": snapshot["human"],
                        "payload": {
                            "controller_group": controller_group,
                            "attempts": [{"base_config": a.get("base_config"), "returncode": a.get("returncode")} for a in attempts],
                        },
                    },
                )
                return 1

            queue_path = decision["queue_path"]
            decision_payload = decision.get("decision_payload", {})
            if not isinstance(decision_payload, dict):
                decision_payload = {}
            queue_prune = _prune_seed_queue(
                queue_path=queue_path,
                app_root=app_root,
                lineage_uid=_decision_lineage_uid(decision_payload),
                decision_payload=decision_payload,
                run_index_groups=run_index_groups_all,
                quarantine_by_group=quarantine_by_group,
            )
            snapshot["queue_prune"] = dict(queue_prune)
            if int(queue_prune.get("rows_after", 0)) <= 0:
                block_reason = str(queue_prune.get("block_reason") or "queue_pruned_empty")
                queue_rel = _safe_rel(queue_path, app_root)
                lineage_recent_summary = dict(queue_prune.get("lineage_recent_summary") or {})
                lineage_positive_evidence = bool(lineage_recent_summary.get("has_positive_coverage_trade_evidence"))
                queue_policy_path = _write_queue_policy_sidecar(
                    queue_path=queue_path,
                    app_root=app_root,
                    planner_policy_hash=planner_policy_hash,
                    selected_lane=selected_lane,
                    selected_lane_index=selected_lane_index,
                    token_rotation=token_rotation,
                    parent_rotation_offset=parent_rotation_offset,
                    parent_diversity_depth=parent_diversity_depth,
                    confirm_replay_hints=confirm_replay_hints,
                    decision_payload=decision_payload,
                    coverage_verified=False,
                    coverage_reason=block_reason,
                    ready_buffer_excluded=True,
                    seed_feasibility_status="blocked",
                    seed_feasibility_reason=block_reason,
                    lineage_positive_evidence=lineage_positive_evidence,
                    recovery_mode="controlled" if controlled_recovery_triggered else "",
                    recovery_reason=str(controlled_recovery_queue_reason or "") if controlled_recovery_triggered else "",
                    recovery_lineage_anchor=str(controlled_recovery_lineage_anchor or ""),
                )
                _decorate_queue_metadata(
                    queue_path=queue_path,
                    planner_policy_hash=planner_policy_hash,
                    queue_policy_path=queue_policy_path,
                    app_root=app_root,
                    coverage_verified=False,
                    coverage_reason=block_reason,
                    ready_buffer_excluded=True,
                    seed_feasibility_status="blocked",
                    seed_feasibility_reason=block_reason,
                    lineage_positive_evidence=lineage_positive_evidence,
                    recovery_mode="controlled" if controlled_recovery_triggered else "",
                    recovery_reason=str(controlled_recovery_queue_reason or "") if controlled_recovery_triggered else "",
                    recovery_lineage_anchor=str(controlled_recovery_lineage_anchor or ""),
                )
                _upsert_orphan_queue(
                    orphan_path=orphan_path,
                    queue_rel=queue_rel,
                    reason=block_reason,
                    cooldown_sec=21600,
                )
                human = "Queue seeder pruned all generated rows due to missing OOS coverage or duplicate seed signatures."
                if block_reason == "MAX_VAR_MULTIPLIER_INVALID":
                    human = "Queue seeder pruned all generated rows: backtest.max_var_multiplier fell below the runtime-safe floor."
                snapshot.update(
                    {
                        "status": "failed",
                        "reason": block_reason,
                        "human": human,
                        "queue_path": queue_rel,
                        "queue_policy_path": _safe_rel(queue_policy_path, app_root),
                    }
                )
                _emit_state(state_path, snapshot)
                _append_log(log_path, snapshot)
                _append_log(
                    incident_path,
                    {
                        "ts": _utc_now_iso(),
                        "agent": "queue_seeder",
                        "kind": "seed_failed",
                        "human": snapshot["human"],
                        "payload": {
                            "queue_path": snapshot["queue_path"],
                            "coverage_rejected": int(queue_prune.get("coverage_rejected", 0) or 0),
                            "dedupe_rejected": int(queue_prune.get("dedupe_rejected", 0) or 0),
                            "missing_months": list(queue_prune.get("missing_months", []) or []),
                        },
                    },
                )
                return 1
            queue_policy_path = _write_queue_policy_sidecar(
                queue_path=queue_path,
                app_root=app_root,
                planner_policy_hash=planner_policy_hash,
                selected_lane=selected_lane,
                selected_lane_index=selected_lane_index,
                token_rotation=token_rotation,
                parent_rotation_offset=parent_rotation_offset,
                parent_diversity_depth=parent_diversity_depth,
                confirm_replay_hints=confirm_replay_hints,
                decision_payload=decision_payload,
                seed_feasibility_status="ok",
                seed_feasibility_reason="",
                lineage_positive_evidence=bool(
                    dict(queue_prune.get("lineage_recent_summary") or {}).get("has_positive_coverage_trade_evidence")
                ),
                recovery_mode="controlled" if controlled_recovery_triggered else "",
                recovery_reason=str(controlled_recovery_queue_reason or "") if controlled_recovery_triggered else "",
                recovery_lineage_anchor=str(controlled_recovery_lineage_anchor or ""),
            )
            _decorate_queue_metadata(
                queue_path=queue_path,
                planner_policy_hash=planner_policy_hash,
                queue_policy_path=queue_policy_path,
                app_root=app_root,
                coverage_verified=True,
                coverage_reason="coverage_verified",
                ready_buffer_excluded=False,
                seed_feasibility_status="ok",
                seed_feasibility_reason="",
                lineage_positive_evidence=bool(
                    dict(queue_prune.get("lineage_recent_summary") or {}).get("has_positive_coverage_trade_evidence")
                ),
                recovery_mode="controlled" if controlled_recovery_triggered else "",
                recovery_reason=str(controlled_recovery_queue_reason or "") if controlled_recovery_triggered else "",
                recovery_lineage_anchor=str(controlled_recovery_lineage_anchor or ""),
            )
            controlled_recovery_attempts_after = controlled_recovery_attempts_remaining
            queue_payload = _load_queue_rows(queue_path)
            snapshot.update(
                {
                    "status": "seeded",
                    "status_detail": "queued",
                    "run_group": run_group,
                    "controller_group": controller_group,
                    "decision_path": _safe_rel(decision["decision_path"], app_root),
                    "queue_path": _safe_rel(queue_path, app_root),
                    "queue_policy_path": _safe_rel(queue_policy_path, app_root),
                    "queue_rows_generated": len(queue_payload),
                    "parent_resolution": dict(decision_payload.get("parent_resolution") or {}),
                    "reason": None,
                    "controlled_recovery": {
                        "active": bool(controlled_recovery_triggered),
                        "reason": str(controlled_recovery_reason or ""),
                        "lineage_anchor": str(controlled_recovery_lineage_anchor or ""),
                        "attempts_before": int(controlled_recovery_attempts_remaining),
                        "attempts_after": int(controlled_recovery_attempts_after),
                        "variants_cap": int(controlled_recovery_variants_cap),
                    },
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
