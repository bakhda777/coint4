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
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows


def _summarize_queues(aggregate_dir: Path) -> tuple[int, int, int, int, int, list[Path]]:
    pending_like_status = {"planned", "queued", "running", "stalled", "failed", "error", "active"}
    dispatchable_status = {"planned", "queued", "running", "failed"}
    total_pending_like = 0
    dispatchable_pending = 0
    executable_pending = 0
    runnable_queue_count = 0
    scanned = 0
    queue_paths: list[Path] = []
    app_root = aggregate_dir.parents[2] if len(aggregate_dir.parents) >= 3 else aggregate_dir
    if not aggregate_dir.exists():
        return (total_pending_like, dispatchable_pending, executable_pending, runnable_queue_count, scanned, queue_paths)

    for queue_file in sorted(aggregate_dir.glob("*/run_queue.csv")):
        if queue_file.parent.name.startswith("."):
            continue
        scanned += 1
        queue_paths.append(queue_file)
        rows = _load_queue_rows(queue_file)
        queue_executable = 0
        for row in rows:
            status = (row.get("status") or "").strip().lower()
            if status in pending_like_status:
                total_pending_like += 1
            if status not in dispatchable_status:
                continue
            dispatchable_pending += 1
            if status == "running":
                executable_pending += 1
                queue_executable += 1
                continue
            config_path = (row.get("config_path") or "").strip()
            if not config_path:
                continue
            cfg_path = Path(config_path)
            if not cfg_path.is_absolute():
                cfg_path = app_root / cfg_path
            if cfg_path.exists():
                executable_pending += 1
                queue_executable += 1
        if queue_executable > 0:
            runnable_queue_count += 1
    return total_pending_like, dispatchable_pending, executable_pending, runnable_queue_count, scanned, queue_paths


def _emit_state(state_path: Path, payload: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _append_log(log_path: Path, payload: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _load_directive(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _load_impossibility_pruner(path: Path) -> dict[str, Any]:
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
        # conservative defaults for impossible-config streaks
        "max_changed_keys_cap": 2,
        "dedupe_distance_floor": 0.02,
        "num_variants_cap": 64,
        "policy_scale": "micro",
    }


def _load_search_blacklist(path: Path) -> dict[str, Any]:
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
        "max_changed_keys_cap": int(caps.get("max_changed_keys_cap") or 4),
        "dedupe_distance_floor": float(caps.get("dedupe_distance_floor") or 0.02),
        "num_variants_cap": int(caps.get("num_variants_cap") or 64),
        "policy_scale": str(caps.get("policy_scale") or "auto"),
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
    ready_depth = len(entries)

    state.update(
        {
            "status": "ok",
            "reason": "threshold_missing",
            "target_depth": int(target_depth) if target_depth is not None else None,
            "refill_threshold": int(refill_threshold) if refill_threshold is not None else None,
            "effective_refill_threshold": int(effective_refill_threshold) if effective_refill_threshold is not None else None,
            "ready_depth": int(ready_depth),
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
        "preferred_contains": [],
        "cooldown_contains": [],
        "preferred_operator_ids": [],
        "cooldown_operator_ids": [],
        "winner_proximate": {"enabled": False, "contains": [], "reason": ""},
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
    lane_weights = payload.get("lane_weights", {})
    if not isinstance(lane_weights, dict):
        lane_weights = {}
    policy_overrides = payload.get("policy_overrides", {})
    if not isinstance(policy_overrides, dict):
        policy_overrides = {}

    state.update(
        {
            "status": "ok",
            "reason": "ok",
            "active": bool(payload.get("active")),
            "preferred_contains": _to_tokens(payload.get("preferred_contains", [])),
            "cooldown_contains": _to_tokens(payload.get("cooldown_contains", [])),
            "preferred_operator_ids": _to_tokens(payload.get("preferred_operator_ids", [])),
            "cooldown_operator_ids": _to_tokens(payload.get("cooldown_operator_ids", [])),
            "winner_proximate": {
                "enabled": bool(winner.get("enabled")),
                "contains": _to_tokens(winner.get("contains", [])),
                "reason": str(winner.get("reason") or "").strip(),
            },
            "lane_weights": {str(k): (_as_int(v, default=0, min_value=0) or 0) for k, v in lane_weights.items()},
            "policy_overrides": dict(policy_overrides),
        }
    )
    return state


def _entry_matches_controller(entry: dict[str, Any], controller_group: str) -> bool:
    wanted = str(controller_group or "").strip()
    if not wanted:
        return True
    group = str(entry.get("controller_group") or entry.get("group") or entry.get("target_group") or "").strip()
    if group:
        return group == wanted
    return True


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
    wanted = {"--repair-mode", "--repair-max-neighbors", "--exclude-knob"}
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
                "ready_buffer_state_path": _safe_rel(ready_buffer_state_path, app_root),
                "ready_queue_buffer": {
                    "exists": bool(ready_buffer_state.get("exists")),
                    "status": str(ready_buffer_state.get("status") or ""),
                    "reason": str(ready_buffer_state.get("reason") or ""),
                    "target_depth": ready_buffer_state.get("target_depth"),
                    "refill_threshold": ready_buffer_state.get("refill_threshold"),
                    "effective_refill_threshold": ready_buffer_state.get("effective_refill_threshold"),
                    "ready_depth": ready_buffer_state.get("ready_depth"),
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

            controller_group = str(args.controller_group).strip() or "autonomous_queue_seeder"
            base_config = str(args.base_config).strip() or "configs/prod_final_budget1000.yaml"
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_group = f"{str(args.run_group_prefix).strip() or 'autonomous_seed'}_{timestamp}"

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
            hard_fail_risk_policy = gate_surrogate.get("hard_fail_risk_policy", {})
            if not isinstance(hard_fail_risk_policy, dict):
                hard_fail_risk_policy = {}

            contains = [token.strip() for token in list(args.contains or []) if token.strip()]
            directive_contains = [str(token).strip() for token in list(directive.get("contains", []) or []) if str(token).strip()]
            contains = _merge_unique(directive_contains, contains)
            directive_winner = directive.get("winner_proximate", {})
            if not isinstance(directive_winner, dict):
                directive_winner = {}
            contains = _merge_unique(_to_tokens(directive_winner.get("contains", [])), contains)
            contains = _merge_unique(_to_tokens((yield_governor.get("winner_proximate", {}) or {}).get("contains", [])), contains)
            contains = _merge_unique(list(yield_governor.get("preferred_contains", []) or []), contains)
            if not contains:
                contains = [controller_group]

            effective_num_variants = int(args.num_variants)
            effective_num_variants_floor = max(1, int(args.num_variants_floor))
            effective_max_changed_keys = int(args.max_changed_keys)
            effective_dedupe_distance = float(args.dedupe_distance)
            effective_policy_scale = str(args.policy_scale)
            effective_repair_mode = bool(args.repair_mode)
            effective_repair_max_neighbors = max(1, int(args.repair_max_neighbors))
            effective_exclude_knobs = [token.strip() for token in list(args.exclude_knob or []) if token and token.strip()]
            extra_cli_args = directive.get("extra_cli_args", {})
            if not isinstance(extra_cli_args, dict):
                extra_cli_args = {}

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

            directive_pruner = directive.get("impossibility_pruner", {})
            if isinstance(directive_pruner, dict) and bool(directive_pruner.get("enabled")):
                try:
                    effective_max_changed_keys = min(
                        int(effective_max_changed_keys),
                        int(directive_pruner.get("max_changed_keys_cap", effective_max_changed_keys)),
                    )
                except Exception:
                    pass
                try:
                    effective_dedupe_distance = max(
                        float(effective_dedupe_distance),
                        float(directive_pruner.get("dedupe_distance_floor", effective_dedupe_distance)),
                    )
                except Exception:
                    pass
                try:
                    effective_num_variants = min(
                        int(effective_num_variants),
                        int(directive_pruner.get("num_variants_cap", effective_num_variants)),
                    )
                except Exception:
                    pass
                pscale = str(directive_pruner.get("policy_scale", "")).strip()
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
            if surrogate_reject_lineages:
                contains = [token for token in contains if token not in set(surrogate_reject_lineages)]
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
            effective_num_variants = max(int(effective_num_variants), int(effective_num_variants_floor))

            snapshot["directive"] = {
                "path": _safe_rel(directive_path, app_root),
                "exists": directive_path.exists(),
                "mode": str(directive.get("mode", "")),
                "dominant_reason": str(directive.get("dominant_reason", "")),
                "contains": contains,
                "policy_scale": effective_policy_scale,
                "num_variants": effective_num_variants,
                "num_variants_floor": effective_num_variants_floor,
                "max_changed_keys": effective_max_changed_keys,
                "dedupe_distance": effective_dedupe_distance,
                "include_stress": bool(args.include_stress),
                "repair_mode": bool(effective_repair_mode),
                "repair_max_neighbors": int(effective_repair_max_neighbors),
                "exclude_knobs": list(effective_exclude_knobs),
                "extra_cli_args": extra_cli_args,
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
                "preferred_contains": list(yield_governor.get("preferred_contains", []) or [])[:8],
                "cooldown_contains": list(yield_governor.get("cooldown_contains", []) or [])[:8],
                "winner_proximate": dict(yield_governor.get("winner_proximate") or {}),
                "lane_weights": dict(yield_governor.get("lane_weights") or {}),
                "policy_overrides": dict(yield_governor.get("policy_overrides") or {}),
            }

            decision_dir = aggregate_dir / controller_group / "decisions"

            fallback_configs = [
                token.strip() for token in list(getattr(args, "fallback_base_config", []) or []) if token.strip()
            ]
            planner_bases: list[str] = []
            for cfg in [base_config, *fallback_configs]:
                if cfg not in planner_bases:
                    planner_bases.append(cfg)

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
                for token in contains:
                    cmd.extend(["--contains", token])
                for raw in args.window:
                    if not raw:
                        continue
                    cmd.extend(["--window", raw])
                cmd.append("--include-stress" if bool(args.include_stress) else "--no-include-stress")
                if bool(effective_repair_mode) and "--repair-mode" in supported_planner_args:
                    cmd.append("--repair-mode")
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

            if decision is None:
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
                emergency_cmd.extend(["--contains", controller_group])
                for raw in args.window:
                    if not raw:
                        continue
                    emergency_cmd.extend(["--window", raw])
                emergency_cmd.append("--include-stress" if bool(args.include_stress) else "--no-include-stress")
                if bool(effective_repair_mode) and "--repair-mode" in supported_planner_args:
                    emergency_cmd.append("--repair-mode")
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
            queue_payload = _load_queue_rows(queue_path)
            snapshot.update(
                {
                    "status": "seeded",
                    "status_detail": "queued",
                    "run_group": run_group,
                    "controller_group": controller_group,
                    "decision_path": _safe_rel(decision["decision_path"], app_root),
                    "queue_path": _safe_rel(queue_path, app_root),
                    "queue_rows_generated": len(queue_payload),
                    "reason": None,
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
