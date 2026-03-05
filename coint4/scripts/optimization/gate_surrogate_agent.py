#!/usr/bin/env python3
"""Queue-level gate surrogate for autonomous search director.

This agent fuses queue status, fullspan state, deterministic quarantine, and
run_index rollup into a per-queue risk decision:
- decision in {allow, refine, reject}
- risk_score in [0, 1]
- reason + evidence payload

Fail-closed behavior:
- if required sources are missing/invalid, policy is disabled and queue
  decisions stay neutral (`refine`, risk=0.5).
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVO_RE = re.compile(r"(?<![A-Za-z0-9])(evo_[0-9a-f]{8,64})(?![A-Za-z0-9])", re.IGNORECASE)

PENDING_STATUSES = {"planned", "queued", "running", "failed", "stalled", "error", "active"}
ERROR_STATUSES = {"failed", "stalled", "error"}
DETERMINISTIC_CODES = {
    "CONFIG_VALIDATION_ERROR",
    "MAX_VAR_MULTIPLIER_INVALID",
    "MAX_CORRELATION_INVALID",
    "NON_POSITIVE_THRESHOLD",
    "INVALID_PARAM",
}
CALIBRATION_MIN_SAMPLE = 100


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


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
    try:
        return int(float(value or 0))
    except Exception:
        return default


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


def safe_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def normalize_queue_key(value: str | Path, *, root: Path) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        path = root / path
    return safe_rel(path, root)


def extract_evo_token(*texts: str) -> str:
    for text in texts:
        val = str(text or "").strip()
        if not val:
            continue
        match = EVO_RE.search(val)
        if match:
            return match.group(1).lower()
    return ""


def extract_lineage_uid(*values: Any) -> str:
    queue: list[Any] = list(values)
    seen: set[int] = set()
    while queue:
        current = queue.pop(0)
        ident = id(current)
        if ident in seen:
            continue
        seen.add(ident)
        if isinstance(current, dict):
            for key in ("lineage_uid", "candidate_uid"):
                value = str(current.get(key) or "").strip().lower()
                if value:
                    return value
            raw_meta = current.get("metadata_json")
            if isinstance(raw_meta, str):
                text = raw_meta.strip()
                if text.startswith("{") or text.startswith("["):
                    try:
                        queue.append(json.loads(text))
                    except Exception:
                        pass
            for value in current.values():
                if isinstance(value, (dict, list)):
                    queue.append(value)
        elif isinstance(current, list):
            queue.extend(current)
        elif isinstance(current, str):
            text = current.strip()
            if text.startswith("{") or text.startswith("["):
                try:
                    queue.append(json.loads(text))
                except Exception:
                    pass
    return ""


def normalize_lookup_path(value: Any, *, root: Path) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        path = root / path
    try:
        return path.resolve().as_posix()
    except Exception:
        return path.as_posix()


def canonical_reason(entry: dict[str, Any]) -> str:
    strict_reason = str(entry.get("strict_gate_reason") or "").strip().upper()
    reject_reason = str(entry.get("rejection_reason") or "").strip().upper()
    contract_reason = str(entry.get("contract_reason") or "").strip().upper()
    merged = strict_reason or reject_reason or contract_reason
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


def collect_queue_files(aggregate_dir: Path) -> list[Path]:
    if not aggregate_dir.exists():
        return []
    out: list[Path] = []
    for queue_path in sorted(aggregate_dir.glob("*/run_queue.csv")):
        if queue_path.parent.name.startswith("."):
            continue
        out.append(queue_path)
    return out


def load_queue_status(queue_path: Path) -> tuple[int, dict[str, int]]:
    if not queue_path.exists():
        return 0, {}
    counts: Counter[str] = Counter()
    rows = 0
    try:
        with queue_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows += 1
                status = str((row or {}).get("status") or "").strip().lower()
                if status:
                    counts[status] += 1
    except Exception:
        return 0, {}
    return rows, dict(counts)


def load_fullspan_state(path: Path, root: Path) -> dict[str, Any]:
    payload = load_json(path, {})
    if not isinstance(payload, dict):
        return {
            "available": path.exists(),
            "valid": False,
            "state_version": None,
            "queues": {},
        }

    queues_raw = payload.get("queues", {})
    queues: dict[str, dict[str, Any]] = {}
    if isinstance(queues_raw, dict):
        for key, entry in queues_raw.items():
            if not isinstance(entry, dict):
                continue
            queue_key = normalize_queue_key(str(key), root=root)
            if queue_key:
                queues[queue_key] = entry

    return {
        "available": path.exists(),
        "valid": True,
        "state_version": payload.get("state_version"),
        "queues": queues,
    }


def load_quarantine(path: Path, root: Path) -> dict[str, Any]:
    payload = load_json(path, {})
    if not isinstance(payload, dict):
        return {
            "available": path.exists(),
            "valid": False,
            "entries": 0,
            "codes": {},
            "dominant_code": "",
            "dominant_ratio": 0.0,
            "active": False,
            "queues": {},
        }

    entries_raw = payload.get("entries", [])
    entries = [entry for entry in entries_raw if isinstance(entry, dict)] if isinstance(entries_raw, list) else []

    by_queue: dict[str, Counter[str]] = defaultdict(Counter)
    codes: Counter[str] = Counter()

    for entry in entries:
        queue_key = normalize_queue_key(str(entry.get("queue") or ""), root=root)
        code = str(entry.get("code") or "").strip().upper()
        if not code:
            continue
        codes[code] += 1
        if queue_key:
            by_queue[queue_key][code] += 1

    dominant_code = ""
    dominant_count = 0
    dominant_ratio = 0.0
    total = sum(codes.values())
    if codes:
        dominant_code, dominant_count = codes.most_common(1)[0]
        if total > 0:
            dominant_ratio = float(dominant_count) / float(total)

    active = bool(total >= 6 and dominant_code in DETERMINISTIC_CODES and dominant_ratio >= 0.40)

    return {
        "available": path.exists(),
        "valid": True,
        "entries": len(entries),
        "codes": dict(codes),
        "dominant_code": dominant_code,
        "dominant_ratio": float(round(dominant_ratio, 6)),
        "active": active,
        "queues": {
            queue_key: {
                "total": int(sum(counter.values())),
                "codes": dict(counter),
            }
            for queue_key, counter in by_queue.items()
        },
    }


def load_queue_lineage_maps(queue_files: list[Path], root: Path) -> dict[str, dict[str, dict[str, str]]]:
    out: dict[str, dict[str, dict[str, str]]] = {}
    for queue_path in queue_files:
        run_group = queue_path.parent.name
        group_entry = out.setdefault(run_group, {"results": {}, "configs": {}})
        try:
            with queue_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if not isinstance(row, dict):
                        continue
                    lineage_uid = extract_lineage_uid(row)
                    if not lineage_uid:
                        lineage_uid = extract_evo_token(
                            str(row.get("config_path") or ""),
                            str(row.get("results_dir") or ""),
                        )
                    if not lineage_uid:
                        continue
                    results_key = normalize_lookup_path(row.get("results_dir"), root=root)
                    config_key = normalize_lookup_path(row.get("config_path"), root=root)
                    if results_key and results_key not in group_entry["results"]:
                        group_entry["results"][results_key] = lineage_uid
                    if config_key and config_key not in group_entry["configs"]:
                        group_entry["configs"][config_key] = lineage_uid
        except Exception:
            continue
    return out


def load_run_index(path: Path, *, root: Path, queue_lineages: dict[str, dict[str, dict[str, str]]]) -> dict[str, Any]:
    if not path.exists():
        return {
            "available": False,
            "valid": False,
            "rows": 0,
            "groups": 0,
            "groups_data": {},
        }

    group_data: dict[str, dict[str, Any]] = {}
    rows = 0

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                run_group = str(row.get("run_group") or "").strip()
                if not run_group:
                    continue

                rows += 1
                entry = group_data.setdefault(
                    run_group,
                    {
                        "rows": 0,
                        "status": Counter(),
                        "metrics_missing": 0,
                        "completed": 0,
                        "completed_zero_activity": 0,
                        "completed_informative": 0,
                        "lineages": Counter(),
                        "legacy_lineages": Counter(),
                    },
                )

                entry["rows"] += 1
                status = str(row.get("status") or "").strip().lower()
                if status:
                    entry["status"][status] += 1

                metrics_present = parse_bool(row.get("metrics_present"), default=False)
                if not metrics_present:
                    entry["metrics_missing"] += 1

                if status == "completed":
                    entry["completed"] += 1
                    trades = parse_float(row.get("total_trades"), 0.0)
                    pairs = parse_float(row.get("total_pairs_traded") or row.get("total_pairs"), 0.0)
                    pnl = parse_float(row.get("total_pnl"), 0.0)
                    informative = bool(trades > 0.0 or pairs > 0.0 or abs(pnl) > 1e-12)
                    if informative:
                        entry["completed_informative"] += 1
                    else:
                        entry["completed_zero_activity"] += 1

                lineage = extract_lineage_uid(row)
                if not lineage:
                    group_lookup = queue_lineages.get(run_group, {})
                    results_lookup = group_lookup.get("results", {}) if isinstance(group_lookup, dict) else {}
                    config_lookup = group_lookup.get("configs", {}) if isinstance(group_lookup, dict) else {}
                    lineage = str(
                        results_lookup.get(normalize_lookup_path(row.get("results_dir"), root=root))
                        or config_lookup.get(normalize_lookup_path(row.get("config_path"), root=root))
                        or ""
                    ).strip().lower()
                legacy_lineage = extract_evo_token(
                    str(row.get("config_path") or ""),
                    str(row.get("run_id") or ""),
                    str(row.get("results_dir") or ""),
                )
                if not lineage:
                    lineage = legacy_lineage
                if lineage:
                    entry["lineages"][lineage] += 1
                if legacy_lineage:
                    entry["legacy_lineages"][legacy_lineage] += 1
    except Exception:
        return {
            "available": True,
            "valid": False,
            "rows": 0,
            "groups": 0,
            "groups_data": {},
        }

    return {
        "available": True,
        "valid": True,
        "rows": rows,
        "groups": len(group_data),
        "groups_data": {
            run_group: {
                "rows": int(data["rows"]),
                "status": dict(data["status"]),
                "metrics_missing": int(data["metrics_missing"]),
                "completed": int(data["completed"]),
                "completed_zero_activity": int(data["completed_zero_activity"]),
                "completed_informative": int(data["completed_informative"]),
                "lineages": dict(data["lineages"]),
                "legacy_lineages": dict(data["legacy_lineages"]),
            }
            for run_group, data in group_data.items()
        },
    }


def load_calibration_state(path: Path, *, default_reject: float, default_refine: float) -> dict[str, Any]:
    payload = load_json(path, {})
    if not isinstance(payload, dict):
        return {
            "available": path.exists(),
            "valid": False,
            "applied": False,
            "applied_reject_threshold": round(default_reject, 4),
            "applied_refine_threshold": round(default_refine, 4),
            "sample_size": 0,
            "reason": "missing_or_invalid",
            "ts": "",
        }

    min_sample_size = parse_int(payload.get("min_sample_size"), CALIBRATION_MIN_SAMPLE)
    recommended_reject = max(0.0, min(1.0, parse_float(payload.get("recommended_reject_threshold"), default_reject)))
    recommended_refine = max(0.0, min(1.0, parse_float(payload.get("recommended_refine_threshold"), default_refine)))
    applied_reject = max(0.0, min(1.0, parse_float(payload.get("applied_reject_threshold"), recommended_reject)))
    applied_refine = max(0.0, min(1.0, parse_float(payload.get("applied_refine_threshold"), recommended_refine)))
    if recommended_reject < recommended_refine:
        recommended_reject, recommended_refine = recommended_refine, recommended_reject
    if applied_reject < applied_refine:
        applied_reject, applied_refine = applied_refine, applied_reject

    sample_size = parse_int(payload.get("sample_size"), 0)
    enabled = bool(payload.get("enabled"))
    applied = bool(payload.get("applied"))
    valid = bool(enabled and applied and sample_size >= min_sample_size)
    return {
        "available": path.exists(),
        "valid": valid,
        "applied": valid,
        "applied_reject_threshold": round(applied_reject if valid else default_reject, 4),
        "applied_refine_threshold": round(applied_refine if valid else default_refine, 4),
        "sample_size": int(sample_size),
        "reason": str(payload.get("reason") or ""),
        "ts": str(payload.get("ts") or ""),
    }


def queue_run_group(queue_key: str) -> str:
    if not queue_key:
        return ""
    path = Path(queue_key)
    if path.name != "run_queue.csv":
        return path.parent.name
    return path.parent.name


def apply_contribution(contribs: list[dict[str, Any]], key: str, delta: float) -> None:
    if abs(delta) < 1e-12:
        return
    contribs.append({"key": key, "delta": round(float(delta), 4)})


def decide_queue(
    *,
    queue_key: str,
    queue_rows: int,
    queue_status: dict[str, int],
    run_group: str,
    run_index_entry: dict[str, Any] | None,
    fullspan_entry: dict[str, Any] | None,
    quarantine_entry: dict[str, Any] | None,
    quarantine_active: bool,
    reject_threshold: float,
    refine_threshold: float,
) -> tuple[dict[str, Any], list[str], bool]:
    risk = 0.15
    contributions: list[dict[str, Any]] = []
    validation_error = False

    queue_completed = int(queue_status.get("completed", 0))
    queue_pending = int(sum(queue_status.get(status, 0) for status in PENDING_STATUSES))
    queue_errors = int(sum(queue_status.get(status, 0) for status in ERROR_STATUSES))
    queue_skipped = int(queue_status.get("skipped", 0))

    if queue_rows <= 0:
        apply_contribution(contributions, "queue_empty", 0.20)
        risk += 0.20
    else:
        pending_ratio = float(queue_pending) / float(queue_rows)
        error_ratio = float(queue_errors) / float(queue_rows)
        skipped_ratio = float(queue_skipped) / float(queue_rows)

        if pending_ratio >= 0.80 and queue_completed == 0:
            apply_contribution(contributions, "queue_pending_backlog", 0.25)
            risk += 0.25
        elif pending_ratio >= 0.50:
            apply_contribution(contributions, "queue_pending_partial", 0.12)
            risk += 0.12

        if error_ratio >= 0.30:
            apply_contribution(contributions, "queue_error_ratio", 0.20)
            risk += 0.20

        if skipped_ratio >= 0.80 and queue_completed == 0:
            apply_contribution(contributions, "queue_skipped_dominant", 0.20)
            risk += 0.20

        if queue_completed > 0 and queue_pending == 0 and queue_errors == 0:
            apply_contribution(contributions, "queue_completed_signal", -0.06)
            risk -= 0.06

    metrics_missing_ratio = 0.0
    zero_activity_ratio = 0.0
    informative_ratio = 0.0
    run_index_status: dict[str, int] = {}
    run_index_rows = 0
    lineages: list[str] = []
    legacy_lineages: list[str] = []

    if isinstance(run_index_entry, dict) and run_index_entry:
        run_index_rows = int(run_index_entry.get("rows", 0) or 0)
        run_index_status = {
            str(k): int(v)
            for k, v in (run_index_entry.get("status", {}) or {}).items()
            if str(k)
        }
        metrics_missing = int(run_index_entry.get("metrics_missing", 0) or 0)
        completed = int(run_index_entry.get("completed", 0) or 0)
        zero_activity = int(run_index_entry.get("completed_zero_activity", 0) or 0)
        informative = int(run_index_entry.get("completed_informative", 0) or 0)

        if run_index_rows > 0:
            metrics_missing_ratio = float(metrics_missing) / float(run_index_rows)
        if completed > 0:
            zero_activity_ratio = float(zero_activity) / float(completed)
            informative_ratio = float(informative) / float(completed)

        if run_index_rows <= 0:
            apply_contribution(contributions, "run_index_empty_group", 0.18)
            risk += 0.18
        else:
            if metrics_missing_ratio >= 0.75:
                apply_contribution(contributions, "metrics_missing_heavy", 0.35)
                risk += 0.35
            elif metrics_missing_ratio >= 0.40:
                apply_contribution(contributions, "metrics_missing_mid", 0.20)
                risk += 0.20
            elif metrics_missing_ratio >= 0.15:
                apply_contribution(contributions, "metrics_missing_light", 0.10)
                risk += 0.10

            if completed == 0:
                apply_contribution(contributions, "no_completed_rows", 0.16)
                risk += 0.16
            else:
                if zero_activity_ratio >= 0.85 and completed >= 4:
                    apply_contribution(contributions, "zero_activity_dominant", 0.35)
                    risk += 0.35
                elif zero_activity_ratio >= 0.60 and completed >= 3:
                    apply_contribution(contributions, "zero_activity_mid", 0.20)
                    risk += 0.20

                if informative_ratio >= 0.60:
                    apply_contribution(contributions, "informative_completed_strong", -0.20)
                    risk -= 0.20
                elif informative_ratio >= 0.30:
                    apply_contribution(contributions, "informative_completed_partial", -0.10)
                    risk -= 0.10

            skipped = int(run_index_status.get("skipped", 0))
            skipped_ratio = float(skipped) / float(run_index_rows)
            if skipped_ratio >= 0.50 and completed == 0:
                apply_contribution(contributions, "run_index_skipped_heavy", 0.18)
                risk += 0.18
            elif skipped_ratio >= 0.40:
                apply_contribution(contributions, "run_index_skipped_mid", 0.10)
                risk += 0.10

        lineage_counter = run_index_entry.get("lineages", {})
        if isinstance(lineage_counter, dict):
            lineages = [
                token
                for token, _count in sorted(
                    ((str(token), int(count)) for token, count in lineage_counter.items() if str(token)),
                    key=lambda item: (-item[1], item[0]),
                )
            ][:6]
        legacy_counter = run_index_entry.get("legacy_lineages", {})
        if isinstance(legacy_counter, dict):
            legacy_lineages = [
                token
                for token, _count in sorted(
                    ((str(token), int(count)) for token, count in legacy_counter.items() if str(token)),
                    key=lambda item: (-item[1], item[0]),
                )
            ][:6]
    else:
        apply_contribution(contributions, "run_index_missing_group", 0.18)
        risk += 0.18

    fullspan_verdict = ""
    fullspan_reason = ""
    fullspan_contract_pass = False
    if isinstance(fullspan_entry, dict) and fullspan_entry:
        fullspan_verdict = str(fullspan_entry.get("promotion_verdict") or "").strip().upper()
        fullspan_reason = canonical_reason(fullspan_entry)
        fullspan_contract_pass = bool(fullspan_entry.get("contract_hard_pass"))
        strict_gate_status = str(fullspan_entry.get("strict_gate_status") or "").strip().upper()

        if fullspan_verdict == "REJECT" or strict_gate_status == "FULLSPAN_PREFILTER_REJECT":
            apply_contribution(contributions, "fullspan_reject", 0.45)
            risk += 0.45
        elif fullspan_verdict == "PROMOTE_ELIGIBLE":
            apply_contribution(contributions, "fullspan_promote_eligible", -0.35)
            risk -= 0.35
        elif fullspan_verdict in {"PROMOTE_PENDING_CONFIRM", "PROMOTE_DEFER_CONFIRM"}:
            apply_contribution(contributions, "fullspan_pending_confirm", 0.08)
            risk += 0.08

        if fullspan_reason == "METRICS_MISSING":
            apply_contribution(contributions, "fullspan_metrics_missing", 0.12)
            risk += 0.12
        elif fullspan_reason and fullspan_reason not in {"UNKNOWN", "PASS", "OK"}:
            apply_contribution(contributions, f"fullspan_{fullspan_reason.lower()}", 0.20)
            risk += 0.20

        if fullspan_contract_pass:
            apply_contribution(contributions, "fullspan_contract_pass", -0.20)
            risk -= 0.20
    else:
        apply_contribution(contributions, "fullspan_missing", 0.02)
        risk += 0.02

    quarantine_total = 0
    quarantine_codes: dict[str, int] = {}
    if isinstance(quarantine_entry, dict) and quarantine_entry:
        quarantine_total = int(quarantine_entry.get("total", 0) or 0)
        quarantine_codes = {
            str(code): int(count)
            for code, count in (quarantine_entry.get("codes", {}) or {}).items()
            if str(code)
        }
        dominant_code = ""
        dominant_count = 0
        if quarantine_codes:
            dominant_code, dominant_count = max(quarantine_codes.items(), key=lambda item: item[1])

        if quarantine_total >= 12:
            apply_contribution(contributions, "quarantine_heavy", 0.65)
            risk += 0.65
        elif quarantine_total >= 6:
            apply_contribution(contributions, "quarantine_mid", 0.45)
            risk += 0.45
        elif quarantine_total >= 3:
            apply_contribution(contributions, "quarantine_light", 0.30)
            risk += 0.30
        elif quarantine_total > 0:
            apply_contribution(contributions, "quarantine_singletons", 0.15)
            risk += 0.15

        if dominant_code in DETERMINISTIC_CODES and dominant_count >= 3:
            validation_error = True
            apply_contribution(contributions, "deterministic_validation_error", 0.20)
            risk += 0.20
    elif quarantine_active:
        apply_contribution(contributions, "quarantine_global_active", 0.03)
        risk += 0.03

    risk = max(0.0, min(1.0, risk))

    if risk >= reject_threshold:
        decision = "reject"
    elif risk >= refine_threshold:
        decision = "refine"
    else:
        decision = "allow"

    positive_contribs = [item for item in contributions if float(item.get("delta", 0.0)) > 0]
    if positive_contribs:
        reason = sorted(positive_contribs, key=lambda item: (-float(item["delta"]), str(item["key"])))[0]["key"]
    else:
        reason = "healthy_signal"

    evidence = {
        "run_group": run_group,
        "queue_rows": int(queue_rows),
        "queue_status": queue_status,
        "run_index_rows": int(run_index_rows),
        "run_index_status": run_index_status,
        "metrics_missing_ratio": round(metrics_missing_ratio, 6),
        "zero_activity_ratio": round(zero_activity_ratio, 6),
        "informative_completed_ratio": round(informative_ratio, 6),
        "fullspan_verdict": fullspan_verdict,
        "fullspan_reason": fullspan_reason,
        "fullspan_contract_pass": bool(fullspan_contract_pass),
        "quarantine_total": int(quarantine_total),
        "quarantine_codes": quarantine_codes,
        "lineage_tokens": lineages,
        "lineage_uids": lineages,
        "legacy_evo_tokens": legacy_lineages,
        "contributions": contributions,
    }

    return (
        {
            "decision": decision,
            "risk_score": round(float(risk), 4),
            "reason": reason,
            "evidence": evidence,
        },
        lineages,
        validation_error,
    )


def neutral_queue_decision(queue_key: str, queue_rows: int, queue_status: dict[str, int]) -> dict[str, Any]:
    run_group = queue_run_group(queue_key)
    return {
        "decision": "refine",
        "risk_score": 0.5,
        "reason": "neutral_missing_inputs",
        "evidence": {
            "run_group": run_group,
            "queue_rows": int(queue_rows),
            "queue_status": queue_status,
            "note": "fail_closed_neutral",
        },
    }


def build_lineage_priority(queue_payloads: dict[str, dict[str, Any]]) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for _queue, payload in queue_payloads.items():
        if not isinstance(payload, dict):
            continue
        decision = str(payload.get("decision") or "").strip().lower()
        if decision == "reject":
            continue
        risk = parse_float(payload.get("risk_score"), 1.0)
        weight = max(0.0, 1.0 - risk)
        if decision == "refine":
            weight *= 0.7
        evidence = payload.get("evidence", {})
        if not isinstance(evidence, dict):
            continue
        tokens = evidence.get("lineage_uids", [])
        if not isinstance(tokens, list) or not tokens:
            tokens = evidence.get("lineage_tokens", [])
        if not isinstance(tokens, list):
            continue
        for idx, token in enumerate(tokens):
            t = str(token or "").strip().lower()
            if not t:
                continue
            rank_penalty = 1.0 / float(1 + idx)
            scores[t] += weight * rank_penalty

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _score in ranked[:12]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build queue-level gate surrogate state.")
    parser.add_argument("--root", default="", help="App root (`coint4/`).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reject-threshold", type=float, default=0.75)
    parser.add_argument("--refine-threshold", type=float, default=0.45)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve() if str(args.root or "").strip() else Path(__file__).resolve().parents[2]

    aggregate_dir = root / "artifacts" / "wfa" / "aggregate"
    state_dir = aggregate_dir / ".autonomous"

    fullspan_path = state_dir / "fullspan_decision_state.json"
    quarantine_path = state_dir / "deterministic_quarantine.json"
    run_index_path = aggregate_dir / "rollup" / "run_index.csv"
    calibration_path = state_dir / "surrogate_calibration_state.json"
    output_path = state_dir / "gate_surrogate_state.json"
    log_path = state_dir / "gate_surrogate.log"
    lock_path = state_dir / "gate_surrogate.lock"

    reject_threshold = max(0.0, min(1.0, float(args.reject_threshold)))
    refine_threshold = max(0.0, min(1.0, float(args.refine_threshold)))
    if reject_threshold < refine_threshold:
        reject_threshold, refine_threshold = refine_threshold, reject_threshold

    state_dir.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return 0

        queue_files = collect_queue_files(aggregate_dir)
        queue_lineages = load_queue_lineage_maps(queue_files, root)
        fullspan = load_fullspan_state(fullspan_path, root)
        quarantine = load_quarantine(quarantine_path, root)
        run_index = load_run_index(run_index_path, root=root, queue_lineages=queue_lineages)
        calibration = load_calibration_state(
            calibration_path,
            default_reject=reject_threshold,
            default_refine=refine_threshold,
        )
        if calibration.get("applied"):
            reject_threshold = parse_float(calibration.get("applied_reject_threshold"), reject_threshold)
            refine_threshold = parse_float(calibration.get("applied_refine_threshold"), refine_threshold)
            if reject_threshold < refine_threshold:
                reject_threshold, refine_threshold = refine_threshold, reject_threshold

        queue_status_map: dict[str, tuple[int, dict[str, int]]] = {}

        for queue_path in queue_files:
            queue_key = safe_rel(queue_path, root)
            queue_status_map[queue_key] = load_queue_status(queue_path)

        all_queue_keys = set(queue_status_map.keys())
        all_queue_keys.update(fullspan.get("queues", {}).keys())
        all_queue_keys.update(quarantine.get("queues", {}).keys())

        inputs_ready = bool(
            fullspan.get("available")
            and fullspan.get("valid")
            and quarantine.get("available")
            and quarantine.get("valid")
            and run_index.get("available")
            and run_index.get("valid")
            and len(all_queue_keys) > 0
        )

        queues_payload: dict[str, dict[str, Any]] = {}
        decision_counts: Counter[str] = Counter()
        validation_hit_count = 0

        for queue_key in sorted(all_queue_keys):
            queue_rows, queue_status = queue_status_map.get(queue_key, (0, {}))
            if not inputs_ready:
                payload = neutral_queue_decision(queue_key, queue_rows, queue_status)
                queues_payload[queue_key] = payload
                decision_counts[payload["decision"]] += 1
                continue

            run_group = queue_run_group(queue_key)
            run_index_entry = (run_index.get("groups_data", {}) or {}).get(run_group)
            fullspan_entry = (fullspan.get("queues", {}) or {}).get(queue_key)
            quarantine_entry = (quarantine.get("queues", {}) or {}).get(queue_key)

            queue_payload, _lineages, validation_error = decide_queue(
                queue_key=queue_key,
                queue_rows=queue_rows,
                queue_status=queue_status,
                run_group=run_group,
                run_index_entry=run_index_entry,
                fullspan_entry=fullspan_entry,
                quarantine_entry=quarantine_entry,
                quarantine_active=bool(quarantine.get("active")),
                reject_threshold=reject_threshold,
                refine_threshold=refine_threshold,
            )
            queues_payload[queue_key] = queue_payload
            decision_counts[queue_payload["decision"]] += 1
            if validation_error:
                validation_hit_count += 1

        lineage_priority = build_lineage_priority(queues_payload) if inputs_ready else []

        repair_enabled = bool(inputs_ready and validation_hit_count > 0)
        repair_mode = {
            "enabled": repair_enabled,
            "validation_neighbor": 1 if repair_enabled else 0,
            "max_neighbor_attempts": 3 if repair_enabled else 0,
        }

        risk_values = [parse_float(payload.get("risk_score"), 0.5) for payload in queues_payload.values()]
        risk_mean = (sum(risk_values) / float(len(risk_values))) if risk_values else 0.0

        high_risk = sorted(
            (
                {
                    "queue": queue,
                    "risk_score": parse_float(payload.get("risk_score"), 0.0),
                    "decision": str(payload.get("decision") or ""),
                    "reason": str(payload.get("reason") or ""),
                }
                for queue, payload in queues_payload.items()
                if parse_float(payload.get("risk_score"), 0.0) >= reject_threshold
            ),
            key=lambda item: (-item["risk_score"], item["queue"]),
        )[:10]

        missing_inputs = []
        if not fullspan.get("available"):
            missing_inputs.append("fullspan_decision_state")
        if not quarantine.get("available"):
            missing_inputs.append("deterministic_quarantine")
        if not run_index.get("available"):
            missing_inputs.append("run_index")
        if len(queue_files) == 0:
            missing_inputs.append("run_queues")

        payload = {
            "version": 1,
            "ts": utc_now_iso(),
            "source": "gate_surrogate_agent",
            "hard_fail_risk_policy": {
                "enabled": bool(inputs_ready),
                "reject_threshold": round(reject_threshold, 4),
                "refine_threshold": round(refine_threshold, 4),
                "source": "calibration" if calibration.get("applied") else "default",
            },
            "lineage_priority": lineage_priority,
            "repair_mode": repair_mode,
            "inputs": {
                "fullspan_decision_state": {
                    "path": safe_rel(fullspan_path, root),
                    "available": bool(fullspan.get("available")),
                    "valid": bool(fullspan.get("valid")),
                    "queue_entries": int(len(fullspan.get("queues", {}))),
                },
                "deterministic_quarantine": {
                    "path": safe_rel(quarantine_path, root),
                    "available": bool(quarantine.get("available")),
                    "valid": bool(quarantine.get("valid")),
                    "entries": int(quarantine.get("entries", 0) or 0),
                    "dominant_code": str(quarantine.get("dominant_code") or ""),
                    "dominant_ratio": float(quarantine.get("dominant_ratio") or 0.0),
                    "active": bool(quarantine.get("active")),
                },
                "run_index": {
                    "path": safe_rel(run_index_path, root),
                    "available": bool(run_index.get("available")),
                    "valid": bool(run_index.get("valid")),
                    "rows": int(run_index.get("rows", 0) or 0),
                    "groups": int(run_index.get("groups", 0) or 0),
                },
                "queue_glob": {
                    "pattern": "artifacts/wfa/aggregate/*/run_queue.csv",
                    "count": int(len(queue_files)),
                },
                "calibration_state": {
                    "path": safe_rel(calibration_path, root),
                    "available": bool(calibration.get("available")),
                    "valid": bool(calibration.get("valid")),
                    "applied": bool(calibration.get("applied")),
                    "sample_size": int(calibration.get("sample_size", 0) or 0),
                    "reason": str(calibration.get("reason") or ""),
                    "applied_reject_threshold": round(parse_float(calibration.get("applied_reject_threshold"), reject_threshold), 4),
                    "applied_refine_threshold": round(parse_float(calibration.get("applied_refine_threshold"), refine_threshold), 4),
                    "ts": str(calibration.get("ts") or ""),
                },
                "missing_required_inputs": missing_inputs,
            },
            "summary": {
                "queues_total": int(len(queues_payload)),
                "decision_counts": dict(decision_counts),
                "risk_mean": round(float(risk_mean), 6),
                "high_risk": high_risk,
                "validation_hit_queues": int(validation_hit_count),
                "mode": "active" if inputs_ready else "neutral",
            },
            "queues": queues_payload,
        }

        if args.dry_run:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            dump_json(output_path, payload)

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{utc_now_iso()} | enabled={int(bool(inputs_ready))} queues={len(queues_payload)} "
                f"allow={decision_counts.get('allow', 0)} refine={decision_counts.get('refine', 0)} "
                f"reject={decision_counts.get('reject', 0)} validation_hits={validation_hit_count} "
                f"dry_run={int(bool(args.dry_run))}\n"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
