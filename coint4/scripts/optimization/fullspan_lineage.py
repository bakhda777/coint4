#!/usr/bin/env python3
"""Lineage utilities for strict fullspan confirmation.

Fail-closed contract:
- confirm replays are counted only from registered confirm queue lineage.
- candidate identity is stable via candidate_uid.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


EVO_TOKEN_RE = re.compile(r"\b(evo_[0-9a-f]{8,64})\b", re.IGNORECASE)


@dataclass(frozen=True)
class ConfirmStats:
    candidate_uid: str
    confirmed_count: int
    confirmed_run_groups: List[str]
    observed_run_groups: List[str]
    confirmed_lineage_keys: List[str]
    observed_lineage_keys: List[str]
    confirmed_group_lineage_keys: Dict[str, List[str]]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_text(value: str | None) -> str:
    return str(value or "").strip()


def _extract_uid_from_metadata_payload(raw: object) -> str:
    queue: list[object] = [raw]
    seen: set[int] = set()
    while queue:
        current = queue.pop(0)
        ident = id(current)
        if ident in seen:
            continue
        seen.add(ident)
        if isinstance(current, dict):
            for key in ("lineage_uid", "candidate_uid"):
                value = _normalize_text(current.get(key))
                if value:
                    return value.lower()
            for value in current.values():
                if isinstance(value, (dict, list)):
                    queue.append(value)
                elif isinstance(value, str):
                    text = _normalize_text(value)
                    if not text:
                        continue
                    if text.startswith("{") or text.startswith("["):
                        try:
                            parsed = json.loads(text)
                        except Exception:
                            continue
                        queue.append(parsed)
        elif isinstance(current, list):
            queue.extend(current)
        elif isinstance(current, str):
            text = _normalize_text(current)
            if text and (text.startswith("{") or text.startswith("[")):
                try:
                    parsed = json.loads(text)
                except Exception:
                    continue
                queue.append(parsed)
    return ""


def derive_candidate_uid(
    *,
    top_run_group: str = "",
    top_variant: str = "",
    top_score: str = "",
    top_config: str = "",
    lineage_uid: str = "",
    top_metadata: str = "",
) -> str:
    explicit_uid = _normalize_text(lineage_uid).lower()
    if explicit_uid:
        return explicit_uid

    metadata_uid = _extract_uid_from_metadata_payload(top_metadata)
    if metadata_uid:
        return metadata_uid

    config_uid = _extract_uid_from_metadata_payload(top_config)
    if config_uid:
        return config_uid

    tokens = [
        _normalize_text(top_variant),
        _normalize_text(top_run_group),
        _normalize_text(top_config),
    ]
    for token in tokens:
        if not token:
            continue
        match = EVO_TOKEN_RE.search(token)
        if match:
            return match.group(1).lower()

    seed_parts = [
        _normalize_text(top_run_group).lower(),
        _normalize_text(top_variant).lower(),
        _normalize_text(top_score),
        _normalize_text(top_config).lower(),
    ]
    if not any(seed_parts):
        return ""
    digest = hashlib.sha1("|".join(seed_parts).encode("utf-8")).hexdigest()[:16]
    return f"cand_{digest}"


def derive_lineage_key(
    *,
    candidate_uid: str,
    source_queue_rel: str,
    top_run_group: str,
    top_variant: str,
    confirm_run_group: str,
    dispatch_id: str = "",
    queue_rel: str = "",
) -> str:
    seed_parts = [
        _normalize_text(candidate_uid).lower(),
        _normalize_text(source_queue_rel).lower(),
        _normalize_text(top_run_group).lower(),
        _normalize_text(top_variant).lower(),
        _normalize_text(confirm_run_group).lower(),
        _normalize_text(dispatch_id).lower(),
        _normalize_text(queue_rel).lower(),
    ]
    digest = hashlib.sha1("|".join(seed_parts).encode("utf-8")).hexdigest()[:20]
    return f"ln_{digest}"


def register_confirm_queue(
    *,
    registry_path: Path,
    queue_rel: str,
    source_queue_rel: str,
    candidate_uid: str,
    lineage_uid: str = "",
    top_run_group: str = "",
    top_variant: str = "",
    dispatch_id: str = "",
) -> dict:
    queue_rel = _normalize_text(queue_rel)
    source_queue_rel = _normalize_text(source_queue_rel)
    candidate_uid = _normalize_text(candidate_uid)
    lineage_uid = _normalize_text(lineage_uid).lower()
    if lineage_uid and not candidate_uid:
        candidate_uid = lineage_uid
    if candidate_uid:
        candidate_uid = candidate_uid.lower()
    top_run_group = _normalize_text(top_run_group)
    top_variant = _normalize_text(top_variant)
    dispatch_id = _normalize_text(dispatch_id)

    if not queue_rel or not candidate_uid:
        return {"ok": False, "reason": "missing_queue_or_candidate_uid"}

    state = _load_json(registry_path, {})
    if not isinstance(state, dict):
        state = {}
    queues = state.get("queues", {})
    if not isinstance(queues, dict):
        queues = {}

    confirm_run_group = Path(queue_rel).parent.name
    lineage_key = derive_lineage_key(
        candidate_uid=candidate_uid,
        source_queue_rel=source_queue_rel,
        top_run_group=top_run_group,
        top_variant=top_variant,
        confirm_run_group=confirm_run_group,
        dispatch_id=dispatch_id,
        queue_rel=queue_rel,
    )
    ts = utc_now_iso()
    entry = {
        "queue_rel": queue_rel,
        "source_queue_rel": source_queue_rel,
        "candidate_uid": candidate_uid,
        "lineage_uid": lineage_uid or candidate_uid,
        "confirm_run_group": confirm_run_group,
        "top_run_group": top_run_group,
        "top_variant": top_variant,
        "dispatch_id": dispatch_id,
        "lineage_key": lineage_key,
        "last_update": ts,
    }

    prev = queues.get(queue_rel)
    if not isinstance(prev, dict):
        entry["created_at"] = ts
    else:
        entry["created_at"] = str(prev.get("created_at") or ts)

    queues[queue_rel] = entry
    state["queues"] = queues
    state["version"] = int(state.get("version", 1) or 1)
    _dump_json(registry_path, state)

    return {
        "ok": True,
        "queue_rel": queue_rel,
        "candidate_uid": candidate_uid,
        "lineage_uid": lineage_uid or candidate_uid,
        "confirm_run_group": confirm_run_group,
        "lineage_key": lineage_key,
    }


def _registry_entries_for_candidate(registry_path: Path, candidate_uid: str) -> list[dict[str, str]]:
    state = _load_json(registry_path, {})
    if not isinstance(state, dict):
        return []
    queues = state.get("queues", {})
    if not isinstance(queues, dict):
        return []

    candidate_uid_l = _normalize_text(candidate_uid).lower()
    entries: list[dict[str, str]] = []
    seen = set()
    for entry in queues.values():
        if not isinstance(entry, dict):
            continue
        uid = _normalize_text(entry.get("candidate_uid") or entry.get("lineage_uid")).lower()
        if not uid or uid != candidate_uid_l:
            continue
        queue_rel = _normalize_text(entry.get("queue_rel"))
        grp = _normalize_text(entry.get("confirm_run_group"))
        if not grp:
            if queue_rel:
                grp = Path(queue_rel).parent.name
        source_queue_rel = _normalize_text(entry.get("source_queue_rel"))
        top_run_group = _normalize_text(entry.get("top_run_group"))
        top_variant = _normalize_text(entry.get("top_variant"))
        dispatch_id = _normalize_text(entry.get("dispatch_id"))
        lineage_key = _normalize_text(entry.get("lineage_key"))
        if not lineage_key:
            lineage_key = derive_lineage_key(
                candidate_uid=candidate_uid,
                source_queue_rel=source_queue_rel,
                top_run_group=top_run_group,
                top_variant=top_variant,
                confirm_run_group=grp,
                dispatch_id=dispatch_id,
                queue_rel=queue_rel,
            )
        dedupe_key = (grp, lineage_key, queue_rel)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        entries.append(
            {
                "queue_rel": queue_rel,
                "source_queue_rel": source_queue_rel,
                "confirm_run_group": grp,
                "lineage_uid": _normalize_text(entry.get("lineage_uid") or uid).lower(),
                "lineage_key": lineage_key,
                "dispatch_id": dispatch_id,
                "top_run_group": top_run_group,
                "top_variant": top_variant,
            }
        )
    return entries


def _is_true(raw: str | None) -> bool:
    return _normalize_text(raw).lower() in {"1", "true", "yes", "y", "on"}


def _run_index_rows(path: Path) -> Iterable[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def count_confirms_by_lineage(*, run_index_path: Path, registry_path: Path, candidate_uid: str) -> ConfirmStats:
    candidate_uid = _normalize_text(candidate_uid)
    if not candidate_uid:
        return ConfirmStats(
            candidate_uid="",
            confirmed_count=0,
            confirmed_run_groups=[],
            observed_run_groups=[],
            confirmed_lineage_keys=[],
            observed_lineage_keys=[],
            confirmed_group_lineage_keys={},
        )

    observed_entries = _registry_entries_for_candidate(registry_path, candidate_uid)
    if not observed_entries:
        return ConfirmStats(
            candidate_uid=candidate_uid,
            confirmed_count=0,
            confirmed_run_groups=[],
            observed_run_groups=[],
            confirmed_lineage_keys=[],
            observed_lineage_keys=[],
            confirmed_group_lineage_keys={},
        )

    observed_groups: list[str] = []
    observed_lineage_keys: list[str] = []
    entries_by_group: Dict[str, list[dict[str, str]]] = {}
    seen_groups = set()
    seen_lineage = set()
    for entry in observed_entries:
        group = _normalize_text(entry.get("confirm_run_group"))
        lineage_key = _normalize_text(entry.get("lineage_key"))
        if group and group not in seen_groups:
            seen_groups.add(group)
            observed_groups.append(group)
        if lineage_key and lineage_key not in seen_lineage:
            seen_lineage.add(lineage_key)
            observed_lineage_keys.append(lineage_key)
        if group:
            entries_by_group.setdefault(group, []).append(entry)

    rows = list(_run_index_rows(run_index_path))
    by_group: Dict[str, dict[str, bool]] = {
        grp: {"has_holdout": False, "has_stress": False} for grp in observed_groups
    }

    for row in rows:
        run_group = _normalize_text(row.get("run_group"))
        if run_group not in by_group:
            continue
        status = _normalize_text(row.get("status")).lower()
        if status != "completed":
            continue
        if not _is_true(row.get("metrics_present")):
            continue

        run_id = _normalize_text(row.get("run_id")).lower()
        if run_id.startswith("holdout_"):
            by_group[run_group]["has_holdout"] = True
        elif run_id.startswith("stress_"):
            by_group[run_group]["has_stress"] = True

    confirmed: list[str] = []
    confirmed_lineage_keys: list[str] = []
    confirmed_group_lineage_keys: dict[str, list[str]] = {}
    seen_confirmed_lineage = set()
    for grp in observed_groups:
        marks = by_group.get(grp, {})
        if bool(marks.get("has_holdout")) and bool(marks.get("has_stress")):
            confirmed.append(grp)
            group_lineage: list[str] = []
            for entry in entries_by_group.get(grp, []):
                lineage_key = _normalize_text(entry.get("lineage_key"))
                if not lineage_key:
                    continue
                if lineage_key not in group_lineage:
                    group_lineage.append(lineage_key)
                if lineage_key not in seen_confirmed_lineage:
                    seen_confirmed_lineage.add(lineage_key)
                    confirmed_lineage_keys.append(lineage_key)
            confirmed_group_lineage_keys[grp] = group_lineage

    return ConfirmStats(
        candidate_uid=candidate_uid,
        confirmed_count=len(confirmed),
        confirmed_run_groups=confirmed,
        observed_run_groups=observed_groups,
        confirmed_lineage_keys=confirmed_lineage_keys,
        observed_lineage_keys=observed_lineage_keys,
        confirmed_group_lineage_keys=confirmed_group_lineage_keys,
    )


def _cmd_derive(args: argparse.Namespace) -> int:
    uid = derive_candidate_uid(
        top_run_group=args.top_run_group,
        top_variant=args.top_variant,
        top_score=args.top_score,
        top_config=args.top_config,
        lineage_uid=args.lineage_uid,
        top_metadata=args.top_metadata,
    )
    if args.json:
        print(json.dumps({"candidate_uid": uid}, ensure_ascii=False))
    else:
        print(uid)
    return 0


def _cmd_register(args: argparse.Namespace) -> int:
    out = register_confirm_queue(
        registry_path=Path(args.registry),
        queue_rel=args.queue_rel,
        source_queue_rel=args.source_queue_rel,
        candidate_uid=args.candidate_uid,
        lineage_uid=args.lineage_uid,
        top_run_group=args.top_run_group,
        top_variant=args.top_variant,
        dispatch_id=args.dispatch_id,
    )
    if args.json:
        print(json.dumps(out, ensure_ascii=False))
    elif out.get("ok"):
        print("ok")
    else:
        print(str(out.get("reason") or "error"))
    return 0 if out.get("ok") else 1


def _cmd_count(args: argparse.Namespace) -> int:
    stats = count_confirms_by_lineage(
        run_index_path=Path(args.run_index),
        registry_path=Path(args.registry),
        candidate_uid=args.candidate_uid,
    )
    payload = {
        "candidate_uid": stats.candidate_uid,
        "confirmed_count": int(stats.confirmed_count),
        "confirmed_run_groups": stats.confirmed_run_groups,
        "observed_run_groups": stats.observed_run_groups,
        "confirmed_lineage_keys": stats.confirmed_lineage_keys,
        "observed_lineage_keys": stats.observed_lineage_keys,
        "confirmed_group_lineage_keys": stats.confirmed_group_lineage_keys,
    }
    if args.value == "count":
        print(payload["confirmed_count"])
    elif args.value == "confirmed_run_groups":
        print("||".join(payload["confirmed_run_groups"]))
    else:
        print(json.dumps(payload, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lineage helper for strict fullspan confirmations.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_derive = sub.add_parser("derive", help="Derive stable candidate_uid.")
    p_derive.add_argument("--top-run-group", default="")
    p_derive.add_argument("--top-variant", default="")
    p_derive.add_argument("--top-score", default="")
    p_derive.add_argument("--top-config", default="")
    p_derive.add_argument("--lineage-uid", default="")
    p_derive.add_argument("--top-metadata", default="")
    p_derive.add_argument("--json", action="store_true")
    p_derive.set_defaults(func=_cmd_derive)

    p_register = sub.add_parser("register", help="Register confirm queue lineage for a candidate_uid.")
    p_register.add_argument("--registry", required=True)
    p_register.add_argument("--queue-rel", required=True)
    p_register.add_argument("--source-queue-rel", required=True)
    p_register.add_argument("--candidate-uid", required=True)
    p_register.add_argument("--lineage-uid", default="")
    p_register.add_argument("--top-run-group", default="")
    p_register.add_argument("--top-variant", default="")
    p_register.add_argument("--dispatch-id", default="")
    p_register.add_argument("--json", action="store_true")
    p_register.set_defaults(func=_cmd_register)

    p_count = sub.add_parser("count", help="Count confirms by lineage registry.")
    p_count.add_argument("--registry", required=True)
    p_count.add_argument("--run-index", required=True)
    p_count.add_argument("--candidate-uid", required=True)
    p_count.add_argument("--value", choices=["json", "count", "confirmed_run_groups"], default="json")
    p_count.set_defaults(func=_cmd_count)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
