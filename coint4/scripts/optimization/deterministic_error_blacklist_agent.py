#!/usr/bin/env python3
"""Build deterministic-error blacklist hints for search seeding.

Consumes deterministic_quarantine.json and produces search_policy_blacklist.json
with conservative generation caps while deterministic config errors dominate.
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


EVO_RE = re.compile(r"\b(evo_[0-9a-f]{8,64})\b", re.IGNORECASE)


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


def extract_evo_token(*texts: str) -> str:
    for text in texts:
        value = str(text or "").strip()
        if not value:
            continue
        match = EVO_RE.search(value)
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


def load_queue_lineage(queue_path: Path, *, root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not queue_path.exists():
        return out
    try:
        with queue_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                results_key = normalize_lookup_path(row.get("results_dir"), root=root)
                if not results_key or results_key in out:
                    continue
                lineage_uid = extract_lineage_uid(row)
                if not lineage_uid:
                    lineage_uid = extract_evo_token(
                        str(row.get("config_path") or ""),
                        str(row.get("results_dir") or ""),
                    )
                if lineage_uid:
                    out[results_key] = lineage_uid
    except Exception:
        return {}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic error blacklist builder.")
    parser.add_argument("--root", default="", help="App root (`coint4/`).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve() if str(args.root or "").strip() else Path(__file__).resolve().parents[2]
    state_dir = root / "artifacts" / "wfa" / "aggregate" / ".autonomous"
    source_path = state_dir / "deterministic_quarantine.json"
    output_path = state_dir / "search_policy_blacklist.json"
    lock_path = state_dir / "deterministic_error_blacklist.lock"
    log_path = state_dir / "deterministic_error_blacklist.log"

    state_dir.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return 0

        source = load_json(source_path, {})
        entries = source.get("entries", []) if isinstance(source, dict) else []
        if not isinstance(entries, list):
            entries = []

        recent = [entry for entry in entries[-400:] if isinstance(entry, dict)]
        codes = Counter()
        lineage_counts = Counter()
        code_lineage_counts: dict[str, Counter[str]] = defaultdict(Counter)
        queue_cache: dict[str, dict[str, str]] = {}

        for entry in recent:
            code = str(entry.get("code") or "").strip().upper()
            if not code:
                continue
            codes[code] += 1
            queue_key = normalize_lookup_path(entry.get("queue"), root=root)
            results_key = normalize_lookup_path(entry.get("results_dir"), root=root)
            lineage_uid = ""
            if queue_key:
                lineage_map = queue_cache.get(queue_key)
                if lineage_map is None:
                    lineage_map = load_queue_lineage(Path(queue_key), root=root)
                    queue_cache[queue_key] = lineage_map
                lineage_uid = str(lineage_map.get(results_key) or "").strip().lower()
            if not lineage_uid:
                lineage_uid = extract_lineage_uid(entry)
            if not lineage_uid:
                lineage_uid = extract_evo_token(
                    str(entry.get("run_id") or ""),
                    str(entry.get("results_dir") or ""),
                    str(entry.get("queue") or ""),
                )
            if lineage_uid:
                lineage_counts[lineage_uid] += 1
                code_lineage_counts[code][lineage_uid] += 1

        total = sum(codes.values())
        dominant_code = ""
        dominant_count = 0
        dominant_ratio = 0.0
        if codes:
            dominant_code, dominant_count = codes.most_common(1)[0]
            if total > 0:
                dominant_ratio = float(dominant_count) / float(total)

        deterministic_codes = {
            "CONFIG_VALIDATION_ERROR",
            "MAX_VAR_MULTIPLIER_INVALID",
            "MAX_CORRELATION_INVALID",
            "NON_POSITIVE_THRESHOLD",
            "INVALID_PARAM",
        }

        active = bool(total >= 6 and dominant_code in deterministic_codes and dominant_ratio >= 0.40)

        blocked_lineage_uids = sorted([token for token, count in lineage_counts.items() if count >= 3])[:200]

        caps = {
            "max_changed_keys_cap": 4,
            "dedupe_distance_floor": 0.06,
            "num_variants_cap": 10,
            "policy_scale": "auto",
        }
        if active:
            if dominant_code == "CONFIG_VALIDATION_ERROR":
                caps = {
                    "max_changed_keys_cap": 2,
                    "dedupe_distance_floor": 0.02,
                    "num_variants_cap": 64,
                    "policy_scale": "micro",
                }
            else:
                caps = {
                    "max_changed_keys_cap": 3,
                    "dedupe_distance_floor": 0.04,
                    "num_variants_cap": 48,
                    "policy_scale": "micro",
                }

        payload = {
            "version": 1,
            "ts": utc_now_iso(),
            "active": active,
            "source": str(source_path),
            "stats": {
                "recent_entries": len(recent),
                "total_coded": total,
                "dominant_code": dominant_code,
                "dominant_count": dominant_count,
                "dominant_ratio": dominant_ratio,
                "codes": dict(codes),
            },
            "blocked_lineage_uids": blocked_lineage_uids,
            "blocked_evo_uids": blocked_lineage_uids,
            "blocked_by_code": {
                code: [token for token, _count in counter.most_common(50)]
                for code, counter in code_lineage_counts.items()
            },
            "recommended_caps": caps,
        }

        if not args.dry_run:
            dump_json(output_path, payload)

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{utc_now_iso()} | active={int(active)} dominant_code={dominant_code or '-'} ratio={dominant_ratio:.3f} "
                f"blocked_lineage={len(blocked_lineage_uids)} dry_run={int(bool(args.dry_run))}\n"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
