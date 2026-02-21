#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from lib_pipeline import (
    DEFAULT_DB_PATH,
    connect_db,
    finish_run,
    loads_json,
    normalize_title,
    sqlite_write_section,
    start_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deduplicate papers by text hash and title/year")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite path")
    return parser.parse_args()


def _find(parent: Dict[str, str], item: str) -> str:
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def _union(parent: Dict[str, str], a: str, b: str) -> None:
    root_a = _find(parent, a)
    root_b = _find(parent, b)
    if root_a != root_b:
        parent[root_b] = root_a


def _clean_duplicate_reasons(reasons_json: str | None) -> list[str]:
    reasons = loads_json(reasons_json, [])
    if not isinstance(reasons, list):
        return []
    result = []
    for reason in reasons:
        if isinstance(reason, str) and reason.startswith("duplicate"):
            continue
        result.append(reason)
    return result


def main() -> int:
    args = parse_args()
    conn = connect_db(db_path=Path(args.db_path))
    run_id = start_run(conn, kind="papers_dedup", params={"db_path": args.db_path})

    rows = conn.execute(
        "SELECT id, title, year, text_sha256, text_len, priority_reasons_json FROM papers"
    ).fetchall()

    parent: Dict[str, str] = {row["id"]: row["id"] for row in rows}

    hard_groups: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        hard_groups[row["text_sha256"]].append(row["id"])

    hard_group_count = 0
    for ids in hard_groups.values():
        if len(ids) < 2:
            continue
        hard_group_count += 1
        base = ids[0]
        for other in ids[1:]:
            _union(parent, base, other)

    soft_groups: Dict[tuple[str, int], List[str]] = defaultdict(list)
    for row in rows:
        norm_title = normalize_title(row["title"])
        if norm_title and row["year"]:
            soft_groups[(norm_title, int(row["year"]))].append(row["id"])

    soft_group_count = 0
    for ids in soft_groups.values():
        if len(ids) < 2:
            continue
        soft_group_count += 1
        base = ids[0]
        for other in ids[1:]:
            _union(parent, base, other)

    components: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        components[_find(parent, row["id"])].append(row["id"])

    row_by_id = {row["id"]: row for row in rows}

    duplicate_groups = 0
    duplicate_papers = 0
    updates: list[tuple[str | None, str, str]] = []

    for ids in components.values():
        if len(ids) < 2:
            paper_id = ids[0]
            reasons = _clean_duplicate_reasons(row_by_id[paper_id]["priority_reasons_json"])
            updates.append((None, json.dumps(reasons, ensure_ascii=False, sort_keys=True), paper_id))
            continue

        duplicate_groups += 1
        duplicate_papers += len(ids)

        sorted_ids = sorted(ids)
        group_seed = "|".join(sorted_ids)
        group_id = f"dup-{hashlib.sha256(group_seed.encode('utf-8')).hexdigest()[:12]}"

        canonical_id = sorted(
            ids,
            key=lambda paper_id: (
                -(row_by_id[paper_id]["text_len"] or 0),
                paper_id,
            ),
        )[0]

        for paper_id in ids:
            reasons = _clean_duplicate_reasons(row_by_id[paper_id]["priority_reasons_json"])
            reasons.append(f"duplicate_group:{group_id}")
            if paper_id == canonical_id:
                reasons.append("duplicate_canonical")
            else:
                reasons.append("duplicate_non_canonical")

            updates.append((group_id, json.dumps(reasons, ensure_ascii=False, sort_keys=True), paper_id))

    batch_size = 100
    for start in range(0, len(updates), batch_size):
        chunk = updates[start : start + batch_size]
        with sqlite_write_section(conn):
            for dup_group_id, reasons_json, paper_id in chunk:
                conn.execute(
                    "UPDATE papers SET dup_group_id = ?, priority_reasons_json = ? WHERE id = ?",
                    (dup_group_id, reasons_json, paper_id),
                )

    stats = {
        "total_papers": len(rows),
        "hard_groups": hard_group_count,
        "soft_groups": soft_group_count,
        "duplicate_groups": duplicate_groups,
        "duplicate_papers": duplicate_papers,
    }
    finish_run(conn, run_id, stats)
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
