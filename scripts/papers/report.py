#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from lib_pipeline import (
    DEFAULT_DB_PATH,
    DEFAULT_SYNTHESIS_DIR,
    DEFAULT_TEXT_DIR,
    connect_db,
    finish_run,
    loads_json,
    start_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build markdown ingest report")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite path")
    parser.add_argument("--text-dir", default=str(DEFAULT_TEXT_DIR), help="Input texts dir")
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_SYNTHESIS_DIR / "ingest_report.md"),
        help="Markdown report path",
    )
    return parser.parse_args()


def _format_title(row) -> str:
    return row["title"] or row["file_name"] or row["id"]


def _top_items(counter: Counter[str], limit: int = 10) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]


def main() -> int:
    args = parse_args()
    db_path = Path(args.db_path)
    text_dir = Path(args.text_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conn = connect_db(db_path)
    run_id = start_run(
        conn,
        kind="papers_report",
        params={"db_path": str(db_path), "text_dir": str(text_dir), "output_path": str(output_path)},
    )

    rows = conn.execute(
        """
        SELECT id, text_path, file_name, title, year, parse_status, parse_error,
               text_len, tags_json, priority_score, priority_reasons_json, dup_group_id
        FROM papers
        """
    ).fetchall()

    total_text_files = len(list(text_dir.rglob("*.txt")))
    db_count = len(rows)

    duplicate_groups = {
        row["dup_group_id"] for row in rows if row["dup_group_id"]
    }
    duplicate_papers = sum(1 for row in rows if row["dup_group_id"])

    method_counter: Counter[str] = Counter()
    practical_counter: Counter[str] = Counter()
    market_counter: Counter[str] = Counter()

    for row in rows:
        tags = loads_json(row["tags_json"], {})
        if not isinstance(tags, dict):
            continue
        for item in tags.get("method_family", []):
            if isinstance(item, str):
                method_counter[item] += 1
        for item in tags.get("practical", []):
            if isinstance(item, str):
                practical_counter[item] += 1
        for item in tags.get("market", []):
            if isinstance(item, str):
                market_counter[item] += 1

    top_priority = sorted(
        rows,
        key=lambda row: (
            -(row["priority_score"] or 0),
            -(row["year"] or 0),
            row["id"],
        ),
    )[:10]

    problematic = [
        row
        for row in rows
        if row["parse_status"] in {"error", "empty", "short"}
        or (row["text_len"] is not None and row["text_len"] < 200)
    ]

    generated_at = datetime.now(timezone.utc).isoformat()
    lines: list[str] = []
    lines.append("# Papers Ingest Report")
    lines.append("")
    lines.append(f"- Generated at: {generated_at}")
    lines.append("")

    lines.append("## Summary")
    lines.append(f"- Total text files discovered: {total_text_files}")
    lines.append(f"- Papers in DB: {db_count}")
    lines.append(f"- Duplicate groups: {len(duplicate_groups)}")
    lines.append(f"- Papers in duplicate groups: {duplicate_papers}")
    lines.append("")

    lines.append("## Top Tags")
    lines.append("### method_family")
    for tag, count in _top_items(method_counter):
        lines.append(f"- {tag}: {count}")
    if not method_counter:
        lines.append("- none")
    lines.append("")

    lines.append("### practical")
    for tag, count in _top_items(practical_counter):
        lines.append(f"- {tag}: {count}")
    if not practical_counter:
        lines.append("- none")
    lines.append("")

    lines.append("### market")
    for tag, count in _top_items(market_counter):
        lines.append(f"- {tag}: {count}")
    if not market_counter:
        lines.append("- none")
    lines.append("")

    lines.append("## Top 10 By Priority")
    if top_priority:
        for index, row in enumerate(top_priority, start=1):
            reasons = loads_json(row["priority_reasons_json"], [])
            if not isinstance(reasons, list):
                reasons = []
            reason_text = ", ".join(str(reason) for reason in reasons[:5]) if reasons else "none"
            title = _format_title(row)
            score = row["priority_score"] if row["priority_score"] is not None else 0
            year = row["year"] if row["year"] is not None else "n/a"
            lines.append(
                f"- {index}. [{score}] {title} (paper_id={row['id']}, year={year}, file={row['file_name']})"
            )
            lines.append(f"- reasons: {reason_text}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Problematic Entries")
    if problematic:
        for row in problematic:
            title = _format_title(row)
            parse_error = row["parse_error"] or "-"
            lines.append(
                f"- paper_id={row['id']}, title={title}, status={row['parse_status']}, "
                f"len={row['text_len']}, error={parse_error}, path={row['text_path']}"
            )
    else:
        lines.append("- none")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")

    stats = {
        "total_text_files": total_text_files,
        "papers_in_db": db_count,
        "duplicate_groups": len(duplicate_groups),
        "duplicate_papers": duplicate_papers,
        "problematic": len(problematic),
        "report_path": str(output_path),
    }

    finish_run(conn, run_id, stats)
    conn.close()
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
