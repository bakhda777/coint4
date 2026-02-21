#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from lib_pipeline import (
    DEFAULT_DB_PATH,
    DEFAULT_DIGESTS_DIR,
    DEFAULT_TEXT_DIR,
    DIGEST_SCHEMA_VERSION,
    build_digest,
    build_excerpt,
    compute_keyword_hits,
    connect_db,
    dumps_json,
    ensure_pipeline_dirs,
    extract_metadata,
    finish_run,
    infer_tags,
    normalize_text,
    paper_id_from_text_sha,
    read_text_file,
    resolve_unique_paper_id,
    sha256_text,
    sqlite_write_section,
    start_run,
    tags_version_matches,
    upsert_fts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index papers/text into sqlite + digest + FTS")
    parser.add_argument("--text-dir", default=str(DEFAULT_TEXT_DIR), help="Directory with .txt files")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite path")
    parser.add_argument("--digests-dir", default=str(DEFAULT_DIGESTS_DIR), help="Digest output dir")
    return parser.parse_args()


def should_skip(existing_row, text_sha256: str) -> bool:
    if existing_row is None:
        return False
    if existing_row["text_sha256"] != text_sha256:
        return False
    if not tags_version_matches(existing_row["tags_json"]):
        return False
    digest_path = existing_row["digest_path"]
    if not digest_path:
        return False
    if not Path(digest_path).exists():
        return False
    return True


def main() -> int:
    args = parse_args()
    text_dir = Path(args.text_dir)
    db_path = Path(args.db_path)
    digests_dir = Path(args.digests_dir)

    ensure_pipeline_dirs(text_dir=text_dir, digests_dir=digests_dir)
    conn = connect_db(db_path)

    files = sorted(path for path in text_dir.rglob("*.txt") if path.is_file())

    run_id = start_run(
        conn,
        kind="papers_index",
        params={
            "text_dir": str(text_dir),
            "db_path": str(db_path),
            "digests_dir": str(digests_dir),
            "digest_schema_version": DIGEST_SCHEMA_VERSION,
        },
    )

    stats: dict[str, int] = {
        "total_files": len(files),
        "new": 0,
        "updated": 0,
        "skipped": 0,
        "errors": 0,
        "decode_fallback": 0,
        "empty_or_short": 0,
    }

    try:
        for path in files:
            rel_path = path.as_posix()
            existing = conn.execute("SELECT * FROM papers WHERE text_path = ?", (rel_path,)).fetchone()

            try:
                raw_text, _, decode_error = read_text_file(path)
                clean_text = normalize_text(raw_text)
                text_len = len(clean_text)
                text_sha256 = sha256_text(clean_text)
                if decode_error:
                    stats["decode_fallback"] += 1

                if should_skip(existing, text_sha256):
                    stats["skipped"] += 1
                    continue

                base_paper_id = paper_id_from_text_sha(text_sha256)
                paper_id = resolve_unique_paper_id(conn, base_paper_id, rel_path)

                metadata = extract_metadata(clean_text)
                digest = build_digest(clean_text)
                excerpt = build_excerpt(clean_text)
                keyword_hits = compute_keyword_hits(clean_text)
                tags = infer_tags(keyword_hits, clean_text)

                parse_status = "ok"
                parse_error = None
                if text_len == 0:
                    parse_status = "empty"
                    parse_error = "empty text"
                    stats["empty_or_short"] += 1
                elif text_len < 200:
                    parse_status = "short"
                    parse_error = "very short text"
                    stats["empty_or_short"] += 1

                if decode_error:
                    parse_error = f"{parse_error + '; ' if parse_error else ''}decode fallback latin-1"

                digest_path = digests_dir / f"{paper_id}.txt"
                digest_path.write_text(digest, encoding="utf-8")
                digest_sha = sha256_text(f"{DIGEST_SCHEMA_VERSION}\n{digest}")

                now = datetime.now(timezone.utc).isoformat()
                added_at = existing["added_at"] if existing else now

                with sqlite_write_section(conn):
                    if existing and existing["id"] != paper_id:
                        conn.execute("DELETE FROM papers_fts WHERE paper_id = ?", (existing["id"],))
                        conn.execute("DELETE FROM papers WHERE id = ?", (existing["id"],))

                    conn.execute(
                        """
                        INSERT INTO papers (
                            id, text_path, text_sha256, file_name, title, year, doi, url,
                            added_at, updated_at, text_len, parse_status, parse_error,
                            digest_path, digest_sha256, tags_json, keyword_hits_json,
                            priority_score, priority_reasons_json, dup_group_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            text_path = excluded.text_path,
                            text_sha256 = excluded.text_sha256,
                            file_name = excluded.file_name,
                            title = excluded.title,
                            year = excluded.year,
                            doi = excluded.doi,
                            url = excluded.url,
                            added_at = excluded.added_at,
                            updated_at = excluded.updated_at,
                            text_len = excluded.text_len,
                            parse_status = excluded.parse_status,
                            parse_error = excluded.parse_error,
                            digest_path = excluded.digest_path,
                            digest_sha256 = excluded.digest_sha256,
                            tags_json = excluded.tags_json,
                            keyword_hits_json = excluded.keyword_hits_json,
                            priority_score = NULL,
                            priority_reasons_json = NULL,
                            dup_group_id = NULL
                        """,
                        (
                            paper_id,
                            rel_path,
                            text_sha256,
                            path.name,
                            metadata["title"],
                            metadata["year"],
                            metadata["doi"],
                            metadata["url"],
                            added_at,
                            now,
                            text_len,
                            parse_status,
                            parse_error,
                            digest_path.as_posix(),
                            digest_sha,
                            dumps_json(tags),
                            dumps_json(keyword_hits),
                            None,
                            None,
                            None,
                        ),
                    )

                    upsert_fts(conn, paper_id=paper_id, title=metadata["title"], digest=digest, excerpt=excerpt)

                if existing:
                    stats["updated"] += 1
                else:
                    stats["new"] += 1

            except Exception as exc:  # noqa: BLE001
                stats["errors"] += 1
                err_now = datetime.now(timezone.utc).isoformat()
                text_sha256 = None
                try:
                    text_sha256 = sha256_text(normalize_text(path.read_text(encoding="utf-8", errors="replace")))
                except Exception:  # noqa: BLE001
                    text_sha256 = ""
                base_id = paper_id_from_text_sha(text_sha256) if text_sha256 else sha256_text(rel_path)[:12]
                paper_id = resolve_unique_paper_id(conn, base_id, rel_path)
                with sqlite_write_section(conn):
                    conn.execute(
                        """
                        INSERT INTO papers (
                            id, text_path, text_sha256, file_name, title, year, doi, url,
                            added_at, updated_at, text_len, parse_status, parse_error,
                            digest_path, digest_sha256, tags_json, keyword_hits_json,
                            priority_score, priority_reasons_json, dup_group_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            text_path = excluded.text_path,
                            text_sha256 = excluded.text_sha256,
                            file_name = excluded.file_name,
                            updated_at = excluded.updated_at,
                            text_len = excluded.text_len,
                            parse_status = excluded.parse_status,
                            parse_error = excluded.parse_error,
                            priority_score = NULL,
                            priority_reasons_json = NULL,
                            dup_group_id = NULL
                        """,
                        (
                            paper_id,
                            rel_path,
                            text_sha256,
                            path.name,
                            None,
                            None,
                            None,
                            None,
                            err_now,
                            err_now,
                            0,
                            "error",
                            str(exc)[:2000],
                            None,
                            None,
                            dumps_json({"_schema_version": DIGEST_SCHEMA_VERSION}),
                            dumps_json({}),
                            None,
                            None,
                            None,
                        ),
                    )

        finish_run(conn, run_id, stats)
        print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
