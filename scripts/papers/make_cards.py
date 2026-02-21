#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from lib_pipeline import (
    CARD_SCHEMA_VERSION,
    DEFAULT_CARDS_DIR,
    DEFAULT_DB_PATH,
    DEFAULT_SYNTHESIS_DIR,
    build_excerpt,
    connect_db,
    extract_section_snippets,
    loads_json,
    normalize_text,
    read_text_file,
)

MAKE_CARDS_VERSION = "make-cards-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper cards via codex exec")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite path")
    parser.add_argument(
        "--queue-path",
        default=str(DEFAULT_SYNTHESIS_DIR / "priority_queue.jsonl"),
        help="Priority queue JSONL",
    )
    parser.add_argument("--cards-dir", default=str(DEFAULT_CARDS_DIR), help="Cards output dir")
    parser.add_argument(
        "--schema-path",
        default=str(DEFAULT_CARDS_DIR / "card.schema.json"),
        help="Card schema JSON path",
    )
    parser.add_argument(
        "--top",
        "--top-n",
        dest="top_n",
        type=int,
        default=int(os.getenv("PAPERS_CARDS_TOP_N", "30")),
        help="How many papers from queue to process",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("PAPERS_CARDS_BATCH_SIZE", "5")),
        help="How many papers to send in one codex exec call",
    )
    parser.add_argument(
        "--codex-cmd",
        default=os.getenv("PAPERS_CARDS_CODEX_CMD", "codex exec"),
        help="Command used for generation, prompt is passed via stdin",
    )
    parser.add_argument("--timeout-sec", type=int, default=120, help="Subprocess timeout for each codex call")
    parser.add_argument("--dry-run", action="store_true", help="Do not invoke codex, only print planned IDs")
    return parser.parse_args()


def _load_schema(schema_path: Path) -> dict[str, Any]:
    data = json.loads(schema_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("schema must be a JSON object")
    return data


def _match_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    return True


def _validate_by_schema(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    errors: list[str] = []

    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        if not _match_type(value, schema_type):
            errors.append(f"{path}: expected {schema_type}")
            return errors
    elif isinstance(schema_type, list):
        if not any(_match_type(value, entry) for entry in schema_type):
            errors.append(f"{path}: expected one of {schema_type}")
            return errors

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        errors.append(f"{path}: value not in enum")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            errors.append(f"{path}: minLength={schema['minLength']}")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            errors.append(f"{path}: maxLength={schema['maxLength']}")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            errors.append(f"{path}: minItems={schema['minItems']}")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            errors.append(f"{path}: maxItems={schema['maxItems']}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(value):
                errors.extend(_validate_by_schema(item, item_schema, path=f"{path}[{idx}]") )

    if isinstance(value, dict):
        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                if key not in value:
                    errors.append(f"{path}: missing required field '{key}'")

        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, child_schema in properties.items():
                if key in value and isinstance(child_schema, dict):
                    errors.extend(_validate_by_schema(value[key], child_schema, path=f"{path}.{key}"))

            if schema.get("additionalProperties") is False:
                for key in value:
                    if key not in properties:
                        errors.append(f"{path}: additionalProperties not allowed ('{key}')")

    return errors


def validate_card_payload(card: dict[str, Any], schema: dict[str, Any]) -> tuple[bool, list[str]]:
    if not isinstance(card, dict):
        return False, ["$ must be object"]
    errors = _validate_by_schema(card, schema)
    return (len(errors) == 0), errors


def _extract_best_json(stdout: str) -> Any:
    decoder = json.JSONDecoder()
    best_payload: Any = None
    best_rank = -1
    for idx, char in enumerate(stdout):
        if char not in "[{":
            continue
        try:
            candidate, _ = decoder.raw_decode(stdout[idx:])
        except json.JSONDecodeError:
            continue
        if not isinstance(candidate, (dict, list)):
            continue
        rank = 0
        if isinstance(candidate, dict) and isinstance(candidate.get("cards"), list):
            rank = 3
        elif isinstance(candidate, dict) and "paper_id" in candidate:
            rank = 2
        elif isinstance(candidate, list) and candidate and isinstance(candidate[0], dict):
            first = candidate[0]
            if "paper_id" in first:
                rank = 2
            else:
                rank = 1
        else:
            rank = 1

        if rank >= best_rank:
            best_rank = rank
            best_payload = candidate
    if best_payload is None:
        raise ValueError("codex output does not contain JSON payload")
    return best_payload


def _coerce_cards(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("cards"), list):
        return [card for card in payload["cards"] if isinstance(card, dict)]
    if isinstance(payload, list):
        return [card for card in payload if isinstance(card, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def _load_queue(queue_path: Path, top_n: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with queue_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                entries.append(row)
    entries.sort(key=lambda entry: (-int(entry.get("priority_score", 0)), entry.get("paper_id", "")))
    return entries[:top_n]


def _cached(card_path: Path, text_sha256: str, schema_version: str) -> bool:
    if not card_path.exists():
        return False
    try:
        payload = json.loads(card_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("text_sha256") == text_sha256 and payload.get("schema_version") == schema_version


def _load_context(text_path: str, digest_path: str | None) -> dict[str, str]:
    digest = ""
    if digest_path:
        path = Path(digest_path)
        if path.exists():
            digest = path.read_text(encoding="utf-8", errors="replace")

    excerpt = ""
    method_snippet = ""
    results_snippet = ""
    conclusion_snippet = ""

    path = Path(text_path)
    if path.exists():
        raw_text, _, _ = read_text_file(path)
        clean = normalize_text(raw_text)
        excerpt = build_excerpt(clean, max_chars=600)
        sections = extract_section_snippets(clean)
        method_snippet = sections.get("method", "")[:250]
        results_snippet = sections.get("results", "")[:250]
        conclusion_snippet = sections.get("conclusion", "")[:250]

    return {
        "digest": digest[:900],
        "excerpt": excerpt,
        "method_snippet": method_snippet,
        "results_snippet": results_snippet,
        "conclusion_snippet": conclusion_snippet,
    }


def _build_batch_prompt(batch_items: list[dict[str, Any]], schema: dict[str, Any], schema_version: str) -> str:
    input_payload = [
        {
            "paper_id": item["paper_id"],
            "title": item["title"],
            "year": item["year"],
            "text_sha256": item["text_sha256"],
            "keyword_hits": item["keyword_hits"],
            "digest": item["digest"],
            "excerpt": item["excerpt"],
            "method_snippet": item["method_snippet"],
            "results_snippet": item["results_snippet"],
            "conclusion_snippet": item["conclusion_snippet"],
        }
        for item in batch_items
    ]

    return (
        "Верни строго JSON-объект без markdown и комментариев в формате {\"cards\": [...]} . "
        "Для каждой статьи из input_papers верни ровно одну карточку. "
        "Используй только переданный контекст, не придумывай факты. "
        f"Обязательно выставь schema_version=\"{schema_version}\".\n\n"
        "Ключевая цель: извлечь прикладные идеи для pair trading с фокусом на Sharpe↑ и DD↓.\n\n"
        f"Schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"input_papers:\n{json.dumps(input_payload, ensure_ascii=False)}\n"
    )


def _run_codex(prompt: str, cmd_text: str, timeout_sec: int) -> subprocess.CompletedProcess[str]:
    cmd = shlex.split(cmd_text)
    return subprocess.run(
        cmd,
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_sec,
    )


def _chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def main() -> int:
    args = parse_args()

    db_path = Path(args.db_path)
    queue_path = Path(args.queue_path)
    cards_dir = Path(args.cards_dir)
    schema_path = Path(args.schema_path)

    cards_dir.mkdir(parents=True, exist_ok=True)

    if not queue_path.exists():
        raise FileNotFoundError(f"Queue not found: {queue_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    schema = _load_schema(schema_path)
    schema_version = str(schema.get("$id") or schema.get("version") or CARD_SCHEMA_VERSION)

    conn = connect_db(db_path, read_only=True)
    queue = _load_queue(queue_path, top_n=args.top_n)

    generated = 0
    skipped = 0
    failed = 0
    error_counter: Counter[str] = Counter()

    pending: list[dict[str, Any]] = []

    for entry in queue:
        paper_id = entry.get("paper_id")
        if not isinstance(paper_id, str):
            failed += 1
            error_counter["invalid_queue_item"] += 1
            continue

        row = conn.execute(
            """
            SELECT id, title, year, text_sha256, text_path, digest_path, keyword_hits_json
            FROM papers WHERE id = ?
            """,
            (paper_id,),
        ).fetchone()

        if not row:
            failed += 1
            error_counter["missing_db_row"] += 1
            continue

        card_path = cards_dir / f"{paper_id}.json"
        if _cached(card_path, row["text_sha256"], schema_version):
            skipped += 1
            continue

        context = _load_context(text_path=row["text_path"], digest_path=row["digest_path"])
        keyword_hits = loads_json(row["keyword_hits_json"], {})
        if not isinstance(keyword_hits, dict):
            keyword_hits = {}

        pending.append(
            {
                "paper_id": paper_id,
                "title": row["title"] or paper_id,
                "year": row["year"],
                "text_sha256": row["text_sha256"],
                "keyword_hits": keyword_hits,
                **context,
            }
        )

    if args.dry_run:
        for item in pending:
            print(f"DRY-RUN {item['paper_id']}")
        print(
            json.dumps(
                {
                    "version": MAKE_CARDS_VERSION,
                    "queued": len(queue),
                    "pending": len(pending),
                    "generated": 0,
                    "skipped": skipped,
                    "failed": failed,
                    "schema_version": schema_version,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        conn.close()
        return 0

    batch_size = max(1, int(args.batch_size))
    for batch in _chunks(pending, batch_size):
        prompt = _build_batch_prompt(batch, schema=schema, schema_version=schema_version)
        try:
            result = _run_codex(prompt, cmd_text=args.codex_cmd, timeout_sec=args.timeout_sec)
        except FileNotFoundError as exc:
            conn.close()
            raise RuntimeError(f"codex command is not available: {args.codex_cmd}") from exc
        except subprocess.TimeoutExpired:
            for item in batch:
                failed += 1
                error_counter["codex_timeout"] += 1
            continue

        if result.returncode != 0:
            for item in batch:
                failed += 1
                error_counter["codex_nonzero_exit"] += 1
            continue

        try:
            payload = _extract_best_json(result.stdout)
            cards = _coerce_cards(payload)
        except ValueError:
            for item in batch:
                failed += 1
                error_counter["json_parse_error"] += 1
            continue

        by_id = {
            card.get("paper_id"): card
            for card in cards
            if isinstance(card.get("paper_id"), str)
        }

        for item in batch:
            paper_id = item["paper_id"]
            card = by_id.get(paper_id)
            if not card:
                failed += 1
                error_counter["missing_card_in_batch_response"] += 1
                continue

            card.setdefault("paper_id", paper_id)
            card.setdefault("title", item["title"])
            card.setdefault("year", item["year"])
            card["text_sha256"] = item["text_sha256"]
            card["schema_version"] = schema_version

            valid, errors = validate_card_payload(card, schema)
            if not valid:
                failed += 1
                error_counter["schema_validation_error"] += 1
                error_path = cards_dir / f"{paper_id}.error.txt"
                error_path.write_text("\n".join(errors), encoding="utf-8")
                continue

            card_path = cards_dir / f"{paper_id}.json"
            card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
            generated += 1

    conn.close()

    top_error_reasons = [
        {"reason": reason, "count": count}
        for reason, count in error_counter.most_common(5)
    ]

    stats = {
        "version": MAKE_CARDS_VERSION,
        "queued": len(queue),
        "pending": len(pending),
        "generated": generated,
        "skipped": skipped,
        "failed": failed,
        "schema_version": schema_version,
        "batch_size": batch_size,
        "top_error_reasons": top_error_reasons,
    }
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
