#!/usr/bin/env python3
"""Write a metadata snapshot for a rollup run_index.csv.

This is meant to "freeze" the exact input used for TOP-N selection by recording
the source path and the sha256 of the file (plus basic CSV shape metadata).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_columns_and_row_count(path: Path) -> Tuple[List[str], int]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return [], 0

        row_count = 0
        for _ in reader:
            row_count += 1

    columns = [str(c).strip() for c in header]
    return columns, row_count


def _try_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _resolve_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: assume we are under coint4/scripts/**.
    return here.parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a metadata snapshot for run_index.csv (sha256 + CSV shape)")
    parser.add_argument(
        "--run-index",
        default="artifacts/wfa/aggregate/rollup/run_index.csv",
        help="Path to rollup run_index.csv (relative to coint4/).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Where to write JSON snapshot (relative to coint4/).",
    )
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument(
        "--refuse-overwrite",
        action="store_true",
        help="Refuse overwriting an existing snapshot (default).",
    )
    overwrite_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing snapshot file.",
    )
    args = parser.parse_args()

    project_root = _resolve_project_root()
    run_index_path = project_root / args.run_index
    output_path = project_root / args.output

    if not run_index_path.exists():
        raise SystemExit(f"run index not found: {run_index_path}")

    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing snapshot: {output_path} (use --overwrite)")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns, row_count = _csv_columns_and_row_count(run_index_path)
    snapshot = {
        "source_path": _try_relpath(run_index_path, project_root),
        "sha256": _sha256_file(run_index_path),
        "generated_at_utc": _utc_now_iso(),
        "row_count": row_count,
        "columns": columns,
    }

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(output_path)

    print(f"Wrote snapshot: {_try_relpath(output_path, project_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
