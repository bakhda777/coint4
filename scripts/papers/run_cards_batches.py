#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lib_pipeline import DEFAULT_SYNTHESIS_DIR


@dataclass
class CmdResult:
    returncode: int
    stdout: str
    stderr: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple diversified card-generation batches")
    parser.add_argument("--python", default=sys.executable, help="Python executable")
    parser.add_argument("--batches", type=int, default=3, help="How many batches to run")
    parser.add_argument("--batch-size", type=int, default=30, help="Papers per batch")
    parser.add_argument("--top-k", type=int, default=180, help="Selection pool bound by priority")
    parser.add_argument("--cards-batch-size", type=int, default=5, help="make_cards --batch-size")
    parser.add_argument("--timeout-sec", type=int, default=240, help="make_cards timeout per codex call")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for deterministic selection")
    parser.add_argument(
        "--select-script",
        default="scripts/papers/select_batch.py",
        help="Path to select_batch.py",
    )
    parser.add_argument(
        "--make-cards-script",
        default="scripts/papers/make_cards.py",
        help="Path to make_cards.py",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_SYNTHESIS_DIR),
        help="Directory for selection files and batch report",
    )
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_SYNTHESIS_DIR / "cards_batches_report.jsonl"),
        help="Output JSONL report path",
    )
    parser.add_argument(
        "--codex-cmd",
        default=None,
        help="Optional override for make_cards --codex-cmd",
    )
    return parser.parse_args()


def _run_command(cmd: list[str]) -> CmdResult:
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return CmdResult(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)


def _extract_json(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("Command output does not contain JSON object")


def _write_report_line(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("", encoding="utf-8")

    total_created = 0
    total_skipped = 0
    total_failed = 0
    completed_batches = 0

    summaries: list[dict[str, Any]] = []

    for batch_index in range(1, max(1, int(args.batches)) + 1):
        selection_path = output_dir / f"selection_batch_{batch_index:02d}.jsonl"

        select_cmd = [
            args.python,
            args.select_script,
            "--batch-size",
            str(int(args.batch_size)),
            "--top-k",
            str(int(args.top_k)),
            "--seed",
            str(int(args.seed) + batch_index - 1),
            "--output-path",
            str(selection_path),
            "--skip-existing-cards",
        ]

        select_res = _run_command(select_cmd)
        if select_res.returncode != 0:
            raise RuntimeError(
                "select_batch failed "
                f"(batch={batch_index}, cmd={shlex.join(select_cmd)}):\n{select_res.stderr.strip() or select_res.stdout.strip()}"
            )
        select_stats = _extract_json(select_res.stdout)
        selected_count = int(select_stats.get("selected", 0) or 0)

        batch_summary: dict[str, Any] = {
            "batch": batch_index,
            "selection_file": str(selection_path),
            "selected": selected_count,
            "select_stats": select_stats,
            "created": 0,
            "skipped": 0,
            "failed": 0,
            "status": "planned",
        }

        if selected_count <= 0:
            batch_summary["status"] = "no_candidates"
            summaries.append(batch_summary)
            _write_report_line(report_path, batch_summary)
            break

        make_cmd = [
            args.python,
            args.make_cards_script,
            "--selection-file",
            str(selection_path),
            "--top",
            str(int(args.batch_size)),
            "--batch-size",
            str(int(args.cards_batch_size)),
            "--timeout-sec",
            str(int(args.timeout_sec)),
        ]
        if args.codex_cmd:
            make_cmd.extend(["--codex-cmd", args.codex_cmd])

        make_res = _run_command(make_cmd)
        if make_res.returncode != 0:
            raise RuntimeError(
                "make_cards failed "
                f"(batch={batch_index}, cmd={shlex.join(make_cmd)}):\n{make_res.stderr.strip() or make_res.stdout.strip()}"
            )

        make_stats = _extract_json(make_res.stdout)

        created = int(make_stats.get("generated", 0) or 0)
        skipped = int(make_stats.get("skipped", 0) or 0)
        failed = int(make_stats.get("failed", 0) or 0)

        total_created += created
        total_skipped += skipped
        total_failed += failed
        completed_batches += 1

        batch_summary.update(
            {
                "status": "completed",
                "created": created,
                "skipped": skipped,
                "failed": failed,
                "make_stats": make_stats,
            }
        )
        summaries.append(batch_summary)
        _write_report_line(report_path, batch_summary)

    final_stats = {
        "batches_requested": int(args.batches),
        "batches_completed": completed_batches,
        "created_total": total_created,
        "skipped_total": total_skipped,
        "failed_total": total_failed,
        "report_path": str(report_path),
        "summaries": summaries,
    }
    print(json.dumps(final_stats, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
