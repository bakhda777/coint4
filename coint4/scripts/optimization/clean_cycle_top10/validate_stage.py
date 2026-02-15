#!/usr/bin/env python3
"""
Fail-fast validations for clean_cycle_top10 E2E stages.

This script is intentionally stdlib-only and safe to run locally after sync_back.
On invariant violation it exits with code 2 (so orchestration can fail fast).
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _die(code: int, msg: str) -> "None":
    _eprint(f"ERROR: {msg}")
    raise SystemExit(code)


def _resolve_app_root() -> Path:
    # repo-root/coint4/scripts/optimization/clean_cycle_top10/validate_stage.py -> parents[3] == app-root (coint4/)
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class QueueInfo:
    path: Path
    rows: int
    statuses: Tuple[str, ...]

    def all_planned(self) -> bool:
        return self.rows > 0 and all(s == "planned" for s in self.statuses)


def _read_queue(path: Path) -> QueueInfo:
    if not path.exists():
        _die(2, f"queue not found: {path}")
    if not path.is_file():
        _die(2, f"queue is not a file: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            _die(2, f"queue has no header: {path}")
        rows: List[dict] = []
        for r in reader:
            # Drop fully-empty lines.
            if not any((v or "").strip() for v in r.values()):
                continue
            rows.append(r)

    statuses = tuple((str(r.get("status") or "").strip().lower()) for r in rows)
    return QueueInfo(path=path, rows=len(rows), statuses=statuses)


def _count_files(root: Path, filename: str) -> int:
    if not root.exists():
        return 0
    if not root.is_dir():
        return 0
    return sum(1 for _ in root.rglob(filename))


def _count_csv_rows(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = 0
        for i, r in enumerate(reader):
            if i == 0:
                continue  # header
            if not r or all((c or "").strip() == "" for c in r):
                continue
            rows += 1
        return rows


def _stage_alias(stage: str) -> str:
    s = stage.strip().lower()
    if s in {"e2e05", "05", "baseline", "baseline_sync_back", "baseline-sync-back"}:
        return "E2E05"
    if s in {"e2e08", "08", "sweeps_sync_back", "sweeps-sync-back", "sweeps"}:
        return "E2E08"
    if s in {"e2e09", "09", "final", "final_rollup", "final-rollup", "sweeps_post", "sweeps-post"}:
        return "E2E09"
    _die(2, f"unknown --stage: {stage!r} (expected E2E05/E2E08/E2E09)")
    raise AssertionError("unreachable")


def _require_not_all_planned(q: QueueInfo) -> None:
    if q.rows <= 0:
        _die(2, f"queue has no rows: {q.path}")
    if q.all_planned():
        _die(2, f"queue status is still all 'planned' (did sync_back happen?): {q.path}")


def _validate_e2e05(*, run: str, allow_baseline_only: bool, app_root: Path) -> None:
    if allow_baseline_only:
        _die(2, "--allow-baseline-only is not applicable for stage E2E05")

    q = _read_queue(
        app_root / "artifacts/wfa/aggregate/clean_cycle_top10" / run / "baseline_run_queue.csv"
    )
    _require_not_all_planned(q)
    if q.rows != 10:
        _die(2, f"baseline queue must have exactly 10 rows (top10); got {q.rows}: {q.path}")

    baseline_dir = app_root / "artifacts/wfa/runs_clean" / run / "baseline_top10"
    eq = _count_files(baseline_dir, "equity_curve.csv")
    if eq != 10:
        _die(
            2,
            f"expected exactly 10 equity_curve.csv under {baseline_dir}, found {eq}",
        )

    print(f"[ok] E2E05: baseline queue rows={q.rows}, equity_curve.csv={eq}")


def _validate_e2e08(*, run: str, allow_baseline_only: bool, app_root: Path) -> None:
    if allow_baseline_only:
        _die(2, "--allow-baseline-only is not applicable for stage E2E08")

    q = _read_queue(
        app_root / "artifacts/wfa/aggregate/clean_cycle_top10" / run / "sweeps_run_queue.csv"
    )
    _require_not_all_planned(q)
    n = q.rows
    if n <= 0:
        _die(2, f"sweeps queue must have at least 1 row; got {n}: {q.path}")

    sweeps_dir = app_root / "artifacts/wfa/runs_clean" / run / "opt_sweeps"
    eq = _count_files(sweeps_dir, "equity_curve.csv")
    if eq != n:
        _die(
            2,
            f"expected exactly N={n} equity_curve.csv under {sweeps_dir}, found {eq}",
        )

    print(f"[ok] E2E08: sweeps queue rows(N)={n}, equity_curve.csv={eq}")


def _validate_e2e09(*, run: str, allow_baseline_only: bool, app_root: Path) -> None:
    q_path = app_root / "artifacts/wfa/aggregate/clean_cycle_top10" / run / "sweeps_run_queue.csv"
    if not q_path.exists():
        if allow_baseline_only:
            print(f"[ok] E2E09: sweeps queue missing and baseline-only explicitly allowed: {q_path}")
            return
        _die(2, f"sweeps queue not found (baseline-only forbidden by default): {q_path}")

    q = _read_queue(q_path)
    n = q.rows
    if n <= 0:
        if allow_baseline_only:
            print(f"[ok] E2E09: sweeps queue has 0 rows and baseline-only explicitly allowed: {q.path}")
            return
        _die(2, f"sweeps queue has 0 rows (baseline-only forbidden by default): {q.path}")

    _require_not_all_planned(q)

    sweeps_dir = app_root / "artifacts/wfa/runs_clean" / run / "opt_sweeps"
    canon = _count_files(sweeps_dir, "canonical_metrics.json")
    if canon != n:
        _die(
            2,
            f"expected exactly N={n} canonical_metrics.json under {sweeps_dir}, found {canon}",
        )

    rollup_csv = app_root / "artifacts/wfa/aggregate/clean_cycle_top10" / run / "rollup_clean_cycle_top10.csv"
    rollup_rows = _count_csv_rows(rollup_csv)
    min_rows = 10 + n
    if rollup_rows < min_rows:
        _die(
            2,
            f"final rollup must include at least 10+N={min_rows} rows (no baseline-only fallback); "
            f"got {rollup_rows}: {rollup_csv}",
        )

    print(f"[ok] E2E09: sweeps canonical_metrics.json={canon} (N={n}), rollup_rows={rollup_rows} (min={min_rows})")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Validate clean_cycle_top10 stage invariants (fail-fast, exit 2)")
    p.add_argument("--run", required=True, help="RUN id, e.g. 20260216_clean_top10")
    p.add_argument("--stage", required=True, help="Stage: E2E05, E2E08, or E2E09")
    p.add_argument(
        "--allow-baseline-only",
        action="store_true",
        help="Opt-in to baseline-only behaviour for the final stage (E2E09) when sweeps are intentionally absent.",
    )
    args = p.parse_args(argv)

    run = str(args.run).strip()
    if not run:
        _die(2, "--run must be non-empty")

    stage = _stage_alias(str(args.stage))
    app_root = _resolve_app_root()

    if stage == "E2E05":
        _validate_e2e05(run=run, allow_baseline_only=bool(args.allow_baseline_only), app_root=app_root)
    elif stage == "E2E08":
        _validate_e2e08(run=run, allow_baseline_only=bool(args.allow_baseline_only), app_root=app_root)
    else:
        _validate_e2e09(run=run, allow_baseline_only=bool(args.allow_baseline_only), app_root=app_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
