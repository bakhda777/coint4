#!/usr/bin/env python3
"""Build clean-cycle rollup CSV/MD (baseline + sweeps) using canonical_metrics.json as source of truth.

Task (C09 in tasks/clean_cycle_top10/prd_clean_cycle_top10.json):
- Generate rollup_clean_cycle_top10.csv in CLEAN_AGG_DIR.
- Use canonical_metrics.json (per results_dir) for canonical_* metrics.
- Compute scalar score = canonical_sharpe - lambda_dd * abs(canonical_max_drawdown_abs).
- Deterministic sorting and explicit filters (completed-only, canonical-present-only by default).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _try_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _resolve_project_root() -> Path:
    """Return app-root (directory containing pyproject.toml)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: assume we are under coint4/scripts/**.
    return here.parents[3]


def _normalize_repo_relative_path(raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    # Some tools may store paths relative to repo root (prefix "coint4/").
    if value.startswith("coint4/"):
        value = value[len("coint4/") :]
    # Strip leading "./" to keep outputs stable.
    while value.startswith("./"):
        value = value[2:]
    return value


def _resolve_under_project(path_str: str, project_root: Path) -> Optional[Path]:
    raw = str(path_str or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    normalized = _normalize_repo_relative_path(raw)
    return project_root / normalized


def _path_for_output(raw: str, project_root: Path) -> str:
    resolved = _resolve_under_project(raw, project_root)
    if resolved is None:
        return ""
    return _try_relpath(resolved, project_root)


def _to_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if int(value) != value:
            return None
        return int(value)
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        if not math.isfinite(out):
            return None
        return out
    s = str(value).strip()
    if not s:
        return None
    try:
        out = float(s)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


WF_KEYS_ORDER = [
    "start_date",
    "end_date",
    "training_period_days",
    "testing_period_days",
    "step_size_days",
    "max_steps",
    "gap_minutes",
    "refit_frequency",
]


def _normalize_walk_forward_subset(wf: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in keys:
        if key in {"start_date", "end_date"}:
            val = wf.get(key)
            out[key] = None if val is None else str(val).strip() or None
        elif key in {"training_period_days", "testing_period_days", "step_size_days", "max_steps", "gap_minutes"}:
            out[key] = _to_int(wf.get(key))
        elif key in {"refit_frequency"}:
            val = wf.get(key)
            out[key] = None if val is None else str(val).strip() or None
        else:
            out[key] = wf.get(key)
    return out


def _load_fixed_windows_fingerprint(*, args: argparse.Namespace, project_root: Path) -> Tuple[str, Dict[str, Any]]:
    fixed_windows_json = str(getattr(args, "fixed_windows_json", "") or "").strip()
    if args.use_definitions or not fixed_windows_json:
        from definitions import FIXED_WINDOWS  # local module (same dir)

        node: Any = FIXED_WINDOWS
    else:
        fixed_path = _resolve_under_project(fixed_windows_json, project_root)
        if fixed_path is None:
            raise SystemExit("--fixed-windows-json is empty")
        if not fixed_path.exists():
            raise SystemExit(f"fixed windows json not found: {fixed_path}")
        node = json.loads(fixed_path.read_text(encoding="utf-8"))

    if not isinstance(node, dict):
        raise SystemExit("FIXED_WINDOWS must be a JSON object/dict")

    wf_node = node.get("walk_forward", node)
    if not isinstance(wf_node, dict):
        raise SystemExit("FIXED_WINDOWS.walk_forward must be a JSON object/dict")

    wf_norm = _normalize_walk_forward_subset(wf_node, WF_KEYS_ORDER)
    payload = json.dumps(wf_norm, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return _sha256_hex(payload), wf_norm


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise SystemExit(f"manifest must be a JSON list: {path}")
        out: List[Dict[str, Any]] = []
        for i, item in enumerate(payload):
            if not isinstance(item, dict):
                raise SystemExit(f"manifest entry #{i} must be an object: {path}")
            out.append(item)
        return out

    if suffix == ".csv":
        out_rows: List[Dict[str, Any]] = []
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                out_rows.append(dict(row))
        return out_rows

    raise SystemExit(f"unsupported manifest format (expected .json or .csv): {path}")


def _load_optional_canonical_metrics(results_dir: Path) -> Tuple[bool, Dict[str, Optional[float]]]:
    path = results_dir / "canonical_metrics.json"
    if not path.exists():
        return False, {
            "canonical_sharpe": None,
            "canonical_pnl_abs": None,
            "canonical_max_drawdown_abs": None,
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"canonical metrics must be a JSON object: {path}")

    metrics_node: Any = payload.get("metrics")
    if isinstance(metrics_node, dict):
        source = metrics_node
    else:
        source = payload

    def _get(key: str) -> Optional[float]:
        return _to_float(source.get(key))

    return True, {
        "canonical_sharpe": _get("canonical_sharpe"),
        "canonical_pnl_abs": _get("canonical_pnl_abs"),
        "canonical_max_drawdown_abs": _get("canonical_max_drawdown_abs"),
    }


def _infer_run_name(*, entry: Dict[str, Any], phase: str, idx: int) -> str:
    explicit = str(entry.get("run_name") or "").strip()
    if explicit:
        return explicit

    rank = entry.get("rank")
    if rank is not None:
        rank_int = _to_int(rank)
        if rank_int is not None and rank_int > 0:
            prefix = "b" if phase == "baseline" else "s"
            return f"{prefix}{rank_int:02d}"

    config_path = str(entry.get("config_path") or "").strip()
    if config_path:
        return Path(config_path).stem

    run_group = str(entry.get("run_group") or "").strip()
    run_id = str(entry.get("run_id") or "").strip()
    if run_group and run_id:
        return f"{run_group}/{run_id}"
    if run_id:
        return run_id
    if run_group:
        return run_group

    results_dir = str(entry.get("results_dir") or "").strip()
    if results_dir:
        return Path(results_dir).name

    return f"{phase}_{idx:04d}"


def _is_completed(status: str) -> bool:
    return str(status or "").strip().lower() == "completed"


def _compute_score(*, sharpe: Optional[float], dd_abs: Optional[float], lambda_dd: float) -> Optional[float]:
    if sharpe is None or dd_abs is None:
        return None
    if not math.isfinite(sharpe) or not math.isfinite(dd_abs):
        return None
    return float(sharpe) - float(lambda_dd) * abs(float(dd_abs))


@dataclass(frozen=True)
class _Row:
    phase: str
    run_name: str
    run_group: str
    run_id: str
    status: str
    results_dir: str
    config_path: str
    config_sha256: str
    canonical_metrics_present: bool
    canonical_sharpe: Optional[float]
    canonical_pnl_abs: Optional[float]
    canonical_max_drawdown_abs: Optional[float]
    score: Optional[float]
    fixed_windows_fingerprint: str


def _sort_key_score(row: _Row) -> Tuple[Any, ...]:
    score_key = float("inf") if row.score is None else -float(row.score)
    sharpe_key = float("inf") if row.canonical_sharpe is None else -float(row.canonical_sharpe)
    dd_key = float("inf") if row.canonical_max_drawdown_abs is None else abs(float(row.canonical_max_drawdown_abs))
    pnl_key = float("inf") if row.canonical_pnl_abs is None else -float(row.canonical_pnl_abs)
    return (
        score_key,
        sharpe_key,
        dd_key,
        pnl_key,
        row.phase,
        row.run_name,
        row.run_group,
        row.run_id,
        row.results_dir,
    )


def _sort_key_multi(row: _Row) -> Tuple[Any, ...]:
    sharpe_key = float("inf") if row.canonical_sharpe is None else -float(row.canonical_sharpe)
    dd_key = float("inf") if row.canonical_max_drawdown_abs is None else abs(float(row.canonical_max_drawdown_abs))
    pnl_key = float("inf") if row.canonical_pnl_abs is None else -float(row.canonical_pnl_abs)
    return (
        sharpe_key,
        dd_key,
        pnl_key,
        row.phase,
        row.run_name,
        row.run_group,
        row.run_id,
        row.results_dir,
    )


def _fmt_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    if not math.isfinite(value):
        return ""
    return str(float(value))


def _write_csv(path: Path, rows: List[_Row]) -> None:
    fieldnames = [
        "phase",
        "run_name",
        "run_group",
        "run_id",
        "status",
        "results_dir",
        "config_path",
        "config_sha256",
        "canonical_metrics_present",
        "canonical_sharpe",
        "canonical_pnl_abs",
        "canonical_max_drawdown_abs",
        "score",
        "fixed_windows_fingerprint",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "phase": r.phase,
                    "run_name": r.run_name,
                    "run_group": r.run_group,
                    "run_id": r.run_id,
                    "status": r.status,
                    "results_dir": r.results_dir,
                    "config_path": r.config_path,
                    "config_sha256": r.config_sha256,
                    "canonical_metrics_present": "true" if r.canonical_metrics_present else "false",
                    "canonical_sharpe": _fmt_float(r.canonical_sharpe),
                    "canonical_pnl_abs": _fmt_float(r.canonical_pnl_abs),
                    "canonical_max_drawdown_abs": _fmt_float(r.canonical_max_drawdown_abs),
                    "score": _fmt_float(r.score),
                    "fixed_windows_fingerprint": r.fixed_windows_fingerprint,
                }
            )
    tmp.replace(path)


def _write_md(
    *,
    path: Path,
    rows: List[_Row],
    sort_mode: str,
    lambda_dd: float,
    filters: List[str],
    fixed_windows_fingerprint: str,
    fixed_windows_walk_forward_norm: Dict[str, Any],
    top_n: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wf_json = json.dumps(fixed_windows_walk_forward_norm, indent=2, sort_keys=True, ensure_ascii=False)
    lines: List[str] = []
    lines.append("# rollup_clean_cycle_top10")
    lines.append("")
    lines.append(f"- sort_mode: {sort_mode}")
    lines.append(f"- score_lambda_dd: {lambda_dd}")
    lines.append(f"- filters: {', '.join(filters) if filters else 'none'}")
    lines.append(f"- fixed_windows_fingerprint: {fixed_windows_fingerprint}")
    lines.append(f"- rows: {len(rows)}")
    lines.append("")
    lines.append("## FIXED_WINDOWS.walk_forward (normalized)")
    lines.append("")
    lines.append("```json")
    lines.append(wf_json)
    lines.append("```")
    lines.append("")
    lines.append(f"## Top-{max(0, int(top_n))}")
    lines.append("")
    lines.append("| rank | phase | run_name | canonical_sharpe | canonical_max_drawdown_abs | canonical_pnl_abs | score | results_dir |")
    lines.append("| ---: | :---: | :------- | ---------------: | ------------------------: | ----------------: | ----: | :---------- |")
    for idx, r in enumerate(rows[: max(0, int(top_n))], 1):
        lines.append(
            "| {rank} | {phase} | {run_name} | {sharpe} | {dd} | {pnl} | {score} | {results_dir} |".format(
                rank=idx,
                phase=r.phase,
                run_name=r.run_name,
                sharpe=_fmt_float(r.canonical_sharpe) or "-",
                dd=_fmt_float(r.canonical_max_drawdown_abs) or "-",
                pnl=_fmt_float(r.canonical_pnl_abs) or "-",
                score=_fmt_float(r.score) or "-",
                results_dir=r.results_dir,
            )
        )
    lines.append("")

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build clean-cycle rollup index (baseline + sweeps) on canonical_*")
    parser.add_argument(
        "--baseline-manifest",
        default="artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/baseline_manifest.json",
        help="Path to baseline manifest (.json/.csv) (relative to coint4/ unless absolute).",
    )
    parser.add_argument(
        "--sweeps-manifest",
        default=None,
        help="Optional path to sweeps manifest (.json/.csv) (relative to coint4/ unless absolute).",
    )
    parser.add_argument(
        "--output-csv",
        default="artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/rollup_clean_cycle_top10.csv",
        help="Where to write rollup_clean_cycle_top10.csv (relative to coint4/ unless absolute).",
    )
    parser.add_argument(
        "--output-md",
        default="artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/rollup_clean_cycle_top10.md",
        help="Where to write rollup_clean_cycle_top10.md (relative to coint4/ unless absolute).",
    )
    parser.add_argument(
        "--lambda-dd",
        "--score-lambda-dd",
        dest="lambda_dd",
        type=float,
        default=0.02,
        help="Score penalty weight for abs(drawdown): score = sharpe - lambda_dd * abs(dd).",
    )
    parser.add_argument(
        "--sort-mode",
        choices=["score", "multi"],
        default="score",
        help="Sorting mode: scalar score or multi-objective (sharpe desc, abs(dd) asc, pnl desc).",
    )
    parser.add_argument(
        "--top-n-md",
        type=int,
        default=20,
        help="How many top rows to include in markdown output.",
    )
    parser.add_argument(
        "--include-noncompleted",
        action="store_true",
        help="Include entries where status is not 'completed'.",
    )
    parser.add_argument(
        "--include-missing-canonical",
        action="store_true",
        help="Include entries missing canonical_metrics.json (canonical_* will be empty).",
    )
    parser.add_argument(
        "--use-definitions",
        action="store_true",
        help="Force FIXED_WINDOWS from definitions.py (default unless --fixed-windows-json is set).",
    )
    parser.add_argument(
        "--fixed-windows-json",
        default="",
        help="Optional FIXED_WINDOWS JSON path to fingerprint (overrides definitions unless --use-definitions is set).",
    )
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument(
        "--refuse-overwrite",
        action="store_true",
        help="Refuse overwriting existing output files (default).",
    )
    overwrite_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    args = parser.parse_args()

    project_root = _resolve_project_root()

    baseline_path = _resolve_under_project(args.baseline_manifest, project_root) or (project_root / args.baseline_manifest)
    sweeps_path = None
    if args.sweeps_manifest:
        sweeps_path = _resolve_under_project(args.sweeps_manifest, project_root) or (project_root / args.sweeps_manifest)
    out_csv = _resolve_under_project(args.output_csv, project_root) or (project_root / args.output_csv)
    out_md = _resolve_under_project(args.output_md, project_root) or (project_root / args.output_md)

    if not baseline_path.exists():
        raise SystemExit(f"baseline manifest not found: {baseline_path}")
    if sweeps_path is not None and not sweeps_path.exists():
        raise SystemExit(f"sweeps manifest not found: {sweeps_path}")

    if (out_csv.exists() or out_md.exists()) and not args.overwrite:
        existing = []
        if out_csv.exists():
            existing.append(str(out_csv))
        if out_md.exists():
            existing.append(str(out_md))
        raise SystemExit("refusing to overwrite existing outputs (use --overwrite):\n- " + "\n- ".join(existing))

    fixed_fp, wf_norm = _load_fixed_windows_fingerprint(args=args, project_root=project_root)

    baseline_entries = _load_manifest(baseline_path)
    sweeps_entries: List[Dict[str, Any]] = _load_manifest(sweeps_path) if sweeps_path is not None else []

    total_entries = len(baseline_entries) + len(sweeps_entries)
    if total_entries <= 0:
        raise SystemExit("no manifest entries to process")

    lambda_dd = float(args.lambda_dd)
    if not math.isfinite(lambda_dd) or lambda_dd < 0:
        raise SystemExit(f"--lambda-dd must be a finite non-negative float: {args.lambda_dd!r}")

    rows: List[_Row] = []
    skipped = 0
    for phase, entries in [("baseline", baseline_entries), ("sweep", sweeps_entries)]:
        for idx, entry in enumerate(entries, 1):
            status_raw = str(entry.get("status") or "").strip()
            if not args.include_noncompleted and not _is_completed(status_raw):
                skipped += 1
                continue

            results_raw = str(entry.get("results_dir") or "").strip()
            if not results_raw:
                skipped += 1
                continue

            results_path = _resolve_under_project(results_raw, project_root)
            canonical_present = False
            canonical: Dict[str, Optional[float]] = {
                "canonical_sharpe": None,
                "canonical_pnl_abs": None,
                "canonical_max_drawdown_abs": None,
            }
            if results_path is not None and results_path.exists():
                canonical_present, canonical = _load_optional_canonical_metrics(results_path)

            if not args.include_missing_canonical and not canonical_present:
                skipped += 1
                continue

            run_name = _infer_run_name(entry=entry, phase=phase, idx=idx)
            run_group = str(entry.get("run_group") or "").strip()
            run_id = str(entry.get("run_id") or "").strip()

            config_raw = str(entry.get("config_path") or "").strip()
            config_path_out = _path_for_output(config_raw, project_root) if config_raw else ""

            config_sha = str(entry.get("config_sha256") or "").strip()
            if not config_sha and config_raw:
                cfg_path = _resolve_under_project(config_raw, project_root)
                if cfg_path is not None and cfg_path.exists() and cfg_path.is_file():
                    config_sha = _sha256_file(cfg_path)

            score = _compute_score(
                sharpe=canonical.get("canonical_sharpe"),
                dd_abs=canonical.get("canonical_max_drawdown_abs"),
                lambda_dd=lambda_dd,
            )

            rows.append(
                _Row(
                    phase=phase,
                    run_name=run_name,
                    run_group=run_group,
                    run_id=run_id,
                    status=status_raw,
                    results_dir=_path_for_output(results_raw, project_root),
                    config_path=config_path_out,
                    config_sha256=config_sha,
                    canonical_metrics_present=canonical_present,
                    canonical_sharpe=canonical.get("canonical_sharpe"),
                    canonical_pnl_abs=canonical.get("canonical_pnl_abs"),
                    canonical_max_drawdown_abs=canonical.get("canonical_max_drawdown_abs"),
                    score=score,
                    fixed_windows_fingerprint=fixed_fp,
                )
            )

    if not rows:
        raise SystemExit("no rows matched filters; nothing to write")

    sort_mode = str(args.sort_mode)
    if sort_mode == "multi":
        rows.sort(key=_sort_key_multi)
    else:
        rows.sort(key=_sort_key_score)

    _write_csv(out_csv, rows)

    filters = []
    if not args.include_noncompleted:
        filters.append("status==completed")
    if not args.include_missing_canonical:
        filters.append("canonical_metrics_present==true")

    _write_md(
        path=out_md,
        rows=rows,
        sort_mode=sort_mode,
        lambda_dd=lambda_dd,
        filters=filters,
        fixed_windows_fingerprint=fixed_fp,
        fixed_windows_walk_forward_norm=wf_norm,
        top_n=args.top_n_md,
    )

    print(f"Wrote rollup CSV: {_try_relpath(out_csv, project_root)} (rows={len(rows)} skipped={skipped})")
    print(f"Wrote rollup MD:  {_try_relpath(out_md, project_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
