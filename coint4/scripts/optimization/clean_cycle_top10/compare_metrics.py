#!/usr/bin/env python3
"""Compare raw metrics (strategy_metrics.csv / run_index.csv) vs canonical_metrics.json.

Task (C08 in tasks/clean_cycle_top10/prd_clean_cycle_top10.json):
- Print a diff report for sharpe/pnl/dd to audit legacy (raw) vs canonical metrics.
- Support sampling (e.g. top-5 + random-5) and diff thresholds.
- Support --dry-run mode that only prints what would be compared (no tree scans).

Typical usage (from coint4/):
  PYTHONPATH=src ./.venv/bin/python scripts/optimization/clean_cycle_top10/compare_metrics.py \
    --manifest artifacts/wfa/aggregate/rollup/run_index.csv \
    --sample top5+random5 --seed 1 --thresholds sharpe=0.05,pnl=1e-6,dd=1e-6
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _resolve_project_root() -> Path:
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
    if value.startswith("coint4/"):
        value = value[len("coint4/") :]
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


def _try_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
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


def _infer_run_id_group(results_dir: Path, project_root: Path) -> Tuple[str, str]:
    run_id = results_dir.name
    run_group = ""
    try:
        rel = results_dir.resolve().relative_to(project_root.resolve())
        parts = rel.parts
        if len(parts) >= 4 and parts[0:3] == ("artifacts", "wfa", "runs"):
            run_group = parts[3]
    except ValueError:
        run_group = ""
    return run_id, run_group


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"manifest must be a JSON list: {path}")
        out: List[Dict[str, Any]] = []
        for i, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"manifest entry #{i} must be an object: {path}")
            out.append(item)
        return out
    if suffix == ".csv":
        out_rows: List[Dict[str, Any]] = []
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                out_rows.append(dict(row))
        return out_rows
    raise ValueError(f"unsupported manifest format (expected .json or .csv): {path}")


def _discover_run_dirs_from_runs_root(runs_root: Path) -> List[Path]:
    if not runs_root.exists():
        return []
    return sorted({p.parent.resolve() for p in runs_root.rglob("strategy_metrics.csv")})


def _load_raw_metrics_from_strategy_csv(run_dir: Path) -> Tuple[bool, Dict[str, Optional[float]]]:
    metrics_path = run_dir / "strategy_metrics.csv"
    if not metrics_path.exists():
        return False, {"sharpe_ratio_abs": None, "total_pnl": None, "max_drawdown_abs": None}
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            return True, {
                "sharpe_ratio_abs": _to_float(row.get("sharpe_ratio_abs")),
                "total_pnl": _to_float(row.get("total_pnl")),
                "max_drawdown_abs": _to_float(row.get("max_drawdown_abs")),
            }
    return True, {"sharpe_ratio_abs": None, "total_pnl": None, "max_drawdown_abs": None}


def _load_canonical_metrics(run_dir: Path) -> Tuple[bool, Dict[str, Optional[float]]]:
    path = run_dir / "canonical_metrics.json"
    if not path.exists():
        return False, {
            "canonical_sharpe": None,
            "canonical_pnl_abs": None,
            "canonical_max_drawdown_abs": None,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"canonical metrics must be a JSON object: {path}")

    # Allow future schema: either flat keys or nested under "metrics".
    metrics_node: Any = payload.get("metrics")
    source: Dict[str, Any] = metrics_node if isinstance(metrics_node, dict) else payload

    return True, {
        "canonical_sharpe": _to_float(source.get("canonical_sharpe")),
        "canonical_pnl_abs": _to_float(source.get("canonical_pnl_abs")),
        "canonical_max_drawdown_abs": _to_float(source.get("canonical_max_drawdown_abs")),
    }


def _fmt(value: Optional[float], *, digits: int) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def _fmt_signed(value: Optional[float], *, digits: int) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{digits}f}"


@dataclass(frozen=True)
class _Target:
    run_dir: Path
    run_id: str
    run_group: str
    status: str
    raw_hint_sharpe: Optional[float]
    raw_hint_pnl: Optional[float]
    raw_hint_dd: Optional[float]


def _dedupe_targets_keep_order(targets: Sequence[_Target]) -> List[_Target]:
    seen: set[Path] = set()
    out: List[_Target] = []
    for t in targets:
        resolved = t.run_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(t)
    return out


def _parse_sample_spec(spec: str) -> Tuple[int, int]:
    raw = str(spec or "").strip().lower()
    if not raw or raw == "all":
        return 0, 0
    top_n = 0
    random_n = 0
    for part in [p.strip() for p in raw.replace(",", "+").split("+") if p.strip()]:
        if part.startswith("top="):
            top_n = int(part.split("=", 1)[1])
        elif part.startswith("top") and part[3:].isdigit():
            top_n = int(part[3:])
        elif part.startswith("random="):
            random_n = int(part.split("=", 1)[1])
        elif part.startswith("random") and part[6:].isdigit():
            random_n = int(part[6:])
        else:
            raise ValueError(f"unsupported --sample part: {part!r}")
    if top_n < 0 or random_n < 0:
        raise ValueError("--sample values must be non-negative")
    return top_n, random_n


def _parse_thresholds(raw: str) -> Dict[str, float]:
    spec = str(raw or "").strip()
    if not spec:
        return {}
    out: Dict[str, float] = {}
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        if "=" not in part:
            raise ValueError(f"invalid threshold item (expected k=v): {part!r}")
        key, value = part.split("=", 1)
        k = key.strip().lower()
        if k not in {"sharpe", "pnl", "dd"}:
            raise ValueError(f"unknown threshold key: {k!r} (expected sharpe,pnl,dd)")
        v = _to_float(value)
        if v is None or v < 0:
            raise ValueError(f"invalid threshold value for {k}: {value!r}")
        out[k] = float(v)
    return out


def _select_targets(
    targets: Sequence[_Target],
    *,
    top_n: int,
    random_n: int,
    seed: int,
    project_root: Path,
    dry_run: bool,
) -> List[_Target]:
    if top_n <= 0 and random_n <= 0:
        return list(targets)

    rng = random.Random(seed)

    # If we only need random sampling, avoid touching per-run files.
    if top_n <= 0 and random_n > 0:
        if dry_run:
            return list(targets)[:0]  # Caller prints plan only; no selection preview needed.
        k = min(random_n, len(targets))
        return rng.sample(list(targets), k=k)

    # For top-N selection we need a sharpe value per target.
    scored: List[Tuple[float, str, str, _Target]] = []
    for t in targets:
        sharpe = t.raw_hint_sharpe
        if sharpe is None and not dry_run:
            present, raw_metrics = _load_raw_metrics_from_strategy_csv(t.run_dir)
            _ = present
            sharpe = raw_metrics.get("sharpe_ratio_abs")
        score = float("-inf") if sharpe is None else float(sharpe)
        scored.append((score, t.run_group, t.run_id, t))

    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    top = [t for _, _, _, t in scored[: min(top_n, len(scored))]]

    remaining = [t for _, _, _, t in scored[min(top_n, len(scored)) :]]
    if random_n <= 0:
        return top
    k = min(random_n, len(remaining))
    rand = rng.sample(remaining, k=k) if k > 0 else []
    return top + rand


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare raw vs canonical metrics and print a diff report")
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Manifest path (.json list or .csv) with results_dir entries (relative to coint4/ unless absolute).",
    )
    parser.add_argument(
        "--runs-root",
        action="append",
        default=[],
        help="Root dir to scan for strategy_metrics.csv (repeatable; relative to coint4/ unless absolute).",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Explicit run dir (repeatable; relative to coint4/ unless absolute).",
    )
    parser.add_argument(
        "--sample",
        default="all",
        help="Sampling spec: all | top5 | random5 | top5+random5 | top=5,random=5",
    )
    parser.add_argument("--seed", type=int, default=1, help="RNG seed for random sampling (default: 1)")
    parser.add_argument(
        "--thresholds",
        default="sharpe=0.05,pnl=1e-6,dd=1e-6",
        help="Abs diff thresholds: sharpe=..,pnl=..,dd=..",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be compared (no scans/reads).")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Always exit 0 (even when diffs exceed thresholds or files are missing).",
    )
    args = parser.parse_args()

    project_root = _resolve_project_root()

    manifest_paths: List[Path] = []
    for raw in args.manifest:
        resolved = _resolve_under_project(raw, project_root)
        if resolved is None:
            raise SystemExit("--manifest path is empty")
        manifest_paths.append(resolved)

    run_dirs: List[Path] = []
    for raw in args.run_dir:
        resolved = _resolve_under_project(raw, project_root)
        if resolved is None:
            continue
        run_dirs.append(resolved)

    runs_roots: List[Path] = []
    for raw in args.runs_root:
        resolved = _resolve_under_project(raw, project_root)
        if resolved is None:
            continue
        runs_roots.append(resolved)

    targets: List[_Target] = []

    for mp in manifest_paths:
        if not mp.exists():
            raise SystemExit(f"manifest not found: {mp}")
        rows = _load_manifest(mp)
        for row in rows:
            results_dir_str = str(row.get("results_dir") or "").strip()
            if not results_dir_str:
                continue
            resolved_run_dir = _resolve_under_project(results_dir_str, project_root)
            if resolved_run_dir is None:
                continue

            run_id = str(row.get("run_id") or "").strip()
            run_group = str(row.get("run_group") or "").strip()
            if not run_id or not run_group:
                inferred_id, inferred_group = _infer_run_id_group(resolved_run_dir, project_root)
                run_id = run_id or inferred_id
                run_group = run_group or inferred_group

            status = str(row.get("status") or "").strip()
            targets.append(
                _Target(
                    run_dir=resolved_run_dir,
                    run_id=run_id,
                    run_group=run_group,
                    status=status,
                    raw_hint_sharpe=_to_float(row.get("sharpe_ratio_abs_raw") or row.get("sharpe_ratio_abs")),
                    raw_hint_pnl=_to_float(row.get("total_pnl")),
                    raw_hint_dd=_to_float(row.get("max_drawdown_abs")),
                )
            )

    for rr in runs_roots:
        if args.dry_run:
            # Avoid scanning large trees in dry-run.
            continue
        for run_dir in _discover_run_dirs_from_runs_root(rr):
            run_id, run_group = _infer_run_id_group(run_dir, project_root)
            targets.append(
                _Target(
                    run_dir=run_dir,
                    run_id=run_id,
                    run_group=run_group,
                    status="",
                    raw_hint_sharpe=None,
                    raw_hint_pnl=None,
                    raw_hint_dd=None,
                )
            )

    for rd in run_dirs:
        run_id, run_group = _infer_run_id_group(rd, project_root)
        targets.append(
            _Target(
                run_dir=rd,
                run_id=run_id,
                run_group=run_group,
                status="",
                raw_hint_sharpe=None,
                raw_hint_pnl=None,
                raw_hint_dd=None,
            )
        )

    targets = _dedupe_targets_keep_order(targets)

    top_n, random_n = _parse_sample_spec(args.sample)
    thresholds = _parse_thresholds(args.thresholds)

    if args.dry_run:
        print("# Raw vs canonical metrics diff (dry-run)")
        print("")
        if manifest_paths:
            for mp in manifest_paths:
                print(f"- manifest: {_try_relpath(mp, project_root)}")
        if runs_roots:
            for rr in runs_roots:
                print(f"- runs_root: {_try_relpath(rr, project_root)} (dry-run: will NOT scan)")
        if run_dirs:
            for rd in run_dirs:
                print(f"- run_dir: {_try_relpath(rd, project_root)}")
        print("")
        if targets:
            print(f"Targets (from manifest/run_dir): {len(targets)}")
            for t in targets[:20]:
                print(f"- {_try_relpath(t.run_dir, project_root)}")
            if len(targets) > 20:
                print(f"- ... ({len(targets) - 20} more)")
        else:
            print("Targets: (unknown without scanning; pass --manifest/--run-dir or drop --dry-run)")
        print("")
        if top_n or random_n:
            print(f"Sample plan: top={top_n} random={random_n} seed={int(args.seed)}")
        else:
            print("Sample plan: all")
        print("")
        print(f"Thresholds (abs diff): {args.thresholds}")
        return 0

    if not targets:
        raise SystemExit("No targets found (use --manifest and/or --runs-root and/or --run-dir)")

    selected = _select_targets(
        targets,
        top_n=top_n,
        random_n=random_n,
        seed=int(args.seed),
        project_root=project_root,
        dry_run=False,
    )
    if not selected:
        raise SystemExit("Empty selection (check --sample)")

    print("# Raw vs canonical metrics diff")
    print("")
    if manifest_paths:
        for mp in manifest_paths:
            print(f"- manifest: {_try_relpath(mp, project_root)}")
    if runs_roots:
        for rr in runs_roots:
            print(f"- runs_root: {_try_relpath(rr, project_root)}")
    print("")
    if top_n or random_n:
        print(f"Sample: top={top_n} random={random_n} seed={int(args.seed)} -> selected={len(selected)}")
    else:
        print(f"Sample: all -> selected={len(selected)}")
    print("")
    print("Definitions:")
    print("- raw = strategy_metrics.csv (or manifest/run_index.csv when fields are present)")
    print("- canonical = canonical_metrics.json (computed from equity_curve.csv)")
    print("- Δ = canonical - raw")
    print("")
    print(f"Thresholds (abs diff): {args.thresholds}")
    print("")
    print(
        "| run_group | run_id | sharpe_raw | sharpe_canon | Δ_sharpe | pnl_raw | pnl_canon | Δ_pnl | dd_raw | dd_canon | Δ_dd | notes |"
    )
    print("| - | - | -: | -: | -: | -: | -: | -: | -: | -: | -: | - |")

    missing_raw = 0
    missing_canon = 0
    over_sharpe = 0
    over_pnl = 0
    over_dd = 0

    any_issues = False

    for t in selected:
        raw_present = False
        raw_metrics: Dict[str, Optional[float]] = {
            "sharpe_ratio_abs": t.raw_hint_sharpe,
            "total_pnl": t.raw_hint_pnl,
            "max_drawdown_abs": t.raw_hint_dd,
        }
        if any(v is None for v in raw_metrics.values()):
            raw_present, loaded = _load_raw_metrics_from_strategy_csv(t.run_dir)
            for k, v in loaded.items():
                if raw_metrics.get(k) is None:
                    raw_metrics[k] = v

        canon_present, canonical = _load_canonical_metrics(t.run_dir)

        raw_sharpe = raw_metrics.get("sharpe_ratio_abs")
        raw_pnl = raw_metrics.get("total_pnl")
        raw_dd = raw_metrics.get("max_drawdown_abs")

        canon_sharpe = canonical.get("canonical_sharpe")
        canon_pnl = canonical.get("canonical_pnl_abs")
        canon_dd = canonical.get("canonical_max_drawdown_abs")

        notes: List[str] = []
        if not raw_present and (t.raw_hint_sharpe is None and t.raw_hint_pnl is None and t.raw_hint_dd is None):
            notes.append("raw:missing")
        if raw_sharpe is None or raw_pnl is None or raw_dd is None:
            notes.append("raw:partial")
        if not canon_present:
            notes.append("canon:missing")
        if canon_sharpe is None or canon_pnl is None or canon_dd is None:
            notes.append("canon:partial")

        d_sharpe = None if raw_sharpe is None or canon_sharpe is None else float(canon_sharpe - raw_sharpe)
        d_pnl = None if raw_pnl is None or canon_pnl is None else float(canon_pnl - raw_pnl)
        d_dd = None if raw_dd is None or canon_dd is None else float(canon_dd - raw_dd)

        def _is_over(key: str, diff: Optional[float]) -> bool:
            thr = thresholds.get(key)
            if thr is None:
                return False
            if diff is None or not math.isfinite(diff):
                return False
            return abs(float(diff)) > float(thr)

        over = []
        if _is_over("sharpe", d_sharpe):
            over.append("sharpe")
            over_sharpe += 1
        if _is_over("pnl", d_pnl):
            over.append("pnl")
            over_pnl += 1
        if _is_over("dd", d_dd):
            over.append("dd")
            over_dd += 1
        if over:
            notes.append("over:" + ",".join(over))

        if raw_sharpe is None and raw_pnl is None and raw_dd is None:
            missing_raw += 1
        if not canon_present:
            missing_canon += 1

        if notes:
            any_issues = True

        print(
            "| {g} | {i} | {rs} | {cs} | {ds} | {rp} | {cp} | {dp} | {rd} | {cd} | {dd} | {n} |".format(
                g=t.run_group,
                i=t.run_id,
                rs=_fmt(raw_sharpe, digits=6),
                cs=_fmt(canon_sharpe, digits=6),
                ds=_fmt_signed(d_sharpe, digits=6),
                rp=_fmt(raw_pnl, digits=6),
                cp=_fmt(canon_pnl, digits=6),
                dp=_fmt_signed(d_pnl, digits=6),
                rd=_fmt(raw_dd, digits=6),
                cd=_fmt(canon_dd, digits=6),
                dd=_fmt_signed(d_dd, digits=6),
                n=";".join(notes),
            )
        )

    print("")
    print("Summary:")
    print(f"- selected: {len(selected)}")
    print(f"- missing_raw: {missing_raw}")
    print(f"- missing_canonical: {missing_canon}")
    print(f"- over_threshold: sharpe={over_sharpe} pnl={over_pnl} dd={over_dd}")

    if args.report_only:
        return 0

    # Fail when we cannot compare reliably or thresholds are exceeded.
    if missing_raw or missing_canon or over_sharpe or over_pnl or over_dd or any_issues:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

