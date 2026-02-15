#!/usr/bin/env python3
"""Select deterministic TOP-N runs from rollup run_index.csv and write baseline_manifest.json.

Goal (C02 in tasks/clean_cycle_top10/prd_clean_cycle_top10.json):
- Create a reproducible TOP-10 manifest with stable ordering and values.
- Include config sha256 and basic artifact presence flags.

Ranking:
- Use canonical_sharpe (from canonical_metrics.json) if present,
  else sharpe_ratio_abs, else sharpe_ratio_abs_raw.
- Stable tie-breakers: sharpe desc, abs(dd) asc, pnl desc, run_group+run_id asc.

Notes:
- The manifest intentionally does not include non-deterministic fields (e.g. current timestamps).
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


def _to_float(value: str) -> Optional[float]:
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


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _try_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _resolve_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: assume we are under coint4/scripts/**.
    return here.parents[3]


def _normalize_repo_relative_path(raw: str) -> str:
    """Normalize a path string so it is project-root relative when possible."""
    value = str(raw or "").strip()
    if not value:
        return ""
    # Some tools/docs may store paths relative to repo root (prefix "coint4/").
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
        raise ValueError(f"canonical metrics must be a JSON object: {path}")

    # Allow future schema: either flat keys or nested under "metrics".
    metrics_node: Any = payload.get("metrics")
    if isinstance(metrics_node, dict):
        source = metrics_node
    else:
        source = payload

    def _get(key: str) -> Optional[float]:
        value = source.get(key)
        if value is None:
            return None
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(out):
            return None
        return out

    return True, {
        "canonical_sharpe": _get("canonical_sharpe"),
        "canonical_pnl_abs": _get("canonical_pnl_abs"),
        "canonical_max_drawdown_abs": _get("canonical_max_drawdown_abs"),
    }


def _pick_first_non_none(candidates: Iterable[Tuple[str, Optional[float]]]) -> Tuple[Optional[float], str]:
    for name, value in candidates:
        if value is None:
            continue
        if not math.isfinite(value):
            continue
        return value, name
    return None, "none"


@dataclass(frozen=True)
class _RunRow:
    run_id: str
    run_group: str
    results_dir: str
    config_path: str
    status: str
    sharpe_ratio_abs: Optional[float]
    sharpe_ratio_abs_raw: Optional[float]
    total_pnl: Optional[float]
    max_drawdown_abs: Optional[float]
    total_trades: Optional[float]
    total_pairs_traded: Optional[float]
    total_costs: Optional[float]
    metrics_present_index: bool


def _read_run_index(path: Path) -> List[_RunRow]:
    rows: List[_RunRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                _RunRow(
                    run_id=(row.get("run_id") or "").strip(),
                    run_group=(row.get("run_group") or "").strip(),
                    results_dir=(row.get("results_dir") or "").strip(),
                    config_path=(row.get("config_path") or "").strip(),
                    status=(row.get("status") or "").strip(),
                    metrics_present_index=_to_bool(row.get("metrics_present") or ""),
                    sharpe_ratio_abs=_to_float(row.get("sharpe_ratio_abs") or ""),
                    sharpe_ratio_abs_raw=_to_float(row.get("sharpe_ratio_abs_raw") or ""),
                    total_pnl=_to_float(row.get("total_pnl") or ""),
                    max_drawdown_abs=_to_float(row.get("max_drawdown_abs") or ""),
                    total_trades=_to_float(row.get("total_trades") or ""),
                    total_pairs_traded=_to_float(row.get("total_pairs_traded") or ""),
                    total_costs=_to_float(row.get("total_costs") or ""),
                )
            )
    return rows


def _sort_key(entry: Dict[str, Any]) -> Tuple[Any, ...]:
    sharpe = entry.get("rank_sharpe")
    dd_abs = entry.get("rank_max_drawdown_abs")
    pnl = entry.get("rank_pnl_abs")

    sharpe_key = float("inf") if sharpe is None else -float(sharpe)
    dd_key = float("inf") if dd_abs is None else abs(float(dd_abs))
    pnl_key = float("inf") if pnl is None else -float(pnl)

    return (
        sharpe_key,
        dd_key,
        pnl_key,
        str(entry.get("run_group") or ""),
        str(entry.get("run_id") or ""),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Select deterministic TOP-N runs and write baseline_manifest.json")
    parser.add_argument(
        "--run-index",
        default="artifacts/wfa/aggregate/rollup/run_index.csv",
        help="Path to rollup run_index.csv (relative to coint4/ unless absolute).",
    )
    parser.add_argument(
        "--output",
        default="artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/baseline_manifest.json",
        help="Where to write baseline_manifest.json (relative to coint4/ unless absolute).",
    )
    parser.add_argument("--top-n", type=int, default=10, help="How many entries to select (default: 10).")
    parser.add_argument(
        "--max-abs-dd",
        type=float,
        default=None,
        help="Optional filter: keep only runs with abs(max_drawdown_abs) <= this threshold.",
    )
    parser.add_argument(
        "--include-noncompleted",
        action="store_true",
        help="Include rows where status is not 'completed'.",
    )
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument(
        "--refuse-overwrite",
        action="store_true",
        help="Refuse overwriting an existing output file (default).",
    )
    overwrite_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file.",
    )
    args = parser.parse_args()

    project_root = _resolve_project_root()
    run_index_path = _resolve_under_project(args.run_index, project_root) or (project_root / args.run_index)
    output_path = _resolve_under_project(args.output, project_root) or (project_root / args.output)

    if not run_index_path.exists():
        raise SystemExit(f"run index not found: {run_index_path}")

    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing manifest: {output_path} (use --overwrite)")

    rows = _read_run_index(run_index_path)
    if not rows:
        raise SystemExit(f"run index is empty: {run_index_path}")

    top_n = max(1, int(args.top_n))
    max_abs_dd = args.max_abs_dd
    if max_abs_dd is not None and max_abs_dd < 0:
        max_abs_dd = abs(float(max_abs_dd))

    candidates: List[Dict[str, Any]] = []
    for row in rows:
        if not row.run_id or not row.results_dir:
            continue
        if not args.include_noncompleted and row.status.strip().lower() != "completed":
            continue

        results_path = _resolve_under_project(row.results_dir, project_root)
        canonical_present = False
        canonical: Dict[str, Optional[float]] = {
            "canonical_sharpe": None,
            "canonical_pnl_abs": None,
            "canonical_max_drawdown_abs": None,
        }
        if results_path and results_path.exists():
            canonical_present, canonical = _load_optional_canonical_metrics(results_path)

        rank_sharpe, rank_sharpe_source = _pick_first_non_none(
            [
                ("canonical_sharpe", canonical.get("canonical_sharpe")),
                ("sharpe_ratio_abs", row.sharpe_ratio_abs),
                ("sharpe_ratio_abs_raw", row.sharpe_ratio_abs_raw),
            ]
        )
        rank_pnl, _ = _pick_first_non_none(
            [
                ("canonical_pnl_abs", canonical.get("canonical_pnl_abs")),
                ("total_pnl", row.total_pnl),
            ]
        )
        rank_dd, _ = _pick_first_non_none(
            [
                ("canonical_max_drawdown_abs", canonical.get("canonical_max_drawdown_abs")),
                ("max_drawdown_abs", row.max_drawdown_abs),
            ]
        )

        if rank_sharpe is None:
            continue
        if max_abs_dd is not None:
            if rank_dd is None:
                continue
            if abs(rank_dd) > max_abs_dd:
                continue

        candidates.append(
            {
                "run_group": row.run_group,
                "run_id": row.run_id,
                "results_dir": _normalize_repo_relative_path(row.results_dir),
                "config_path": _normalize_repo_relative_path(row.config_path) or None,
                "status": row.status,
                "rank_sharpe": rank_sharpe,
                "rank_sharpe_source": rank_sharpe_source,
                "rank_pnl_abs": rank_pnl,
                "rank_max_drawdown_abs": rank_dd,
                "canonical_metrics_present": canonical_present,
                "canonical_sharpe": canonical.get("canonical_sharpe"),
                "canonical_pnl_abs": canonical.get("canonical_pnl_abs"),
                "canonical_max_drawdown_abs": canonical.get("canonical_max_drawdown_abs"),
                "metrics_present_index": row.metrics_present_index,
                "sharpe_ratio_abs": row.sharpe_ratio_abs,
                "sharpe_ratio_abs_raw": row.sharpe_ratio_abs_raw,
                "total_pnl": row.total_pnl,
                "max_drawdown_abs": row.max_drawdown_abs,
                "total_trades": row.total_trades,
                "total_pairs_traded": row.total_pairs_traded,
                "total_costs": row.total_costs,
            }
        )

    candidates.sort(key=_sort_key)
    selected = candidates[:top_n]

    if len(selected) != top_n:
        raise SystemExit(f"not enough eligible runs: need {top_n}, got {len(selected)}")

    # Enrich with sha256 + file existence flags.
    manifest: List[Dict[str, Any]] = []
    for idx, entry in enumerate(selected, 1):
        results_dir = str(entry.get("results_dir") or "")
        results_path = _resolve_under_project(results_dir, project_root) if results_dir else None
        equity_present = bool(results_path and (results_path / "equity_curve.csv").exists())
        metrics_present = bool(results_path and (results_path / "strategy_metrics.csv").exists())

        config_sha256 = None
        config_path = entry.get("config_path")
        resolved_cfg = _resolve_under_project(str(config_path or ""), project_root) if config_path else None
        if resolved_cfg and resolved_cfg.exists() and resolved_cfg.is_file():
            config_sha256 = _sha256_file(resolved_cfg)

        enriched = dict(entry)
        enriched["rank"] = idx
        enriched["equity_present"] = equity_present
        enriched["metrics_present"] = metrics_present
        enriched["config_sha256"] = config_sha256
        manifest.append(enriched)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(output_path)

    print(f"Wrote baseline manifest: {_try_relpath(output_path, project_root)}")
    print(f"Source run_index.csv: {_try_relpath(run_index_path, project_root)} (sha256={_sha256_file(run_index_path)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

