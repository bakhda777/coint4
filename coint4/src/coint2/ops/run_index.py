"""Index builder for WFA run artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from coint2.core.sharpe import compute_sharpe_ratio_abs_from_equity_curve_csv


@dataclass
class RunIndexEntry:
    """Single run index entry."""

    run_id: str
    run_group: str
    results_dir: str
    metrics_path: str
    config_path: str
    status: str
    metrics_present: bool
    sharpe_ratio_abs: Optional[float]
    sharpe_ratio_abs_raw: Optional[float]
    sharpe_ratio_on_returns: Optional[float]
    total_pnl: Optional[float]
    max_drawdown_abs: Optional[float]
    max_drawdown_on_equity: Optional[float]
    total_trades: Optional[float]
    total_pairs_traded: Optional[float]
    total_costs: Optional[float]
    total_days: Optional[float]
    volatility: Optional[float]
    win_rate: Optional[float]
    best_pair_pnl: Optional[float]
    worst_pair_pnl: Optional[float]
    avg_pnl_per_pair: Optional[float]


def _normalize_path(path: Path, project_root: Path) -> str:
    target = path
    if not target.is_absolute():
        target = project_root / target
    return target.resolve().as_posix()


def _relative_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _infer_group(results_dir: Path, project_root: Path) -> Tuple[str, str]:
    run_id = results_dir.name
    run_group = ""
    try:
        rel = results_dir.resolve().relative_to(project_root.resolve())
        parts = rel.parts
        if len(parts) >= 4 and parts[0:3] == ("artifacts", "wfa", "runs"):
            run_group = parts[3]
        elif len(parts) >= 4 and parts[0:3] == ("artifacts", "wfa", "runs_clean"):
            run_group = parts[3]
    except ValueError:
        run_group = ""

    # Some pipelines store holdout/stress as a folder and the leaf is the base id.
    # Normalize into the same `holdout_<id>` / `stress_<id>` convention expected by
    # robust rankers.
    parent = results_dir.parent.name.strip().lower()
    if parent in {"holdout", "stress"}:
        prefix = f"{parent}_"
        if not run_id.startswith(prefix):
            run_id = f"{prefix}{run_id}"
    return run_id, run_group


def _load_metrics(metrics_path: Path) -> Dict[str, Optional[float]]:
    with metrics_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            return {key: _to_float(value) for key, value in row.items()}
    return {}


def _compute_sharpe_from_equity_curve(
    results_dir: Path,
    *,
    days_per_year: float = 365.0,
) -> Optional[float]:
    return compute_sharpe_ratio_abs_from_equity_curve_csv(
        results_dir / "equity_curve.csv",
        days_per_year=days_per_year,
    )


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    stripped = str(value).strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def _status_rank(status: str) -> int:
    normalized = status.lower()
    order = {
        "completed": 3,
        "running": 2,
        "active": 2,
        "stalled": 1,
        "planned": 0,
        "partial": 0,
    }
    return order.get(normalized, 0)


def load_run_queues(
    queue_paths: Iterable[Path], project_root: Path
) -> Dict[str, Dict[str, str]]:
    """Load run queues into a map keyed by absolute results_dir."""
    queue_map: Dict[str, Dict[str, str]] = {}
    for queue_path in queue_paths:
        if not queue_path.exists():
            continue
        with queue_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                results_dir = (row.get("results_dir") or "").strip()
                config_path = (row.get("config_path") or "").strip()
                status = (row.get("status") or "").strip()
                if not results_dir:
                    continue
                results_abs = _normalize_path(Path(results_dir), project_root)
                existing = queue_map.get(results_abs)
                if existing is None or _status_rank(status) > _status_rank(
                    existing.get("status", "")
                ):
                    queue_map[results_abs] = {
                        "status": status,
                        "config_path": config_path,
                    }
    return queue_map


def _load_canonical_metrics(canonical_path: Path) -> Dict[str, Optional[float]]:
    try:
        payload = json.loads(canonical_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(payload, dict):
        return {}

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return {}

    def _num(key: str) -> Optional[float]:
        value = metrics.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    return {
        "canonical_sharpe": _num("canonical_sharpe"),
        "canonical_pnl_abs": _num("canonical_pnl_abs"),
        "canonical_max_drawdown_abs": _num("canonical_max_drawdown_abs"),
    }


def build_run_index(
    runs_dir: Path, queue_paths: Iterable[Path], project_root: Path
) -> List[RunIndexEntry]:
    """Build run index entries from metrics and queue data."""
    queue_map = load_run_queues(queue_paths, project_root)
    entries: Dict[str, RunIndexEntry] = {}

    for results_abs, queue_info in queue_map.items():
        results_path = Path(results_abs)
        run_id, run_group = _infer_group(results_path, project_root)
        entry = RunIndexEntry(
            run_id=run_id,
            run_group=run_group,
            results_dir=_relative_path(results_path, project_root),
            metrics_path="",
            config_path=queue_info.get("config_path", ""),
            status=queue_info.get("status", ""),
            metrics_present=False,
            sharpe_ratio_abs=None,
            sharpe_ratio_abs_raw=None,
            sharpe_ratio_on_returns=None,
            total_pnl=None,
            max_drawdown_abs=None,
            max_drawdown_on_equity=None,
            total_trades=None,
            total_pairs_traded=None,
            total_costs=None,
            total_days=None,
            volatility=None,
            win_rate=None,
            best_pair_pnl=None,
            worst_pair_pnl=None,
            avg_pnl_per_pair=None,
        )
        entries[results_abs] = entry

    # Fast-path: populate metrics for all queue-referenced results dirs (runs and runs_clean).
    # This avoids relying on `runs_dir` being "the right root" when different pipelines write
    # to different trees (e.g. artifacts/wfa/runs_clean/**).
    for results_abs, entry in list(entries.items()):
        results_path = Path(results_abs)
        metrics_path = results_path / "strategy_metrics.csv"
        if not metrics_path.exists():
            continue

        metrics = _load_metrics(metrics_path)
        entry.metrics_present = True
        entry.metrics_path = _relative_path(metrics_path, project_root)
        entry.sharpe_ratio_abs_raw = metrics.get("sharpe_ratio_abs")
        entry.sharpe_ratio_on_returns = metrics.get("sharpe_ratio_on_returns")

        computed_sharpe = _compute_sharpe_from_equity_curve(results_path)
        entry.sharpe_ratio_abs = (
            computed_sharpe if computed_sharpe is not None else entry.sharpe_ratio_abs_raw
        )
        entry.total_pnl = metrics.get("total_pnl")
        entry.max_drawdown_abs = metrics.get("max_drawdown_abs")
        entry.max_drawdown_on_equity = metrics.get("max_drawdown_on_equity")
        entry.total_trades = metrics.get("total_trades")
        entry.total_pairs_traded = metrics.get("total_pairs_traded")
        entry.total_costs = metrics.get("total_costs")
        entry.total_days = metrics.get("total_days")
        entry.volatility = metrics.get("volatility")
        entry.win_rate = metrics.get("win_rate")
        entry.best_pair_pnl = metrics.get("best_pair_pnl")
        entry.worst_pair_pnl = metrics.get("worst_pair_pnl")
        entry.avg_pnl_per_pair = metrics.get("avg_pnl_per_pair")

        canonical_path = results_path / "canonical_metrics.json"
        if canonical_path.exists():
            canonical = _load_canonical_metrics(canonical_path)
            canonical_sharpe = canonical.get("canonical_sharpe")
            canonical_pnl_abs = canonical.get("canonical_pnl_abs")
            canonical_max_dd_abs = canonical.get("canonical_max_drawdown_abs")
            if canonical_sharpe is not None:
                entry.sharpe_ratio_abs = canonical_sharpe
            if canonical_pnl_abs is not None:
                entry.total_pnl = canonical_pnl_abs
            if canonical_max_dd_abs is not None:
                entry.max_drawdown_abs = canonical_max_dd_abs

    for metrics_path in runs_dir.rglob("strategy_metrics.csv"):
        results_path = metrics_path.parent
        results_abs = _normalize_path(results_path, project_root)
        if results_abs in entries:
            # Already processed from a run_queue entry.
            continue
        metrics = _load_metrics(metrics_path)
        run_id, run_group = _infer_group(results_path, project_root)
        entry = entries.get(results_abs)
        if entry is None:
            entry = RunIndexEntry(
                run_id=run_id,
                run_group=run_group,
                results_dir=_relative_path(results_path, project_root),
                metrics_path="",
                config_path="",
                status="unknown",
                metrics_present=False,
                sharpe_ratio_abs=None,
                sharpe_ratio_abs_raw=None,
                sharpe_ratio_on_returns=None,
                total_pnl=None,
                max_drawdown_abs=None,
                max_drawdown_on_equity=None,
                total_trades=None,
                total_pairs_traded=None,
                total_costs=None,
                total_days=None,
                volatility=None,
                win_rate=None,
                best_pair_pnl=None,
                worst_pair_pnl=None,
                avg_pnl_per_pair=None,
            )
            entries[results_abs] = entry

        entry.metrics_present = True
        entry.metrics_path = _relative_path(metrics_path, project_root)
        entry.sharpe_ratio_abs_raw = metrics.get("sharpe_ratio_abs")
        entry.sharpe_ratio_on_returns = metrics.get("sharpe_ratio_on_returns")

        computed_sharpe = _compute_sharpe_from_equity_curve(results_path)
        entry.sharpe_ratio_abs = (
            computed_sharpe if computed_sharpe is not None else entry.sharpe_ratio_abs_raw
        )
        entry.total_pnl = metrics.get("total_pnl")
        entry.max_drawdown_abs = metrics.get("max_drawdown_abs")
        entry.max_drawdown_on_equity = metrics.get("max_drawdown_on_equity")
        entry.total_trades = metrics.get("total_trades")
        entry.total_pairs_traded = metrics.get("total_pairs_traded")
        entry.total_costs = metrics.get("total_costs")
        entry.total_days = metrics.get("total_days")
        entry.volatility = metrics.get("volatility")
        entry.win_rate = metrics.get("win_rate")
        entry.best_pair_pnl = metrics.get("best_pair_pnl")
        entry.worst_pair_pnl = metrics.get("worst_pair_pnl")
        entry.avg_pnl_per_pair = metrics.get("avg_pnl_per_pair")

        canonical_path = results_path / "canonical_metrics.json"
        if canonical_path.exists():
            canonical = _load_canonical_metrics(canonical_path)
            canonical_sharpe = canonical.get("canonical_sharpe")
            canonical_pnl_abs = canonical.get("canonical_pnl_abs")
            canonical_max_dd_abs = canonical.get("canonical_max_drawdown_abs")
            if canonical_sharpe is not None:
                entry.sharpe_ratio_abs = canonical_sharpe
            if canonical_pnl_abs is not None:
                entry.total_pnl = canonical_pnl_abs
            if canonical_max_dd_abs is not None:
                entry.max_drawdown_abs = canonical_max_dd_abs

    return list(entries.values())


def write_run_index_csv(path: Path, entries: Sequence[RunIndexEntry]) -> None:
    """Write run index entries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(entries[0]).keys()) if entries else [
        "run_id",
        "run_group",
        "results_dir",
        "metrics_path",
        "config_path",
        "status",
        "metrics_present",
        "sharpe_ratio_abs",
        "sharpe_ratio_abs_raw",
        "sharpe_ratio_on_returns",
        "total_pnl",
        "max_drawdown_abs",
        "max_drawdown_on_equity",
        "total_trades",
        "total_pairs_traded",
        "total_costs",
        "total_days",
        "volatility",
        "win_rate",
        "best_pair_pnl",
        "worst_pair_pnl",
        "avg_pnl_per_pair",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            writer.writerow(asdict(entry))


def write_run_index_json(path: Path, entries: Sequence[RunIndexEntry]) -> None:
    """Write run index entries to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(entry) for entry in entries]
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
