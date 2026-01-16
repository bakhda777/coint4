"""Index builder for WFA run artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


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
    total_pnl: Optional[float]
    max_drawdown_abs: Optional[float]
    total_trades: Optional[float]
    total_pairs_traded: Optional[float]
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
    except ValueError:
        run_group = ""
    return run_id, run_group


def _load_metrics(metrics_path: Path) -> Dict[str, Optional[float]]:
    with metrics_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            return {key: _to_float(value) for key, value in row.items()}
    return {}


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
            total_pnl=None,
            max_drawdown_abs=None,
            total_trades=None,
            total_pairs_traded=None,
            best_pair_pnl=None,
            worst_pair_pnl=None,
            avg_pnl_per_pair=None,
        )
        entries[results_abs] = entry

    for metrics_path in runs_dir.rglob("strategy_metrics.csv"):
        results_path = metrics_path.parent
        results_abs = _normalize_path(results_path, project_root)
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
                total_pnl=None,
                max_drawdown_abs=None,
                total_trades=None,
                total_pairs_traded=None,
                best_pair_pnl=None,
                worst_pair_pnl=None,
                avg_pnl_per_pair=None,
            )
            entries[results_abs] = entry

        entry.metrics_present = True
        entry.metrics_path = _relative_path(metrics_path, project_root)
        entry.sharpe_ratio_abs = metrics.get("sharpe_ratio_abs")
        entry.total_pnl = metrics.get("total_pnl")
        entry.max_drawdown_abs = metrics.get("max_drawdown_abs")
        entry.total_trades = metrics.get("total_trades")
        entry.total_pairs_traded = metrics.get("total_pairs_traded")
        entry.best_pair_pnl = metrics.get("best_pair_pnl")
        entry.worst_pair_pnl = metrics.get("worst_pair_pnl")
        entry.avg_pnl_per_pair = metrics.get("avg_pnl_per_pair")

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
        "total_pnl",
        "max_drawdown_abs",
        "total_trades",
        "total_pairs_traded",
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

