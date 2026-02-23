"""Index builder for WFA run artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from coint2.core.performance import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from coint2.core.sharpe import compute_sharpe_ratio_abs_from_equity_curve_csv

try:
    import yaml
except Exception:  # noqa: BLE001
    yaml = None


@dataclass
class RunIndexEntry:
    """Single run index entry."""

    run_id: str
    run_group: str
    results_dir: str
    metrics_path: str
    config_path: str
    universe_path: str
    universe_tag: str
    universe_pairs_count: Optional[int]
    denylist_count: int
    denylist_hash: str
    status: str
    metrics_present: bool
    sharpe_ratio_abs: Optional[float]
    sharpe_ratio_abs_raw: Optional[float]
    sharpe_ratio_on_returns: Optional[float]
    psr: Optional[float]
    dsr: Optional[float]
    dsr_trials: Optional[int]
    total_pnl: Optional[float]
    max_drawdown_abs: Optional[float]
    max_drawdown_on_equity: Optional[float]
    total_trades: Optional[float]
    total_pairs_traded: Optional[float]
    total_costs: Optional[float]
    total_days: Optional[float]
    expected_test_days: Optional[float]
    observed_test_days: Optional[float]
    coverage_ratio: Optional[float]
    zero_pnl_days: Optional[float]
    zero_pnl_days_pct: Optional[float]
    missing_test_days: Optional[float]
    volatility: Optional[float]
    win_rate: Optional[float]
    best_pair_pnl: Optional[float]
    worst_pair_pnl: Optional[float]
    avg_pnl_per_pair: Optional[float]
    tail_loss_pair_total_abs: Optional[float]
    tail_loss_worst_pair: str
    tail_loss_worst_pair_pnl: Optional[float]
    tail_loss_worst_pair_share: Optional[float]
    tail_loss_period_total_abs: Optional[float]
    tail_loss_worst_period: str
    tail_loss_worst_period_pnl: Optional[float]
    tail_loss_worst_period_share: Optional[float]


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


def _load_equity_returns(results_dir: Path) -> List[float]:
    """Load per-step returns from equity_curve.csv."""
    equity_curve_path = results_dir / "equity_curve.csv"
    if not equity_curve_path.exists():
        return []

    returns: List[float] = []
    prev_ts: Optional[datetime] = None
    prev_equity: Optional[float] = None

    with equity_curve_path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2:
                continue
            try:
                ts = datetime.fromisoformat(str(row[0]).strip())
                equity = float(row[1])
            except (TypeError, ValueError):
                continue
            if prev_ts is None or prev_equity is None or prev_equity == 0:
                prev_ts = ts
                prev_equity = equity
                continue
            if (ts - prev_ts).total_seconds() <= 0:
                prev_ts = ts
                prev_equity = equity
                continue
            returns.append((equity - prev_equity) / prev_equity)
            prev_ts = ts
            prev_equity = equity
    return returns


def _load_daily_pnl_as_returns(results_dir: Path) -> List[float]:
    """Fallback: use daily_pnl.csv values as return-like series."""
    daily_path = results_dir / "daily_pnl.csv"
    if not daily_path.exists():
        return []

    with daily_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if not fields:
            return []
        pnl_key: Optional[str] = None
        for field in fields:
            if "pnl" in field.lower():
                pnl_key = field
                break
        if pnl_key is None and len(fields) >= 2:
            pnl_key = fields[1]
        if pnl_key is None:
            return []

        values: List[float] = []
        for row in reader:
            value = _to_float(str(row.get(pnl_key) or ""))
            if value is not None:
                values.append(float(value))
        return values


def _parse_iso_date(raw: str) -> Optional[date]:
    token = str(raw or "").strip()
    if not token:
        return None
    token = token.rstrip("Z")
    try:
        return datetime.fromisoformat(token).date()
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(token[:10]).date()
    except (TypeError, ValueError):
        return None


def _coverage_from_daily_pnl_csv(
    daily_pnl_path: Path,
    *,
    start_date: str,
    end_date: str,
    eps: float = 1e-12,
) -> Dict[str, Optional[float]]:
    """Compute coverage metrics from daily_pnl.csv using only config dates for expected days."""
    try:
        start = datetime.fromisoformat(str(start_date).strip()).date()
        end = datetime.fromisoformat(str(end_date).strip()).date()
    except ValueError:
        return {}

    expected = int((end - start).days + 1) if end >= start else 0
    if expected <= 0:
        return {
            "expected_test_days": float(expected),
            "observed_test_days": 0.0,
            "coverage_ratio": float("nan"),
            "zero_pnl_days": 0.0,
            "zero_pnl_days_pct": float("nan"),
            "missing_test_days": float("nan"),
        }

    if not daily_pnl_path.exists():
        return {
            "expected_test_days": float(expected),
            "observed_test_days": 0.0,
            "coverage_ratio": 0.0,
            "zero_pnl_days": 0.0,
            "zero_pnl_days_pct": 0.0,
            "missing_test_days": float(expected),
        }

    pnl_by_day: Dict[date, float] = {}
    try:
        with daily_pnl_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            if not fields:
                return {}
            date_key = fields[0]
            pnl_key: Optional[str] = None
            for field in fields:
                if "pnl" in field.lower():
                    pnl_key = field
                    break
            if pnl_key is None and len(fields) >= 2:
                pnl_key = fields[1]
            if pnl_key is None:
                return {}

            for row in reader:
                day = _parse_iso_date(str(row.get(date_key) or ""))
                if day is None:
                    continue
                pnl = _to_float(str(row.get(pnl_key) or ""))
                if pnl is None:
                    continue
                pnl_by_day[day] = float(pnl_by_day.get(day, 0.0) + float(pnl))
    except OSError:
        return {}

    observed = int(len(pnl_by_day))
    zero_days = int(sum(1 for pnl in pnl_by_day.values() if abs(float(pnl)) < float(eps)))
    coverage = float(observed) / float(expected)
    zero_pct = float(zero_days) / float(expected)
    missing = float(expected - observed)

    return {
        "expected_test_days": float(expected),
        "observed_test_days": float(observed),
        "coverage_ratio": float(coverage),
        "zero_pnl_days": float(zero_days),
        "zero_pnl_days_pct": float(zero_pct),
        "missing_test_days": float(missing),
    }


def _safe_psr_dsr(
    *,
    returns: Sequence[float],
    trials: Optional[int],
) -> Tuple[Optional[float], Optional[float]]:
    if len(returns) < 2:
        return None, None
    try:
        psr_raw = probabilistic_sharpe_ratio(list(returns), benchmark_sr=0.0)
        psr = float(psr_raw)
        if not math.isfinite(psr) or not (0.0 <= psr <= 1.0):
            psr = None
    except Exception:  # noqa: BLE001
        psr = None

    dsr: Optional[float] = None
    if trials is not None and int(trials) >= 2:
        try:
            dsr_raw = deflated_sharpe_ratio(list(returns), trials=int(trials))
            dsr = float(dsr_raw)
            if not math.isfinite(dsr):
                dsr = None
        except Exception:  # noqa: BLE001
            dsr = None
    return psr, dsr


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


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _normalize_symbols(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    normalized: List[str] = []
    for item in values:
        token = str(item).strip().upper()
        if token:
            normalized.append(token)
    return sorted(set(normalized))


def _symbols_hash(symbols: Sequence[str]) -> str:
    if not symbols:
        return ""
    payload = "\n".join(sorted(symbols))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _load_yaml_dict(path: Path) -> Dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_repo_path(raw_path: str, project_root: Path) -> Optional[Path]:
    candidate = str(raw_path or "").strip()
    if not candidate:
        return None
    path = Path(candidate)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _extract_universe_context(
    *,
    config_path: str,
    project_root: Path,
    config_cache: Dict[str, Dict[str, Any]],
    universe_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    context: Dict[str, Any] = {
        "universe_path": "",
        "universe_tag": "",
        "universe_pairs_count": None,
        "denylist_count": 0,
        "denylist_hash": "",
    }
    config_abs = _resolve_repo_path(config_path, project_root)
    if config_abs is None:
        return context

    config_key = config_abs.as_posix()
    payload = config_cache.get(config_key)
    if payload is None:
        payload = _load_yaml_dict(config_abs)
        config_cache[config_key] = payload
    if not payload:
        return context

    data_filters = payload.get("data_filters")
    if isinstance(data_filters, dict):
        denylisted = _normalize_symbols(data_filters.get("exclude_symbols"))
        context["denylist_count"] = len(denylisted)
        context["denylist_hash"] = _symbols_hash(denylisted)

    walk_forward = payload.get("walk_forward")
    pairs_file = ""
    if isinstance(walk_forward, dict):
        pairs_file = str(walk_forward.get("pairs_file") or "").strip()
    if not pairs_file:
        return context

    universe_abs = _resolve_repo_path(pairs_file, project_root)
    if universe_abs is None:
        return context

    context["universe_path"] = _relative_path(universe_abs, project_root)
    context["universe_tag"] = universe_abs.stem

    universe_key = universe_abs.as_posix()
    universe_payload = universe_cache.get(universe_key)
    if universe_payload is None:
        universe_payload = _load_yaml_dict(universe_abs)
        universe_cache[universe_key] = universe_payload

    if not universe_payload:
        return context

    pairs = universe_payload.get("pairs")
    if isinstance(pairs, list):
        context["universe_pairs_count"] = len(pairs)

    metadata = universe_payload.get("metadata")
    if not isinstance(metadata, dict):
        return context

    for key in ("tag", "version", "name", "generated"):
        value = str(metadata.get(key) or "").strip()
        if value:
            context["universe_tag"] = value
            break

    pruning = metadata.get("pruning")
    if isinstance(pruning, dict):
        denylisted = _normalize_symbols(pruning.get("denylisted_symbols"))
        if denylisted:
            context["denylist_count"] = len(denylisted)
            context["denylist_hash"] = _symbols_hash(denylisted)
        if context["universe_pairs_count"] is None:
            remaining_pairs = _to_int(pruning.get("remaining_pairs"))
            if remaining_pairs is not None:
                context["universe_pairs_count"] = remaining_pairs

    return context


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


def _tail_loss_bucket_metrics(pnl_by_bucket: Dict[str, float]) -> Dict[str, Optional[float] | str]:
    loss_total_abs = float(sum(-pnl for pnl in pnl_by_bucket.values() if pnl < 0))
    if loss_total_abs <= 0.0:
        return {
            "loss_total_abs": 0.0,
            "worst_bucket": "",
            "worst_bucket_pnl": None,
            "worst_bucket_share": None,
        }

    worst_bucket, worst_bucket_pnl = min(pnl_by_bucket.items(), key=lambda kv: kv[1])
    if worst_bucket_pnl >= 0:
        return {
            "loss_total_abs": loss_total_abs,
            "worst_bucket": "",
            "worst_bucket_pnl": None,
            "worst_bucket_share": None,
        }

    return {
        "loss_total_abs": loss_total_abs,
        "worst_bucket": str(worst_bucket),
        "worst_bucket_pnl": float(worst_bucket_pnl),
        "worst_bucket_share": float(abs(worst_bucket_pnl) / loss_total_abs),
    }


def _load_tail_loss_diagnostics(results_dir: Path) -> Dict[str, Optional[float] | str]:
    empty: Dict[str, Optional[float] | str] = {
        "tail_loss_pair_total_abs": None,
        "tail_loss_worst_pair": "",
        "tail_loss_worst_pair_pnl": None,
        "tail_loss_worst_pair_share": None,
        "tail_loss_period_total_abs": None,
        "tail_loss_worst_period": "",
        "tail_loss_worst_period_pnl": None,
        "tail_loss_worst_period_share": None,
    }

    trade_stats_path = results_dir / "trade_statistics.csv"
    if not trade_stats_path.exists():
        return empty

    pair_pnl: Dict[str, float] = {}
    period_pnl: Dict[str, float] = {}
    try:
        with trade_stats_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                pnl = _to_float(str(row.get("total_pnl") or ""))
                if pnl is None:
                    continue
                pair = str(row.get("pair") or "").strip()
                period = str(row.get("period") or "").strip()
                if pair:
                    pair_pnl[pair] = float(pair_pnl.get(pair, 0.0) + pnl)
                if period:
                    period_pnl[period] = float(period_pnl.get(period, 0.0) + pnl)
    except OSError:
        return empty

    pair_metrics = _tail_loss_bucket_metrics(pair_pnl)
    period_metrics = _tail_loss_bucket_metrics(period_pnl)
    return {
        "tail_loss_pair_total_abs": pair_metrics.get("loss_total_abs"),
        "tail_loss_worst_pair": str(pair_metrics.get("worst_bucket") or ""),
        "tail_loss_worst_pair_pnl": pair_metrics.get("worst_bucket_pnl"),
        "tail_loss_worst_pair_share": pair_metrics.get("worst_bucket_share"),
        "tail_loss_period_total_abs": period_metrics.get("loss_total_abs"),
        "tail_loss_worst_period": str(period_metrics.get("worst_bucket") or ""),
        "tail_loss_worst_period_pnl": period_metrics.get("worst_bucket_pnl"),
        "tail_loss_worst_period_share": period_metrics.get("worst_bucket_share"),
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
            universe_path="",
            universe_tag="",
            universe_pairs_count=None,
            denylist_count=0,
            denylist_hash="",
            status=queue_info.get("status", ""),
            metrics_present=False,
            sharpe_ratio_abs=None,
            sharpe_ratio_abs_raw=None,
            sharpe_ratio_on_returns=None,
            psr=None,
            dsr=None,
            dsr_trials=None,
            total_pnl=None,
            max_drawdown_abs=None,
            max_drawdown_on_equity=None,
            total_trades=None,
            total_pairs_traded=None,
            total_costs=None,
            total_days=None,
            expected_test_days=None,
            observed_test_days=None,
            coverage_ratio=None,
            zero_pnl_days=None,
            zero_pnl_days_pct=None,
            missing_test_days=None,
            volatility=None,
            win_rate=None,
            best_pair_pnl=None,
            worst_pair_pnl=None,
            avg_pnl_per_pair=None,
            tail_loss_pair_total_abs=None,
            tail_loss_worst_pair="",
            tail_loss_worst_pair_pnl=None,
            tail_loss_worst_pair_share=None,
            tail_loss_period_total_abs=None,
            tail_loss_worst_period="",
            tail_loss_worst_period_pnl=None,
            tail_loss_worst_period_share=None,
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
        entry.expected_test_days = metrics.get("expected_test_days")
        entry.observed_test_days = metrics.get("observed_test_days")
        entry.coverage_ratio = metrics.get("coverage_ratio")
        entry.zero_pnl_days = metrics.get("zero_pnl_days")
        entry.zero_pnl_days_pct = metrics.get("zero_pnl_days_pct")
        entry.missing_test_days = metrics.get("missing_test_days")
        entry.volatility = metrics.get("volatility")
        entry.win_rate = metrics.get("win_rate")
        entry.best_pair_pnl = metrics.get("best_pair_pnl")
        entry.worst_pair_pnl = metrics.get("worst_pair_pnl")
        entry.avg_pnl_per_pair = metrics.get("avg_pnl_per_pair")
        tail_diag = _load_tail_loss_diagnostics(results_path)
        entry.tail_loss_pair_total_abs = tail_diag.get("tail_loss_pair_total_abs")
        entry.tail_loss_worst_pair = str(tail_diag.get("tail_loss_worst_pair") or "")
        entry.tail_loss_worst_pair_pnl = tail_diag.get("tail_loss_worst_pair_pnl")
        entry.tail_loss_worst_pair_share = tail_diag.get("tail_loss_worst_pair_share")
        entry.tail_loss_period_total_abs = tail_diag.get("tail_loss_period_total_abs")
        entry.tail_loss_worst_period = str(tail_diag.get("tail_loss_worst_period") or "")
        entry.tail_loss_worst_period_pnl = tail_diag.get("tail_loss_worst_period_pnl")
        entry.tail_loss_worst_period_share = tail_diag.get("tail_loss_worst_period_share")

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
                universe_path="",
                universe_tag="",
                universe_pairs_count=None,
                denylist_count=0,
                denylist_hash="",
                status="unknown",
                metrics_present=False,
                sharpe_ratio_abs=None,
                sharpe_ratio_abs_raw=None,
                sharpe_ratio_on_returns=None,
                psr=None,
                dsr=None,
                dsr_trials=None,
                total_pnl=None,
                max_drawdown_abs=None,
                max_drawdown_on_equity=None,
                total_trades=None,
                total_pairs_traded=None,
                total_costs=None,
                total_days=None,
                expected_test_days=None,
                observed_test_days=None,
                coverage_ratio=None,
                zero_pnl_days=None,
                zero_pnl_days_pct=None,
                missing_test_days=None,
                volatility=None,
                win_rate=None,
                best_pair_pnl=None,
                worst_pair_pnl=None,
                avg_pnl_per_pair=None,
                tail_loss_pair_total_abs=None,
                tail_loss_worst_pair="",
                tail_loss_worst_pair_pnl=None,
                tail_loss_worst_pair_share=None,
                tail_loss_period_total_abs=None,
                tail_loss_worst_period="",
                tail_loss_worst_period_pnl=None,
                tail_loss_worst_period_share=None,
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
        entry.expected_test_days = metrics.get("expected_test_days")
        entry.observed_test_days = metrics.get("observed_test_days")
        entry.coverage_ratio = metrics.get("coverage_ratio")
        entry.zero_pnl_days = metrics.get("zero_pnl_days")
        entry.zero_pnl_days_pct = metrics.get("zero_pnl_days_pct")
        entry.missing_test_days = metrics.get("missing_test_days")
        entry.volatility = metrics.get("volatility")
        entry.win_rate = metrics.get("win_rate")
        entry.best_pair_pnl = metrics.get("best_pair_pnl")
        entry.worst_pair_pnl = metrics.get("worst_pair_pnl")
        entry.avg_pnl_per_pair = metrics.get("avg_pnl_per_pair")
        tail_diag = _load_tail_loss_diagnostics(results_path)
        entry.tail_loss_pair_total_abs = tail_diag.get("tail_loss_pair_total_abs")
        entry.tail_loss_worst_pair = str(tail_diag.get("tail_loss_worst_pair") or "")
        entry.tail_loss_worst_pair_pnl = tail_diag.get("tail_loss_worst_pair_pnl")
        entry.tail_loss_worst_pair_share = tail_diag.get("tail_loss_worst_pair_share")
        entry.tail_loss_period_total_abs = tail_diag.get("tail_loss_period_total_abs")
        entry.tail_loss_worst_period = str(tail_diag.get("tail_loss_worst_period") or "")
        entry.tail_loss_worst_period_pnl = tail_diag.get("tail_loss_worst_period_pnl")
        entry.tail_loss_worst_period_share = tail_diag.get("tail_loss_worst_period_share")

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

    config_cache: Dict[str, Dict[str, Any]] = {}
    universe_cache: Dict[str, Dict[str, Any]] = {}
    for entry in entries.values():
        config_payload: Dict[str, Any] = {}
        if entry.config_path:
            config_abs = _resolve_repo_path(entry.config_path, project_root)
            if config_abs is not None:
                key = config_abs.as_posix()
                config_payload = config_cache.get(key) or {}
                if not config_payload:
                    config_payload = _load_yaml_dict(config_abs)
                    config_cache[key] = config_payload

        context = _extract_universe_context(
            config_path=entry.config_path,
            project_root=project_root,
            config_cache=config_cache,
            universe_cache=universe_cache,
        )
        entry.universe_path = str(context.get("universe_path") or "")
        entry.universe_tag = str(context.get("universe_tag") or "")
        entry.universe_pairs_count = _to_int(context.get("universe_pairs_count"))
        denylist_count = _to_int(context.get("denylist_count"))
        entry.denylist_count = int(denylist_count or 0)
        entry.denylist_hash = str(context.get("denylist_hash") or "")

        # Coverage metrics (fail-closed): prefer values from strategy_metrics.csv, but when absent
        # (legacy runs) compute from daily_pnl.csv + config walk_forward dates.
        if entry.metrics_present and (entry.coverage_ratio is None or not math.isfinite(float(entry.coverage_ratio))):
            walk_forward = config_payload.get("walk_forward")
            if isinstance(walk_forward, dict):
                wf_start = str(walk_forward.get("start_date") or "").strip()
                wf_end = str(walk_forward.get("end_date") or "").strip()
                if wf_start and wf_end:
                    results_abs = _resolve_repo_path(entry.results_dir, project_root)
                    if results_abs is not None:
                        cov = _coverage_from_daily_pnl_csv(
                            results_abs / "daily_pnl.csv",
                            start_date=wf_start,
                            end_date=wf_end,
                        )
                        if cov:
                            entry.expected_test_days = cov.get("expected_test_days")
                            entry.observed_test_days = cov.get("observed_test_days")
                            entry.coverage_ratio = cov.get("coverage_ratio")
                            entry.zero_pnl_days = cov.get("zero_pnl_days")
                            entry.zero_pnl_days_pct = cov.get("zero_pnl_days_pct")
                            entry.missing_test_days = cov.get("missing_test_days")

    # Statistical confidence metrics for Sharpe reliability.
    # trials ~= number of tested configs with metrics in the same run_group.
    trials_by_group: Dict[str, int] = {}
    total_trials = 0
    for entry in entries.values():
        if not entry.metrics_present:
            continue
        total_trials += 1
        key = str(entry.run_group or "")
        trials_by_group[key] = int(trials_by_group.get(key, 0) + 1)
    default_trials = total_trials if total_trials >= 2 else None

    for entry in entries.values():
        if not entry.metrics_present:
            continue
        group_key = str(entry.run_group or "")
        group_trials = trials_by_group.get(group_key, 0)
        trials = group_trials if group_trials >= 2 else default_trials
        entry.dsr_trials = int(trials) if trials is not None else None

        results_abs = _resolve_repo_path(entry.results_dir, project_root)
        if results_abs is None:
            continue
        returns = _load_equity_returns(results_abs)
        if len(returns) < 2:
            returns = _load_daily_pnl_as_returns(results_abs)
        entry.psr, entry.dsr = _safe_psr_dsr(returns=returns, trials=entry.dsr_trials)

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
        "universe_path",
        "universe_tag",
        "universe_pairs_count",
        "denylist_count",
        "denylist_hash",
        "status",
        "metrics_present",
        "sharpe_ratio_abs",
        "sharpe_ratio_abs_raw",
        "sharpe_ratio_on_returns",
        "psr",
        "dsr",
        "dsr_trials",
        "total_pnl",
        "max_drawdown_abs",
        "max_drawdown_on_equity",
        "total_trades",
        "total_pairs_traded",
        "total_costs",
        "total_days",
        "expected_test_days",
        "observed_test_days",
        "coverage_ratio",
        "zero_pnl_days",
        "zero_pnl_days_pct",
        "missing_test_days",
        "volatility",
        "win_rate",
        "best_pair_pnl",
        "worst_pair_pnl",
        "avg_pnl_per_pair",
        "tail_loss_pair_total_abs",
        "tail_loss_worst_pair",
        "tail_loss_worst_pair_pnl",
        "tail_loss_worst_pair_share",
        "tail_loss_period_total_abs",
        "tail_loss_worst_period",
        "tail_loss_worst_period_pnl",
        "tail_loss_worst_period_share",
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
