#!/usr/bin/env python3
"""
Rebuild a top-N robust ranking from the rollup run_index.csv.

Definition (matches scripts/optimization/rank_multiwindow_robust_runs.py):
  robust_sharpe_window = min(holdout_sharpe, stress_sharpe)
  dd_pct_window        = max(abs(holdout_dd_pct), abs(stress_dd_pct))

Across OOS windows for the same variant:
  worst_robust_sharpe = min_window(robust_sharpe_window)
  worst_dd_pct        = max_window(dd_pct_window)

This script is meant for auditing:
  - It uses the canonical Sharpe already present in run_index.csv: sharpe_ratio_abs
    (computed from equity_curve.csv by the rollup builder when possible).
  - It prints top-N rows with paths to YAML + metrics for the worst robust window.
  - It flags duplicate parameter sets among the top-N (hash over YAML with
    window-specific keys stripped).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "coint4"

_OOS_RE = re.compile(r"_oos(\d{8})_(\d{8})")


@dataclass(frozen=True)
class Entry:
    run_id: str
    run_group: str
    results_dir: str
    metrics_path: str
    config_path: str
    status: str
    metrics_present: bool
    sharpe: Optional[float]
    dd_pct: Optional[float]
    pnl: Optional[float]
    trades: Optional[float]
    pairs: Optional[float]
    costs: Optional[float]


@dataclass(frozen=True)
class WindowRow:
    window: str
    robust_sharpe: float
    dd_pct: float
    holdout: Entry
    stress: Entry


@dataclass(frozen=True)
class VariantRow:
    run_group: str
    variant_id: str
    windows: List[WindowRow]

    worst_robust_sharpe: float
    avg_robust_sharpe: float
    worst_dd_pct: float
    avg_dd_pct: float

    worst_window: WindowRow
    worst_dd_window: WindowRow

    sample_config_path: str
    sample_config_repo_path: str
    params_sig: str


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _to_float(value: str) -> Optional[float]:
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _kind_and_base_id(run_id: str) -> Tuple[Optional[str], str]:
    if run_id.startswith("holdout_"):
        return "holdout", run_id[len("holdout_") :]
    if run_id.startswith("stress_"):
        return "stress", run_id[len("stress_") :]
    return None, run_id


def _matches_all(text: str, needles: Iterable[str]) -> bool:
    hay = text.lower()
    for needle in needles:
        if needle.lower() not in hay:
            return False
    return True


def _parse_window(base_id: str) -> str:
    m = _OOS_RE.search(base_id)
    if not m:
        return "-"
    return f"{m.group(1)}-{m.group(2)}"


def _variant_id(base_id: str) -> str:
    return _OOS_RE.sub("", base_id)


def _repo_rel_app_path(app_rel: str) -> str:
    """Convert an app-root-relative path to a repo-root-relative one."""
    p = (app_rel or "").strip()
    if not p:
        return ""
    if p.startswith("coint4/"):
        return p
    return f"coint4/{p}"


def _resolve_config_path(config_path: str) -> Optional[Path]:
    p = (config_path or "").strip()
    if not p:
        return None
    path = Path(p)
    if path.is_absolute():
        return path if path.exists() else None
    # Most configs in run_index.csv are relative to APP_ROOT (coint4/).
    cand1 = APP_ROOT / path
    if cand1.exists():
        return cand1
    cand2 = PROJECT_ROOT / path
    if cand2.exists():
        return cand2
    return None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonicalize_for_param_sig(cfg: object) -> object:
    """
    Strip window/run-output specific keys so that configs that only differ by
    WF date-range or logging don't look different.
    """
    if not isinstance(cfg, dict):
        return cfg

    out = {}
    for k, v in cfg.items():
        if k in {"results_dir", "logging"}:
            continue
        if k == "walk_forward" and isinstance(v, dict):
            wf = {wk: wv for wk, wv in v.items() if wk not in {"start_date", "end_date"}}
            out[k] = _canonicalize_for_param_sig(wf)
            continue
        out[k] = _canonicalize_for_param_sig(v)
    return out


def _config_param_sig(config_path: str) -> Tuple[str, str]:
    """
    Returns:
      - repo_rel_path (string, possibly empty)
      - params_sig (short sha)
    """
    cfg_path = _resolve_config_path(config_path)
    if cfg_path is None:
        return "", "NA"
    repo_rel = cfg_path.resolve()
    try:
        repo_rel = repo_rel.relative_to(PROJECT_ROOT.resolve())
        repo_rel_s = repo_rel.as_posix()
    except ValueError:
        repo_rel_s = cfg_path.resolve().as_posix()

    try:
        raw = cfg_path.read_bytes()
    except OSError:
        return repo_rel_s, "NA"

    try:
        cfg_obj = yaml.safe_load(raw)
    except Exception:
        # Fall back to raw hash if YAML is not parseable.
        return repo_rel_s, _sha256_bytes(raw)[:12]

    canon = _canonicalize_for_param_sig(cfg_obj)
    canon_json = json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return repo_rel_s, _sha256_bytes(canon_json)[:12]


def _fmt(value: Optional[float], *, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit top-N robust variants from rollup run_index.csv")
    parser.add_argument(
        "--run-index",
        default="coint4/artifacts/wfa/aggregate/rollup/run_index.csv",
        help="Path to rollup run_index.csv (repo-relative).",
    )
    parser.add_argument(
        "--contains",
        action="append",
        default=[],
        help="Substring filter (repeatable). All must match combined run metadata.",
    )
    parser.add_argument("--top", type=int, default=20, help="How many variants to print.")
    parser.add_argument(
        "--include-noncompleted",
        action="store_true",
        help="Include variants where at least one window is not 'completed'.",
    )
    parser.add_argument("--min-windows", type=int, default=3, help="Require at least this many windows per variant.")
    parser.add_argument("--min-trades", type=int, default=200, help="Require min(total_trades) across windows >= this.")
    parser.add_argument("--min-pairs", type=int, default=20, help="Require min(total_pairs_traded) across windows >= this.")
    parser.add_argument(
        "--max-dd-pct",
        type=float,
        default=0.15,
        help="Gate: max window drawdown on equity (abs) must be <= this (default: 0.15 = 15%%).",
    )
    args = parser.parse_args()

    run_index_path = (PROJECT_ROOT / args.run_index).resolve() if not Path(args.run_index).is_absolute() else Path(args.run_index).resolve()
    if not run_index_path.exists():
        raise SystemExit(f"run index not found: {run_index_path}")

    entries: List[Entry] = []
    with run_index_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            entries.append(
                Entry(
                    run_id=(row.get("run_id") or "").strip(),
                    run_group=(row.get("run_group") or "").strip(),
                    results_dir=(row.get("results_dir") or "").strip(),
                    metrics_path=(row.get("metrics_path") or "").strip(),
                    config_path=(row.get("config_path") or "").strip(),
                    status=(row.get("status") or "").strip(),
                    metrics_present=_to_bool(row.get("metrics_present") or ""),
                    sharpe=_to_float(row.get("sharpe_ratio_abs") or ""),
                    dd_pct=_to_float(row.get("max_drawdown_on_equity") or ""),
                    pnl=_to_float(row.get("total_pnl") or ""),
                    trades=_to_float(row.get("total_trades") or ""),
                    pairs=_to_float(row.get("total_pairs_traded") or ""),
                    costs=_to_float(row.get("total_costs") or ""),
                )
            )

    # Step 1: pair holdout/stress within each run_group by base_id.
    paired: Dict[Tuple[str, str], Dict[str, Entry]] = {}
    for e in entries:
        kind, base_id = _kind_and_base_id(e.run_id)
        if kind not in {"holdout", "stress"}:
            continue

        meta = " | ".join([e.run_group, base_id, e.run_id, e.config_path, e.metrics_path, e.results_dir, e.status])
        if args.contains and not _matches_all(meta, args.contains):
            continue

        key = (e.run_group, base_id)
        slot = paired.setdefault(key, {})
        slot[kind] = e

    # Step 2: compute per-window robust metrics and aggregate by variant.
    windows_by_variant: Dict[Tuple[str, str], List[WindowRow]] = {}
    for (run_group, base_id), pair in paired.items():
        h = pair.get("holdout")
        s = pair.get("stress")
        if h is None or s is None:
            continue
        if not (h.metrics_present and s.metrics_present):
            continue
        if h.sharpe is None or s.sharpe is None:
            continue
        if h.dd_pct is None or s.dd_pct is None:
            continue
        if not args.include_noncompleted:
            if h.status.lower() != "completed" or s.status.lower() != "completed":
                continue

        robust_sharpe = min(h.sharpe, s.sharpe)
        dd_pct = max(abs(h.dd_pct), abs(s.dd_pct))
        window = _parse_window(base_id)
        variant = _variant_id(base_id)
        windows_by_variant.setdefault((run_group, variant), []).append(
            WindowRow(window=window, robust_sharpe=robust_sharpe, dd_pct=dd_pct, holdout=h, stress=s)
        )

    variants: List[VariantRow] = []
    for (run_group, variant_id), items in windows_by_variant.items():
        if len(items) < max(1, args.min_windows):
            continue

        # Single pass across windows (avoid repeated min/max/sum passes).
        n = 0
        worst_dd = 0.0
        worst_robust = float("inf")
        sum_robust = 0.0
        sum_dd = 0.0
        min_trades = float("inf")
        min_pairs = float("inf")
        for it in items:
            n += 1
            if it.dd_pct > worst_dd:
                worst_dd = it.dd_pct
            if it.robust_sharpe < worst_robust:
                worst_robust = it.robust_sharpe
            sum_robust += it.robust_sharpe
            sum_dd += it.dd_pct
            trades = it.holdout.trades or 0.0
            pairs = it.holdout.pairs or 0.0
            if trades < min_trades:
                min_trades = trades
            if pairs < min_pairs:
                min_pairs = pairs

        if args.max_dd_pct is not None and worst_dd > max(0.0, args.max_dd_pct):
            continue
        if args.min_trades is not None and min_trades < args.min_trades:
            continue
        if args.min_pairs is not None and min_pairs < args.min_pairs:
            continue

        avg_robust = sum_robust / n
        avg_dd = sum_dd / n

        # Worst robust window: lowest robust_sharpe; tie-break by larger dd, then window.
        worst_window = min(items, key=lambda it: (it.robust_sharpe, -it.dd_pct, it.window))
        # Worst DD window: largest dd_pct; tie-break by lower robust_sharpe, then window.
        worst_dd_window = min(items, key=lambda it: (-it.dd_pct, it.robust_sharpe, it.window))

        sample_cfg = worst_window.holdout.config_path or worst_window.stress.config_path
        # NOTE: we defer YAML parsing/hash until after ranking (top-N only).

        variants.append(
            VariantRow(
                run_group=run_group,
                variant_id=variant_id,
                windows=sorted(items, key=lambda it: it.window),
                worst_robust_sharpe=worst_robust,
                avg_robust_sharpe=avg_robust,
                worst_dd_pct=worst_dd,
                avg_dd_pct=avg_dd,
                worst_window=worst_window,
                worst_dd_window=worst_dd_window,
                sample_config_path=sample_cfg,
                sample_config_repo_path="",
                params_sig="NA",
            )
        )

    sort_key = lambda v: (v.worst_robust_sharpe, v.avg_robust_sharpe, -v.worst_dd_pct)
    top_n = max(1, args.top)
    top = heapq.nlargest(top_n, variants, key=sort_key)
    top.sort(key=sort_key, reverse=True)
    # Enrich top with params_sig only after ranking (I/O heavy).
    top_enriched: List[VariantRow] = []
    for v in top:
        cfg_repo_path, params_sig = _config_param_sig(v.sample_config_path)
        top_enriched.append(
            VariantRow(
                run_group=v.run_group,
                variant_id=v.variant_id,
                windows=v.windows,
                worst_robust_sharpe=v.worst_robust_sharpe,
                avg_robust_sharpe=v.avg_robust_sharpe,
                worst_dd_pct=v.worst_dd_pct,
                avg_dd_pct=v.avg_dd_pct,
                worst_window=v.worst_window,
                worst_dd_window=v.worst_dd_window,
                sample_config_path=v.sample_config_path,
                sample_config_repo_path=cfg_repo_path,
                params_sig=params_sig,
            )
        )

    if not top_enriched:
        print("No variants matched (check filters/gates or ensure rollup index is up to date).")
        return 1

    print(f"run_index: {run_index_path.as_posix()}")
    print(f"paired_windows: {sum(len(v.windows) for v in variants)} (variants_after_gates={len(variants)})")
    print("")

    print(
        "| rank | worst_robust_sh | avg_robust_sh | worst_dd_pct | windows | params_sig | run_group | variant_id | sample_config | worst_window | holdout_metrics | stress_metrics |"
    )
    print("|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|")
    for idx, v in enumerate(top_enriched, 1):
        w = v.worst_window
        hold_metrics = _repo_rel_app_path(w.holdout.metrics_path)
        stress_metrics = _repo_rel_app_path(w.stress.metrics_path)
        sample_cfg = v.sample_config_repo_path or _repo_rel_app_path(v.sample_config_path)
        print(
            "| {rank} | {worst} | {avg} | {wdd} | {n} | {sig} | {group} | {variant} | {cfg} | {win} | {hm} | {sm} |".format(
                rank=idx,
                worst=_fmt(v.worst_robust_sharpe, digits=3),
                avg=_fmt(v.avg_robust_sharpe, digits=3),
                wdd=_fmt(v.worst_dd_pct, digits=3),
                n=len(v.windows),
                sig=v.params_sig,
                group=v.run_group,
                variant=v.variant_id,
                cfg=sample_cfg or "-",
                win=w.window,
                hm=hold_metrics or "-",
                sm=stress_metrics or "-",
            )
        )

    # Duplicate param signatures within the top-N.
    sig_map: Dict[str, List[Tuple[int, VariantRow]]] = {}
    for idx, v in enumerate(top_enriched, 1):
        sig_map.setdefault(v.params_sig, []).append((idx, v))

    dups = [(sig, items) for sig, items in sig_map.items() if sig != "NA" and len(items) > 1]
    if dups:
        print("")
        print("duplicate_params_sig_in_top:")
        for sig, items in sorted(dups, key=lambda x: (-len(x[1]), x[0])):
            # Keep this compact but actionable: ranks + config paths.
            ranks = ",".join(str(it[0]) for it in items)
            cfgs = ", ".join((it[1].sample_config_repo_path or _repo_rel_app_path(it[1].sample_config_path) or "-") for it in items)
            print(f"- params_sig={sig} ranks=[{ranks}] configs=[{cfgs}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
