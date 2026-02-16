#!/usr/bin/env python3
"""Budget $1000 optimization autopilot (VPS WFA -> canonical metrics -> robust ranking).

Goal: reduce manual loop "run -> wait -> ask Codex -> generate next queue".
This tool performs a coordinate-ascent style micro-sweep over a list of knobs:
  - generate multi-window holdout+stress queue (small)
  - run it on VPS via scripts/remote/run_server_job.sh (STOP_AFTER=1)
  - post-process locally (sync statuses, canonical_metrics.json, rollup run_index)
  - select best by worst-window robust Sharpe with DD + sanity gates
  - stop when there is no sufficient improvement for N rounds in a row

Outputs (canonical locations; see AGENTS.md):
  - queues (tracked):     coint4/artifacts/wfa/aggregate/<run_group>/run_queue.csv
  - configs (tracked):    coint4/configs/budget1000_autopilot/<run_group>/*.yaml
  - heavy artifacts:      coint4/artifacts/wfa/runs_clean/<run_group>/**   (do NOT commit)
  - controller state:     coint4/artifacts/wfa/aggregate/<controller>/state.json
  - final report (docs):  docs/budget1000_autopilot_final_YYYYMMDD.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


_OOS_RE = re.compile(r"_oos(\d{8})_(\d{8})")


STRESS_OVERRIDES: Dict[str, Any] = {
    "backtest.commission_pct": 0.0006,
    "backtest.commission_rate_per_leg": 0.0006,
    "backtest.slippage_pct": 0.001,
    "backtest.slippage_stress_multiplier": 2.0,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_app_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_repo_root(app_root: Path) -> Path:
    # Repo root is parent of coint4/ (app-root dir).
    return app_root.parent


def _resolve_under_root(path: str, *, root: Path) -> Path:
    raw = str(path or "").strip()
    if not raw:
        raise ValueError("empty path")
    p = Path(raw)
    if p.is_absolute():
        return p
    return root / raw


def _venv_python(app_root: Path) -> Path:
    candidate = app_root / ".venv" / "bin" / "python"
    if candidate.exists():
        return candidate
    candidate = app_root / ".venv" / "bin" / "python3"
    if candidate.exists():
        return candidate
    return Path(sys.executable)


def _run(cmd: List[str], *, cwd: Path, env: dict, check: bool = True) -> int:
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return int(proc.returncode)


def _git(cmd: List[str], *, repo_root: Path, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *cmd],
        cwd=str(repo_root),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or f"git {' '.join(cmd)} failed rc={proc.returncode}")
    return proc.stdout


def _unique(items: Iterable[Any]) -> List[Any]:
    seen: set[Any] = set()
    out: List[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def set_nested(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node: Dict[str, Any] = cfg
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def get_nested(cfg: Dict[str, Any], dotted_key: str) -> Any:
    parts = dotted_key.split(".")
    node: Any = cfg
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted_key)
        node = node[part]
    return node


def encode_value(value: Any) -> str:
    if isinstance(value, bool):
        return "T" if value else "F"
    if isinstance(value, float):
        return str(value).replace(".", "p").replace("-", "m")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        if re.match(r"\d{4}-\d{2}-\d{2}", value):
            return value.replace("-", "")
        return value.replace(".", "p").replace("-", "m")
    if value is None:
        return "null"
    return str(value).replace(".", "p").replace("-", "m")


def short_key_for_tag(dotted_key: str) -> str:
    short_keys = {
        "walk_forward.start_date": "oos",
        "walk_forward.end_date": "to",
        "walk_forward.max_steps": "ms",
        "portfolio.risk_per_position_pct": "risk",
        "portfolio.max_active_positions": "maxpos",
        "backtest.zscore_entry_threshold": "z",
        "backtest.zscore_exit": "exit",
        "backtest.min_spread_move_sigma": "ms",
        "backtest.pair_stop_loss_usd": "slusd",
        "backtest.pair_stop_loss_zscore": "slz",
        "backtest.stop_loss_multiplier": "slm",
        "backtest.time_stop_multiplier": "ts",
        "backtest.portfolio_daily_stop_pct": "dstop",
        "backtest.max_var_multiplier": "vm",
        "filter_params.min_beta": "min_beta",
        "pair_selection.min_correlation": "corr",
        "pair_selection.coint_pvalue_threshold": "pv",
    }
    return short_keys.get(dotted_key, dotted_key.split(".")[-1])


def make_tag(dotted_key: str, value: Any) -> str:
    return f"{short_key_for_tag(dotted_key)}{encode_value(value)}"


def _variant_id(base_id: str) -> str:
    return _OOS_RE.sub("", base_id)


def _parse_window(base_id: str) -> str:
    m = _OOS_RE.search(base_id)
    if not m:
        return "-"
    return f"{m.group(1)}-{m.group(2)}"


def _kind_and_base_id(run_id: str) -> Tuple[Optional[str], str]:
    if run_id.startswith("holdout_"):
        return "holdout", run_id[len("holdout_") :]
    if run_id.startswith("stress_"):
        return "stress", run_id[len("stress_") :]
    return None, run_id


@dataclass(frozen=True)
class _RunIndexRow:
    run_id: str
    run_group: str
    config_path: str
    results_dir: str
    status: str
    metrics_present: bool
    sharpe: Optional[float]
    dd_pct: Optional[float]
    trades: Optional[float]
    pairs: Optional[float]


@dataclass(frozen=True)
class SelectionResult:
    run_group: str
    variant_id: str
    score: float
    worst_robust_sharpe: float
    avg_robust_sharpe: float
    worst_dd_pct: float
    avg_dd_pct: float
    windows: int
    sample_config_path: str


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_run_index(run_index_path: Path) -> List[_RunIndexRow]:
    import csv

    rows: List[_RunIndexRow] = []
    with run_index_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                _RunIndexRow(
                    run_id=(row.get("run_id") or "").strip(),
                    run_group=(row.get("run_group") or "").strip(),
                    config_path=(row.get("config_path") or "").strip(),
                    results_dir=(row.get("results_dir") or "").strip(),
                    status=(row.get("status") or "").strip(),
                    metrics_present=_to_bool(row.get("metrics_present") or ""),
                    sharpe=_to_float(row.get("sharpe_ratio_abs") or ""),
                    dd_pct=_to_float(row.get("max_drawdown_on_equity") or ""),
                    trades=_to_float(row.get("total_trades") or ""),
                    pairs=_to_float(row.get("total_pairs_traded") or ""),
                )
            )
    return rows


def select_best_multiwindow(
    *,
    run_index_path: Path,
    run_group: str,
    min_windows: int,
    min_trades: int,
    min_pairs: int,
    max_dd_pct: Optional[float],
    dd_target_pct: Optional[float],
    dd_penalty: float,
) -> SelectionResult:
    rows = _load_run_index(run_index_path)
    rows = [r for r in rows if r.run_group == run_group]

    paired: Dict[str, Dict[str, _RunIndexRow]] = {}
    for r in rows:
        kind, base_id = _kind_and_base_id(r.run_id)
        if kind not in {"holdout", "stress"}:
            continue
        if not r.metrics_present:
            continue
        if r.sharpe is None or r.dd_pct is None:
            continue
        if r.status.lower() != "completed":
            continue
        paired.setdefault(base_id, {})[kind] = r

    windows_by_variant: Dict[str, List[Tuple[str, float, float, _RunIndexRow]]] = {}
    for base_id, pair in paired.items():
        h = pair.get("holdout")
        s = pair.get("stress")
        if h is None or s is None:
            continue

        robust_sharpe = min(float(h.sharpe), float(s.sharpe))
        dd_pct = max(abs(float(h.dd_pct)), abs(float(s.dd_pct)))
        window = _parse_window(base_id)
        variant = _variant_id(base_id)
        windows_by_variant.setdefault(variant, []).append((window, robust_sharpe, dd_pct, h))

    candidates: List[SelectionResult] = []
    for variant, items in windows_by_variant.items():
        if len(items) < max(1, int(min_windows)):
            continue
        worst_dd = max(item[2] for item in items)
        if max_dd_pct is not None and worst_dd > max(0.0, float(max_dd_pct)):
            continue

        holdout_rows = [item[3] for item in items]
        min_tr = min(int(r.trades or 0.0) for r in holdout_rows)
        min_pr = min(int(r.pairs or 0.0) for r in holdout_rows)
        if min_tr < int(min_trades) or min_pr < int(min_pairs):
            continue

        worst_robust = min(item[1] for item in items)
        avg_robust = sum(item[1] for item in items) / len(items)
        avg_dd = sum(item[2] for item in items) / len(items)
        penalty = 0.0
        if dd_target_pct is not None:
            penalty = max(0.0, float(worst_dd) - float(dd_target_pct)) * max(0.0, float(dd_penalty))
        score = float(worst_robust) - float(penalty)
        sample_cfg = holdout_rows[0].config_path
        candidates.append(
            SelectionResult(
                run_group=run_group,
                variant_id=variant,
                score=score,
                worst_robust_sharpe=float(worst_robust),
                avg_robust_sharpe=float(avg_robust),
                worst_dd_pct=float(worst_dd),
                avg_dd_pct=float(avg_dd),
                windows=len(items),
                sample_config_path=sample_cfg,
            )
        )

    if not candidates:
        raise SystemExit(
            f"No candidates matched for run_group={run_group} (check gates/sanity or ensure rollup is rebuilt)."
        )

    candidates.sort(
        key=lambda c: (c.score, c.worst_robust_sharpe, c.avg_robust_sharpe, -c.worst_dd_pct),
        reverse=True,
    )
    return candidates[0]


def _dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _build_filename(base_name: str, sweep_tags: List[str], kind: str) -> str:
    clean = base_name
    for prefix in ("holdout_", "stress_"):
        if clean.startswith(prefix):
            clean = clean[len(prefix) :]
            break
    if clean.endswith(".yaml"):
        clean = clean[:-5]

    has_oos_sweep = any(t.startswith("oos") for t in sweep_tags)
    if has_oos_sweep:
        clean = re.sub(r"_oos\d{8}_\d{8}", "", clean)

    suffix = "_".join(sweep_tags) if sweep_tags else ""
    if suffix:
        return f"{kind}_{clean}_{suffix}"
    return f"{kind}_{clean}"


def _generate_sweep_queue(
    *,
    app_root: Path,
    run_group: str,
    base_config_path: Path,
    windows: List[Tuple[str, str]],
    knob_key: str,
    knob_values: List[Any],
    configs_dir_rel: str,
    queue_dir_rel: str,
    runs_dir_rel: str,
    seen_signatures: Optional[set[str]] = None,
) -> Path:
    sys.path.insert(0, str(app_root / "src"))
    from coint2.ops.run_queue import RunQueueEntry, write_run_queue  # type: ignore

    if not base_config_path.exists():
        raise SystemExit(f"Base config not found: {base_config_path}")

    base_cfg = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    if not isinstance(base_cfg, dict):
        raise SystemExit(f"Invalid YAML (expected dict): {base_config_path}")

    configs_dir = app_root / configs_dir_rel / run_group
    queue_dir = app_root / queue_dir_rel / run_group

    entries: List[RunQueueEntry] = []
    seen = set(seen_signatures or set())

    base_name = base_config_path.name
    for start_date, end_date in windows:
        # OOS tag in the same convention as generate_configs.py
        oos_tag = f"oos{encode_value(start_date)}_{encode_value(end_date)}"
        for value in knob_values:
            sweep_tags = [oos_tag, make_tag(knob_key, value)]

            holdout_cfg = json.loads(json.dumps(base_cfg))  # cheap deep copy (dict-only)
            set_nested(holdout_cfg, "walk_forward.start_date", start_date)
            set_nested(holdout_cfg, "walk_forward.end_date", end_date)
            set_nested(holdout_cfg, knob_key, value)
            if not _is_valid_config_combo(holdout_cfg):
                continue
            holdout_sig = _config_signature(holdout_cfg)
            if holdout_sig in seen:
                continue

            holdout_name = _build_filename(base_name, sweep_tags, "holdout")
            holdout_yaml = configs_dir / f"{holdout_name}.yaml"
            holdout_results_dir = f"{runs_dir_rel}/{run_group}/{holdout_name}"

            stress_cfg = json.loads(json.dumps(holdout_cfg))
            for skey, sval in STRESS_OVERRIDES.items():
                set_nested(stress_cfg, skey, sval)
            if not _is_valid_config_combo(stress_cfg):
                continue
            stress_sig = _config_signature(stress_cfg)
            if stress_sig in seen:
                continue
            stress_name = _build_filename(base_name, sweep_tags, "stress")
            stress_yaml = configs_dir / f"{stress_name}.yaml"
            stress_results_dir = f"{runs_dir_rel}/{run_group}/{stress_name}"

            configs_dir.mkdir(parents=True, exist_ok=True)
            holdout_yaml.write_text(yaml.dump(holdout_cfg, default_flow_style=False, allow_unicode=True), encoding="utf-8")
            stress_yaml.write_text(yaml.dump(stress_cfg, default_flow_style=False, allow_unicode=True), encoding="utf-8")

            entries.append(
                RunQueueEntry(
                    config_path=str(holdout_yaml.relative_to(app_root)),
                    results_dir=holdout_results_dir,
                    status="planned",
                )
            )
            entries.append(
                RunQueueEntry(
                    config_path=str(stress_yaml.relative_to(app_root)),
                    results_dir=stress_results_dir,
                    status="planned",
                )
            )

            seen.add(holdout_sig)
            seen.add(stress_sig)

    queue_path = queue_dir / "run_queue.csv"
    if not entries:
        raise ValueError(
            f"Queue generation produced 0 entries for run_group={run_group} "
            "(all candidates were duplicates or invalid combinations)."
        )
    write_run_queue(queue_path, entries)
    return queue_path


def _queue_is_fully_completed(queue_path: Path, *, app_root: Path) -> bool:
    """Return True if every queue entry is completed and has local metrics present.

    This allows safe re-runs/resumes without re-spending VPS time when results are already synced back.
    """
    sys.path.insert(0, str(app_root / "src"))
    from coint2.ops.run_queue import load_run_queue  # type: ignore

    entries = load_run_queue(queue_path)
    if not entries:
        return False

    for entry in entries:
        if (entry.status or "").strip().lower() != "completed":
            return False
        results_dir = (entry.results_dir or "").strip()
        if not results_dir:
            return False
        results_path = _resolve_under_root(results_dir, root=app_root)
        if not (results_path / "strategy_metrics.csv").exists():
            return False
    return True


def _candidate_values(
    *, current: float, op: str, step: float, candidates: List[int], min_value: Optional[float], max_value: Optional[float]
) -> List[float]:
    values: List[float] = []
    for i in candidates:
        if op == "add":
            v = float(current) + float(i) * float(step)
        elif op == "mul":
            v = float(current) * (1.0 + float(i) * float(step))
        else:
            raise ValueError(f"Unsupported op: {op!r} (expected 'add' or 'mul')")
        if min_value is not None:
            v = max(float(min_value), v)
        if max_value is not None:
            v = min(float(max_value), v)
        values.append(v)
    # Stable de-dup with rounding to avoid float jitter in filenames.
    rounded = [round(v, 10) for v in values]
    return _unique(rounded)


def _safe_knob_index(raw: Any, *, knobs_count: int, default: int = 0) -> int:
    if knobs_count <= 0:
        return 0
    try:
        idx = int(raw)
    except (TypeError, ValueError):
        idx = int(default)
    if idx < 0 or idx >= knobs_count:
        return max(0, min(int(default), knobs_count - 1))
    return idx


def _next_alternative_knob_index(*, knobs_count: int, exclude_index: Optional[int], start_index: int = 0) -> int:
    if knobs_count <= 0:
        return 0
    if knobs_count == 1:
        return 0
    start = int(start_index) % knobs_count
    for shift in range(knobs_count):
        idx = (start + shift) % knobs_count
        if exclude_index is not None and idx == int(exclude_index):
            continue
        return idx
    return 0


def _branch_candidate_offsets(*, candidates: List[int], branch: str) -> List[int]:
    base = sorted(_unique(int(x) for x in candidates))
    if not base:
        return []
    if branch != "local_refine":
        return base
    if len(base) <= 3:
        return base
    center_idx = min(range(len(base)), key=lambda i: abs(base[i]))
    left_idx = max(0, center_idx - 1)
    right_idx = min(len(base) - 1, center_idx + 1)
    return _unique([base[left_idx], base[center_idx], base[right_idx]])


def _config_signature(cfg: Dict[str, Any]) -> str:
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _get_optional_float(cfg: Dict[str, Any], dotted_key: str) -> Optional[float]:
    try:
        value = get_nested(cfg, dotted_key)
    except KeyError:
        return None
    return _to_float(value)


def _is_valid_config_combo(cfg: Dict[str, Any]) -> bool:
    try:
        start = str(get_nested(cfg, "walk_forward.start_date"))
        end = str(get_nested(cfg, "walk_forward.end_date"))
        if start > end:
            return False
    except KeyError:
        pass

    risk = _get_optional_float(cfg, "portfolio.risk_per_position_pct")
    if risk is not None and not (0.0 < float(risk) <= 1.0):
        return False

    max_var = _get_optional_float(cfg, "backtest.max_var_multiplier")
    if max_var is not None and float(max_var) < 1.0:
        return False

    stop_usd = _get_optional_float(cfg, "backtest.pair_stop_loss_usd")
    if stop_usd is not None and float(stop_usd) <= 0.0:
        return False

    daily_stop = _get_optional_float(cfg, "backtest.portfolio_daily_stop_pct")
    if daily_stop is not None and not (0.0 < float(daily_stop) < 1.0):
        return False

    z_entry = _get_optional_float(cfg, "backtest.zscore_entry_threshold")
    if z_entry is not None and float(z_entry) <= 0.0:
        return False
    z_exit = _get_optional_float(cfg, "backtest.zscore_exit")
    if z_exit is not None and float(z_exit) < 0.0:
        return False
    if z_entry is not None and z_exit is not None and abs(float(z_exit)) > abs(float(z_entry)):
        return False

    return True


def _collect_seen_config_signatures(
    *,
    app_root: Path,
    queue_dir_rel: str,
    run_group_prefix: str,
    exclude_run_group: Optional[str] = None,
) -> set[str]:
    queue_root = app_root / queue_dir_rel
    if not queue_root.exists():
        return set()

    seen: set[str] = set()
    pattern = f"{run_group_prefix}_r*_*"
    for queue_path in sorted(queue_root.glob(f"{pattern}/run_queue.csv")):
        run_group = queue_path.parent.name
        if exclude_run_group and run_group == exclude_run_group:
            continue
        for cfg_rel in _queue_config_paths(queue_path):
            cfg_path = _resolve_under_root(cfg_rel, root=app_root)
            if not cfg_path.exists():
                continue
            try:
                payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict):
                seen.add(_config_signature(payload))
    return seen


def _ensure_tracked_for_sync_up(
    *,
    repo_root: Path,
    paths_rel_repo: List[Path],
    auto_stage: bool,
) -> None:
    if not paths_rel_repo:
        return

    missing: List[Path] = []
    ignored_blocking: List[Path] = []
    ignored_controller_state: List[Path] = []

    def _is_ignored_by_git(path_rel_repo: Path) -> bool:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", "--", str(path_rel_repo)],
            cwd=str(repo_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return proc.returncode == 0

    def _is_controller_state(path_rel_repo: Path) -> bool:
        p = path_rel_repo.as_posix().lstrip("/")
        return p.startswith("coint4/artifacts/wfa/aggregate/") and p.endswith("/state.json")

    for p in paths_rel_repo:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(p)],
            cwd=str(repo_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if tracked.returncode == 0:
            continue
        if _is_ignored_by_git(p):
            if _is_controller_state(p):
                ignored_controller_state.append(p)
            else:
                ignored_blocking.append(p)
            continue
        missing.append(p)

    if ignored_blocking:
        formatted = "\n".join(f"- {p}" for p in ignored_blocking[:50])
        raise SystemExit(
            "SYNC_UP=1 cannot auto-stage ignored files. Remove them from tracked list or update .gitignore. Ignored:\n"
            + formatted
            + ("\n- ... (more)" if len(ignored_blocking) > 50 else "")
        )

    if ignored_controller_state:
        formatted = "\n".join(f"- {p}" for p in ignored_controller_state[:20])
        print(
            "[autopilot] skip git-add for ignored controller state (local-only):\n"
            + formatted
            + ("\n- ... (more)" if len(ignored_controller_state) > 20 else ""),
            file=sys.stderr,
        )

    if missing and not auto_stage:
        formatted = "\n".join(f"- {p}" for p in missing[:50])
        raise SystemExit(
            "SYNC_UP=1 requires files to be tracked (git add / commit). Missing:\n"
            + formatted
            + ("\n- ... (more)" if len(missing) > 50 else "")
        )

    if missing:
        _git(["add", "--"] + [str(p) for p in missing], repo_root=repo_root, check=True)


def _queue_config_paths(queue_path: Path) -> List[str]:
    import csv

    out: List[str] = []
    with queue_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            p = (row.get("config_path") or "").strip()
            if p:
                out.append(p)
    return _unique(out)


def _run_vps_queue(
    *,
    repo_root: Path,
    run_group: str,
    queue_rel_app_root: str,
    sync_up: bool,
    stop_after: bool,
) -> None:
    script_path = repo_root / "coint4" / "scripts" / "remote" / "run_server_job.sh"
    if not script_path.exists():
        raise SystemExit(f"Missing remote runner: {script_path}")

    env = os.environ.copy()
    env["SYNC_UP"] = "1" if sync_up else "0"
    env["STOP_AFTER"] = "1" if stop_after else "0"
    env["SYNC_BACK"] = "1"
    # Fetch only what we need to avoid rsync writing into root-owned dirs.
    env["SYNC_PATHS"] = " ".join(
        [
            f"coint4/artifacts/wfa/runs_clean/{run_group}",
            f"coint4/artifacts/wfa/aggregate/{run_group}",
        ]
    )

    cmd = [
        "bash",
        str(script_path),
        "bash",
        "scripts/optimization/watch_wfa_queue.sh",
        "--queue",
        queue_rel_app_root,
    ]
    _run(cmd, cwd=repo_root, env=env, check=True)


def _load_state(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_state(state_path: Path, state: Dict[str, Any]) -> None:
    state["updated_at_utc"] = _utc_now_iso()
    _dump_json(state_path, state)


def _best_score(state: Dict[str, Any]) -> float:
    score = _to_float(state.get("best_score"))
    if score is not None:
        return float(score)
    best = state.get("current_best")
    if isinstance(best, dict):
        score = _to_float(best.get("score"))
        if score is not None:
            return float(score)
    return float("-inf")


def _normalize_stop_state(state: Dict[str, Any]) -> None:
    best_score = _best_score(state)
    state["best_score"] = None if best_score == float("-inf") else float(best_score)
    try:
        streak = int(state.get("no_improvement_streak") or 0)
    except (TypeError, ValueError):
        streak = 0
    state["no_improvement_streak"] = max(0, streak)
    reason = state.get("stop_reason")
    if reason is None:
        state["stop_reason"] = None
    else:
        text = str(reason).strip()
        state["stop_reason"] = text or None


def _compact_best(best: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(best, dict):
        return None
    return {
        "run_group": str(best.get("run_group") or ""),
        "variant_id": str(best.get("variant_id") or ""),
        "score": _to_float(best.get("score")),
        "worst_robust_sharpe": _to_float(best.get("worst_robust_sharpe")),
        "worst_dd_pct": _to_float(best.get("worst_dd_pct")),
        "sample_config_path": str(best.get("sample_config_path") or ""),
    }


def _metric_delta(*, before: Optional[Dict[str, Any]], after: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    b = _to_float(before.get(key))
    a = _to_float(after.get(key))
    if b is None or a is None:
        return None
    return float(a - b)


def _round_decision(
    *,
    current_round: int,
    max_rounds: int,
    stopped_by_no_improvement: bool,
    stop_reason: Optional[str],
) -> Dict[str, Any]:
    if stopped_by_no_improvement:
        return {
            "action": "stop",
            "reason": str(stop_reason or "no_improvement_streak_reached"),
            "next_round": None,
        }
    if int(current_round) >= int(max_rounds):
        return {
            "action": "stop",
            "reason": f"max_rounds_reached: max_rounds={int(max_rounds)}",
            "next_round": None,
        }
    return {
        "action": "continue",
        "reason": "continue_to_next_round",
        "next_round": int(current_round) + 1,
    }


def _format_metric(value: Any) -> str:
    v = _to_float(value)
    if v is None:
        return "n/a"
    return f"{float(v):.6f}"


def _render_best_compact(best: Optional[Dict[str, Any]]) -> str:
    if not isinstance(best, dict):
        return "n/a"
    run_group = str(best.get("run_group") or "")
    variant_id = str(best.get("variant_id") or "")
    score = _format_metric(best.get("score"))
    robust = _format_metric(best.get("worst_robust_sharpe"))
    dd = _format_metric(best.get("worst_dd_pct"))
    return (
        f"run_group=`{run_group}`, variant_id=`{variant_id}`, "
        f"score=`{score}`, worst_robust_sharpe=`{robust}`, worst_dd_pct=`{dd}`"
    )


def _render_round_analysis_md(payload: Dict[str, Any]) -> str:
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    lines: List[str] = []
    lines.append(f"# Round {int(payload.get('round') or 0):02d} analysis")
    lines.append("")
    lines.append(f"- generated_at_utc: `{payload.get('generated_at_utc') or ''}`")
    lines.append(f"- best_before_round: {_render_best_compact(payload.get('best_before_round'))}")
    lines.append(f"- best_after_round: {_render_best_compact(payload.get('best_after_round'))}")
    lines.append(f"- delta_score: `{_format_metric(payload.get('delta_score'))}`")
    lines.append(f"- delta_worst_robust_sharpe: `{_format_metric(payload.get('delta_worst_robust_sharpe'))}`")
    lines.append(f"- delta_worst_dd_pct: `{_format_metric(payload.get('delta_worst_dd_pct'))}`")
    lines.append(f"- decision: `{decision.get('action') or ''}`")
    lines.append(f"- decision_reason: `{decision.get('reason') or ''}`")
    lines.append(f"- next_round: `{decision.get('next_round')}`")
    return "\n".join(lines) + "\n"


def _write_round_analysis(
    *,
    controller_dir: Path,
    payload: Dict[str, Any],
) -> None:
    round_num = int(payload.get("round") or 0)
    out_dir = controller_dir / "round_analysis"
    base = out_dir / f"round_{round_num:02d}"
    _dump_json(base.with_suffix(".json"), payload)
    _write_text(base.with_suffix(".md"), _render_round_analysis_md(payload))


def _apply_no_improvement_round(
    *,
    state: Dict[str, Any],
    improved_in_round: bool,
    no_improvement_rounds: int,
    min_improvement: float,
) -> bool:
    if improved_in_round:
        state["no_improvement_streak"] = 0
        return False
    streak = int(state.get("no_improvement_streak") or 0) + 1
    state["no_improvement_streak"] = streak
    if streak < int(no_improvement_rounds):
        return False
    state["done"] = True
    state["stop_reason"] = (
        "no_improvement_streak_reached: "
        f"streak={streak}, rounds={int(no_improvement_rounds)}, min_improvement={float(min_improvement):.6g}"
    )
    return True


def _final_report_path(*, repo_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return repo_root / "docs" / f"budget1000_autopilot_final_{stamp}.md"


def _render_final_report(*, state: Dict[str, Any]) -> str:
    best = state.get("current_best") or {}
    history = state.get("history") or []
    lines: List[str] = []
    lines.append("# Budget $1000 autopilot: итог")
    lines.append("")
    lines.append(f"Generated at (UTC): {state.get('updated_at_utc') or _utc_now_iso()}")
    lines.append(f"stop_reason: `{state.get('stop_reason') or ''}`")
    lines.append(f"best_score: `{state.get('best_score')}`")
    lines.append(f"no_improvement_streak: `{state.get('no_improvement_streak')}`")
    lines.append("")
    lines.append("## Лучший найденный кандидат (stop-condition достигнут)")
    lines.append("")
    lines.append(f"- run_group: `{best.get('run_group','')}`")
    lines.append(f"- variant_id: `{best.get('variant_id','')}`")
    lines.append(f"- score: `{best.get('score','')}`")
    lines.append(f"- worst-window robust Sharpe: `{best.get('worst_robust_sharpe','')}`")
    lines.append(f"- worst-window DD pct: `{best.get('worst_dd_pct','')}`")
    lines.append(f"- sample config: `{best.get('sample_config_path','')}`")
    lines.append("")
    lines.append("## История шагов (суммарно)")
    lines.append("")
    if not history:
        lines.append("- (нет записей)")
    else:
        for item in history[-50:]:
            lines.append(
                "- {run_group} | knob={knob} | score={score:.3f} | improved={improved}".format(
                    run_group=item.get("run_group", ""),
                    knob=item.get("knob_key", ""),
                    score=float(item.get("best_score") or 0.0),
                    improved=bool(item.get("improved") or False),
                )
            )
        if len(history) > 50:
            lines.append(f"- ... ({len(history) - 50} older)")
    lines.append("")
    lines.append("## Примечания")
    lines.append("")
    lines.append("- База: multi-window worst-case robust Sharpe = min_window(min(holdout, stress)).")
    lines.append("- Score (если включён dd_penalty): score = worst_robust_sharpe - dd_penalty * max(0, worst_dd_pct - dd_target_pct).")
    lines.append("- Heavy execution выполняется только на VPS через run_server_job.sh (STOP_AFTER=1).")
    lines.append("- Heavy артефакты `coint4/artifacts/wfa/runs_clean/**` не коммитить.")
    lines.append("")
    return "\n".join(lines)


def _derive_next_queue_plan(
    *,
    next_round: int,
    knobs_count: int,
    previous_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if knobs_count <= 0:
        return {
            "round": int(next_round),
            "branch": "fallback_knob",
            "knob_index": 0,
            "reason": "no_knobs_configured",
        }

    prev = previous_result if isinstance(previous_result, dict) else {}
    prev_round = int(prev.get("round") or 0)
    prev_improved = bool(prev.get("improved") or False)
    prev_knob = _safe_knob_index(prev.get("knob_index"), knobs_count=knobs_count, default=0)

    if int(next_round) <= 1:
        return {
            "round": int(next_round),
            "branch": "fallback_knob",
            "knob_index": 0,
            "reason": "initial_round",
        }

    if prev_round == int(next_round) - 1 and prev_improved:
        return {
            "round": int(next_round),
            "branch": "local_refine",
            "knob_index": prev_knob,
            "reason": "winner_improved_previous_round",
        }

    alt_idx = _next_alternative_knob_index(
        knobs_count=knobs_count,
        exclude_index=prev_knob if prev_round == int(next_round) - 1 else None,
        start_index=prev_knob + 1 if prev_round == int(next_round) - 1 else 0,
    )
    return {
        "round": int(next_round),
        "branch": "fallback_knob",
        "knob_index": int(alt_idx),
        "reason": "no_improvement_or_no_winner_previous_round",
    }


def _normalize_queue_plan(
    *,
    raw_plan: Any,
    current_round: int,
    knobs_count: int,
    fallback_plan: Dict[str, Any],
) -> Dict[str, Any]:
    plan = raw_plan if isinstance(raw_plan, dict) else {}
    if int(plan.get("round") or -1) != int(current_round):
        plan = fallback_plan

    branch = str(plan.get("branch") or "").strip()
    if branch not in {"local_refine", "fallback_knob"}:
        branch = str(fallback_plan.get("branch") or "fallback_knob")

    knob_index = _safe_knob_index(
        plan.get("knob_index"),
        knobs_count=knobs_count,
        default=int(fallback_plan.get("knob_index") or 0),
    )
    reason = str(plan.get("reason") or fallback_plan.get("reason") or "").strip() or "unspecified"
    return {
        "round": int(current_round),
        "branch": branch,
        "knob_index": int(knob_index),
        "reason": reason,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Budget1000 autopilot: VPS WFA sweeps + robust selection")
    parser.add_argument(
        "--config",
        required=True,
        help="Autopilot YAML config (relative to app-root coint4/ unless absolute).",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from controller state if present.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset controller state (creates a timestamped .bak copy if state.json exists).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate next queue only; do not run VPS.")
    args = parser.parse_args(argv)

    app_root = _resolve_app_root()
    repo_root = _resolve_repo_root(app_root)
    py = _venv_python(app_root)

    config_path = _resolve_under_root(args.config, root=app_root)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise SystemExit(f"Invalid YAML config: {config_path}")

    base_config_rel = str(cfg.get("base_config") or "").strip()
    if not base_config_rel:
        raise SystemExit("config.base_config is required")

    run_group_prefix = str(cfg.get("run_group_prefix") or "").strip()
    if not run_group_prefix:
        raise SystemExit("config.run_group_prefix is required (e.g. 20260215_budget1000_ap)")

    controller_group = str(cfg.get("controller_group") or f"{run_group_prefix}_autopilot").strip()
    controller_dir = app_root / "artifacts" / "wfa" / "aggregate" / controller_group
    state_path = controller_dir / "state.json"

    selection_cfg = cfg.get("selection") or {}
    max_dd_pct_raw = selection_cfg.get("max_dd_pct", None)
    max_dd_pct = None if max_dd_pct_raw is None else float(max_dd_pct_raw)
    dd_target_pct_raw = selection_cfg.get("dd_target_pct", None)
    dd_target_pct = None if dd_target_pct_raw is None else float(dd_target_pct_raw)
    dd_penalty = float(selection_cfg.get("dd_penalty") or 0.0)
    min_windows = int(selection_cfg.get("min_windows") or 3)
    min_trades = int(selection_cfg.get("min_trades") or 10)
    min_pairs = int(selection_cfg.get("min_pairs") or 1)

    search_cfg = cfg.get("search") or {}
    max_rounds = int(search_cfg.get("max_rounds") or 3)
    min_improvement = float(search_cfg.get("min_improvement") or 0.02)
    no_improvement_rounds = int(search_cfg.get("no_improvement_rounds") or 1)
    if no_improvement_rounds < 1:
        raise SystemExit("config.search.no_improvement_rounds must be >= 1")
    knobs = search_cfg.get("knobs") or []
    if not isinstance(knobs, list) or not knobs:
        raise SystemExit("config.search.knobs must be a non-empty list")

    windows_cfg = cfg.get("windows") or []
    if not isinstance(windows_cfg, list) or not windows_cfg:
        raise SystemExit("config.windows must be a non-empty list of [start,end] pairs")
    windows: List[Tuple[str, str]] = []
    for item in windows_cfg:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise SystemExit("config.windows entries must be [start_date,end_date]")
        windows.append((str(item[0]), str(item[1])))

    exec_cfg = cfg.get("execution") or {}
    sync_up = bool(exec_cfg.get("sync_up", True))
    stop_after = bool(exec_cfg.get("stop_after", True))
    bar_minutes = float(exec_cfg.get("bar_minutes") or 15.0)
    overwrite_canonical = bool(exec_cfg.get("overwrite_canonical", True))

    git_cfg = cfg.get("git") or {}
    auto_stage = bool(git_cfg.get("auto_stage", True))

    configs_dir_rel = str(cfg.get("configs_dir") or "configs/budget1000_autopilot").strip()
    queue_dir_rel = str(cfg.get("queue_dir") or "artifacts/wfa/aggregate").strip()
    runs_dir_rel = str(cfg.get("runs_dir") or "artifacts/wfa/runs_clean").strip()

    state: Dict[str, Any] = {}
    if args.reset and state_path.exists():
        backup = state_path.with_suffix(state_path.suffix + f".bak_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
        backup.write_text(state_path.read_text(encoding="utf-8"), encoding="utf-8")
    elif state_path.exists() and not args.resume:
        raise SystemExit(f"State already exists: {state_path} (use --resume or --reset)")

    if args.resume and not args.reset:
        state = _load_state(state_path)

    if not state:
        state = {
            "schema_version": 1,
            "created_at_utc": _utc_now_iso(),
            "updated_at_utc": _utc_now_iso(),
            "config_path": str(config_path.relative_to(app_root)) if config_path.is_relative_to(app_root) else str(config_path),
            "controller_group": controller_group,
            "run_group_prefix": run_group_prefix,
            "base_config_path": base_config_rel,
            "round": 1,
            "knob_index": 0,
            "current_best": None,
            "best_score": None,
            "history": [],
            "no_improvement_streak": 0,
            "stop_reason": None,
            "done": False,
            "final_report": None,
            "round_context": None,
            "next_queue_plan": None,
            "last_round_result": None,
        }
        _save_state(state_path, state)

    before_normalize = json.dumps(state, sort_keys=True)
    _normalize_stop_state(state)
    if state.get("done") is not True and state.get("stop_reason") is not None:
        state["stop_reason"] = None
    if before_normalize != json.dumps(state, sort_keys=True):
        _save_state(state_path, state)

    if state.get("done") is True:
        print(f"[autopilot] done=true, nothing to do (state: {state_path})")
        return 0

    current_round = int(state.get("round") or 1)

    env_py = os.environ.copy()
    env_py["PYTHONPATH"] = str(app_root / "src")

    rollup_run_index = app_root / "artifacts" / "wfa" / "aggregate" / "rollup" / "run_index.csv"

    while current_round <= max_rounds:
        improved_in_round = False
        round_context = state.get("round_context")
        if not isinstance(round_context, dict) or int(round_context.get("round") or -1) != current_round:
            round_context = {
                "round": current_round,
                "started_at_utc": _utc_now_iso(),
                "best_before_round": _compact_best(state.get("current_best")),
            }
            state["round_context"] = round_context
            _save_state(state_path, state)
        round_best_before = _compact_best((round_context or {}).get("best_before_round"))

        fallback_plan = _derive_next_queue_plan(
            next_round=current_round,
            knobs_count=len(knobs),
            previous_result=state.get("last_round_result") if isinstance(state.get("last_round_result"), dict) else None,
        )
        round_plan = _normalize_queue_plan(
            raw_plan=state.get("next_queue_plan"),
            current_round=current_round,
            knobs_count=len(knobs),
            fallback_plan=fallback_plan,
        )
        state["next_queue_plan"] = round_plan
        state["knob_index"] = int(round_plan.get("knob_index") or 0)
        _save_state(state_path, state)

        attempt_specs: List[Tuple[int, str, str]] = [
            (
                int(round_plan.get("knob_index") or 0),
                str(round_plan.get("branch") or "fallback_knob"),
                str(round_plan.get("reason") or "unspecified"),
            )
        ]
        if str(round_plan.get("branch") or "") == "local_refine" and len(knobs) > 1:
            alt_idx = _next_alternative_knob_index(
                knobs_count=len(knobs),
                exclude_index=int(round_plan.get("knob_index") or 0),
                start_index=int(round_plan.get("knob_index") or 0) + 1,
            )
            if alt_idx != int(round_plan.get("knob_index") or 0):
                attempt_specs.append((int(alt_idx), "fallback_knob", "fallback_after_local_refine_empty"))

        selection: Optional[SelectionResult] = None
        run_group = ""
        queue_path: Optional[Path] = None
        used_knob_index = int(round_plan.get("knob_index") or 0)
        used_knob_key = ""
        used_branch = str(round_plan.get("branch") or "fallback_knob")
        used_plan_reason = str(round_plan.get("reason") or "unspecified")
        last_queue_error: Optional[str] = None

        for attempt_knob_index, attempt_branch, attempt_reason in attempt_specs:
            knob = knobs[attempt_knob_index]
            if not isinstance(knob, dict):
                raise SystemExit("Each knob must be a mapping with key/op/step/candidates")
            knob_key = str(knob.get("key") or "").strip()
            op = str(knob.get("op") or "").strip()
            step = float(knob.get("step") or 0.0)
            candidates = knob.get("candidates") or []
            if not knob_key or op not in {"add", "mul"} or step <= 0 or not isinstance(candidates, list):
                raise SystemExit(f"Invalid knob definition: {knob}")

            candidates_int = [int(x) for x in candidates]
            effective_candidates = _branch_candidate_offsets(candidates=candidates_int, branch=attempt_branch)
            if not effective_candidates:
                last_queue_error = (
                    f"knob={knob_key}: no effective candidates after branch={attempt_branch} normalization"
                )
                continue

            min_value = _to_float(knob.get("min")) if "min" in knob else None
            max_value = _to_float(knob.get("max")) if "max" in knob else None

            # IMPORTANT: always derive base_config_path from state so coordinate-ascent
            # updates apply immediately within the same process run.
            base_config_path = str(state.get("base_config_path") or base_config_rel)
            base_cfg_path = _resolve_under_root(base_config_path, root=app_root)
            base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))
            if not isinstance(base_cfg, dict):
                raise SystemExit(f"Invalid base YAML: {base_cfg_path}")
            current_value = float(get_nested(base_cfg, knob_key))
            values = _candidate_values(
                current=current_value,
                op=op,
                step=step,
                candidates=effective_candidates,
                min_value=min_value,
                max_value=max_value,
            )
            if not values:
                last_queue_error = f"knob={knob_key}: value grid is empty after min/max clipping"
                continue

            knob_slug = short_key_for_tag(knob_key)
            run_group = f"{run_group_prefix}_r{current_round:02d}_{knob_slug}"
            queue_path = app_root / queue_dir_rel / run_group / "run_queue.csv"
            spec_path = queue_path.parent / "sweep_spec.json"

            expected_spec = {
                "run_group": run_group,
                "base_config_path": base_config_path,
                "queue_branch": attempt_branch,
                "knob": {
                    "key": knob_key,
                    "op": op,
                    "step": step,
                    "candidates": candidates_int,
                    "effective_candidates": effective_candidates,
                    "values": values,
                    "current_value": current_value,
                },
                "windows": windows,
            }

            def _spec_matches() -> bool:
                if not spec_path.exists():
                    return False
                try:
                    payload = json.loads(spec_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    return False
                if not isinstance(payload, dict):
                    return False
                for key in ("run_group", "base_config_path", "windows", "queue_branch"):
                    if payload.get(key) != expected_spec.get(key):
                        return False
                knob_payload = payload.get("knob")
                if not isinstance(knob_payload, dict):
                    return False
                for key in ("key", "op", "step", "candidates", "effective_candidates", "values"):
                    if knob_payload.get(key) != expected_spec["knob"].get(key):
                        return False
                return True

            if not queue_path.exists() or not _spec_matches():
                # Rebuild queue from current round plan, skipping duplicates and invalid combos.
                seen_signatures = _collect_seen_config_signatures(
                    app_root=app_root,
                    queue_dir_rel=queue_dir_rel,
                    run_group_prefix=run_group_prefix,
                    exclude_run_group=run_group,
                )
                try:
                    queue_path = _generate_sweep_queue(
                        app_root=app_root,
                        run_group=run_group,
                        base_config_path=base_cfg_path,
                        windows=windows,
                        knob_key=knob_key,
                        knob_values=values,
                        configs_dir_rel=configs_dir_rel,
                        queue_dir_rel=queue_dir_rel,
                        runs_dir_rel=runs_dir_rel,
                        seen_signatures=seen_signatures,
                    )
                except ValueError as exc:
                    last_queue_error = str(exc)
                    continue

                _dump_json(
                    spec_path,
                    {
                        "generated_at_utc": _utc_now_iso(),
                        **expected_spec,
                    },
                )

            queue_configs = _queue_config_paths(queue_path)
            if not queue_configs:
                last_queue_error = f"Queue has 0 entries: {queue_path}"
                continue

            # Ensure queue + configs are tracked for SYNC_UP=1.
            # NOTE: SYNC_UP=1 uploads only `git ls-files` => stage every referenced config file
            # (not just the directory pathspec), otherwise VPS execution will see missing configs.
            tracked_paths: List[Path] = []
            tracked_paths.append(Path("coint4") / queue_path.relative_to(app_root))
            for cfg_rel in queue_configs:
                tracked_paths.append(Path("coint4") / cfg_rel)
            _ensure_tracked_for_sync_up(
                repo_root=repo_root,
                paths_rel_repo=tracked_paths,
                auto_stage=auto_stage,
            )

            if args.dry_run:
                print(f"[autopilot] dry-run: generated/ready queue: {queue_path.relative_to(app_root)}")
                return 0

            # Run on VPS (heavy) unless the queue is already completed locally.
            if _queue_is_fully_completed(queue_path, app_root=app_root):
                print(f"[autopilot] queue already completed, skipping VPS run: {queue_path.relative_to(app_root)}")
            else:
                _run_vps_queue(
                    repo_root=repo_root,
                    run_group=run_group,
                    queue_rel_app_root=str(queue_path.relative_to(app_root)),
                    sync_up=sync_up,
                    stop_after=stop_after,
                )

            # Local postprocess: canonical metrics + rollup rebuild.
            post_cmd = [
                str(py),
                "scripts/optimization/postprocess_queue.py",
                "--queue",
                str(queue_path.relative_to(app_root)),
                "--bar-minutes",
                str(bar_minutes),
                "--build-rollup",
            ]
            if overwrite_canonical:
                post_cmd.append("--overwrite-canonical")
            _run(post_cmd, cwd=app_root, env=env_py, check=True)

            # Selection for this sweep.
            selection = select_best_multiwindow(
                run_index_path=rollup_run_index,
                run_group=run_group,
                min_windows=min_windows,
                min_trades=min_trades,
                min_pairs=min_pairs,
                max_dd_pct=max_dd_pct,
                dd_target_pct=dd_target_pct,
                dd_penalty=dd_penalty,
            )
            used_knob_index = int(attempt_knob_index)
            used_knob_key = knob_key
            used_branch = attempt_branch
            used_plan_reason = attempt_reason
            break

        if selection is None:
            raise SystemExit(
                f"Unable to build/execute non-empty queue for round={current_round}: {last_queue_error or 'unknown_error'}"
            )

        prev_score = _best_score(state)
        improved = bool(selection.score > prev_score + float(min_improvement))
        if improved:
            improved_in_round = True
            state["current_best"] = {
                "updated_at_utc": _utc_now_iso(),
                "run_group": selection.run_group,
                "variant_id": selection.variant_id,
                "score": selection.score,
                "worst_robust_sharpe": selection.worst_robust_sharpe,
                "avg_robust_sharpe": selection.avg_robust_sharpe,
                "worst_dd_pct": selection.worst_dd_pct,
                "avg_dd_pct": selection.avg_dd_pct,
                "windows": selection.windows,
                "sample_config_path": selection.sample_config_path,
            }
            state["best_score"] = float(selection.score)
            state["no_improvement_streak"] = 0
            state["stop_reason"] = None
            state["base_config_path"] = selection.sample_config_path

        state.setdefault("history", []).append(
            {
                "at_utc": _utc_now_iso(),
                "round": current_round,
                "knob_index": used_knob_index,
                "knob_key": used_knob_key,
                "queue_branch": used_branch,
                "queue_plan_reason": used_plan_reason,
                "run_group": run_group,
                "best_variant_id": selection.variant_id,
                "best_score": selection.score,
                "best_worst_robust_sharpe": selection.worst_robust_sharpe,
                "best_worst_dd_pct": selection.worst_dd_pct,
                "improved": improved,
            }
        )
        state["knob_index"] = int(used_knob_index)
        state["last_round_result"] = {
            "round": current_round,
            "knob_index": int(used_knob_index),
            "knob_key": used_knob_key,
            "branch": used_branch,
            "improved": bool(improved),
            "run_group": run_group,
        }
        _save_state(state_path, state)

        # End of round.
        stop_by_no_improvement = _apply_no_improvement_round(
            state=state,
            improved_in_round=improved_in_round,
            no_improvement_rounds=no_improvement_rounds,
            min_improvement=min_improvement,
        )

        round_best_after = _compact_best(state.get("current_best"))
        decision = _round_decision(
            current_round=current_round,
            max_rounds=max_rounds,
            stopped_by_no_improvement=stop_by_no_improvement,
            stop_reason=state.get("stop_reason"),
        )
        next_queue_plan = None
        if str(decision.get("action") or "") == "continue":
            next_queue_plan = _derive_next_queue_plan(
                next_round=current_round + 1,
                knobs_count=len(knobs),
                previous_result=state.get("last_round_result") if isinstance(state.get("last_round_result"), dict) else None,
            )
        state["next_queue_plan"] = next_queue_plan
        round_payload = {
            "generated_at_utc": _utc_now_iso(),
            "controller_group": controller_group,
            "round": current_round,
            "best_before_round": round_best_before,
            "best_after_round": round_best_after,
            "delta_score": _metric_delta(before=round_best_before, after=round_best_after, key="score"),
            "delta_worst_robust_sharpe": _metric_delta(
                before=round_best_before,
                after=round_best_after,
                key="worst_robust_sharpe",
            ),
            "delta_worst_dd_pct": _metric_delta(before=round_best_before, after=round_best_after, key="worst_dd_pct"),
            "decision": decision,
            "queue_plan": {
                "requested": round_plan,
                "executed": {
                    "knob_index": int(used_knob_index),
                    "knob_key": used_knob_key,
                    "branch": used_branch,
                    "reason": used_plan_reason,
                    "run_group": run_group,
                },
                "next_round_plan": next_queue_plan,
            },
        }
        _write_round_analysis(controller_dir=controller_dir, payload=round_payload)

        if stop_by_no_improvement:
            report_path = _final_report_path(repo_root=repo_root)
            state["final_report"] = str(report_path.relative_to(repo_root))
            _save_state(state_path, state)
            _write_text(report_path, _render_final_report(state=state))
            print(f"[autopilot] stop-condition reached ({state.get('stop_reason')}), report: {report_path}")
            return 0
        if current_round >= max_rounds:
            break

        # New round around the updated best.
        current_round += 1
        state["round"] = current_round
        state["knob_index"] = _safe_knob_index(
            (next_queue_plan or {}).get("knob_index") if isinstance(next_queue_plan, dict) else 0,
            knobs_count=len(knobs),
            default=0,
        )
        state["round_context"] = {
            "round": current_round,
            "started_at_utc": _utc_now_iso(),
            "best_before_round": _compact_best(state.get("current_best")),
        }
        _save_state(state_path, state)

    # Safety cap: reached max_rounds.
    state["done"] = True
    state["stop_reason"] = f"max_rounds_reached: max_rounds={max_rounds}"
    report_path = _final_report_path(repo_root=repo_root)
    state["final_report"] = str(report_path.relative_to(repo_root))
    _save_state(state_path, state)
    _write_text(report_path, _render_final_report(state=state))
    print(f"[autopilot] max_rounds reached ({state.get('stop_reason')}), report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
