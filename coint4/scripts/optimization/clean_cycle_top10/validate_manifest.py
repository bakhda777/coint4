#!/usr/bin/env python3
"""Validate baseline_manifest.json and check FIXED_WINDOWS compatibility.

Task (C03 in tasks/clean_cycle_top10/prd_clean_cycle_top10.json):
- Validate manifest schema and basic path integrity.
- Extract walk_forward from configs and verify it matches FIXED_WINDOWS.walk_forward.
- Support --report-only mode which prints a report but does not fail.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _to_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if not s:
        return None
    if s in {"1", "true", "yes", "y"}:
        return True
    if s in {"0", "false", "no", "n"}:
        return False
    return None


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
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


def _norm_date_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    s = str(value).strip()
    return s or None


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
            out[key] = _norm_date_str(wf.get(key))
        elif key in {"training_period_days", "testing_period_days", "step_size_days", "max_steps", "gap_minutes"}:
            out[key] = _to_int(wf.get(key))
        elif key in {"refit_frequency"}:
            value = wf.get(key)
            out[key] = None if value is None else str(value).strip()
        else:
            out[key] = wf.get(key)
    return out


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


def _load_expected_walk_forward(*, args: argparse.Namespace, project_root: Path) -> Dict[str, Any]:
    if args.use_definitions:
        from definitions import FIXED_WINDOWS  # local module (same dir)

        node = FIXED_WINDOWS
    else:
        fixed_path = _resolve_under_project(args.fixed_windows_json, project_root)
        if fixed_path is None:
            raise ValueError("--fixed-windows-json is empty")
        if not fixed_path.exists():
            raise FileNotFoundError(f"fixed windows json not found: {fixed_path}")
        node = json.loads(fixed_path.read_text(encoding="utf-8"))

    if not isinstance(node, dict):
        raise ValueError("FIXED_WINDOWS must be a JSON object/dict")

    wf_node = node.get("walk_forward", node)
    if not isinstance(wf_node, dict):
        raise ValueError("FIXED_WINDOWS.walk_forward must be a JSON object/dict")

    # Normalize expected values for stable comparisons.
    return _normalize_walk_forward_subset(wf_node, WF_KEYS_ORDER)


def _load_config_walk_forward(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must parse into a mapping/object: {path}")
    node = payload.get("walk_forward")
    if not isinstance(node, dict):
        raise ValueError(f"config is missing walk_forward mapping: {path}")
    return node


@dataclass(frozen=True)
class _Issue:
    kind: str  # "error" | "warn"
    message: str


def _require_field(entry: Dict[str, Any], key: str) -> Tuple[bool, Any]:
    if key not in entry:
        return False, None
    return True, entry.get(key)


def _validate_entry_schema(entry: Dict[str, Any], idx: int) -> List[_Issue]:
    issues: List[_Issue] = []

    def err(msg: str) -> None:
        issues.append(_Issue("error", f"entry[{idx}]: {msg}"))

    required_str = ["run_group", "run_id", "results_dir"]
    for key in required_str:
        ok, val = _require_field(entry, key)
        if not ok:
            err(f"missing required field: {key}")
            continue
        if not str(val or "").strip():
            err(f"empty required field: {key}")

    required_any = [
        "config_path",
        "rank_sharpe",
        "rank_pnl_abs",
        "rank_max_drawdown_abs",
        "total_trades",
        "total_costs",
        "equity_present",
        "metrics_present",
        "config_sha256",
    ]
    for key in required_any:
        ok, _ = _require_field(entry, key)
        if not ok:
            err(f"missing required field: {key}")

    # Type-ish checks (best-effort, keeping CSV manifests supported).
    equity_present = _to_bool(entry.get("equity_present"))
    if equity_present is None:
        err("equity_present must be a boolean")
    metrics_present = _to_bool(entry.get("metrics_present"))
    if metrics_present is None:
        err("metrics_present must be a boolean")

    sharpe = _to_float(entry.get("rank_sharpe"))
    if sharpe is None:
        err("rank_sharpe must be a finite number")

    # Optional (but usually present).
    cfg_sha = entry.get("config_sha256")
    if cfg_sha is not None:
        s = str(cfg_sha).strip()
        if s and (len(s) != 64 or any(c not in "0123456789abcdef" for c in s.lower())):
            err("config_sha256 must be a 64-hex sha256 (or null)")

    return issues


def _validate_paths_and_hashes(
    entry: Dict[str, Any], idx: int, project_root: Path, strict: bool
) -> Tuple[List[_Issue], Optional[Path], Optional[Path]]:
    issues: List[_Issue] = []

    def push(kind: str, msg: str) -> None:
        issues.append(_Issue(kind, f"entry[{idx}]: {msg}"))

    results_dir = str(entry.get("results_dir") or "").strip()
    results_path = _resolve_under_project(results_dir, project_root)
    if results_path is None:
        push("error", "results_dir is empty")
    elif not results_path.exists():
        push("error" if strict else "warn", f"results_dir not found: {results_path}")
    elif not results_path.is_dir():
        push("error" if strict else "warn", f"results_dir is not a directory: {results_path}")
    else:
        # Cross-check presence flags, if possible.
        equity_present = _to_bool(entry.get("equity_present"))
        eq_exists = (results_path / "equity_curve.csv").exists()
        if equity_present is not None and equity_present != eq_exists:
            push(
                "error" if strict else "warn",
                f"equity_present mismatch: manifest={equity_present} disk={eq_exists} ({results_path})",
            )
        metrics_present = _to_bool(entry.get("metrics_present"))
        m_exists = (results_path / "strategy_metrics.csv").exists()
        if metrics_present is not None and metrics_present != m_exists:
            push(
                "error" if strict else "warn",
                f"metrics_present mismatch: manifest={metrics_present} disk={m_exists} ({results_path})",
            )

    config_path_raw = entry.get("config_path")
    config_path_str = "" if config_path_raw is None else str(config_path_raw).strip()
    config_path = _resolve_under_project(config_path_str, project_root) if config_path_str else None
    if config_path is None:
        push("error", "config_path is empty")
    elif not config_path.exists():
        push("error" if strict else "warn", f"config_path not found: {config_path}")
    elif not config_path.is_file():
        push("error" if strict else "warn", f"config_path is not a file: {config_path}")
    else:
        expected_sha = entry.get("config_sha256")
        actual_sha = _sha256_file(config_path)
        if expected_sha is None or not str(expected_sha).strip():
            push("error" if strict else "warn", f"config_sha256 missing (actual sha256={actual_sha})")
        elif str(expected_sha).strip().lower() != actual_sha.lower():
            push(
                "error" if strict else "warn",
                f"config_sha256 mismatch: manifest={expected_sha} actual={actual_sha} ({config_path})",
            )

    return issues, results_path, config_path


def _compare_walk_forward(
    actual_wf: Dict[str, Any], expected_norm: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Tuple[Any, Any]]]:
    actual_norm = _normalize_walk_forward_subset(actual_wf, expected_norm.keys())
    diffs: Dict[str, Tuple[Any, Any]] = {}
    for key, expected_val in expected_norm.items():
        got_val = actual_norm.get(key)
        if got_val != expected_val:
            diffs[key] = (expected_val, got_val)
    return actual_norm, diffs


def _print_report(
    *,
    manifest_path: Path,
    project_root: Path,
    entries: List[Dict[str, Any]],
    expected_wf: Dict[str, Any],
    issues: List[_Issue],
    window_counts: Counter[str],
    wf_mismatches: List[Tuple[str, str, Dict[str, Tuple[Any, Any]]]],
) -> None:
    print("== baseline_manifest validator ==")
    print(f"manifest: {_try_relpath(manifest_path, project_root)}")
    print(f"entries: {len(entries)}")
    print(f"expected FIXED_WINDOWS.walk_forward subset keys: {', '.join(WF_KEYS_ORDER)}")
    print("expected walk_forward:")
    print(json.dumps({k: expected_wf.get(k) for k in WF_KEYS_ORDER}, indent=2, sort_keys=True))
    print("")

    errs = [i for i in issues if i.kind == "error"]
    warns = [i for i in issues if i.kind == "warn"]
    print(f"schema/path issues: errors={len(errs)} warnings={len(warns)}")
    for item in errs + warns:
        print(f"- [{item.kind}] {item.message}")
    print("")

    print("walk_forward window frequencies among manifest entries:")
    if not window_counts:
        print("(none)")
    else:
        for sig, count in sorted(window_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"- {count}x {sig}")
    print("")

    if wf_mismatches:
        print("walk_forward mismatches vs FIXED_WINDOWS:")
        for run_id, cfg_path, diffs in wf_mismatches:
            print(f"- run_id={run_id} config={cfg_path}")
            for key in WF_KEYS_ORDER:
                if key not in diffs:
                    continue
                exp, got = diffs[key]
                print(f"  {key}: expected={exp!r} got={got!r}")
    else:
        print("walk_forward mismatches vs FIXED_WINDOWS: none")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate baseline_manifest and check configs match FIXED_WINDOWS.walk_forward"
    )
    parser.add_argument(
        "--manifest",
        default="artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/baseline_manifest.json",
        help="Path to baseline_manifest.(json|csv) (relative to coint4/ unless absolute).",
    )
    fixed_group = parser.add_mutually_exclusive_group(required=True)
    fixed_group.add_argument(
        "--fixed-windows-json",
        help="JSON with either FIXED_WINDOWS object (with walk_forward) or a raw walk_forward dict.",
    )
    fixed_group.add_argument(
        "--use-definitions",
        action="store_true",
        help="Use built-in definitions from scripts/optimization/clean_cycle_top10/definitions.py.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--strict", action="store_true", help="Fail on any issues (default).")
    mode.add_argument("--report-only", action="store_true", help="Print report but exit 0 even on mismatches.")
    args = parser.parse_args()

    strict = not bool(args.report_only)
    project_root = _resolve_project_root()
    manifest_path = _resolve_under_project(args.manifest, project_root) or (project_root / args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")

    expected_wf = _load_expected_walk_forward(args=args, project_root=project_root)

    entries = _load_manifest(manifest_path)
    issues: List[_Issue] = []

    # Validate manifest entry schema + paths/hashes.
    for idx, entry in enumerate(entries):
        issues.extend(_validate_entry_schema(entry, idx))
        path_issues, _, _ = _validate_paths_and_hashes(entry, idx, project_root, strict=strict)
        issues.extend(path_issues)

    # Extract walk_forward from configs and compare.
    window_counts: Counter[str] = Counter()
    wf_mismatches: List[Tuple[str, str, Dict[str, Tuple[Any, Any]]]] = []
    for idx, entry in enumerate(entries):
        run_id = str(entry.get("run_id") or "").strip() or f"entry[{idx}]"
        cfg_raw = entry.get("config_path")
        cfg_str = "" if cfg_raw is None else str(cfg_raw).strip()
        cfg_path = _resolve_under_project(cfg_str, project_root) if cfg_str else None
        if cfg_path is None or not cfg_path.exists() or not cfg_path.is_file():
            continue

        try:
            wf = _load_config_walk_forward(cfg_path)
            actual_norm, diffs = _compare_walk_forward(wf, expected_wf)
        except Exception as exc:  # noqa: BLE001 - want a robust report for CLI use
            issues.append(_Issue("error" if strict else "warn", f"entry[{idx}]: walk_forward parse failed: {exc}"))
            continue

        sig = json.dumps({k: actual_norm.get(k) for k in WF_KEYS_ORDER}, sort_keys=True)
        window_counts[sig] += 1
        if diffs:
            wf_mismatches.append((run_id, _try_relpath(cfg_path, project_root), diffs))

    _print_report(
        manifest_path=manifest_path,
        project_root=project_root,
        entries=entries,
        expected_wf=expected_wf,
        issues=issues,
        window_counts=window_counts,
        wf_mismatches=wf_mismatches,
    )

    if args.report_only:
        return 0

    has_errors = any(i.kind == "error" for i in issues)
    if has_errors or wf_mismatches:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

