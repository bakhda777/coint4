#!/usr/bin/env python3
"""Autonomous optimization loop: batch -> powered run -> rollup -> rank -> next batch."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(payload, dict):
        return payload
    return {}


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:  # noqa: BLE001
        return str(path)


@dataclass
class QueueTarget:
    run_group: str
    queue_path: Path
    source: str
    ready_rows: int


@dataclass
class RankResult:
    ok: bool
    source: str
    score: Optional[float]
    worst_robust_sharpe: Optional[float]
    worst_dd_pct: Optional[float]
    run_name: str
    config_path: str
    details: str


@dataclass
class CodexDecision:
    payload: Dict[str, Any]
    decision_json_path: Path
    context_path: Path
    decision_md_path: Path
    exec_log_path: Path


class AutonomousOptimizer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        # .../coint4/scripts/optimization/autonomous_optimize.py -> parents[2] == coint4/
        self.app_root = Path(__file__).resolve().parents[2]
        self.repo_root = self.app_root.parent
        self.state_dir = self.app_root / "artifacts" / "optimization_state"
        self.state_path = self.state_dir / "autonomous_state.json"
        self.main_log_path = self.state_dir / "autonomous_service.log"
        self.decisions_dir = self.state_dir / "decisions"
        self.rollup_csv = self.app_root / "artifacts" / "wfa" / "aggregate" / "rollup" / "run_index.csv"
        self.bridge11_path = self.app_root / "configs" / "autopilot" / "budget1000_batch_loop_bridge11_20260217.yaml"
        self.baseline_queue = (
            self.app_root
            / "artifacts"
            / "wfa"
            / "aggregate"
            / "20260219_budget1000_bl11_baseline_mini_risk"
            / "run_queue_mini.csv"
        )
        self.best_params_path = self.repo_root / "docs" / "best_params_latest.yaml"
        self.final_report_path = self.repo_root / "docs" / "final_report_latest.md"
        self.rank_results_dir = self.app_root / "artifacts" / "optimization_state" / "rank_results"
        self.python_exec = self.repo_root / ".venv" / "bin" / "python"
        self.powered_runner = self.app_root / "scripts" / "optimization" / "run_wfa_queue_powered.py"
        self.ensure_next_batch = self.app_root / "scripts" / "optimization" / "loop_orchestrator" / "ensure_next_batch.py"
        self.build_run_index = self.app_root / "scripts" / "optimization" / "build_run_index.py"
        self.rank_script = self.app_root / "scripts" / "optimization" / "rank_multiwindow_robust_runs.py"
        self.codex_schema_path = (
            self.app_root
            / "scripts"
            / "optimization"
            / "schemas"
            / "autopilot_decision.schema.json"
        )
        self._current_queue_for_ranking: Optional[Path] = None
        self._exit_after_iteration_wait: bool = False
        self._last_queue_selection_error: str = ""
        self._last_wait_reason: str = ""

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.decisions_dir.mkdir(parents=True, exist_ok=True)
        self.rank_results_dir.mkdir(parents=True, exist_ok=True)
        self.best_params_path.parent.mkdir(parents=True, exist_ok=True)
        self.final_report_path.parent.mkdir(parents=True, exist_ok=True)

        self.stop_policy = self._load_stop_policy()

    def log(self, message: str) -> None:
        line = f"{_utc_now()} {message}"
        print(line, flush=True)
        with self.main_log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _load_stop_policy(self) -> Dict[str, Any]:
        defaults = {
            "max_rounds": 5,
            "no_improvement_rounds": 2,
            "min_improvement": 0.02,
            "require_all_knobs_before_stop": True,
            "min_queue_entries": 60,
            "min_windows": 3,
            "min_trades": 200,
            "min_pairs": 20,
            "max_dd_pct": 0.14,
        }
        policy = dict(defaults)
        defaults_used: list[str] = []

        payload = _read_yaml(self.bridge11_path)
        search = payload.get("search") if isinstance(payload.get("search"), dict) else {}
        selection = payload.get("selection") if isinstance(payload.get("selection"), dict) else {}

        for key in ("max_rounds", "no_improvement_rounds", "min_queue_entries"):
            value = _to_int(search.get(key))
            if value is None:
                defaults_used.append(key)
            else:
                policy[key] = value

        min_improvement = _to_float(search.get("min_improvement"))
        if min_improvement is None:
            defaults_used.append("min_improvement")
        else:
            policy["min_improvement"] = float(min_improvement)

        require_all = search.get("require_all_knobs_before_stop")
        if require_all is None:
            defaults_used.append("require_all_knobs_before_stop")
        else:
            policy["require_all_knobs_before_stop"] = bool(require_all)

        for key in ("min_windows", "min_trades", "min_pairs"):
            value = _to_int(selection.get(key))
            if value is None:
                defaults_used.append(key)
            else:
                policy[key] = value

        max_dd_pct = _to_float(selection.get("max_dd_pct"))
        if max_dd_pct is None:
            defaults_used.append("max_dd_pct")
        else:
            policy["max_dd_pct"] = float(max_dd_pct)

        policy["defaults_used"] = sorted(set(defaults_used))
        policy["source"] = _safe_rel(self.bridge11_path, self.app_root) if self.bridge11_path.exists() else "defaults"
        return policy

    def _default_state(self) -> Dict[str, Any]:
        return {
            "status": "running",
            "iteration": 0,
            "current_run_group": "",
            "iteration_started_utc": "",
            "best_score": None,
            "best_run_name": "",
            "best_config_path": "",
            "no_improvement_streak": 0,
            "last_error": "",
            "last_iteration_phase": "",
            "ignored_run_groups": [],
            "ignored_queue_paths": [],
            "last_decision_id": "",
            "last_decision_action": "",
            "last_decision_explanation_md": "",
            "last_progress_log_utc": "",
            "last_updated_utc": _utc_now(),
        }

    def load_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return self._default_state()
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            self.log("autonomous: failed to parse state json, resetting to defaults")
            return self._default_state()
        if not isinstance(payload, dict):
            return self._default_state()
        state = self._default_state()
        state.update(payload)
        return state

    def save_state(self, state: Dict[str, Any]) -> None:
        for key in ("ignored_run_groups", "ignored_queue_paths"):
            values = state.get(key)
            if not isinstance(values, list):
                state[key] = []
                continue
            normalized = []
            seen: set[str] = set()
            for item in values:
                text = str(item).strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                normalized.append(text)
            state[key] = normalized
        state["last_updated_utc"] = _utc_now()
        self.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.app_root / "src")
        return env

    def _decision_stamp(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    def _sanitize_external_text(self, text: str) -> str:
        lines: list[str] = []
        for raw_line in str(text or "").splitlines():
            line = raw_line
            upper = line.upper()
            for marker in ("API_KEY", "SECRET", "TOKEN", "PASSWORD", "X-API-KEY"):
                if marker in upper and "=" in line:
                    key = line.split("=", 1)[0].strip()
                    line = f"{key}=<REDACTED>"
                    upper = line.upper()
            if "X-API-KEY" in upper:
                line = "X-API-KEY: <REDACTED>"
            lines.append(line)
        return "\n".join(lines)

    def _is_within(self, path: Path, root: Path) -> bool:
        try:
            path.resolve().relative_to(root.resolve())
            return True
        except Exception:  # noqa: BLE001
            return False

    def _resolve_repo_path(self, raw_path: str) -> Path:
        candidate = Path(str(raw_path or "").strip())
        if not candidate.as_posix():
            raise ValueError("Empty path in decision payload")
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.repo_root / candidate).resolve()
        if not self._is_within(resolved, self.repo_root):
            raise ValueError(f"Path outside repo: {raw_path}")
        return resolved

    def _resolve_app_relative(self, raw_path: str) -> str:
        path = self._resolve_repo_path(raw_path)
        if not self._is_within(path, self.app_root):
            raise ValueError(f"Path must be under app root: {raw_path}")
        return _safe_rel(path, self.app_root).replace("\\", "/")

    def _deep_merge(self, base: Any, override: Any) -> Any:
        if isinstance(base, dict) and isinstance(override, dict):
            merged = copy.deepcopy(base)
            for key, value in override.items():
                if key in merged:
                    merged[key] = self._deep_merge(merged[key], value)
                else:
                    merged[key] = copy.deepcopy(value)
            return merged
        return copy.deepcopy(override)

    def _latest_rank_result_payload(self) -> Optional[Dict[str, Any]]:
        files = sorted(self.rank_results_dir.glob("*_latest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            return None
        try:
            payload = json.loads(files[0].read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(payload, dict):
            return None
        payload["__path"] = _safe_rel(files[0], self.repo_root)
        return payload

    def _run_index_summary(self, limit: int = 80) -> Dict[str, Any]:
        summary: Dict[str, Any] = {"exists": self.rollup_csv.exists(), "total_rows": 0, "status_counts": {}, "sample_rows": []}
        if not self.rollup_csv.exists():
            return summary
        rows: list[dict[str, str]] = []
        status_counts: dict[str, int] = {}
        try:
            with self.rollup_csv.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    summary["total_rows"] += 1
                    status = str(row.get("status") or "").strip().lower() or "unknown"
                    status_counts[status] = status_counts.get(status, 0) + 1
                    if len(rows) < limit:
                        rows.append(
                            {
                                "run_group": str(row.get("run_group") or "").strip(),
                                "run_id": str(row.get("run_id") or "").strip(),
                                "status": status,
                                "metrics_present": str(row.get("metrics_present") or "").strip(),
                                "config_path": str(row.get("config_path") or "").strip(),
                                "results_dir": str(row.get("results_dir") or "").strip(),
                                "sharpe_ratio_abs": str(row.get("sharpe_ratio_abs") or "").strip(),
                                "max_drawdown_on_equity": str(row.get("max_drawdown_on_equity") or "").strip(),
                            }
                        )
        except Exception as exc:  # noqa: BLE001
            summary["error"] = f"{type(exc).__name__}:{exc}"
            return summary
        summary["status_counts"] = status_counts
        summary["sample_rows"] = rows
        return summary

    def _resolve_queue_for_run_group(self, run_group: str) -> Optional[Path]:
        group = str(run_group or "").strip()
        if not group:
            return None
        base = self.app_root / "artifacts" / "wfa" / "aggregate" / group
        if not base.exists():
            return None
        candidates: list[Path] = []
        for name in ("run_queue_mini.csv", "run_queue.csv"):
            path = base / name
            if path.exists():
                candidates.append(path)
        for path in sorted(base.glob("run_queue*.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
            if path not in candidates:
                candidates.append(path)
        return candidates[0] if candidates else None

    def _extract_queue_filters(self, queue_rows: list[dict[str, str]]) -> Dict[str, set[str]]:
        filters = {"run_ids": set(), "results_dirs": set(), "config_paths": set()}
        for row in queue_rows:
            run_id = str(
                row.get("run_name")
                or row.get("run_id")
                or row.get("name")
                or ""
            ).strip()
            if run_id:
                filters["run_ids"].add(run_id)
            config_path = str(row.get("config_path") or "").strip().replace("\\", "/")
            if config_path:
                filters["config_paths"].add(config_path)
            results_dir = str(row.get("results_dir") or "").strip().replace("\\", "/")
            if results_dir:
                filters["results_dirs"].add(results_dir)
        return filters

    def _row_matches_queue_filters(
        self,
        row: dict[str, str],
        *,
        run_group: str,
        filters: Dict[str, set[str]],
    ) -> bool:
        row_group = str(row.get("run_group") or "").strip()
        row_run_id = str(row.get("run_id") or "").strip()
        row_config = str(row.get("config_path") or "").strip().replace("\\", "/")
        row_results = str(row.get("results_dir") or "").strip().replace("\\", "/")

        has_filters = bool(filters["run_ids"] or filters["config_paths"] or filters["results_dirs"])
        if has_filters:
            if row_run_id and row_run_id in filters["run_ids"]:
                return True
            if row_config and row_config in filters["config_paths"]:
                return True
            if row_results and row_results in filters["results_dirs"]:
                return True
        return bool(run_group and row_group == run_group)

    def _bridge11_knobs_snapshot(self) -> Dict[str, Any]:
        payload = _read_yaml(self.bridge11_path)
        search = payload.get("search") if isinstance(payload.get("search"), dict) else {}
        selection = payload.get("selection") if isinstance(payload.get("selection"), dict) else {}
        batch_policy = payload.get("batch_policy") if isinstance(payload.get("batch_policy"), dict) else {}
        knobs_raw = search.get("knobs") if isinstance(search.get("knobs"), list) else []

        knobs: list[Dict[str, Any]] = []
        for knob in knobs_raw:
            if not isinstance(knob, dict):
                continue
            candidates = knob.get("candidates")
            candidate_list = candidates if isinstance(candidates, list) else []
            knobs.append(
                {
                    "key": str(knob.get("key") or "").strip(),
                    "op": str(knob.get("op") or "").strip(),
                    "step": knob.get("step"),
                    "min": knob.get("min"),
                    "max": knob.get("max"),
                    "candidates_count": len(candidate_list),
                    "candidates_preview": candidate_list[:8],
                }
            )
        return {
            "base_config": str(payload.get("base_config") or "").strip(),
            "windows": payload.get("windows") if isinstance(payload.get("windows"), list) else [],
            "selection_gates": selection,
            "batch_policy": batch_policy,
            "search": {
                "max_rounds": search.get("max_rounds"),
                "no_improvement_rounds": search.get("no_improvement_rounds"),
                "min_improvement": search.get("min_improvement"),
                "min_queue_entries": search.get("min_queue_entries"),
                "require_all_knobs_before_stop": search.get("require_all_knobs_before_stop"),
            },
            "knobs": knobs,
        }

    def _collect_decision_evidence(self, state: Dict[str, Any], *, row_limit: int = 200) -> Dict[str, Any]:
        run_group = str(state.get("current_run_group") or "").strip()
        queue_path = self._resolve_queue_for_run_group(run_group) if run_group else None

        queue_rows: list[dict[str, str]] = []
        queue_read_error = ""
        if queue_path is not None and queue_path.exists():
            try:
                queue_rows = self._read_queue_rows(queue_path)
            except Exception as exc:  # noqa: BLE001
                queue_read_error = f"{type(exc).__name__}:{exc}"

        queue_filters = self._extract_queue_filters(queue_rows)
        status_counts: Dict[str, int] = {}
        matched_rows: list[dict[str, str]] = []
        matched_total = 0
        completed_count = 0
        metrics_true_completed = 0
        metrics_false_completed = 0
        objective_available_count = 0

        if self.rollup_csv.exists():
            try:
                with self.rollup_csv.open("r", encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle):
                        if not self._row_matches_queue_filters(row, run_group=run_group, filters=queue_filters):
                            continue
                        matched_total += 1
                        status = str(row.get("status") or "").strip().lower() or "unknown"
                        status_counts[status] = status_counts.get(status, 0) + 1

                        is_completed = status == "completed"
                        metrics_present = _to_bool(row.get("metrics_present"))
                        if is_completed:
                            completed_count += 1
                            if metrics_present:
                                metrics_true_completed += 1
                            else:
                                metrics_false_completed += 1

                        if _to_float(row.get("score")) is not None:
                            objective_available_count += 1

                        if len(matched_rows) < row_limit:
                            matched_rows.append(
                                {
                                    "run_id": str(row.get("run_id") or "").strip(),
                                    "status": status,
                                    "metrics_present": str(row.get("metrics_present") or "").strip(),
                                    "config_path": str(row.get("config_path") or "").strip(),
                                    "results_dir": str(row.get("results_dir") or "").strip(),
                                    "score": str(row.get("score") or "").strip(),
                                    "worst_robust_sharpe": str(
                                        row.get("worst_robust_sharpe") or row.get("sharpe_ratio_abs") or ""
                                    ).strip(),
                                    "worst_dd_pct": str(
                                        row.get("worst_dd_pct") or row.get("max_drawdown_on_equity") or ""
                                    ).strip(),
                                }
                            )
            except Exception as exc:  # noqa: BLE001
                queue_read_error = queue_read_error or f"ROLLUP_READ_ERROR:{type(exc).__name__}:{exc}"

        candidates: list[Dict[str, Any]] = []
        for row in matched_rows:
            if str(row.get("status") or "").strip() != "completed":
                continue
            if not _to_bool(row.get("metrics_present")):
                continue
            objective = _to_float(row.get("score"))
            worst_sharpe = _to_float(row.get("worst_robust_sharpe"))
            worst_dd = _to_float(row.get("worst_dd_pct"))
            candidates.append(
                {
                    "run_id": str(row.get("run_id") or "").strip(),
                    "config_path": str(row.get("config_path") or "").strip(),
                    "results_dir": str(row.get("results_dir") or "").strip(),
                    "best_score": objective,
                    "worst_robust_sharpe": worst_sharpe,
                    "worst_dd_pct": worst_dd,
                    "ranking_basis": "objective_score" if objective is not None else "sharpe_proxy",
                }
            )

        candidates.sort(
            key=lambda row: (
                _to_float(row.get("best_score")) if row.get("best_score") is not None else float("-inf"),
                _to_float(row.get("worst_robust_sharpe")) if row.get("worst_robust_sharpe") is not None else float("-inf"),
            ),
            reverse=True,
        )
        if not any(row.get("best_score") is not None for row in candidates):
            candidates.sort(
                key=lambda row: (
                    _to_float(row.get("worst_robust_sharpe")) if row.get("worst_robust_sharpe") is not None else float("-inf"),
                ),
                reverse=True,
            )

        computed_missing_reasons: list[str] = []
        blocking_missing_reasons: list[str] = []

        if not self.rollup_csv.exists():
            computed_missing_reasons.append("ROLLUP_NOT_UPDATED")
            blocking_missing_reasons.append("ROLLUP_NOT_UPDATED")
        if queue_path is None and run_group:
            computed_missing_reasons.append("QUEUE_CONTEXT_MISSING")
        if queue_rows and matched_total == 0:
            computed_missing_reasons.append("ROLLUP_NOT_UPDATED")
            blocking_missing_reasons.append("ROLLUP_NOT_UPDATED")
        if completed_count == 0:
            computed_missing_reasons.append("NO_COMPLETED_RUNS")
            blocking_missing_reasons.append("NO_COMPLETED_RUNS")
        if completed_count > 0 and metrics_true_completed == 0:
            computed_missing_reasons.append("NO_CANONICAL_METRICS")
            blocking_missing_reasons.append("NO_CANONICAL_METRICS")
        if run_group and not self._rank_result_path(run_group).exists() and completed_count == 0:
            computed_missing_reasons.append("RANK_RESULT_MISSING")

        computed_missing_reasons = sorted(set(computed_missing_reasons))
        blocking_missing_reasons = sorted(set(blocking_missing_reasons))

        completed_threshold = max(1, int(self.stop_policy.get("min_windows") or 3))
        if completed_threshold < 3:
            completed_threshold = 3

        return {
            "current_run_group": run_group,
            "current_queue_path": _safe_rel(queue_path, self.repo_root) if queue_path else "",
            "queue_entry_count": len(queue_rows),
            "queue_ready_rows": sum(
                1 for row in queue_rows if str(row.get("status") or "planned").strip().lower() in {"planned", "stalled"}
            ),
            "queue_filters_counts": {
                "run_ids": len(queue_filters["run_ids"]),
                "config_paths": len(queue_filters["config_paths"]),
                "results_dirs": len(queue_filters["results_dirs"]),
            },
            "rollup_rows_scanned_limit": row_limit,
            "rollup_rows_matched": matched_total,
            "status_counts": status_counts,
            "completed_count": completed_count,
            "metrics_present_true_completed": metrics_true_completed,
            "metrics_present_false_completed": metrics_false_completed,
            "objective_available_count": objective_available_count,
            "completed_threshold_for_next_batch": completed_threshold,
            "decision_data_ready": bool(
                completed_count >= completed_threshold and metrics_true_completed > 0 and not blocking_missing_reasons
            ),
            "computed_missing_reasons": computed_missing_reasons,
            "blocking_missing_reasons": blocking_missing_reasons,
            "top_candidates": candidates[:10],
            "queue_read_error": queue_read_error,
        }

    def _parse_utc(self, value: Any) -> Optional[datetime]:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    def _extract_last_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        for line in reversed([chunk.strip() for chunk in str(text or "").splitlines() if chunk.strip()]):
            if not (line.startswith("{") and line.endswith("}")):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    def _probe_remote_queue_counts(self, queue_path: Path) -> Optional[Dict[str, Any]]:
        queue_rel = _safe_rel(queue_path, self.app_root)
        cmd = [
            str(self.python_exec),
            str(self.powered_runner),
            "--queue",
            queue_rel,
            "--probe-queue",
            "true",
            "--compute-host",
            "85.198.90.128",
            "--remote-repo",
            "auto",
            "--ssh-user",
            "root",
            "--ssh-port",
            "22",
            "--preflight",
            "false",
            "--sync-inputs",
            "false",
            "--wait-completion",
            "false",
            "--postprocess",
            "false",
            "--poweroff",
            "false",
            "--bootstrap-repo",
            "false",
            "--bootstrap-venv",
            "false",
            "--max-retries",
            "1",
            "--backoff-seconds",
            "1",
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                env=self._env(),
                text=True,
                capture_output=True,
                check=False,
            )
        except Exception:  # noqa: BLE001
            return None
        if proc.returncode != 0:
            return None
        payload = self._extract_last_json_object(proc.stdout or "")
        if payload is None:
            payload = self._extract_last_json_object((proc.stdout or "") + "\n" + (proc.stderr or ""))
        if not isinstance(payload, dict) or not bool(payload.get("ok")):
            return None
        counts_raw = payload.get("counts")
        counts: Dict[str, int] = {}
        if isinstance(counts_raw, dict):
            for key, value in counts_raw.items():
                name = str(key or "").strip().lower()
                if not name:
                    continue
                try:
                    counts[name] = int(value or 0)
                except (TypeError, ValueError):
                    counts[name] = 0
        try:
            total = int(payload.get("total") or 0)
        except (TypeError, ValueError):
            total = 0
        if total <= 0:
            total = int(sum(counts.values()))
        return {
            "counts": counts,
            "total": total,
            "has_metrics": bool(payload.get("has_metrics")),
        }

    def _apply_remote_probe_to_evidence(
        self,
        evidence: Dict[str, Any],
        remote_probe: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not isinstance(remote_probe, dict):
            return evidence
        counts_raw = remote_probe.get("counts")
        remote_counts: Dict[str, int] = {}
        if isinstance(counts_raw, dict):
            for key, value in counts_raw.items():
                name = str(key or "").strip().lower()
                if not name:
                    continue
                try:
                    remote_counts[name] = int(value or 0)
                except (TypeError, ValueError):
                    remote_counts[name] = 0
        try:
            remote_total = int(remote_probe.get("total") or 0)
        except (TypeError, ValueError):
            remote_total = 0
        if remote_total <= 0:
            remote_total = int(sum(remote_counts.values()))
        remote_has_metrics = bool(remote_probe.get("has_metrics"))

        patched = dict(evidence)
        patched["remote_counts"] = remote_counts
        patched["remote_total"] = remote_total
        patched["remote_has_metrics"] = remote_has_metrics

        local_completed = int(patched.get("completed_count") or 0)
        remote_completed = int(remote_counts.get("completed", 0) or 0)
        if local_completed == 0 and remote_completed > 0:
            computed = [
                str(item).strip()
                for item in list(patched.get("computed_missing_reasons") or [])
                if str(item).strip()
            ]
            blocking = [
                str(item).strip()
                for item in list(patched.get("blocking_missing_reasons") or [])
                if str(item).strip()
            ]
            computed = [item for item in computed if item != "NO_COMPLETED_RUNS"]
            blocking = [item for item in blocking if item != "NO_COMPLETED_RUNS"]
            if "ROLLUP_LAGGING_REMOTE" not in computed:
                computed.append("ROLLUP_LAGGING_REMOTE")
            if "ROLLUP_LAGGING_REMOTE" not in blocking:
                blocking.append("ROLLUP_LAGGING_REMOTE")
            patched["computed_missing_reasons"] = sorted(set(computed))
            patched["blocking_missing_reasons"] = sorted(set(blocking))
            patched["decision_data_ready"] = False
        return patched

    def log_progress_snapshot(
        self,
        state: Dict[str, Any],
        *,
        phase: str,
        min_interval_sec: int = 60,
    ) -> bool:
        now_utc = _utc_now()
        now_dt = self._parse_utc(now_utc)
        last_dt = self._parse_utc(state.get("last_progress_log_utc"))
        if now_dt is not None and last_dt is not None:
            if (now_dt - last_dt).total_seconds() < max(1, int(min_interval_sec)):
                return False

        evidence = self._collect_decision_evidence(state, row_limit=200)
        remote_probe: Optional[Dict[str, Any]] = None
        queue_rel = str(evidence.get("current_queue_path") or "").strip() or "n/a"
        phase_state = str(state.get("last_iteration_phase") or "").strip().lower()
        phase_hint = str(phase or "").strip().lower()
        should_probe_remote = phase_state in {"started", "waiting_codex"} or phase_hint in {
            "started",
            "waiting_codex",
            "wait:waiting_codex",
        }
        if should_probe_remote and queue_rel != "n/a":
            queue_path = self.repo_root / queue_rel
            if not queue_path.exists():
                queue_path = self.app_root / queue_rel
            try:
                active = queue_path.exists() and self._is_powered_runner_active_for_queue(queue_path)
            except Exception:  # noqa: BLE001
                active = False
            if active:
                remote_probe = self._probe_remote_queue_counts(queue_path)
        evidence = self._apply_remote_probe_to_evidence(evidence, remote_probe)
        status_counts = evidence.get("status_counts") if isinstance(evidence.get("status_counts"), dict) else {}
        planned = int(status_counts.get("planned", 0) or 0)
        running = int(status_counts.get("running", 0) or 0)
        stalled = int(status_counts.get("stalled", 0) or 0)
        completed = int(evidence.get("completed_count", 0) or 0)
        metrics_true = int(evidence.get("metrics_present_true_completed", 0) or 0)
        threshold = int(evidence.get("completed_threshold_for_next_batch", 0) or 0)
        ready = bool(evidence.get("decision_data_ready"))
        missing_reasons = list(evidence.get("blocking_missing_reasons") or evidence.get("computed_missing_reasons") or [])
        missing = ",".join(str(x).strip() for x in missing_reasons if str(x).strip()) or "-"

        remaining_completed = max(0, threshold - completed)
        remaining_metrics = 0 if metrics_true > 0 else 1
        remaining_to_unblock = max(remaining_completed, remaining_metrics)

        iteration_started = self._parse_utc(state.get("iteration_started_utc"))
        elapsed_sec = 0
        if now_dt is not None and iteration_started is not None:
            elapsed_sec = max(0, int((now_dt - iteration_started).total_seconds()))

        run_group = str(evidence.get("current_run_group") or "").strip() or "n/a"
        remote_probe_suffix = ""
        remote_counts = evidence.get("remote_counts") if isinstance(evidence.get("remote_counts"), dict) else {}
        if remote_counts:
            try:
                remote_total = int(evidence.get("remote_total") or 0)
            except (TypeError, ValueError):
                remote_total = 0
            remote_has_metrics = bool(evidence.get("remote_has_metrics"))
            remote_probe_suffix = (
                " remote_counts={counts} remote_total={total} remote_has_metrics={has_metrics}".format(
                    counts=json.dumps(remote_counts, ensure_ascii=False, sort_keys=True),
                    total=remote_total,
                    has_metrics=str(remote_has_metrics).lower(),
                )
            )
        self.log(
            "progress phase={phase} run_group={run_group} queue={queue} planned={planned} "
            "running={running} stalled={stalled} completed={completed} "
            "metrics_true_completed={metrics_true} threshold={threshold} ready={ready} "
            "missing={missing} remaining_to_unblock={remaining} elapsed_sec={elapsed}{remote_suffix}".format(
                phase=str(phase or "").strip() or "snapshot",
                run_group=run_group,
                queue=queue_rel,
                planned=planned,
                running=running,
                stalled=stalled,
                completed=completed,
                metrics_true=metrics_true,
                threshold=threshold,
                ready=str(ready).lower(),
                missing=missing,
                remaining=remaining_to_unblock,
                elapsed=elapsed_sec,
                remote_suffix=remote_probe_suffix,
            )
        )
        state["last_progress_log_utc"] = now_utc
        return True

    def _build_decision_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        recent_logs = sorted((self.state_dir / "iterations").glob("iter_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        recent_log_paths = [_safe_rel(path, self.repo_root) for path in recent_logs[:10]]
        evidence = self._collect_decision_evidence(state, row_limit=200)
        queue_rel = str(evidence.get("current_queue_path") or "").strip()
        remote_probe: Optional[Dict[str, Any]] = None
        if queue_rel:
            queue_path = self.repo_root / queue_rel
            if not queue_path.exists():
                queue_path = self.app_root / queue_rel
            try:
                active = queue_path.exists() and self._is_powered_runner_active_for_queue(queue_path)
            except Exception:  # noqa: BLE001
                active = False
            if active:
                remote_probe = self._probe_remote_queue_counts(queue_path)
        evidence = self._apply_remote_probe_to_evidence(evidence, remote_probe)
        context = {
            "generated_at_utc": _utc_now(),
            "repo_root": str(self.repo_root),
            "app_root": str(self.app_root),
            "state": state,
            "stop_policy": self.stop_policy,
            "canonical_sources": {
                "run_index_csv": _safe_rel(self.rollup_csv, self.repo_root),
                "rank_results_dir": _safe_rel(self.rank_results_dir, self.repo_root),
                "bridge11_config": _safe_rel(self.bridge11_path, self.repo_root),
            },
            "run_index_summary": self._run_index_summary(limit=80),
            "latest_rank_result": self._latest_rank_result_payload(),
            "evidence": evidence,
            "knobs_snapshot": self._bridge11_knobs_snapshot(),
            "policy": {
                "allow_changes": [
                    "queue generation",
                    "parameter overrides",
                    "blacklists and filters",
                    "config ranges under repo",
                    "file_edits within repo root",
                ],
                "must_keep": [
                    "canonical metrics formulas unchanged",
                    "WFA windows and metric source of truth unchanged unless explicit decision rationale",
                    "no secrets in logs/files",
                ],
                "decision_rule_wait": (
                    "Use next_action='wait' only when evidence.blocking_missing_reasons is non-empty. "
                    "List every blocking reason in stop_reason and human_explanation_md."
                ),
                "decision_rule_when_ready": (
                    "If evidence.completed_count >= evidence.completed_threshold_for_next_batch and "
                    "metrics_present_true_completed > 0, default to run_next_batch unless stop=true is justified."
                ),
            },
            "recent_iteration_logs": recent_log_paths,
            "runtime_constraints": {
                "must_use_powered_runner": _safe_rel(self.powered_runner, self.repo_root),
                "compute_host": "85.198.90.128",
                "parallel": 1,
                "poweroff": True,
                "postprocess": True,
                "wait_completion": True,
            },
            "guardrails": [
                "No secrets in outputs.",
                "Do not change canonical metric formulas.",
                "Decision and stop must be via Codex JSON schema output.",
            ],
        }
        return context

    def _decision_prompt(self, context: Dict[str, Any]) -> str:
        payload = json.dumps(context, ensure_ascii=False, indent=2)
        return (
            "Ты — decision engine для автономного оптимизатора coint4.\n"
            "Цель: максимизировать objective bridge11 (worst_robust_sharpe - dd_penalty) с учётом risk gates.\n"
            "Можешь менять параметры, фильтры, blacklist'ы и формировать следующий batch в пределах репозитория.\n"
            "Ограничения: не менять формулы метрик и окна WFA; не выводить секреты.\n"
            "Верни РОВНО один JSON-объект по схеме.\n"
            "next_action='wait' разрешён ТОЛЬКО когда context.evidence.blocking_missing_reasons непустой.\n"
            "Если выбираешь wait: перечисли ВСЕ blocking_missing_reasons в stop_reason и подробно в human_explanation_md.\n"
            "Если completed_count >= completed_threshold_for_next_batch и есть metrics_present_true_completed > 0,\n"
            "безосновательный wait запрещён: выбери run_next_batch или stop с проверяемым объяснением.\n"
            "Для run_next_batch: сформируй actionable queue_entries (обычно 10-25, если нет явных ограничений),\n"
            "каждому entry дай notes, а для file_edits добавь rationale.\n"
            "Если нужно менять параметры вне стандартных overrides, используй file_edits.\n"
            "stop=true ставь только когда обоснованно считаешь, что дальше улучшать некуда.\n\n"
            f"Контекст:\n{payload}\n"
        )

    def _validate_decision(self, payload: Dict[str, Any]) -> Optional[str]:
        required = [
            "decision_id",
            "stop",
            "stop_reason",
            "human_explanation_md",
            "next_run_group",
            "next_queue_path",
            "queue_entries",
            "file_edits",
            "constraints",
            "next_action",
            "wait_seconds",
        ]
        for key in required:
            if key not in payload:
                return f"MISSING_FIELD:{key}"
        if not isinstance(payload.get("decision_id"), str):
            return "INVALID_FIELD:decision_id"
        if not isinstance(payload.get("stop"), bool):
            return "INVALID_FIELD:stop"
        if payload.get("next_action") not in {"run_next_batch", "wait", "stop"}:
            return "INVALID_FIELD:next_action"
        wait_seconds = payload.get("wait_seconds")
        if not isinstance(wait_seconds, int) or wait_seconds <= 0:
            return "INVALID_FIELD:wait_seconds"
        queue_entries = payload.get("queue_entries")
        if not isinstance(queue_entries, list):
            return "INVALID_FIELD:queue_entries"
        for idx, entry in enumerate(queue_entries):
            if not isinstance(entry, dict):
                return f"INVALID_QUEUE_ENTRY:{idx}"
            for key in ("config_path", "status", "results_dir", "notes", "overrides"):
                if key not in entry:
                    return f"MISSING_QUEUE_ENTRY_FIELD:{idx}:{key}"
            if str(entry.get("status") or "").strip() != "planned":
                return f"INVALID_QUEUE_ENTRY_STATUS:{idx}"
            if not isinstance(entry.get("overrides"), dict):
                return f"INVALID_QUEUE_ENTRY_OVERRIDES:{idx}"
        file_edits = payload.get("file_edits")
        if not isinstance(file_edits, list):
            return "INVALID_FIELD:file_edits"
        for idx, edit in enumerate(file_edits):
            if not isinstance(edit, dict):
                return f"INVALID_FILE_EDIT:{idx}"
            for key in ("path", "op", "content", "rationale"):
                if key not in edit:
                    return f"MISSING_FILE_EDIT_FIELD:{idx}:{key}"
            if str(edit.get("op") or "").strip() not in {"write_text", "write_yaml", "write_json", "append_text"}:
                return f"INVALID_FILE_EDIT_OP:{idx}"
        constraints = payload.get("constraints")
        if not isinstance(constraints, dict):
            return "INVALID_FIELD:constraints"
        if constraints.get("allow_anything_in_repo") is not True:
            return "INVALID_FIELD:constraints.allow_anything_in_repo"
        if not isinstance(constraints.get("must_keep"), list):
            return "INVALID_FIELD:constraints.must_keep"
        return None

    def _persist_decision_memo(
        self,
        *,
        decision_payload: Dict[str, Any],
        decision_md_path: Path,
    ) -> None:
        summary = [
            f"# Decision {decision_payload.get('decision_id')}",
            "",
            f"- stop: {decision_payload.get('stop')}",
            f"- next_action: {decision_payload.get('next_action')}",
            f"- next_run_group: {decision_payload.get('next_run_group')}",
            f"- next_queue_path: {decision_payload.get('next_queue_path')}",
            f"- queue_entries: {len(list(decision_payload.get('queue_entries') or []))}",
            "",
            "## Human Explanation",
            "",
            str(decision_payload.get("human_explanation_md") or "").strip(),
            "",
            "## Raw Decision",
            "",
            "```json",
            json.dumps(decision_payload, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
        decision_md_path.parent.mkdir(parents=True, exist_ok=True)
        decision_md_path.write_text("\n".join(summary), encoding="utf-8")

    def _log_human_explanation(self, explanation_md: str) -> None:
        self.log("codex human_explanation_md START")
        for line in str(explanation_md or "").splitlines() or [""]:
            self.log(f"  {line}")
        self.log("codex human_explanation_md END")

    def decide_with_codex(self, state: Dict[str, Any]) -> Optional[CodexDecision]:
        stamp = self._decision_stamp()
        context_path = self.decisions_dir / f"context_{stamp}.json"
        decision_json_path = self.decisions_dir / f"decision_{stamp}.json"
        decision_md_path = self.decisions_dir / f"decision_{stamp}.md"
        exec_log_path = self.decisions_dir / f"codex_exec_{stamp}.jsonl"

        context = self._build_decision_context(state)
        context_path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
        prompt = self._decision_prompt(context)

        cmd = [
            "codex",
            "exec",
            "--cd",
            str(self.repo_root),
            "--output-schema",
            str(self.codex_schema_path),
            "--output-last-message",
            str(decision_json_path),
            "--json",
            "-",
        ]
        try:
            proc = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError:
            self._last_wait_reason = "CODEX_EXEC_UNAVAILABLE"
            return None
        except Exception as exc:  # noqa: BLE001
            self._last_wait_reason = f"CODEX_EXEC_ERROR:{type(exc).__name__}"
            return None

        event = {
            "ts": _utc_now(),
            "cmd": cmd,
            "rc": int(proc.returncode),
            "stdout": self._sanitize_external_text(proc.stdout or ""),
            "stderr": self._sanitize_external_text(proc.stderr or ""),
            "context_path": _safe_rel(context_path, self.repo_root),
            "decision_json_path": _safe_rel(decision_json_path, self.repo_root),
        }
        exec_log_path.write_text(json.dumps(event, ensure_ascii=False) + "\n", encoding="utf-8")

        if proc.returncode != 0:
            self._last_wait_reason = f"CODEX_EXEC_RC{proc.returncode}"
            return None
        if not decision_json_path.exists():
            self._last_wait_reason = "CODEX_DECISION_MISSING"
            return None
        try:
            payload = json.loads(decision_json_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            self._last_wait_reason = f"CODEX_DECISION_INVALID_JSON:{type(exc).__name__}"
            return None
        if not isinstance(payload, dict):
            self._last_wait_reason = "CODEX_DECISION_INVALID_PAYLOAD"
            return None
        validation_error = self._validate_decision(payload)
        if validation_error:
            self._last_wait_reason = f"CODEX_DECISION_SCHEMA:{validation_error}"
            return None

        self._persist_decision_memo(decision_payload=payload, decision_md_path=decision_md_path)
        self.log(
            "codex decision_id={decision_id} next_action={next_action} stop={stop}".format(
                decision_id=str(payload.get("decision_id") or "").strip(),
                next_action=str(payload.get("next_action") or "").strip(),
                stop=bool(payload.get("stop")),
            )
        )
        self._log_human_explanation(str(payload.get("human_explanation_md") or ""))
        return CodexDecision(
            payload=payload,
            decision_json_path=decision_json_path,
            context_path=context_path,
            decision_md_path=decision_md_path,
            exec_log_path=exec_log_path,
        )

    def _apply_file_edits(self, decision: CodexDecision) -> None:
        edits = list(decision.payload.get("file_edits") or [])
        for edit in edits:
            path = self._resolve_repo_path(str(edit.get("path") or ""))
            op = str(edit.get("op") or "").strip()
            content = edit.get("content")
            path.parent.mkdir(parents=True, exist_ok=True)
            if op == "write_text":
                path.write_text(str(content), encoding="utf-8")
            elif op == "append_text":
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(str(content))
            elif op == "write_yaml":
                if isinstance(content, str):
                    path.write_text(str(content), encoding="utf-8")
                else:
                    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
            elif op == "write_json":
                if isinstance(content, str):
                    try:
                        parsed = json.loads(content)
                    except Exception as exc:  # noqa: BLE001
                        raise ValueError(f"Invalid JSON content for {path}: {type(exc).__name__}") from exc
                    path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                else:
                    path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            else:
                raise ValueError(f"Unsupported file edit op: {op}")

    def _materialize_queue_from_decision(self, decision: CodexDecision) -> QueueTarget:
        payload = decision.payload
        run_group = str(payload.get("next_run_group") or "").strip()
        if not run_group:
            raise ValueError("Decision missing next_run_group")
        queue_path = self._resolve_repo_path(str(payload.get("next_queue_path") or ""))
        queue_entries = list(payload.get("queue_entries") or [])
        if not queue_entries:
            raise ValueError("Decision next_action=run_next_batch but queue_entries is empty")

        base_cfg = _read_yaml(self.bridge11_path)
        rows: list[dict[str, str]] = []
        for entry in queue_entries:
            config_abs = self._resolve_repo_path(str(entry.get("config_path") or ""))
            if not self._is_within(config_abs, self.app_root):
                raise ValueError(f"config_path must be inside app root: {config_abs}")
            overrides = entry.get("overrides") if isinstance(entry.get("overrides"), dict) else {}
            cfg_payload = self._deep_merge(base_cfg, overrides)
            config_abs.parent.mkdir(parents=True, exist_ok=True)
            config_abs.write_text(yaml.safe_dump(cfg_payload, sort_keys=False), encoding="utf-8")

            results_raw = str(entry.get("results_dir") or "").strip()
            if not results_raw:
                raise ValueError("queue entry results_dir is empty")
            results_candidate = Path(results_raw)
            if results_candidate.is_absolute():
                if not self._is_within(results_candidate, self.app_root):
                    raise ValueError(f"results_dir must be under app root: {results_raw}")
                results_rel = _safe_rel(results_candidate, self.app_root).replace("\\", "/")
            else:
                results_rel = results_candidate.as_posix()

            rows.append(
                {
                    "config_path": _safe_rel(config_abs, self.app_root).replace("\\", "/"),
                    "results_dir": results_rel,
                    "status": "planned",
                }
            )

        queue_path.parent.mkdir(parents=True, exist_ok=True)
        with queue_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["config_path", "results_dir", "status"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        queue_path_app = queue_path
        if not self._is_within(queue_path_app, self.app_root):
            raise ValueError(f"next_queue_path must be under app root: {queue_path}")
        return QueueTarget(
            run_group=run_group,
            queue_path=queue_path_app,
            source="codex_decision",
            ready_rows=len(rows),
        )

    def _read_queue_rows(self, queue_path: Path) -> list[dict[str, str]]:
        with queue_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]

    def _count_ready_rows(self, queue_path: Path) -> int:
        try:
            rows = self._read_queue_rows(queue_path)
        except Exception:  # noqa: BLE001
            return 0
        ready = 0
        for row in rows:
            status = str(row.get("status") or "planned").strip().lower()
            if status in {"planned", "stalled"}:
                ready += 1
        return ready

    def _is_demo_queue(self, queue_path: Path) -> bool:
        run_group = str(queue_path.parent.name or "").strip().lower()
        if run_group.startswith("demo"):
            return True
        rel = _safe_rel(queue_path, self.app_root).replace("\\", "/").lower()
        return "/demo" in rel or rel.startswith("demo/")

    def _queue_selection_log_path(self) -> Path:
        path = self.app_root / "artifacts" / "optimization_state" / "iterations" / "queue_selection.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _queue_target_for_run_group(self, run_group: str, *, source: str) -> Optional[QueueTarget]:
        group = str(run_group or "").strip()
        if not group:
            return None
        queue_path = self._resolve_queue_for_run_group(group)
        if queue_path is None:
            return None
        return QueueTarget(
            run_group=group,
            queue_path=queue_path,
            source=source,
            ready_rows=self._count_ready_rows(queue_path),
        )

    def _is_powered_runner_active_for_queue(self, queue_path: Path) -> bool:
        queue_abs = str(queue_path.resolve())
        queue_rel_repo = _safe_rel(queue_path, self.repo_root).replace("\\", "/")
        queue_rel_app = _safe_rel(queue_path, self.app_root).replace("\\", "/")
        try:
            proc = subprocess.run(
                ["ps", "-eo", "pid,args"],
                text=True,
                capture_output=True,
                check=False,
            )
        except Exception:  # noqa: BLE001
            return False
        if proc.returncode != 0:
            return False
        for raw_line in (proc.stdout or "").splitlines():
            line = raw_line.strip()
            if "run_wfa_queue_powered.py" not in line:
                continue
            if queue_abs in line or queue_rel_repo in line or queue_rel_app in line:
                return True
        return False

    def _append_queue_selection_failure(
        self,
        *,
        cmd: list[str],
        return_code: int,
        stdout: str,
        stderr: str,
    ) -> None:
        body = (
            f"CMD: {' '.join(cmd)}\n"
            f"RC: {return_code}\n"
            "STDOUT:\n"
            f"{(stdout or '').rstrip()}\n"
            "STDERR:\n"
            f"{(stderr or '').rstrip()}\n"
        )
        self._append_iteration_log(self._queue_selection_log_path(), "ENSURE_NEXT_BATCH_FAIL", body)

    def _is_ignored_queue(self, queue_path: Path, state: Dict[str, Any]) -> bool:
        ignored_run_groups = {str(x).strip() for x in list(state.get("ignored_run_groups") or []) if str(x).strip()}
        ignored_queue_paths = {
            str(x).strip().replace("\\", "/")
            for x in list(state.get("ignored_queue_paths") or [])
            if str(x).strip()
        }
        run_group = str(queue_path.parent.name or "").strip()
        queue_rel = _safe_rel(queue_path, self.app_root).replace("\\", "/")
        if run_group in ignored_run_groups:
            return True
        if queue_rel in ignored_queue_paths:
            return True
        return False

    def _record_ignored_queue(self, state: Dict[str, Any], queue: QueueTarget) -> None:
        run_groups = list(state.get("ignored_run_groups") or [])
        queue_paths = list(state.get("ignored_queue_paths") or [])
        run_group = str(queue.run_group).strip()
        queue_rel = _safe_rel(queue.queue_path, self.app_root).replace("\\", "/")
        if run_group and run_group not in run_groups:
            run_groups.append(run_group)
        if queue_rel and queue_rel not in queue_paths:
            queue_paths.append(queue_rel)
        state["ignored_run_groups"] = run_groups
        state["ignored_queue_paths"] = queue_paths

    def _quarantine_demo_queues(self) -> None:
        aggregate = self.app_root / "artifacts" / "wfa" / "aggregate"
        if not aggregate.exists():
            return
        quarantine_root = aggregate / "_quarantine_demo"
        for child in sorted(aggregate.iterdir()):
            if not child.is_dir():
                continue
            name = str(child.name or "")
            if name == "_quarantine_demo":
                continue
            if not name.lower().startswith("demo"):
                continue
            quarantine_root.mkdir(parents=True, exist_ok=True)
            target = quarantine_root / name
            if target.exists():
                target = quarantine_root / f"{name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            try:
                shutil.move(str(child), str(target))
                self.log(
                    "autonomous: quarantine demo queue dir "
                    f"{_safe_rel(child, self.app_root)} -> {_safe_rel(target, self.app_root)}"
                )
            except Exception as exc:  # noqa: BLE001
                self.log(
                    "autonomous: failed to quarantine demo queue dir "
                    f"{_safe_rel(child, self.app_root)} ({type(exc).__name__})"
                )

    def _discover_ready_queues(self) -> list[QueueTarget]:
        aggregate = self.app_root / "artifacts" / "wfa" / "aggregate"
        out: list[QueueTarget] = []
        if not aggregate.exists():
            return out
        patterns = ("run_queue.csv", "run_queue_mini.csv")
        candidates: list[Path] = []
        for pattern in patterns:
            candidates.extend(aggregate.glob(f"*/{pattern}"))
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for queue_path in candidates:
            if self._is_demo_queue(queue_path):
                continue
            ready = self._count_ready_rows(queue_path)
            if ready <= 0:
                continue
            out.append(
                QueueTarget(
                    run_group=queue_path.parent.name,
                    queue_path=queue_path,
                    source="discover_ready_queues",
                    ready_rows=ready,
                )
            )
        return out

    def _prioritize_queues(self, queues: list[QueueTarget]) -> list[QueueTarget]:
        preferred = [item for item in queues if "budget1000_bl11_" in str(item.run_group)]
        others = [item for item in queues if item not in preferred]
        return preferred + others

    def _extract_json_from_output(self, text: str) -> Optional[Dict[str, Any]]:
        lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
        for line in reversed(lines):
            if not (line.startswith("{") and line.endswith("}")):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    def _run_codex_json(self, *, prompt: str, schema: Dict[str, Any], timeout_sec: int = 180) -> Optional[Dict[str, Any]]:
        if not bool(self.args.use_codex_exec):
            return None
        codex_bin = shutil.which("codex")
        if not codex_bin:
            self.log("autonomous: codex exec requested but codex binary not found; fallback deterministic")
            return None

        with tempfile.TemporaryDirectory(prefix="autonomous_codex_") as tmp_dir:
            tmp = Path(tmp_dir)
            schema_path = tmp / "schema.json"
            out_path = tmp / "out.json"
            schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
            cmd = [
                codex_bin,
                "exec",
                "-C",
                str(self.repo_root),
                "-s",
                "read-only",
                "--skip-git-repo-check",
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(out_path),
            ]
            if str(self.args.codex_model or "").strip():
                cmd.extend(["-m", str(self.args.codex_model).strip()])

            self.log("autonomous: codex exec start")
            proc = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                capture_output=True,
                check=False,
                timeout=max(30, int(timeout_sec)),
            )
            if proc.returncode != 0:
                self.log(f"autonomous: codex exec failed rc={proc.returncode}; fallback deterministic")
                return None
            if not out_path.exists():
                self.log("autonomous: codex exec missing output file; fallback deterministic")
                return None
            try:
                payload = json.loads(out_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.log("autonomous: codex exec output is not valid json; fallback deterministic")
                return None
            if not isinstance(payload, dict):
                self.log("autonomous: codex exec output schema mismatch; fallback deterministic")
                return None
            return payload

    def _select_queue_with_codex(self, candidates: list[QueueTarget], state: Dict[str, Any]) -> Optional[QueueTarget]:
        if not candidates:
            return None
        schema = {
            "type": "object",
            "properties": {
                "queue_rel": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["queue_rel", "reason"],
            "additionalProperties": False,
        }
        items = [
            {
                "run_group": item.run_group,
                "queue_rel": _safe_rel(item.queue_path, self.app_root),
                "ready_rows": item.ready_rows,
            }
            for item in candidates[:15]
        ]
        prompt = (
            "Выбери одну очередь для следующего батча оптимизации.\n"
            "Критерий: предпочесть самую актуальную и пригодную к продолжению.\n"
            f"Кандидаты JSON: {json.dumps(items, ensure_ascii=False)}\n"
            "Верни только queue_rel и reason."
        )
        payload = self._run_codex_json(prompt=prompt, schema=schema, timeout_sec=120)
        if not payload:
            return None
        queue_rel = str(payload.get("queue_rel") or "").strip()
        if not queue_rel:
            return None
        queue_path = self.app_root / queue_rel
        if not queue_path.exists():
            self.log(f"autonomous: codex suggested missing queue {queue_rel}, fallback deterministic")
            return None
        if self._is_demo_queue(queue_path):
            self.log(f"autonomous: skip demo queue from codex suggestion: {queue_rel}")
            return None
        if self._is_ignored_queue(queue_path, state):
            self.log(f"autonomous: codex suggested ignored queue {queue_rel}, fallback deterministic")
            return None
        ready = self._count_ready_rows(queue_path)
        if ready <= 0:
            self.log(f"autonomous: codex suggested queue without ready rows {queue_rel}, fallback deterministic")
            return None
        self.log(f"autonomous: codex selected queue={queue_rel}")
        return QueueTarget(
            run_group=queue_path.parent.name,
            queue_path=queue_path,
            source="codex_exec",
            ready_rows=ready,
        )

    def _select_queue_via_ensure_next_batch(self, state: Dict[str, Any]) -> Optional[QueueTarget]:
        if not self.ensure_next_batch.exists():
            return None
        cmd = [
            str(self.python_exec),
            "scripts/optimization/loop_orchestrator/ensure_next_batch.py",
            "--dry-run",
            "--queue",
            "auto",
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(self.app_root),
            env=self._env(),
            text=True,
            capture_output=True,
            check=False,
        )
        merged = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        if proc.returncode != 0:
            self._last_queue_selection_error = f"ENSURE_NEXT_BATCH_RC{proc.returncode}"
            self._append_queue_selection_failure(
                cmd=cmd,
                return_code=int(proc.returncode),
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
            )
            self.log(f"autonomous: ensure_next_batch dry-run failed rc={proc.returncode}")
            return None
        payload = self._extract_json_from_output(merged)
        if not payload:
            self.log("autonomous: ensure_next_batch returned no json payload")
            return None
        queue_rel = str(payload.get("queue_rel") or "").strip()
        if not queue_rel:
            self.log("autonomous: ensure_next_batch json has empty queue_rel")
            return None
        queue_path = self.app_root / queue_rel
        if not queue_path.exists():
            self.log(f"autonomous: ensure_next_batch queue missing on disk: {queue_rel}")
            return None
        if self._is_demo_queue(queue_path):
            self.log(f"autonomous: skip demo queue from ensure_next_batch: {queue_rel}")
            return None
        if self._is_ignored_queue(queue_path, state):
            self.log(f"autonomous: ensure_next_batch suggested ignored queue: {queue_rel}")
            return None
        ready = self._count_ready_rows(queue_path)
        if ready <= 0:
            self.log(f"autonomous: ensure_next_batch queue has no ready rows: {queue_rel}")
            return None
        return QueueTarget(
            run_group=str(payload.get("run_group") or queue_path.parent.name).strip() or queue_path.parent.name,
            queue_path=queue_path,
            source="ensure_next_batch",
            ready_rows=ready,
        )

    def select_next_queue(self, *, state: Dict[str, Any]) -> Optional[QueueTarget]:
        self._last_queue_selection_error = ""
        has_winner = bool(str(state.get("best_run_name") or "").strip()) or (_to_float(state.get("best_score")) is not None)
        if (not has_winner) and self.baseline_queue.exists() and not self._is_demo_queue(self.baseline_queue):
            if self._is_ignored_queue(self.baseline_queue, state):
                self.log(
                    "autonomous: baseline queue is ignored "
                    f"{_safe_rel(self.baseline_queue, self.app_root)}"
                )
            else:
                ready = self._count_ready_rows(self.baseline_queue)
                if ready > 0:
                    return QueueTarget(
                        run_group=self.baseline_queue.parent.name,
                        queue_path=self.baseline_queue,
                        source="baseline_mini_first",
                        ready_rows=ready,
                    )
                self.log(
                    "autonomous: baseline queue has no ready rows "
                    f"{_safe_rel(self.baseline_queue, self.app_root)}"
                )

        ready_queues = self._prioritize_queues(self._discover_ready_queues())
        ready_queues = [item for item in ready_queues if not self._is_ignored_queue(item.queue_path, state)]

        codex_pick = self._select_queue_with_codex(ready_queues, state)
        if codex_pick is not None:
            return codex_pick

        ensured = self._select_queue_via_ensure_next_batch(state)
        if ensured is not None:
            return ensured

        if ready_queues:
            top = ready_queues[0]
            top.source = "latest_ready_fallback"
            return top

        if not self._last_queue_selection_error:
            self._last_queue_selection_error = "NO_VALID_QUEUES"
        return None

    def _iteration_log_path(self, run_group: str, iteration: int) -> Path:
        safe_group = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(run_group))
        path = (
            self.app_root
            / "artifacts"
            / "optimization_state"
            / "iterations"
            / f"iter_{int(iteration) + 1}_{safe_group}.log"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _append_iteration_log(self, path: Path, title: str, content: str) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{_utc_now()}] {title}\n")
            handle.write(content.rstrip() + "\n")

    def _run_subprocess(
        self,
        *,
        cmd: list[str],
        cwd: Path,
        iteration_log: Path,
        env: Optional[Dict[str, str]] = None,
    ) -> subprocess.CompletedProcess[str]:
        self._append_iteration_log(iteration_log, "CMD", " ".join(cmd))
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        stdout_text = proc.stdout or ""
        stderr_text = proc.stderr or ""
        self._append_iteration_log(iteration_log, f"RC={proc.returncode}", "")
        if stdout_text.strip():
            self._append_iteration_log(iteration_log, "STDOUT", stdout_text)
        if stderr_text.strip():
            self._append_iteration_log(iteration_log, "STDERR", stderr_text)
        return proc

    def _run_powered_queue(self, queue: QueueTarget, iteration_log: Path) -> int:
        queue_rel = _safe_rel(queue.queue_path, self.app_root)
        parallel_env = str(os.environ.get("COINT4_COMPUTE_PARALLEL", "auto") or "").strip()
        executor_statuses = "auto"
        watchdog_raw = str(os.environ.get("COINT4_WATCHDOG", "true") or "").strip()
        watchdog_enabled = _to_bool(watchdog_raw) if watchdog_raw else True
        stale_raw = str(os.environ.get("COINT4_WATCHDOG_STALE_SEC", "900") or "").strip()
        try:
            watchdog_stale_sec = int(stale_raw)
        except (TypeError, ValueError):
            watchdog_stale_sec = 900
        if watchdog_stale_sec <= 0:
            watchdog_stale_sec = 900
        if not parallel_env:
            parallel_env = "auto"
        if parallel_env.lower() != "auto":
            try:
                parsed_parallel = int(parallel_env)
            except (TypeError, ValueError):
                parallel_env = "auto"
            else:
                if parsed_parallel <= 0:
                    parallel_env = "auto"
                else:
                    parallel_env = str(parsed_parallel)
        self.log(
            "autonomous: executor_statuses={statuses} compute_parallel={parallel} watchdog={watchdog} stale_sec={stale}".format(
                statuses=executor_statuses,
                parallel=parallel_env,
                watchdog=str(bool(watchdog_enabled)).lower(),
                stale=int(watchdog_stale_sec),
            )
        )
        cmd = [
            str(self.python_exec),
            str(self.powered_runner),
            "--queue",
            queue_rel,
            "--statuses",
            executor_statuses,
            "--parallel",
            parallel_env,
            "--compute-host",
            "85.198.90.128",
            "--watchdog",
            "true" if bool(watchdog_enabled) else "false",
            "--watchdog-stale-sec",
            str(int(watchdog_stale_sec)),
            "--remote-repo",
            "auto",
            "--sync-inputs",
            "true",
            "--sync-configs-bulk",
            "true",
            "--preflight",
            "true",
            "--bootstrap-repo",
            "true",
            "--bootstrap-remote-dir",
            "/opt/coint4",
            "--bootstrap-venv",
            "true",
            "--poweroff",
            "true",
            "--postprocess",
            "true",
            "--wait-completion",
            "true",
            "--wait-timeout-sec",
            str(int(self.args.wait_timeout_sec)),
            "--wait-poll-sec",
            str(int(self.args.wait_poll_sec)),
        ]
        proc = self._run_subprocess(
            cmd=cmd,
            cwd=self.repo_root,
            env=self._env(),
            iteration_log=iteration_log,
        )
        return int(proc.returncode)

    def _rebuild_rollup(self, iteration_log: Path) -> int:
        cmd = [
            str(self.python_exec),
            "scripts/optimization/build_run_index.py",
            "--output-dir",
            "artifacts/wfa/aggregate/rollup",
        ]
        proc = self._run_subprocess(cmd=cmd, cwd=self.app_root, env=self._env(), iteration_log=iteration_log)
        return int(proc.returncode)

    def _parse_ranker_table(self, text: str) -> Optional[RankResult]:
        for line in (text or "").splitlines():
            stripped = line.strip()
            if not stripped.startswith("|"):
                continue
            cols = [part.strip() for part in stripped.strip("|").split("|")]
            if len(cols) < 9:
                continue
            if cols[0] != "1":
                continue
            score = _to_float(cols[1])
            worst_dd = _to_float(cols[3])
            run_name = str(cols[7]).strip()
            cfg = str(cols[8]).strip()
            if score is None or not run_name:
                continue
            return RankResult(
                ok=True,
                source="rank_multiwindow_robust_runs.py",
                score=score,
                worst_robust_sharpe=score,
                worst_dd_pct=worst_dd,
                run_name=run_name,
                config_path=cfg,
                details="parsed ranker markdown row",
            )
        return None

    def _fallback_rank_from_run_index(self, run_group: str) -> RankResult:
        if not self.rollup_csv.exists():
            return RankResult(
                ok=False,
                source="fallback_run_index_sharpe",
                score=None,
                worst_robust_sharpe=None,
                worst_dd_pct=None,
                run_name="",
                config_path="",
                details="run_index.csv not found",
            )
        queue_filters = {
            "run_ids": set(),
            "results_dirs": set(),
            "config_paths": set(),
        }
        queue_path = getattr(self, "_current_queue_for_ranking", None)
        if isinstance(queue_path, Path) and queue_path.exists():
            try:
                for row in self._read_queue_rows(queue_path):
                    run_id = str(
                        row.get("run_name")
                        or row.get("run_id")
                        or row.get("name")
                        or ""
                    ).strip()
                    if run_id:
                        queue_filters["run_ids"].add(run_id)
                    results_dir = str(row.get("results_dir") or "").strip().replace("\\", "/")
                    if results_dir:
                        queue_filters["results_dirs"].add(results_dir)
                    config_path = str(row.get("config_path") or "").strip().replace("\\", "/")
                    if config_path:
                        queue_filters["config_paths"].add(config_path)
            except Exception:  # noqa: BLE001
                pass

        def _is_candidate(row: dict[str, str]) -> bool:
            run_id = str(row.get("run_id") or "").strip()
            results_dir = str(row.get("results_dir") or "").strip().replace("\\", "/")
            config_path = str(row.get("config_path") or "").strip().replace("\\", "/")
            if queue_filters["run_ids"] and run_id in queue_filters["run_ids"]:
                return True
            if queue_filters["results_dirs"] and results_dir in queue_filters["results_dirs"]:
                return True
            if queue_filters["config_paths"] and config_path in queue_filters["config_paths"]:
                return True
            if not any(queue_filters.values()):
                return str(row.get("run_group") or "").strip() == run_group
            return False

        best_row: Optional[dict[str, str]] = None
        best_score: Optional[float] = None
        with self.rollup_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not _is_candidate(row):
                    continue
                if str(row.get("status") or "").strip().lower() != "completed":
                    continue
                metrics_present = _to_bool(row.get("metrics_present"))
                if not metrics_present:
                    continue
                sharpe = _to_float(row.get("sharpe_ratio_abs"))
                if sharpe is None:
                    continue
                if best_score is None or sharpe > best_score:
                    best_score = sharpe
                    best_row = row
        if best_row is None:
            return RankResult(
                ok=False,
                source="fallback_run_index_sharpe",
                score=None,
                worst_robust_sharpe=None,
                worst_dd_pct=None,
                run_name="",
                config_path="",
                details="no completed metrics rows for run_group",
            )
        return RankResult(
            ok=True,
            source="fallback_run_index_sharpe",
            score=best_score,
            worst_robust_sharpe=best_score,
            worst_dd_pct=_to_float(best_row.get("max_drawdown_on_equity")),
            run_name=str(best_row.get("run_id") or "").strip(),
            config_path=str(best_row.get("config_path") or "").strip(),
            details="fallback by max sharpe_ratio_abs in run_index",
        )

    def _rank_result_path(self, run_group: str) -> Path:
        return self.rank_results_dir / f"{run_group}_latest.json"

    def _rank_from_remote_result(self, run_group: str) -> RankResult:
        path = self._rank_result_path(run_group)
        if not path.exists():
            return RankResult(
                ok=False,
                source="remote_rank_result",
                score=None,
                worst_robust_sharpe=None,
                worst_dd_pct=None,
                run_name="",
                config_path="",
                details="RANK_RESULT_MISSING",
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            return RankResult(
                ok=False,
                source="remote_rank_result",
                score=None,
                worst_robust_sharpe=None,
                worst_dd_pct=None,
                run_name="",
                config_path="",
                details=f"RANK_RESULT_INVALID_JSON:{type(exc).__name__}",
            )
        if not isinstance(payload, dict):
            return RankResult(
                ok=False,
                source="remote_rank_result",
                score=None,
                worst_robust_sharpe=None,
                worst_dd_pct=None,
                run_name="",
                config_path="",
                details="RANK_RESULT_INVALID_PAYLOAD",
            )

        score = _to_float(payload.get("best_score"))
        worst_robust = _to_float(payload.get("worst_robust_sharpe"))
        if score is None:
            score = worst_robust
        if worst_robust is None:
            worst_robust = score
        rank = RankResult(
            ok=bool(payload.get("ok")),
            source="remote_rank_result",
            score=score,
            worst_robust_sharpe=worst_robust,
            worst_dd_pct=_to_float(payload.get("worst_dd_pct")),
            run_name=str(payload.get("best_run_name") or "").strip(),
            config_path=str(payload.get("best_config_path") or "").strip(),
            details=str(payload.get("details") or "").strip(),
        )
        if rank.ok and rank.score is not None and rank.run_name:
            return rank
        return RankResult(
            ok=False,
            source=rank.source,
            score=None,
            worst_robust_sharpe=rank.worst_robust_sharpe,
            worst_dd_pct=rank.worst_dd_pct,
            run_name=rank.run_name,
            config_path=rank.config_path,
            details=rank.details or "RANK_RESULT_NOT_READY",
        )

    def _latest_powered_fail_reason(self, run_group: str) -> str:
        logs_dir = self.app_root / "artifacts" / "wfa" / "aggregate" / run_group / "logs"
        if not logs_dir.exists():
            return "POWERED_LOG_NOT_FOUND"
        powered_logs = sorted(logs_dir.glob("powered*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not powered_logs:
            return "POWERED_LOG_NOT_FOUND"
        latest = powered_logs[0]
        try:
            text = latest.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return "POWERED_LOG_READ_ERROR"
        reason = ""
        for line in text.splitlines():
            marker = "powered: FAIL reason="
            if marker in line:
                reason = line.split(marker, 1)[1].strip()
        if reason:
            return reason
        return "POWERED_RC_NONZERO_NO_REASON"

    def rank_run_group(self, run_group: str, iteration_log: Path) -> RankResult:
        cmd = [
            str(self.python_exec),
            "scripts/optimization/rank_multiwindow_robust_runs.py",
            "--run-index",
            "artifacts/wfa/aggregate/rollup/run_index.csv",
            "--contains",
            run_group,
            "--top",
            "1",
            "--min-windows",
            str(int(self.stop_policy["min_windows"])),
            "--min-trades",
            str(int(self.stop_policy["min_trades"])),
            "--min-pairs",
            str(int(self.stop_policy["min_pairs"])),
            "--max-dd-pct",
            str(float(self.stop_policy["max_dd_pct"])),
        ]
        proc = self._run_subprocess(cmd=cmd, cwd=self.app_root, env=self._env(), iteration_log=iteration_log)
        if proc.returncode == 0:
            parsed = self._parse_ranker_table(proc.stdout or "")
            if parsed is not None:
                return parsed
        self.log(
            f"autonomous: ranker fallback for run_group={run_group} rc={proc.returncode}"
        )
        return self._fallback_rank_from_run_index(run_group)

    def _run_index_rows(self) -> list[dict[str, str]]:
        if not self.rollup_csv.exists():
            return []
        with self.rollup_csv.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    def _has_completed_metrics_for_queue(self, queue_path: Path) -> tuple[bool, str]:
        try:
            queue_rows = self._read_queue_rows(queue_path)
        except Exception as exc:  # noqa: BLE001
            return False, f"queue_read_error:{type(exc).__name__}"

        run_ids: set[str] = set()
        results_dirs: set[str] = set()
        config_paths: set[str] = set()
        for row in queue_rows:
            run_id = str(
                row.get("run_name")
                or row.get("run_id")
                or row.get("name")
                or ""
            ).strip()
            if run_id:
                run_ids.add(run_id)
            results_dir = str(row.get("results_dir") or "").strip().replace("\\", "/")
            if results_dir:
                results_dirs.add(results_dir)
            config_path = str(row.get("config_path") or "").strip().replace("\\", "/")
            if config_path:
                config_paths.add(config_path)

        if not (run_ids or results_dirs or config_paths):
            return False, "queue_has_no_match_keys"

        matched = 0
        completed = 0
        for row in self._run_index_rows():
            run_id = str(row.get("run_id") or "").strip()
            results_dir = str(row.get("results_dir") or "").strip().replace("\\", "/")
            config_path = str(row.get("config_path") or "").strip().replace("\\", "/")
            is_match = (
                (run_id in run_ids if run_ids else False)
                or (results_dir in results_dirs if results_dirs else False)
                or (config_path in config_paths if config_paths else False)
            )
            if not is_match:
                continue
            matched += 1
            status = str(row.get("status") or "").strip().lower()
            metrics_present = _to_bool(row.get("metrics_present"))
            if status == "completed" and metrics_present:
                completed += 1

        if completed > 0:
            return True, f"completed_candidates={completed}"
        return False, f"matched={matched},completed=0"

    def _resolve_best_config_source(self, state: Dict[str, Any]) -> Optional[Path]:
        run_group = str(state.get("current_run_group") or "").strip()
        run_name = str(state.get("best_run_name") or "").strip()
        cfg = str(state.get("best_config_path") or "").strip()

        rows = self._run_index_rows()
        for row in rows:
            if run_group and str(row.get("run_group") or "").strip() != run_group:
                continue
            if run_name and str(row.get("run_id") or "").strip() != run_name:
                continue
            results_dir = str(row.get("results_dir") or "").strip()
            if results_dir:
                run_dir = self.app_root / results_dir
                snapshot = run_dir / "config_snapshot.yaml"
                if snapshot.exists():
                    return snapshot
        if cfg:
            cfg_path = Path(cfg)
            if not cfg_path.is_absolute():
                cfg_path = self.app_root / cfg_path
            if cfg_path.exists():
                return cfg_path
        return None

    def write_best_params(self, state: Dict[str, Any]) -> None:
        source = self._resolve_best_config_source(state)
        if source is None:
            placeholder = (
                "# No winner yet\n"
                f"updated_utc: {_utc_now()}\n"
                f"status: {state.get('status')}\n"
                "note: winner config will be written after first successful ranking cycle.\n"
            )
            self.best_params_path.write_text(placeholder, encoding="utf-8")
            self.log("autonomous: wrote placeholder best params (winner not available yet)")
            return
        self.best_params_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        self.log(f"autonomous: wrote best params {self.best_params_path}")

    def _deterministic_report(self, state: Dict[str, Any], *, defaults_used: list[str]) -> str:
        progress = self._collect_decision_evidence(state, row_limit=200)
        lines = []
        lines.append("# Autonomous Optimization Final Report")
        lines.append("")
        lines.append(f"- Updated (UTC): {_utc_now()}")
        lines.append(f"- Status: {state.get('status')}")
        lines.append(f"- Iterations executed: {state.get('iteration')}")
        lines.append(f"- Current run_group: {state.get('current_run_group')}")
        lines.append(f"- Last error: {state.get('last_error') or 'none'}")
        lines.append(f"- Last decision id: {state.get('last_decision_id') or 'n/a'}")
        lines.append(f"- Last decision action: {state.get('last_decision_action') or 'n/a'}")
        lines.append(f"- Best run: {state.get('best_run_name') or 'n/a'}")
        lines.append(f"- Best config path: {state.get('best_config_path') or 'n/a'}")
        best_score = state.get("best_score")
        lines.append(f"- Best score (rank objective): {best_score if best_score is not None else 'n/a'}")
        lines.append("")
        lines.append("## Objective and gates")
        lines.append("- Ranker: `scripts/optimization/rank_multiwindow_robust_runs.py`")
        lines.append("- Score: `worst_robust_sharpe` (as implemented in ranker)")
        lines.append(f"- Gates: min_windows={self.stop_policy['min_windows']}, min_trades={self.stop_policy['min_trades']}, min_pairs={self.stop_policy['min_pairs']}, max_dd_pct={self.stop_policy['max_dd_pct']}")
        lines.append("")
        lines.append("## Stop criteria")
        lines.append(f"- max_rounds={self.stop_policy['max_rounds']}")
        lines.append(f"- no_improvement_rounds={self.stop_policy['no_improvement_rounds']}")
        lines.append(f"- min_improvement={self.stop_policy['min_improvement']}")
        lines.append(f"- require_all_knobs_before_stop={self.stop_policy['require_all_knobs_before_stop']}")
        lines.append("")
        if defaults_used:
            lines.append("## Defaults")
            lines.append(
                "- Used defaults for: " + ", ".join(defaults_used)
            )
            lines.append("- Source config missing/partial; defaults were applied safely.")
            lines.append("")
        lines.append("## Canonical sources")
        lines.append("- `coint4/artifacts/wfa/aggregate/rollup/run_index.csv`")
        lines.append("- `canonical_metrics.json` in each completed run dir")
        lines.append("- `docs/best_params_latest.yaml`")
        lines.append("")
        status_counts = progress.get("status_counts") if isinstance(progress.get("status_counts"), dict) else {}
        lines.append("## Current Progress Snapshot")
        lines.append(f"- Run group: {progress.get('current_run_group') or 'n/a'}")
        lines.append(f"- Queue: {progress.get('current_queue_path') or 'n/a'}")
        lines.append(
            "- Status counts: planned={planned}, running={running}, stalled={stalled}, completed={completed}".format(
                planned=int(status_counts.get("planned", 0) or 0),
                running=int(status_counts.get("running", 0) or 0),
                stalled=int(status_counts.get("stalled", 0) or 0),
                completed=int(progress.get("completed_count", 0) or 0),
            )
        )
        lines.append(
            "- completed metrics_present=True: {value}".format(
                value=int(progress.get("metrics_present_true_completed", 0) or 0)
            )
        )
        lines.append(
            "- threshold (completed for next batch): {value}".format(
                value=int(progress.get("completed_threshold_for_next_batch", 0) or 0)
            )
        )
        missing_reasons = list(progress.get("blocking_missing_reasons") or progress.get("computed_missing_reasons") or [])
        lines.append(
            "- missing reasons: {value}".format(
                value=", ".join(str(x).strip() for x in missing_reasons if str(x).strip()) or "none"
            )
        )
        lines.append(f"- decision_data_ready: {bool(progress.get('decision_data_ready'))}")
        lines.append("")
        explanation = str(state.get("last_decision_explanation_md") or "").strip()
        if explanation:
            lines.append("## Last Codex Explanation")
            lines.append(explanation)
            lines.append("")
        return "\n".join(lines)

    def write_final_report(self, state: Dict[str, Any]) -> None:
        defaults_used = list(self.stop_policy.get("defaults_used") or [])
        if bool(self.args.use_codex_exec):
            schema = {
                "type": "object",
                "properties": {"markdown": {"type": "string"}},
                "required": ["markdown"],
                "additionalProperties": False,
            }
            prompt = (
                "Сформируй краткий markdown final report для автономного оптимизатора.\n"
                f"State JSON: {json.dumps(state, ensure_ascii=False)}\n"
                f"Policy JSON: {json.dumps(self.stop_policy, ensure_ascii=False)}\n"
                "Укажи objective/gates/stop criteria и winner."
            )
            payload = self._run_codex_json(prompt=prompt, schema=schema, timeout_sec=180)
            if payload and str(payload.get("markdown") or "").strip():
                self.final_report_path.write_text(str(payload["markdown"]).strip() + "\n", encoding="utf-8")
                self.log(f"autonomous: wrote final report via codex {self.final_report_path}")
                return

        body = self._deterministic_report(state, defaults_used=defaults_used)
        self.final_report_path.write_text(body, encoding="utf-8")
        self.log(f"autonomous: wrote final report {self.final_report_path}")

    def _improved(self, *, current: Optional[float], best: Optional[float]) -> bool:
        if current is None:
            return False
        if best is None:
            return True
        threshold = float(self.stop_policy["min_improvement"])
        return current > (best + threshold)

    def _effective_max_iterations(self) -> int:
        cfg_max = int(self.stop_policy["max_rounds"])
        if self.args.max_iterations is None:
            return cfg_max
        return max(1, int(self.args.max_iterations))

    def _should_stop(self, state: Dict[str, Any]) -> tuple[bool, str]:
        # Stop criteria are delegated to Codex decisions (stop=true).
        _ = state
        return False, ""

    def _apply_decision_state(self, state: Dict[str, Any], decision: CodexDecision) -> None:
        payload = decision.payload
        state["last_decision_id"] = str(payload.get("decision_id") or "").strip()
        state["last_decision_action"] = str(payload.get("next_action") or "").strip()
        state["last_decision_explanation_md"] = str(payload.get("human_explanation_md") or "").strip()

    def _wait_and_exit(
        self,
        *,
        state: Dict[str, Any],
        reason: str,
        phase: str,
    ) -> Dict[str, Any]:
        if phase.startswith("waiting") or phase == "skipping_queue":
            self.log_progress_snapshot(state, phase=f"wait:{phase}")
        state["status"] = "running"
        state["last_error"] = reason
        state["last_iteration_phase"] = phase
        self.save_state(state)
        self.write_best_params(state)
        self.write_final_report(state)
        self._exit_after_iteration_wait = True
        return state

    def run_iteration(self, *, state: Dict[str, Any]) -> Dict[str, Any]:
        self._exit_after_iteration_wait = False
        iteration = int(state.get("iteration") or 0)

        decision = self.decide_with_codex(state)
        if decision is None:
            wait_reason = str(self._last_wait_reason or "WAITING_CODEX_OR_DATA").strip()
            self.log(f"autonomous: codex decision unavailable ({wait_reason})")
            return self._wait_and_exit(state=state, reason=wait_reason, phase="waiting_codex")

        self._apply_decision_state(state, decision)
        payload = decision.payload
        next_action = str(payload.get("next_action") or "").strip()
        stop = bool(payload.get("stop"))

        if stop or next_action == "stop":
            state["status"] = "done"
            state["last_error"] = str(payload.get("stop_reason") or "").strip()
            state["last_iteration_phase"] = "stopped_by_codex"
            self.save_state(state)
            self.write_best_params(state)
            self.write_final_report(state)
            return state

        if next_action == "wait":
            wait_reason = str(payload.get("stop_reason") or "").strip() or "WAITING_CODEX_OR_DATA"
            evidence = self._collect_decision_evidence(state, row_limit=200)
            blocking_reasons = {
                str(item).strip().upper()
                for item in list(evidence.get("blocking_missing_reasons") or [])
                if str(item).strip()
            }
            if "NO_COMPLETED_RUNS" in blocking_reasons:
                queue_target = self._queue_target_for_run_group(
                    str(state.get("current_run_group") or "").strip(),
                    source="wait_data_collection",
                )
                if queue_target is not None and queue_target.ready_rows > 0:
                    if not self._is_powered_runner_active_for_queue(queue_target.queue_path):
                        if not str(state.get("iteration_started_utc") or "").strip():
                            state["iteration_started_utc"] = _utc_now()
                        iteration_log = self._iteration_log_path(
                            queue_target.run_group,
                            int(state.get("iteration") or 0),
                        )
                        self.log(
                            "autonomous: launched data-collection run for queue because NO_COMPLETED_RUNS"
                        )
                        self.log_progress_snapshot(state, phase="pre-data-collection", min_interval_sec=1)
                        rc_data_collection = self._run_powered_queue(
                            queue_target,
                            iteration_log=iteration_log,
                        )
                        self.log_progress_snapshot(state, phase="post-data-collection", min_interval_sec=1)
                        if rc_data_collection != 0:
                            fail_reason = self._latest_powered_fail_reason(queue_target.run_group)
                            fail_reason_norm = str(fail_reason or "").strip().upper() or "UNKNOWN"
                            return self._wait_and_exit(
                                state=state,
                                reason=f"POWERED_WAIT:{fail_reason_norm}",
                                phase="waiting_powered",
                            )
                    else:
                        self.log(
                            "autonomous: data-collection run already active for queue "
                            f"{_safe_rel(queue_target.queue_path, self.app_root)}"
                        )
            return self._wait_and_exit(state=state, reason=wait_reason, phase="waiting_codex")

        if next_action != "run_next_batch":
            return self._wait_and_exit(
                state=state,
                reason=f"INVALID_CODEX_ACTION:{next_action or 'EMPTY'}",
                phase="waiting_codex",
            )

        try:
            self._apply_file_edits(decision)
            queue = self._materialize_queue_from_decision(decision)
        except Exception as exc:  # noqa: BLE001
            return self._wait_and_exit(
                state=state,
                reason=f"DECISION_APPLY_ERROR:{type(exc).__name__}",
                phase="waiting_codex",
            )

        if self._is_demo_queue(queue.queue_path):
            self._record_ignored_queue(state, queue)
            return self._wait_and_exit(
                state=state,
                reason="SKIP_QUEUE:DEMO_QUEUE",
                phase="skipping_queue",
            )
        if self._is_ignored_queue(queue.queue_path, state):
            return self._wait_and_exit(
                state=state,
                reason="SKIP_QUEUE:IGNORED_QUEUE",
                phase="skipping_queue",
            )
        if queue.ready_rows < 1:
            return self._wait_and_exit(
                state=state,
                reason="NO_VALID_QUEUES",
                phase="waiting_queue",
            )

        state["iteration"] = iteration + 1
        state["current_run_group"] = queue.run_group
        state["iteration_started_utc"] = _utc_now()
        state["status"] = "running"
        state["last_error"] = ""
        state["last_iteration_phase"] = "started"
        self.save_state(state)
        self.write_final_report(state)

        iteration_log = self._iteration_log_path(queue.run_group, iteration)
        self.log(
            f"autonomous: iteration={iteration + 1} queue={_safe_rel(queue.queue_path, self.app_root)} source={queue.source} ready={queue.ready_rows}"
        )
        self.log_progress_snapshot(state, phase="pre-run")

        if bool(self.args.plan_only):
            return self._wait_and_exit(state=state, reason="PLAN_ONLY_MODE", phase="planned")

        rc_powered = self._run_powered_queue(queue, iteration_log=iteration_log)
        self.log_progress_snapshot(state, phase="post-run")
        if rc_powered != 0:
            fail_reason = self._latest_powered_fail_reason(queue.run_group)
            fail_reason_norm = str(fail_reason or "").strip().upper() or "UNKNOWN"
            if fail_reason_norm in {
                "LOCAL_CONFIG_MISSING",
                "QUEUE_MISSING",
                "QUEUE_PARSE_ERROR",
                "QUEUE_REL_EMPTY",
            } or fail_reason_norm.startswith("QUEUE_"):
                self._record_ignored_queue(state, queue)
                return self._wait_and_exit(
                    state=state,
                    reason=f"SKIP_QUEUE:{fail_reason_norm}",
                    phase="skipping_queue",
                )
            return self._wait_and_exit(
                state=state,
                reason=f"POWERED_WAIT:{fail_reason_norm}",
                phase="waiting_powered",
            )

        if bool(self.args.local_rollup_rebuild):
            rc_rollup = self._rebuild_rollup(iteration_log=iteration_log)
            if rc_rollup != 0:
                return self._wait_and_exit(
                    state=state,
                    reason=f"BUILD_RUN_INDEX_RC{rc_rollup}",
                    phase="waiting_rollup",
                )
        else:
            self.log("autonomous: skip local build_run_index (using powered sync-back rollup)")

        follow_decision = self.decide_with_codex(state)
        if follow_decision is None:
            wait_reason = str(self._last_wait_reason or "WAITING_CODEX_OR_DATA").strip()
            return self._wait_and_exit(state=state, reason=wait_reason, phase="waiting_codex")

        self._apply_decision_state(state, follow_decision)
        follow_payload = follow_decision.payload
        follow_action = str(follow_payload.get("next_action") or "").strip()
        if bool(follow_payload.get("stop")) or follow_action == "stop":
            state["status"] = "done"
            state["last_error"] = str(follow_payload.get("stop_reason") or "").strip()
            state["last_iteration_phase"] = "stopped_by_codex"
            self.save_state(state)
            self.write_best_params(state)
            self.write_final_report(state)
            return state
        if follow_action == "wait":
            wait_reason = str(follow_payload.get("stop_reason") or "").strip() or "WAITING_CODEX_OR_DATA"
            return self._wait_and_exit(state=state, reason=wait_reason, phase="waiting_codex")

        rank = self._rank_from_remote_result(queue.run_group)
        if rank.ok and rank.score is not None:
            prev_best = _to_float(state.get("best_score"))
            if self._improved(current=rank.score, best=prev_best):
                state["best_score"] = rank.score
                state["best_run_name"] = rank.run_name
                state["best_config_path"] = rank.config_path
                state["no_improvement_streak"] = 0
            else:
                state["no_improvement_streak"] = int(state.get("no_improvement_streak") or 0) + 1

        state["status"] = "running"
        state["last_error"] = "NEXT_BATCH_READY_FROM_CODEX"
        state["last_iteration_phase"] = "rank_ok"
        self.save_state(state)
        self.write_best_params(state)
        self.write_final_report(state)
        self._exit_after_iteration_wait = True
        return state

    def run(self) -> int:
        state = self.load_state()
        self._quarantine_demo_queues()
        self.log(
            f"autonomous: start once={bool(self.args.once)} until_done={bool(self.args.until_done)} "
            f"max_iterations={self.args.max_iterations} plan_only={bool(self.args.plan_only)} "
            f"wait_timeout_sec={int(self.args.wait_timeout_sec)} wait_poll_sec={int(self.args.wait_poll_sec)} "
            f"local_rollup_rebuild={bool(self.args.local_rollup_rebuild)}"
        )

        if state.get("status") == "done" and not bool(self.args.once):
            self.log("autonomous: state is done; nothing to do")
            return 0

        loop_forever = bool(self.args.until_done) and not bool(self.args.once)
        if not bool(self.args.until_done) and not bool(self.args.once):
            loop_forever = True

        try:
            while True:
                stop, reason = self._should_stop(state)
                if stop and not bool(self.args.plan_only):
                    state["status"] = "done"
                    self.save_state(state)
                    self.write_best_params(state)
                    self.write_final_report(state)
                    self.log(f"autonomous: stop ({reason})")
                    return 0

                state = self.run_iteration(state=state)
                if str(state.get("status") or "").strip().lower() == "done":
                    self.log("autonomous: done by codex decision")
                    return 0
                if self._exit_after_iteration_wait:
                    self.log("autonomous: wait mode, exit 0 and continue on next timer/service run")
                    return 0
                if bool(self.args.once):
                    self.log("autonomous: once mode complete")
                    return 0
                if bool(self.args.plan_only):
                    self.log("autonomous: plan-only complete")
                    return 0
                if not loop_forever:
                    self.log("autonomous: single pass complete")
                    return 0
        except Exception as exc:  # noqa: BLE001
            state["status"] = "running"
            state["last_error"] = f"{type(exc).__name__}: {exc}"
            self.save_state(state)
            self.write_best_params(state)
            self.write_final_report(state)
            self.log(f"autonomous: FAIL {type(exc).__name__}: {exc}")
            with self.main_log_path.open("a", encoding="utf-8") as handle:
                traceback.print_exc(file=handle)
            return 1


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous optimization orchestrator")
    parser.add_argument("--once", action="store_true", help="Run only one iteration.")
    parser.add_argument("--until-done", action="store_true", help="Run iterations until stop criteria.")
    parser.add_argument("--max-iterations", type=int, default=None, help="Override max iterations.")
    parser.add_argument("--use-codex-exec", action="store_true", help="Enable optional codex exec helpers.")
    parser.add_argument("--codex-model", default="", help="Optional model override for codex exec.")
    parser.add_argument("--plan-only", action="store_true", help="Prepare/plan one iteration without powered run.")
    parser.add_argument(
        "--wait-timeout-sec",
        type=int,
        default=21600,
        help="Timeout (seconds) passed to powered runner wait-completion.",
    )
    parser.add_argument(
        "--wait-poll-sec",
        type=int,
        default=60,
        help="Polling interval (seconds) passed to powered runner wait-completion.",
    )
    parser.add_argument(
        "--local-rollup-rebuild",
        action="store_true",
        help="Rebuild run_index locally after powered runner (default: disabled, rely on powered sync-back).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    runner = AutonomousOptimizer(args)
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
