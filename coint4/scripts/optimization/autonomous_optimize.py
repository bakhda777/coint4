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
import time
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


@dataclass
class EvolutionPlan:
    queue: QueueTarget
    controller_group: str
    run_prefix: str
    decision_path: Path


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
        self.evolve_next_batch = self.app_root / "scripts" / "optimization" / "evolve_next_batch.py"
        self.reflect_next_action = self.app_root / "scripts" / "optimization" / "reflect_next_action.py"
        self.build_factor_pool = self.app_root / "scripts" / "optimization" / "build_factor_pool.py"
        self.codex_schema_path = (
            self.app_root
            / "scripts"
            / "optimization"
            / "schemas"
            / "autopilot_decision.schema.json"
        )
        self._current_queue_for_ranking: Optional[Path] = None
        self._exit_after_iteration_wait: bool = False
        self._next_iteration_delay_sec: int = max(1, int(getattr(self.args, "wait_poll_sec", 60) or 60))
        self._last_queue_selection_error: str = ""
        self._last_wait_reason: str = ""
        self._codex_auth_checked: bool = False
        self._codex_auth_ready: bool = False
        self._codex_auth_reason: str = ""
        self._codex_exec_home: str = str(os.environ.get("HOME") or "").strip()

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
            "min_pnl": 0.0,
            "min_psr": None,
            "min_dsr": None,
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
        min_pnl = _to_float(selection.get("min_pnl"))
        if min_pnl is None:
            defaults_used.append("min_pnl")
        else:
            policy["min_pnl"] = float(min_pnl)

        min_psr = _to_float(selection.get("min_psr"))
        if min_psr is None:
            defaults_used.append("min_psr")
        else:
            policy["min_psr"] = float(min_psr)

        min_dsr = _to_float(selection.get("min_dsr"))
        if min_dsr is None:
            defaults_used.append("min_dsr")
        else:
            policy["min_dsr"] = float(min_dsr)

        policy["defaults_used"] = sorted(set(defaults_used))
        policy["source"] = _safe_rel(self.bridge11_path, self.app_root) if self.bridge11_path.exists() else "defaults"
        return policy

    def _fixed_walk_forward_period(self) -> tuple[Optional[str], Optional[str]]:
        payload = _read_yaml(self.bridge11_path)
        windows = payload.get("windows") if isinstance(payload.get("windows"), list) else []
        starts: list[str] = []
        ends: list[str] = []
        for item in windows:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            start = str(item[0] or "").strip()
            end = str(item[1] or "").strip()
            if start and end:
                starts.append(start)
                ends.append(end)
        if starts and ends:
            return min(starts), max(ends)

        base_config_rel = str(payload.get("base_config") or "").strip()
        if base_config_rel:
            try:
                base_config_path = self._resolve_app_path(base_config_rel)
            except Exception:  # noqa: BLE001
                base_config_path = None
            if base_config_path is not None:
                base_payload = _read_yaml(base_config_path)
                walk_forward = base_payload.get("walk_forward") if isinstance(base_payload.get("walk_forward"), dict) else {}
                start = str(walk_forward.get("start_date") or "").strip()
                end = str(walk_forward.get("end_date") or "").strip()
                if start and end:
                    return start, end

        return None, None

    def _apply_fixed_walk_forward_period(self, cfg_payload: Dict[str, Any]) -> None:
        fixed_start, fixed_end = self._fixed_walk_forward_period()
        if not fixed_start or not fixed_end:
            return
        walk_forward = cfg_payload.get("walk_forward")
        if not isinstance(walk_forward, dict):
            walk_forward = {}
            cfg_payload["walk_forward"] = walk_forward
        walk_forward["start_date"] = fixed_start
        walk_forward["end_date"] = fixed_end

    def _planner_mode(self) -> str:
        raw_mode = str(getattr(self.args, "planner_mode", "") or "").strip().lower()
        if raw_mode in {"legacy", "evolution"}:
            return raw_mode
        return "legacy"

    def _bridge11_payload(self) -> Dict[str, Any]:
        payload = _read_yaml(self.bridge11_path)
        if isinstance(payload, dict):
            return payload
        return {}

    def _bridge11_base_config_rel(self) -> str:
        payload = self._bridge11_payload()
        raw_base = str(payload.get("base_config") or "").strip()
        if raw_base:
            return raw_base
        return "configs/prod_final_budget1000.yaml"

    def _bridge11_windows(self) -> list[tuple[str, str]]:
        payload = self._bridge11_payload()
        windows = payload.get("windows") if isinstance(payload.get("windows"), list) else []
        out: list[tuple[str, str]] = []
        for item in windows:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            start = str(item[0] or "").strip()
            end = str(item[1] or "").strip()
            if not start or not end:
                continue
            out.append((start, end))
        return out

    def _evolution_controller_group(self, state: Dict[str, Any]) -> str:
        existing = str(state.get("evolution_controller_group") or "").strip()
        if existing:
            return existing
        cli_value = str(getattr(self.args, "evolution_controller_group", "") or "").strip()
        if cli_value:
            state["evolution_controller_group"] = cli_value
            return cli_value
        env_value = str(os.environ.get("COINT4_EVOLUTION_CONTROLLER_GROUP") or "").strip()
        if env_value:
            state["evolution_controller_group"] = env_value
            return env_value
        fallback = f"{datetime.now(timezone.utc).strftime('%Y%m%d')}_autonomous_evolution"
        state["evolution_controller_group"] = fallback
        return fallback

    def _evolution_run_prefix(self, state: Dict[str, Any]) -> str:
        existing = str(state.get("evolution_run_prefix") or "").strip()
        if existing:
            return existing
        cli_value = str(getattr(self.args, "evolution_run_prefix", "") or "").strip()
        if cli_value:
            state["evolution_run_prefix"] = cli_value
            return cli_value
        env_value = str(os.environ.get("COINT4_EVOLUTION_RUN_PREFIX") or "").strip()
        if env_value:
            state["evolution_run_prefix"] = env_value
            return env_value
        fallback = "autonomous_evo"
        state["evolution_run_prefix"] = fallback
        return fallback

    def _next_evolution_run_group(self, state: Dict[str, Any], *, iteration: int) -> str:
        prefix = self._evolution_run_prefix(state)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_i{int(iteration) + 1:03d}_{stamp}"

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
            "planner_mode": "",
            "evolution_controller_group": "",
            "evolution_run_prefix": "",
            "last_evolution_decision_path": "",
            "last_evolution_reflection_path": "",
            "trajectory_memory": [],
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
        self._normalize_trajectory_memory(state)
        state["last_updated_utc"] = _utc_now()
        self.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _trajectory_memory_limit(self) -> int:
        value = _to_int(os.environ.get("COINT4_TRAJECTORY_MEMORY_MAX"))
        if isinstance(value, int) and value > 0:
            return min(1000, value)
        return 200

    def _normalize_trajectory_memory(self, state: Dict[str, Any]) -> None:
        values = state.get("trajectory_memory")
        if not isinstance(values, list):
            state["trajectory_memory"] = []
            return
        normalized: list[Dict[str, Any]] = []
        for item in values:
            if not isinstance(item, dict):
                continue
            action = str(item.get("action") or "").strip()
            result = str(item.get("result") or "").strip()
            reflection = str(item.get("reflection") or "").strip()
            if not (action or result or reflection):
                continue
            iteration = _to_int(item.get("iteration"))
            normalized.append(
                {
                    "ts_utc": str(item.get("ts_utc") or "").strip() or _utc_now(),
                    "iteration": int(iteration or 0),
                    "run_group": str(item.get("run_group") or "").strip(),
                    "action": action,
                    "result": result,
                    "reflection": reflection,
                }
            )
        limit = self._trajectory_memory_limit()
        if len(normalized) > limit:
            normalized = normalized[-limit:]
        state["trajectory_memory"] = normalized

    def _append_trajectory_memory(
        self,
        state: Dict[str, Any],
        *,
        action: str,
        result: str,
        reflection: str,
    ) -> None:
        values = state.get("trajectory_memory")
        if not isinstance(values, list):
            values = []
        action_text = str(action or "").strip()
        result_text = str(result or "").strip()
        reflection_text = str(reflection or "").strip()
        if not (action_text or result_text or reflection_text):
            return
        iteration = _to_int(state.get("iteration"))
        values.append(
            {
                "ts_utc": _utc_now(),
                "iteration": int(iteration or 0),
                "run_group": str(state.get("current_run_group") or "").strip(),
                "action": action_text,
                "result": result_text,
                "reflection": reflection_text,
            }
        )
        limit = self._trajectory_memory_limit()
        if len(values) > limit:
            values = values[-limit:]
        state["trajectory_memory"] = values

    def _trajectory_memory_tail(self, state: Dict[str, Any], *, limit: int = 20) -> list[Dict[str, Any]]:
        values = state.get("trajectory_memory")
        if not isinstance(values, list):
            return []
        out: list[Dict[str, Any]] = []
        for item in values:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "ts_utc": str(item.get("ts_utc") or "").strip(),
                    "iteration": int(_to_int(item.get("iteration")) or 0),
                    "run_group": str(item.get("run_group") or "").strip(),
                    "action": str(item.get("action") or "").strip(),
                    "result": str(item.get("result") or "").strip(),
                    "reflection": str(item.get("reflection") or "").strip(),
                }
            )
        if len(out) > limit:
            return out[-limit:]
        return out

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

    def _classify_codex_exec_failure(self, *, stdout: str, stderr: str) -> str:
        blob = f"{stdout}\n{stderr}".lower()
        auth_markers = (
            "missing bearer",
            "unauthorized",
            "authentication",
            "codex login",
            "login --with-api-key",
            "api key",
            "incorrect api key",
            "401",
        )
        if any(marker in blob for marker in auth_markers):
            return "CODEX_AUTH_REQUIRED"
        network_markers = (
            "stream disconnected",
            "error sending request",
            "dns error",
            "operation not permitted",
            "timed out",
            "connection reset",
            "transport channel closed",
            "network",
        )
        if any(marker in blob for marker in network_markers):
            return "CODEX_EXEC_NETWORK"
        return ""

    def _codex_auth_mode(self) -> str:
        raw_mode = str(os.environ.get("COINT4_CODEX_AUTH_MODE") or "subscription").strip().lower()
        if raw_mode in {"subscription", "chatgpt", "device-auth", "device_auth", "device"}:
            return "subscription"
        if raw_mode in {"api-key", "api_key", "apikey", "with-api-key"}:
            return "api-key"
        if raw_mode == "auto":
            return "auto"
        return "subscription"

    def _candidate_codex_homes(self) -> list[str]:
        homes: list[str] = []

        def _append(raw: Any) -> None:
            value = str(raw or "").strip()
            if not value:
                return
            value = str(Path(value).expanduser())
            if value not in homes:
                homes.append(value)

        _append(os.environ.get("HOME"))
        _append(self._codex_exec_home)
        _append(os.environ.get("COINT4_CODEX_AUTH_HOME"))
        try:
            import pwd  # Unix-only; fallback handled by exception.

            _append(pwd.getpwuid(os.getuid()).pw_dir)
        except Exception:  # noqa: BLE001
            pass
        try:
            if hasattr(os, "geteuid") and int(os.geteuid()) == 0:
                _append("/root")
        except Exception:  # noqa: BLE001
            pass
        if not homes:
            homes.append(str(Path.home()))
        return homes

    def _codex_subprocess_env(
        self,
        *,
        auth_mode: Optional[str] = None,
        home_override: Optional[str] = None,
    ) -> Dict[str, str]:
        mode = str(auth_mode or self._codex_auth_mode()).strip().lower()
        env = os.environ.copy()
        selected_home = str(home_override or self._codex_exec_home or "").strip()
        if selected_home:
            env["HOME"] = selected_home
        if mode == "subscription":
            # Force device/session auth path for subscription mode.
            env.pop("OPENAI_API_KEY", None)
        return env

    def _ensure_codex_auth_ready(self) -> bool:
        if self._codex_auth_checked:
            return self._codex_auth_ready

        self._codex_auth_checked = True
        auth_mode = self._codex_auth_mode()
        if auth_mode == "subscription" and str(os.environ.get("OPENAI_API_KEY") or "").strip():
            self.log("autonomous: subscription mode active; ignore OPENAI_API_KEY for codex exec")

        candidate_homes = self._candidate_codex_homes()
        for candidate_home in candidate_homes:
            codex_env = self._codex_subprocess_env(auth_mode=auth_mode, home_override=candidate_home)
            try:
                status_proc = subprocess.run(
                    ["codex", "login", "status"],
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=20,
                    env=codex_env,
                )
            except FileNotFoundError:
                self._codex_auth_reason = "CODEX_EXEC_UNAVAILABLE"
                self.log("autonomous: codex auth check failed (codex binary not found)")
                self._codex_auth_ready = False
                return False
            except subprocess.TimeoutExpired:
                self._codex_auth_reason = "CODEX_AUTH_STATUS_TIMEOUT"
                self.log("autonomous: codex auth status timeout")
                self._codex_auth_ready = False
                return False
            except Exception as exc:  # noqa: BLE001
                self._codex_auth_reason = f"CODEX_AUTH_STATUS_ERROR:{type(exc).__name__}"
                self.log(f"autonomous: codex auth status error={type(exc).__name__}")
                self._codex_auth_ready = False
                return False

            if status_proc.returncode == 0:
                self._codex_exec_home = str(codex_env.get("HOME") or "").strip()
                self._codex_auth_ready = True
                self._codex_auth_reason = ""
                current_home = str(os.environ.get("HOME") or "").strip()
                if self._codex_exec_home and self._codex_exec_home != current_home:
                    self.log(
                        "autonomous: codex auth ready via existing codex session "
                        f"(fallback HOME={self._codex_exec_home})"
                    )
                else:
                    self.log("autonomous: codex auth ready via existing codex session")
                return True

        if auth_mode == "subscription":
            self._codex_auth_reason = "CODEX_AUTH_REQUIRED"
            self.log(
                "autonomous: codex login required (subscription mode). "
                "Run `codex login --device-auth` under the service user."
            )
            self._codex_auth_ready = False
            return False

        api_key = str(os.environ.get("OPENAI_API_KEY") or "").strip()
        if not api_key:
            self._codex_auth_reason = "CODEX_AUTH_MISSING"
            self.log("autonomous: codex auth missing (OPENAI_API_KEY is empty for api-key mode)")
            self._codex_auth_ready = False
            return False

        bootstrap_home = candidate_homes[0] if candidate_homes else ""
        bootstrap_env = self._codex_subprocess_env(auth_mode=auth_mode, home_override=bootstrap_home)
        try:
            proc = subprocess.run(
                ["codex", "login", "--with-api-key"],
                input=api_key + "\n",
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
                env=bootstrap_env,
            )
        except FileNotFoundError:
            self._codex_auth_reason = "CODEX_EXEC_UNAVAILABLE"
            self.log("autonomous: codex auth bootstrap failed (codex binary not found)")
            self._codex_auth_ready = False
            return False
        except subprocess.TimeoutExpired:
            self._codex_auth_reason = "CODEX_AUTH_BOOTSTRAP_TIMEOUT"
            self.log("autonomous: codex auth bootstrap timeout")
            self._codex_auth_ready = False
            return False
        except Exception as exc:  # noqa: BLE001
            self._codex_auth_reason = f"CODEX_AUTH_BOOTSTRAP_ERROR:{type(exc).__name__}"
            self.log(f"autonomous: codex auth bootstrap error={type(exc).__name__}")
            self._codex_auth_ready = False
            return False

        if proc.returncode != 0:
            self._codex_auth_reason = "CODEX_AUTH_BOOTSTRAP_FAILED"
            self.log(f"autonomous: codex auth bootstrap failed rc={proc.returncode}")
            self._codex_auth_ready = False
            return False

        self._codex_exec_home = str(bootstrap_env.get("HOME") or "").strip()
        self._codex_auth_ready = True
        self._codex_auth_reason = ""
        self.log(
            "autonomous: codex auth ready via OPENAI_API_KEY bootstrap"
            + (f" (home={self._codex_exec_home})" if self._codex_exec_home else "")
        )
        return True

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

    def _resolve_app_path(self, raw_path: str) -> Path:
        raw = str(raw_path or "").strip()
        candidate = Path(raw)
        if not candidate.as_posix():
            raise ValueError("Empty path in decision payload")
        if candidate.is_absolute():
            resolved = candidate.resolve()
            if self._is_within(resolved, self.app_root):
                return resolved
            raise ValueError(f"Path must be under app root: {raw_path}")

        repo_candidate = (self.repo_root / candidate).resolve()
        if self._is_within(repo_candidate, self.app_root):
            return repo_candidate

        app_candidate = (self.app_root / candidate).resolve()
        if self._is_within(app_candidate, self.app_root):
            return app_candidate
        raise ValueError(f"Path must be under app root: {raw_path}")

    def _resolve_app_relative(self, raw_path: str) -> str:
        path = self._resolve_app_path(raw_path)
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
        rows_head: list[dict[str, str]] = []
        rows_recent: list[dict[str, str]] = []
        status_counts: dict[str, int] = {}
        try:
            with self.rollup_csv.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    summary["total_rows"] += 1
                    status = str(row.get("status") or "").strip().lower() or "unknown"
                    status_counts[status] = status_counts.get(status, 0) + 1
                    row_payload = {
                        "run_group": str(row.get("run_group") or "").strip(),
                        "run_id": str(row.get("run_id") or "").strip(),
                        "status": status,
                        "metrics_present": str(row.get("metrics_present") or "").strip(),
                        "config_path": str(row.get("config_path") or "").strip(),
                        "results_dir": str(row.get("results_dir") or "").strip(),
                        "sharpe_ratio_abs": str(row.get("sharpe_ratio_abs") or "").strip(),
                        "max_drawdown_on_equity": str(row.get("max_drawdown_on_equity") or "").strip(),
                    }
                    if len(rows_head) < limit:
                        rows_head.append(row_payload)
                    rows_recent.append(row_payload)
                    if len(rows_recent) > limit:
                        rows_recent.pop(0)
        except Exception as exc:  # noqa: BLE001
            summary["error"] = f"{type(exc).__name__}:{exc}"
            return summary
        summary["status_counts"] = status_counts
        # Keep backward compatibility for consumers that read sample_rows only,
        # but prefer recent rows so decision context sees latest batches.
        summary["sample_rows"] = rows_recent
        summary["sample_rows_head"] = rows_head
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
        fixed_start, fixed_end = self._fixed_walk_forward_period()
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
            "fixed_walk_forward": {"start_date": fixed_start, "end_date": fixed_end},
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
        historical_status_counts: Dict[str, int] = {}
        matched_rows: list[dict[str, str]] = []
        historical_candidates: list[Dict[str, Any]] = []
        matched_total = 0
        completed_count = 0
        metrics_true_completed = 0
        metrics_false_completed = 0
        objective_available_count = 0
        historical_completed_count = 0
        historical_metrics_true_completed = 0
        historical_metrics_false_completed = 0
        historical_objective_available_count = 0

        if self.rollup_csv.exists():
            try:
                with self.rollup_csv.open("r", encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle):
                        status = str(row.get("status") or "").strip().lower() or "unknown"
                        historical_status_counts[status] = historical_status_counts.get(status, 0) + 1

                        is_completed = status == "completed"
                        metrics_present = _to_bool(row.get("metrics_present"))
                        objective = _to_float(row.get("score"))
                        if is_completed:
                            historical_completed_count += 1
                            if metrics_present:
                                historical_metrics_true_completed += 1
                            else:
                                historical_metrics_false_completed += 1
                        if objective is not None:
                            historical_objective_available_count += 1
                        if is_completed and metrics_present:
                            historical_candidates.append(
                                {
                                    "run_group": str(row.get("run_group") or "").strip(),
                                    "run_id": str(row.get("run_id") or "").strip(),
                                    "config_path": str(row.get("config_path") or "").strip(),
                                    "results_dir": str(row.get("results_dir") or "").strip(),
                                    "best_score": objective,
                                    "total_pnl": _to_float(row.get("total_pnl")),
                                    "worst_robust_sharpe": _to_float(
                                        row.get("worst_robust_sharpe") or row.get("sharpe_ratio_abs")
                                    ),
                                    "worst_dd_pct": _to_float(
                                        row.get("worst_dd_pct") or row.get("max_drawdown_on_equity")
                                    ),
                                    "ranking_basis": "objective_score" if objective is not None else "sharpe_proxy",
                                }
                            )

                        if not self._row_matches_queue_filters(row, run_group=run_group, filters=queue_filters):
                            continue

                        matched_total += 1
                        status_counts[status] = status_counts.get(status, 0) + 1
                        if is_completed:
                            completed_count += 1
                            if metrics_present:
                                metrics_true_completed += 1
                            else:
                                metrics_false_completed += 1

                        if objective is not None:
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
                                    "total_pnl": str(row.get("total_pnl") or "").strip(),
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
                    "total_pnl": _to_float(row.get("total_pnl")),
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
        historical_candidates.sort(
            key=lambda row: (
                _to_float(row.get("best_score")) if row.get("best_score") is not None else float("-inf"),
                _to_float(row.get("worst_robust_sharpe")) if row.get("worst_robust_sharpe") is not None else float("-inf"),
            ),
            reverse=True,
        )
        if not any(row.get("best_score") is not None for row in historical_candidates):
            historical_candidates.sort(
                key=lambda row: (
                    _to_float(row.get("worst_robust_sharpe")) if row.get("worst_robust_sharpe") is not None else float("-inf"),
                ),
                reverse=True,
            )

        computed_missing_reasons: list[str] = []
        blocking_missing_reasons: list[str] = []
        try:
            iteration_idx = int(state.get("iteration") or 0)
        except (TypeError, ValueError):
            iteration_idx = 0
        historical_bootstrap_mode = bool(iteration_idx <= 0 and historical_metrics_true_completed > 0)

        if not self.rollup_csv.exists():
            computed_missing_reasons.append("ROLLUP_NOT_UPDATED")
            blocking_missing_reasons.append("ROLLUP_NOT_UPDATED")
        if queue_path is None and run_group:
            computed_missing_reasons.append("QUEUE_CONTEXT_MISSING")
        if queue_rows and matched_total == 0:
            if not historical_bootstrap_mode:
                computed_missing_reasons.append("ROLLUP_NOT_UPDATED")
                blocking_missing_reasons.append("ROLLUP_NOT_UPDATED")
        if completed_count == 0:
            if historical_metrics_true_completed > 0:
                # Local queue has no completed rows yet, but historical rollup still provides
                # usable completed evidence for decision making.
                computed_missing_reasons.append("NO_COMPLETED_RUNS_CURRENT_QUEUE")
            else:
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
            "historical_status_counts": historical_status_counts,
            "historical_completed_count": historical_completed_count,
            "historical_metrics_present_true_completed": historical_metrics_true_completed,
            "historical_metrics_present_false_completed": historical_metrics_false_completed,
            "historical_objective_available_count": historical_objective_available_count,
            "historical_data_ready": bool(historical_metrics_true_completed > 0),
            "historical_bootstrap_mode": historical_bootstrap_mode,
            "completed_threshold_for_next_batch": completed_threshold,
            "decision_data_ready": bool(
                completed_count >= completed_threshold and metrics_true_completed > 0 and not blocking_missing_reasons
            ),
            "computed_missing_reasons": computed_missing_reasons,
            "blocking_missing_reasons": blocking_missing_reasons,
            "top_candidates": candidates[:10],
            "historical_top_candidates": historical_candidates[:10],
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
            computed = [
                item
                for item in computed
                if item not in {"NO_COMPLETED_RUNS", "NO_COMPLETED_RUNS_CURRENT_QUEUE"}
            ]
            blocking = [
                item
                for item in blocking
                if item not in {"NO_COMPLETED_RUNS", "NO_COMPLETED_RUNS_CURRENT_QUEUE"}
            ]
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
        top_candidate = None
        top_candidates = evidence.get("top_candidates")
        if isinstance(top_candidates, list) and top_candidates:
            first = top_candidates[0]
            if isinstance(first, dict):
                top_candidate = first
        if top_candidate is None:
            historical = evidence.get("historical_top_candidates")
            if isinstance(historical, list) and historical:
                first_hist = historical[0]
                if isinstance(first_hist, dict):
                    top_candidate = first_hist

        top_score = _to_float(top_candidate.get("best_score")) if isinstance(top_candidate, dict) else None
        top_pnl = _to_float(top_candidate.get("total_pnl")) if isinstance(top_candidate, dict) else None
        top_dd = _to_float(top_candidate.get("worst_dd_pct")) if isinstance(top_candidate, dict) else None
        top_score_text = "n/a" if top_score is None else f"{top_score:.6f}"
        top_pnl_text = "n/a" if top_pnl is None else f"{top_pnl:.2f}"
        top_dd_text = "n/a" if top_dd is None else f"{top_dd:.6f}"
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
            "missing={missing} remaining_to_unblock={remaining} elapsed_sec={elapsed} "
            "top_score={top_score} top_pnl={top_pnl} top_dd={top_dd}{remote_suffix}".format(
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
                top_score=top_score_text,
                top_pnl=top_pnl_text,
                top_dd=top_dd_text,
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
            "trajectory_memory": self._trajectory_memory_tail(state, limit=20),
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
                    "List every blocking reason in stop_reason and human_explanation_md. "
                    "If only current queue is empty but historical_data_ready=true, wait is optional."
                ),
                "decision_rule_when_ready": (
                    "If evidence.completed_count >= evidence.completed_threshold_for_next_batch and "
                    "metrics_present_true_completed > 0, default to run_next_batch unless stop=true is justified."
                ),
                "historical_fallback_rule": (
                    "When evidence.historical_data_ready=true and evidence.historical_top_candidates is non-empty, "
                    "you may decide stop or run_next_batch even if current queue has no completed rows."
                ),
                "analysis_scope_rule": (
                    "Always analyze both the current queue/batch and the global rollup index "
                    "(canonical_sources.run_index_csv). Never make decisions from only the latest batch."
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
            "Если completed в текущей очереди нет, но historical_data_ready=true и есть historical_top_candidates,\n"
            "используй historical evidence и не утверждай NO_COMPLETED_RUNS без проверки historical_* полей.\n"
            "Анализируй не только последний batch: обязательно проверяй глобальный rollup index "
            "(context.canonical_sources.run_index_csv) и кросс-batch динамику.\n"
            "Используй context.trajectory_memory как краткую память прошлых action/result/reflection между итерациями.\n"
            "В критериях отбора строго соблюдай stop_policy, включая min_pnl.\n"
            "Периоды walk_forward фиксируются вне агента (по bridge11 windows), их подбирать/менять не нужно.\n"
            "Для run_next_batch: сформируй actionable queue_entries (обычно 10-25, если нет явных ограничений),\n"
            "каждому entry дай notes, а для file_edits добавь rationale.\n"
            "В human_explanation_md обязательно укажи ключевые P&L и DD для текущего лучшего кандидата\n"
            "(или явно напиши, что данных по P&L/DD пока нет).\n"
            "Пути next_queue_path/config_path/results_dir задавай внутри app-root coint4 "
            "(допустимы форматы coint4/... или app-relative вроде configs/... и artifacts/...).\n"
            "config_path делай уникальными и предпочтительно под configs/_autopilot_batches/<next_run_group>/.\n"
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
        allow_anything = constraints.get("allow_anything_in_repo")
        if isinstance(allow_anything, str):
            lowered = allow_anything.strip().lower()
            if lowered in {"true", "false"}:
                allow_anything = lowered == "true"
                constraints["allow_anything_in_repo"] = allow_anything
        if allow_anything is None:
            allow_anything = False
            constraints["allow_anything_in_repo"] = allow_anything
        if not isinstance(allow_anything, bool):
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
        if not bool(self.args.use_codex_exec):
            self._last_wait_reason = "CODEX_EXEC_DISABLED"
            return None
        auth_mode = self._codex_auth_mode()
        if not self._ensure_codex_auth_ready():
            self._last_wait_reason = self._codex_auth_reason or "CODEX_AUTH_MISSING"
            return None
        codex_env = self._codex_subprocess_env(auth_mode=auth_mode)
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
            "-C",
            str(self.repo_root),
            "-s",
            "read-only",
            "--skip-git-repo-check",
            "--ephemeral",
            "--output-schema",
            str(self.codex_schema_path),
            "--output-last-message",
            str(decision_json_path),
            "--json",
            "-",
        ]
        timeout_default = max(300, int(getattr(self.args, "wait_poll_sec", 60) or 60) * 3)
        timeout_override = _to_int(os.environ.get("COINT4_CODEX_EXEC_TIMEOUT_SEC"))
        timeout_sec = timeout_default
        if isinstance(timeout_override, int) and timeout_override > 0:
            timeout_sec = max(30, int(timeout_override))
        heartbeat_sec = _to_int(os.environ.get("COINT4_CODEX_EXEC_HEARTBEAT_SEC"))
        if not isinstance(heartbeat_sec, int) or heartbeat_sec <= 0:
            heartbeat_sec = min(30, max(5, timeout_sec // 10))
        max_attempts = max(1, int(os.environ.get("COINT4_CODEX_EXEC_ATTEMPTS", "3") or "3"))
        backoff_sec = max(1, int(os.environ.get("COINT4_CODEX_EXEC_BACKOFF_SEC", "5") or "5"))
        last_reason = "CODEX_EXEC_UNKNOWN"
        prompt_bytes = len(prompt.encode("utf-8", errors="ignore"))

        for attempt in range(1, max_attempts + 1):
            try:
                self.log(
                    "autonomous: codex exec start attempt={attempt}/{total} timeout={timeout}s heartbeat={heartbeat}s "
                    "prompt_bytes={prompt_bytes} home={home} user={user}".format(
                        attempt=attempt,
                        total=max_attempts,
                        timeout=timeout_sec,
                        heartbeat=heartbeat_sec,
                        prompt_bytes=prompt_bytes,
                        home=codex_env.get("HOME", ""),
                        user=codex_env.get("USER", ""),
                    )
                )
                started_at = time.monotonic()
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=codex_env,
                )
                first_input: Optional[str] = prompt
                stdout_text = ""
                stderr_text = ""
                while True:
                    elapsed_sec = int(max(0, time.monotonic() - started_at))
                    remaining_sec = timeout_sec - elapsed_sec
                    if remaining_sec <= 0:
                        proc.kill()
                        kill_stdout, kill_stderr = proc.communicate()
                        stdout_text = kill_stdout or stdout_text
                        stderr_text = kill_stderr or stderr_text
                        raise subprocess.TimeoutExpired(
                            cmd=cmd,
                            timeout=timeout_sec,
                            output=stdout_text,
                            stderr=stderr_text,
                        )
                    slice_timeout = min(heartbeat_sec, max(1, remaining_sec))
                    try:
                        stdout_text, stderr_text = proc.communicate(input=first_input, timeout=slice_timeout)
                        break
                    except subprocess.TimeoutExpired as timeout_exc:
                        first_input = None
                        if isinstance(timeout_exc.output, str) and timeout_exc.output:
                            stdout_text = timeout_exc.output
                        if isinstance(timeout_exc.stderr, str) and timeout_exc.stderr:
                            stderr_text = timeout_exc.stderr
                        elapsed_sec = int(max(0, time.monotonic() - started_at))
                        if elapsed_sec >= timeout_sec:
                            proc.kill()
                            kill_stdout, kill_stderr = proc.communicate()
                            stdout_text = kill_stdout or stdout_text
                            stderr_text = kill_stderr or stderr_text
                            raise subprocess.TimeoutExpired(
                                cmd=cmd,
                                timeout=timeout_sec,
                                output=stdout_text,
                                stderr=stderr_text,
                            )
                        self.log(
                            "autonomous: codex attempt {attempt}/{total} running elapsed={elapsed}s/{timeout}s".format(
                                attempt=attempt,
                                total=max_attempts,
                                elapsed=elapsed_sec,
                                timeout=timeout_sec,
                            )
                        )
                        continue

                proc_result = subprocess.CompletedProcess(
                    args=cmd,
                    returncode=int(proc.returncode or 0),
                    stdout=stdout_text or "",
                    stderr=stderr_text or "",
                )
                event = {
                    "ts": _utc_now(),
                    "attempt": int(attempt),
                    "max_attempts": int(max_attempts),
                    "cmd": cmd,
                    "rc": int(proc_result.returncode),
                    "elapsed_sec": int(max(0, time.monotonic() - started_at)),
                    "timeout_sec": int(timeout_sec),
                    "stdout": self._sanitize_external_text(proc_result.stdout or ""),
                    "stderr": self._sanitize_external_text(proc_result.stderr or ""),
                    "context_path": _safe_rel(context_path, self.repo_root),
                    "decision_json_path": _safe_rel(decision_json_path, self.repo_root),
                    "home": codex_env.get("HOME", ""),
                    "user": codex_env.get("USER", ""),
                    "auth_mode": auth_mode,
                    "openai_api_key_present": bool(str(codex_env.get("OPENAI_API_KEY") or "").strip()),
                }
                with exec_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            except FileNotFoundError:
                last_reason = "CODEX_EXEC_UNAVAILABLE"
                break
            except subprocess.TimeoutExpired as exc:
                stdout_text = self._sanitize_external_text(str(getattr(exc, "output", "") or ""))
                stderr_text = self._sanitize_external_text(str(getattr(exc, "stderr", "") or ""))
                timeout_event = {
                    "ts": _utc_now(),
                    "attempt": int(attempt),
                    "max_attempts": int(max_attempts),
                    "cmd": cmd,
                    "rc": None,
                    "elapsed_sec": int(timeout_sec),
                    "timeout_sec": int(timeout_sec),
                    "stdout": stdout_text,
                    "stderr": stderr_text,
                    "context_path": _safe_rel(context_path, self.repo_root),
                    "decision_json_path": _safe_rel(decision_json_path, self.repo_root),
                    "home": codex_env.get("HOME", ""),
                    "user": codex_env.get("USER", ""),
                    "auth_mode": auth_mode,
                    "openai_api_key_present": bool(str(codex_env.get("OPENAI_API_KEY") or "").strip()),
                }
                with exec_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(timeout_event, ensure_ascii=False) + "\n")
                classified = self._classify_codex_exec_failure(stdout=stdout_text, stderr=stderr_text)
                last_reason = classified or "CODEX_EXEC_TIMEOUT"
                if last_reason == "CODEX_AUTH_REQUIRED":
                    self.log("autonomous: codex timeout contains auth markers; stop retries")
                    break
                if attempt < max_attempts:
                    sleep_sec = backoff_sec * attempt
                    self.log(
                        "autonomous: codex attempt {attempt}/{total} timeout={timeout}s; retry in {sleep}s".format(
                            attempt=attempt,
                            total=max_attempts,
                            timeout=timeout_sec,
                            sleep=sleep_sec,
                        )
                    )
                    time.sleep(sleep_sec)
                continue
            except Exception as exc:  # noqa: BLE001
                last_reason = f"CODEX_EXEC_ERROR:{type(exc).__name__}"
                if attempt < max_attempts:
                    sleep_sec = backoff_sec * attempt
                    self.log(
                        "autonomous: codex attempt {attempt}/{total} error={reason}; retry in {sleep}s".format(
                            attempt=attempt,
                            total=max_attempts,
                            reason=last_reason,
                            sleep=sleep_sec,
                        )
                    )
                    time.sleep(sleep_sec)
                continue

            if proc_result.returncode != 0:
                classified = self._classify_codex_exec_failure(
                    stdout=proc_result.stdout or "",
                    stderr=proc_result.stderr or "",
                )
                last_reason = classified or f"CODEX_EXEC_RC{proc_result.returncode}"
                if last_reason == "CODEX_AUTH_REQUIRED":
                    self.log("autonomous: codex auth required in runtime HOME; stop retries")
            elif not decision_json_path.exists():
                last_reason = "CODEX_DECISION_MISSING"
            else:
                try:
                    payload = json.loads(decision_json_path.read_text(encoding="utf-8"))
                except Exception as exc:  # noqa: BLE001
                    last_reason = f"CODEX_DECISION_INVALID_JSON:{type(exc).__name__}"
                else:
                    if not isinstance(payload, dict):
                        last_reason = "CODEX_DECISION_INVALID_PAYLOAD"
                    else:
                        validation_error = self._validate_decision(payload)
                        if validation_error:
                            last_reason = f"CODEX_DECISION_SCHEMA:{validation_error}"
                        else:
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

            is_retriable = (
                last_reason not in {"CODEX_EXEC_UNAVAILABLE", "CODEX_AUTH_REQUIRED"}
                and not last_reason.startswith("CODEX_DECISION_SCHEMA:")
            )
            if is_retriable and attempt < max_attempts:
                sleep_sec = backoff_sec * attempt
                self.log(
                    "autonomous: codex attempt {attempt}/{total} failed ({reason}); retry in {sleep}s".format(
                        attempt=attempt,
                        total=max_attempts,
                        reason=last_reason,
                        sleep=sleep_sec,
                    )
                )
                time.sleep(sleep_sec)
                continue
            break

        self._last_wait_reason = last_reason
        return None

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
        queue_path = self._resolve_app_path(str(payload.get("next_queue_path") or ""))
        queue_entries = list(payload.get("queue_entries") or [])
        if not queue_entries:
            raise ValueError("Decision next_action=run_next_batch but queue_entries is empty")

        base_cfg = _read_yaml(self.bridge11_path)
        rows: list[dict[str, str]] = []
        seen_config_paths: set[str] = set()
        generated_prefix = f"configs/_autopilot_batches/{run_group}/"
        generated_dir = self.app_root / "configs" / "_autopilot_batches" / run_group
        for idx, entry in enumerate(queue_entries):
            requested_config_abs = self._resolve_app_path(str(entry.get("config_path") or ""))
            requested_config_rel = _safe_rel(requested_config_abs, self.app_root).replace("\\", "/")

            results_raw = str(entry.get("results_dir") or "").strip()
            if not results_raw:
                raise ValueError("queue entry results_dir is empty")
            results_abs = self._resolve_app_path(results_raw)
            results_rel = _safe_rel(results_abs, self.app_root).replace("\\", "/")

            # Keep provided config paths only when they are already in the per-run-group
            # batch namespace and unique. Otherwise materialize into generated files.
            if (
                requested_config_rel.startswith(generated_prefix)
                and requested_config_rel not in seen_config_paths
            ):
                config_abs = requested_config_abs
                config_rel = requested_config_rel
            else:
                config_abs = generated_dir / f"entry_{idx:03d}.yaml"
                config_rel = _safe_rel(config_abs, self.app_root).replace("\\", "/")
                dedupe_idx = 1
                while config_rel in seen_config_paths:
                    config_abs = generated_dir / f"entry_{idx:03d}_{dedupe_idx:02d}.yaml"
                    config_rel = _safe_rel(config_abs, self.app_root).replace("\\", "/")
                    dedupe_idx += 1

            overrides = entry.get("overrides") if isinstance(entry.get("overrides"), dict) else {}
            cfg_payload = self._deep_merge(base_cfg, overrides)
            self._apply_fixed_walk_forward_period(cfg_payload)
            config_abs.parent.mkdir(parents=True, exist_ok=True)
            config_abs.write_text(yaml.safe_dump(cfg_payload, sort_keys=False), encoding="utf-8")
            seen_config_paths.add(config_rel)

            rows.append(
                {
                    "config_path": config_rel,
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

        return QueueTarget(
            run_group=run_group,
            queue_path=queue_path,
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

    def _queue_status_counts(self, queue_path: Path) -> Dict[str, int]:
        try:
            rows = self._read_queue_rows(queue_path)
        except Exception:  # noqa: BLE001
            return {}
        counts: Dict[str, int] = {}
        for row in rows:
            status = str(row.get("status") or "planned").strip().lower() or "planned"
            counts[status] = int(counts.get(status, 0) or 0) + 1
        return counts

    @staticmethod
    def _is_stalled_only_counts(counts: Dict[str, int]) -> bool:
        stalled = int(counts.get("stalled", 0) or 0)
        completed = int(counts.get("completed", 0) or 0)
        pending_total = sum(
            int(counts.get(name, 0) or 0)
            for name in ("planned", "running", "queued", "pending", "active", "partial")
        )
        return stalled > 0 and completed == 0 and pending_total == 0

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
        codex_env = self._codex_subprocess_env()

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
            try:
                proc = subprocess.run(
                    cmd,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=max(30, int(timeout_sec)),
                    env=codex_env,
                )
            except subprocess.TimeoutExpired as exc:
                # Fail-open: codex helper timeouts should never crash the main loop.
                stdout_text = self._sanitize_external_text(str(getattr(exc, "output", "") or ""))
                stderr_text = self._sanitize_external_text(str(getattr(exc, "stderr", "") or ""))
                self.log(
                    "autonomous: codex exec timeout timeout={timeout}s; fallback deterministic stdout={stdout} stderr={stderr}".format(
                        timeout=int(timeout_sec),
                        stdout=stdout_text[-500:] if stdout_text else "",
                        stderr=stderr_text[-500:] if stderr_text else "",
                    )
                )
                return None
            except Exception as exc:  # noqa: BLE001
                self.log(f"autonomous: codex exec error {type(exc).__name__}: {exc}; fallback deterministic")
                return None
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

    def select_next_queue(self, *, state: Dict[str, Any], allow_codex_pick: bool = True) -> Optional[QueueTarget]:
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

        if allow_codex_pick:
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
        lines = [line.strip() for line in (text or "").splitlines() if line.strip().startswith("|")]
        if len(lines) < 3:
            return None

        header = [part.strip() for part in lines[0].strip("|").split("|")]
        if not header:
            return None

        for line in lines[2:]:
            cols = [part.strip() for part in line.strip("|").split("|")]
            if len(cols) != len(header):
                continue
            row = dict(zip(header, cols))
            if str(row.get("rank") or "").strip() != "1":
                continue

            score = _to_float(row.get("score"))
            worst_robust = _to_float(row.get("worst_robust_sh"))
            worst_dd = _to_float(row.get("worst_dd_pct"))
            run_name = str(row.get("variant_id") or "").strip()
            cfg = str(row.get("sample_config") or "").strip()
            if score is None or not run_name:
                continue
            if worst_robust is None:
                worst_robust = score
            return RankResult(
                ok=True,
                source="rank_multiwindow_robust_runs.py",
                score=score,
                worst_robust_sharpe=worst_robust,
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
        min_pnl = _to_float(self.stop_policy.get("min_pnl"))
        if min_pnl is None:
            min_pnl = 0.0
        min_psr = _to_float(self.stop_policy.get("min_psr"))
        min_dsr = _to_float(self.stop_policy.get("min_dsr"))
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
                pnl = _to_float(row.get("total_pnl"))
                if pnl is None or pnl < min_pnl:
                    continue
                psr = _to_float(row.get("psr"))
                if min_psr is not None and (psr is None or psr < min_psr):
                    continue
                dsr = _to_float(row.get("dsr"))
                if min_dsr is not None and (dsr is None or dsr < min_dsr):
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
            "--min-pnl",
            str(float(self.stop_policy["min_pnl"])),
        ]
        min_psr = _to_float(self.stop_policy.get("min_psr"))
        if min_psr is not None:
            cmd.extend(["--min-psr", str(float(min_psr))])
        min_dsr = _to_float(self.stop_policy.get("min_dsr"))
        if min_dsr is not None:
            cmd.extend(["--min-dsr", str(float(min_dsr))])
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
        lines.append(
            f"- Gates: min_windows={self.stop_policy['min_windows']}, min_trades={self.stop_policy['min_trades']}, "
            f"min_pairs={self.stop_policy['min_pairs']}, max_dd_pct={self.stop_policy['max_dd_pct']}, "
            f"min_pnl={self.stop_policy['min_pnl']}"
        )
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
        top_candidate: Optional[Dict[str, Any]] = None
        top_candidates = progress.get("top_candidates")
        if isinstance(top_candidates, list) and top_candidates:
            first = top_candidates[0]
            if isinstance(first, dict):
                top_candidate = first
        if top_candidate is None:
            historical_candidates = progress.get("historical_top_candidates")
            if isinstance(historical_candidates, list) and historical_candidates:
                first_hist = historical_candidates[0]
                if isinstance(first_hist, dict):
                    top_candidate = first_hist
        if isinstance(top_candidate, dict):
            lines.append(
                "- Top candidate: run_id={run_id}, score={score}, pnl={pnl}, dd={dd}".format(
                    run_id=str(top_candidate.get("run_id") or "n/a"),
                    score=top_candidate.get("best_score"),
                    pnl=top_candidate.get("total_pnl"),
                    dd=top_candidate.get("worst_dd_pct"),
                )
            )
        else:
            lines.append("- Top candidate: n/a")
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

    def write_final_report(self, state: Dict[str, Any], *, prefer_codex: bool = True) -> None:
        defaults_used = list(self.stop_policy.get("defaults_used") or [])
        if prefer_codex and bool(self.args.use_codex_exec):
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
                "Укажи objective/gates/stop criteria и winner. Обязательно добавь P&L и DD по winner (или явно укажи, что данных нет)."
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
        wait_seconds: Optional[int] = None,
        prefer_codex_report: bool = True,
    ) -> Dict[str, Any]:
        if phase.startswith("waiting") or phase == "skipping_queue":
            self.log_progress_snapshot(state, phase=f"wait:{phase}")
        delay = wait_seconds
        fallback = int(getattr(self.args, "wait_poll_sec", 60) or 60)
        if fallback <= 0:
            fallback = 60
        max_wait_env = _to_int(os.environ.get("COINT4_MAX_WAIT_SECONDS"))
        max_wait = max_wait_env if isinstance(max_wait_env, int) and max_wait_env > 0 else fallback
        if not isinstance(delay, int) or delay <= 0:
            delay = fallback
        if delay > max_wait:
            self.log(
                "autonomous: clamp wait_seconds {value}s -> {limit}s".format(
                    value=int(delay),
                    limit=int(max_wait),
                )
            )
            delay = max_wait
        action = str(state.get("last_decision_action") or "").strip()
        if not action:
            action = "wait" if phase.startswith("waiting") else str(phase or "wait").strip()
        reflection = str(state.get("last_decision_explanation_md") or "").strip() or reason
        self._append_trajectory_memory(state, action=action, result=phase, reflection=reflection)
        self._next_iteration_delay_sec = int(delay)
        state["status"] = "running"
        state["last_error"] = reason
        state["last_iteration_phase"] = phase
        self.save_state(state)
        self.write_best_params(state)
        self.write_final_report(state, prefer_codex=prefer_codex_report)
        self._exit_after_iteration_wait = True
        return state

    def _plan_next_queue_with_evolution(
        self,
        *,
        state: Dict[str, Any],
        iteration: int,
        run_group: str,
        iteration_log: Path,
    ) -> EvolutionPlan:
        if not self.evolve_next_batch.exists():
            raise RuntimeError(f"missing planner script: {self.evolve_next_batch}")

        controller_group = self._evolution_controller_group(state)
        run_prefix = self._evolution_run_prefix(state)
        base_config = self._bridge11_base_config_rel()
        windows = self._bridge11_windows()
        num_variants = _to_int(getattr(self.args, "evolution_num_variants", None))
        if not isinstance(num_variants, int) or num_variants <= 0:
            num_variants = 12

        contains_tokens = [
            str(token).strip()
            for token in list(getattr(self.args, "evolution_contains", []) or [])
            if str(token).strip()
        ]
        if not contains_tokens:
            contains_tokens.append(run_prefix)
        previous_group = str(state.get("current_run_group") or "").strip()
        if previous_group and previous_group not in contains_tokens:
            contains_tokens.append(previous_group)

        decision_dir = self.app_root / "artifacts" / "wfa" / "aggregate" / controller_group / "decisions"
        decision_dir.mkdir(parents=True, exist_ok=True)
        known_decisions = {path.resolve() for path in decision_dir.glob("*.json")}
        planner_env = self._env()
        if bool(self.args.use_codex_exec):
            if not self._ensure_codex_auth_ready():
                reason = str(self._codex_auth_reason or "CODEX_AUTH_REQUIRED").strip()
                raise RuntimeError(reason)
            selected_home = str(self._codex_exec_home or "").strip()
            if selected_home:
                planner_env["HOME"] = selected_home

        cmd = [
            str(self.python_exec),
            "scripts/optimization/evolve_next_batch.py",
            "--base-config",
            base_config,
            "--controller-group",
            controller_group,
            "--run-group",
            run_group,
            "--num-variants",
            str(int(num_variants)),
            "--ir-mode",
            str(getattr(self.args, "evolution_ir_mode", "patch_ast") or "patch_ast"),
            "--dedupe-distance",
            str(float(getattr(self.args, "evolution_dedupe_distance", 0.0) or 0.0)),
            "--max-changed-keys",
            str(int(getattr(self.args, "evolution_max_changed_keys", 8) or 8)),
            "--policy-scale",
            str(getattr(self.args, "evolution_policy_scale", "auto") or "auto"),
            "--ast-max-complexity-score",
            str(float(getattr(self.args, "evolution_ast_max_complexity_score", 120.0) or 120.0)),
            "--ast-max-redundancy-similarity",
            str(float(getattr(self.args, "evolution_ast_max_redundancy_similarity", 1.0) or 1.0)),
            "--patch-max-attempts",
            str(int(getattr(self.args, "evolution_patch_max_attempts", 8) or 8)),
        ]
        for token in contains_tokens:
            cmd.extend(["--contains", token])
        for start, end in windows:
            cmd.extend(["--window", f"{start},{end}"])

        if bool(self.args.use_codex_exec):
            llm_model = str(getattr(self.args, "codex_model", "") or "").strip() or str(
                getattr(self.args, "evolution_llm_model", "gpt-5.2") or "gpt-5.2"
            ).strip()
            cmd.extend(
                [
                    "--llm-propose",
                    "--llm-model",
                    llm_model,
                    "--llm-effort",
                    str(getattr(self.args, "evolution_llm_effort", "high") or "high"),
                    "--llm-timeout-sec",
                    str(int(getattr(self.args, "evolution_llm_timeout_sec", 300) or 300)),
                ]
            )
            if bool(getattr(self.args, "evolution_llm_verify_semantic", True)):
                cmd.append("--llm-verify-semantic")

        proc = self._run_subprocess(
            cmd=cmd,
            cwd=self.app_root,
            env=planner_env,
            iteration_log=iteration_log,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"EVOLVE_NEXT_BATCH_RC{proc.returncode}")

        decision_paths = sorted(decision_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not decision_paths:
            raise RuntimeError("EVOLUTION_DECISION_MISSING")
        latest_new = [path for path in decision_paths if path.resolve() not in known_decisions]
        decision_path = latest_new[0] if latest_new else decision_paths[0]

        try:
            decision_payload = json.loads(decision_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"EVOLUTION_DECISION_INVALID_JSON:{type(exc).__name__}") from exc
        if not isinstance(decision_payload, dict):
            raise RuntimeError("EVOLUTION_DECISION_INVALID_PAYLOAD")

        queue_rel = str(decision_payload.get("queue_path") or "").strip()
        if not queue_rel:
            raise RuntimeError("EVOLUTION_QUEUE_PATH_EMPTY")
        queue_path = self._resolve_app_path(queue_rel)
        if not queue_path.exists():
            raise RuntimeError("EVOLUTION_QUEUE_PATH_MISSING")
        ready_rows = self._count_ready_rows(queue_path)
        if ready_rows <= 0:
            raise RuntimeError("EVOLUTION_QUEUE_EMPTY")

        planned_run_group = str(decision_payload.get("run_group") or "").strip() or run_group
        decision_id = str(decision_payload.get("decision_id") or "").strip() or decision_path.stem
        state["last_decision_id"] = decision_id
        state["last_decision_action"] = "run_next_batch_evolution"
        state["last_decision_explanation_md"] = (
            f"evolution planner: controller_group={controller_group}, "
            f"run_group={planned_run_group}, queue={_safe_rel(queue_path, self.app_root)}"
        )
        state["planner_mode"] = "evolution"
        state["evolution_controller_group"] = controller_group
        state["evolution_run_prefix"] = run_prefix
        state["last_evolution_decision_path"] = _safe_rel(decision_path, self.repo_root)

        queue = QueueTarget(
            run_group=planned_run_group,
            queue_path=queue_path,
            source="evolution_planner",
            ready_rows=ready_rows,
        )
        return EvolutionPlan(
            queue=queue,
            controller_group=controller_group,
            run_prefix=run_prefix,
            decision_path=decision_path,
        )

    def _run_evolution_reflection(
        self,
        *,
        plan: EvolutionPlan,
        iteration_log: Path,
    ) -> Dict[str, Any]:
        if not self.reflect_next_action.exists():
            return {}
        reflections_dir = self.app_root / "artifacts" / "wfa" / "aggregate" / plan.controller_group / "reflections"
        reflections_dir.mkdir(parents=True, exist_ok=True)
        reflection_path = reflections_dir / f"reflection_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"

        cmd = [
            str(self.python_exec),
            "scripts/optimization/reflect_next_action.py",
            "--decision",
            str(plan.decision_path),
            "--run-index",
            str(self.rollup_csv),
            "--contains",
            plan.queue.run_group,
            "--output-json",
            str(reflection_path),
        ]
        reflection_env = self._env()
        if bool(self.args.use_codex_exec):
            if not self._ensure_codex_auth_ready():
                return {}
            selected_home = str(self._codex_exec_home or "").strip()
            if selected_home:
                reflection_env["HOME"] = selected_home
            llm_model = str(getattr(self.args, "codex_model", "") or "").strip() or str(
                getattr(self.args, "evolution_llm_model", "gpt-5.2") or "gpt-5.2"
            ).strip()
            cmd.extend(
                [
                    "--llm-critic",
                    "--llm-model",
                    llm_model,
                    "--llm-timeout-sec",
                    str(int(getattr(self.args, "evolution_llm_timeout_sec", 300) or 300)),
                ]
            )

        proc = self._run_subprocess(
            cmd=cmd,
            cwd=self.app_root,
            env=reflection_env,
            iteration_log=iteration_log,
        )
        if proc.returncode != 0 or not reflection_path.exists():
            return {}
        try:
            payload = json.loads(reflection_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
        if not isinstance(payload, dict):
            return {}
        payload["__path"] = _safe_rel(reflection_path, self.repo_root)
        return payload

    def _build_factor_pool_from_evolution(self, *, plan: EvolutionPlan, iteration_log: Path) -> None:
        if not self.build_factor_pool.exists():
            return
        output_dir = self.app_root / "artifacts" / "wfa" / "aggregate" / plan.controller_group
        output_dir.mkdir(parents=True, exist_ok=True)
        pool_json = output_dir / f"factor_pool_{plan.queue.run_group}.json"
        pool_md = output_dir / f"factor_pool_{plan.queue.run_group}.md"
        cmd = [
            str(self.python_exec),
            "scripts/optimization/build_factor_pool.py",
            "--controller-group",
            plan.controller_group,
            "--run-index",
            str(self.rollup_csv),
            "--decisions-dir",
            str(output_dir / "decisions"),
            "--contains",
            plan.run_prefix,
            "--top",
            "20",
            "--output-json",
            str(pool_json),
            "--output-md",
            str(pool_md),
        ]
        proc = self._run_subprocess(
            cmd=cmd,
            cwd=self.app_root,
            env=self._env(),
            iteration_log=iteration_log,
        )
        if proc.returncode != 0:
            self.log(f"autonomous: factor_pool build skipped rc={proc.returncode}")

    def _run_iteration_evolution(self, *, state: Dict[str, Any]) -> Dict[str, Any]:
        self._exit_after_iteration_wait = False
        self._next_iteration_delay_sec = max(1, int(getattr(self.args, "wait_poll_sec", 60) or 60))
        iteration = int(state.get("iteration") or 0)
        planned_run_group = self._next_evolution_run_group(state, iteration=iteration)
        iteration_log = self._iteration_log_path(planned_run_group, iteration)

        try:
            plan = self._plan_next_queue_with_evolution(
                state=state,
                iteration=iteration,
                run_group=planned_run_group,
                iteration_log=iteration_log,
            )
        except Exception as exc:  # noqa: BLE001
            reason = f"EVOLUTION_PLAN_ERROR:{type(exc).__name__}:{exc}"
            current_group = str(state.get("current_run_group") or "").strip()
            current_queue = self._queue_target_for_run_group(
                current_group,
                source="evolution_plan_error_reuse_queue",
            )
            if (
                current_queue is not None
                and current_queue.ready_rows > 0
                and not self._is_demo_queue(current_queue.queue_path)
                and not self._is_ignored_queue(current_queue.queue_path, state)
            ):
                self.log(
                    "autonomous: evolution planner failed; reuse existing ready queue "
                    f"{_safe_rel(current_queue.queue_path, self.app_root)} reason={reason}"
                )
                return self._run_iteration_local(state=state)
            return self._wait_and_exit(
                state=state,
                reason=reason,
                phase="waiting_plan",
            )

        queue = plan.queue
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
        state["planner_mode"] = "evolution"
        self.save_state(state)
        self.write_final_report(state)

        self.log(
            f"autonomous: evolution iteration={iteration + 1} queue={_safe_rel(queue.queue_path, self.app_root)} "
            f"source={queue.source} ready={queue.ready_rows}"
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

        queue_counts = self._queue_status_counts(queue.queue_path)
        has_completed_metrics, metrics_reason = self._has_completed_metrics_for_queue(queue.queue_path)
        if self._is_stalled_only_counts(queue_counts) and not has_completed_metrics:
            self.log(
                "autonomous: skip local stalled-only queue {queue} counts={counts} metrics={metrics}".format(
                    queue=_safe_rel(queue.queue_path, self.app_root),
                    counts=json.dumps(queue_counts, ensure_ascii=False, sort_keys=True),
                    metrics=str(metrics_reason or "NO_DATA"),
                )
            )
            self._record_ignored_queue(state, queue)
            return self._wait_and_exit(
                state=state,
                reason="SKIP_QUEUE:LOCAL_STALLED_ONLY",
                phase="skipping_queue",
                wait_seconds=1,
            )

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
            state["last_error"] = "RANK_UPDATED"
            state["last_iteration_phase"] = "rank_ok"
        else:
            state["last_error"] = f"RANK_NOT_READY:{rank.details or 'NO_DATA'}"
            state["last_iteration_phase"] = "rank_pending"

        reflection_payload = self._run_evolution_reflection(plan=plan, iteration_log=iteration_log)
        next_action = str(reflection_payload.get("next_action") or "").strip() if reflection_payload else ""
        reflection_text = str(reflection_payload.get("reflection") or "").strip() if reflection_payload else ""
        if not reflection_text:
            if rank.ok and rank.score is not None:
                reflection_text = (
                    f"score={rank.score:.6f}; run_name={rank.run_name or 'n/a'}; "
                    f"config_path={rank.config_path or 'n/a'}; source={rank.source}"
                )
            else:
                reflection_text = f"rank_not_ready:{rank.details or 'NO_DATA'}"
        state["last_evolution_reflection_path"] = str(reflection_payload.get("__path") if reflection_payload else "")

        if next_action == "stop":
            state["status"] = "done"
            state["last_error"] = str(reflection_payload.get("reflection") or "REFLECTION_STOP").strip()
            state["last_iteration_phase"] = "stopped_by_reflection"
            self._append_trajectory_memory(
                state,
                action="stop",
                result="stopped_by_reflection",
                reflection=reflection_text,
            )
            self._build_factor_pool_from_evolution(plan=plan, iteration_log=iteration_log)
            self.save_state(state)
            self.write_best_params(state)
            self.write_final_report(state)
            return state

        action = next_action or "run_next_batch"
        self._append_trajectory_memory(
            state,
            action=action,
            result=str(state.get("last_iteration_phase") or "rank_pending"),
            reflection=reflection_text,
        )
        self._build_factor_pool_from_evolution(plan=plan, iteration_log=iteration_log)
        self.save_state(state)
        self.write_best_params(state)
        self.write_final_report(state)
        self._next_iteration_delay_sec = 0
        self._exit_after_iteration_wait = True
        return state

    def _run_iteration_local(self, *, state: Dict[str, Any]) -> Dict[str, Any]:
        self._exit_after_iteration_wait = False
        self._next_iteration_delay_sec = max(1, int(getattr(self.args, "wait_poll_sec", 60) or 60))
        iteration = int(state.get("iteration") or 0)

        queue = self.select_next_queue(state=state, allow_codex_pick=False)
        if queue is None:
            reason = str(self._last_queue_selection_error or "NO_VALID_QUEUES").strip() or "NO_VALID_QUEUES"
            return self._wait_and_exit(
                state=state,
                reason=reason,
                phase="waiting_queue",
                wait_seconds=min(10, self._next_iteration_delay_sec),
            )
        if self._is_demo_queue(queue.queue_path):
            self._record_ignored_queue(state, queue)
            return self._wait_and_exit(
                state=state,
                reason="SKIP_QUEUE:DEMO_QUEUE",
                phase="skipping_queue",
                wait_seconds=1,
            )
        if self._is_ignored_queue(queue.queue_path, state):
            return self._wait_and_exit(
                state=state,
                reason="SKIP_QUEUE:IGNORED_QUEUE",
                phase="skipping_queue",
                wait_seconds=1,
            )
        if queue.ready_rows < 1:
            return self._wait_and_exit(
                state=state,
                reason="NO_VALID_QUEUES",
                phase="waiting_queue",
                wait_seconds=min(10, self._next_iteration_delay_sec),
            )
        remote_probe = self._probe_remote_queue_counts(queue.queue_path)
        if isinstance(remote_probe, dict):
            counts_raw = remote_probe.get("counts")
            counts = counts_raw if isinstance(counts_raw, dict) else {}
            stalled = int(counts.get("stalled", 0) or 0)
            planned = int(counts.get("planned", 0) or 0)
            running = int(counts.get("running", 0) or 0)
            completed = int(counts.get("completed", 0) or 0)
            has_metrics = bool(remote_probe.get("has_metrics"))
            if stalled > 0 and planned == 0 and running == 0 and completed == 0 and not has_metrics:
                self.log(
                    "autonomous: skip stalled-only remote queue {queue}".format(
                        queue=_safe_rel(queue.queue_path, self.app_root),
                    )
                )
                self._record_ignored_queue(state, queue)
                return self._wait_and_exit(
                    state=state,
                    reason="SKIP_QUEUE:REMOTE_STALLED_ONLY",
                    phase="skipping_queue",
                    wait_seconds=1,
                )

        state["last_decision_id"] = f"local-{self._decision_stamp()}"
        state["last_decision_action"] = "run_existing_queue"
        state["last_decision_explanation_md"] = (
            "Локальный deterministic режим: выбран существующий ready queue без Codex exec."
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
            f"autonomous: local iteration={iteration + 1} queue={_safe_rel(queue.queue_path, self.app_root)} source={queue.source} ready={queue.ready_rows}"
        )
        self.log_progress_snapshot(state, phase="pre-run")

        if bool(self.args.plan_only):
            return self._wait_and_exit(state=state, reason="PLAN_ONLY_MODE", phase="planned", wait_seconds=1)

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
                "REMOTE_EXEC_FAILED",
                "SERVSPACEERROR",
                "SERVSPACE_ERROR",
            } or fail_reason_norm.startswith("QUEUE_"):
                self._record_ignored_queue(state, queue)
                return self._wait_and_exit(
                    state=state,
                    reason=f"SKIP_QUEUE:{fail_reason_norm}",
                    phase="skipping_queue",
                    wait_seconds=1,
                )
            return self._wait_and_exit(
                state=state,
                reason=f"POWERED_WAIT:{fail_reason_norm}",
                phase="waiting_powered",
                wait_seconds=min(10, self._next_iteration_delay_sec),
            )

        if bool(self.args.local_rollup_rebuild):
            rc_rollup = self._rebuild_rollup(iteration_log=iteration_log)
            if rc_rollup != 0:
                return self._wait_and_exit(
                    state=state,
                    reason=f"BUILD_RUN_INDEX_RC{rc_rollup}",
                    phase="waiting_rollup",
                    wait_seconds=min(10, self._next_iteration_delay_sec),
                )
        else:
            self.log("autonomous: skip local build_run_index (using powered sync-back rollup)")

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
            state["last_error"] = "LOCAL_NEXT_BATCH_READY"
            state["last_iteration_phase"] = "rank_ok"
        else:
            state["last_error"] = f"RANK_NOT_READY:{rank.details or 'NO_DATA'}"
            state["last_iteration_phase"] = "rank_pending"

        reflection = ""
        if rank.ok and rank.score is not None:
            reflection = (
                f"score={rank.score:.6f}; run_name={rank.run_name or 'n/a'}; "
                f"config_path={rank.config_path or 'n/a'}; source={rank.source}"
            )
        else:
            reflection = f"rank_not_ready:{rank.details or 'NO_DATA'}"
        self._append_trajectory_memory(
            state,
            action=str(state.get("last_decision_action") or "run_existing_queue"),
            result=str(state.get("last_iteration_phase") or "rank_pending"),
            reflection=reflection,
        )
        state["status"] = "running"
        self.save_state(state)
        self.write_best_params(state)
        self.write_final_report(state)
        self._next_iteration_delay_sec = min(5, max(1, int(getattr(self.args, "wait_poll_sec", 60) or 60)))
        self._exit_after_iteration_wait = True
        return state

    def run_iteration(self, *, state: Dict[str, Any]) -> Dict[str, Any]:
        if self._planner_mode() == "evolution":
            return self._run_iteration_evolution(state=state)

        if not bool(self.args.use_codex_exec):
            return self._run_iteration_local(state=state)

        self._exit_after_iteration_wait = False
        self._next_iteration_delay_sec = max(1, int(getattr(self.args, "wait_poll_sec", 60) or 60))
        iteration = int(state.get("iteration") or 0)

        decision = self.decide_with_codex(state)
        if decision is None:
            wait_reason = str(self._last_wait_reason or "WAITING_CODEX_OR_DATA").strip()
            self.log(f"autonomous: codex decision unavailable ({wait_reason})")
            if wait_reason.startswith("CODEX_AUTH_") or wait_reason == "CODEX_AUTH_MISSING":
                return self._wait_and_exit(
                    state=state,
                    reason=wait_reason,
                    phase="waiting_codex_auth",
                    wait_seconds=1800,
                    prefer_codex_report=False,
                )
            return self._wait_and_exit(state=state, reason=wait_reason, phase="waiting_codex")

        self._apply_decision_state(state, decision)
        payload = decision.payload
        next_action = str(payload.get("next_action") or "").strip()
        stop = bool(payload.get("stop"))

        if stop or next_action == "stop":
            state["status"] = "done"
            state["last_error"] = str(payload.get("stop_reason") or "").strip()
            state["last_iteration_phase"] = "stopped_by_codex"
            reflection = str(payload.get("human_explanation_md") or "").strip() or str(payload.get("stop_reason") or "").strip()
            self._append_trajectory_memory(
                state,
                action=next_action or "stop",
                result="stopped_by_codex",
                reflection=reflection,
            )
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
            return self._wait_and_exit(
                state=state,
                reason=wait_reason,
                phase="waiting_codex",
                wait_seconds=_to_int(payload.get("wait_seconds")),
            )

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
        state["last_error"] = "RANK_UPDATED"
        state["last_iteration_phase"] = "rank_ok"
        reflection = ""
        if rank.ok and rank.score is not None:
            reflection = (
                f"score={rank.score:.6f}; run_name={rank.run_name or 'n/a'}; "
                f"config_path={rank.config_path or 'n/a'}; source={rank.source}"
            )
        else:
            reflection = f"rank_not_ready:{rank.details or 'NO_DATA'}"
        self._append_trajectory_memory(
            state,
            action=next_action or str(state.get("last_decision_action") or "run_next_batch"),
            result="rank_ok",
            reflection=reflection,
        )
        self.save_state(state)
        self.write_best_params(state)
        self.write_final_report(state)
        self._next_iteration_delay_sec = 0
        self._exit_after_iteration_wait = True
        return state

    def run(self) -> int:
        state = self.load_state()
        self._quarantine_demo_queues()
        self.log(
            f"autonomous: start once={bool(self.args.once)} until_done={bool(self.args.until_done)} "
            f"max_iterations={self.args.max_iterations} plan_only={bool(self.args.plan_only)} "
            f"planner_mode={self._planner_mode()} "
            f"use_codex_exec={bool(self.args.use_codex_exec)} "
            f"wait_timeout_sec={int(self.args.wait_timeout_sec)} wait_poll_sec={int(self.args.wait_poll_sec)} "
            f"local_rollup_rebuild={bool(self.args.local_rollup_rebuild)}"
        )

        if state.get("status") == "done" and not bool(self.args.once):
            if bool(getattr(self.args, "resume", False)):
                self.log("autonomous: state is done; resume=true; continue")
                state["status"] = "running"
                self.save_state(state)
            else:
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
                    if bool(self.args.plan_only):
                        self.log("autonomous: plan-only complete")
                        return 0
                    if bool(self.args.once):
                        self.log("autonomous: once mode complete")
                        return 0
                    if bool(self.args.until_done):
                        phase = str(state.get("last_iteration_phase") or "").strip() or "waiting"
                        delay_sec = int(self._next_iteration_delay_sec or 0)
                        if delay_sec > 0:
                            self.log(
                                f"autonomous: wait phase={phase}, sleep {delay_sec}s and continue in-process (--until-done)"
                            )
                            time.sleep(delay_sec)
                        else:
                            self.log(
                                f"autonomous: continue immediately phase={phase} in-process (--until-done)"
                            )
                        state = self.load_state()
                        continue
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last saved iteration even if the state is marked done.",
    )
    parser.add_argument("--max-iterations", type=int, default=None, help="Override max iterations.")
    parser.add_argument(
        "--planner-mode",
        choices=["legacy", "evolution"],
        default="evolution",
        help="Planner mode: legacy Codex decision schema or evolution (ConfigPatch AST pipeline).",
    )
    parser.add_argument("--use-codex-exec", action="store_true", help="Enable optional codex exec helpers.")
    parser.add_argument("--codex-model", default="", help="Optional model override for codex exec.")
    parser.add_argument(
        "--evolution-controller-group",
        default="",
        help="Override controller group for evolution planner state/decisions.",
    )
    parser.add_argument(
        "--evolution-run-prefix",
        default="autonomous_evo",
        help="Run group prefix for evolution planner batches.",
    )
    parser.add_argument(
        "--evolution-contains",
        action="append",
        default=[],
        help="Optional contains token for evolution diagnostics filtering (repeatable).",
    )
    parser.add_argument(
        "--evolution-num-variants",
        type=int,
        default=12,
        help="Number of variants per evolution batch.",
    )
    parser.add_argument(
        "--evolution-ir-mode",
        choices=["knob", "patch_ast"],
        default="patch_ast",
        help="Evolution candidate IR mode.",
    )
    parser.add_argument(
        "--evolution-dedupe-distance",
        type=float,
        default=0.0,
        help="Evolution dedupe distance for generated candidates.",
    )
    parser.add_argument(
        "--evolution-max-changed-keys",
        type=int,
        default=8,
        help="Max changed keys per evolution candidate.",
    )
    parser.add_argument(
        "--evolution-policy-scale",
        choices=["auto", "micro", "macro"],
        default="auto",
        help="Evolution operator policy scale.",
    )
    parser.add_argument(
        "--evolution-ast-max-complexity-score",
        type=float,
        default=120.0,
        help="Max complexity score for patch_ast mode.",
    )
    parser.add_argument(
        "--evolution-ast-max-redundancy-similarity",
        type=float,
        default=1.0,
        help="Max redundancy similarity for patch_ast mode.",
    )
    parser.add_argument(
        "--evolution-patch-max-attempts",
        type=int,
        default=8,
        help="Max generation attempts per variant in patch_ast mode.",
    )
    parser.add_argument(
        "--evolution-llm-model",
        default="gpt-5.2",
        help="LLM model for evolution proposer/critic when --use-codex-exec is enabled.",
    )
    parser.add_argument(
        "--evolution-llm-effort",
        default="high",
        help="LLM effort hint for evolution proposer.",
    )
    parser.add_argument(
        "--evolution-llm-timeout-sec",
        type=int,
        default=300,
        help="Timeout for LLM proposer/critic calls in evolution planner.",
    )
    parser.add_argument(
        "--evolution-llm-verify-semantic",
        action="store_true",
        default=True,
        help="Enable LLM semantic consistency verification in patch_ast mode.",
    )
    parser.add_argument(
        "--no-evolution-llm-verify-semantic",
        action="store_false",
        dest="evolution_llm_verify_semantic",
        help="Disable LLM semantic consistency verification in patch_ast mode.",
    )
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
