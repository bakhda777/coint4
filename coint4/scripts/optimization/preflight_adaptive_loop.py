#!/usr/bin/env python3
"""Preflight adaptive-loop config and safely initialize controller state.

Behavior:
- Validates closed-loop search guardrails:
  - search.max_rounds <= 0
  - search.require_all_knobs_before_stop == true
  - search.min_queue_entries >= 12
- Reads controller state from artifacts/wfa/aggregate/<controller_group>/state.json
- Chooses safe init mode from factual state:
  - missing state or done=true  -> --reset --dry-run
  - done=false                  -> --resume --dry-run

Run from app root (coint4/):
  PYTHONPATH=src ./.venv/bin/python scripts/optimization/preflight_adaptive_loop.py \
    --config configs/autopilot/budget1000_closed_loop_20260216.yaml --apply
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import yaml


def _resolve_app_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_under_root(path: str, *, root: Path) -> Path:
    candidate = Path(str(path).strip())
    if candidate.is_absolute():
        return candidate
    return root / candidate


def _venv_python(app_root: Path) -> Path:
    py = app_root / ".venv" / "bin" / "python"
    if py.exists():
        return py
    py3 = app_root / ".venv" / "bin" / "python3"
    if py3.exists():
        return py3
    return Path(sys.executable)


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid YAML config: {path}")
    return data


def _load_state(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid JSON state payload: {path}")
    return data


def _validate_search(cfg: dict[str, Any], *, config_path: Path) -> None:
    search = cfg.get("search") or {}
    if not isinstance(search, dict):
        raise SystemExit(f"Invalid search section in {config_path}")

    max_rounds_raw = search.get("max_rounds")
    try:
        max_rounds = int(max_rounds_raw)
    except (TypeError, ValueError):
        raise SystemExit("config.search.max_rounds must be an integer") from None
    if max_rounds > 0:
        raise SystemExit(
            "Adaptive-loop preflight failed: config.search.max_rounds must be <= 0 (no hard cap)."
        )

    require_all = search.get("require_all_knobs_before_stop")
    if require_all is not True:
        raise SystemExit(
            "Adaptive-loop preflight failed: config.search.require_all_knobs_before_stop must be true."
        )

    min_queue_raw = search.get("min_queue_entries")
    try:
        min_queue = int(min_queue_raw)
    except (TypeError, ValueError):
        raise SystemExit("config.search.min_queue_entries must be an integer") from None
    if min_queue < 12:
        raise SystemExit(
            "Adaptive-loop preflight failed: config.search.min_queue_entries must be >= 12."
        )


def _choose_mode(state: Optional[dict[str, Any]]) -> tuple[str, list[str], str]:
    if not state:
        return (
            "reset",
            ["--reset", "--dry-run"],
            "state_absent",
        )

    if bool(state.get("done")):
        return (
            "reset",
            ["--reset", "--dry-run"],
            "state_done_true",
        )

    return (
        "resume",
        ["--resume", "--dry-run"],
        "state_done_false",
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Preflight adaptive-loop and controller init mode")
    parser.add_argument(
        "--config",
        default="configs/autopilot/budget1000_closed_loop_20260216.yaml",
        help="Autopilot config path (relative to coint4/ unless absolute)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Run autopilot with selected safe mode; default is print-only",
    )
    args = parser.parse_args(argv)

    app_root = _resolve_app_root()
    config_path = _resolve_under_root(args.config, root=app_root)
    cfg = _load_yaml(config_path)
    _validate_search(cfg, config_path=config_path)

    controller_group = str(cfg.get("controller_group") or "").strip()
    if not controller_group:
        run_prefix = str(cfg.get("run_group_prefix") or "").strip()
        if not run_prefix:
            raise SystemExit("config.controller_group or config.run_group_prefix is required")
        controller_group = f"{run_prefix}_autopilot"

    state_path = app_root / "artifacts" / "wfa" / "aggregate" / controller_group / "state.json"
    state = _load_state(state_path)
    mode, mode_flags, reason = _choose_mode(state)

    py = _venv_python(app_root)
    cmd = [
        str(py),
        "scripts/optimization/autopilot_budget1000.py",
        "--config",
        str(config_path.relative_to(app_root)) if config_path.is_relative_to(app_root) else str(config_path),
        *mode_flags,
    ]

    print("[preflight] config:", config_path, flush=True)
    print("[preflight] controller_group:", controller_group, flush=True)
    print("[preflight] state_path:", state_path, flush=True)
    print("[preflight] selected_mode:", mode, flush=True)
    print("[preflight] reason:", reason, flush=True)
    print("[preflight] command:", " ".join(cmd), flush=True)

    if not args.apply:
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = str(app_root / "src")
    result = subprocess.run(cmd, cwd=str(app_root), env=env, check=False)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
