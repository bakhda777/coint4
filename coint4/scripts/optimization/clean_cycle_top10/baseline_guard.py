"""Baseline freeze guard for clean-cycle TOP-10.

Baseline is considered "frozen" when a sentinel file exists in BASELINE_DIR:
  BASELINE_FROZEN.txt

Any script/runner that writes into BASELINE_DIR should call `refuse_if_frozen(...)`
unless the user explicitly opted into overwrite via a flag (e.g. --allow-overwrite).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


SENTINEL_FILENAME = "BASELINE_FROZEN.txt"


def sentinel_path(baseline_dir: Path) -> Path:
    return Path(baseline_dir) / SENTINEL_FILENAME


def is_frozen(baseline_dir: Path) -> bool:
    return sentinel_path(baseline_dir).exists()


def read_sentinel(baseline_dir: Path) -> Dict[str, Any]:
    path = sentinel_path(baseline_dir)
    if not path.exists():
        raise FileNotFoundError(str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in sentinel: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"sentinel must be a JSON object: {path}")
    return payload


def refuse_if_frozen(*, baseline_dir: Path, allow_overwrite: bool, action: str) -> None:
    """Refuse writes into baseline_dir if frozen and overwrite is not explicitly allowed."""
    if allow_overwrite:
        return
    path = sentinel_path(baseline_dir)
    if path.exists():
        raise SystemExit(
            f"refusing to {action}: baseline is frozen (sentinel exists: {path}). "
            "Use --allow-overwrite to proceed."
        )

