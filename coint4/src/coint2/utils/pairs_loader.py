"""Utilities for loading pair universes from YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml


def _parse_pair_entry(entry: Any) -> dict[str, Any] | None:
    if isinstance(entry, str):
        if "/" not in entry:
            return None
        sym1, sym2 = entry.split("/", 1)
        return {"symbol1": sym1, "symbol2": sym2, "beta": 1.0, "alpha": 0.0}

    if not isinstance(entry, dict):
        return None

    sym1 = entry.get("symbol1")
    sym2 = entry.get("symbol2")
    if not sym1 or not sym2:
        pair_str = entry.get("pair")
        if isinstance(pair_str, str) and "/" in pair_str:
            sym1, sym2 = pair_str.split("/", 1)

    if not sym1 or not sym2:
        return None

    metrics = entry.get("metrics") or {}
    beta = entry.get("beta", metrics.get("beta", 1.0))
    alpha = entry.get("alpha", metrics.get("alpha", 0.0))

    return {"symbol1": sym1, "symbol2": sym2, "beta": beta, "alpha": alpha}


def load_pairs(pairs_file: str | Path) -> list[dict[str, Any]]:
    """Load pairs from a universe YAML file (supports multiple formats)."""
    path = Path(pairs_file)
    if not path.exists():
        raise FileNotFoundError(f"Pairs file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    pairs_data = data.get("pairs", data) if isinstance(data, dict) else data
    if not isinstance(pairs_data, Iterable):
        return []

    pairs: list[dict[str, Any]] = []
    for entry in pairs_data:
        parsed = _parse_pair_entry(entry)
        if parsed:
            pairs.append(parsed)
    return pairs


def load_pair_tuples(pairs_file: str | Path) -> list[tuple[str, str]]:
    """Load pairs and return as (symbol1, symbol2) tuples."""
    pairs = load_pairs(pairs_file)
    seen = set()
    tuples: list[tuple[str, str]] = []
    for entry in pairs:
        pair = (entry["symbol1"], entry["symbol2"])
        if pair in seen:
            continue
        seen.add(pair)
        tuples.append(pair)
    return tuples
