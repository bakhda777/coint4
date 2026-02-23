#!/usr/bin/env python3
"""Build a wide pairs universe for dd-focus fast-loops (tailguard).

Goal: fix dd-focus width (min_pairs>=20) without touching risk/stop axes by
providing a larger fixed `walk_forward.pairs_file` universe.

Notes:
- This script is deterministic: stable ordering + exact dedup on (symbol1, symbol2).
- Supports multiple source YAML formats via `coint2.utils.pairs_loader.load_pairs`.
- Applies denylist from a reference universe (default: pruned_v2).

Run (from app-root `coint4/`):
  PYTHONPATH=src ./.venv/bin/python scripts/universe/build_ddfocus_wide_pairs_universe.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from coint2.utils.pairs_loader import load_pairs


def _quote_ccy(symbol: str) -> str:
    for suffix in ("USDT", "USDC", "BUSD", "FDUSD"):
        if str(symbol).endswith(suffix):
            return suffix
    return "OTHER"


def _load_denylist(path: Path) -> List[str]:
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return []
    deny = data.get("metadata", {}).get("pruning", {}).get("denylisted_symbols", [])
    if not isinstance(deny, list):
        return []
    out: List[str] = []
    for sym in deny:
        sym = str(sym or "").strip()
        if sym:
            out.append(sym)
    return sorted(set(out))


def _iter_source_pairs(sources: Iterable[Path]) -> Iterable[Tuple[str, str]]:
    for src in sources:
        for rec in load_pairs(src):
            s1 = str(rec.get("symbol1") or "").strip()
            s2 = str(rec.get("symbol2") or "").strip()
            if not s1 or not s2 or s1 == s2:
                continue
            yield (s1, s2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default="configs/universe/ddfocus_wide_pairs_universe_v01.yaml",
        help="Output universe path (relative to app-root).",
    )
    ap.add_argument(
        "--source",
        action="append",
        default=[
            "configs/universe/pruned_v2_pairs_universe.yaml",
            "artifacts/universe/20260119_relaxed8_strict_preholdout_v2/pairs_universe.yaml",
            "artifacts/universe/20260119_relaxed6_w3m2_fixed/pairs_universe.yaml",
            "artifacts/universe/20260119_relaxed5_w3m2_fixed/pairs_universe.yaml",
        ],
        help="Source universe YAML (repeatable). Order matters for stable output.",
    )
    ap.add_argument(
        "--denylist-from",
        default="configs/universe/pruned_v2_pairs_universe.yaml",
        help="Universe YAML to source denylisted_symbols from (metadata.pruning.denylisted_symbols).",
    )
    ap.add_argument(
        "--same-quote-only",
        action="store_true",
        help="Keep only pairs where both legs share the same quote (USDT/USDC/BUSD/FDUSD).",
    )
    args = ap.parse_args()

    app_root = Path(__file__).resolve().parents[2]
    out_path = (app_root / str(args.out)).resolve()
    sources = [(app_root / str(p)).resolve() for p in list(args.source or [])]
    deny_src = (app_root / str(args.denylist_from)).resolve()

    missing = [p for p in sources if not p.exists()]
    if missing:
        raise SystemExit("Missing sources:\n" + "\n".join([f"- {p}" for p in missing]))

    deny = set(_load_denylist(deny_src))

    seen: set[Tuple[str, str]] = set()
    pairs: list[dict[str, Any]] = []
    dropped_deny = 0
    dropped_quote = 0
    for s1, s2 in _iter_source_pairs(sources):
        if s1 in deny or s2 in deny:
            dropped_deny += 1
            continue
        if args.same_quote_only and _quote_ccy(s1) != _quote_ccy(s2):
            dropped_quote += 1
            continue
        key = (s1, s2)
        if key in seen:
            continue
        seen.add(key)
        pairs.append({"symbol1": s1, "symbol2": s2})

    meta: Dict[str, Any] = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "type": "ddfocus_wide_pairs_universe",
        "sources": [str(p.relative_to(app_root)) for p in sources],
        "denylist_from": str(deny_src.relative_to(app_root)) if deny_src.exists() else str(deny_src),
        "denylisted_symbols": sorted(deny),
        "filters": {
            "same_quote_only": bool(args.same_quote_only),
        },
        "counts": {
            "pairs_total": len(pairs),
            "dropped_denylist_hits": int(dropped_deny),
            "dropped_quote_mismatch": int(dropped_quote),
        },
    }
    payload = {"metadata": meta, "pairs": pairs}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.dump(payload, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    print(f"✅ Wrote: {out_path.relative_to(app_root)}")
    print(f"  pairs: {len(pairs)}")
    if deny:
        print(f"  denylisted_symbols: {len(deny)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

