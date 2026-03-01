#!/usr/bin/env python3
"""Build a QuantaAlpha-style "final factor pool" (pair-crypto adaptation).

Pool source of truth:
- rollup run_index.csv (metrics across windows)
- evolution decisions (hypothesis/IR/lineage) when available
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from coint2.ops.evolution_targeting import VariantDiagnostics, build_variant_diagnostics


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _extract_candidate_id_from_variant_id(variant_id: str) -> str | None:
    text = str(variant_id or "").strip()
    marker = "_v"
    pos = text.find(marker)
    if pos < 0:
        return None
    rest = text[pos:]
    parts = rest.split("_", 2)
    if len(parts) < 3:
        return None
    candidate = str(parts[2]).strip()
    return candidate or None


def _matches_all(text: str, needles: Sequence[str]) -> bool:
    hay = str(text or "").lower()
    for needle in needles:
        token = str(needle or "").strip().lower()
        if token and token not in hay:
            return False
    return True


def _load_run_index_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


@dataclass(frozen=True, slots=True)
class _CandidateMeta:
    candidate_id: str
    parents: tuple[str, ...]
    patch_ir: dict[str, Any] | None
    patch_path: str | None


def _load_candidate_meta(decisions_dir: Path, *, controller_group: str | None) -> dict[str, _CandidateMeta]:
    by_id: dict[str, _CandidateMeta] = {}
    if not decisions_dir.exists():
        return by_id

    for path in sorted(decisions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if controller_group is not None and str(payload.get("controller_group") or "").strip() != controller_group:
            continue
        proposals = payload.get("proposals")
        if not isinstance(proposals, list):
            continue
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            cid = str(proposal.get("candidate_id") or "").strip()
            if not cid:
                continue
            parents_raw = proposal.get("parents") or []
            parents = tuple(str(item) for item in parents_raw if str(item).strip()) if isinstance(parents_raw, list) else ()
            patch_ir = proposal.get("patch_ir") if isinstance(proposal.get("patch_ir"), dict) else None
            patch_path = str(proposal.get("patch_path") or "").strip() or None
            if cid in by_id:
                continue
            by_id[cid] = _CandidateMeta(candidate_id=cid, parents=parents, patch_ir=patch_ir, patch_path=patch_path)
    return by_id


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Factor Pool (Pair-Crypto)",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- controller_group: `{payload['controller_group']}`",
        f"- run_index: `{payload['run_index']}`",
        f"- contains: `{', '.join(payload['contains'])}`",
        f"- top: `{payload['top_n']}`",
        "",
        "## Top variants",
        "",
        "| rank | candidate_id | windows | worst_robust_sharpe | worst_dd_pct | trades | pairs | parents | thesis |",
        "|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in payload.get("top", []):
        thesis = str(row.get("hypothesis_thesis") or "").strip().replace("\n", " ")
        if len(thesis) > 120:
            thesis = thesis[:117] + "..."
        parents = ",".join(row.get("parents") or [])
        lines.append(
            "| {rank} | {candidate_id} | {windows} | {worst_robust_sharpe:.4f} | {worst_dd_pct:.4f} | {trades:.0f} | {pairs:.0f} | {parents} | {thesis} |".format(
                rank=row["rank"],
                candidate_id=row.get("candidate_id") or "-",
                windows=row.get("windows") or 0,
                worst_robust_sharpe=float(row.get("worst_robust_sharpe") or 0.0),
                worst_dd_pct=float(row.get("worst_dd_pct") or 0.0),
                trades=float(row.get("worst_trades") or 0.0),
                pairs=float(row.get("worst_pairs") or 0.0),
                parents=parents or "-",
                thesis=thesis or "-",
            )
        )
    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build factor pool from run_index + evolution decisions.")
    parser.add_argument("--controller-group", required=True)
    parser.add_argument("--run-index", required=True, help="Path to rollup run_index.csv.")
    parser.add_argument("--decisions-dir", required=True, help="Directory with decision JSON files.")
    parser.add_argument("--contains", action="append", default=[], help="Filter token over run metadata (repeatable).")
    parser.add_argument("--top", type=int, default=20, help="Top variants to include.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    controller_group = str(args.controller_group or "").strip()
    run_index_path = Path(args.run_index)
    decisions_dir = Path(args.decisions_dir)
    contains = [str(token).strip() for token in list(args.contains or []) if str(token).strip()]

    rows = _load_run_index_rows(run_index_path)
    if contains:
        rows = [row for row in rows if _matches_all(" | ".join(row.values()), contains)]

    diagnostics = build_variant_diagnostics(rows, contains=contains, include_noncompleted=False)
    meta = _load_candidate_meta(decisions_dir, controller_group=controller_group)

    top_n = max(1, int(args.top))
    out_rows: list[dict[str, Any]] = []
    for rank, diag in enumerate(diagnostics[:top_n], start=1):
        candidate_id = _extract_candidate_id_from_variant_id(diag.variant_id)
        candidate_meta = meta.get(candidate_id or "")
        hypothesis_thesis = None
        factors = None
        patch_path = None
        parents = ()
        if candidate_meta is not None and candidate_meta.patch_ir is not None:
            hyp = candidate_meta.patch_ir.get("hypothesis")
            if isinstance(hyp, dict):
                hypothesis_thesis = str(hyp.get("thesis") or "").strip() or None
            raw_factors = candidate_meta.patch_ir.get("factors")
            if isinstance(raw_factors, list):
                factors = raw_factors
            patch_path = candidate_meta.patch_path
            parents = candidate_meta.parents

        out_rows.append(
            {
                "rank": rank,
                "run_group": diag.run_group,
                "variant_id": diag.variant_id,
                "candidate_id": candidate_id,
                "sample_config_path": diag.sample_config_path,
                "windows": diag.windows,
                "worst_robust_sharpe": diag.worst_robust_sharpe,
                "worst_dd_pct": diag.worst_dd_pct,
                "worst_trades": diag.worst_trades,
                "worst_pairs": diag.worst_pairs,
                "parents": list(parents),
                "patch_path": patch_path,
                "hypothesis_thesis": hypothesis_thesis,
                "factors": factors,
            }
        )

    payload = {
        "generated_at": _utc_now(),
        "controller_group": controller_group,
        "run_index": str(run_index_path),
        "contains": contains,
        "top_n": top_n,
        "top": out_rows,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_render_markdown(payload), encoding="utf-8")

    print(f"Wrote JSON: {output_json}")
    print(f"Wrote MD:   {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

