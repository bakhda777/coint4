#!/usr/bin/env python3
"""Transfer/generalization report across OOS windows for pair-crypto variants."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


_OOS_RE = re.compile(r"_oos(\d{8})_(\d{8})")


@dataclass(frozen=True)
class _Entry:
    run_id: str
    run_group: str
    config_path: str
    status: str
    metrics_present: bool
    sharpe: float | None
    dd_pct: float | None
    pnl: float | None


def _to_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _kind_and_base_id(run_id: str) -> tuple[str | None, str]:
    text = str(run_id or "").strip()
    if text.startswith("holdout_"):
        return "holdout", text[len("holdout_") :]
    if text.startswith("stress_"):
        return "stress", text[len("stress_") :]
    return None, text


def _variant_id(base_id: str) -> str:
    return _OOS_RE.sub("", str(base_id or ""))


def _parse_window(base_id: str) -> str:
    match = _OOS_RE.search(base_id)
    if not match:
        return "-"
    return f"{match.group(1)}-{match.group(2)}"


def _matches_all(text: str, needles: Sequence[str]) -> bool:
    hay = str(text or "").lower()
    for needle in needles:
        token = str(needle or "").strip().lower()
        if token and token not in hay:
            return False
    return True


def _load_rows(path: Path) -> list[_Entry]:
    rows: list[_Entry] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                _Entry(
                    run_id=str(row.get("run_id") or "").strip(),
                    run_group=str(row.get("run_group") or "").strip(),
                    config_path=str(row.get("config_path") or "").strip(),
                    status=str(row.get("status") or "").strip(),
                    metrics_present=_to_bool(row.get("metrics_present")),
                    sharpe=_to_float(row.get("worst_robust_sharpe") or row.get("sharpe_ratio_abs")),
                    dd_pct=_to_float(row.get("worst_dd_pct") or row.get("max_drawdown_on_equity")),
                    pnl=_to_float(row.get("worst_robust_pnl") or row.get("total_pnl")),
                )
            )
    return rows


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Transfer / Generalization Report",
        "",
        f"- source: `{payload['run_index']}`",
        f"- contains: `{', '.join(payload['contains'])}`",
        f"- variants: `{payload['variants_count']}`",
        "",
        "## Top variants",
        "",
        "| variant | windows | worst_robust_sharpe | std_robust_sharpe | transfer_score | worst_dd_pct |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("top", []):
        lines.append(
            "| {variant_id} | {windows} | {worst_robust_sharpe:.4f} | {std_robust_sharpe:.4f} | "
            "{transfer_score:.4f} | {worst_dd_pct:.4f} |".format(**row)
        )
    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build transfer/generalization report from run_index.")
    parser.add_argument("--run-index", required=True, help="Path to rollup run_index.csv.")
    parser.add_argument("--contains", action="append", default=[], help="Filter token over metadata (repeatable).")
    parser.add_argument("--top", type=int, default=20, help="Top variants to include in report.")
    parser.add_argument("--output-json", required=True, help="Output JSON path.")
    parser.add_argument("--output-md", help="Optional output Markdown path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_index_path = Path(args.run_index)
    rows = _load_rows(run_index_path)
    contains = [str(token).strip() for token in list(args.contains or []) if str(token).strip()]

    by_variant: dict[str, dict[str, Any]] = {}
    for row in rows:
        kind, base_id = _kind_and_base_id(row.run_id)
        if kind == "stress":
            continue

        meta = " | ".join((row.run_group, row.run_id, row.config_path))
        if contains and not _matches_all(meta, contains):
            continue
        if row.status.lower() != "completed" or not row.metrics_present:
            continue
        if row.sharpe is None or row.dd_pct is None:
            continue

        robust_sharpe = float(row.sharpe)
        robust_dd = abs(float(row.dd_pct))
        robust_pnl = float(row.pnl) if row.pnl is not None else None

        variant_id = _variant_id(base_id)
        state = by_variant.setdefault(
            variant_id,
            {
                "run_group": row.run_group,
                "variant_id": variant_id,
                "windows": [],
                "sample_config_path": row.config_path,
                "robust_sharpes": [],
                "robust_dds": [],
                "robust_pnls": [],
            },
        )
        state["windows"].append(_parse_window(base_id))
        state["robust_sharpes"].append(robust_sharpe)
        state["robust_dds"].append(robust_dd)
        if robust_pnl is not None:
            state["robust_pnls"].append(robust_pnl)

    summary_rows: list[dict[str, Any]] = []
    for variant_id, state in by_variant.items():
        sharpes = state["robust_sharpes"]
        if not sharpes:
            continue
        avg = sum(sharpes) / len(sharpes)
        var = sum((value - avg) ** 2 for value in sharpes) / len(sharpes)
        std = math.sqrt(max(0.0, var))
        worst = min(sharpes)
        worst_dd = max(state["robust_dds"]) if state["robust_dds"] else 0.0
        worst_pnl = min(state["robust_pnls"]) if state["robust_pnls"] else None
        transfer_score = worst - std
        summary_rows.append(
            {
                "run_group": state["run_group"],
                "variant_id": variant_id,
                "sample_config_path": state["sample_config_path"],
                "windows": len(state["windows"]),
                "worst_robust_sharpe": worst,
                "avg_robust_sharpe": avg,
                "std_robust_sharpe": std,
                "worst_dd_pct": worst_dd,
                "worst_robust_pnl": worst_pnl,
                "transfer_score": transfer_score,
                "window_tags": sorted(set(state["windows"])),
            }
        )

    summary_rows.sort(key=lambda row: row["transfer_score"], reverse=True)
    payload = {
        "run_index": str(run_index_path),
        "contains": contains,
        "variants_count": len(summary_rows),
        "top": summary_rows[: max(1, int(args.top))],
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote JSON: {output_json}")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(_render_markdown(payload), encoding="utf-8")
        print(f"Wrote MD:   {output_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
