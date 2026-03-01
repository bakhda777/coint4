#!/usr/bin/env python3
"""Build reproducibility + ablation parity report for evolution decisions."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class _RunScore:
    run_group: str
    best_score: float | None
    best_worst_robust_sharpe: float | None
    worst_dd_pct: float | None


def _to_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _load_run_scores(path: Path) -> dict[str, _RunScore]:
    by_group: dict[str, dict[str, float | None]] = {}
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            group = str(row.get("run_group") or "").strip()
            if not group:
                continue
            score = _to_float(row.get("score"))
            sharpe = _to_float(row.get("worst_robust_sharpe") or row.get("sharpe_ratio_abs"))
            dd = _to_float(row.get("worst_dd_pct") or row.get("max_drawdown_on_equity"))
            state = by_group.setdefault(group, {"best_score": None, "best_sharpe": None, "worst_dd": None})
            prev_score = state["best_score"]
            if score is not None and (prev_score is None or score > prev_score):
                state["best_score"] = score
            prev_sharpe = state["best_sharpe"]
            if sharpe is not None and (prev_sharpe is None or sharpe > prev_sharpe):
                state["best_sharpe"] = sharpe
            prev_dd = state["worst_dd"]
            if dd is not None:
                dd_abs = abs(dd)
                if prev_dd is None or dd_abs > prev_dd:
                    state["worst_dd"] = dd_abs
    return {
        group: _RunScore(
            run_group=group,
            best_score=payload["best_score"],
            best_worst_robust_sharpe=payload["best_sharpe"],
            worst_dd_pct=payload["worst_dd"],
        )
        for group, payload in by_group.items()
    }


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Reproducibility + Ablation Parity Report",
        "",
        f"- controller_group: `{payload['controller_group']}`",
        f"- decisions_total: `{payload['decisions_total']}`",
        f"- reproducibility_ok: `{payload['reproducibility_ok']}`",
        "",
        "## Ablation",
        "",
        "| ablation | decisions | avg_best_score | avg_best_worst_robust_sharpe | avg_worst_dd_pct |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload.get("ablation", []):
        lines.append(
            "| {name} | {decisions} | {avg_best_score} | {avg_best_worst_robust_sharpe} | {avg_worst_dd_pct} |".format(
                name=row["name"],
                decisions=row["decisions"],
                avg_best_score="-" if row["avg_best_score"] is None else f"{row['avg_best_score']:.4f}",
                avg_best_worst_robust_sharpe="-"
                if row["avg_best_worst_robust_sharpe"] is None
                else f"{row['avg_best_worst_robust_sharpe']:.4f}",
                avg_worst_dd_pct="-" if row["avg_worst_dd_pct"] is None else f"{row['avg_worst_dd_pct']:.4f}",
            )
        )
    lines.extend(
        [
            "",
            "## Parity checklist",
            "",
            f"- evaluator_v2_present: `{payload['parity']['evaluator_v2_present']}`",
            f"- proposer_script_present: `{payload['parity']['proposer_script_present']}`",
            f"- critic_script_present: `{payload['parity']['critic_script_present']}`",
            f"- orchestrator_script_present: `{payload['parity']['orchestrator_script_present']}`",
            f"- transfer_report_script_present: `{payload['parity']['transfer_report_script_present']}`",
            f"- patch_ast_module_present: `{payload['parity']['patch_ast_module_present']}`",
            f"- patch_gates_module_present: `{payload['parity']['patch_gates_module_present']}`",
            f"- semantic_gate_script_present: `{payload['parity']['semantic_gate_script_present']}`",
            f"- factor_pool_script_present: `{payload['parity']['factor_pool_script_present']}`",
            f"- quantaalpha_parity_doc_present: `{payload['parity']['quantaalpha_parity_doc_present']}`",
        ]
    )
    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build parity/ablation report for evolution loop.")
    parser.add_argument("--controller-group", required=True)
    parser.add_argument("--run-index", required=True, help="Path to rollup run_index.csv.")
    parser.add_argument("--decisions-dir", required=True, help="Path to decisions directory.")
    parser.add_argument("--output-json", required=True, help="Output JSON path.")
    parser.add_argument("--output-md", required=True, help="Output Markdown path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_scores = _load_run_scores(Path(args.run_index))
    decisions_dir = Path(args.decisions_dir)
    decisions_payload: list[dict[str, Any]] = []
    for path in sorted(decisions_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        if str(payload.get("controller_group") or "").strip() != str(args.controller_group):
            continue
        decisions_payload.append(payload)

    ablation_rows = [
        {"name": "llm_off", "decisions": 0, "scores": [], "sharpes": [], "dds": []},
        {"name": "llm_on", "decisions": 0, "scores": [], "sharpes": [], "dds": []},
    ]
    for payload in decisions_payload:
        llm_used = bool(((payload.get("llm_policy") or {}) if isinstance(payload.get("llm_policy"), dict) else {}).get("used"))
        bucket = ablation_rows[1] if llm_used else ablation_rows[0]
        bucket["decisions"] += 1
        group = str(payload.get("run_group") or "").strip()
        score = run_scores.get(group)
        if score is None:
            continue
        if score.best_score is not None:
            bucket["scores"].append(score.best_score)
        if score.best_worst_robust_sharpe is not None:
            bucket["sharpes"].append(score.best_worst_robust_sharpe)
        if score.worst_dd_pct is not None:
            bucket["dds"].append(score.worst_dd_pct)

    ablation = []
    for row in ablation_rows:
        ablation.append(
            {
                "name": row["name"],
                "decisions": row["decisions"],
                "avg_best_score": _avg(row["scores"]),
                "avg_best_worst_robust_sharpe": _avg(row["sharpes"]),
                "avg_worst_dd_pct": _avg(row["dds"]),
            }
        )

    reproducibility_ok = True
    for payload in decisions_payload:
        rng = payload.get("rng")
        if not isinstance(rng, dict):
            reproducibility_ok = False
            break
        if str(rng.get("algorithm") or "") != "PCG64":
            reproducibility_ok = False
            break
        if "state" not in rng:
            reproducibility_ok = False
            break

    repo_root = Path(__file__).resolve().parents[3]
    parity = {
        "evaluator_v2_present": (repo_root / "coint4/src/coint2/ops/evaluator.py").exists(),
        "proposer_script_present": (repo_root / "coint4/scripts/optimization/evolve_next_batch.py").exists(),
        "critic_script_present": (repo_root / "coint4/scripts/optimization/reflect_next_action.py").exists(),
        "orchestrator_script_present": (repo_root / "coint4/scripts/optimization/evolution_orchestrate.py").exists(),
        "transfer_report_script_present": (repo_root / "coint4/scripts/optimization/transfer_generalization_report.py").exists(),
        "patch_ast_module_present": (repo_root / "coint4/src/coint2/ops/config_patch_ast.py").exists(),
        "patch_gates_module_present": (repo_root / "coint4/src/coint2/ops/config_patch_gates.py").exists(),
        "semantic_gate_script_present": (repo_root / "coint4/scripts/optimization/semantic_consistency_gate.py").exists(),
        "factor_pool_script_present": (repo_root / "coint4/scripts/optimization/build_factor_pool.py").exists(),
        "quantaalpha_parity_doc_present": (repo_root / "docs/quantaalpha_paircrypto_parity.md").exists(),
    }

    payload = {
        "controller_group": args.controller_group,
        "decisions_total": len(decisions_payload),
        "reproducibility_ok": reproducibility_ok,
        "ablation": ablation,
        "parity": parity,
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
