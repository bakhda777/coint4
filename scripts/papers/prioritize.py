#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from lib_pipeline import (
    DEFAULT_DB_PATH,
    DEFAULT_SYNTHESIS_DIR,
    connect_db,
    finish_run,
    loads_json,
    sqlite_write_section,
    start_run,
)

METRIC_VALUE_PATTERNS = {
    "sharpe": re.compile(r"\bsharpe\b[^\n]{0,30}?[-+]?\d+(?:\.\d+)?", re.IGNORECASE),
    "cagr": re.compile(r"\bcagr\b[^\n]{0,30}?[-+]?\d+(?:\.\d+)?", re.IGNORECASE),
    "mdd": re.compile(r"\b(?:max(?:imum)?\s+drawdown|mdd|drawdown)\b[^\n]{0,30}?[-+]?\d+(?:\.\d+)?", re.IGNORECASE),
    "turnover": re.compile(r"\bturnover\b[^\n]{0,30}?[-+]?\d+(?:\.\d+)?", re.IGNORECASE),
    "sortino": re.compile(r"\bsortino\b[^\n]{0,30}?[-+]?\d+(?:\.\d+)?", re.IGNORECASE),
    "calmar": re.compile(r"\bcalmar\b[^\n]{0,30}?[-+]?\d+(?:\.\d+)?", re.IGNORECASE),
    "win_rate": re.compile(r"\bwin\s*rate\b[^\n]{0,30}?[-+]?\d+(?:\.\d+)?", re.IGNORECASE),
}

RESULT_MARKERS = re.compile(
    r"\b(experiment(?:s)?|backtest(?:ing)?|out[- ]of[- ]sample|oos|walk[- ]forward|results?|ablation|"
    r"sensitivity|bootstrap|cross[- ]validation|rolling window(?:s)?)\b",
    re.IGNORECASE,
)

COST_PATTERNS = {
    "funding": re.compile(r"\bfunding(?:\s+rate(?:s)?)?\b", re.IGNORECASE),
    "perpetual": re.compile(r"\bperpetual(?:s)?\b|\bperp(?:s)?\b", re.IGNORECASE),
    "borrow": re.compile(r"\bborrow(?:ing)?\b|\bloan\s+rate\b|\bmargin\s+interest\b", re.IGNORECASE),
    "fees": re.compile(r"\bfee(?:s)?\b|\bcommission(?:s)?\b|\btransaction(?:al)?\s+cost(?:s)?\b", re.IGNORECASE),
    "slippage": re.compile(r"\bslippage\b", re.IGNORECASE),
    "turnover_control": re.compile(r"\bturnover\b|\bchurn\b|\btrade\s+frequency\b", re.IGNORECASE),
    "spread": re.compile(r"\bbid[- ]ask\s+spread\b|\bspread\s+cost\b", re.IGNORECASE),
}

ROBUSTNESS_PATTERNS = {
    "oos": re.compile(r"\bout[- ]of[- ]sample\b|\boos\b", re.IGNORECASE),
    "wfa": re.compile(r"\bwalk[- ]forward\b|\bwfa\b", re.IGNORECASE),
    "rolling": re.compile(r"\brolling\s+window(?:s)?\b|\bexpanding\s+window(?:s)?\b", re.IGNORECASE),
    "robustness": re.compile(r"\brobust(?:ness)?\b|\bsensitivity\s+analysis\b", re.IGNORECASE),
    "stress": re.compile(r"\bstress\s+test(?:s)?\b|\bbootstrap\b|\bcross[- ]validation\b|\bablation\b", re.IGNORECASE),
}

IMPLEMENTATION_PATTERNS = (
    re.compile(r"\bpseudocode\b|\balgorithm\b|\bstep\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bparameter(?:s)?\b|\bhyperparameter(?:s)?\b|\bthreshold(?:s)?\b", re.IGNORECASE),
    re.compile(r"\blookback\b|\bwindow\s*size\b|\bhalf[- ]life\b", re.IGNORECASE),
    re.compile(r"\bentry\b[^\n]{0,24}\bexit\b|\btrading\s+rule(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bhedge\s+ratio\b|\bbeta\b|\blambda\b|\bz[- ]score\b", re.IGNORECASE),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assign paper priority scores and build queue")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite path")
    parser.add_argument(
        "--queue-path",
        default=str(DEFAULT_SYNTHESIS_DIR / "priority_queue.jsonl"),
        help="Output JSONL queue",
    )
    return parser.parse_args()


def _load_digest(digest_path: str | None) -> str:
    if not digest_path:
        return ""
    path = Path(digest_path)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _truthy_hit(hits: dict[str, Any], key: str) -> bool:
    return bool(hits.get(key, False))


def _contains(pattern: re.Pattern[str], text: str) -> bool:
    return pattern.search(text) is not None


def _metric_subscore(text: str) -> tuple[int, list[str], bool]:
    score = 0
    reasons: list[str] = []
    metric_hits = 0
    for name, pattern in METRIC_VALUE_PATTERNS.items():
        if _contains(pattern, text):
            metric_hits += 1
            score += 2
            reasons.append(f"metric:{name}")

    numeric_tokens = len(re.findall(r"\b\d+(?:\.\d+)?(?:%|bps|x)?\b", text))
    numeric_bonus = min(4, numeric_tokens // 6)
    score += numeric_bonus
    if numeric_bonus > 0:
        reasons.append(f"metric_numeric_density:+{numeric_bonus}")

    has_results_signal = metric_hits >= 2 or _contains(RESULT_MARKERS, text)
    return min(18, score), reasons, has_results_signal


def _is_review_like(title: str, digest: str) -> bool:
    combined = f"{title} {digest}".lower()
    reviewish = re.search(r"\b(survey|review|literature review|overview|state of the art)\b", combined) is not None
    metric_density = len(re.findall(r"\b\d+(?:\.\d+)?(?:%|bps|x)?\b", digest))
    has_result_terms = _contains(RESULT_MARKERS, combined)
    return reviewish and metric_density < 5 and not has_result_terms


def _subscore_costs(text: str, hits: dict[str, Any]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    if _truthy_hit(hits, "funding") or _contains(COST_PATTERNS["funding"], text):
        score += 5
        reasons.append("cost_funding")
    if _truthy_hit(hits, "perpetual") or _contains(COST_PATTERNS["perpetual"], text):
        score += 4
        reasons.append("cost_perpetual")
    if _contains(COST_PATTERNS["borrow"], text):
        score += 3
        reasons.append("cost_borrow")
    if _truthy_hit(hits, "transaction_costs") or _contains(COST_PATTERNS["fees"], text):
        score += 4
        reasons.append("cost_fees")
    if _truthy_hit(hits, "slippage") or _contains(COST_PATTERNS["slippage"], text):
        score += 4
        reasons.append("cost_slippage")
    if _truthy_hit(hits, "turnover") or _contains(COST_PATTERNS["turnover_control"], text):
        score += 3
        reasons.append("cost_turnover_control")
    if _contains(COST_PATTERNS["spread"], text):
        score += 2
        reasons.append("cost_spread")

    return min(25, score), reasons


def _subscore_robustness(text: str, hits: dict[str, Any]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    if _truthy_hit(hits, "out_of_sample") or _contains(ROBUSTNESS_PATTERNS["oos"], text):
        score += 6
        reasons.append("robust_oos")
    if _truthy_hit(hits, "walk_forward") or _contains(ROBUSTNESS_PATTERNS["wfa"], text):
        score += 5
        reasons.append("robust_wfa")
    if _contains(ROBUSTNESS_PATTERNS["rolling"], text):
        score += 4
        reasons.append("robust_rolling_windows")
    if _truthy_hit(hits, "robustness") or _contains(ROBUSTNESS_PATTERNS["robustness"], text):
        score += 4
        reasons.append("robust_sensitivity")
    if _contains(ROBUSTNESS_PATTERNS["stress"], text):
        score += 3
        reasons.append("robust_stress_tests")

    return min(22, score), reasons


def _subscore_method_novelty(text: str, hits: dict[str, Any]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    if _truthy_hit(hits, "kalman"):
        score += 4
        reasons.append("method_kalman")
    if _truthy_hit(hits, "copula"):
        score += 4
        reasons.append("method_copula")
    if _truthy_hit(hits, "regime") or _truthy_hit(hits, "structural_break"):
        score += 4
        reasons.append("method_regime_structural")
    if _truthy_hit(hits, "ou") or _truthy_hit(hits, "half_life"):
        score += 2
        reasons.append("method_ou_half_life")
    if _truthy_hit(hits, "pca") or _truthy_hit(hits, "clustering"):
        score += 2
        reasons.append("method_pca_clustering")
    if _truthy_hit(hits, "rl") or re.search(r"\bmachine learning\b|\bdeep learning\b|\bneural\b", text):
        score += 3
        reasons.append("method_ml_rl")

    return min(18, score), reasons


def _subscore_market(text: str, hits: dict[str, Any]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    if _truthy_hit(hits, "crypto"):
        score += 7
        reasons.append("market_crypto")
    if _truthy_hit(hits, "perpetual"):
        score += 3
        reasons.append("market_perpetual")
    if _truthy_hit(hits, "funding"):
        score += 3
        reasons.append("market_funding")
    if _truthy_hit(hits, "bybit") or _truthy_hit(hits, "binance"):
        score += 2
        reasons.append("market_exchange_specific")
    return min(15, score), reasons


def _subscore_implementation(text: str) -> tuple[int, list[str]]:
    density = sum(1 for pattern in IMPLEMENTATION_PATTERNS if _contains(pattern, text))
    score = min(10, density * 2)
    reasons = [f"implementation_density:{density}"] if density > 0 else []
    return score, reasons


def _subscore_risk_controls(hits: dict[str, Any]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    if _truthy_hit(hits, "drawdown"):
        score += 2
        reasons.append("risk_drawdown")
    if _truthy_hit(hits, "stop_loss"):
        score += 2
        reasons.append("risk_stop_loss")
    if _truthy_hit(hits, "position_sizing"):
        score += 2
        reasons.append("risk_position_sizing")
    if _truthy_hit(hits, "leverage"):
        score += 2
        reasons.append("risk_leverage")
    return min(8, score), reasons


def main() -> int:
    args = parse_args()
    db_path = Path(args.db_path)
    queue_path = Path(args.queue_path)
    queue_path.parent.mkdir(parents=True, exist_ok=True)

    conn = connect_db(db_path)
    run_id = start_run(conn, kind="papers_prioritize", params={"db_path": str(db_path), "queue_path": str(queue_path)})

    rows = conn.execute(
        """
        SELECT id, text_path, file_name, title, year, digest_path,
               keyword_hits_json, dup_group_id, text_len
        FROM papers
        """
    ).fetchall()

    duplicates: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["dup_group_id"]:
            duplicates[row["dup_group_id"]].append(row["id"])

    row_by_id = {row["id"]: row for row in rows}
    canonical_by_group: dict[str, str] = {}
    for group_id, ids in duplicates.items():
        canonical_by_group[group_id] = sorted(
            ids,
            key=lambda paper_id: (
                -(row_by_id[paper_id]["text_len"] or 0),
                paper_id,
            ),
        )[0]

    queue_entries: list[dict[str, Any]] = []
    db_updates: list[tuple[int, str, str]] = []

    for row in rows:
        hits = loads_json(row["keyword_hits_json"], {})
        if not isinstance(hits, dict):
            hits = {}

        digest = _load_digest(row["digest_path"])
        title = row["title"] or row["file_name"] or row["id"]
        combined = f"{title}\n{digest}".lower()

        market_subscore, market_reasons = _subscore_market(combined, hits)
        costs_subscore, cost_reasons = _subscore_costs(combined, hits)
        robustness_subscore, robust_reasons = _subscore_robustness(combined, hits)
        method_novelty_subscore, method_reasons = _subscore_method_novelty(combined, hits)
        implementation_subscore, implementation_reasons = _subscore_implementation(combined)
        risk_subscore, risk_reasons = _subscore_risk_controls(hits)
        metrics_subscore, metric_reasons, has_results_signal = _metric_subscore(combined)

        penalties = 0
        penalty_reasons: list[str] = []

        review_like = _is_review_like(title=title, digest=digest)
        if review_like:
            penalties += 20
            penalty_reasons.append("penalty_review_without_experiments")

        text_len = int(row["text_len"] or 0)
        text_len_penalty = 0
        if text_len < 1200:
            text_len_penalty = 16
        elif text_len < 2500:
            text_len_penalty = 10
        elif text_len < 4000:
            text_len_penalty = 4
        if text_len_penalty > 0:
            penalties += text_len_penalty
            penalty_reasons.append(f"penalty_short_text:{text_len_penalty}")

        if not has_results_signal:
            penalties += 8
            penalty_reasons.append("penalty_missing_results_signal")

        dup_group_id = row["dup_group_id"]
        is_duplicate = 0
        if dup_group_id:
            canonical_id = canonical_by_group.get(dup_group_id)
            if canonical_id and canonical_id != row["id"]:
                penalties += 30
                is_duplicate = 1
                penalty_reasons.append("penalty_duplicate_non_canonical")
            else:
                penalty_reasons.append("duplicate_canonical")

        raw_score = (
            market_subscore
            + costs_subscore
            + robustness_subscore
            + method_novelty_subscore
            + implementation_subscore
            + metrics_subscore
            + risk_subscore
            - penalties
        )
        score = max(0, min(100, int(raw_score)))

        reasons: list[str] = []
        reasons.extend(market_reasons)
        reasons.extend(cost_reasons)
        reasons.extend(robust_reasons)
        reasons.extend(method_reasons)
        reasons.extend(implementation_reasons)
        reasons.extend(metric_reasons)
        reasons.extend(risk_reasons)
        reasons.extend(penalty_reasons)

        sort_key = [
            score,
            robustness_subscore,
            costs_subscore,
            method_novelty_subscore,
            -is_duplicate,
            -text_len_penalty,
        ]

        db_updates.append((score, json.dumps(reasons, ensure_ascii=False, sort_keys=True), row["id"]))
        queue_entries.append(
            {
                "paper_id": row["id"],
                "priority_score": score,
                "title": row["title"],
                "year": row["year"],
                "reasons": reasons,
                "text_path": row["text_path"],
                "robustness_subscore": robustness_subscore,
                "costs_subscore": costs_subscore,
                "method_novelty_subscore": method_novelty_subscore,
                "is_duplicate": is_duplicate,
                "text_len_penalty": text_len_penalty,
                "sort_key": sort_key,
            }
        )

    batch_size = 100
    for start in range(0, len(db_updates), batch_size):
        chunk = db_updates[start : start + batch_size]
        with sqlite_write_section(conn):
            for score, reasons_json, paper_id in chunk:
                conn.execute(
                    "UPDATE papers SET priority_score = ?, priority_reasons_json = ? WHERE id = ?",
                    (score, reasons_json, paper_id),
                )

    queue_entries.sort(
        key=lambda entry: (
            -int(entry["sort_key"][0]),
            -int(entry["sort_key"][1]),
            -int(entry["sort_key"][2]),
            -int(entry["sort_key"][3]),
            -int(entry["sort_key"][4]),
            -int(entry["sort_key"][5]),
            -(int(entry["year"]) if entry.get("year") else 0),
            entry.get("paper_id", ""),
        )
    )

    with queue_path.open("w", encoding="utf-8") as fh:
        for entry in queue_entries:
            fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

    stats = {
        "scored": len(queue_entries),
        "queue_path": str(queue_path),
        "top_score": queue_entries[0]["priority_score"] if queue_entries else None,
        "score_spread": {
            "max": max((entry["priority_score"] for entry in queue_entries), default=None),
            "min": min((entry["priority_score"] for entry in queue_entries), default=None),
        },
    }
    finish_run(conn, run_id, stats)
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
