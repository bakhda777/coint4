#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lib_pipeline import DEFAULT_CARDS_DIR, DEFAULT_SYNTHESIS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize markdown docs from cards")
    parser.add_argument("--cards-dir", default=str(DEFAULT_CARDS_DIR), help="Directory with *.json cards")
    parser.add_argument("--output-dir", default=str(DEFAULT_SYNTHESIS_DIR), help="Synthesis output dir")
    parser.add_argument("--top-ideas", type=int, default=10, help="How many ideas to keep in executive summary")
    return parser.parse_args()


def _load_cards(cards_dir: Path) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for path in sorted(cards_dir.glob("*.json")):
        if path.name.endswith(".schema.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("paper_id"), str):
            cards.append(payload)
    return cards


def _title(card: dict[str, Any]) -> str:
    return str(card.get("title") or card.get("paper_id") or "unknown")


def _paper_id(card: dict[str, Any]) -> str:
    return str(card.get("paper_id") or "unknown")


def _year(card: dict[str, Any]) -> str:
    value = card.get("year")
    return str(value) if value is not None else "n/a"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _as_list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str):
            normalized = item.strip()
            if normalized:
                result.append(normalized)
    return result


def _impact_weight(impact: str) -> int:
    lowered = impact.lower()
    if lowered == "high":
        return 15
    if lowered == "medium":
        return 8
    if lowered == "low":
        return 3
    return 0


def _effort_penalty(effort: str) -> int:
    lowered = effort.lower()
    if lowered == "high":
        return 10
    if lowered == "medium":
        return 4
    if lowered == "low":
        return 0
    return 5


def _classify_bucket(idea_text: str, pitfalls_text: str) -> str:
    text = f"{idea_text} {pitfalls_text}".lower()

    if re.search(r"\boverfit|curve[- ]fit|fragile|leverage\s+spike|no\s+oos|look[- ]ahead|survivorship\b", text):
        return "danger"
    if re.search(r"\b(kalman|hedge\s+ratio|dynamic\s+hedge|ou|half[- ]life|spread\s+model|copula)\b", text):
        return "method_upgrade"
    if re.search(r"\b(drawdown|stop[- ]loss|risk\s+limit|regime|stability|volatility|kill\s+switch)\b", text):
        return "dd_cheap"
    if re.search(r"\b(cost|fee|slippage|funding|turnover|execution|borrow)\b", text):
        return "sharpe_cheap"
    return "method_upgrade"


def _normalize_experiment(card: dict[str, Any], experiment: dict[str, Any]) -> dict[str, Any]:
    idea = _text(experiment.get("idea"))
    expected_effect = _text(experiment.get("expected_effect"))
    why = _text(experiment.get("why"))
    impact = _text(experiment.get("impact") or "medium").lower()
    effort = _text(experiment.get("effort") or "medium").lower()
    wfa_checks = _as_list_of_strings(experiment.get("wfa_checks"))
    if not wfa_checks:
        wfa_checks = [
            "OOS Sharpe",
            "Max Drawdown",
            "Turnover",
            "PSR / стабильность по окнам",
        ]

    pitfalls_joined = " ".join(_as_list_of_strings(card.get("pitfalls")))
    bucket = _classify_bucket(f"{idea} {expected_effect} {why}", pitfalls_joined)

    actionability_score = card.get("actionability_score")
    if not isinstance(actionability_score, int):
        actionability_score = 50

    priority_score = actionability_score + _impact_weight(impact) - _effort_penalty(effort)
    if bucket == "danger":
        priority_score -= 10

    return {
        "paper_id": _paper_id(card),
        "title": _title(card),
        "idea": idea,
        "expected_effect": expected_effect or "Sharpe↑/DD↓ в зависимости от настройки",
        "why": why,
        "impact": impact,
        "effort": effort,
        "wfa_checks": wfa_checks,
        "bucket": bucket,
        "priority_score": max(0, min(100, priority_score)),
        "actionability_score": actionability_score,
    }


def _collect_experiments(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    for card in cards:
        entries = card.get("actionable_experiments")
        if not isinstance(entries, list):
            continue
        for experiment in entries:
            if not isinstance(experiment, dict):
                continue
            normalized = _normalize_experiment(card, experiment)
            if normalized["idea"]:
                experiments.append(normalized)
    return experiments


def build_methods_overview(cards: list[dict[str, Any]]) -> str:
    now = datetime.now(timezone.utc).isoformat()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for card in cards:
        families = card.get("method_family")
        if not isinstance(families, list) or not families:
            groups["unspecified"].append(card)
            continue
        for family in families:
            if isinstance(family, str) and family.strip():
                groups[family.strip()].append(card)

    lines = ["# Methods Overview", "", f"- Generated at: {now}", "", f"- Cards processed: {len(cards)}", ""]
    for family in sorted(groups):
        lines.append(f"## {family}")
        family_cards = sorted(
            groups[family],
            key=lambda item: (
                -(int(item.get("actionability_score", 0)) if isinstance(item.get("actionability_score"), int) else 0),
                _paper_id(item),
            ),
        )
        for card in family_cards:
            findings = _as_list_of_strings(card.get("key_findings"))
            top_finding = findings[0] if findings else ""
            lines.append(
                f"- paper_id={_paper_id(card)} | title={_title(card)} | year={_year(card)} | "
                f"actionability={card.get('actionability_score', 'n/a')}"
            )
            if top_finding:
                lines.append(f"- key finding: {top_finding}")
        lines.append("")
    if not groups:
        lines.extend(["## No Data", "- No cards found", ""])
    return "\n".join(lines)


def build_pitfalls(cards: list[dict[str, Any]]) -> str:
    now = datetime.now(timezone.utc).isoformat()
    counter: Counter[str] = Counter()
    refs: dict[str, set[str]] = defaultdict(set)

    for card in cards:
        paper_id = _paper_id(card)
        for pitfall in _as_list_of_strings(card.get("pitfalls")):
            counter[pitfall] += 1
            refs[pitfall].add(paper_id)

    lines = ["# Pitfalls", "", f"- Generated at: {now}", "", f"- Cards processed: {len(cards)}", ""]
    if not counter:
        lines.append("- No pitfalls found in cards")
        lines.append("")
        return "\n".join(lines)

    for pitfall, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        linked = ", ".join(sorted(refs[pitfall]))
        lines.append(f"- {pitfall}")
        lines.append(f"- mentions: {count}")
        lines.append(f"- supporting papers: {linked}")
    lines.append("")
    return "\n".join(lines)


def build_backlog(experiments: list[dict[str, Any]], cards_count: int) -> str:
    now = datetime.now(timezone.utc).isoformat()

    bucket_titles = {
        "sharpe_cheap": "Самое дешёвое улучшение Sharpe",
        "dd_cheap": "Самое дешёвое улучшение DD",
        "method_upgrade": "Методические апгрейды спреда/hedge ratio",
        "danger": "Опасные/хрупкие идеи",
    }

    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in bucket_titles}
    for experiment in experiments:
        grouped.setdefault(experiment["bucket"], []).append(experiment)

    lines = [
        "# Backlog Experiments",
        "",
        f"- Generated at: {now}",
        f"- Cards processed: {cards_count}",
        f"- Experiments extracted: {len(experiments)}",
        "",
    ]

    for bucket in ("sharpe_cheap", "dd_cheap", "method_upgrade", "danger"):
        lines.append(f"## {bucket_titles[bucket]}")
        bucket_items = sorted(
            grouped.get(bucket, []),
            key=lambda item: (-int(item["priority_score"]), item["paper_id"], item["idea"]),
        )[:15]

        if not bucket_items:
            lines.append("- Нет идей в этой группе")
            lines.append("")
            continue

        for item in bucket_items:
            checks = ", ".join(item["wfa_checks"])
            lines.append(f"- Что меняем: {item['idea']}")
            lines.append(f"- Ожидаемый эффект: {item['expected_effect']}")
            lines.append(f"- Поддержка (paper_id): {item['paper_id']}")
            lines.append(f"- Что проверять в WFA/robust ranking: {checks}")
            if item["why"]:
                lines.append(f"- Почему это может сработать: {item['why']}")
        lines.append("")

    return "\n".join(lines)


def build_executive_summary(experiments: list[dict[str, Any]], top_ideas: int) -> tuple[str, list[dict[str, Any]]]:
    now = datetime.now(timezone.utc).isoformat()
    ordered = sorted(
        experiments,
        key=lambda item: (
            -int(item["actionability_score"]),
            -int(item["priority_score"]),
            item["paper_id"],
            item["idea"],
        ),
    )

    top = ordered[:top_ideas]

    lines = [
        "# Executive Summary",
        "",
        f"- Generated at: {now}",
        f"- Ideas considered: {len(experiments)}",
        f"- Top ideas listed: {len(top)}",
        "",
        "## Что внедрять в стратегию в первую очередь",
    ]

    if not top:
        lines.append("- Недостаточно карточек с actionable_experiments")
        lines.append("")
        return "\n".join(lines), []

    for idx, item in enumerate(top, start=1):
        lines.append(
            f"- {idx}. [{item['actionability_score']}] {item['idea']} | эффект: {item['expected_effect']} | "
            f"paper_id={item['paper_id']}"
        )
    lines.append("")

    lines.append("## Почему это приоритет")
    lines.append("- Идеи ранжированы по actionability_score из карточек и по ожидаемому impact/effort.")
    lines.append("- При равенстве приоритет отдан идеям с прямым влиянием на Sharpe↑ и DD↓ в OOS/WFA.")
    lines.append("")

    return "\n".join(lines), top


def main() -> int:
    args = parse_args()
    cards_dir = Path(args.cards_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cards = _load_cards(cards_dir)
    experiments = _collect_experiments(cards)

    methods_path = output_dir / "methods_overview.md"
    pitfalls_path = output_dir / "pitfalls.md"
    backlog_path = output_dir / "backlog_experiments.md"
    executive_path = output_dir / "executive_summary.md"

    methods_path.write_text(build_methods_overview(cards), encoding="utf-8")
    pitfalls_path.write_text(build_pitfalls(cards), encoding="utf-8")
    backlog_path.write_text(build_backlog(experiments, cards_count=len(cards)), encoding="utf-8")
    executive_summary_text, top_ideas = build_executive_summary(experiments, top_ideas=args.top_ideas)
    executive_path.write_text(executive_summary_text, encoding="utf-8")

    print(
        json.dumps(
            {
                "cards": len(cards),
                "experiments": len(experiments),
                "methods_overview": str(methods_path),
                "pitfalls": str(pitfalls_path),
                "backlog": str(backlog_path),
                "executive_summary": str(executive_path),
                "top_ideas": [
                    {
                        "rank": idx + 1,
                        "paper_id": item["paper_id"],
                        "actionability_score": item["actionability_score"],
                        "idea": item["idea"],
                        "expected_effect": item["expected_effect"],
                    }
                    for idx, item in enumerate(top_ideas)
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
