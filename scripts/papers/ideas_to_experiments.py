#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lib_pipeline import DEFAULT_SYNTHESIS_DIR

DEFAULT_REGISTRY_PATH = Path("coint4/experiments/registry.jsonl")
DEFAULT_SCHEMA_PATH = Path("coint4/experiments/registry.schema.json")

IDEA_LINE_PREFIX = "- Что меняем:"
EFFECT_LINE_PREFIX = "- Ожидаемый эффект:"
PAPERS_LINE_PREFIX = "- Поддержка (paper_id):"
CHECKS_LINE_PREFIX = "- Что проверять в WFA/robust ranking:"
WHY_LINE_PREFIX = "- Почему это может сработать:"

EXEC_LINE_RE = re.compile(
    r"^\s*-\s*\d+\.\s*\[(?P<score>\d+)\]\s*(?P<body>.+?)\s*\|\s*эффект:\s*(?P<effect>.+?)\s*\|\s*paper_id=(?P<paper_id>[A-Za-z0-9._-]+)\s*$"
)
PAPER_ID_RE = re.compile(r"\b[0-9a-f]{12}(?:-[0-9a-f]{6}(?:-\d+)?)?\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert synthesized paper ideas into executable experiment registry")
    parser.add_argument(
        "--backlog-path",
        default=str(DEFAULT_SYNTHESIS_DIR / "backlog_experiments.md"),
        help="Path to backlog_experiments.md",
    )
    parser.add_argument(
        "--executive-path",
        default=str(DEFAULT_SYNTHESIS_DIR / "executive_summary.md"),
        help="Path to executive_summary.md",
    )
    parser.add_argument(
        "--registry-path",
        default=str(DEFAULT_REGISTRY_PATH),
        help="Registry JSONL path",
    )
    parser.add_argument(
        "--schema-path",
        default=str(DEFAULT_SCHEMA_PATH),
        help="Registry schema JSON path",
    )
    parser.add_argument("--top", type=int, default=10, help="How many top ideas to map at minimum")
    parser.add_argument("--owner", default="unassigned", help="Owner for new experiments")
    return parser.parse_args()


def _norm(text: str) -> str:
    value = re.sub(r"\s+", " ", text.strip().lower())
    value = re.sub(r"[^a-z0-9а-яё ]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value


def _parse_paper_ids(text: str) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    for match in PAPER_ID_RE.finditer(text.lower()):
        value = match.group(0)
        if value not in out:
            out.append(value)
    return out


def _parse_backlog(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    section = "Unknown"
    current: dict[str, Any] | None = None

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("## "):
            section = line[3:].strip()
            continue

        if line.startswith(IDEA_LINE_PREFIX):
            if current and current.get("idea"):
                records.append(current)
            current = {
                "section": section,
                "idea": line[len(IDEA_LINE_PREFIX) :].strip(),
                "expected_effect": "",
                "linked_papers": [],
                "checks": "",
                "why": "",
            }
            continue

        if not current:
            continue

        if line.startswith(EFFECT_LINE_PREFIX):
            current["expected_effect"] = line[len(EFFECT_LINE_PREFIX) :].strip()
            continue
        if line.startswith(PAPERS_LINE_PREFIX):
            paper_text = line[len(PAPERS_LINE_PREFIX) :].strip()
            current["linked_papers"] = _parse_paper_ids(paper_text)
            continue
        if line.startswith(CHECKS_LINE_PREFIX):
            current["checks"] = line[len(CHECKS_LINE_PREFIX) :].strip()
            continue
        if line.startswith(WHY_LINE_PREFIX):
            current["why"] = line[len(WHY_LINE_PREFIX) :].strip()
            continue

    if current and current.get("idea"):
        records.append(current)

    return records


def _parse_executive(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    items: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        m = EXEC_LINE_RE.match(line)
        if not m:
            continue
        title = m.group("body").strip()
        items.append(
            {
                "idea": title,
                "actionability_score": int(m.group("score")),
                "expected_effect": m.group("effect").strip(),
                "paper_id": m.group("paper_id").strip().lower(),
            }
        )
    return items


def _effect_direction(text: str, metric: str) -> str:
    t = text.lower()
    if metric == "sharpe":
        if re.search(r"(сниж|уменьш|drop|fall).{0,20}sharpe|sharpe.{0,20}(сниж|уменьш|drop|fall)", t):
            return "down"
        if "sharpe" in t and re.search(r"(рост|улучш|повыш|up)", t):
            return "up"
        return "up" if "sharpe" in t else "neutral"

    if metric == "drawdown":
        if re.search(r"(сниж|уменьш|меньш|down|lower).{0,24}(drawdown|dd|просад)", t):
            return "down"
        if re.search(r"(drawdown|dd|просад).{0,24}(сниж|уменьш|меньш|down|lower)", t):
            return "down"
        return "neutral"

    if metric == "turnover":
        if re.search(r"(turnover|оборот).{0,20}(сниж|уменьш|down|lower)", t):
            return "down"
        if re.search(r"(сниж|уменьш|down|lower).{0,20}(turnover|оборот)", t):
            return "down"
        return "neutral"

    return "neutral"


def _build_expected_effect(text: str) -> dict[str, str]:
    return {
        "sharpe": _effect_direction(text, "sharpe"),
        "drawdown": _effect_direction(text, "drawdown"),
        "turnover": _effect_direction(text, "turnover"),
        "notes": text or "Ожидается улучшение risk-adjusted профиля в OOS/WFA.",
    }


def _suggest_changes(idea: str) -> list[str]:
    text = idea.lower()
    changes: list[str] = []

    if any(token in text for token in ("cost", "fee", "slippage", "funding", "turnover")):
        changes.extend(
            [
                "coint4/src/coint2/pipeline/cost_model.py",
                "coint4/src/coint2/pipeline/walk_forward_orchestrator.py",
                "configs/*.yaml (backtest.cost_*)",
            ]
        )
    if any(token in text for token in ("stop", "drawdown", "regime", "risk", "volatility")):
        changes.extend(
            [
                "coint4/src/coint2/pipeline/walk_forward_orchestrator.py",
                "coint4/src/coint2/utils/config.py (backtest/portfolio risk params)",
            ]
        )
    if any(token in text for token in ("kalman", "hedge", "ou", "half-life", "cointegration", "copula")):
        changes.extend(
            [
                "coint4/src/coint2/core/pair_backtester.py",
                "coint4/src/coint2/pipeline/filters.py",
            ]
        )

    if not changes:
        changes = [
            "coint4/src/coint2/pipeline/walk_forward_orchestrator.py",
            "configs/*.yaml",
        ]

    out: list[str] = []
    for entry in changes:
        if entry not in out:
            out.append(entry)
    return out


def _derive_effort(idea: str) -> str:
    text = idea.lower()
    if any(token in text for token in ("reinforcement", "rl", "deep learning", "neural", "genetic", "hmm")):
        return "L"
    if any(token in text for token in ("copula", "kalman", "regime", "factor", "optimization", "cluster")):
        return "M"
    return "S"


def _build_acceptance(idea: str, expected: dict[str, str], checks_text: str) -> list[str]:
    criteria: list[str] = []

    criteria.append("Net метрики считаются на fully-costed бэктесте (fees/slippage/funding), gross и net публикуются рядом.")

    if expected["sharpe"] == "up":
        criteria.append("net sharpe_ratio_abs >= baseline * 1.05 на aggregate OOS.")
    elif expected["sharpe"] == "down":
        criteria.append("net sharpe_ratio_abs не хуже baseline более чем на 2% (guardrail).")

    if expected["drawdown"] == "down":
        criteria.append("max_drawdown_on_equity <= baseline * 0.90 (более глубокая просадка не допускается).")

    if expected["turnover"] == "down":
        criteria.append("turnover_ratio <= baseline * 0.90 при сохранении net_total_pnl >= baseline.")
    else:
        criteria.append("turnover_ratio не растет более чем на 15% относительно baseline.")

    criteria.append("Улучшение подтверждается минимум в 60% walk-forward OOS окон; худшее окно не уходит в критический DD.")

    if checks_text:
        short_checks = [chunk.strip() for chunk in checks_text.split(",") if chunk.strip()]
        if short_checks:
            criteria.append(f"Доп. контроль из synthesis: {short_checks[0]}")

    # Keep deterministic order and unique values.
    unique: list[str] = []
    for item in criteria:
        if item not in unique:
            unique.append(item)
    return unique


def _build_rollback(idea: str) -> str:
    text = idea.lower()
    if any(token in text for token in ("cost", "fee", "slippage", "funding", "turnover")):
        return "Отключить COSTS_ENABLED=0 и вернуть baseline cost-сценарий в конфиге; перезапустить WFA для подтверждения rollback."
    return "Вернуть baseline-параметры эксперимента в конфиге, отключить feature-flag и пересобрать run_index/ранжирование."


def _experiment_id(title: str) -> str:
    digest = hashlib.sha1(_norm(title).encode("utf-8")).hexdigest()[:8].upper()
    return f"EXP-{digest}"


def _read_registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _load_schema(schema_path: Path) -> dict[str, Any]:
    if not schema_path.exists():
        return {}
    try:
        payload = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _validate_entry(entry: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = schema.get("required", [])
    if isinstance(required, list):
        for key in required:
            if key not in entry:
                errors.append(f"missing required field: {key}")

    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        status_schema = properties.get("status")
        effort_schema = properties.get("effort")
        if isinstance(status_schema, dict) and isinstance(status_schema.get("enum"), list):
            if entry.get("status") not in status_schema["enum"]:
                errors.append("invalid status enum")
        if isinstance(effort_schema, dict) and isinstance(effort_schema.get("enum"), list):
            if entry.get("effort") not in effort_schema["enum"]:
                errors.append("invalid effort enum")

    return errors


def main() -> int:
    args = parse_args()

    backlog_path = Path(args.backlog_path)
    executive_path = Path(args.executive_path)
    registry_path = Path(args.registry_path)
    schema_path = Path(args.schema_path)

    registry_path.parent.mkdir(parents=True, exist_ok=True)

    backlog = _parse_backlog(backlog_path)
    executive = _parse_executive(executive_path)

    backlog_by_norm = {_norm(item.get("idea", "")): item for item in backlog if item.get("idea")}

    selected: list[dict[str, Any]] = []
    used_norm: set[str] = set()

    target_n = max(1, int(args.top))

    # Priority source: executive summary top ideas.
    for item in executive:
        if len(selected) >= target_n:
            break
        idea = str(item.get("idea", "")).strip()
        if not idea:
            continue
        idea_norm = _norm(idea)
        if idea_norm in used_norm:
            continue

        linked = [str(item.get("paper_id", "")).lower()] if item.get("paper_id") else []
        backlog_item = backlog_by_norm.get(idea_norm)
        if backlog_item is None:
            # Soft match if wording changed slightly.
            for key, candidate in backlog_by_norm.items():
                if idea_norm in key or key in idea_norm:
                    backlog_item = candidate
                    break

        expected_text = str(item.get("expected_effect") or "").strip()
        rationale = ""
        checks = ""
        if backlog_item:
            expected_text = expected_text or str(backlog_item.get("expected_effect") or "").strip()
            rationale = str(backlog_item.get("why") or "").strip()
            checks = str(backlog_item.get("checks") or "").strip()
            for pid in backlog_item.get("linked_papers", []):
                if pid not in linked:
                    linked.append(pid)

        if not rationale:
            rationale = "Идея попала в top executive summary как прикладная и имеет потенциал улучшения Sharpe/DD в OOS."

        selected.append(
            {
                "idea": idea,
                "expected_text": expected_text,
                "rationale": rationale,
                "checks": checks,
                "linked_papers": [pid for pid in linked if pid],
                "source": [str(executive_path), str(backlog_path)],
            }
        )
        used_norm.add(idea_norm)

    # Fill from backlog if executive had less than target_n.
    if len(selected) < target_n:
        for item in backlog:
            if len(selected) >= target_n:
                break
            idea = str(item.get("idea") or "").strip()
            if not idea:
                continue
            idea_norm = _norm(idea)
            if idea_norm in used_norm:
                continue
            selected.append(
                {
                    "idea": idea,
                    "expected_text": str(item.get("expected_effect") or "").strip(),
                    "rationale": str(item.get("why") or "").strip() or "Выбрано из backlog synthesis.",
                    "checks": str(item.get("checks") or "").strip(),
                    "linked_papers": item.get("linked_papers", []),
                    "source": [str(backlog_path)],
                }
            )
            used_norm.add(idea_norm)

    now_iso = datetime.now(timezone.utc).isoformat()

    existing = _read_registry(registry_path)
    existing_by_title = {_norm(str(item.get("title") or "")): item for item in existing}

    generated_entries: list[dict[str, Any]] = []
    for item in selected:
        title = item["idea"]
        norm_title = _norm(title)
        expected_effect = _build_expected_effect(item.get("expected_text", ""))
        entry = {
            "experiment_id": _experiment_id(title),
            "title": title,
            "rationale": item.get("rationale") or "Практическая идея из synthesis на основе карточек статей.",
            "expected_effect": expected_effect,
            "changes": _suggest_changes(title),
            "acceptance": _build_acceptance(title, expected_effect, str(item.get("checks") or "")),
            "rollback": _build_rollback(title),
            "linked_papers": sorted({pid for pid in item.get("linked_papers", []) if pid}),
            "status": "todo",
            "owner": str(args.owner),
            "created_at": now_iso,
            "effort": _derive_effort(title),
            "source": item.get("source", []),
        }

        old = existing_by_title.get(norm_title)
        if old:
            entry["experiment_id"] = old.get("experiment_id") or entry["experiment_id"]
            entry["status"] = old.get("status") or entry["status"]
            entry["owner"] = old.get("owner") or entry["owner"]
            entry["created_at"] = old.get("created_at") or entry["created_at"]
            if old.get("completion_notes"):
                entry["completion_notes"] = old.get("completion_notes")

        generated_entries.append(entry)

    generated_ids = {item["experiment_id"] for item in generated_entries}
    merged = []
    merged.extend(generated_entries)

    for item in existing:
        exp_id = str(item.get("experiment_id") or "")
        if not exp_id or exp_id in generated_ids:
            continue
        merged.append(item)

    schema = _load_schema(schema_path)
    validation_errors: list[str] = []
    for item in merged:
        for err in _validate_entry(item, schema):
            validation_errors.append(f"{item.get('experiment_id', 'unknown')}: {err}")

    if validation_errors:
        error_path = registry_path.with_suffix(".errors.log")
        error_path.write_text("\n".join(validation_errors), encoding="utf-8")

    # Stable output: generated first, then remaining items by id.
    generated_map = {item["experiment_id"]: idx for idx, item in enumerate(generated_entries)}
    merged.sort(
        key=lambda item: (
            0 if item.get("experiment_id") in generated_map else 1,
            generated_map.get(item.get("experiment_id"), 10**9),
            str(item.get("experiment_id") or ""),
        )
    )

    with registry_path.open("w", encoding="utf-8") as fh:
        for item in merged:
            fh.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    stats = {
        "registry_path": str(registry_path),
        "source_backlog": str(backlog_path),
        "source_executive": str(executive_path),
        "selected_ideas": len(selected),
        "generated_or_updated": len(generated_entries),
        "total_registry_entries": len(merged),
        "validation_errors": len(validation_errors),
    }
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
