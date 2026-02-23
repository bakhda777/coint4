#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lib_pipeline import (
    CARD_SCHEMA_VERSION,
    DEFAULT_CARDS_DIR,
    DEFAULT_DB_PATH,
    DEFAULT_SYNTHESIS_DIR,
    connect_db,
    loads_json,
)

CLUSTERS = (
    "cointegration",
    "kalman",
    "copula",
    "regime",
    "execution_costs",
    "ml_rl",
    "ou_model",
    "clustering_pca",
)


@dataclass(frozen=True)
class Candidate:
    paper_id: str
    priority_score: int
    title: str | None
    year: int | None
    text_sha256: str
    text_len: int
    method_tags: frozenset[str]
    practical_tags: frozenset[str]
    market_tags: frozenset[str]
    clusters: frozenset[str]

    @property
    def all_tags(self) -> tuple[str, ...]:
        tags = [f"method:{tag}" for tag in sorted(self.method_tags)]
        tags.extend(f"practical:{tag}" for tag in sorted(self.practical_tags))
        tags.extend(f"market:{tag}" for tag in sorted(self.market_tags))
        return tuple(tags)


@dataclass
class SelectionItem:
    candidate: Candidate
    reasons: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build diversified paper selection batch for card generation")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite path")
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_SYNTHESIS_DIR / "selection.jsonl"),
        help="Selection JSONL output",
    )
    parser.add_argument("--batch-size", type=int, default=30, help="How many papers to select")
    parser.add_argument("--top-k", type=int, default=180, help="Upper boundary by priority ranking")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument(
        "--min-per-cluster",
        type=int,
        default=3,
        help="Minimum guaranteed picks per method cluster (if available)",
    )
    parser.add_argument(
        "--novelty-ratio",
        type=float,
        default=0.25,
        help="Share of batch allocated to novelty picks (0..1)",
    )
    parser.add_argument("--cards-dir", default=str(DEFAULT_CARDS_DIR), help="Cards directory for cache skipping")
    parser.add_argument(
        "--schema-path",
        default=str(DEFAULT_CARDS_DIR / "card.schema.json"),
        help="Card schema path to validate cache version",
    )
    parser.add_argument(
        "--skip-existing-cards",
        action="store_true",
        default=True,
        help="Skip candidates that already have valid cached card",
    )
    parser.add_argument(
        "--include-existing-cards",
        action="store_true",
        help="Do not skip candidates with valid cached cards",
    )
    return parser.parse_args()


def _tie_key(seed: int, paper_id: str) -> str:
    return hashlib.sha256(f"{seed}:{paper_id}".encode("utf-8")).hexdigest()


def _cluster_members(method_tags: set[str], practical_tags: set[str]) -> set[str]:
    out: set[str] = set()
    if "cointegration" in method_tags or "distance" in method_tags:
        out.add("cointegration")
    if "kalman" in method_tags:
        out.add("kalman")
    if "copula" in method_tags:
        out.add("copula")
    if "regime" in method_tags or "volatility_filter" in method_tags:
        out.add("regime")
    if "execution_costs" in practical_tags:
        out.add("execution_costs")
    if "ml" in method_tags or "rl" in method_tags:
        out.add("ml_rl")
    if "ou_model" in method_tags:
        out.add("ou_model")
    if "clustering" in method_tags or "pca" in method_tags:
        out.add("clustering_pca")
    return out


def _load_schema_version(schema_path: Path) -> str:
    if not schema_path.exists():
        return CARD_SCHEMA_VERSION
    try:
        payload = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return CARD_SCHEMA_VERSION
    if not isinstance(payload, dict):
        return CARD_SCHEMA_VERSION
    return str(payload.get("$id") or payload.get("version") or CARD_SCHEMA_VERSION)


def _has_valid_cached_card(cards_dir: Path, candidate: Candidate, schema_version: str) -> bool:
    card_path = cards_dir / f"{candidate.paper_id}.json"
    if not card_path.exists():
        return False
    try:
        payload = json.loads(card_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    return (
        payload.get("text_sha256") == candidate.text_sha256
        and payload.get("schema_version") == schema_version
    )


def _canonical_ids(rows: list[Any]) -> set[str]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    canonical: set[str] = set()

    for row in rows:
        group = row["dup_group_id"]
        if not group:
            canonical.add(row["id"])
            continue
        grouped[str(group)].append(row)

    for _, group_rows in grouped.items():
        chosen = sorted(
            group_rows,
            key=lambda item: (
                -(int(item["text_len"] or 0)),
                item["id"],
            ),
        )[0]
        canonical.add(chosen["id"])

    return canonical


def _candidate_from_row(row: Any) -> Candidate:
    tags = loads_json(row["tags_json"], {})
    if not isinstance(tags, dict):
        tags = {}

    method_tags = {
        item.strip()
        for item in tags.get("method_family", [])
        if isinstance(item, str) and item.strip()
    }
    practical_tags = {
        item.strip()
        for item in tags.get("practical", [])
        if isinstance(item, str) and item.strip()
    }
    market_tags = {
        item.strip()
        for item in tags.get("market", [])
        if isinstance(item, str) and item.strip()
    }

    clusters = _cluster_members(method_tags, practical_tags)

    return Candidate(
        paper_id=str(row["id"]),
        priority_score=int(row["priority_score"] or 0),
        title=row["title"],
        year=int(row["year"]) if row["year"] is not None else None,
        text_sha256=str(row["text_sha256"]),
        text_len=int(row["text_len"] or 0),
        method_tags=frozenset(method_tags),
        practical_tags=frozenset(practical_tags),
        market_tags=frozenset(market_tags),
        clusters=frozenset(clusters),
    )


def _novelty_score(candidate: Candidate, tag_freq: Counter[str], cluster_freq: Counter[str]) -> float:
    rarity = 0.0
    for tag in candidate.all_tags:
        rarity += 1.0 / float(max(1, tag_freq[tag]))
    for cluster in candidate.clusters:
        rarity += 0.75 / float(max(1, cluster_freq[cluster]))
    return rarity


def _priority_sort(candidates: list[Candidate], seed: int) -> list[Candidate]:
    return sorted(
        candidates,
        key=lambda item: (
            -item.priority_score,
            -(item.year or 0),
            _tie_key(seed, item.paper_id),
        ),
    )


def _append_reason(item: SelectionItem, reason: str) -> None:
    if reason not in item.reasons:
        item.reasons.append(reason)


def main() -> int:
    args = parse_args()

    db_path = Path(args.db_path)
    output_path = Path(args.output_path)
    cards_dir = Path(args.cards_dir)
    schema_path = Path(args.schema_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    skip_existing_cards = bool(args.skip_existing_cards and not args.include_existing_cards)
    schema_version = _load_schema_version(schema_path)

    conn = connect_db(db_path, read_only=True)
    rows = conn.execute(
        """
        SELECT id, title, year, text_sha256, text_len, tags_json,
               priority_score, dup_group_id
        FROM papers
        ORDER BY COALESCE(priority_score, 0) DESC, COALESCE(year, 0) DESC, id ASC
        """
    ).fetchall()
    conn.close()

    canonical_ids = _canonical_ids(rows)

    filtered: list[Candidate] = []
    excluded_noncanonical = 0
    excluded_cached = 0

    for row in rows:
        paper_id = str(row["id"])
        if paper_id not in canonical_ids:
            excluded_noncanonical += 1
            continue

        candidate = _candidate_from_row(row)

        if skip_existing_cards and _has_valid_cached_card(cards_dir, candidate, schema_version):
            excluded_cached += 1
            continue

        filtered.append(candidate)

    ranked = _priority_sort(filtered, seed=args.seed)
    top_k = max(1, int(args.top_k))
    candidate_pool = ranked[:top_k]

    batch_size = max(1, int(args.batch_size))

    if not candidate_pool:
        output_path.write_text("", encoding="utf-8")
        stats = {
            "selected": 0,
            "batch_size": batch_size,
            "top_k": top_k,
            "candidate_pool": 0,
            "excluded_noncanonical": excluded_noncanonical,
            "excluded_cached": excluded_cached,
            "schema_version": schema_version,
        }
        print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
        return 0

    selected: dict[str, SelectionItem] = {}

    def add_candidate(candidate: Candidate, reason: str) -> None:
        item = selected.get(candidate.paper_id)
        if item is None:
            selected[candidate.paper_id] = SelectionItem(candidate=candidate, reasons=[reason])
        else:
            _append_reason(item, reason)

    per_cluster_target = max(1, min(int(args.min_per_cluster), max(1, batch_size // max(1, len(CLUSTERS)))))

    # 1) Cluster coverage picks.
    for cluster in CLUSTERS:
        if len(selected) >= batch_size:
            break
        cluster_candidates = [
            cand
            for cand in candidate_pool
            if cluster in cand.clusters and cand.paper_id not in selected
        ]
        cluster_candidates = _priority_sort(cluster_candidates, seed=args.seed)
        take_n = min(per_cluster_target, len(cluster_candidates), batch_size - len(selected))
        for cand in cluster_candidates[:take_n]:
            add_candidate(cand, f"cluster_quota:{cluster}")

    # Prepare counters for novelty scoring.
    tag_freq: Counter[str] = Counter()
    cluster_freq: Counter[str] = Counter()
    for cand in candidate_pool:
        tag_freq.update(cand.all_tags)
        cluster_freq.update(cand.clusters)

    current_cluster_counts: Counter[str] = Counter()
    for entry in selected.values():
        current_cluster_counts.update(entry.candidate.clusters)

    # 2) Novelty picks from underrepresented/rare tags.
    novelty_ratio = min(0.95, max(0.0, float(args.novelty_ratio)))
    novelty_target = min(batch_size - len(selected), max(0, int(round(batch_size * novelty_ratio))))
    if novelty_target > 0:
        desired_cluster_count = max(1, batch_size // max(1, len(CLUSTERS)))

        novelty_candidates = [cand for cand in candidate_pool if cand.paper_id not in selected]
        novelty_candidates = sorted(
            novelty_candidates,
            key=lambda cand: (
                -sum(max(0, desired_cluster_count - current_cluster_counts[c]) for c in cand.clusters),
                -_novelty_score(cand, tag_freq, cluster_freq),
                -cand.priority_score,
                _tie_key(args.seed, cand.paper_id),
            ),
        )

        for cand in novelty_candidates[:novelty_target]:
            add_candidate(cand, "novelty_pick")
            current_cluster_counts.update(cand.clusters)

    # 3) Priority fill to requested batch size.
    if len(selected) < batch_size:
        fill_candidates = [cand for cand in candidate_pool if cand.paper_id not in selected]
        for cand in _priority_sort(fill_candidates, seed=args.seed)[: batch_size - len(selected)]:
            add_candidate(cand, "priority_fill")

    selected_items = sorted(
        selected.values(),
        key=lambda item: (
            -item.candidate.priority_score,
            -(item.candidate.year or 0),
            _tie_key(args.seed, item.candidate.paper_id),
        ),
    )

    selected_items = selected_items[:batch_size]

    output_lines: list[str] = []
    selected_cluster_counts: Counter[str] = Counter()

    for rank, item in enumerate(selected_items, start=1):
        cand = item.candidate
        selected_cluster_counts.update(cand.clusters)
        reasons = ["top_k_priority_pool", *item.reasons]

        output_lines.append(
            json.dumps(
                {
                    "paper_id": cand.paper_id,
                    "priority_score": cand.priority_score,
                    "title": cand.title,
                    "year": cand.year,
                    "selection_rank": rank,
                    "selection_reasons": reasons,
                    "clusters": sorted(cand.clusters),
                    "method_tags": sorted(cand.method_tags),
                    "practical_tags": sorted(cand.practical_tags),
                    "market_tags": sorted(cand.market_tags),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    output_path.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")

    stats = {
        "selected": len(selected_items),
        "batch_size": batch_size,
        "top_k": top_k,
        "candidate_pool": len(candidate_pool),
        "excluded_noncanonical": excluded_noncanonical,
        "excluded_cached": excluded_cached,
        "schema_version": schema_version,
        "clusters_covered": sorted([name for name, count in selected_cluster_counts.items() if count > 0]),
        "cluster_counts": dict(sorted(selected_cluster_counts.items())),
        "output_path": str(output_path),
        "seed": int(args.seed),
    }
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
