#!/usr/bin/env python3
"""Generate next evolution batch (targeted mutation) from rollup diagnostics.

Outputs:
  - configs: coint4/configs/evolution/<run_group>/*.yaml
  - queue: coint4/artifacts/wfa/aggregate/<run_group>/run_queue.csv
  - search space: coint4/artifacts/wfa/aggregate/<run_group>/search_space.md
  - decision/state:
      coint4/artifacts/wfa/aggregate/<controller_group>/decisions/*.json
      coint4/artifacts/wfa/aggregate/<controller_group>/evolution_state.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import yaml

from coint2.ops.evolution_targeting import (
    FailureAssessment,
    FailureThresholds,
    TargetedOperatorPlan,
    VariantDiagnostics,
    build_variant_diagnostics,
    infer_failure_mode,
    select_operator_plan,
)
from coint2.ops.config_patch_gates import PatchGateLimits, evaluate_patch_gates
from coint2.ops.config_patch_ir import validate_patch_ir_payload
from coint2.ops.genome import (
    Genome,
    KnobSpec,
    canonicalize_object,
    get_dotted,
    genome_distance_weighted_l1_v1,
    genome_from_config,
    knob_specs_from_dicts,
    load_effective_yaml_config,
)
from coint2.ops.mutation_ops import apply_operator_v1

# Strict fullspan contract requires paired holdout+stress generation by default.
STRESS_OVERRIDES: dict[str, Any] = {
    "backtest.commission_pct": 0.0006,
    "backtest.commission_rate_per_leg": 0.0006,
    "backtest.slippage_pct": 0.001,
    "backtest.slippage_stress_multiplier": 2.0,
}

DEFAULT_KNOB_SPACE: list[dict[str, Any]] = [
    {"key": "portfolio.risk_per_position_pct", "type": "float", "min": 0.003, "max": 0.03, "step": 0.001, "round_to": 4, "weight": 2.0, "norm": "range"},
    {"key": "portfolio.max_active_positions", "type": "int", "min": 4, "max": 40, "step": 1, "weight": 1.5, "norm": "range"},
    {"key": "backtest.portfolio_daily_stop_pct", "type": "float", "min": 0.005, "max": 0.08, "step": 0.002, "round_to": 4, "weight": 1.5, "norm": "range"},
    {"key": "backtest.max_var_multiplier", "type": "float", "min": 0.9, "max": 1.2, "step": 0.005, "round_to": 4, "weight": 1.5, "norm": "range"},
    {"key": "backtest.pair_stop_loss_usd", "type": "float", "min": 1.0, "max": 30.0, "step": 0.5, "round_to": 3, "weight": 1.0, "norm": "range"},
    {"key": "backtest.pair_stop_loss_zscore", "type": "float", "min": 1.0, "max": 6.0, "step": 0.1, "round_to": 3, "weight": 1.0, "norm": "range"},
    {"key": "backtest.stop_loss_multiplier", "type": "float", "min": 1.0, "max": 6.0, "step": 0.1, "round_to": 3, "weight": 1.0, "norm": "range"},
    {"key": "backtest.time_stop_multiplier", "type": "float", "min": 0.5, "max": 6.0, "step": 0.1, "round_to": 3, "weight": 1.0, "norm": "range"},
    {"key": "backtest.zscore_entry_threshold", "type": "float", "min": 0.6, "max": 3.0, "step": 0.05, "round_to": 3, "weight": 1.8, "norm": "range"},
    {"key": "backtest.zscore_exit", "type": "float", "min": 0.02, "max": 1.2, "step": 0.02, "round_to": 3, "weight": 1.2, "norm": "range"},
    {"key": "backtest.rolling_window", "type": "int", "min": 24, "max": 240, "step": 12, "weight": 1.2, "norm": "range"},
    # AppConfig.backtest.cooldown_hours is int (strict). Keep evolution knob int to avoid fractional values that fail
    # pydantic validation on the runner.
    {"key": "backtest.cooldown_hours", "type": "int", "min": 0, "max": 24, "step": 1, "weight": 1.2, "norm": "range"},
    {"key": "backtest.min_spread_move_sigma", "type": "float", "min": 0.0, "max": 1.2, "step": 0.05, "round_to": 3, "weight": 1.0, "norm": "range"},
    {"key": "pair_selection.max_pairs", "type": "int", "min": 8, "max": 96, "step": 2, "weight": 1.8, "norm": "range"},
    {"key": "pair_selection.min_correlation", "type": "float", "min": 0.1, "max": 0.9, "step": 0.02, "round_to": 3, "weight": 1.5, "norm": "range"},
    {"key": "pair_selection.coint_pvalue_threshold", "type": "float", "min": 0.01, "max": 0.5, "step": 0.01, "round_to": 3, "weight": 1.5, "norm": "range"},
    {"key": "pair_selection.lookback_days", "type": "int", "min": 30, "max": 180, "step": 10, "weight": 1.2, "norm": "range"},
    {"key": "pair_selection.ssd_top_n", "type": "int", "min": 5000, "max": 100000, "step": 5000, "weight": 1.2, "norm": "range"},
    {"key": "pair_selection.pvalue_top_n", "type": "int", "min": 200, "max": 10000, "step": 200, "weight": 1.2, "norm": "range"},
    {"key": "pair_selection.kpss_pvalue_threshold", "type": "float", "min": 0.01, "max": 1.0, "step": 0.05, "round_to": 3, "weight": 1.0, "norm": "range"},
    {"key": "pair_selection.max_hurst_exponent", "type": "float", "min": 0.4, "max": 0.95, "step": 0.02, "round_to": 3, "weight": 1.1, "norm": "range"},
    {"key": "pair_selection.min_mean_crossings", "type": "int", "min": 0, "max": 12, "step": 1, "weight": 1.1, "norm": "range"},
    {"key": "pair_selection.max_half_life_days", "type": "float", "min": 5.0, "max": 180.0, "step": 5.0, "round_to": 3, "weight": 1.1, "norm": "range"},
    {"key": "pair_selection.enable_pair_tradeability_filter", "type": "bool", "weight": 1.0, "norm": "none"},
    {"key": "pair_selection.min_volume_usd_24h", "type": "float", "min": 0.0, "max": 50000000.0, "step": 250000.0, "round_to": 1, "weight": 1.0, "norm": "range"},
    {"key": "pair_selection.min_days_live", "type": "int", "min": 0, "max": 180, "step": 5, "weight": 1.0, "norm": "range"},
    {"key": "pair_selection.max_funding_rate_abs", "type": "float", "min": 0.00005, "max": 0.05, "step": 0.001, "round_to": 6, "weight": 1.0, "norm": "range"},
    {"key": "pair_selection.max_tick_size_pct", "type": "float", "min": 0.00005, "max": 0.02, "step": 0.001, "round_to": 6, "weight": 1.0, "norm": "range"},
    {"key": "data_processing.min_history_ratio", "type": "float", "min": 0.5, "max": 1.0, "step": 0.05, "round_to": 3, "weight": 1.0, "norm": "range"},
    {"key": "filter_params.max_hurst_exponent", "type": "float", "min": 0.4, "max": 0.95, "step": 0.02, "round_to": 3, "weight": 1.2, "norm": "range"},
    {"key": "filter_params.min_mean_crossings", "type": "int", "min": 0, "max": 8, "step": 1, "weight": 1.2, "norm": "range"},
    {"key": "filter_params.max_half_life_days", "type": "int", "min": 5, "max": 180, "step": 5, "weight": 1.2, "norm": "range"},
]

LlmOperatorKind = Literal["mutate_step_v1", "coordinate_sweep_v1", "random_restart_v1", "crossover_uniform_v1"]
_LLM_ALLOWED_OPERATOR_KINDS: tuple[LlmOperatorKind, ...] = (
    "mutate_step_v1",
    "coordinate_sweep_v1",
    "random_restart_v1",
    "crossover_uniform_v1",
)


@dataclass(frozen=True, slots=True)
class CandidateProposal:
    candidate_id: str
    operator_id: str
    parents: tuple[str, ...]
    genome: Genome
    changed_keys: tuple[str, ...]
    nearest_id: str
    nearest_distance: float
    notes: str
    patch_ir: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class LlmPolicyPlan:
    enabled: bool
    used: bool
    source: str
    reason: str
    model: str
    effort: str
    operator_plan: TargetedOperatorPlan
    payload: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class ParentResolution:
    diagnostics: tuple[VariantDiagnostics, ...]
    top_diag: VariantDiagnostics | None
    winner_proximate_requested: bool
    winner_proximate_tokens: tuple[str, ...]
    winner_proximate_resolved: bool
    preferred_parent_source: str
    winner_proximate_fallback_reason: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_app_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _ensure_local_optimization_scripts_on_path() -> None:
    """Ensure sibling optimization scripts are importable even when loaded via importlib."""

    scripts_dir = Path(__file__).resolve().parent
    token = str(scripts_dir)
    if token not in sys.path:
        sys.path.insert(0, token)


def _resolve_under_root(path: str, *, root: Path) -> Path:
    candidate = Path(str(path or "").strip())
    if not candidate.as_posix():
        raise ValueError("empty path")
    if candidate.is_absolute():
        return candidate
    return root / candidate


def _relative_to_root(path: Path, *, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:  # noqa: BLE001
        return path.as_posix()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    return rows


def _derive_lineage_uid(
    *,
    controller_group: str,
    run_group: str,
    generation: int,
    variant_index: int,
    candidate_id: str,
    operator_id: str,
    parents: Sequence[str],
) -> str:
    seed = [
        str(controller_group or "").strip().lower(),
        str(run_group or "").strip().lower(),
        str(int(generation)),
        str(int(variant_index)),
        str(candidate_id or "").strip().lower(),
        str(operator_id or "").strip().lower(),
        "|".join(str(parent or "").strip().lower() for parent in list(parents or [])),
    ]
    digest = hashlib.sha1("|".join(seed).encode("utf-8")).hexdigest()[:20]
    return f"lnuid_{digest}"


def _write_run_queue_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = ["config_path", "results_dir", "status"]
    ordered_extras = ["lineage_uid", "metadata_json"]
    extras_seen: list[str] = []
    extras_set: set[str] = set()

    for row in rows:
        if not isinstance(row, Mapping):
            continue
        for key in row.keys():
            name = str(key or "").strip()
            if not name or name in base_fields or name in extras_set:
                continue
            extras_set.add(name)
            extras_seen.append(name)

    fieldnames = list(base_fields)
    for key in ordered_extras:
        if key in extras_set:
            fieldnames.append(key)
    for key in extras_seen:
        if key not in ordered_extras:
            fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload: dict[str, str] = {}
            for key in fieldnames:
                value = row.get(key) if isinstance(row, Mapping) else ""
                if key == "metadata_json":
                    if isinstance(value, str):
                        payload[key] = value
                    elif value is None:
                        payload[key] = ""
                    else:
                        payload[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
                else:
                    payload[key] = str(value or "").strip()
            writer.writerow(payload)


def _load_knob_space(path: str | None) -> list[KnobSpec]:
    if not path:
        return knob_specs_from_dicts(DEFAULT_KNOB_SPACE)
    payload_path = Path(path)
    if not payload_path.exists():
        raise SystemExit(f"knob-space file not found: {payload_path}")
    if payload_path.suffix.lower() == ".json":
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    else:
        payload = yaml.safe_load(payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit("knob-space payload must be a list")
    return knob_specs_from_dicts(payload)


def _parse_windows(*, raw_values: Sequence[str], base_cfg: Mapping[str, Any]) -> list[tuple[str, str]]:
    windows: list[tuple[str, str]] = []
    for token in raw_values:
        text = str(token or "").strip()
        if not text:
            continue
        parts = [part.strip() for part in text.split(",")]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise SystemExit(f"invalid --window format: {text} (expected YYYY-MM-DD,YYYY-MM-DD)")
        windows.append((parts[0], parts[1]))
    if windows:
        return windows

    walk_forward = base_cfg.get("walk_forward")
    if not isinstance(walk_forward, Mapping):
        raise SystemExit("base config has no walk_forward section and no --window passed")
    start = str(walk_forward.get("start_date") or "").strip()
    end = str(walk_forward.get("end_date") or "").strip()
    if not start or not end:
        raise SystemExit("base config walk_forward.start_date/end_date is empty; pass explicit --window")
    return [(start, end)]


def _set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _apply_genome_overrides(cfg: dict[str, Any], genome: Genome) -> None:
    for dotted_key, value in genome.items():
        _set_nested(cfg, dotted_key, value)


def _apply_materialized_patch(cfg: dict[str, Any], patch: Mapping[str, Any]) -> None:
    for key, value in patch.items():
        if str(key) == "base_config":
            continue
        if isinstance(value, Mapping):
            child = cfg.get(key)
            if not isinstance(child, dict):
                child = {}
                cfg[key] = child
            _apply_materialized_patch(child, value)
            continue
        cfg[key] = value


def _candidate_id(
    *,
    generation: int,
    parent_id: str,
    operator_id: str,
    genome: Genome,
) -> str:
    encoded = json.dumps(
        canonicalize_object(
            {
                "generation": generation,
                "parent_id": parent_id,
                "operator_id": operator_id,
                "genome": genome,
            }
        ),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:12]
    return f"evo_{digest}"


def _changed_keys(parent: Mapping[str, Any], child: Mapping[str, Any]) -> tuple[str, ...]:
    changed = [key for key in sorted(child.keys()) if parent.get(key) != child.get(key)]
    return tuple(changed)


def _genome_fingerprint(genome: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        canonicalize_object(dict(genome)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_path_for_lookup(path_raw: Any, *, root: Path) -> str:
    token = str(path_raw or "").strip()
    if not token:
        return ""
    path = Path(token)
    if not path.is_absolute():
        path = root / path
    try:
        return path.resolve().as_posix()
    except Exception:  # noqa: BLE001
        return path.as_posix()


def _load_validation_error_genome_counts(
    *,
    app_root: Path,
    knob_space: Sequence[KnobSpec],
) -> dict[str, int]:
    quarantine_path = app_root / "artifacts" / "wfa" / "aggregate" / ".autonomous" / "deterministic_quarantine.json"
    if not quarantine_path.exists():
        return {}

    try:
        payload = json.loads(quarantine_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(payload, dict):
        return {}
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return {}

    queue_row_cache: dict[str, dict[str, Path]] = {}
    config_fingerprint_cache: dict[str, str] = {}
    counts: dict[str, int] = {}

    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("code") or "").strip() != "CONFIG_VALIDATION_ERROR":
            continue

        queue_key = _normalize_path_for_lookup(entry.get("queue"), root=app_root)
        results_key = _normalize_path_for_lookup(entry.get("results_dir"), root=app_root)
        if not queue_key or not results_key:
            continue

        queue_mapping = queue_row_cache.get(queue_key)
        if queue_mapping is None:
            queue_mapping = {}
            queue_path = Path(queue_key)
            for row in _read_csv_rows(queue_path):
                row_results = _normalize_path_for_lookup(row.get("results_dir"), root=app_root)
                cfg_raw = str(row.get("config_path") or "").strip()
                if not row_results or not cfg_raw:
                    continue
                try:
                    queue_mapping[row_results] = _resolve_under_root(cfg_raw, root=app_root)
                except Exception:  # noqa: BLE001
                    continue
            queue_row_cache[queue_key] = queue_mapping

        cfg_path = queue_mapping.get(results_key)
        if cfg_path is None:
            continue
        cfg_key = cfg_path.as_posix()
        fp = config_fingerprint_cache.get(cfg_key)
        if fp is None:
            if not cfg_path.exists():
                continue
            try:
                cfg = load_effective_yaml_config(cfg_path)
                genome = genome_from_config(cfg, knob_space=knob_space)
                fp = _genome_fingerprint(genome)
            except Exception:  # noqa: BLE001
                continue
            config_fingerprint_cache[cfg_key] = fp

        counts[fp] = int(counts.get(fp, 0)) + 1
    return counts


def _matches_all(text: str, needles: Sequence[str]) -> bool:
    hay = str(text or "").lower()
    for needle in needles:
        token = str(needle or "").strip().lower()
        if token and token not in hay:
            return False
    return True


def _collect_seen_genomes(
    *,
    rows: Sequence[Mapping[str, Any]],
    contains: Sequence[str],
    app_root: Path,
    knob_space: Sequence[KnobSpec],
    max_items: int,
) -> list[tuple[str, Genome]]:
    seen_hashes: set[str] = set()
    candidates: list[tuple[str, Genome]] = []
    for row in rows:
        meta = " | ".join(
            (
                str(row.get("run_group") or "").strip(),
                str(row.get("run_id") or "").strip(),
                str(row.get("config_path") or "").strip(),
                str(row.get("results_dir") or "").strip(),
            )
        )
        if contains and not _matches_all(meta, contains):
            continue
        cfg_path_raw = str(row.get("config_path") or "").strip()
        if not cfg_path_raw:
            continue
        cfg_path = _resolve_under_root(cfg_path_raw, root=app_root)
        if not cfg_path.exists():
            continue
        try:
            cfg = load_effective_yaml_config(cfg_path)
            genome = genome_from_config(cfg, knob_space=knob_space)
        except Exception:  # noqa: BLE001
            continue
        encoded = json.dumps(canonicalize_object(genome), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        if encoded in seen_hashes:
            continue
        seen_hashes.add(encoded)
        run_id = str(row.get("run_id") or "").strip()
        candidates.append((run_id, genome))
        if len(candidates) >= max_items:
            break
    return candidates


def _resolve_parent_diagnostics(
    *,
    rows: Sequence[Mapping[str, Any]],
    contains: Sequence[str],
    winner_proximate_tokens: Sequence[str],
    include_noncompleted: bool,
) -> ParentResolution:
    preferred_tokens = tuple(dict.fromkeys(str(token).strip() for token in winner_proximate_tokens if str(token).strip()))
    generic_diagnostics = tuple(
        build_variant_diagnostics(
            rows,
            contains=contains,
            contains_mode="all",
            include_noncompleted=include_noncompleted,
        )
    )
    generic_top = generic_diagnostics[0] if generic_diagnostics else None
    if not preferred_tokens:
        return ParentResolution(
            diagnostics=generic_diagnostics,
            top_diag=generic_top,
            winner_proximate_requested=False,
            winner_proximate_tokens=(),
            winner_proximate_resolved=False,
            preferred_parent_source="generic_contains" if generic_top else "base_config",
            winner_proximate_fallback_reason="",
        )

    preferred_diagnostics = tuple(
        build_variant_diagnostics(
            rows,
            contains=preferred_tokens,
            contains_mode="any",
            include_noncompleted=include_noncompleted,
        )
    )
    preferred_top = preferred_diagnostics[0] if preferred_diagnostics else None
    if preferred_top is not None:
        return ParentResolution(
            diagnostics=preferred_diagnostics,
            top_diag=preferred_top,
            winner_proximate_requested=True,
            winner_proximate_tokens=preferred_tokens,
            winner_proximate_resolved=True,
            preferred_parent_source="winner_proximate_any_match",
            winner_proximate_fallback_reason="",
        )
    return ParentResolution(
        diagnostics=generic_diagnostics,
        top_diag=generic_top,
        winner_proximate_requested=True,
        winner_proximate_tokens=preferred_tokens,
        winner_proximate_resolved=False,
        preferred_parent_source="generic_contains_fallback" if generic_top else "base_config_fallback",
        winner_proximate_fallback_reason="no_matching_rows_for_any_winner_proximate_token",
    )


def _load_patch_zoo_from_decisions(decisions_dir: Path, *, max_items: int = 200) -> list[dict[str, Any]]:
    """Load previously accepted patch IR entries from decision artifacts (best-effort)."""

    from coint2.ops.config_patch_ast import ast_from_factors  # local import for CLI startup speed

    if not decisions_dir.exists():
        return []
    out: list[dict[str, Any]] = []
    candidates = sorted(decisions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        proposals = payload.get("proposals")
        if not isinstance(proposals, list):
            continue
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            patch_ir = proposal.get("patch_ir")
            if not isinstance(patch_ir, dict):
                continue
            if validate_patch_ir_payload(patch_ir):
                continue
            factors = patch_ir.get("factors")
            if not isinstance(factors, list) or not factors:
                continue
            hypothesis = patch_ir.get("hypothesis")
            if not isinstance(hypothesis, dict):
                hypothesis = {}
            materialized_patch = patch_ir.get("materialized_patch")
            if not isinstance(materialized_patch, dict):
                materialized_patch = {}
            candidate_id = str(proposal.get("candidate_id") or "").strip() or path.stem
            try:
                ast = ast_from_factors(factors)
            except Exception:  # noqa: BLE001
                continue
            out.append(
                {
                    "candidate_id": candidate_id,
                    "hypothesis": hypothesis,
                    "factors": factors,
                    "materialized_patch": materialized_patch,
                    "ast": ast,
                }
            )
            if len(out) >= max_items:
                return out
    return out


def _nearest_distance(
    genome: Genome,
    *,
    seen_pool: Sequence[tuple[str, Genome]],
    knob_space: Sequence[KnobSpec],
) -> tuple[str, float]:
    nearest_id = ""
    nearest_distance = float("inf")
    for seen_id, seen_genome in seen_pool:
        distance = genome_distance_weighted_l1_v1(
            genome,
            seen_genome,
            knob_space=knob_space,
            normalization="range",
        )
        if distance < nearest_distance:
            nearest_distance = float(distance)
            nearest_id = str(seen_id or "").strip()
    return nearest_id, nearest_distance


def _parents_for_operator(
    kind: str,
    *,
    parent_genome: Genome,
    seen_pool: Sequence[tuple[str, Genome]],
    rng: np.random.Generator,
) -> list[Mapping[str, Any]]:
    if kind == "random_restart_v1":
        return []
    if kind == "crossover_uniform_v1":
        others = [genome for _run_id, genome in seen_pool if genome != parent_genome]
        if not others:
            return [parent_genome, parent_genome]
        parent_b = others[int(rng.integers(0, len(others)))]
        return [parent_genome, parent_b]
    return [parent_genome]


def _build_candidate(
    *,
    child: Genome,
    parent_genome: Genome,
    parent_id: str,
    parents: Sequence[str],
    operator_id: str,
    generation: int,
    seen_pool: Sequence[tuple[str, Genome]],
    knob_space: Sequence[KnobSpec],
    patch_ir: dict[str, Any] | None = None,
    notes: str = "targeted mutation proposal",
) -> CandidateProposal:
    nearest_id, nearest_distance = _nearest_distance(child, seen_pool=seen_pool, knob_space=knob_space)
    candidate_id = _candidate_id(
        generation=generation,
        parent_id=parent_id,
        operator_id=operator_id,
        genome=child,
    )
    changed = _changed_keys(parent_genome, child)
    return CandidateProposal(
        candidate_id=candidate_id,
        operator_id=operator_id,
        parents=tuple(str(item) for item in parents if str(item).strip()) or (parent_id,),
        genome=child,
        changed_keys=changed,
        nearest_id=nearest_id,
        nearest_distance=nearest_distance,
        notes=notes,
        patch_ir=patch_ir,
    )


def _generate_proposals(
    *,
    num_variants: int,
    parent_genome: Genome,
    parent_id: str,
    knob_space: Sequence[KnobSpec],
    operator_plan: TargetedOperatorPlan,
    failure_assessment: FailureAssessment,
    generation: int,
    rng: np.random.Generator,
    seen_pool: list[tuple[str, Genome]],
    dedupe_distance: float,
    max_changed_keys: int,
    blocked_candidate_ids: set[str] | None = None,
) -> list[CandidateProposal]:
    _ = failure_assessment
    operator_kind = str(operator_plan.operator_kind or "").strip()
    if operator_kind not in {"mutate_step_v1", "coordinate_sweep_v1", "random_restart_v1", "crossover_uniform_v1"}:
        raise SystemExit(f"unsupported operator kind: {operator_kind}")

    proposals: list[CandidateProposal] = []
    local_seen: set[str] = set()
    blocked_ids = blocked_candidate_ids if blocked_candidate_ids is not None else set()
    max_attempts = max(128, num_variants * 64)
    attempts = 0

    while len(proposals) < num_variants and attempts < max_attempts:
        attempts += 1
        parents = _parents_for_operator(
            operator_kind,
            parent_genome=parent_genome,
            seen_pool=seen_pool,
            rng=rng,
        )
        children = apply_operator_v1(
            operator_kind,
            parents=parents,
            knob_space=knob_space,
            rng=rng,
            budget=int(operator_plan.budget),
            params=operator_plan.params,
        )
        for child in children:
            candidate = _build_candidate(
                child=child,
                parent_genome=parent_genome,
                parent_id=parent_id,
                parents=[parent_id],
                operator_id="op_targeted_primary",
                generation=generation,
                seen_pool=seen_pool,
                knob_space=knob_space,
            )
            if len(candidate.changed_keys) > max_changed_keys:
                continue
            if candidate.nearest_id and candidate.nearest_distance <= dedupe_distance:
                continue
            if candidate.candidate_id in blocked_ids:
                continue
            if candidate.candidate_id in local_seen:
                continue
            local_seen.add(candidate.candidate_id)
            blocked_ids.add(candidate.candidate_id)
            proposals.append(candidate)
            seen_pool.append((candidate.candidate_id, candidate.genome))
            if len(proposals) >= num_variants:
                break

    if not proposals:
        # Emergency escape hatch: never leave the autonomous cycle without at least
        # one candidate when the standard operator plan collapses into duplicates.
        fallback_budget = max(1, min(int(max_changed_keys), 8))
        for fallback_kind in ("random_restart_v1", "mutate_step_v1", "coordinate_sweep_v1"):
            try:
                fallback_children = apply_operator_v1(
                    fallback_kind,
                    parents=[parent_genome],
                    knob_space=knob_space,
                    rng=rng,
                    budget=fallback_budget,
                    params={},
                )
            except Exception:
                continue
            for child in fallback_children:
                fallback_candidate = _build_candidate(
                    child=child,
                    parent_genome=parent_genome,
                    parent_id=parent_id,
                    parents=[parent_id],
                    operator_id=f"op_emergency_{fallback_kind}",
                    generation=generation,
                    seen_pool=seen_pool,
                    knob_space=knob_space,
                )
                if len(fallback_candidate.changed_keys) <= 0:
                    continue
                if fallback_candidate.candidate_id in blocked_ids or fallback_candidate.candidate_id in local_seen:
                    continue
                local_seen.add(fallback_candidate.candidate_id)
                blocked_ids.add(fallback_candidate.candidate_id)
                proposals.append(fallback_candidate)
                seen_pool.append((fallback_candidate.candidate_id, fallback_candidate.genome))
                break
            if proposals:
                break

    if not proposals:
        raise SystemExit("unable to generate non-duplicate candidates (check knob-space or dedupe threshold)")
    return proposals


def _candidate_passes_app_config_validation(
    *,
    parent_cfg: Mapping[str, Any],
    candidate: CandidateProposal,
) -> bool:
    # Keep import local to avoid startup overhead in default (repair=off) mode.
    from coint2.utils.config import AppConfig

    payload = copy.deepcopy(dict(parent_cfg))
    changed = {key: candidate.genome.get(key) for key in candidate.changed_keys if key in candidate.genome}
    _apply_genome_overrides(payload, changed)
    try:
        AppConfig(**payload)
    except Exception:  # noqa: BLE001
        return False
    return True


def _build_validation_neighbor_proposals(
    *,
    proposal: CandidateProposal,
    risk_count: int,
    repair_max_neighbors: int,
    exclude_knobs: set[str],
    parent_genome: Genome,
    parent_id: str,
    parent_cfg: Mapping[str, Any],
    knob_space: Sequence[KnobSpec],
    generation: int,
    rng: np.random.Generator,
    seen_pool: list[tuple[str, Genome]],
    dedupe_distance: float,
    max_changed_keys: int,
    blocked_ids: set[str],
) -> list[CandidateProposal]:
    out: list[CandidateProposal] = []
    if repair_max_neighbors <= 0:
        return out

    ordered_keys = list(proposal.changed_keys)
    ordered_keys.extend(spec.key for spec in knob_space if spec.key not in ordered_keys)
    ordered_keys = [key for key in ordered_keys if key not in exclude_knobs]
    if not ordered_keys:
        return out

    local_fingerprints: set[str] = {_genome_fingerprint(proposal.genome)}
    key_order = {key: idx for idx, key in enumerate(ordered_keys)}
    neighbor_pool: list[tuple[float, int, str, Genome]] = []
    for key in ordered_keys:
        try:
            children = apply_operator_v1(
                "coordinate_sweep_v1",
                parents=[proposal.genome],
                knob_space=knob_space,
                rng=rng,
                budget=3,
                params={"keys": [key], "max_keys": 1},
            )
        except Exception:  # noqa: BLE001
            continue
        for child in children:
            fp = _genome_fingerprint(child)
            if fp in local_fingerprints:
                continue
            local_fingerprints.add(fp)
            dist = genome_distance_weighted_l1_v1(
                proposal.genome,
                child,
                knob_space=knob_space,
                normalization="range",
            )
            neighbor_pool.append((float(dist), int(key_order.get(key, 9999)), key, child))

    if not neighbor_pool:
        return out
    neighbor_pool.sort(key=lambda item: (item[0], item[1], item[2]))

    for _distance, _key_idx, key, child in neighbor_pool:
        if len(out) >= int(repair_max_neighbors):
            break
        candidate = _build_candidate(
            child=child,
            parent_genome=parent_genome,
            parent_id=parent_id,
            parents=list(proposal.parents) if proposal.parents else [parent_id],
            operator_id="op_validation_neighbor",
            generation=generation,
            seen_pool=seen_pool,
            knob_space=knob_space,
            notes=(
                f"validation-neighbor repair for {proposal.candidate_id}: "
                f"risk_count={int(risk_count)} key={key}"
            ),
        )
        if len(candidate.changed_keys) > int(max_changed_keys):
            continue
        if candidate.nearest_id and candidate.nearest_distance <= float(dedupe_distance):
            continue
        if candidate.candidate_id in blocked_ids:
            continue
        if not _candidate_passes_app_config_validation(parent_cfg=parent_cfg, candidate=candidate):
            continue
        blocked_ids.add(candidate.candidate_id)
        seen_pool.append((candidate.candidate_id, candidate.genome))
        out.append(candidate)
    return out


def _apply_validation_neighbor_repair(
    *,
    proposals: Sequence[CandidateProposal],
    app_root: Path,
    repair_max_neighbors: int,
    exclude_knobs: set[str],
    parent_cfg: Mapping[str, Any],
    parent_genome: Genome,
    parent_id: str,
    knob_space: Sequence[KnobSpec],
    generation: int,
    rng: np.random.Generator,
    seen_pool: Sequence[tuple[str, Genome]],
    dedupe_distance: float,
    max_changed_keys: int,
) -> tuple[list[CandidateProposal], dict[str, int]]:
    summary: dict[str, int] = {
        "risk_matched": 0,
        "replaced": 0,
        "neighbors_added": 0,
        "fallback_kept": 0,
    }
    baseline = list(proposals)
    if not baseline:
        return baseline, summary

    risk_counts_by_genome = _load_validation_error_genome_counts(app_root=app_root, knob_space=knob_space)
    if not risk_counts_by_genome:
        return baseline, summary

    min_risk_count = 2
    risky: list[tuple[CandidateProposal, int]] = []
    safe: list[CandidateProposal] = []
    for proposal in baseline:
        risk_count = int(risk_counts_by_genome.get(_genome_fingerprint(proposal.genome), 0))
        if risk_count >= min_risk_count:
            risky.append((proposal, risk_count))
        else:
            safe.append(proposal)

    if not risky:
        return baseline, summary
    summary["risk_matched"] = len(risky)

    target_size = len(baseline)
    blocked_ids: set[str] = {item.candidate_id for item in safe}
    risky_ids = {item.candidate_id for item, _risk_count in risky}
    local_seen_pool: list[tuple[str, Genome]] = [item for item in seen_pool if item[0] not in risky_ids]
    repaired: list[CandidateProposal] = list(safe)

    for proposal, risk_count in risky:
        if len(repaired) >= target_size:
            break
        neighbors = _build_validation_neighbor_proposals(
            proposal=proposal,
            risk_count=risk_count,
            repair_max_neighbors=repair_max_neighbors,
            exclude_knobs=exclude_knobs,
            parent_genome=parent_genome,
            parent_id=parent_id,
            parent_cfg=parent_cfg,
            knob_space=knob_space,
            generation=generation,
            rng=rng,
            seen_pool=local_seen_pool,
            dedupe_distance=dedupe_distance,
            max_changed_keys=max_changed_keys,
            blocked_ids=blocked_ids,
        )
        if neighbors:
            remaining = max(0, target_size - len(repaired))
            if remaining > 0:
                repaired.extend(neighbors[:remaining])
            summary["replaced"] += 1
            summary["neighbors_added"] += len(neighbors[:remaining])
        else:
            if proposal.candidate_id not in blocked_ids:
                repaired.append(proposal)
                blocked_ids.add(proposal.candidate_id)
                summary["fallback_kept"] += 1

    if len(repaired) < target_size:
        for proposal, _risk_count in risky:
            if len(repaired) >= target_size:
                break
            if proposal.candidate_id in blocked_ids:
                continue
            repaired.append(proposal)
            blocked_ids.add(proposal.candidate_id)
            summary["fallback_kept"] += 1

    return repaired[:target_size], summary


def math_is_finite(value: Any) -> bool:
    if value is None:
        return False
    try:
        out = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(out)


def _decision_id(controller_group: str, run_group: str, generation: int) -> str:
    safe_controller = str(controller_group or "").strip().replace("/", "_")
    safe_run = str(run_group or "").strip().replace("/", "_")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"evo_{safe_controller}_{safe_run}_g{int(generation):02d}_{stamp}".replace("-", "_")


def _fmt_optional(value: float | None, *, digits: int = 4) -> str:
    if value is None or not math_is_finite(value):
        return "-"
    return f"{float(value):.{digits}f}"


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _apply_policy_scale(
    *,
    operator_plan: TargetedOperatorPlan,
    policy_scale: str,
    seen_pool_count: int,
) -> TargetedOperatorPlan:
    scale = str(policy_scale or "auto").strip().lower()
    if scale not in {"auto", "micro", "macro"}:
        return operator_plan
    if scale == "micro":
        keys = list(operator_plan.params.get("keys") or [])
        params = {"keys": keys[:3] if keys else keys}
        return TargetedOperatorPlan(
            failure_mode=operator_plan.failure_mode,
            operator_kind="mutate_step_v1",
            budget=max(8, min(int(operator_plan.budget), 24)),
            params=params,
            hypothesis=f"{operator_plan.hypothesis} [policy_scale=micro]",
        )
    if scale == "macro":
        keys = list(operator_plan.params.get("keys") or [])
        if seen_pool_count >= 2:
            return TargetedOperatorPlan(
                failure_mode=operator_plan.failure_mode,
                operator_kind="crossover_uniform_v1",
                budget=max(int(operator_plan.budget), 24),
                params={"keys": keys},
                hypothesis=f"{operator_plan.hypothesis} [policy_scale=macro+crossover]",
            )
        return TargetedOperatorPlan(
            failure_mode=operator_plan.failure_mode,
            operator_kind="coordinate_sweep_v1",
            budget=max(int(operator_plan.budget), 24),
            params={"keys": keys[:4], "max_keys": min(4, len(keys)) if keys else 2},
            hypothesis=f"{operator_plan.hypothesis} [policy_scale=macro+sweep]",
        )
    if operator_plan.operator_kind == "random_restart_v1" and seen_pool_count >= 2:
        return TargetedOperatorPlan(
            failure_mode=operator_plan.failure_mode,
            operator_kind="crossover_uniform_v1",
            budget=max(int(operator_plan.budget), 16),
            params={"keys": list(operator_plan.params.get("keys") or [])},
            hypothesis=f"{operator_plan.hypothesis} [policy_scale=auto->crossover]",
        )
    return operator_plan


def _build_llm_policy_schema(
    *,
    allowed_operator_kinds: Sequence[str],
    max_keys: int,
    max_budget: int,
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["operator_kind", "budget", "keys", "max_keys", "reasoning"],
        "properties": {
            "operator_kind": {"type": "string", "enum": list(allowed_operator_kinds)},
            "budget": {"type": "integer", "minimum": 1, "maximum": int(max_budget)},
            "keys": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": int(max_keys)},
            "max_keys": {"type": "integer", "minimum": 1, "maximum": int(max_keys)},
            "reasoning": {"type": "string", "minLength": 1, "maxLength": 2000},
        },
    }


def _run_codex_json(
    *,
    prompt: str,
    schema: dict[str, Any],
    model_name: str,
    timeout_sec: int,
    codex_bin: str,
    repo_root: Path,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="evolve_llm_") as temp_dir:
        temp = Path(temp_dir)
        schema_path = temp / "output_schema.json"
        out_path = temp / "output_last_message.json"
        schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

        command = [
            codex_bin,
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(out_path),
        ]
        if model_name:
            command.extend(["--model", model_name])
        command.append(prompt)

        proc = subprocess.run(
            command,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(10, int(timeout_sec)),
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"llm_codex_exec failed rc={proc.returncode}: {proc.stderr.strip()}")
        if not out_path.exists():
            raise RuntimeError("llm_codex_exec returned no --output-last-message file")
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("llm_codex_exec output is not a JSON object")
        return payload


def _sanitize_llm_keys(value: Any, *, knob_space: Sequence[KnobSpec], max_keys: int) -> list[str]:
    allowed = {spec.key for spec in knob_space}
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        if text not in allowed:
            continue
        if text in out:
            continue
        out.append(text)
        if len(out) >= max(1, int(max_keys)):
            break
    return out


def _build_llm_policy_prompt(
    *,
    diagnostics: VariantDiagnostics | None,
    failure: FailureAssessment,
    base_plan: TargetedOperatorPlan,
    thresholds: FailureThresholds,
    windows: Sequence[tuple[str, str]],
    knob_space: Sequence[KnobSpec],
    max_keys: int,
    max_budget: int,
    allowed_operator_kinds: Sequence[str],
    reasoning_effort_hint: str,
) -> str:
    diagnostics_payload: dict[str, Any] = {
        "failure_mode": failure.failure_mode,
        "triggers": list(failure.triggers),
        "base_operator_plan": {
            "operator_kind": base_plan.operator_kind,
            "budget": base_plan.budget,
            "keys": list(base_plan.params.get("keys") or []),
            "max_keys": base_plan.params.get("max_keys"),
            "hypothesis": base_plan.hypothesis,
        },
        "thresholds": {
            "min_windows": thresholds.min_windows,
            "min_trades": thresholds.min_trades,
            "min_pairs": thresholds.min_pairs,
            "max_dd_pct": thresholds.max_dd_pct,
            "max_zero_pair_steps_pct": thresholds.max_zero_pair_steps_pct,
            "max_tail_pair_share": thresholds.max_tail_pair_share,
            "max_tail_period_share": thresholds.max_tail_period_share,
        },
        "windows": [{"start": start, "end": end} for start, end in windows],
        "diagnostics": None
        if diagnostics is None
        else {
            "run_group": diagnostics.run_group,
            "variant_id": diagnostics.variant_id,
            "sample_config_path": diagnostics.sample_config_path,
            "windows": diagnostics.windows,
            "worst_robust_sharpe": diagnostics.worst_robust_sharpe,
            "worst_dd_pct": diagnostics.worst_dd_pct,
            "worst_trades": diagnostics.worst_trades,
            "worst_pairs": diagnostics.worst_pairs,
            "worst_zero_pair_steps_pct": diagnostics.worst_zero_pair_steps_pct,
            "worst_tail_pair_share": diagnostics.worst_tail_pair_share,
            "worst_tail_period_share": diagnostics.worst_tail_period_share,
        },
        "knob_space_keys": [spec.key for spec in knob_space],
    }

    return (
        "You are an evolution policy planner for pair-trading configs.\n"
        "Return strictly one JSON object matching the output schema.\n"
        "Choose one operator and keep proposal conservative and actionable.\n"
        "Hard constraints:\n"
        f"- operator_kind must be one of: {', '.join(allowed_operator_kinds)}\n"
        f"- budget must be <= {int(max_budget)}\n"
        f"- keys length must be between 1 and {int(max_keys)}\n"
        "- keys must belong to knob_space_keys exactly.\n"
        f"- reasoning_effort_hint={reasoning_effort_hint}\n\n"
        "Context JSON:\n"
        f"{json.dumps(diagnostics_payload, ensure_ascii=False, sort_keys=True)}"
    )


def _resolve_llm_policy_plan(
    *,
    enabled: bool,
    model: str,
    effort: str,
    timeout_sec: int,
    codex_bin: str,
    repo_root: Path,
    diagnostics: VariantDiagnostics | None,
    failure: FailureAssessment,
    base_plan: TargetedOperatorPlan,
    thresholds: FailureThresholds,
    windows: Sequence[tuple[str, str]],
    knob_space: Sequence[KnobSpec],
    max_keys: int,
    max_budget: int,
    max_changed_keys: int,
) -> LlmPolicyPlan:
    if not enabled:
        return LlmPolicyPlan(
            enabled=False,
            used=False,
            source="disabled",
            reason="llm policy disabled",
            model=model,
            effort=effort,
            operator_plan=base_plan,
            payload=None,
        )

    _ = max_changed_keys
    schema = _build_llm_policy_schema(
        allowed_operator_kinds=_LLM_ALLOWED_OPERATOR_KINDS,
        max_keys=max_keys,
        max_budget=max_budget,
    )
    prompt = _build_llm_policy_prompt(
        diagnostics=diagnostics,
        failure=failure,
        base_plan=base_plan,
        thresholds=thresholds,
        windows=windows,
        knob_space=knob_space,
        max_keys=max_keys,
        max_budget=max_budget,
        allowed_operator_kinds=_LLM_ALLOWED_OPERATOR_KINDS,
        reasoning_effort_hint=effort,
    )
    try:
        payload = _run_codex_json(
            prompt=prompt,
            schema=schema,
            model_name=model,
            timeout_sec=timeout_sec,
            codex_bin=codex_bin,
            repo_root=repo_root,
        )
    except Exception as exc:  # noqa: BLE001
        return LlmPolicyPlan(
            enabled=True,
            used=False,
            source="llm_codex_exec",
            reason=f"llm policy fallback: {type(exc).__name__}: {exc}",
            model=model,
            effort=effort,
            operator_plan=base_plan,
            payload=None,
        )

    operator_kind = str(payload.get("operator_kind") or "").strip()
    if operator_kind not in _LLM_ALLOWED_OPERATOR_KINDS:
        return LlmPolicyPlan(
            enabled=True,
            used=False,
            source="llm_codex_exec",
            reason=f"invalid operator_kind from llm: {operator_kind!r}",
            model=model,
            effort=effort,
            operator_plan=base_plan,
            payload=payload,
        )
    keys = _sanitize_llm_keys(payload.get("keys"), knob_space=knob_space, max_keys=max_keys)
    if not keys:
        return LlmPolicyPlan(
            enabled=True,
            used=False,
            source="llm_codex_exec",
            reason="llm policy fallback: empty or invalid keys",
            model=model,
            effort=effort,
            operator_plan=base_plan,
            payload=payload,
        )
    budget = _coerce_int(payload.get("budget"), default=base_plan.budget)
    budget = max(1, min(int(max_budget), budget))
    override_plan = TargetedOperatorPlan(
        failure_mode=failure.failure_mode,
        operator_kind=operator_kind,
        budget=budget,
        params={"keys": keys, "max_keys": min(max_keys, _coerce_int(payload.get("max_keys"), default=max_keys))},
        hypothesis=f"LLM policy override for failure_mode={failure.failure_mode}",
    )
    return LlmPolicyPlan(
        enabled=True,
        used=True,
        source="llm_codex_exec",
        reason=str(payload.get("reasoning") or "").strip(),
        model=model,
        effort=effort,
        operator_plan=override_plan,
        payload=payload,
    )


def _infer_factor_category(target_key: str) -> str:
    root = str(target_key or "").split(".", 1)[0]
    if root in {"risk", "portfolio", "guards"}:
        return "risk"
    if root in {"pair_selection", "filter_params"}:
        return "selection"
    if root in {"data_filters", "data_processing"}:
        return "data"
    if root in {"time"}:
        return "execution"
    return "signal"


def _default_expected_effect(*, failure_mode: str) -> dict[str, Any]:
    mode = str(failure_mode or "").strip().lower()
    if mode == "dd":
        return {"sharpe": "up", "drawdown": "down", "turnover": "neutral", "notes": "tighter risk/guards"}
    if mode == "trades":
        return {"sharpe": "neutral", "drawdown": "neutral", "turnover": "up", "notes": "more activity via thresholds"}
    if mode == "zero_pair":
        return {"sharpe": "neutral", "drawdown": "neutral", "turnover": "up", "notes": "broader pair filters"}
    if mode == "tail":
        return {"sharpe": "neutral", "drawdown": "down", "turnover": "neutral", "notes": "reduce tail concentration"}
    return {"sharpe": "up", "drawdown": "neutral", "turnover": "neutral", "notes": "balanced tweak"}


def _build_llm_patch_schema(*, max_factors: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["title", "thesis", "confidence", "expected_effect", "factors", "wfa_checks", "reasoning"],
        "properties": {
            "hypothesis_id": {"type": "string"},
            "title": {"type": "string", "minLength": 8, "maxLength": 200},
            "thesis": {"type": "string", "minLength": 16, "maxLength": 1000},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "expected_effect": {
                "type": "object",
                "additionalProperties": False,
                "required": ["sharpe", "drawdown", "turnover"],
                "properties": {
                    "sharpe": {"type": "string", "enum": ["up", "down", "neutral"]},
                    "drawdown": {"type": "string", "enum": ["up", "down", "neutral"]},
                    "turnover": {"type": "string", "enum": ["up", "down", "neutral"]},
                    "notes": {"type": "string", "minLength": 1, "maxLength": 300},
                },
            },
            "factors": {
                "type": "array",
                "minItems": 1,
                "maxItems": int(max(1, max_factors)),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["target_key", "op", "value", "rationale"],
                    "properties": {
                        "factor_id": {"type": "string"},
                        "category": {
                            "type": "string",
                            "enum": [
                                "signal",
                                "selection",
                                "risk",
                                "execution",
                                "cost",
                                "regime",
                                "portfolio",
                                "data",
                            ],
                        },
                        "target_key": {"type": "string", "pattern": "^[A-Za-z0-9_]+(\\.[A-Za-z0-9_]+)+$"},
                        "op": {"type": "string", "enum": ["set", "scale", "offset", "enable", "disable"]},
                        "value": {"type": ["number", "integer", "string", "boolean", "null"]},
                        "bounds": {
                            "type": ["object", "null"],
                            "additionalProperties": False,
                            "required": ["lower", "upper"],
                            "properties": {"lower": {"type": "number"}, "upper": {"type": "number"}},
                        },
                        "rationale": {"type": "string", "minLength": 8, "maxLength": 500},
                    },
                },
            },
            "wfa_checks": {
                "type": "array",
                "minItems": 1,
                "maxItems": 12,
                "items": {"type": "string"},
            },
            "reasoning": {"type": "string", "minLength": 1, "maxLength": 2000},
        },
    }


def _build_llm_patch_prompt(
    *,
    failure: FailureAssessment,
    operator_plan: TargetedOperatorPlan,
    thresholds: FailureThresholds,
    knob_space: Sequence[KnobSpec],
    allowed_keys: Sequence[str],
    max_factors: int,
    existing_key_sets: Sequence[Sequence[str]],
    existing_root_signatures: Sequence[Sequence[str]],
    effort_hint: str,
) -> str:
    knob_payload = [
        {
            "key": spec.key,
            "type": spec.type,
            "min": spec.min,
            "max": spec.max,
            "step": spec.step,
            "choices": spec.choices,
        }
        for spec in knob_space
    ]
    context = {
        "failure_mode": failure.failure_mode,
        "triggers": list(failure.triggers),
        "operator_kind": operator_plan.operator_kind,
        "operator_keys": list(allowed_keys),
        "max_factors": int(max_factors),
        "thresholds": {
            "min_windows": thresholds.min_windows,
            "min_trades": thresholds.min_trades,
            "min_pairs": thresholds.min_pairs,
            "max_dd_pct": thresholds.max_dd_pct,
            "max_zero_pair_steps_pct": thresholds.max_zero_pair_steps_pct,
            "max_tail_pair_share": thresholds.max_tail_pair_share,
            "max_tail_period_share": thresholds.max_tail_period_share,
        },
        "knob_space": knob_payload,
        "avoid_key_sets": [list(keys) for keys in existing_key_sets[-20:]],
        "avoid_root_signatures": [list(roots) for roots in existing_root_signatures[-20:]],
        "effort_hint": str(effort_hint),
    }
    return (
        "You are generating a *pair-crypto* config patch hypothesis.\n"
        "Return strictly one JSON object matching the output schema.\n"
        "Hard constraints:\n"
        "- Use only target_key from operator_keys.\n"
        f"- Factors count must be 1..{int(max_factors)}.\n"
        "- Be conservative: prefer small, testable changes.\n"
        "- Diversity: avoid repeating avoid_key_sets and avoid_root_signatures.\n\n"
        "Context JSON:\n"
        f"{json.dumps(context, ensure_ascii=False, sort_keys=True)}"
    )


def _sanitize_llm_patch_payload(
    payload: dict[str, Any],
    *,
    generation: int,
    variant_index: int,
    failure_mode: str,
    allowed_keys: Sequence[str],
    knob_space: Sequence[KnobSpec],
    max_factors: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    allowed = {str(key) for key in allowed_keys}
    knob_by_key = {spec.key: spec for spec in knob_space}

    title = str(payload.get("title") or f"AUTO hypothesis g{generation:02d} v{variant_index:03d}").strip()
    if len(title) < 8:
        title = f"AUTO hypothesis g{generation:02d} v{variant_index:03d}"

    thesis = str(payload.get("thesis") or "").strip()
    if len(thesis) < 16:
        thesis = f"{failure_mode}: auto patch candidate for pair-crypto evolution."

    hypothesis_id = str(payload.get("hypothesis_id") or "").strip()
    if not hypothesis_id:
        hypothesis_id = f"HYP-AUTO_G{generation:02d}_V{variant_index:03d}"

    confidence_raw = payload.get("confidence")
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.6
    confidence = min(1.0, max(0.0, confidence))

    expected_effect = payload.get("expected_effect")
    if not isinstance(expected_effect, dict):
        expected_effect = _default_expected_effect(failure_mode=failure_mode)

    raw_factors = payload.get("factors")
    factors_out: list[dict[str, Any]] = []
    if isinstance(raw_factors, list):
        for item in raw_factors:
            if not isinstance(item, dict):
                continue
            target_key = str(item.get("target_key") or "").strip()
            if not target_key or target_key not in allowed:
                continue
            op = str(item.get("op") or "").strip()
            if op not in {"set", "scale", "offset", "enable", "disable"}:
                continue
            spec = knob_by_key.get(target_key)
            if spec is None:
                continue
            factor_id = str(item.get("factor_id") or "").strip() or f"F-AUTO{len(factors_out)+1:02d}"
            category = str(item.get("category") or _infer_factor_category(target_key)).strip()
            if category not in {"signal", "selection", "risk", "execution", "cost", "regime", "portfolio", "data"}:
                category = _infer_factor_category(target_key)
            rationale = str(item.get("rationale") or f"auto factor for {target_key}").strip()
            if len(rationale) < 8:
                rationale = f"auto factor for {target_key}"

            factor: dict[str, Any] = {
                "factor_id": factor_id,
                "category": category,
                "target_key": target_key,
                "op": op,
                "value": item.get("value"),
                "rationale": rationale,
            }
            bounds = item.get("bounds")
            if bounds is None and spec.type in {"int", "float"} and spec.min is not None and spec.max is not None and op in {"set", "scale", "offset"}:
                factor["bounds"] = {"lower": float(spec.min), "upper": float(spec.max)}
            elif isinstance(bounds, dict):
                factor["bounds"] = bounds

            factors_out.append(factor)
            if len(factors_out) >= max(1, int(max_factors)):
                break

    # Unique target keys only (fail-closed: keep first occurrence)
    deduped: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for factor in factors_out:
        key = str(factor.get("target_key") or "").strip()
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(factor)
    factors_out = deduped

    if not factors_out:
        raise ValueError("llm patch payload produced no valid factors after sanitization")

    wfa_checks = payload.get("wfa_checks")
    checks_out: list[str] = []
    if isinstance(wfa_checks, list):
        for item in wfa_checks:
            token = str(item or "").strip()
            if token:
                checks_out.append(token)
    if not checks_out:
        checks_out = ["oos_sharpe", "worst_window_dd", "trade_count", "tail_loss_concentration"]

    hypothesis = {
        "hypothesis_id": hypothesis_id,
        "title": title,
        "thesis": thesis,
        "priority": 1,
        "confidence": confidence,
        "expected_effect": expected_effect,
        "factors": factors_out,
        "wfa_checks": checks_out[:12],
    }
    return hypothesis, factors_out


def _deterministic_patch_candidate(
    *,
    parent_cfg: Mapping[str, Any],
    generation: int,
    variant_index: int,
    failure_mode: str,
    allowed_keys: Sequence[str],
    knob_space: Sequence[KnobSpec],
    rng: np.random.Generator,
    max_factors: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    knob_by_key = {spec.key: spec for spec in knob_space}
    keys = [key for key in allowed_keys if key in knob_by_key]
    if not keys:
        keys = [spec.key for spec in knob_space]

    k = int(rng.integers(1, max(2, min(int(max_factors), len(keys) + 1))))
    selected = [keys[idx] for idx in rng.choice(len(keys), size=min(k, len(keys)), replace=False)]

    factors: list[dict[str, Any]] = []
    for i, key in enumerate(selected, start=1):
        spec = knob_by_key[key]
        current = None
        try:
            current = get_dotted(parent_cfg, key)
        except Exception:  # noqa: BLE001
            current = None

        op = "set"
        value: Any = None
        if spec.type == "bool":
            value = not bool(current) if isinstance(current, bool) else True
        elif spec.type == "categorical" and spec.choices:
            choices = [c for c in spec.choices if c != current]
            value = choices[0] if choices else (spec.choices[0] if spec.choices else current)
        elif spec.type == "string":
            value = str(current or "").strip() or "auto"
        else:
            # numeric
            lo = float(spec.min) if spec.min is not None else None
            hi = float(spec.max) if spec.max is not None else None
            step = float(spec.step) if spec.step is not None and spec.step > 0 else None
            base = float(current) if isinstance(current, (int, float)) and not isinstance(current, bool) else (lo or 0.0)
            direction = -1.0 if float(rng.random()) < 0.5 else 1.0
            if step is None:
                step = (hi - lo) / 10.0 if (lo is not None and hi is not None and hi > lo) else 0.1
            candidate = base + direction * step
            if lo is not None:
                candidate = max(lo, candidate)
            if hi is not None:
                candidate = min(hi, candidate)
            value = int(candidate) if spec.type == "int" else float(candidate)

        factor: dict[str, Any] = {
            "factor_id": f"F-AUTO{i:02d}",
            "category": _infer_factor_category(key),
            "target_key": key,
            "op": op,
            "value": value,
            "bounds": None if spec.min is None or spec.max is None else {"lower": float(spec.min), "upper": float(spec.max)},
            "rationale": f"deterministic tweak for {key}",
        }
        if factor.get("bounds") is None:
            factor.pop("bounds", None)
        factors.append(factor)

    hypothesis = {
        "hypothesis_id": f"HYP-DETER_G{generation:02d}_V{variant_index:03d}",
        "title": f"DETER hypothesis g{generation:02d} v{variant_index:03d}",
        "thesis": f"{failure_mode}: deterministic patch candidate (no LLM).",
        "priority": 2,
        "confidence": 0.4,
        "expected_effect": _default_expected_effect(failure_mode=failure_mode),
        "factors": factors,
        "wfa_checks": ["oos_sharpe", "worst_window_dd", "trade_count"],
    }
    return hypothesis, factors


def _flatten_patch_leaves(patch: Mapping[str, Any], *, prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for key, value in patch.items():
        if str(key) == "base_config":
            continue
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.extend(_flatten_patch_leaves(value, prefix=dotted))
            continue
        out.append((dotted, value))
    return out


def _factors_from_materialized_patch(
    patch: Mapping[str, Any],
    *,
    allowed_keys: set[str],
    max_items: int,
    prefix: str,
) -> list[dict[str, Any]]:
    factors: list[dict[str, Any]] = []
    for idx, (key, value) in enumerate(_flatten_patch_leaves(patch), start=1):
        target_key = str(key).strip()
        if not target_key or target_key not in allowed_keys:
            continue
        factors.append(
            {
                "factor_id": f"F-{prefix}{idx:02d}",
                "category": _infer_factor_category(target_key),
                "target_key": target_key,
                "op": "set",
                "value": value,
                "rationale": f"crossover inherit {target_key}",
            }
        )
        if len(factors) >= max(1, int(max_items)):
            break
    # Deduplicate by target_key
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for factor in factors:
        key = str(factor.get("target_key") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(factor)
    return out


def _segment_crossover_candidate(
    *,
    parent_a: Mapping[str, Any],
    parent_b: Mapping[str, Any],
    generation: int,
    variant_index: int,
    failure_mode: str,
    allowed_keys: Sequence[str],
    rng: np.random.Generator,
    max_factors: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], tuple[str, str]]:
    allowed = {str(key) for key in allowed_keys if str(key).strip()}
    if not allowed:
        raise ValueError("empty allowed_keys for crossover")

    patch_a = parent_a.get("materialized_patch")
    patch_b = parent_b.get("materialized_patch")
    if not isinstance(patch_a, Mapping) or not isinstance(patch_b, Mapping):
        raise ValueError("missing parent materialized_patch")

    factors_a = _factors_from_materialized_patch(patch_a, allowed_keys=allowed, max_items=max_factors, prefix="A")
    factors_b = _factors_from_materialized_patch(patch_b, allowed_keys=allowed, max_items=max_factors, prefix="B")
    if not factors_a or not factors_b:
        raise ValueError("parents have no usable factors for crossover")

    def _group(items: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            key = str(item.get("target_key") or "").strip()
            root = key.split(".", 1)[0] if key else ""
            if not root:
                continue
            grouped.setdefault(root, []).append(dict(item))
        return grouped

    grouped_a = _group(factors_a)
    grouped_b = _group(factors_b)
    roots = sorted(set(grouped_a).union(grouped_b))
    if not roots:
        raise ValueError("no roots for crossover")

    chosen: dict[str, str] = {}
    for root in roots:
        if root in grouped_a and root in grouped_b:
            chosen[root] = "a" if float(rng.random()) < 0.5 else "b"
        elif root in grouped_a:
            chosen[root] = "a"
        else:
            chosen[root] = "b"

    # Ensure both parents contribute when possible.
    if len(roots) >= 2:
        if all(src == "a" for src in chosen.values()):
            chosen[roots[-1]] = "b"
        if all(src == "b" for src in chosen.values()):
            chosen[roots[-1]] = "a"

    merged: list[dict[str, Any]] = []
    for root in roots:
        src = chosen[root]
        merged.extend(grouped_a[root] if src == "a" else grouped_b[root])

    # Deduplicate by target_key and cap to max_factors.
    out: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in merged:
        key = str(item.get("target_key") or "").strip()
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        out.append(item)
    if len(out) > max(1, int(max_factors)):
        keep = rng.choice(len(out), size=int(max_factors), replace=False)
        out = [out[int(i)] for i in sorted(keep)]

    parent_a_id = str(parent_a.get("candidate_id") or "").strip() or "parent_a"
    parent_b_id = str(parent_b.get("candidate_id") or "").strip() or "parent_b"
    hypothesis = {
        "hypothesis_id": f"HYP-XOVER_G{generation:02d}_V{variant_index:03d}",
        "title": f"XOVER g{generation:02d} v{variant_index:03d}",
        "thesis": f"{failure_mode}: segment-crossover between {parent_a_id} and {parent_b_id}.",
        "priority": 2,
        "confidence": 0.5,
        "expected_effect": _default_expected_effect(failure_mode=failure_mode),
        "factors": out,
        "wfa_checks": ["oos_sharpe", "worst_window_dd", "trade_count", "tail_loss_concentration"],
    }
    return hypothesis, out, (parent_a_id, parent_b_id)


def _factor_root_signature(factors: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    roots: set[str] = set()
    for factor in factors:
        key = str(factor.get("target_key") or "").strip()
        root = key.split(".", 1)[0] if key else ""
        if root:
            roots.add(root)
    return tuple(sorted(roots))


def _segment_mutation_preferred_roots(failure_mode: str) -> tuple[str, ...]:
    mode = str(failure_mode or "").strip().lower()
    if mode == "dd":
        return ("portfolio", "backtest", "risk", "guards")
    if mode == "trades":
        return ("backtest", "time", "portfolio")
    if mode == "zero_pair":
        return ("pair_selection", "filter_params", "data_filters", "data_processing")
    if mode == "tail":
        return ("portfolio", "backtest", "pair_selection", "guards")
    return ("portfolio", "backtest", "pair_selection", "filter_params")


def _mutate_factor_for_segment(
    factor: Mapping[str, Any],
    *,
    spec: KnobSpec | None,
    rng: np.random.Generator,
) -> dict[str, Any]:
    out = dict(factor)
    value = out.get("value")
    if spec is None:
        out["op"] = "set"
        return out

    if spec.type == "bool":
        current = bool(value) if isinstance(value, bool) else False
        out["op"] = "set"
        out["value"] = not current
        return out
    if spec.type == "categorical" and spec.choices:
        choices = [choice for choice in list(spec.choices) if choice != value]
        out["op"] = "set"
        out["value"] = choices[0] if choices else spec.choices[0]
        return out
    if spec.type == "string":
        out["op"] = "set"
        out["value"] = str(value or "").strip() or "auto"
        return out

    # Numeric branch: mutate by one step (or small range fraction) and keep bounds.
    lo = float(spec.min) if spec.min is not None else None
    hi = float(spec.max) if spec.max is not None else None
    step = float(spec.step) if spec.step is not None and float(spec.step) > 0 else None
    base = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else (lo or 0.0)
    if step is None:
        if lo is not None and hi is not None and hi > lo:
            step = (hi - lo) / 10.0
        else:
            step = 0.1
    direction = -1.0 if float(rng.random()) < 0.5 else 1.0
    candidate = base + direction * float(step)
    if lo is not None:
        candidate = max(lo, candidate)
    if hi is not None:
        candidate = min(hi, candidate)
    out["op"] = "set"
    if spec.type == "int":
        out["value"] = int(round(candidate))
    else:
        out["value"] = float(candidate)
    return out


def _segment_mutation_candidate(
    *,
    parent_entry: Mapping[str, Any],
    generation: int,
    variant_index: int,
    failure_mode: str,
    allowed_keys: Sequence[str],
    knob_space: Sequence[KnobSpec],
    rng: np.random.Generator,
    max_factors: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    allowed = {str(key) for key in allowed_keys if str(key).strip()}
    if not allowed:
        raise ValueError("empty allowed_keys for segment mutation")
    raw_factors = parent_entry.get("factors")
    if not isinstance(raw_factors, list) or not raw_factors:
        raise ValueError("parent entry has no factors")

    factors_seed: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in raw_factors:
        if not isinstance(item, Mapping):
            continue
        target_key = str(item.get("target_key") or "").strip()
        if not target_key or target_key not in allowed or target_key in seen_keys:
            continue
        seen_keys.add(target_key)
        factor = dict(item)
        factor["target_key"] = target_key
        factor["op"] = str(item.get("op") or "set").strip() or "set"
        factor["rationale"] = str(item.get("rationale") or f"inherit {target_key}").strip() or f"inherit {target_key}"
        factors_seed.append(factor)
    if not factors_seed:
        raise ValueError("parent factors do not intersect allowed_keys")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for factor in factors_seed:
        key = str(factor.get("target_key") or "").strip()
        root = key.split(".", 1)[0] if key else ""
        if not root:
            continue
        grouped.setdefault(root, []).append(factor)
    if not grouped:
        raise ValueError("unable to build groups for segment mutation")

    preferred = [root for root in _segment_mutation_preferred_roots(failure_mode) if root in grouped]
    roots_pool = preferred if preferred else sorted(grouped.keys())
    mutate_root = str(roots_pool[int(rng.integers(0, len(roots_pool)))])

    knob_by_key = {spec.key: spec for spec in knob_space}
    out: list[dict[str, Any]] = []
    for root in sorted(grouped.keys()):
        for index, factor in enumerate(grouped[root], start=1):
            current = dict(factor)
            key = str(current.get("target_key") or "").strip()
            if root == mutate_root:
                mutated = _mutate_factor_for_segment(current, spec=knob_by_key.get(key), rng=rng)
                mutated["factor_id"] = str(current.get("factor_id") or f"F-MUT{index:02d}")
                mutated["category"] = str(current.get("category") or _infer_factor_category(key)).strip() or _infer_factor_category(key)
                mutated["rationale"] = f"segment_mutation[{mutate_root}] for {key}"
                out.append(mutated)
            else:
                current["rationale"] = f"frozen prefix segment[{root}]"
                out.append(current)

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for factor in out:
        key = str(factor.get("target_key") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(factor)
    out = deduped
    if len(out) > max(1, int(max_factors)):
        # Keep mutated root first, then trim deterministically by order.
        out.sort(key=lambda item: (0 if str(item.get("target_key") or "").startswith(f"{mutate_root}.") else 1, str(item.get("target_key") or "")))
        out = out[: int(max_factors)]
    if not out:
        raise ValueError("segment mutation produced no factors")

    parent_id = str(parent_entry.get("candidate_id") or "").strip() or "parent_patch"
    hypothesis = {
        "hypothesis_id": f"HYP-MUTSEG_G{generation:02d}_V{variant_index:03d}",
        "title": f"MUTSEG g{generation:02d} v{variant_index:03d}",
        "thesis": (
            f"{failure_mode}: segment-mutation on root `{mutate_root}` from {parent_id} "
            f"with frozen non-target segments."
        ),
        "priority": 2,
        "confidence": 0.5,
        "expected_effect": _default_expected_effect(failure_mode=failure_mode),
        "factors": out,
        "wfa_checks": ["oos_sharpe", "worst_window_dd", "trade_count", "tail_loss_concentration"],
    }
    return hypothesis, out, parent_id


def _extract_candidate_id_from_variant_id(variant_id: str) -> str | None:
    text = str(variant_id or "").strip()
    marker = "_v"
    pos = text.find(marker)
    if pos < 0:
        return None
    # expected: <run_group>_vNNN_<candidate_id>
    rest = text[pos:]
    parts = rest.split("_", 2)
    if len(parts) < 3:
        return None
    candidate = str(parts[2]).strip()
    return candidate or None


def _generate_patch_ast_proposals(
    *,
    num_variants: int,
    parent_cfg: Mapping[str, Any],
    parent_cfg_ref: str,
    parent_genome: Genome,
    parent_id: str,
    knob_space: Sequence[KnobSpec],
    operator_plan: TargetedOperatorPlan,
    failure: FailureAssessment,
    thresholds: FailureThresholds,
    diagnostics: Sequence[VariantDiagnostics],
    generation: int,
    rng: np.random.Generator,
    seen_pool: list[tuple[str, Genome]],
    dedupe_distance: float,
    max_changed_keys: int,
    use_llm: bool,
    llm_model: str,
    llm_effort: str,
    llm_timeout_sec: int,
    llm_codex_bin: str,
    repo_root: Path,
    decisions_dir: Path,
    gate_limits: PatchGateLimits,
    llm_verify_semantic: bool,
    max_attempts_per_variant: int,
) -> list[CandidateProposal]:
    _ensure_local_optimization_scripts_on_path()
    from hypothesis_factor_dsl import materialize_hypothesis_factor_patch
    from semantic_consistency_gate import run_semantic_gate

    allowed_keys = [str(key) for key in list(operator_plan.params.get("keys") or []) if str(key).strip()]
    if not allowed_keys:
        allowed_keys = [spec.key for spec in knob_space]
    allowed_set = {spec.key for spec in knob_space}
    allowed_keys = [key for key in allowed_keys if key in allowed_set]
    if not allowed_keys:
        allowed_keys = [spec.key for spec in knob_space]

    zoo_entries = _load_patch_zoo_from_decisions(decisions_dir, max_items=200)
    zoo_ast = [(str(entry.get("candidate_id") or ""), entry.get("ast")) for entry in zoo_entries if entry.get("ast") is not None]
    zoo_by_id = {str(entry.get("candidate_id") or "").strip(): entry for entry in zoo_entries}
    batch_ast_zoo: list[tuple[str, Any]] = []
    existing_key_sets: list[list[str]] = []
    existing_root_signatures: list[tuple[str, ...]] = []

    schema = _build_llm_patch_schema(max_factors=int(max_changed_keys))

    proposals: list[CandidateProposal] = []
    local_seen: set[str] = set()
    operator_kind = str(operator_plan.operator_kind or "").strip()

    for variant_index in range(1, int(num_variants) + 1):
        accepted = False
        for _attempt in range(max(1, int(max_attempts_per_variant))):
            hypothesis: dict[str, Any]
            factors: list[dict[str, Any]]
            source: str
            parents_for_candidate: list[str] = [parent_id]
            parent_id_for_candidate = parent_id

            has_candidate = False
            if operator_kind == "crossover_uniform_v1" and len(zoo_entries) >= 2:
                ranked: list[dict[str, Any]] = []
                seen_ids: set[str] = set()
                for diag in diagnostics:
                    cid = _extract_candidate_id_from_variant_id(str(diag.variant_id))
                    if not cid or cid in seen_ids:
                        continue
                    entry = zoo_by_id.get(cid)
                    if entry is None:
                        continue
                    ranked.append(entry)
                    seen_ids.add(cid)
                    if len(ranked) >= 12:
                        break
                pool = ranked if len(ranked) >= 2 else list(zoo_entries)
                pool = pool[: min(12, len(pool))]
                if len(pool) >= 2:
                    idxs = rng.choice(len(pool), size=2, replace=False)
                    pa = pool[int(idxs[0])]
                    pb = pool[int(idxs[1])]
                    try:
                        hypothesis, factors, parent_pair = _segment_crossover_candidate(
                            parent_a=pa,
                            parent_b=pb,
                            generation=generation,
                            variant_index=variant_index,
                            failure_mode=str(failure.failure_mode or ""),
                            allowed_keys=allowed_keys,
                            rng=rng,
                            max_factors=int(max_changed_keys),
                        )
                        parents_for_candidate = [parent_pair[0], parent_pair[1]]
                        parent_id_for_candidate = f"parents::{parent_pair[0]}+{parent_pair[1]}"
                        source = "crossover_patch_v1"
                        has_candidate = True
                    except Exception:  # noqa: BLE001
                        has_candidate = False

            if not has_candidate:
                if operator_kind in {"mutate_step_v1", "coordinate_sweep_v1"} and zoo_entries:
                    ranked: list[dict[str, Any]] = []
                    seen_ids: set[str] = set()
                    for diag in diagnostics:
                        cid = _extract_candidate_id_from_variant_id(str(diag.variant_id))
                        if not cid or cid in seen_ids:
                            continue
                        entry = zoo_by_id.get(cid)
                        if entry is None:
                            continue
                        ranked.append(entry)
                        seen_ids.add(cid)
                        if len(ranked) >= 12:
                            break
                    pool = ranked if ranked else list(zoo_entries)
                    if pool:
                        parent_patch = pool[int(rng.integers(0, len(pool)))]
                        try:
                            hypothesis, factors, parent_patch_id = _segment_mutation_candidate(
                                parent_entry=parent_patch,
                                generation=generation,
                                variant_index=variant_index,
                                failure_mode=str(failure.failure_mode or ""),
                                allowed_keys=allowed_keys,
                                knob_space=knob_space,
                                rng=rng,
                                max_factors=int(max_changed_keys),
                            )
                            parents_for_candidate = [parent_patch_id]
                            parent_id_for_candidate = f"parent::{parent_patch_id}"
                            source = "segment_mutation_patch_v1"
                            has_candidate = True
                        except Exception:  # noqa: BLE001
                            has_candidate = False

            if not has_candidate:
                if use_llm:
                    prompt = _build_llm_patch_prompt(
                        failure=failure,
                        operator_plan=operator_plan,
                        thresholds=thresholds,
                        knob_space=knob_space,
                        allowed_keys=allowed_keys,
                        max_factors=int(max_changed_keys),
                        existing_key_sets=existing_key_sets,
                        existing_root_signatures=existing_root_signatures,
                        effort_hint=str(llm_effort),
                    )
                    try:
                        payload = _run_codex_json(
                            prompt=prompt,
                            schema=schema,
                            model_name=str(llm_model),
                            timeout_sec=int(llm_timeout_sec),
                            codex_bin=str(llm_codex_bin),
                            repo_root=repo_root,
                        )
                        hypothesis, factors = _sanitize_llm_patch_payload(
                            payload,
                            generation=generation,
                            variant_index=variant_index,
                            failure_mode=str(failure.failure_mode or ""),
                            allowed_keys=allowed_keys,
                            knob_space=knob_space,
                            max_factors=int(max_changed_keys),
                        )
                    except Exception:  # noqa: BLE001
                        continue
                    source = "llm_patch_v1"
                else:
                    hypothesis, factors = _deterministic_patch_candidate(
                        parent_cfg=parent_cfg,
                        generation=generation,
                        variant_index=variant_index,
                        failure_mode=str(failure.failure_mode or ""),
                        allowed_keys=allowed_keys,
                        knob_space=knob_space,
                        rng=rng,
                        max_factors=int(max_changed_keys),
                    )
                    source = "deterministic_patch_v1"

            root_signature = _factor_root_signature(factors)
            diversity_target = max(2, min(int(num_variants), 4))
            if (
                root_signature
                and root_signature in existing_root_signatures
                and len(existing_root_signatures) < diversity_target
            ):
                continue

            existing_key_sets.append([str(item.get("target_key") or "").strip() for item in factors])
            if root_signature:
                existing_root_signatures.append(root_signature)

            dsl_payload = {
                "dsl_version": "pair_crypto_hypothesis_factor.v1",
                "strategy_scope": "pair_crypto",
                "metadata": {"generated_by": source, "generated_at": _utc_now_iso(), "source": source},
                "hypotheses": [hypothesis],
            }
            try:
                patch = materialize_hypothesis_factor_patch(
                    dsl_payload,
                    hypothesis_id=str(hypothesis.get("hypothesis_id") or ""),
                    base_config=dict(parent_cfg),
                    base_config_ref=parent_cfg_ref,
                )
            except Exception:  # noqa: BLE001
                continue

            gate = evaluate_patch_gates(
                factors,
                zoo=[(cid, ast) for cid, ast in (zoo_ast + batch_ast_zoo)],
                limits=gate_limits,
            )
            if not gate.ok:
                continue

            semantic = run_semantic_gate(
                hypothesis=hypothesis,
                factors=factors,
                materialized_patch=patch,
                use_llm=bool(llm_verify_semantic and use_llm),
                model=str(llm_model),
                effort=str(llm_effort),
                codex_bin=str(llm_codex_bin),
                timeout_sec=int(llm_timeout_sec),
                repo_root=repo_root,
                fail_open_on_llm_error=False,
            )
            if not semantic.ok:
                continue

            child_cfg_payload = copy.deepcopy(dict(parent_cfg))
            _apply_materialized_patch(child_cfg_payload, patch)
            child_genome = genome_from_config(child_cfg_payload, knob_space=knob_space)

            patch_ir = {
                "ir_version": "config_patch_ast.v1",
                "source": source,
                "parents": list(parents_for_candidate),
                "hypothesis": hypothesis,
                "factors": factors,
                "materialized_patch": patch,
                "gates": {
                    "complexity": gate.complexity,
                    "redundancy": gate.redundancy,
                    "limits": {
                        "max_complexity_score": gate_limits.max_complexity_score,
                        "max_redundancy_similarity": gate_limits.max_redundancy_similarity,
                        "alpha_sl": gate_limits.alpha_sl,
                        "alpha_pc": gate_limits.alpha_pc,
                        "alpha_feat": gate_limits.alpha_feat,
                    },
                },
                "semantic_gate": {
                    "ok": semantic.ok,
                    "source": semantic.source,
                    "reasons": list(semantic.reasons),
                    "model": semantic.model,
                    "effort": semantic.effort,
                    "error": semantic.error,
                },
            }
            if validate_patch_ir_payload(patch_ir):
                continue

            candidate = _build_candidate(
                child=child_genome,
                parent_genome=parent_genome,
                parent_id=parent_id_for_candidate,
                parents=parents_for_candidate,
                operator_id="op_targeted_primary",
                generation=generation,
                seen_pool=seen_pool,
                knob_space=knob_space,
                patch_ir=patch_ir,
            )
            if len(candidate.changed_keys) > int(max_changed_keys):
                continue
            if candidate.nearest_id and candidate.nearest_distance <= float(dedupe_distance):
                continue
            if candidate.candidate_id in local_seen:
                continue
            local_seen.add(candidate.candidate_id)
            proposals.append(candidate)
            seen_pool.append((candidate.candidate_id, candidate.genome))
            batch_ast_zoo.append((candidate.candidate_id, gate.ast))
            accepted = True
            break

        if not accepted:
            print(
                f"warn: unable to generate patch_ast proposal {variant_index}/{num_variants} "
                f"after {max_attempts_per_variant} attempts (check gates/allowed_keys/llm)",
                file=sys.stderr,
            )
            continue

    return proposals


def _top_up_with_genome_proposals(
    *,
    existing_proposals: Sequence[CandidateProposal],
    target_num_variants: int,
    parent_genome: Genome,
    parent_id: str,
    knob_space: Sequence[KnobSpec],
    operator_plan: TargetedOperatorPlan,
    failure_assessment: FailureAssessment,
    generation: int,
    rng: np.random.Generator,
    seen_pool: list[tuple[str, Genome]],
    dedupe_distance: float,
    max_changed_keys: int,
) -> list[CandidateProposal]:
    missing = max(0, int(target_num_variants) - len(existing_proposals))
    if missing <= 0:
        return []

    blocked_ids: set[str] = {str(item.candidate_id) for item in existing_proposals}
    key_pool = [spec.key for spec in knob_space]
    base_budget = max(8, int(operator_plan.budget))

    def _keys(limit: int) -> list[str]:
        if not key_pool:
            return []
        return list(key_pool[: max(1, min(limit, len(key_pool)))])

    fallback_plans: list[TargetedOperatorPlan] = [
        operator_plan,
        TargetedOperatorPlan(
            failure_mode=operator_plan.failure_mode,
            operator_kind="mutate_step_v1",
            budget=max(12, min(base_budget, 32)),
            params={"keys": _keys(12)},
            hypothesis=f"{operator_plan.hypothesis} [topup_mutate]",
        ),
        TargetedOperatorPlan(
            failure_mode=operator_plan.failure_mode,
            operator_kind="coordinate_sweep_v1",
            budget=max(8, min(base_budget, 24)),
            params={"keys": _keys(10)},
            hypothesis=f"{operator_plan.hypothesis} [topup_sweep]",
        ),
        TargetedOperatorPlan(
            failure_mode=operator_plan.failure_mode,
            operator_kind="random_restart_v1",
            budget=max(12, min(base_budget, 48)),
            params={"keys": _keys(16)},
            hypothesis=f"{operator_plan.hypothesis} [topup_restart]",
        ),
    ]

    topup: list[CandidateProposal] = []
    relax_schedule = [
        (float(dedupe_distance), int(max_changed_keys)),
        (max(0.0, float(dedupe_distance) * 0.50), max(int(max_changed_keys), 4)),
        (max(0.0, float(dedupe_distance) * 0.25), max(int(max_changed_keys), 6)),
        (0.0, max(int(max_changed_keys), 8)),
    ]
    for idx, plan in enumerate(fallback_plans):
        remaining = missing - len(topup)
        if remaining <= 0:
            break
        relax_idx = min(idx, len(relax_schedule) - 1)
        dedupe_current, max_keys_current = relax_schedule[relax_idx]
        try:
            batch = _generate_proposals(
                num_variants=remaining,
                parent_genome=parent_genome,
                parent_id=parent_id,
                knob_space=knob_space,
                operator_plan=plan,
                failure_assessment=failure_assessment,
                generation=generation,
                rng=rng,
                seen_pool=seen_pool,
                dedupe_distance=float(dedupe_current),
                max_changed_keys=int(max_keys_current),
                blocked_candidate_ids=blocked_ids,
            )
        except SystemExit:
            continue
        topup.extend(batch)

    # Final safety valve: if strict dedupe/similarity still leaves a tiny batch,
    # force-fill with one-step mutations from parent/accepted genomes.
    remaining_after_topup = missing - len(topup)
    if remaining_after_topup > 0 and knob_space:
        mutation_bases: list[Genome] = [dict(parent_genome)]
        mutation_bases.extend(dict(item.genome) for item in existing_proposals)
        mutation_bases.extend(dict(item.genome) for item in topup)
        key_pool = [spec.key for spec in knob_space]
        genome_hashes: set[str] = {
            json.dumps(canonicalize_object(item.genome), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            for item in list(existing_proposals) + list(topup)
        }
        max_attempts = max(512, remaining_after_topup * 256)
        attempts = 0
        while len(topup) < missing and attempts < max_attempts:
            attempts += 1
            base = mutation_bases[int(rng.integers(0, len(mutation_bases)))]
            key = key_pool[int(rng.integers(0, len(key_pool)))]
            try:
                children = apply_operator_v1(
                    "mutate_step_v1",
                    parents=[base],
                    knob_space=knob_space,
                    rng=rng,
                    budget=1,
                    params={"key": key},
                )
            except Exception:
                continue
            if not children:
                continue

            child = children[0]
            encoded = json.dumps(canonicalize_object(child), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            if encoded in genome_hashes:
                continue

            candidate = _build_candidate(
                child=child,
                parent_genome=parent_genome,
                parent_id=parent_id,
                parents=[parent_id],
                operator_id=f"op_topup_force_mutate_{attempts}",
                generation=generation,
                seen_pool=seen_pool,
                knob_space=knob_space,
            )
            if len(candidate.changed_keys) > max(int(max_changed_keys), 8):
                continue
            if candidate.candidate_id in blocked_ids:
                continue

            blocked_ids.add(candidate.candidate_id)
            genome_hashes.add(encoded)
            topup.append(candidate)
            seen_pool.append((candidate.candidate_id, candidate.genome))
            mutation_bases.append(dict(candidate.genome))

    return topup


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return payload if isinstance(payload, dict) else {}


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _default_invalid_proposal_index_path(app_root: Path) -> Path:
    return app_root / "artifacts" / "wfa" / "aggregate" / ".autonomous" / "invalid_proposal_index.json"


def _load_invalid_proposal_index(path: Path) -> dict[str, Any]:
    payload = _load_state(path)
    entries = payload.get("entries")
    if not isinstance(entries, list):
        entries = []
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        fingerprint = str(entry.get("fingerprint") or "").strip()
        code = str(entry.get("code") or "").strip().upper()
        if not fingerprint or not code:
            continue
        normalized.append(
            {
                "fingerprint": fingerprint,
                "code": code,
                "reason": str(entry.get("reason") or "").strip(),
                "occurrences": max(1, _coerce_int(entry.get("occurrences"), default=1)),
                "operator_id": str(entry.get("operator_id") or "").strip(),
                "candidate_id": str(entry.get("candidate_id") or "").strip(),
                "changed_keys": [str(key) for key in list(entry.get("changed_keys") or []) if str(key).strip()],
                "first_seen_at": str(entry.get("first_seen_at") or "").strip(),
                "last_seen_at": str(entry.get("last_seen_at") or "").strip(),
            }
        )
    return {
        "schema_version": "v1",
        "updated_at": str(payload.get("updated_at") or "").strip(),
        "entries": normalized,
    }


def _upsert_invalid_proposal_index_entry(
    *,
    state: dict[str, Any],
    fingerprint: str,
    code: str,
    reason: str,
    operator_id: str,
    candidate_id: str,
    changed_keys: Sequence[str],
) -> dict[str, Any]:
    entries = state.setdefault("entries", [])
    if not isinstance(entries, list):
        entries = []
        state["entries"] = entries
    now = _utc_now_iso()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("fingerprint") or "").strip() != fingerprint:
            continue
        entry["code"] = code
        entry["reason"] = reason
        entry["operator_id"] = operator_id
        entry["candidate_id"] = candidate_id
        entry["changed_keys"] = [str(key) for key in list(changed_keys or []) if str(key).strip()]
        entry["occurrences"] = max(1, _coerce_int(entry.get("occurrences"), default=1)) + 1
        if not str(entry.get("first_seen_at") or "").strip():
            entry["first_seen_at"] = now
        entry["last_seen_at"] = now
        state["updated_at"] = now
        return entry

    entry = {
        "fingerprint": fingerprint,
        "code": code,
        "reason": reason,
        "occurrences": 1,
        "operator_id": operator_id,
        "candidate_id": candidate_id,
        "changed_keys": [str(key) for key in list(changed_keys or []) if str(key).strip()],
        "first_seen_at": now,
        "last_seen_at": now,
    }
    entries.append(entry)
    state["updated_at"] = now
    return entry


def _find_invalid_proposal_index_entry(state: Mapping[str, Any], fingerprint: str) -> Mapping[str, Any] | None:
    entries = state.get("entries")
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("fingerprint") or "").strip() == fingerprint:
            return entry
    return None


def _dump_invalid_proposal_index(path: Path, payload: Mapping[str, Any]) -> None:
    entries = payload.get("entries")
    if not isinstance(entries, list):
        entries = []
    normalized = {
        "schema_version": "v1",
        "updated_at": str(payload.get("updated_at") or _utc_now_iso()).strip() or _utc_now_iso(),
        "entries": entries[-10000:],
    }
    _dump_json(path, normalized)


def _build_invalid_proposal_fingerprint(
    *,
    proposal: CandidateProposal,
    ir_mode: str,
    patch_payload: Mapping[str, Any] | None,
) -> str:
    changed_payload = {
        key: proposal.genome.get(key)
        for key in proposal.changed_keys
        if key in proposal.genome
    }
    fingerprint_payload = canonicalize_object(
        {
            "ir_mode": str(ir_mode),
            "operator_id": str(proposal.operator_id),
            "changed_keys": list(proposal.changed_keys),
            "changed_values": changed_payload,
            "patch": dict(patch_payload or {}),
        }
    )
    raw = json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"invalid_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:20]}"


def _preflight_validate_materialized_candidate(
    *,
    parent_cfg: Mapping[str, Any],
    proposal: CandidateProposal,
    patch_payload: Mapping[str, Any] | None,
    window_start: str,
    window_end: str,
    include_stress: bool,
    enforce_app_config_validation: bool,
) -> tuple[str | None, str]:
    from coint2.utils.config import AppConfig

    if proposal.patch_ir is not None and not isinstance(proposal.patch_ir, dict):
        return "PATCH_IR_INVALID", "proposal.patch_ir must be a mapping"
    if proposal.patch_ir is not None:
        patch_ir_errors = validate_patch_ir_payload(proposal.patch_ir)
        if patch_ir_errors:
            return "PATCH_IR_INVALID", f"proposal.patch_ir failed validate_patch_ir_payload: {patch_ir_errors}"

    holdout_cfg_payload = copy.deepcopy(dict(parent_cfg))
    try:
        if patch_payload is not None:
            _apply_materialized_patch(holdout_cfg_payload, patch_payload)
        else:
            changed = {key: proposal.genome.get(key) for key in proposal.changed_keys if key in proposal.genome}
            _apply_genome_overrides(holdout_cfg_payload, changed)
        _set_nested(holdout_cfg_payload, "walk_forward.start_date", window_start)
        _set_nested(holdout_cfg_payload, "walk_forward.end_date", window_end)
    except Exception as exc:  # noqa: BLE001
        return "CONFIG_VALIDATION_ERROR", f"materialization failed: {type(exc).__name__}: {exc}"

    backtest_block = holdout_cfg_payload.get("backtest") or {}
    try:
        max_var_multiplier = float(backtest_block.get("max_var_multiplier"))
    except Exception:  # noqa: BLE001
        max_var_multiplier = None
    if max_var_multiplier is not None and max_var_multiplier <= 1.0:
        return "MAX_VAR_MULTIPLIER_INVALID", f"backtest.max_var_multiplier={max_var_multiplier:.6f} <= 1.0"

    if enforce_app_config_validation:
        try:
            AppConfig(**holdout_cfg_payload)
        except Exception as exc:  # noqa: BLE001
            return "CONFIG_VALIDATION_ERROR", f"holdout AppConfig validation failed: {type(exc).__name__}: {exc}"

        if include_stress:
            stress_cfg_payload = copy.deepcopy(holdout_cfg_payload)
            for dotted_key, value in STRESS_OVERRIDES.items():
                _set_nested(stress_cfg_payload, dotted_key, value)
            try:
                AppConfig(**stress_cfg_payload)
            except Exception as exc:  # noqa: BLE001
                return "CONFIG_VALIDATION_ERROR", f"stress AppConfig validation failed: {type(exc).__name__}: {exc}"

    return None, ""


def _baseline_parent_supports_app_config_validation(
    *,
    parent_cfg: Mapping[str, Any],
    window_start: str,
    window_end: str,
    include_stress: bool,
) -> bool:
    from coint2.utils.config import AppConfig

    payload = copy.deepcopy(dict(parent_cfg))
    _set_nested(payload, "walk_forward.start_date", window_start)
    _set_nested(payload, "walk_forward.end_date", window_end)
    try:
        AppConfig(**payload)
    except Exception:  # noqa: BLE001
        return False
    if include_stress:
        stress_payload = copy.deepcopy(payload)
        for dotted_key, value in STRESS_OVERRIDES.items():
            _set_nested(stress_payload, dotted_key, value)
        try:
            AppConfig(**stress_payload)
        except Exception:  # noqa: BLE001
            return False
    return True


def filter_invalid_proposals_before_materialization(
    *,
    proposals: Sequence[CandidateProposal],
    app_root: Path,
    invalid_index_path: Path,
    parent_cfg: Mapping[str, Any],
    windows: Sequence[tuple[str, str]],
    include_stress: bool,
    ir_mode: str,
    persist_state: bool,
) -> tuple[list[CandidateProposal], dict[str, Any], dict[str, Any]]:
    state = _load_invalid_proposal_index(invalid_index_path)
    accepted: list[CandidateProposal] = []
    summary: dict[str, Any] = {
        "checked": 0,
        "accepted": 0,
        "skipped_invalid": 0,
        "skipped_quarantined": 0,
        "codes": {},
        "entries": [],
        "state_path": _relative_to_root(invalid_index_path, root=app_root),
    }
    code_counter: dict[str, int] = {}
    if not proposals:
        return accepted, summary, state
    if not windows:
        raise ValueError("windows must not be empty")

    first_window_start, first_window_end = windows[0]
    enforce_app_config_validation = _baseline_parent_supports_app_config_validation(
        parent_cfg=parent_cfg,
        window_start=first_window_start,
        window_end=first_window_end,
        include_stress=bool(include_stress),
    )
    state_dirty = False
    quarantine_codes = {"MAX_VAR_MULTIPLIER_INVALID", "CONFIG_VALIDATION_ERROR", "PATCH_IR_INVALID"}
    for proposal in proposals:
        summary["checked"] += 1
        patch_payload: dict[str, Any] | None = None
        if str(ir_mode) == "patch_ast" and isinstance(proposal.patch_ir, dict):
            raw_patch = proposal.patch_ir.get("materialized_patch")
            if isinstance(raw_patch, dict):
                patch_payload = raw_patch

        fingerprint = _build_invalid_proposal_fingerprint(
            proposal=proposal,
            ir_mode=str(ir_mode),
            patch_payload=patch_payload,
        )
        existing = _find_invalid_proposal_index_entry(state, fingerprint)
        existing_code = str(existing.get("code") or "").strip().upper() if isinstance(existing, Mapping) else ""
        if existing_code in quarantine_codes and _coerce_int(existing.get("occurrences"), default=1) >= 1:
            entry = _upsert_invalid_proposal_index_entry(
                state=state,
                fingerprint=fingerprint,
                code=existing_code,
                reason=str(existing.get("reason") or "quarantined repeat invalid proposal").strip(),
                operator_id=proposal.operator_id,
                candidate_id=proposal.candidate_id,
                changed_keys=proposal.changed_keys,
            )
            state_dirty = True
            code_counter[existing_code] = code_counter.get(existing_code, 0) + 1
            summary["skipped_quarantined"] += 1
            summary["entries"].append(
                {
                    "candidate_id": proposal.candidate_id,
                    "fingerprint": fingerprint,
                    "action": "quarantined",
                    "code": existing_code,
                    "reason": str(entry.get("reason") or "").strip(),
                }
            )
            continue

        code, reason = _preflight_validate_materialized_candidate(
            parent_cfg=parent_cfg,
            proposal=proposal,
            patch_payload=patch_payload,
            window_start=first_window_start,
            window_end=first_window_end,
            include_stress=bool(include_stress),
            enforce_app_config_validation=bool(enforce_app_config_validation),
        )
        if code:
            _upsert_invalid_proposal_index_entry(
                state=state,
                fingerprint=fingerprint,
                code=code,
                reason=reason,
                operator_id=proposal.operator_id,
                candidate_id=proposal.candidate_id,
                changed_keys=proposal.changed_keys,
            )
            state_dirty = True
            code_counter[code] = code_counter.get(code, 0) + 1
            summary["skipped_invalid"] += 1
            summary["entries"].append(
                {
                    "candidate_id": proposal.candidate_id,
                    "fingerprint": fingerprint,
                    "action": "invalid",
                    "code": code,
                    "reason": reason,
                }
            )
            continue

        accepted.append(proposal)

    summary["accepted"] = len(accepted)
    summary["codes"] = code_counter
    summary["enforce_app_config_validation"] = bool(enforce_app_config_validation)
    if persist_state and state_dirty:
        _dump_invalid_proposal_index(invalid_index_path, state)
    return accepted, summary, state


def _render_knob_space_payload(knob_space: Sequence[KnobSpec]) -> list[dict[str, Any]]:
    return [
        {
            "key": spec.key,
            "type": spec.type,
            "min": spec.min,
            "max": spec.max,
            "step": spec.step,
            "choices": spec.choices,
            "round_to": spec.round_to,
            "weight": spec.weight,
            "norm": spec.norm,
        }
        for spec in knob_space
    ]


def _render_search_space_md(
    *,
    run_group: str,
    controller_group: str,
    base_config_rel: str,
    parent_config_rel: str,
    windows: Sequence[tuple[str, str]],
    diagnostics: VariantDiagnostics | None,
    failure: FailureAssessment,
    operator_plan: TargetedOperatorPlan,
    thresholds: FailureThresholds,
    proposals: Sequence[CandidateProposal],
    dedupe_distance: float,
    max_changed_keys: int,
    llm_policy: LlmPolicyPlan,
    policy_scale: str,
    repair_mode: str,
    repair_max_neighbors: int,
    exclude_knobs: Sequence[str],
    repair_summary: Mapping[str, int],
    parent_resolution: ParentResolution,
) -> str:
    lines = [
        f"# Evolution search space: `{run_group}`",
        "",
        f"- controller_group: `{controller_group}`",
        f"- base_config: `{base_config_rel}`",
        f"- parent_config: `{parent_config_rel}`",
        f"- winner_proximate_requested: `{parent_resolution.winner_proximate_requested}`",
        f"- winner_proximate_resolved: `{parent_resolution.winner_proximate_resolved}`",
        f"- preferred_parent_source: `{parent_resolution.preferred_parent_source}`",
        f"- winner_proximate_tokens: `{','.join(parent_resolution.winner_proximate_tokens) or '-'}`",
        f"- winner_proximate_fallback_reason: `{parent_resolution.winner_proximate_fallback_reason or '-'}`",
        f"- operator: `{operator_plan.operator_kind}`",
        f"- budget: `{operator_plan.budget}`",
        f"- failure_mode: `{failure.failure_mode}`",
        f"- triggers: `{'; '.join(failure.triggers)}`",
        f"- llm_policy_used: `{llm_policy.used}`",
        f"- llm_model: `{llm_policy.model}`",
        f"- llm_effort: `{llm_policy.effort}`",
        f"- llm_source: `{llm_policy.source}`",
        f"- llm_reason: `{llm_policy.reason}`",
        f"- policy_scale: `{policy_scale}`",
        f"- repair_mode: `{repair_mode}`",
        f"- repair_max_neighbors: `{int(repair_max_neighbors)}`",
        f"- repair_exclude_knobs: `{','.join(str(key) for key in exclude_knobs) or '-'}`",
        f"- repair_risk_matched: `{int(repair_summary.get('risk_matched', 0))}`",
        f"- repair_replaced: `{int(repair_summary.get('replaced', 0))}`",
        f"- repair_neighbors_added: `{int(repair_summary.get('neighbors_added', 0))}`",
        f"- repair_fallback_kept: `{int(repair_summary.get('fallback_kept', 0))}`",
        f"- dedupe_distance: `{dedupe_distance}`",
        f"- max_changed_keys: `{max_changed_keys}`",
        f"- min_windows: `{thresholds.min_windows}`",
        f"- min_trades: `{thresholds.min_trades}`",
        f"- min_pairs: `{thresholds.min_pairs}`",
        f"- max_dd_pct: `{thresholds.max_dd_pct}`",
        "",
        "## Windows",
    ]
    for start, end in windows:
        lines.append(f"- `{start}` -> `{end}`")
    lines.append("")
    lines.append("## Diagnostics (selected parent)")
    if diagnostics is None:
        lines.append("- no matching completed holdout rows found in run_index.")
    else:
        lines.extend(
            [
                f"- run_group: `{diagnostics.run_group}`",
                f"- variant_id: `{diagnostics.variant_id}`",
                f"- sample_config_path: `{diagnostics.sample_config_path}`",
                f"- windows: `{diagnostics.windows}`",
                f"- worst_robust_sharpe: `{_fmt_optional(diagnostics.worst_robust_sharpe)}`",
                f"- worst_dd_pct: `{_fmt_optional(diagnostics.worst_dd_pct)}`",
                f"- worst_trades: `{_fmt_optional(diagnostics.worst_trades, digits=2)}`",
                f"- worst_pairs: `{_fmt_optional(diagnostics.worst_pairs, digits=2)}`",
                f"- worst_zero_pair_steps_pct: `{_fmt_optional(diagnostics.worst_zero_pair_steps_pct)}`",
                f"- worst_tail_pair_share: `{_fmt_optional(diagnostics.worst_tail_pair_share)}`",
                f"- worst_tail_period_share: `{_fmt_optional(diagnostics.worst_tail_period_share)}`",
            ]
        )
    lines.extend(["", "## Proposals"])
    for candidate in proposals:
        lines.append(
            f"- `{candidate.candidate_id}` | changed={','.join(candidate.changed_keys)} "
            f"| nearest={candidate.nearest_id or '-'} ({_fmt_optional(candidate.nearest_distance)})"
        )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    include_stress_default = _env_bool("EVOLVE_NEXT_BATCH_INCLUDE_STRESS", True)
    parser = argparse.ArgumentParser(description="Targeted mutation planner for next evolution batch.")
    parser.add_argument("--base-config", required=True, help="Seed/base YAML config (relative to coint4/ unless absolute).")
    parser.add_argument("--run-group", required=True, help="Output run group for generated queue/configs.")
    parser.add_argument("--controller-group", required=True, help="Controller group for evolution state/decisions.")
    parser.add_argument(
        "--run-index",
        default="artifacts/wfa/aggregate/rollup/run_index.csv",
        help="Rollup run_index CSV (relative to coint4/).",
    )
    parser.add_argument(
        "--contains",
        action="append",
        default=[],
        help="run_index filter token (repeatable). Defaults to --controller-group when omitted.",
    )
    parser.add_argument(
        "--winner-proximate-token",
        action="append",
        default=[],
        help="Preferred lineage/run_group token for explicit winner-proximate parent resolution (OR semantics, repeatable).",
    )
    parser.add_argument("--window", action="append", default=[], help="OOS window format: YYYY-MM-DD,YYYY-MM-DD (repeatable).")
    parser.add_argument(
        "--include-stress",
        dest="include_stress",
        action="store_true",
        help="Generate paired stress configs for each holdout variant/window.",
    )
    parser.add_argument(
        "--no-include-stress",
        dest="include_stress",
        action="store_false",
        help="Disable stress pair generation (debug only; breaks strict fullspan pairing).",
    )
    parser.set_defaults(include_stress=include_stress_default)
    parser.add_argument("--num-variants", type=int, default=12, help="How many candidate variants to generate.")
    parser.add_argument("--seed", type=int, default=20260226, help="Deterministic RNG seed.")
    parser.add_argument(
        "--ir-mode",
        choices=["knob", "patch_ast"],
        default="knob",
        help="Candidate IR mode: knob (genome mutation) or patch_ast (ConfigPatch AST + gates).",
    )
    parser.add_argument("--knob-space", help="Optional JSON/YAML list of KnobSpec-like dicts.")
    parser.add_argument("--dedupe-distance", type=float, default=0.06, help="Reject candidates closer than this weighted L1 distance.")
    parser.add_argument("--max-changed-keys", type=int, default=3, help="Complexity budget per candidate.")
    parser.add_argument(
        "--repair-mode",
        choices=["off", "validation_neighbor"],
        default="off",
        help="Post-generation repair mode for high-risk validation-error candidates.",
    )
    parser.add_argument(
        "--repair-max-neighbors",
        type=int,
        default=2,
        help="Max validation-neighbor replacements generated per risky candidate.",
    )
    parser.add_argument(
        "--exclude-knob",
        action="append",
        default=[],
        help="Knob key excluded from validation-neighbor repair mutations (repeatable).",
    )
    parser.add_argument("--configs-dir", default="configs/evolution", help="Base directory for generated configs.")
    parser.add_argument("--queue-dir", default="artifacts/wfa/aggregate", help="Base directory for run_queue/search_space.")
    parser.add_argument("--runs-dir", default="artifacts/wfa/runs", help="Base runs directory for queue results_dir paths.")
    parser.add_argument("--state-path", help="Override evolution state path.")
    parser.add_argument("--decision-dir", help="Override decision output directory.")
    parser.add_argument(
        "--invalid-proposal-state-path",
        help="Override persistent invalid proposal index path (default: artifacts/wfa/aggregate/.autonomous/invalid_proposal_index.json).",
    )
    parser.add_argument("--include-noncompleted", action="store_true", help="Allow non-completed rows in diagnostics.")
    parser.add_argument("--min-windows", type=int, default=3)
    parser.add_argument("--min-trades", type=float, default=200.0)
    parser.add_argument("--min-pairs", type=float, default=20.0)
    parser.add_argument("--max-dd-pct", type=float, default=0.14)
    parser.add_argument("--max-zero-pair-steps-pct", type=float, default=0.2)
    parser.add_argument("--max-tail-pair-share", type=float, default=0.45)
    parser.add_argument("--max-tail-period-share", type=float, default=0.6)
    parser.add_argument(
        "--policy-scale",
        choices=["auto", "micro", "macro"],
        default="auto",
        help="Operator policy mode: micro (safe local), macro (broad/crossover), auto.",
    )
    parser.add_argument("--llm-propose", action="store_true", help="Enable LLM policy override via codex exec.")
    parser.add_argument("--llm-model", default="gpt-5.2", help="Model for codex exec when --llm-propose is set.")
    parser.add_argument("--llm-effort", default="xhigh", help="Reasoning effort hint passed into LLM context.")
    parser.add_argument("--llm-timeout-sec", type=int, default=180, help="Timeout for codex exec in seconds.")
    parser.add_argument("--llm-codex-bin", default="codex", help="codex executable for LLM policy call.")
    parser.add_argument("--llm-max-budget", type=int, default=96, help="Upper budget bound allowed from LLM policy.")
    parser.add_argument("--llm-max-keys", type=int, default=6, help="Max key count allowed from LLM policy.")
    parser.add_argument("--llm-verify-semantic", action="store_true", help="Enable LLM semantic consistency verifier (patch_ast mode).")
    parser.add_argument("--ast-max-complexity-score", type=float, default=60.0, help="Max complexity score for ConfigPatch AST candidates.")
    parser.add_argument(
        "--ast-max-redundancy-similarity",
        type=float,
        default=0.85,
        help="Max redundancy similarity vs zoo/batch for ConfigPatch AST candidates.",
    )
    parser.add_argument("--patch-max-attempts", type=int, default=8, help="Max attempts per variant in patch_ast mode.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files; print summary only.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if int(args.num_variants) < 1:
        raise SystemExit("--num-variants must be >= 1")
    if int(args.max_changed_keys) < 1:
        raise SystemExit("--max-changed-keys must be >= 1")
    if float(args.dedupe_distance) < 0.0:
        raise SystemExit("--dedupe-distance must be >= 0")
    if int(args.llm_timeout_sec) < 10:
        raise SystemExit("--llm-timeout-sec must be >= 10")
    if int(args.llm_max_budget) < 1:
        raise SystemExit("--llm-max-budget must be >= 1")
    if int(args.llm_max_keys) < 1:
        raise SystemExit("--llm-max-keys must be >= 1")
    if float(args.ast_max_complexity_score) <= 0.0:
        raise SystemExit("--ast-max-complexity-score must be > 0")
    if not (0.0 <= float(args.ast_max_redundancy_similarity) <= 1.0):
        raise SystemExit("--ast-max-redundancy-similarity must be in [0,1]")
    if int(args.patch_max_attempts) < 1:
        raise SystemExit("--patch-max-attempts must be >= 1")
    if int(args.repair_max_neighbors) < 1:
        raise SystemExit("--repair-max-neighbors must be >= 1")

    app_root = _resolve_app_root()
    repo_root = app_root.parent
    run_index_path = _resolve_under_root(str(args.run_index), root=app_root)
    base_config_path = _resolve_under_root(str(args.base_config), root=app_root)
    if not base_config_path.exists():
        raise SystemExit(f"base config not found: {base_config_path}")

    try:
        base_cfg = load_effective_yaml_config(base_config_path)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"invalid base config: {exc}") from exc

    windows = _parse_windows(raw_values=args.window, base_cfg=base_cfg)
    knob_space = _load_knob_space(args.knob_space)
    exclude_knobs = {str(key).strip() for key in list(args.exclude_knob or []) if str(key).strip()}
    known_knobs = {spec.key for spec in knob_space}
    unknown_exclude = sorted(exclude_knobs - known_knobs)
    if unknown_exclude:
        raise SystemExit(f"--exclude-knob contains unknown keys: {unknown_exclude}")
    contains = [str(token).strip() for token in list(args.contains or []) if str(token).strip()]
    if not contains:
        contains = [str(args.controller_group).strip()]
    winner_proximate_tokens = [
        str(token).strip() for token in list(args.winner_proximate_token or []) if str(token).strip()
    ]

    run_group = str(args.run_group or "").strip()
    controller_group = str(args.controller_group or "").strip()
    if not run_group:
        raise SystemExit("empty --run-group")
    if not controller_group:
        raise SystemExit("empty --controller-group")

    rows = _read_csv_rows(run_index_path)
    parent_resolution = _resolve_parent_diagnostics(
        rows=rows,
        contains=contains,
        winner_proximate_tokens=winner_proximate_tokens,
        include_noncompleted=bool(args.include_noncompleted),
    )
    diagnostics = list(parent_resolution.diagnostics)
    top_diag = parent_resolution.top_diag

    thresholds = FailureThresholds(
        min_windows=int(args.min_windows),
        min_trades=float(args.min_trades),
        min_pairs=float(args.min_pairs),
        max_dd_pct=float(args.max_dd_pct),
        max_zero_pair_steps_pct=float(args.max_zero_pair_steps_pct),
        max_tail_pair_share=float(args.max_tail_pair_share),
        max_tail_period_share=float(args.max_tail_period_share),
    )
    failure = infer_failure_mode(top_diag, thresholds=thresholds)
    base_operator_plan = select_operator_plan(failure.failure_mode, variants_count=int(args.num_variants))
    llm_policy = _resolve_llm_policy_plan(
        enabled=bool(args.llm_propose),
        model=str(args.llm_model),
        effort=str(args.llm_effort),
        timeout_sec=int(args.llm_timeout_sec),
        codex_bin=str(args.llm_codex_bin),
        repo_root=repo_root,
        diagnostics=top_diag,
        failure=failure,
        base_plan=base_operator_plan,
        thresholds=thresholds,
        windows=windows,
        knob_space=knob_space,
        max_keys=int(args.llm_max_keys),
        max_budget=int(args.llm_max_budget),
        max_changed_keys=int(args.max_changed_keys),
    )
    operator_plan = llm_policy.operator_plan

    default_state_path = app_root / "artifacts" / "wfa" / "aggregate" / str(args.controller_group) / "evolution_state.json"
    state_path = _resolve_under_root(str(args.state_path), root=app_root) if args.state_path else default_state_path
    state_payload = _load_state(state_path)
    generation = int(state_payload.get("generation") or -1) + 1
    seed = int(state_payload.get("seed") or int(args.seed))

    parent_config_path = base_config_path
    if top_diag and top_diag.sample_config_path:
        sample = _resolve_under_root(top_diag.sample_config_path, root=app_root)
        if sample.exists():
            parent_config_path = sample
    parent_cfg = load_effective_yaml_config(parent_config_path)
    parent_genome = genome_from_config(parent_cfg, knob_space=knob_space)
    parent_id = f"parent::{_relative_to_root(parent_config_path, root=app_root)}"
    parent_cfg_ref = _relative_to_root(parent_config_path, root=app_root)

    seen_pool = _collect_seen_genomes(
        rows=rows,
        contains=contains,
        app_root=app_root,
        knob_space=knob_space,
        max_items=2000,
    )
    operator_plan = _apply_policy_scale(
        operator_plan=operator_plan,
        policy_scale=str(args.policy_scale),
        seen_pool_count=len(seen_pool),
    )

    decision_dir = (
        _resolve_under_root(str(args.decision_dir), root=app_root)
        if args.decision_dir
        else (app_root / "artifacts" / "wfa" / "aggregate" / controller_group / "decisions")
    )
    if args.invalid_proposal_state_path:
        invalid_proposal_state_path = _resolve_under_root(str(args.invalid_proposal_state_path), root=app_root)
    else:
        invalid_proposal_state_path = None
    rng_state_seed = int(seed + generation)
    rng = np.random.default_rng(rng_state_seed)
    if str(args.ir_mode) == "patch_ast":
        def _patch_ast_generate(llm_verify_semantic: bool) -> list[CandidateProposal]:
            try:
                return _generate_patch_ast_proposals(
                    num_variants=int(args.num_variants),
                    parent_cfg=parent_cfg,
                    parent_cfg_ref=parent_cfg_ref,
                    parent_genome=parent_genome,
                    parent_id=parent_id,
                    knob_space=knob_space,
                    operator_plan=operator_plan,
                    failure=failure,
                    thresholds=thresholds,
                    diagnostics=diagnostics,
                    generation=generation,
                    rng=rng,
                    seen_pool=seen_pool,
                    dedupe_distance=float(args.dedupe_distance),
                    max_changed_keys=int(args.max_changed_keys),
                    use_llm=bool(args.llm_propose),
                    llm_model=str(args.llm_model),
                    llm_effort=str(args.llm_effort),
                    llm_timeout_sec=int(args.llm_timeout_sec),
                    llm_codex_bin=str(args.llm_codex_bin),
                    repo_root=repo_root,
                    decisions_dir=decision_dir,
                    gate_limits=PatchGateLimits(
                        max_complexity_score=float(args.ast_max_complexity_score),
                        max_redundancy_similarity=float(args.ast_max_redundancy_similarity),
                    ),
                    llm_verify_semantic=bool(llm_verify_semantic),
                    max_attempts_per_variant=int(args.patch_max_attempts),
                )
            except SystemExit as exc:
                print(
                    f"patch_ast generation failed (llm_verify_semantic={bool(llm_verify_semantic)}): {exc}",
                    file=sys.stderr,
                )
                return []
            except Exception as exc:  # noqa: BLE001
                print(
                    "patch_ast generation exception "
                    f"(llm_verify_semantic={bool(llm_verify_semantic)}): {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                return []

        target_variants = int(args.num_variants)
        proposals = _patch_ast_generate(bool(args.llm_verify_semantic))
        if not proposals and bool(args.llm_verify_semantic):
            print(
                "semantic gate fallback: no proposals with llm_verify_semantic=true; "
                "retry with deterministic semantic checks only"
            )
            proposals = _patch_ast_generate(False)
        if not proposals:
            print(
                "patch_ast fallback: no proposals generated; retry with deterministic genome proposals",
                file=sys.stderr,
            )
            proposals = _generate_proposals(
                num_variants=target_variants,
                parent_genome=parent_genome,
                parent_id=parent_id,
                knob_space=knob_space,
                operator_plan=operator_plan,
                failure_assessment=failure,
                generation=generation,
                rng=rng,
                seen_pool=seen_pool,
                dedupe_distance=float(args.dedupe_distance),
                max_changed_keys=int(args.max_changed_keys),
            )
        if len(proposals) < target_variants:
            missing = target_variants - len(proposals)
            print(
                "patch_ast top-up fallback: "
                f"generated {len(proposals)}/{target_variants}; "
                f"filling remaining {missing} with deterministic genome proposals",
                file=sys.stderr,
            )
            proposals.extend(
                _top_up_with_genome_proposals(
                    existing_proposals=proposals,
                    target_num_variants=target_variants,
                    parent_genome=parent_genome,
                    parent_id=parent_id,
                    knob_space=knob_space,
                    operator_plan=operator_plan,
                    failure_assessment=failure,
                    generation=generation,
                    rng=rng,
                    seen_pool=seen_pool,
                    dedupe_distance=float(args.dedupe_distance),
                    max_changed_keys=int(args.max_changed_keys),
                )
            )
        if len(proposals) < target_variants:
            print(
                "warn: incomplete proposal batch after top-up "
                f"({len(proposals)}/{target_variants}); continuing with partial batch",
                file=sys.stderr,
            )
    else:
        proposals = _generate_proposals(
            num_variants=int(args.num_variants),
            parent_genome=parent_genome,
            parent_id=parent_id,
            knob_space=knob_space,
            operator_plan=operator_plan,
            failure_assessment=failure,
            generation=generation,
            rng=rng,
            seen_pool=seen_pool,
            dedupe_distance=float(args.dedupe_distance),
            max_changed_keys=int(args.max_changed_keys),
        )

    repair_summary: dict[str, int] = {
        "risk_matched": 0,
        "replaced": 0,
        "neighbors_added": 0,
        "fallback_kept": 0,
    }
    if str(args.repair_mode) == "validation_neighbor":
        proposals, repair_summary = _apply_validation_neighbor_repair(
            proposals=proposals,
            app_root=app_root,
            repair_max_neighbors=int(args.repair_max_neighbors),
            exclude_knobs=exclude_knobs,
            parent_cfg=parent_cfg,
            parent_genome=parent_genome,
            parent_id=parent_id,
            knob_space=knob_space,
            generation=generation,
            rng=rng,
            seen_pool=seen_pool,
            dedupe_distance=float(args.dedupe_distance),
            max_changed_keys=int(args.max_changed_keys),
        )

    queue_base = _resolve_under_root(str(args.queue_dir), root=app_root) / run_group
    queue_path = queue_base / "run_queue.csv"
    search_space_path = queue_base / "search_space.md"
    decision_id = _decision_id(controller_group=controller_group, run_group=run_group, generation=generation)
    decision_path = decision_dir / f"{decision_id}.json"
    configs_base = _resolve_under_root(str(args.configs_dir), root=app_root) / run_group
    runs_base = _resolve_under_root(str(args.runs_dir), root=app_root) / run_group
    if invalid_proposal_state_path is None:
        outputs_external = any(
            base.resolve() != app_root.resolve() and app_root.resolve() not in base.resolve().parents
            for base in (queue_base.parent, configs_base.parent, runs_base.parent, state_path.parent, decision_dir)
        )
        invalid_proposal_state_path = (
            state_path.parent / "invalid_proposal_index.json"
            if outputs_external
            else _default_invalid_proposal_index_path(app_root)
        )

    proposals, invalid_firewall_summary, invalid_proposal_state = filter_invalid_proposals_before_materialization(
        proposals=proposals,
        app_root=app_root,
        invalid_index_path=invalid_proposal_state_path,
        parent_cfg=parent_cfg,
        windows=windows,
        include_stress=bool(args.include_stress),
        ir_mode=str(args.ir_mode),
        persist_state=not bool(args.dry_run),
    )

    queue_rows: list[dict[str, Any]] = []
    decision_proposals: list[dict[str, Any]] = []
    created_files: list[str] = []

    for idx, proposal in enumerate(proposals, start=1):
        lineage_uid = _derive_lineage_uid(
            controller_group=controller_group,
            run_group=run_group,
            generation=generation,
            variant_index=idx,
            candidate_id=proposal.candidate_id,
            operator_id=proposal.operator_id,
            parents=proposal.parents,
        )
        patch_payload: dict[str, Any] | None = None
        patch_rel: str | None = None
        if str(args.ir_mode) == "patch_ast" and isinstance(proposal.patch_ir, dict):
            raw_patch = proposal.patch_ir.get("materialized_patch")
            if isinstance(raw_patch, dict):
                patch_payload = raw_patch
                patch_rel = f"{_relative_to_root(configs_base, root=app_root)}/{run_group}_v{idx:03d}_{proposal.candidate_id}.patch.yaml"

        for window_start, window_end in windows:
            window_tag = f"{window_start.replace('-', '')}_{window_end.replace('-', '')}"
            variant_tag = f"{run_group}_v{idx:03d}_{proposal.candidate_id}_oos{window_tag}"
            holdout_run_id = f"holdout_{variant_tag}"

            holdout_cfg_payload = copy.deepcopy(parent_cfg)
            if patch_payload is not None:
                _apply_materialized_patch(holdout_cfg_payload, patch_payload)
            else:
                # Apply only changed knobs to avoid injecting unrelated nulls when knob-space expands.
                changed = {key: proposal.genome.get(key) for key in proposal.changed_keys if key in proposal.genome}
                _apply_genome_overrides(holdout_cfg_payload, changed)
            _set_nested(holdout_cfg_payload, "walk_forward.start_date", window_start)
            _set_nested(holdout_cfg_payload, "walk_forward.end_date", window_end)
            holdout_cfg_rel = f"{_relative_to_root(configs_base, root=app_root)}/{variant_tag}.yaml"
            holdout_cfg_path = app_root / holdout_cfg_rel

            if patch_payload is not None and not args.dry_run:
                patch_path = app_root / str(patch_rel)
                if not patch_path.exists():
                    patch_path.parent.mkdir(parents=True, exist_ok=True)
                    patch_path.write_text(yaml.safe_dump(patch_payload, sort_keys=True), encoding="utf-8")
                    created_files.append(_relative_to_root(patch_path, root=app_root))

            if not args.dry_run:
                holdout_cfg_path.parent.mkdir(parents=True, exist_ok=True)
                holdout_cfg_path.write_text(yaml.safe_dump(holdout_cfg_payload, sort_keys=True), encoding="utf-8")
                created_files.append(_relative_to_root(holdout_cfg_path, root=app_root))

            queue_rows.append(
                {
                    "config_path": _relative_to_root(holdout_cfg_path, root=app_root),
                    "results_dir": _relative_to_root(runs_base / holdout_run_id, root=app_root),
                    "status": "planned",
                    "lineage_uid": lineage_uid,
                    "metadata_json": {
                        "lineage_uid": lineage_uid,
                        "variant_index": int(idx),
                        "candidate_id": proposal.candidate_id,
                        "operator_id": proposal.operator_id,
                        "parents": list(proposal.parents),
                        "window_start": window_start,
                        "window_end": window_end,
                        "profile": "holdout",
                        "ir_mode": str(args.ir_mode),
                    },
                }
            )

            if bool(args.include_stress):
                stress_cfg_payload = copy.deepcopy(holdout_cfg_payload)
                for dotted_key, value in STRESS_OVERRIDES.items():
                    _set_nested(stress_cfg_payload, dotted_key, value)

                stress_cfg_rel = f"{_relative_to_root(configs_base, root=app_root)}/{variant_tag}_stress.yaml"
                stress_cfg_path = app_root / stress_cfg_rel
                stress_run_id = f"stress_{variant_tag}"

                if not args.dry_run:
                    stress_cfg_path.parent.mkdir(parents=True, exist_ok=True)
                    stress_cfg_path.write_text(yaml.safe_dump(stress_cfg_payload, sort_keys=True), encoding="utf-8")
                    created_files.append(_relative_to_root(stress_cfg_path, root=app_root))

                queue_rows.append(
                    {
                        "config_path": _relative_to_root(stress_cfg_path, root=app_root),
                        "results_dir": _relative_to_root(runs_base / stress_run_id, root=app_root),
                        "status": "planned",
                        "lineage_uid": lineage_uid,
                        "metadata_json": {
                            "lineage_uid": lineage_uid,
                            "variant_index": int(idx),
                            "candidate_id": proposal.candidate_id,
                            "operator_id": proposal.operator_id,
                            "parents": list(proposal.parents),
                            "window_start": window_start,
                            "window_end": window_end,
                            "profile": "stress",
                            "ir_mode": str(args.ir_mode),
                        },
                    }
                )

        similarity_payload = None
        if proposal.nearest_id and math_is_finite(proposal.nearest_distance):
            similarity_payload = {"nearest_id": proposal.nearest_id, "distance": float(proposal.nearest_distance)}

        decision_proposals.append(
            {
                "candidate_id": proposal.candidate_id,
                "lineage_uid": lineage_uid,
                "operator_id": proposal.operator_id,
                "parents": list(proposal.parents),
                "genome": proposal.genome,
                "materialization": {
                    "config_path": f"{_relative_to_root(configs_base, root=app_root)}/{run_group}_v{idx:03d}_{proposal.candidate_id}_oos{windows[0][0].replace('-', '')}_{windows[0][1].replace('-', '')}.yaml",
                    "results_dir": _relative_to_root(runs_base / f"holdout_{run_group}_v{idx:03d}_{proposal.candidate_id}_oos{windows[0][0].replace('-', '')}_{windows[0][1].replace('-', '')}", root=app_root),
                    "status": "planned",
                    "lineage_uid": lineage_uid,
                },
                "paired_holdout_stress": bool(args.include_stress),
                "ir_mode": str(args.ir_mode),
                "patch_path": patch_rel,
                "patch_ir": proposal.patch_ir,
                "notes": proposal.notes,
                "similarity": similarity_payload,
            }
        )

    if not queue_rows:
        raise SystemExit("no valid queue entries were generated after preflight firewall")

    search_space_md = _render_search_space_md(
        run_group=run_group,
        controller_group=controller_group,
        base_config_rel=_relative_to_root(base_config_path, root=app_root),
        parent_config_rel=_relative_to_root(parent_config_path, root=app_root),
        windows=windows,
        diagnostics=top_diag,
        failure=failure,
        operator_plan=operator_plan,
        thresholds=thresholds,
        proposals=proposals,
        dedupe_distance=float(args.dedupe_distance),
        max_changed_keys=int(args.max_changed_keys),
        llm_policy=llm_policy,
        policy_scale=str(args.policy_scale),
        repair_mode=str(args.repair_mode),
        repair_max_neighbors=int(args.repair_max_neighbors),
        exclude_knobs=sorted(exclude_knobs),
        repair_summary=repair_summary,
        parent_resolution=parent_resolution,
    )

    decision_payload = {
        "schema_version": "v1",
        "decision_id": decision_id,
        "engine_id": "evolve_next_batch",
        "created_at": _utc_now_iso(),
        "generation": generation,
        "seed": seed,
        "rng": {"algorithm": "PCG64", "state": rng.bit_generator.state, "notes": "seed + generation"},
        "controller_group": controller_group,
        "run_group": run_group,
        "ir_mode": str(args.ir_mode),
        "queue_path": _relative_to_root(queue_path, root=app_root),
        "base_config_path": _relative_to_root(base_config_path, root=app_root),
        "knob_space": _render_knob_space_payload(knob_space),
        "patch_ast_gates": None
        if str(args.ir_mode) != "patch_ast"
        else {
            "max_complexity_score": float(args.ast_max_complexity_score),
            "max_redundancy_similarity": float(args.ast_max_redundancy_similarity),
            "llm_verify_semantic": bool(args.llm_verify_semantic),
        },
        "reward": {
            "profile": "strict_fullspan_holdout_stress_v1",
            "hard_gates": {
                "require_metrics": True,
                "min_windows": thresholds.min_windows,
                "min_trades": thresholds.min_trades,
                "min_pairs": thresholds.min_pairs,
                "max_dd_pct": thresholds.max_dd_pct,
                "min_pnl": 0.0,
            },
            "dd_penalty": {"target_pct": thresholds.max_dd_pct, "k": 1.0},
            "fullspan_policy_v1": {
                "paired_holdout_stress": bool(args.include_stress),
                "stress_overrides": STRESS_OVERRIDES if bool(args.include_stress) else {},
            },
        },
        "similarity": {
            "metric": "genome_weighted_l1_v1",
            "dedupe_threshold": float(args.dedupe_distance),
            "normalization": "range",
            "key_weights": None,
        },
        "operators": [
            {
                "operator_id": "op_targeted_primary",
                "kind": operator_plan.operator_kind,
                "budget": int(operator_plan.budget),
                "params": operator_plan.params,
            }
        ],
        "llm_policy": {
            "enabled": llm_policy.enabled,
            "used": llm_policy.used,
            "source": llm_policy.source,
            "reason": llm_policy.reason,
            "model": llm_policy.model,
            "effort": llm_policy.effort,
            "payload": llm_policy.payload,
        },
        "policy_scale": str(args.policy_scale),
        "parent_resolution": {
            "winner_proximate_requested": parent_resolution.winner_proximate_requested,
            "winner_proximate_tokens": list(parent_resolution.winner_proximate_tokens),
            "winner_proximate_resolved": parent_resolution.winner_proximate_resolved,
            "preferred_parent_source": parent_resolution.preferred_parent_source,
            "winner_proximate_fallback_reason": parent_resolution.winner_proximate_fallback_reason,
        },
        "repair": {
            "mode": str(args.repair_mode),
            "max_neighbors": int(args.repair_max_neighbors),
            "exclude_knobs": sorted(exclude_knobs),
            "summary": dict(repair_summary),
        },
        "preflight_firewall": {
            "invalid_proposal_state_path": _relative_to_root(invalid_proposal_state_path, root=app_root),
            "summary": invalid_firewall_summary,
            "tracked_entries": len(list(invalid_proposal_state.get("entries") or [])),
        },
        "lineage": {
            "uid_field": "lineage_uid",
            "metadata_field": "metadata_json",
            "version": "v1",
            "rows": len(queue_rows),
            "unique_uids": len({str(row.get("lineage_uid") or "") for row in queue_rows if str(row.get("lineage_uid") or "")}),
        },
        "proposals": decision_proposals,
        "outputs": {
            "state_path": _relative_to_root(state_path, root=app_root),
            "decision_path": _relative_to_root(decision_path, root=app_root),
            "created_files": sorted(created_files),
        },
        "notes_md": (
            f"failure_mode={failure.failure_mode}; triggers={'; '.join(failure.triggers)}; "
            f"{operator_plan.hypothesis}; llm_used={llm_policy.used}; llm_source={llm_policy.source}"
        ),
    }

    state_history = list(state_payload.get("history") or [])
    state_history.append(
        {
            "generation": generation,
            "decision_id": decision_id,
            "run_group": run_group,
            "ir_mode": str(args.ir_mode),
            "failure_mode": failure.failure_mode,
            "triggers": list(failure.triggers),
            "variants": len(proposals),
            "llm_used": llm_policy.used,
            "llm_source": llm_policy.source,
            "queue_path": _relative_to_root(queue_path, root=app_root),
            "parent_resolution": {
                "winner_proximate_requested": parent_resolution.winner_proximate_requested,
                "winner_proximate_resolved": parent_resolution.winner_proximate_resolved,
                "preferred_parent_source": parent_resolution.preferred_parent_source,
                "winner_proximate_fallback_reason": parent_resolution.winner_proximate_fallback_reason,
            },
            "repair_mode": str(args.repair_mode),
            "repair_summary": dict(repair_summary),
            "invalid_firewall_summary": dict(invalid_firewall_summary),
            "created_at": _utc_now_iso(),
        }
    )
    state_payload_out = {
        "engine_id": "evolve_next_batch",
        "updated_at": _utc_now_iso(),
        "generation": generation,
        "seed": seed,
        "base_config_path": _relative_to_root(base_config_path, root=app_root),
        "parent_config_path": _relative_to_root(parent_config_path, root=app_root),
        "last_run_group": run_group,
        "last_failure_mode": failure.failure_mode,
        "last_decision_id": decision_id,
        "history": state_history[-50:],
    }

    if args.dry_run:
        print(f"[dry-run] run_group={run_group}")
        print(f"[dry-run] controller_group={controller_group}")
        print(f"[dry-run] ir_mode={args.ir_mode}")
        print(f"[dry-run] failure_mode={failure.failure_mode}")
        print(f"[dry-run] triggers={' | '.join(failure.triggers)}")
        print(
            "[dry-run] parent_resolution="
            + json.dumps(
                {
                    "winner_proximate_requested": parent_resolution.winner_proximate_requested,
                    "winner_proximate_tokens": list(parent_resolution.winner_proximate_tokens),
                    "winner_proximate_resolved": parent_resolution.winner_proximate_resolved,
                    "preferred_parent_source": parent_resolution.preferred_parent_source,
                    "winner_proximate_fallback_reason": parent_resolution.winner_proximate_fallback_reason,
                },
                ensure_ascii=False,
            )
        )
        print(f"[dry-run] policy_scale={args.policy_scale}")
        print(f"[dry-run] repair_mode={args.repair_mode} max_neighbors={int(args.repair_max_neighbors)} exclude={sorted(exclude_knobs)}")
        print(f"[dry-run] repair_summary={json.dumps(repair_summary, ensure_ascii=False)}")
        print(f"[dry-run] invalid_firewall={json.dumps(invalid_firewall_summary, ensure_ascii=False)}")
        print(f"[dry-run] operator={operator_plan.operator_kind} params={json.dumps(operator_plan.params, ensure_ascii=False)}")
        print(f"[dry-run] llm_used={llm_policy.used} source={llm_policy.source} reason={llm_policy.reason}")
        print(f"[dry-run] generated_variants={len(proposals)} windows={len(windows)} queue_rows={len(queue_rows)}")
        print(f"[dry-run] queue={_relative_to_root(queue_path, root=app_root)}")
        print(f"[dry-run] search_space={_relative_to_root(search_space_path, root=app_root)}")
        print(f"[dry-run] decision={_relative_to_root(decision_path, root=app_root)}")
        return 0

    queue_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    _write_run_queue_rows(queue_path, queue_rows)
    search_space_path.write_text(search_space_md, encoding="utf-8")
    _dump_json(decision_path, decision_payload)
    _dump_json(state_path, state_payload_out)

    print(f"Wrote queue:       {queue_path} ({len(queue_rows)} rows)")
    print(f"Wrote search:      {search_space_path}")
    print(f"Wrote decision:    {decision_path}")
    print(f"Wrote state:       {state_path}")
    print(f"Failure mode:      {failure.failure_mode}")
    print(f"Operator:          {operator_plan.operator_kind}")
    print(f"LLM policy used:   {llm_policy.used} ({llm_policy.source})")
    print(f"Preflight invalid: {json.dumps(invalid_firewall_summary['codes'], ensure_ascii=False)}")
    print(f"Generated variants:{len(proposals)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
