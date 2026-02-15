#!/usr/bin/env python3
"""Generate clean-cycle sweep YAML configs strictly from a single baseline config.

Task: C14 (tasks/clean_cycle_top10/prd_clean_cycle_top10.json)

Notes:
- This generator is baseline-only: it never chains sweeps from existing sweep configs.
- All generated configs are forced to use FIXED_WINDOWS.walk_forward (clean-cycle invariant).
- Sweep grid logic (parse/permutations/tags) is reused from scripts/optimization/generate_configs.py
  to avoid duplicating the sweep semantics.

Example (run from app-root `coint4/`):

  PYTHONPATH=src ./.venv/bin/python scripts/optimization/clean_cycle_top10/make_sweep_configs.py \
    --baseline-config configs/clean_cycle_top10/baseline/b01_*.yaml \
    --sweep 'backtest.min_spread_move_sigma=[0.10,0.15,0.20]' \
    --out-dir configs/clean_cycle_top10/sweeps/min_spread_move_sigma \
    --manifest-out artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/sweeps_manifest_ms.json \
    --dry-run
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from definitions import CLEAN_AGG_DIR, FIXED_WINDOWS


_SAFE_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")
_UNDERSCORES_RE = re.compile(r"_+")
_BASELINE_PREFIX_RE = re.compile(r"^(?P<prefix>b[0-9]{2})_")

_WF_REQUIRED_KEYS = [
    "start_date",
    "end_date",
    "training_period_days",
    "testing_period_days",
    "step_size_days",
    "max_steps",
    "gap_minutes",
    "refit_frequency",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve_project_root() -> Path:
    """Return app-root (directory containing pyproject.toml)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: assume we are under coint4/scripts/**.
    return here.parents[3]


def _normalize_repo_relative_path(raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    if value.startswith("coint4/"):
        value = value[len("coint4/") :]
    while value.startswith("./"):
        value = value[2:]
    return value


def _resolve_under_project(path_str: str, project_root: Path) -> Optional[Path]:
    raw = str(path_str or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    normalized = _normalize_repo_relative_path(raw)
    return project_root / normalized


def _ensure_under_project(path: Path, project_root: Path) -> Path:
    resolved = path.resolve()
    root = project_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"refusing to operate outside project root: {resolved}") from exc
    return resolved


def _try_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _slugify(value: str) -> str:
    raw = str(value or "").strip()
    slug = _SAFE_SLUG_RE.sub("_", raw)
    slug = _UNDERSCORES_RE.sub("_", slug).strip("_")
    if not slug:
        slug = "run"

    # Keep filenames within typical filesystem limits and deterministic.
    if len(slug) > 200:
        digest8 = hashlib.sha256(slug.encode("utf-8")).hexdigest()[:8]
        slug = f"{slug[:191]}_{digest8}"
    return slug


def _load_generate_configs_module() -> Any:
    """Dynamic import of scripts/optimization/generate_configs.py to reuse sweep logic."""
    gen_path = Path(__file__).resolve().parents[1] / "generate_configs.py"
    if not gen_path.exists():
        raise SystemExit(f"generate_configs.py not found: {gen_path}")

    spec = importlib.util.spec_from_file_location("_coint4_generate_configs", gen_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"failed to load module spec: {gen_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _to_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if int(value) != value:
            return None
        return int(value)
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _require_str(value: Any, *, label: str) -> str:
    s = str(value or "").strip()
    if not s:
        raise SystemExit(f"{label} must be a non-empty string")
    return s


def _load_fixed_walk_forward() -> Dict[str, Any]:
    node: Any = FIXED_WINDOWS
    if not isinstance(node, dict):
        raise SystemExit("FIXED_WINDOWS must be a JSON object/dict")
    wf = node.get("walk_forward", node)
    if not isinstance(wf, dict):
        raise SystemExit("FIXED_WINDOWS.walk_forward must be a JSON object/dict")

    missing = [k for k in _WF_REQUIRED_KEYS if k not in wf]
    if missing:
        raise SystemExit("FIXED_WINDOWS.walk_forward missing keys: " + ", ".join(missing))

    out: Dict[str, Any] = {}
    out["start_date"] = _require_str(wf.get("start_date"), label="FIXED_WINDOWS.walk_forward.start_date")
    out["end_date"] = _require_str(wf.get("end_date"), label="FIXED_WINDOWS.walk_forward.end_date")

    for key in ["training_period_days", "testing_period_days", "step_size_days", "max_steps", "gap_minutes"]:
        value = _to_int(wf.get(key))
        if value is None:
            raise SystemExit(f"FIXED_WINDOWS.walk_forward.{key} must be an int")
        out[key] = value

    refit = wf.get("refit_frequency")
    out["refit_frequency"] = None if refit is None else str(refit).strip()

    max_steps = int(out.get("max_steps") or 0)
    if max_steps <= 0:
        raise SystemExit("FIXED_WINDOWS.walk_forward.max_steps must be a positive int")
    if max_steps > 5:
        raise SystemExit("unsafe FIXED_WINDOWS.walk_forward.max_steps (must be <= 5 for queue-compatible runs)")

    return out


def _patch_walk_forward(config: Dict[str, Any], fixed_wf: Dict[str, Any], *, cfg_label: str) -> None:
    wf = config.get("walk_forward")
    if not isinstance(wf, dict):
        raise SystemExit(f"config missing walk_forward mapping: {cfg_label}")
    for key, value in fixed_wf.items():
        wf[key] = value


def _parse_zip_keys(raw_values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for raw in raw_values:
        for part in str(raw).split(","):
            key = part.strip()
            if key:
                out.append(key)
    return out


def _build_permutations(
    *,
    gen: Any,
    sweeps: List[Tuple[str, List[Any]]],
    zip_mode: bool,
    zip_keys: List[str],
) -> List[List[Tuple[str, Any]]]:
    if zip_mode and zip_keys:
        raise SystemExit("--zip and --zip-keys are mutually exclusive")

    if zip_keys:
        sweep_keys = {k for k, _ in sweeps}
        missing = sorted(set(zip_keys) - sweep_keys)
        if missing:
            raise SystemExit(f"--zip-keys contains keys not present in --sweep: {missing}")

        zip_key_set = set(zip_keys)
        zipped = [s for s in sweeps if s[0] in zip_key_set]
        others = [s for s in sweeps if s[0] not in zip_key_set]
        zipped_perms = gen.generate_permutations(zipped, zip_mode=True)
        other_perms = gen.generate_permutations(others, zip_mode=False)
        return [z + o for z in zipped_perms for o in other_perms]

    return gen.generate_permutations(sweeps, zip_mode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate clean-cycle sweep YAML configs from a single baseline config.")
    parser.add_argument(
        "--baseline-config",
        required=True,
        help="Baseline YAML config path (relative to app-root `coint4/` unless absolute).",
    )
    parser.add_argument(
        "--sweep",
        "--sweep-spec",
        dest="sweep_specs",
        action="append",
        default=[],
        help="Sweep spec: 'key=[val1,val2,...]' (repeatable). Values are parsed as JSON.",
    )
    parser.add_argument("--zip", action="store_true", dest="zip_mode", help="Force zipped (not cartesian) iteration.")
    parser.add_argument(
        "--zip-keys",
        action="append",
        default=[],
        help="Comma-separated sweep keys to zip together (partial zip). Example: --zip-keys a,b",
    )
    parser.add_argument(
        "--out-dir",
        default="configs/clean_cycle_top10/sweeps",
        help="Directory for generated YAML configs (relative to app-root `coint4/` unless absolute).",
    )
    parser.add_argument(
        "--manifest-out",
        default=f"{CLEAN_AGG_DIR}/sweeps_manifest.json",
        help="Where to write sweeps_manifest.json (relative to app-root `coint4/` unless absolute).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan without writing files.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output YAML files and manifest.",
    )
    args = parser.parse_args()

    project_root = _resolve_project_root()
    gen = _load_generate_configs_module()
    fixed_wf = _load_fixed_walk_forward()

    baseline_path = _resolve_under_project(args.baseline_config, project_root)
    if baseline_path is None:
        raise SystemExit("--baseline-config is empty")
    baseline_path = _ensure_under_project(baseline_path, project_root)
    if not baseline_path.exists():
        raise SystemExit(f"baseline config not found: {_try_relpath(baseline_path, project_root)}")

    out_dir = _resolve_under_project(args.out_dir, project_root) or (project_root / args.out_dir)
    out_dir = _ensure_under_project(out_dir, project_root)

    manifest_out = _resolve_under_project(args.manifest_out, project_root) or (project_root / args.manifest_out)
    manifest_out = _ensure_under_project(manifest_out, project_root)

    if manifest_out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing manifest (use --overwrite): {_try_relpath(manifest_out, project_root)}")

    if not args.sweep_specs:
        raise SystemExit("at least one --sweep is required (this tool is baseline-only, not a baseline copier)")

    sweeps = [gen.parse_sweep(s) for s in args.sweep_specs]
    sweep_keys = [k for k, _ in sweeps]
    forbidden = sorted([k for k in sweep_keys if str(k).startswith("walk_forward.")])
    if forbidden:
        raise SystemExit(
            "refusing sweep over walk_forward.* keys (clean-cycle invariant: FIXED_WINDOWS.walk_forward is fixed): "
            + ", ".join(forbidden)
        )

    zip_keys = _parse_zip_keys(args.zip_keys)
    permutations = _build_permutations(
        gen=gen,
        sweeps=sweeps,
        zip_mode=bool(args.zip_mode),
        zip_keys=zip_keys,
    )

    base_cfg_raw = yaml.safe_load(baseline_path.read_text(encoding="utf-8")) or {}
    if not isinstance(base_cfg_raw, dict):
        raise SystemExit(f"baseline config must parse into a mapping/object: {_try_relpath(baseline_path, project_root)}")

    baseline_sha256 = _sha256_file(baseline_path)
    baseline_stem = baseline_path.stem
    m = _BASELINE_PREFIX_RE.match(baseline_stem)
    baseline_prefix = m.group("prefix") if m else baseline_stem

    if not permutations:
        raise SystemExit("no permutations to generate (unexpected)")

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: List[Dict[str, Any]] = []
    planned_paths: List[Path] = []
    for idx, combo in enumerate(permutations, 1):
        cfg = copy.deepcopy(base_cfg_raw)
        tags: List[str] = []
        params_list: List[Dict[str, Any]] = []
        for key, value in combo:
            gen.set_nested(cfg, key, value)
            tags.append(gen.make_tag(key, value))
            params_list.append({"key": key, "value": value})

        # Enforce clean-cycle invariant.
        _patch_walk_forward(cfg, fixed_wf, cfg_label=_try_relpath(baseline_path, project_root))

        sweep_id = f"s{idx:03d}"
        slug = _slugify(f"{baseline_prefix}__" + "_".join(tags))
        out_path = out_dir / f"{sweep_id}_{slug}.yaml"

        if out_path.exists() and not args.overwrite:
            raise SystemExit(f"refusing to overwrite existing output (use --overwrite): {_try_relpath(out_path, project_root)}")

        if args.dry_run:
            params_str = ", ".join(f"{p['key']}={p['value']}" for p in params_list)
            print(f"DRY: {out_path.name} <= {baseline_path.name} + {params_str}")
        else:
            dumped = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, default_flow_style=False)
            if not dumped.endswith("\n"):
                dumped += "\n"
            out_path.write_text(dumped, encoding="utf-8")

        planned_paths.append(out_path)

        manifest_entries.append(
            {
                "run_group": "clean_cycle_top10",
                "run_id": out_path.stem,
                "run_name": sweep_id,
                "status": "planned",
                "baseline_config_path": _try_relpath(baseline_path, project_root),
                "baseline_config_sha256": baseline_sha256,
                "config_path": _try_relpath(out_path, project_root),
                "config_sha256": None if args.dry_run else _sha256_text(out_path.read_text(encoding="utf-8")),
                "sweep_params": params_list,
                "sweep_tags": tags,
                "zip_mode": bool(args.zip_mode),
                "zip_keys": zip_keys,
            }
        )

    if args.dry_run:
        print(f"DRY: would write manifest: {_try_relpath(manifest_out, project_root)} ({len(manifest_entries)} entries)")
        return 0

    # Write manifest deterministically (no timestamps).
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_out.with_suffix(manifest_out.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest_entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(manifest_out)

    print(f"Wrote {len(planned_paths)} sweep config(s) to: {_try_relpath(out_dir, project_root)}")
    print(f"Wrote manifest: {_try_relpath(manifest_out, project_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
