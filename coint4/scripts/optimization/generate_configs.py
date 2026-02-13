#!/usr/bin/env python3
"""Generate WFA config permutations from a base YAML + sweep specification.

Usage examples:

  # OOS window sweep
  python scripts/optimization/generate_configs.py \
    --base configs/sprint34_tp15_tr90_holdout.yaml \
    --tag oos_validation \
    --sweep 'walk_forward.start_date=["2022-06-01","2023-10-01","2024-05-01"]' \
    --sweep 'walk_forward.end_date=["2023-04-30","2024-09-30","2025-06-30"]' \
    --output-dir configs/oos_validation/ \
    --queue-dir artifacts/wfa/aggregate/oos_validation/

  # Notional sweep (single param)
  python scripts/optimization/generate_configs.py \
    --base configs/sprint34_tp15_tr90_holdout.yaml \
    --tag notional_sweep \
    --sweep 'portfolio.max_notional_per_trade=[500,1000,2000,5000,10000]' \
    --output-dir configs/notional_sweep/ \
    --queue-dir artifacts/wfa/aggregate/notional_sweep/

  # With paired holdout/stress generation
  python scripts/optimization/generate_configs.py \
    --base configs/sprint34_tp15_tr90_holdout.yaml \
    --tag oos_validation \
    --sweep 'walk_forward.start_date=["2022-06-01"]' \
    --sweep 'walk_forward.end_date=["2023-04-30"]' \
    --with-stress \
    --output-dir configs/oos_validation/ \
    --queue-dir artifacts/wfa/aggregate/oos_validation/

Sweep modes:
  - Multiple --sweep flags with same-length arrays: zipped (paired) iteration
  - Single --sweep flag: iterate over values
  - Multiple --sweep flags with different lengths: cartesian product
  - Use --zip to force zipped iteration (arrays must be same length)
  - Use --zip-keys to zip only a subset of keys (e.g. pair start/end dates) while
    keeping other sweeps as cartesian products.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

# Add project root to path for imports
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root / "src"))

from coint2.ops.run_queue import RunQueueEntry, write_run_queue


# --- Stress config overrides ---
STRESS_OVERRIDES = {
    "backtest.commission_pct": 0.0006,
    "backtest.commission_rate_per_leg": 0.0006,
    "backtest.slippage_pct": 0.001,
    "backtest.slippage_stress_multiplier": 2.0,
}


def parse_sweep(spec: str) -> Tuple[str, List[Any]]:
    """Parse a sweep spec like 'walk_forward.start_date=["2022-06-01","2023-10-01"]'."""
    key, _, raw_values = spec.partition("=")
    key = key.strip()
    raw_values = raw_values.strip()
    if not key or not raw_values:
        raise ValueError(f"Invalid sweep spec: {spec!r}")
    values = json.loads(raw_values)
    if not isinstance(values, list):
        values = [values]
    return key, values


def set_nested(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using dotted key notation."""
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node:
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def get_nested(cfg: Dict[str, Any], dotted_key: str) -> Any:
    """Get a value from a nested dict using dotted key notation."""
    parts = dotted_key.split(".")
    node = cfg
    for part in parts:
        node = node[part]
    return node


def encode_value(value: Any) -> str:
    """Encode a value for use in filename (matching project conventions)."""
    if isinstance(value, bool):
        return "T" if value else "F"
    if isinstance(value, float):
        return str(value).replace(".", "p").replace("-", "m")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        # Dates: 2022-06-01 -> 20220601
        if re.match(r"\d{4}-\d{2}-\d{2}", value):
            return value.replace("-", "")
        return value.replace(".", "p").replace("-", "m")
    if value is None:
        return "null"
    return str(value).replace(".", "p").replace("-", "m")


def make_tag(key: str, value: Any) -> str:
    """Create a filename tag from a key-value pair."""
    # Use short key names for common parameters
    short_keys = {
        "walk_forward.start_date": "oos",
        "walk_forward.end_date": "to",
        "walk_forward.max_steps": "ms",
        "portfolio.max_notional_per_trade": "maxnot",
        "portfolio.initial_capital": "cap",
        "portfolio.risk_per_position_pct": "risk",
        "portfolio.max_active_positions": "maxpos",
        "backtest.zscore_entry_threshold": "z",
        "backtest.zscore_exit": "exit",
        "backtest.commission_pct": "comm",
        "backtest.slippage_pct": "slip",
        "backtest.rolling_window": "rw",
        "backtest.min_spread_move_sigma": "ms",
        "backtest.pair_stop_loss_usd": "slusd",
        "backtest.pair_stop_loss_zscore": "slz",
        "backtest.stop_loss_multiplier": "slm",
        "backtest.time_stop_multiplier": "ts",
        "backtest.portfolio_daily_stop_pct": "dstop",
        "pair_selection.max_hurst_exponent": "hx",
        "pair_selection.max_pairs": "mp",
        "pair_selection.min_correlation": "corr",
        "pair_selection.coint_pvalue_threshold": "pv",
    }
    short = short_keys.get(key, key.split(".")[-1])
    return f"{short}{encode_value(value)}"


def build_filename(base_name: str, sweep_tags: List[str], kind: str) -> str:
    """Build config filename from base name + sweep tags."""
    # Remove holdout_/stress_ prefix and .yaml suffix from base
    clean = base_name
    for prefix in ("holdout_", "stress_"):
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
            break
    if clean.endswith(".yaml"):
        clean = clean[:-5]

    # Remove old OOS tags if we're sweeping OOS dates
    oos_pattern = r"_oos\d{8}_\d{8}"
    has_oos_sweep = any(t.startswith("oos") for t in sweep_tags)
    if has_oos_sweep:
        clean = re.sub(oos_pattern, "", clean)

    suffix = "_".join(sweep_tags) if sweep_tags else ""
    if suffix:
        name = f"{kind}_{clean}_{suffix}"
    else:
        name = f"{kind}_{clean}"
    return name


def generate_permutations(
    sweeps: List[Tuple[str, List[Any]]], zip_mode: bool
) -> List[List[Tuple[str, Any]]]:
    """Generate all parameter combinations."""
    if not sweeps:
        return [[]]

    if zip_mode:
        lengths = [len(vals) for _, vals in sweeps]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"--zip requires all sweeps to have same length, got {lengths}"
            )
        return [
            [(key, vals[i]) for key, vals in sweeps]
            for i in range(lengths[0])
        ]

    # Check if all same length -> auto-zip
    lengths = [len(vals) for _, vals in sweeps]
    if len(set(lengths)) == 1 and len(sweeps) > 1:
        return [
            [(key, vals[i]) for key, vals in sweeps]
            for i in range(lengths[0])
        ]

    # Cartesian product
    value_lists = [[(key, v) for v in vals] for key, vals in sweeps]
    return [list(combo) for combo in itertools.product(*value_lists)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate WFA config permutations from base YAML + sweep spec"
    )
    parser.add_argument("--base", required=True, help="Base YAML config path")
    parser.add_argument("--tag", required=True, help="Run tag for directory naming")
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        help="Sweep spec: 'key=[val1,val2,...]' (repeatable)",
    )
    parser.add_argument(
        "--with-stress",
        action="store_true",
        help="Generate paired stress configs alongside holdout",
    )
    parser.add_argument("--zip", action="store_true", dest="zip_mode",
                        help="Force zipped (not cartesian) iteration")
    parser.add_argument(
        "--zip-keys",
        action="append",
        default=[],
        help=(
            "Comma-separated sweep keys to zip together (partial zip). Example: "
            "--zip-keys walk_forward.start_date,walk_forward.end_date"
        ),
    )
    parser.add_argument("--output-dir", required=True, help="Directory for generated YAML configs")
    parser.add_argument("--queue-dir", required=True, help="Directory for run_queue.csv")
    parser.add_argument("--runs-dir", default="artifacts/wfa/runs",
                        help="Base directory for WFA run results")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without writing files")

    args = parser.parse_args()

    base_path = Path(args.base)
    if not base_path.exists():
        print(f"Error: base config not found: {base_path}", file=sys.stderr)
        sys.exit(1)

    with base_path.open() as f:
        base_cfg = yaml.safe_load(f)

    sweeps = [parse_sweep(s) for s in args.sweep]

    zip_keys: List[str] = []
    for raw in args.zip_keys:
        for part in str(raw).split(","):
            key = part.strip()
            if key:
                zip_keys.append(key)

    if args.zip_mode and zip_keys:
        raise ValueError("--zip and --zip-keys are mutually exclusive")

    if zip_keys:
        sweep_keys = {k for k, _ in sweeps}
        missing = sorted(set(zip_keys) - sweep_keys)
        if missing:
            raise ValueError(f"--zip-keys contains keys not present in --sweep: {missing}")

        zipped = [s for s in sweeps if s[0] in set(zip_keys)]
        others = [s for s in sweeps if s[0] not in set(zip_keys)]

        zipped_perms = generate_permutations(zipped, zip_mode=True)
        other_perms = generate_permutations(others, zip_mode=False)
        permutations = [z + o for z in zipped_perms for o in other_perms]
    else:
        permutations = generate_permutations(sweeps, args.zip_mode)

    output_dir = Path(args.output_dir)
    queue_dir = Path(args.queue_dir)
    runs_base = args.runs_dir

    entries: List[RunQueueEntry] = []
    base_name = base_path.name

    print(f"Base config: {base_path}")
    print(f"Sweeps: {len(sweeps)} parameters, {len(permutations)} combinations")
    if args.with_stress:
        print(f"With stress: {len(permutations) * 2} total configs")
    print()

    for combo in permutations:
        # Build holdout config
        holdout_cfg = copy.deepcopy(base_cfg)
        sweep_tags = []
        oos_start = None
        oos_end = None

        for key, value in combo:
            set_nested(holdout_cfg, key, value)
            sweep_tags.append(make_tag(key, value))
            if key == "walk_forward.start_date":
                oos_start = encode_value(value)
            if key == "walk_forward.end_date":
                oos_end = encode_value(value)

        # Build OOS tag if dates were swept
        if oos_start and oos_end:
            # Replace individual oos/to tags with combined
            sweep_tags = [
                t for t in sweep_tags
                if not t.startswith("oos") and not t.startswith("to")
            ]
            sweep_tags.insert(0, f"oos{oos_start}_{oos_end}")

        holdout_name = build_filename(base_name, sweep_tags, "holdout")
        holdout_yaml = output_dir / f"{holdout_name}.yaml"
        holdout_results = f"{runs_base}/{args.tag}/{holdout_name}"

        if args.dry_run:
            print(f"  [holdout] {holdout_yaml.name}")
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            with holdout_yaml.open("w") as f:
                yaml.dump(holdout_cfg, f, default_flow_style=False, allow_unicode=True)

        entries.append(RunQueueEntry(
            config_path=str(holdout_yaml),
            results_dir=holdout_results,
            status="planned",
        ))

        # Build stress config if requested
        if args.with_stress:
            stress_cfg = copy.deepcopy(holdout_cfg)
            for skey, sval in STRESS_OVERRIDES.items():
                set_nested(stress_cfg, skey, sval)

            stress_name = build_filename(base_name, sweep_tags, "stress")
            stress_yaml = output_dir / f"{stress_name}.yaml"
            stress_results = f"{runs_base}/{args.tag}/{stress_name}"

            if args.dry_run:
                print(f"  [stress]  {stress_yaml.name}")
            else:
                with stress_yaml.open("w") as f:
                    yaml.dump(stress_cfg, f, default_flow_style=False, allow_unicode=True)

            entries.append(RunQueueEntry(
                config_path=str(stress_yaml),
                results_dir=stress_results,
                status="planned",
            ))

    # Write run queue
    queue_path = queue_dir / "run_queue.csv"
    if args.dry_run:
        print(f"\nQueue: {queue_path} ({len(entries)} entries)")
    else:
        write_run_queue(queue_path, entries)
        print(f"\nGenerated {len(entries)} configs in {output_dir}")
        print(f"Queue written to {queue_path}")


if __name__ == "__main__":
    main()
