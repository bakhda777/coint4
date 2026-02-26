"""Tests for config genome/fingerprint helpers."""

from __future__ import annotations

from pathlib import Path

from coint2.ops.genome import (
    KnobSpec,
    fingerprint_yaml_config,
    genome_distance_weighted_l1_v1,
    genome_from_config,
    load_effective_yaml_config,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_fingerprint_stable_across_key_order(tmp_path: Path) -> None:
    a = _write(
        tmp_path / "a.yaml",
        """
backtest:
  zscore_entry_threshold: 1.15
portfolio:
  max_active_positions: 18
""".lstrip(),
    )
    b = _write(
        tmp_path / "b.yaml",
        """
portfolio:
  max_active_positions: 18
backtest:
  zscore_entry_threshold: 1.15
""".lstrip(),
    )

    assert fingerprint_yaml_config(a) == fingerprint_yaml_config(b)


def test_fingerprint_materializes_base_config(tmp_path: Path) -> None:
    base = _write(
        tmp_path / "base.yaml",
        """
backtest:
  zscore_entry_threshold: 1.15
portfolio:
  max_active_positions: 18
results_dir: results
""".lstrip(),
    )
    override = _write(
        tmp_path / "override.yaml",
        """
base_config: base.yaml
portfolio:
  max_active_positions: 20
""".lstrip(),
    )

    effective = load_effective_yaml_config(override)
    assert effective["portfolio"]["max_active_positions"] == 20
    assert effective["backtest"]["zscore_entry_threshold"] == 1.15

    # Fingerprint must change when the effective config changes.
    fp_override = fingerprint_yaml_config(override)
    _write(
        override,
        """
base_config: base.yaml
portfolio:
  max_active_positions: 21
""".lstrip(),
    )
    fp_override2 = fingerprint_yaml_config(override)
    assert fp_override != fp_override2

    # Sanity: base fingerprint differs from override fingerprint.
    assert fingerprint_yaml_config(base) != fp_override


def test_fingerprint_rounds_float_jitter(tmp_path: Path) -> None:
    a = _write(
        tmp_path / "a.yaml",
        """
backtest:
  max_var_multiplier: 0.30000000000000004
""".lstrip(),
    )
    b = _write(
        tmp_path / "b.yaml",
        """
backtest:
  max_var_multiplier: 0.3
""".lstrip(),
    )

    assert fingerprint_yaml_config(a) == fingerprint_yaml_config(b)


def test_genome_distance_weighted_l1_v1_range_norm() -> None:
    knob_space = [
        KnobSpec(
            key="backtest.zscore_entry_threshold",
            type="float",
            min=0.0,
            max=2.0,
            weight=2.0,
            norm="range",
        ),
        KnobSpec(
            key="portfolio.max_active_positions",
            type="int",
            min=1.0,
            max=21.0,
            weight=1.0,
            norm="range",
        ),
        KnobSpec(
            key="pair_selection.require_same_quote",
            type="bool",
            weight=0.5,
            norm="none",
        ),
    ]

    cfg_a = {
        "backtest": {"zscore_entry_threshold": 1.15},
        "portfolio": {"max_active_positions": 18},
        "pair_selection": {"require_same_quote": False},
    }
    cfg_b = {
        "backtest": {"zscore_entry_threshold": 1.25},
        "portfolio": {"max_active_positions": 20},
        "pair_selection": {"require_same_quote": True},
    }

    a = genome_from_config(cfg_a, knob_space=knob_space)
    b = genome_from_config(cfg_b, knob_space=knob_space)

    # z: |1.15-1.25|/2.0 * 2.0 = 0.1
    # pos: |18-20|/20.0 * 1.0 = 0.1
    # bool: different -> 0.5
    assert genome_distance_weighted_l1_v1(a, b, knob_space=knob_space) == 0.7

