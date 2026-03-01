"""Tests for safe mutation operator registry (MVP)."""

from __future__ import annotations

import numpy as np

from coint2.ops.genome import KnobSpec
from coint2.ops.mutation_ops import (
    OPERATOR_REGISTRY_V1,
    apply_operator_v1,
    coordinate_sweep_v1,
    crossover_uniform_v1,
    mutate_step_v1,
    random_restart_v1,
    sanitize_genome,
)


def _knob_space() -> list[KnobSpec]:
    return [
        KnobSpec(
            key="backtest.max_var_multiplier",
            type="float",
            min=0.1,
            max=0.5,
            step=0.1,
            round_to=2,
            weight=2.0,
        ),
        KnobSpec(
            key="portfolio.max_active_positions",
            type="int",
            min=10,
            max=20,
            step=5,
            weight=1.0,
        ),
        KnobSpec(
            key="pair_selection.require_same_quote",
            type="bool",
            weight=0.5,
        ),
        KnobSpec(
            key="pair_selection.mode",
            type="categorical",
            choices=["a", "b", "c"],
            weight=1.0,
        ),
    ]


def test_registry_contains_mvp_kinds() -> None:
    assert set(OPERATOR_REGISTRY_V1) == {
        "mutate_step_v1",
        "crossover_uniform_v1",
        "random_restart_v1",
        "coordinate_sweep_v1",
    }


def test_sanitize_genome_drops_non_knob_keys() -> None:
    knob_space = _knob_space()
    genome = {
        "backtest.max_var_multiplier": 0.3,
        "portfolio.max_active_positions": 15,
        "pair_selection.require_same_quote": False,
        "pair_selection.mode": "b",
        "evil.key": 123,
    }
    sanitized = sanitize_genome(genome, knob_space=knob_space)
    assert "evil.key" not in sanitized
    assert set(sanitized) == {spec.key for spec in knob_space}


def test_mutate_step_v1_forced_numeric_key_changes_by_step() -> None:
    knob_space = _knob_space()
    rng = np.random.default_rng(7)
    parent = {
        "backtest.max_var_multiplier": 0.3,
        "portfolio.max_active_positions": 15,
        "pair_selection.require_same_quote": False,
        "pair_selection.mode": "b",
    }
    child = mutate_step_v1(parent, knob_space=knob_space, rng=rng, params={"key": "backtest.max_var_multiplier"})
    assert child["portfolio.max_active_positions"] == 15
    assert child["pair_selection.require_same_quote"] is False
    assert child["pair_selection.mode"] == "b"
    assert child["backtest.max_var_multiplier"] in {0.2, 0.4}


def test_mutate_step_v1_forced_bool_key_flips() -> None:
    knob_space = _knob_space()
    rng = np.random.default_rng(1)
    parent = {
        "backtest.max_var_multiplier": 0.3,
        "portfolio.max_active_positions": 15,
        "pair_selection.require_same_quote": False,
        "pair_selection.mode": "b",
    }
    child = mutate_step_v1(parent, knob_space=knob_space, rng=rng, params={"key": "pair_selection.require_same_quote"})
    assert child["pair_selection.require_same_quote"] is True


def test_crossover_uniform_v1_uses_only_parent_values() -> None:
    knob_space = _knob_space()
    rng = np.random.default_rng(42)
    a = {
        "backtest.max_var_multiplier": 0.2,
        "portfolio.max_active_positions": 10,
        "pair_selection.require_same_quote": False,
        "pair_selection.mode": "a",
    }
    b = {
        "backtest.max_var_multiplier": 0.5,
        "portfolio.max_active_positions": 20,
        "pair_selection.require_same_quote": True,
        "pair_selection.mode": "c",
    }
    child = crossover_uniform_v1(a, b, knob_space=knob_space, rng=rng)
    for key in (spec.key for spec in knob_space):
        assert child[key] in {a[key], b[key]}


def test_random_restart_v1_respects_bounds_and_choices() -> None:
    knob_space = _knob_space()
    rng = np.random.default_rng(123)
    genome = random_restart_v1(knob_space=knob_space, rng=rng)
    assert genome["backtest.max_var_multiplier"] in {0.1, 0.2, 0.3, 0.4, 0.5}
    assert genome["portfolio.max_active_positions"] in {10, 15, 20}
    assert genome["pair_selection.require_same_quote"] in {True, False}
    assert genome["pair_selection.mode"] in {"a", "b", "c"}


def test_coordinate_sweep_v1_budget_and_order() -> None:
    knob_space = _knob_space()
    rng = np.random.default_rng(999)
    parent = {
        "backtest.max_var_multiplier": 0.3,
        "portfolio.max_active_positions": 15,
        "pair_selection.require_same_quote": False,
        "pair_selection.mode": "b",
    }
    sweep = coordinate_sweep_v1(
        parent,
        knob_space=knob_space,
        rng=rng,
        budget=5,
        params={"keys": ["portfolio.max_active_positions", "backtest.max_var_multiplier"]},
    )
    assert len(sweep) == 5
    # Stable cartesian order: (pos-5, mv-0.1) first.
    assert sweep[0]["portfolio.max_active_positions"] == 10
    assert sweep[0]["backtest.max_var_multiplier"] == 0.2


def test_coordinate_sweep_v1_respects_max_keys_when_keys_provided() -> None:
    knob_space = _knob_space()
    rng = np.random.default_rng(123)
    parent = {
        "backtest.max_var_multiplier": 0.3,
        "portfolio.max_active_positions": 15,
        "pair_selection.require_same_quote": False,
        "pair_selection.mode": "b",
    }
    sweep = coordinate_sweep_v1(
        parent,
        knob_space=knob_space,
        rng=rng,
        budget=20,
        params={
            "keys": [
                "portfolio.max_active_positions",
                "backtest.max_var_multiplier",
                "pair_selection.require_same_quote",
            ],
            "max_keys": 1,
        },
    )
    assert sweep
    watched = [
        "portfolio.max_active_positions",
        "backtest.max_var_multiplier",
        "pair_selection.require_same_quote",
    ]
    union_changed = set()
    for child in sweep:
        for key in watched:
            if child[key] != parent[key]:
                union_changed.add(key)
    assert len(union_changed) <= 1


def test_coordinate_sweep_v1_supports_bool_keys() -> None:
    knob_space = _knob_space()
    rng = np.random.default_rng(0)
    parent = {
        "backtest.max_var_multiplier": 0.3,
        "portfolio.max_active_positions": 15,
        "pair_selection.require_same_quote": False,
        "pair_selection.mode": "b",
    }
    sweep = coordinate_sweep_v1(
        parent,
        knob_space=knob_space,
        rng=rng,
        budget=10,
        params={"keys": ["pair_selection.require_same_quote"]},
    )
    assert {bool(item["pair_selection.require_same_quote"]) for item in sweep} == {False, True}


def test_apply_operator_v1_dispatch_smoke() -> None:
    knob_space = _knob_space()
    rng = np.random.default_rng(0)
    parent = {
        "backtest.max_var_multiplier": 0.3,
        "portfolio.max_active_positions": 15,
        "pair_selection.require_same_quote": False,
        "pair_selection.mode": "b",
    }
    out = apply_operator_v1("mutate_step_v1", parents=[parent], knob_space=knob_space, rng=rng, params={"key": "pair_selection.mode"})
    assert len(out) == 1
    assert out[0]["pair_selection.mode"] in {"a", "c"}
