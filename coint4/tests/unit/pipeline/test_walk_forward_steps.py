from __future__ import annotations

import pandas as pd

from coint2.pipeline.walk_forward_orchestrator import _build_walk_forward_steps


def test_build_walk_forward_steps_clamps_to_end_ts_inclusive_without_overshoot() -> None:
    bar_delta = pd.Timedelta(minutes=15)
    start = pd.Timestamp("2023-06-29 00:00:00")
    end_inclusive = pd.Timestamp("2024-06-27 23:45:00")

    steps = _build_walk_forward_steps(
        start_date=start,
        end_ts_inclusive=end_inclusive,
        training_period_days=90.0,
        testing_period_days=15.0,
        bar_delta=bar_delta,
    )

    assert len(steps) == 25
    assert steps[0][2] == start
    assert steps[-1][3] == end_inclusive
    assert steps[-1][2] == pd.Timestamp("2024-06-23 00:00:00")

    # No overlap between testing windows: next_start == prev_end + bar_delta.
    for prev, nxt in zip(steps, steps[1:]):
        assert nxt[2] == prev[3] + bar_delta

    # End never exceeds the configured inclusive end.
    assert all(step[3] <= end_inclusive for step in steps)


def test_build_walk_forward_steps_returns_empty_when_end_before_start() -> None:
    bar_delta = pd.Timedelta(minutes=15)
    steps = _build_walk_forward_steps(
        start_date=pd.Timestamp("2024-01-10 00:00:00"),
        end_ts_inclusive=pd.Timestamp("2024-01-09 23:45:00"),
        training_period_days=90.0,
        testing_period_days=15.0,
        bar_delta=bar_delta,
    )
    assert steps == []

