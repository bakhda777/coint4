from __future__ import annotations

from coint2.pipeline.walk_forward_orchestrator import _compute_pair_trailing_streaks


def test_trailing_streak_when_pair_not_recent_then_zero() -> None:
    history_window = [
        [("A", "B"), ("C", "D")],
        [("A", "B")],
        [("A", "B"), ("E", "F")],
    ]

    streaks = _compute_pair_trailing_streaks(history_window)

    assert streaks[("A", "B")] == 3
    assert streaks[("E", "F")] == 1
    assert streaks[("C", "D")] == 0


def test_trailing_streak_when_history_has_gap_then_streak_stops_on_gap() -> None:
    history_window = [
        [("A", "B")],
        [("A", "B")],
        [("X", "Y")],
        [("A", "B")],
    ]

    streaks = _compute_pair_trailing_streaks(history_window)

    assert streaks[("A", "B")] == 1
    assert streaks[("X", "Y")] == 0
