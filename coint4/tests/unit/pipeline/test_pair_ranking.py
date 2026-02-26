import pytest

from coint2.pipeline.pair_ranking import apply_entry_rank, rank_pairs


@pytest.mark.unit
def test_rank_pairs_when_spread_std_then_sorts_by_abs_std_desc() -> None:
    pairs = [
        ("A", "B", 1.0, 0.0, 1.0, {"half_life": 10, "mean_crossings": 5, "pvalue": 0.01}),
        ("C", "D", 1.0, 0.0, 3.0, {"half_life": 10, "mean_crossings": 5, "pvalue": 0.01}),
        ("E", "F", 1.0, 0.0, 2.0, {"half_life": 10, "mean_crossings": 5, "pvalue": 0.01}),
    ]

    ranked, weights = rank_pairs(pairs, "spread_std")

    assert [f"{p[0]}-{p[1]}" for p in ranked] == ["C-D", "E-F", "A-B"]
    assert weights["C-D"] == 1.0
    assert weights["A-B"] == 0.0


@pytest.mark.unit
def test_rank_pairs_when_composite_v1_then_prefers_mr_strength_over_std() -> None:
    # Pair A-B has better mean reversion (lower half-life, more crossings, lower pvalue),
    # but lower std. Composite should still rank it above C-D.
    pairs = [
        ("A", "B", 1.0, 0.0, 1.0, {"half_life": 5, "mean_crossings": 10, "pvalue": 0.001}),
        ("C", "D", 1.0, 0.0, 3.0, {"half_life": 20, "mean_crossings": 3, "pvalue": 0.04}),
    ]

    ranked, weights = rank_pairs(pairs, "composite_v1")

    assert [f"{p[0]}-{p[1]}" for p in ranked] == ["A-B", "C-D"]
    assert weights["A-B"] > weights["C-D"]


@pytest.mark.unit
def test_apply_entry_rank_when_abs_signal_then_no_change() -> None:
    assert (
        apply_entry_rank(
            2.0,
            pair_quality=1.0,
            entry_rank_mode="abs_signal",
            pair_quality_alpha=1.0,
        )
        == 2.0
    )


@pytest.mark.unit
def test_apply_entry_rank_when_abs_signal_x_pair_quality_then_scaled() -> None:
    # pair_quality in [0,1], centered at 0.5 => multiplier in [1-alpha, 1+alpha].
    assert apply_entry_rank(
        2.0,
        pair_quality=1.0,
        entry_rank_mode="abs_signal_x_pair_quality",
        pair_quality_alpha=0.5,
    ) == pytest.approx(3.0)  # 2.0 * 1.5
    assert apply_entry_rank(
        2.0,
        pair_quality=0.0,
        entry_rank_mode="abs_signal_x_pair_quality",
        pair_quality_alpha=0.5,
    ) == pytest.approx(1.0)  # 2.0 * 0.5

