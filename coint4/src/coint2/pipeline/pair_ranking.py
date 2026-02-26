from __future__ import annotations

import math
from typing import Any

PairPayload = tuple[str, str, float, float, float, dict[str, Any]]


def _is_finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except Exception:
        return False


def _rank_feature(values: list[float], *, higher_is_better: bool) -> list[float]:
    """Return rank-based weights in [0, 1], where 1.0 is best and NaN/None -> 0.0."""
    ranks = [0.0] * len(values)
    finite_idx = [i for i, v in enumerate(values) if _is_finite(v)]
    if not finite_idx:
        return ranks

    finite_sorted = sorted(
        finite_idx,
        key=lambda i: float(values[i]),
        reverse=bool(higher_is_better),
    )
    n = len(finite_sorted)
    if n == 1:
        ranks[finite_sorted[0]] = 1.0
        return ranks
    for pos, i in enumerate(finite_sorted):
        ranks[i] = 1.0 - (pos / (n - 1))
    return ranks


def rank_pairs(
    pairs: list[PairPayload],
    rank_mode: str | None,
) -> tuple[list[PairPayload], dict[str, float]]:
    """Sort pairs by a chosen quality score and return per-pair weights in [0, 1].

    The weights are intended to be used for portfolio entry ranking under maxpos.
    """
    mode = (rank_mode or "spread_std").strip().lower()

    weights: list[float]
    if mode in {"spread_std", "std", "vol", "spread_vol"}:
        std_abs = [
            abs(float(std)) if _is_finite(std) else float("nan")
            for (_s1, _s2, _beta, _mean, std, _metrics) in pairs
        ]
        weights = _rank_feature(std_abs, higher_is_better=True)
    elif mode in {"composite_v1", "quality_v1"}:
        mean_crossings: list[float] = []
        half_life: list[float] = []
        pvalue: list[float] = []
        spread_std: list[float] = []
        ecm_tstat: list[float] = []
        beta_drift: list[float] = []
        has_ecm = False
        has_beta_drift = False

        for (_s1, _s2, _beta, _mean, std, metrics) in pairs:
            metrics = metrics or {}
            mean_crossings.append(
                float(metrics.get("mean_crossings"))
                if _is_finite(metrics.get("mean_crossings"))
                else float("nan")
            )
            half_life.append(
                float(metrics.get("half_life")) if _is_finite(metrics.get("half_life")) else float("nan")
            )
            pvalue.append(
                float(metrics.get("pvalue")) if _is_finite(metrics.get("pvalue")) else float("nan")
            )
            spread_std.append(abs(float(std)) if _is_finite(std) else float("nan"))

            raw_ecm = metrics.get("ecm_alpha_tstat")
            if _is_finite(raw_ecm):
                has_ecm = True
                ecm_tstat.append(float(raw_ecm))
            else:
                ecm_tstat.append(float("nan"))

            raw_beta_drift = metrics.get("beta_drift_ratio")
            if _is_finite(raw_beta_drift):
                has_beta_drift = True
                beta_drift.append(float(raw_beta_drift))
            else:
                beta_drift.append(float("nan"))

        feature_ranks: list[list[float]] = [
            _rank_feature(mean_crossings, higher_is_better=True),
            _rank_feature(half_life, higher_is_better=False),
            _rank_feature(pvalue, higher_is_better=False),
            _rank_feature(spread_std, higher_is_better=True),
        ]
        # ECM t-stat: more negative is better.
        if has_ecm:
            feature_ranks.append(_rank_feature(ecm_tstat, higher_is_better=False))
        if has_beta_drift:
            feature_ranks.append(_rank_feature(beta_drift, higher_is_better=False))

        denom = float(len(feature_ranks)) if feature_ranks else 1.0
        weights = [0.0] * len(pairs)
        for i in range(len(pairs)):
            weights[i] = float(sum(fr[i] for fr in feature_ranks) / denom)
    else:
        # Fail-closed to legacy behavior.
        std_abs = [
            abs(float(std)) if _is_finite(std) else float("nan")
            for (_s1, _s2, _beta, _mean, std, _metrics) in pairs
        ]
        weights = _rank_feature(std_abs, higher_is_better=True)

    scored: list[tuple[PairPayload, float, str]] = []
    for pair, w in zip(pairs, weights, strict=False):
        s1, s2, *_rest = pair
        scored.append((pair, float(w), f"{s1}-{s2}"))

    # Deterministic: secondary sort by pair name, then stable sort by score.
    scored.sort(key=lambda x: x[2])
    scored.sort(key=lambda x: x[1], reverse=True)

    sorted_pairs = [pair for (pair, _w, _name) in scored]
    weight_map = {name: w for (_pair, w, name) in scored}
    return sorted_pairs, weight_map


def apply_entry_rank(
    base_strength: float,
    *,
    pair_quality: float | None,
    entry_rank_mode: str | None,
    pair_quality_alpha: float,
) -> float:
    """Adjust entry signal strength based on the chosen entry-ranking mode."""
    try:
        strength = float(base_strength)
    except Exception:
        return 0.0
    if not math.isfinite(strength):
        return 0.0

    mode = (entry_rank_mode or "abs_signal").strip().lower()
    if mode in {"abs_signal", "abs_score", "abs_z"}:
        return strength

    if mode in {"abs_signal_x_pair_quality", "abs_signal_times_pair_quality", "abs_z_x_pair_quality"}:
        try:
            alpha = float(pair_quality_alpha or 0.0)
        except Exception:
            alpha = 0.0
        if alpha <= 0.0:
            return strength

        try:
            q = float(pair_quality) if pair_quality is not None else 0.5
        except Exception:
            q = 0.5
        if not math.isfinite(q):
            q = 0.5

        # pair_quality is expected in [0, 1], with 0.5 meaning "neutral".
        mult = 1.0 + alpha * (q - 0.5) * 2.0
        if mult < 0.0:
            mult = 0.0
        return strength * mult

    # Unknown mode: keep behavior stable.
    return strength

