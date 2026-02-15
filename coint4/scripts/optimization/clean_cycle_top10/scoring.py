"""Scoring helpers for clean-cycle rollups.

Canonical score (scalar):
  score = canonical_sharpe - lambda_dd * abs(canonical_max_drawdown_abs)
"""

from __future__ import annotations

import math
from typing import Optional


DEFAULT_LAMBDA_DD = 0.02


def compute_score(
    *,
    canonical_sharpe: Optional[float],
    canonical_max_drawdown_abs: Optional[float],
    lambda_dd: float = DEFAULT_LAMBDA_DD,
) -> Optional[float]:
    """Return scalar score or None when inputs are missing/non-finite."""
    if canonical_sharpe is None or canonical_max_drawdown_abs is None:
        return None
    if not math.isfinite(canonical_sharpe) or not math.isfinite(canonical_max_drawdown_abs):
        return None
    if not math.isfinite(lambda_dd) or lambda_dd < 0:
        return None
    return float(canonical_sharpe) - float(lambda_dd) * abs(float(canonical_max_drawdown_abs))

