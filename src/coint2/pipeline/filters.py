import numpy as np
import pandas as pd
from coint2.core.math_utils import calculate_half_life
# from statsmodels.tsa.stattools import coint  # Заменено на fast_coint
from coint2.core.fast_coint import fast_coint

import logging

def filter_pairs_by_coint_and_half_life(
    pairs: list[tuple[str, str]],
    price_df: pd.DataFrame,
    pvalue_threshold: float = 0.05,
    min_half_life: float = 2,
    max_half_life: float = 100
) -> list[tuple[str, str, float, float, float]]:
    """
    Фильтрует пары по p-value коинтеграции и half-life спреда.
    Возвращает пары с параметрами (s1, s2, beta, mean, std), прошедшие оба фильтра.
    """
    logger = logging.getLogger("pair_filter")
    logger.info(f"[ФИЛЬТР] На входе после SSD: {len(pairs)} пар")
    coint_passed = []
    for s1, s2 in pairs:
        pair_data = price_df[[s1, s2]].dropna()
        if pair_data.empty or pair_data[s2].var() == 0:
            continue
        # Коинтеграционный тест (Engle-Granger) - ускоренная версия
        try:
            _score, pvalue, _ = fast_coint(pair_data[s1], pair_data[s2], trend='n')
        except Exception:
            continue
        if pvalue >= pvalue_threshold:
            continue
        coint_passed.append((s1, s2))
    logger.info(f"[ФИЛЬТР] После фильтра по коинтеграции (p<{pvalue_threshold}): {len(coint_passed)} пар")
    result = []
    for s1, s2 in coint_passed:
        pair_data = price_df[[s1, s2]].dropna()
        beta = pair_data[s1].cov(pair_data[s2]) / pair_data[s2].var()
        spread = pair_data[s1] - beta * pair_data[s2]
        hl = calculate_half_life(spread)
        if not (min_half_life < hl < max_half_life):
            continue
        mean = spread.mean()
        std = spread.std()
        result.append((s1, s2, beta, mean, std))
    logger.info(f"[ФИЛЬТР] После фильтра по half-life ({min_half_life}<hl<{max_half_life}): {len(result)} пар")
    return result
