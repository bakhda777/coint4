import numpy as np
import pandas as pd
from coint2.core.math_utils import calculate_half_life
from coint2.core.fast_coint import fast_coint
from coint2.analysis.pair_filter import calculate_hurst_exponent
from statsmodels.tsa.stattools import kpss  # KPSS тест стационарности
from statsmodels.tools.sm_exceptions import InterpolationWarning
import warnings
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# --- Placeholder market data helpers ---
_market_cache: dict[str, tuple[float,float,float]] = {}

def _get_market_metrics(symbol: str) -> tuple[float, float, float]:
    """Return (avg_volume_usd, bid_ask_pct, avg_funding_pct).
    In production replace with real data source.
    """
    if symbol in _market_cache:
        return _market_cache[symbol]
    # TODO: integrate real market data. For now use safe defaults.
    _market_cache[symbol] = (float('inf'), 0.0, 0.0)
    return _market_cache[symbol]

import logging
from typing import List, Tuple, Optional, Dict, Any

# Разумные границы для коэффициента beta
MIN_BETA = 0.1
MAX_BETA = 10.0

def filter_pairs_by_coint_and_half_life(
    pairs: List[Tuple[str, str]],
    price_df: pd.DataFrame,
    pvalue_threshold: float = 0.05,
    min_beta: float = MIN_BETA,
    max_beta: float = MAX_BETA,
    min_half_life: float = 2,
    max_half_life: float = 100,
    min_mean_crossings: int = 8,
    min_history_ratio: float = 0.8,
    liquidity_usd_daily: float = 1_000_000.0,
    max_bid_ask_pct: float = 0.2,
    max_avg_funding_pct: float = 0.03,
    save_filter_reasons: bool = True,
    kpss_pvalue_threshold: float = 0.05,
    max_hurst_exponent: float = 0.5,
    *,
    stable_tokens: Optional[List[str]] = None,
) -> List[Tuple[str, str, float, float, float, Dict[str, Any]]]:
    """
    Фильтрует пары по ключевым критериям качества:
    1. p-value коинтеграции
    2. коэффициент beta
    3. half-life спреда
    4. Количество пересечений среднего
    
    Возвращает пары с параметрами (s1, s2, beta, mean, std, metrics), прошедшие все фильтры.
    """
    logger = logging.getLogger("pair_filter")
    if stable_tokens is None:
        stable_tokens = [
            "USDT",
            "USDC",
            "BUSD",
            "TUSD",
            "DAI",
            "USD",
            "USDP",
            "PAX",
            "SUSD",
            "GUSD",
        ]
    def _is_stable(sym: str) -> bool:
        return sym in stable_tokens
    logger.info(f"[ФИЛЬТР] На входе после SSD: {len(pairs)} пар")
    
    # Словарь для отслеживания причин фильтрации по категориям
    filter_stats = {
        'total': len(pairs),
        'pvalue': 0,
        'beta': 0,
        'half_life': 0,
        'crossings': 0,
        'hurst': 0,
    }
    
    # Фильтр 1: Коинтеграция (p-value)
    coint_passed = []
    filter_reasons: list[tuple[str,str,str]] = []  # (s1,s2,reason)
    for s1, s2 in pairs:
        # --- History coverage check ---
        hist_ratio1 = price_df[s1].notna().mean()
        hist_ratio2 = price_df[s2].notna().mean()
        if min(hist_ratio1, hist_ratio2) < min_history_ratio:
            filter_reasons.append((s1, s2, 'history'))
            filter_stats['history'] = filter_stats.get('history', 0) + 1
            continue

        # --- Liquidity & market microstructure checks ---
        vol1, ba1, fund1 = _get_market_metrics(s1)
        vol2, ba2, fund2 = _get_market_metrics(s2)
        if min(vol1, vol2) < liquidity_usd_daily or max(ba1, ba2) > max_bid_ask_pct or max(abs(fund1), abs(fund2)) > max_avg_funding_pct:
            filter_reasons.append((s1, s2, 'liquidity'))
            filter_stats['liquidity'] = filter_stats.get('liquidity', 0) + 1
            continue
        pair_data = price_df[[s1, s2]].dropna()
        if pair_data.empty or pair_data[s2].var() == 0:
            continue
            
        # Корреляция между активами (для метрик)
        corr = pair_data[s1].corr(pair_data[s2])
            
        # Коинтеграционный тест (Engle-Granger) - ускоренная версия
        try:
            _score, pvalue, _ = fast_coint(pair_data[s1], pair_data[s2], trend='n')
        except Exception:
            continue
            
        if pvalue >= pvalue_threshold:
            filter_reasons.append((s1, s2, 'pvalue'))
            filter_stats['pvalue'] += 1
            continue
            
        coint_passed.append((s1, s2, corr, pvalue))
        

    logger.info(f"[ФИЛЬТР] После фильтра коинтеграции: {len(coint_passed)} пар")

    total_pairs_coint = len(coint_passed)
    half_life_passed = 0
    beta_passed = 0

    # Первичный проход — вычисляем half-life и std, собираем метрики
    tmp_stats: list[tuple[str, str, float, float, float, float, float, float]] = []  # s1,s2,beta,mean,std,hl_days,mean_crossings,pvalue
    for s1, s2, corr, pvalue in coint_passed:
        pair_data = price_df[[s1, s2]].dropna()
        beta = pair_data[s1].cov(pair_data[s2]) / pair_data[s2].var()

        if not (min_beta <= abs(beta) <= max_beta):
            filter_reasons.append((s1, s2, f"beta_out_of_range ({beta:.2f})"))
            filter_stats['beta'] += 1
            continue

        beta_passed += 1

        spread = pair_data[s1] - beta * pair_data[s2]

        hurst_exponent = calculate_hurst_exponent(spread)
        if hurst_exponent > max_hurst_exponent:
            filter_reasons.append((s1, s2, f"hurst_too_high ({hurst_exponent:.2f})"))
            filter_stats['hurst'] += 1
            continue

        # Расчёт half-life (в барах)
        hl_bars = calculate_half_life(spread)
        # Определяем таймфрейм в минутах по разнице индексов
        try:
            bar_minutes = int((pair_data.index[1] - pair_data.index[0]).total_seconds() / 60)
            if bar_minutes <= 0:
                bar_minutes = 15
        except Exception:
            bar_minutes = 15  # fallback
        hl_days = hl_bars * bar_minutes / 1440
        if not (min_half_life < hl_days < max_half_life):
            filter_reasons.append((s1, s2, 'half_life'))
            filter_stats['half_life'] += 1
            continue
        half_life_passed += 1
        hl = hl_days  # сохраняем в днях для метрик
            
        # KPSS тест стационарности спреда
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=InterpolationWarning)
                p_kpss = kpss(spread, regression='c', nlags='auto')[1]
            if p_kpss < kpss_pvalue_threshold:
                filter_reasons.append((s1, s2, 'kpss'))
                filter_stats['kpss'] = filter_stats.get('kpss', 0) + 1
                continue  # отвергаем стационарность тренда
        except Exception:
            pass  # если тест не применим, пропускаем

        # Расчет стандартного отклонения спреда
        mean = spread.mean()
        std = spread.std()
        # отложим проверку std/other until we вычислим адаптивные пороги
        
        tmp_stats.append((s1, s2, beta, mean, std, hl_days, 0.0, pvalue))  # mean_cross later

    # --- Second pass: crossings & metrics ---
    mean_cross_passed = 0
    result = []
    for (s1, s2, beta, mean, std, hl_days, _, pvalue) in tmp_stats:

        pair_data = price_df[[s1, s2]].dropna()
        spread = pair_data[s1] - beta * pair_data[s2]
        z_score = (spread - mean) / std
        mean_crossings = ((z_score[:-1] * z_score[1:]) < 0).sum()
        # динамический минимум пересечений
        train_bars = len(spread)
        hl_bars_est = hl_days * 1440 / 15  # приблизительно, bar_minutes=15
        
        # Значительно снижаем порог пересечений (было: делим на hl_bars_est)
        dynamic_min_cross = max(1, int(train_bars / max(hl_bars_est * 10, 1)))
        # Фильтр временно отключен для диагностики
        if False and mean_crossings < min(2, dynamic_min_cross):  # Минимум 2 пересечения
            filter_reasons.append((s1, s2, 'crossings'))
            filter_stats['crossings'] += 1
            continue
        mean_cross_passed += 1

        # Вычисляем среднее абсолютное отклонение для метрики
        mean_abs_dev = np.abs(spread - mean).mean()
            
        # Сохраняем расширенные метрики для пары
        metrics = {
            'half_life': hl,
            'correlation': corr,
            'mean_crossings': mean_crossings,
            'spread_std': std,
            'pvalue': pvalue
        }
        
        result.append((s1, s2, beta, mean, std, metrics))
        
    # Логирование итогов по этапам
    # Рассчитываем процент отфильтрованных пар для каждой причины
    total_pairs = len(pairs)
    filter_percentages = {reason: (count / total_pairs * 100) for reason, count in filter_stats.items() if count > 0}
    
    # Детальная статистика по фильтрам
    logger.info(f"[ФИЛЬТР] Статистика по причинам фильтрации:")
    for reason, percent in sorted(filter_percentages.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  • {reason}: {filter_stats[reason]} пар ({percent:.1f}%)")
    
    logger.info(
        f"[ФИЛЬТР] Beta range {min_beta}-{max_beta}: {total_pairs_coint} → {beta_passed} пар"
    )
    logger.info(f"[ФИЛЬТР] Half-life: {total_pairs_coint} → {half_life_passed} пар")
    logger.info(f"[ФИЛЬТР] Mean crossings: {len(tmp_stats)} → {mean_cross_passed} пар")
    logger.info(f"[ФИЛЬТР] После всех фильтров: {len(result)} пар ({len(result)/total_pairs*100:.1f}% от исходного числа)")

    # Сохраняем причины отсева в CSV
    if save_filter_reasons and filter_reasons:
        out_dir = Path('results')
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = out_dir / f'filter_reasons_{ts}.csv'
        pd.DataFrame(filter_reasons, columns=['s1','s2','reason']).to_csv(out_path, index=False)
        logger.info(f"Причины отсева сохранены в {out_path}")

    return result
