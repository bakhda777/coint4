import numpy as np
import pandas as pd
from coint2.core.math_utils import calculate_half_life, count_mean_crossings
from coint2.core.fast_coint import fast_coint
from coint2.analysis.pair_filter import calculate_hurst_exponent
from statsmodels.tsa.stattools import kpss  # KPSS тест стационарности
from statsmodels.tools.sm_exceptions import InterpolationWarning
import statsmodels.api as sm
import warnings
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import logging
from typing import List, Tuple, Dict, Any, Optional

# --- Placeholder market data helpers ---
_market_cache: dict[str, tuple[float,float,float]] = {}

def _get_market_metrics(symbol: str) -> tuple[float, float, float]:
    """Return (avg_volume_usd, bid_ask_pct, avg_funding_pct).
    In production replace with real data source.
    """
    if symbol in _market_cache:
        return _market_cache[symbol]
    # WARNING: Using placeholder data - integrate real market data in production
    _market_cache[symbol] = (float('inf'), 0.0, 0.0)
    return _market_cache[symbol]

import logging
from typing import List, Tuple, Optional, Dict, Any

# Разумные границы для коэффициента beta
MIN_BETA = 0.1
MAX_BETA = 10.0

def enhanced_pair_screening(
    pairs: List[Tuple[str, str]],
    price_df: pd.DataFrame,
    pvalue_threshold: float = 0.05,
    max_half_life_bars: int = 96,  # 96 * 15min = 24 hours for 15-min data
    min_daily_volume_usd: float = 50_000.0,  # $50k per leg
    min_beta: float = MIN_BETA,
    max_beta: float = MAX_BETA,
    min_mean_crossings: int = 8,
    min_history_ratio: float = 0.8,
    save_filter_reasons: bool = False,
    kpss_pvalue_threshold: float = 0.05,
    max_hurst_exponent: float = 0.5,
    *,
    stable_tokens: Optional[List[str]] = None,
) -> List[Tuple[str, str, float, float, float, Dict[str, Any]]]:
    """Enhanced pair screening with strict criteria for 15-minute data.
     
     Criteria:
     1. p-value коинтеграции < 0.05
     2. half-life < N бар (default 96 bars = 24 hours for 15-min data)
     3. среднедневной объём ≥ 50k $ на каждую leg
     
     Parameters
     ----------
     pairs : List[Tuple[str, str]]
         List of symbol pairs to screen
     price_df : pd.DataFrame
         Price data with columns for each symbol
     pvalue_threshold : float, default 0.05
         Maximum p-value for cointegration test
     max_half_life_bars : int, default 96
         Maximum half-life in bars (96 * 15min = 24 hours)
     min_daily_volume_usd : float, default 50_000.0
         Minimum average daily volume in USD per leg
     
     Returns
     -------
     List[Tuple[str, str, float, float, float, Dict[str, Any]]]
         Filtered pairs with (s1, s2, beta, mean, std, metrics)
     """
    logger = logging.getLogger("enhanced_pair_screening")
    if stable_tokens is None:
        stable_tokens = [
            "USDT", "USDC", "BUSD", "TUSD", "DAI", "USD", "USDP", "PAX", "SUSD", "GUSD"
        ]
    
    def _is_stable(sym: str) -> bool:
        return sym in stable_tokens
    
    logger.info(f"[ENHANCED SCREENING] Starting with {len(pairs)} pairs")
    
    # Statistics tracking
    filter_stats = {
        "p_value_failed": 0,
        "half_life_failed": 0,
        "volume_failed": 0,
        "beta_failed": 0,
        "data_insufficient": 0,
        "kpss_failed": 0,
        "hurst_failed": 0,
        "total_passed": 0
    }
    
    filtered_pairs = []
    filter_reasons = []
    
    for s1, s2 in pairs:
        try:
            # Check if symbols exist in price data
            if s1 not in price_df.columns or s2 not in price_df.columns:
                filter_stats["data_insufficient"] += 1
                if save_filter_reasons:
                    filter_reasons.append((s1, s2, "missing_price_data"))
                continue
            
            # Get price series
            y = price_df[s1].dropna()
            x = price_df[s2].dropna()
            
            # Align series
            common_idx = y.index.intersection(x.index)
            if len(common_idx) < 100:  # Minimum data requirement
                filter_stats["data_insufficient"] += 1
                if save_filter_reasons:
                    filter_reasons.append((s1, s2, "insufficient_data"))
                continue
            
            y_aligned = y.loc[common_idx]
            x_aligned = x.loc[common_idx]
            
            # 1. Cointegration test (p-value criterion)
            try:
                _, pvalue, _ = fast_coint(y_aligned, x_aligned, trend='n')
                if pvalue > pvalue_threshold:
                    filter_stats["p_value_failed"] += 1
                    if save_filter_reasons:
                        filter_reasons.append((s1, s2, f"p_value_{pvalue:.4f}"))
                    continue
            except Exception as e:
                filter_stats["data_insufficient"] += 1
                if save_filter_reasons:
                    filter_reasons.append((s1, s2, f"coint_error_{str(e)[:50]}"))
                continue
            
            # Calculate regression parameters
            try:
                X = sm.add_constant(x_aligned)
                model = sm.OLS(y_aligned, X).fit()
                beta = model.params.iloc[1]
                
                # 2. Beta criterion
                if not (min_beta <= abs(beta) <= max_beta):
                    filter_stats["beta_failed"] += 1
                    if save_filter_reasons:
                        filter_reasons.append((s1, s2, f"beta_{beta:.4f}"))
                    continue
                
                # Calculate spread and statistics
                spread = y_aligned - beta * x_aligned
                
                # Check for empty spread to avoid numpy warnings
                if len(spread) == 0 or spread.isna().all():
                    filter_stats["spread_empty"] = filter_stats.get("spread_empty", 0) + 1
                    if save_filter_reasons:
                        filter_reasons.append((s1, s2, "spread_empty"))
                    continue
                
                mean_spread = spread.mean()
                std_spread = spread.std()
                
                # 3. Half-life criterion (in bars for 15-min data)
                half_life_bars = calculate_half_life(spread)
                if half_life_bars > max_half_life_bars:
                    filter_stats["half_life_failed"] += 1
                    if save_filter_reasons:
                        filter_reasons.append((s1, s2, f"half_life_{half_life_bars:.1f}_bars"))
                    continue
                
                # 4. Volume criterion (estimate daily volume)
                # For 15-min data: 96 bars per day
                bars_per_day = 96
                recent_data_days = min(30, len(common_idx) // bars_per_day)  # Last 30 days or available
                
                if recent_data_days > 0:
                    recent_periods = recent_data_days * bars_per_day
                    recent_y = y_aligned.tail(recent_periods)
                    recent_x = x_aligned.tail(recent_periods)
                    
                    # Estimate daily volume (simplified - using price * typical volume multiplier)
                    # This is a placeholder - in real implementation, you'd use actual volume data
                    
                    # Check for empty recent data to avoid numpy warnings
                    if len(recent_y) == 0 or recent_y.isna().all() or len(recent_x) == 0 or recent_x.isna().all():
                        filter_stats["recent_data_empty"] = filter_stats.get("recent_data_empty", 0) + 1
                        if save_filter_reasons:
                            filter_reasons.append((s1, s2, "recent_data_empty"))
                        continue
                    
                    avg_price_s1 = recent_y.mean()
                    avg_price_s2 = recent_x.mean()
                    
                    # Placeholder volume check - in practice, you'd get this from market data
                    # For now, we'll use a simplified heuristic based on price volatility
                    volatility_s1 = recent_y.pct_change().std() * np.sqrt(bars_per_day)
                    volatility_s2 = recent_x.pct_change().std() * np.sqrt(bars_per_day)
                    
                    # Estimate volume based on volatility (higher vol usually means higher volume)
                    # This is a rough approximation - replace with actual volume data
                    estimated_volume_s1 = avg_price_s1 * 1000 * (1 + volatility_s1 * 10)
                    estimated_volume_s2 = avg_price_s2 * 1000 * (1 + volatility_s2 * 10)
                    
                    if (estimated_volume_s1 < min_daily_volume_usd or 
                        estimated_volume_s2 < min_daily_volume_usd):
                        filter_stats["volume_failed"] += 1
                        if save_filter_reasons:
                            filter_reasons.append((s1, s2, f"volume_s1_{estimated_volume_s1:.0f}_s2_{estimated_volume_s2:.0f}"))
                        continue
                
                # Additional quality checks
                
                # KPSS test for spread stationarity
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", InterpolationWarning)
                        kpss_stat, kpss_pvalue, _, _ = kpss(spread, regression='c')
                    
                    if kpss_pvalue < kpss_pvalue_threshold:
                        filter_stats["kpss_failed"] += 1
                        if save_filter_reasons:
                            filter_reasons.append((s1, s2, f"kpss_pvalue_{kpss_pvalue:.4f}"))
                        continue
                except Exception:
                    # If KPSS fails, skip this pair
                    filter_stats["kpss_failed"] += 1
                    if save_filter_reasons:
                        filter_reasons.append((s1, s2, "kpss_error"))
                    continue
                
                # Hurst exponent check
                try:
                    hurst = calculate_hurst_exponent(spread)
                    if hurst > max_hurst_exponent:
                        filter_stats["hurst_failed"] += 1
                        if save_filter_reasons:
                            filter_reasons.append((s1, s2, f"hurst_{hurst:.4f}"))
                        continue
                except Exception:
                    # If Hurst calculation fails, continue (don't filter out)
                    pass
                
                # Mean crossings check
                mean_crossings = count_mean_crossings(spread)
                if mean_crossings < min_mean_crossings:
                    if save_filter_reasons:
                        filter_reasons.append((s1, s2, f"mean_crossings_{mean_crossings}"))
                    continue
                
                # If we reach here, the pair passed all filters
                metrics = {
                    "p_value": pvalue,
                    "half_life_bars": half_life_bars,
                    "beta": beta,
                    "mean_crossings": mean_crossings,
                    "hurst_exponent": hurst if 'hurst' in locals() else 0.5,
                    "kpss_pvalue": kpss_pvalue if 'kpss_pvalue' in locals() else 1.0,
                    "estimated_volume_s1": estimated_volume_s1 if 'estimated_volume_s1' in locals() else 0,
                    "estimated_volume_s2": estimated_volume_s2 if 'estimated_volume_s2' in locals() else 0,
                }
                
                filtered_pairs.append((s1, s2, beta, mean_spread, std_spread, metrics))
                filter_stats["total_passed"] += 1
                
            except Exception as e:
                filter_stats["data_insufficient"] += 1
                if save_filter_reasons:
                    filter_reasons.append((s1, s2, f"calculation_error_{str(e)[:50]}"))
                continue
                
        except Exception as e:
            filter_stats["data_insufficient"] += 1
            if save_filter_reasons:
                filter_reasons.append((s1, s2, f"general_error_{str(e)[:50]}"))
            continue
    
    # Log results
    logger.info(f"[ENHANCED SCREENING] Results:")
    logger.info(f"  Total pairs processed: {len(pairs)}")
    logger.info(f"  Passed all filters: {filter_stats['total_passed']}")
    logger.info(f"  Failed p-value: {filter_stats['p_value_failed']}")
    logger.info(f"  Failed half-life: {filter_stats['half_life_failed']}")
    logger.info(f"  Failed volume: {filter_stats['volume_failed']}")
    logger.info(f"  Failed beta: {filter_stats['beta_failed']}")
    logger.info(f"  Failed KPSS: {filter_stats['kpss_failed']}")
    logger.info(f"  Failed Hurst: {filter_stats['hurst_failed']}")
    logger.info(f"  Data insufficient: {filter_stats['data_insufficient']}")
    
    # Save filter reasons if requested
    if save_filter_reasons and filter_reasons:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filter_file = Path(f"enhanced_filter_reasons_{timestamp}.csv")
        
        with open(filter_file, 'w', newline='') as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(["symbol1", "symbol2", "filter_reason"])
            writer.writerows(filter_reasons)
        
        logger.info(f"Filter reasons saved to: {filter_file}")
    
    return filtered_pairs
    
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
    save_filter_reasons: bool = False,
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
    logger.info(f"[ФИЛЬТР] Применяем оптимизированный порядок: Коинтеграция → Beta → Half-life → Mean crossings → Hurst → KPSS → Market microstructure")
    
    # Словарь для отслеживания причин фильтрации по категориям
    filter_stats = {
        'total': len(pairs),
        'insufficient_data': 0,
        'zero_variance': 0,
        'pvalue': 0,
        'beta': 0,
        'half_life': 0,
        'crossings': 0,
        'hurst': 0,
        'kpss': 0,
        'history': 0,
        'liquidity': 0,
    }
    
    # Фильтр 1: Коинтеграция (p-value) - основной критерий качества
    coint_passed = []
    filter_reasons: list[tuple[str,str,str]] = []  # (s1,s2,reason)
    for s1, s2 in pairs:
        pair_data = price_df[[s1, s2]].dropna()
        if pair_data.empty or len(pair_data) < 30:  # Минимум 30 наблюдений для статистических тестов
            filter_reasons.append((s1, s2, 'insufficient_data'))
            filter_stats['insufficient_data'] += 1
            continue
        if pair_data[s2].var() == 0:
            filter_reasons.append((s1, s2, 'zero_variance'))
            filter_stats['zero_variance'] += 1
            continue
            
        # Корреляция между активами (для метрик)
        corr = pair_data[s1].corr(pair_data[s2])
            
        # Коинтеграционный тест (Engle-Granger) - ускоренная версия
        try:
            _score, pvalue, _ = fast_coint(pair_data[s1], pair_data[s2], trend='n')
            if pvalue is None or np.isnan(pvalue):
                filter_reasons.append((s1, s2, 'invalid_pvalue'))
                filter_stats['invalid_pvalue'] = filter_stats.get('invalid_pvalue', 0) + 1
                continue
        except Exception as e:
            logger.debug(f"Ошибка коинтеграционного теста для пары {s1}-{s2}: {str(e)}")
            filter_reasons.append((s1, s2, 'coint_test_error'))
            filter_stats['coint_test_error'] = filter_stats.get('coint_test_error', 0) + 1
            continue
            
        if pvalue >= pvalue_threshold:
            filter_reasons.append((s1, s2, 'pvalue'))
            filter_stats['pvalue'] += 1
            continue
            
        coint_passed.append((s1, s2, corr, pvalue))
        

    logger.info(f"[ФИЛЬТР] После фильтра коинтеграции: {len(coint_passed)} пар")

    # Фильтр 2: Beta range проверка (очень быстро)
    beta_passed = []
    for s1, s2, corr, pvalue in coint_passed:
        pair_data = price_df[[s1, s2]].dropna()
        beta = pair_data[s1].cov(pair_data[s2]) / pair_data[s2].var()

        if not (min_beta <= abs(beta) <= max_beta):
            filter_reasons.append((s1, s2, f"beta_out_of_range ({beta:.2f})"))
            filter_stats['beta'] += 1
            continue

        beta_passed.append((s1, s2, corr, pvalue, beta))
    
    logger.info(f"[ФИЛЬТР] После фильтра beta: {len(beta_passed)} пар")

    # Фильтр 3: Half-life расчет (быстро)
    half_life_passed = []
    for s1, s2, corr, pvalue, beta in beta_passed:
        pair_data = price_df[[s1, s2]].dropna()
        spread = pair_data[s1] - beta * pair_data[s2]

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
        
        half_life_passed.append((s1, s2, corr, pvalue, beta, hl_days, spread))
    
    logger.info(f"[ФИЛЬТР] После фильтра half-life: {len(half_life_passed)} пар")

    # Фильтр 4: Mean crossings (быстро)
    crossings_passed = []
    for s1, s2, corr, pvalue, beta, hl_days, spread in half_life_passed:
        mean_crossings = count_mean_crossings(spread)
        if mean_crossings < min_mean_crossings:
            filter_reasons.append((s1, s2, 'crossings'))
            filter_stats['crossings'] += 1
            continue
        
        crossings_passed.append((s1, s2, corr, pvalue, beta, hl_days, spread, mean_crossings))
    
    logger.info(f"[ФИЛЬТР] После фильтра mean crossings: {len(crossings_passed)} пар")

    # Фильтр 5: Hurst exponent (умеренно)
    hurst_passed = []
    for s1, s2, corr, pvalue, beta, hl_days, spread, mean_crossings in crossings_passed:
        hurst_exponent = calculate_hurst_exponent(spread)
        if hurst_exponent > max_hurst_exponent:
            filter_reasons.append((s1, s2, f"hurst_too_high ({hurst_exponent:.2f})"))
            filter_stats['hurst'] += 1
            continue
        
        hurst_passed.append((s1, s2, corr, pvalue, beta, hl_days, spread, mean_crossings))
    
    logger.info(f"[ФИЛЬТР] После фильтра Hurst: {len(hurst_passed)} пар")

    # Фильтр 6: KPSS тест (медленно)
    kpss_passed = []
    for s1, s2, corr, pvalue, beta, hl_days, spread, mean_crossings in hurst_passed:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=InterpolationWarning)
                p_kpss = kpss(spread, regression='c', nlags='auto')[1]
            if p_kpss < kpss_pvalue_threshold:
                filter_reasons.append((s1, s2, 'kpss'))
                filter_stats['kpss'] += 1
                continue  # отвергаем стационарность тренда
        except Exception:
            pass  # если тест не применим, пропускаем
        
        kpss_passed.append((s1, s2, corr, pvalue, beta, hl_days, spread, mean_crossings))
    
    logger.info(f"[ФИЛЬТР] После фильтра KPSS: {len(kpss_passed)} пар")

    # Фильтр 7: Market microstructure (liquidity, bid-ask, funding)
    microstructure_passed = []
    for s1, s2, corr, pvalue, beta, hl_days, spread, mean_crossings in kpss_passed:
        # --- History coverage check ---
        hist_ratio1 = price_df[s1].notna().mean()
        hist_ratio2 = price_df[s2].notna().mean()
        if min(hist_ratio1, hist_ratio2) < min_history_ratio:
            filter_reasons.append((s1, s2, 'history'))
            filter_stats['history'] += 1
            continue

        # --- Liquidity & market microstructure checks ---
        vol1, ba1, fund1 = _get_market_metrics(s1)
        vol2, ba2, fund2 = _get_market_metrics(s2)
        if min(vol1, vol2) < liquidity_usd_daily or max(ba1, ba2) > max_bid_ask_pct or max(abs(fund1), abs(fund2)) > max_avg_funding_pct:
            filter_reasons.append((s1, s2, 'liquidity'))
            filter_stats['liquidity'] += 1
            continue
        
        # Расчет стандартного отклонения спреда
        if len(spread) == 0 or spread.isna().all():
            filter_reasons.append((s1, s2, 'spread_empty_stats'))
            filter_stats['spread_empty_stats'] = filter_stats.get('spread_empty_stats', 0) + 1
            continue
        
        mean = spread.mean()
        std = spread.std()
        
        microstructure_passed.append((s1, s2, corr, pvalue, beta, hl_days, mean, std, mean_crossings))
    
    logger.info(f"[ФИЛЬТР] После фильтра market microstructure: {len(microstructure_passed)} пар")

    # Формируем финальный результат
    result = []
    for s1, s2, corr, pvalue, beta, hl_days, mean, std, mean_crossings in microstructure_passed:
        # Сохраняем расширенные метрики для пары
        metrics = {
            'half_life': hl_days,
            'correlation': corr,
            'mean_crossings': mean_crossings,
            'spread_std': std,
            'pvalue': pvalue
        }
        
        result.append((s1, s2, beta, mean, std, metrics))
        
    # Логирование итогов по этапам
    total_pairs = len(pairs)
    
    # Детальная статистика по фильтрам в порядке применения
    logger.info(f"[ФИЛЬТР] Статистика фильтрации (оптимизированный порядок):")
    logger.info(f"  1. SSD → Коинтеграция: {total_pairs} → {len(coint_passed)} пар")
    logger.info(f"  2. Коинтеграция → Beta: {len(coint_passed)} → {len(beta_passed)} пар")
    logger.info(f"  3. Beta → Half-life: {len(beta_passed)} → {len(half_life_passed)} пар")
    logger.info(f"  4. Half-life → Mean crossings: {len(half_life_passed)} → {len(crossings_passed)} пар")
    logger.info(f"  5. Mean crossings → Hurst: {len(crossings_passed)} → {len(hurst_passed)} пар")
    logger.info(f"  6. Hurst → KPSS: {len(hurst_passed)} → {len(kpss_passed)} пар")
    logger.info(f"  7. KPSS → Market microstructure: {len(kpss_passed)} → {len(microstructure_passed)} пар")
    
    # Рассчитываем процент отфильтрованных пар для каждой причины
    filter_percentages = {reason: (count / total_pairs * 100) for reason, count in filter_stats.items() if count > 0}
    
    # Детальная статистика по причинам отсева
    logger.info(f"[ФИЛЬТР] Причины отсева:")
    for reason, percent in sorted(filter_percentages.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  • {reason}: {filter_stats[reason]} пар ({percent:.1f}%)")
    
    logger.info(f"[ФИЛЬТР] Итого: {total_pairs} → {len(result)} пар ({len(result)/total_pairs*100:.1f}% прошли все фильтры)")

    # Сохраняем причины отсева в CSV
    if save_filter_reasons and filter_reasons:
        out_dir = Path('results')
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = out_dir / f'filter_reasons_{ts}.csv'
        pd.DataFrame(filter_reasons, columns=['s1','s2','reason']).to_csv(out_path, index=False)
        logger.info(f"Причины отсева сохранены в {out_path}")

    return result
