import logging

import dask
import pandas as pd
from dask import delayed
# from statsmodels.tsa.stattools import coint  # Заменено на fast_coint

from coint2.core import math_utils
from coint2.core.fast_coint import fast_coint
from coint2.utils.config import AppConfig
from coint2.utils.timing_utils import logged_time, time_block, ProgressTracker

# Настройка логгера
logger = logging.getLogger(__name__)


@delayed
def _test_pair_for_tradability(
    handler,
    symbol1: str,
    symbol2: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    min_half_life: float,
    max_half_life: float,
    min_crossings: int,
) -> tuple[str, str] | None:
    """Lazy tradability filter for a pair."""
    # Обеспечиваем, что даты в наивном формате (без timezone)
    if start_date.tzinfo is not None:
        start_date = start_date.tz_localize(None)
    if end_date.tzinfo is not None:
        end_date = end_date.tz_localize(None)
    
    logger.debug(f"Проверка tradability для пары {symbol1}-{symbol2} ({start_date} - {end_date})")
    
    pair_data = handler.load_pair_data(symbol1, symbol2, start_date, end_date)
    if pair_data.empty or len(pair_data.columns) < 2:
        logger.debug(f"Пара {symbol1}-{symbol2}: нет данных или недостаточно столбцов")
        return None

    logger.debug(f"Пара {symbol1}-{symbol2}: получены данные размером {len(pair_data)}")
    
    y = pair_data[symbol1]
    x = pair_data[symbol2]
    beta = y.cov(x) / x.var()
    spread = y - beta * x

    spread_np = spread.to_numpy()
    half_life = math_utils.half_life_numba(spread_np)
    logger.debug(
        f"Пара {symbol1}-{symbol2}: half_life = {half_life:.2f} "
        f"(требуется {min_half_life:.2f}-{max_half_life:.2f})"
    )
    
    if half_life < min_half_life or half_life > max_half_life:
        logger.debug(f"Пара {symbol1}-{symbol2}: отклонена по half_life")
        return None

    crossings = math_utils.mean_crossings_numba(spread_np)
    logger.debug(f"Пара {symbol1}-{symbol2}: mean_crossings = {crossings} (требуется мин. {min_crossings})")
    
    if crossings < min_crossings:
        logger.debug(f"Пара {symbol1}-{symbol2}: отклонена по mean_crossings")
        return None

    logger.debug(f"Пара {symbol1}-{symbol2}: прошла фильтр tradability")
    return symbol1, symbol2


def _coint_test(series1: pd.Series, series2: pd.Series) -> float:
    """Run fast cointegration test and return p-value."""
    _score, pvalue, _ = fast_coint(series1, series2, trend='n')
    return pvalue


@delayed
def _test_pair_for_coint(
    handler,
    symbol1: str,
    symbol2: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    p_value_threshold: float,
) -> tuple[str, str, float, float, float] | None:
    """Lazy test for a single pair using provided handler and dates."""
    # Обеспечиваем, что даты в наивном формате (без timezone)
    if start_date.tzinfo is not None:
        start_date = start_date.tz_localize(None)
    if end_date.tzinfo is not None:
        end_date = end_date.tz_localize(None)
    
    logger.debug(f"Проверка коинтеграции для пары {symbol1}-{symbol2} ({start_date} - {end_date})")
    
    pair_data = handler.load_pair_data(symbol1, symbol2, start_date, end_date)
    if pair_data.empty or len(pair_data.columns) < 2:
        logger.debug(f"Пара {symbol1}-{symbol2}: нет данных или недостаточно столбцов для коинтеграции")
        return None

    logger.debug(f"Пара {symbol1}-{symbol2}: получены данные размером {len(pair_data)} для теста коинтеграции")
    
    # Проверяем типы данных и логируем
    if (
        not pd.api.types.is_numeric_dtype(pair_data[symbol1])
        or not pd.api.types.is_numeric_dtype(pair_data[symbol2])
    ):
        logger.warning(
            "Пара %s-%s: данные не числовые: %s, %s",
            symbol1,
            symbol2,
            pair_data[symbol1].dtype,
            pair_data[symbol2].dtype,
        )
        return None
    
    # Проверяем на наличие NaN, если более 10% - отбрасываем пару
    nan_ratio1 = pair_data[symbol1].isna().mean()
    nan_ratio2 = pair_data[symbol2].isna().mean()
    if nan_ratio1 > 0.1 or nan_ratio2 > 0.1:
        logger.debug(f"Пара {symbol1}-{symbol2}: слишком много NaN ({nan_ratio1:.2f}, {nan_ratio2:.2f})")
        return None
    
    # Очищаем данные от NaN одновременно по обоим столбцам, чтобы сохранить синхронизацию
    clean_data = pair_data[[symbol1, symbol2]].dropna()
    pvalue = _coint_test(clean_data[symbol1], clean_data[symbol2])
    logger.debug(f"Пара {symbol1}-{symbol2}: p-value = {pvalue:.4f} (требуется < {p_value_threshold:.4f})")
    
    if pvalue >= p_value_threshold:
        logger.debug(f"Пара {symbol1}-{symbol2}: отклонена по p-value коинтеграции")
        return None

    y = pair_data[symbol1]
    x = pair_data[symbol2]
    beta = y.cov(x) / x.var()
    spread = y - beta * x
    mean = spread.mean()
    std = spread.std()
    
    logger.debug(f"Пара {symbol1}-{symbol2}: успешно прошла фильтр коинтеграции")
    return symbol1, symbol2, beta, mean, std


@logged_time("find_cointegrated_pairs")
def find_cointegrated_pairs(
    handler,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cfg: AppConfig,
) -> list[tuple[str, str, float, float, float]]:
    """Find cointegrated pairs using SSD pre-filtering."""
    
    # Обеспечиваем, что даты в наивном формате (без timezone)
    if start_date.tzinfo is not None:
        logger.debug(f"Удаляю timezone из start_date: {start_date}")
        start_date = start_date.tz_localize(None)
    if end_date.tzinfo is not None:
        logger.debug(f"Удаляю timezone из end_date: {end_date}")
        end_date = end_date.tz_localize(None)

    logger.info(f"Поиск коинтегрированных пар в периоде {start_date} - {end_date}")
    p_value_threshold = cfg.pair_selection.coint_pvalue_threshold
    logger.info(
        "Настройки фильтрации: p_value < %s, half_life = %s-%s дней, min_crossings = %s",
        p_value_threshold,
        cfg.pair_selection.min_half_life_days,
        cfg.pair_selection.max_half_life_days,
        cfg.pair_selection.min_mean_crossings,
    )

    # Stage 1: SSD pre-filter
    logger.info("🔍 ЭТАП 1: Загрузка данных и SSD-фильтрация")
    with time_block("loading and normalizing data for SSD"):
        normalized = handler.load_and_normalize_data(start_date, end_date)
        if normalized.empty or len(normalized.columns) < 2:
            logger.warning("Не удалось загрузить данные или недостаточно символов")
            return []
        
        logger.info(
            f"Загружены данные для {len(normalized.columns)} символов за период "
            f"{normalized.index[0]} - {normalized.index[-1]}"
        )

    # Calculate theoretical number of pairs
    num_symbols = len(normalized.columns)
    theoretical_pairs = (num_symbols * (num_symbols - 1)) // 2
    logger.info(f"Теоретическое число пар: {theoretical_pairs:,} из {num_symbols} символов")

    with time_block("SSD computation"):
        logger.info(
            f"Выполняю SSD-фильтрацию без ограничения (всего {theoretical_pairs:,} пар)"
        )
        ssd = math_utils.calculate_ssd(
            normalized,
            top_k=None,
        )
        top_pairs = ssd.index.tolist()
        logger.info(f"SSD отобрал {len(top_pairs):,} из {theoretical_pairs:,} пар ({len(top_pairs)/theoretical_pairs*100:.1f}%)")

    # Stage 2: tradability filter
    logger.info("🔍 ЭТАП 2: Проверка tradability (half-life, mean crossings)")
    with time_block("tradability testing"):
        lazy_tradable = []
        for s1, s2 in top_pairs:
            task = _test_pair_for_tradability(
                handler,
                s1,
                s2,
                start_date,
                end_date,
                cfg.pair_selection.min_half_life_days,
                cfg.pair_selection.max_half_life_days,
                cfg.pair_selection.min_mean_crossings,
            )
            lazy_tradable.append(task)

        logger.info(f"Запускаю параллельную проверку tradability для {len(lazy_tradable):,} пар")
        tradable_results = dask.compute(*lazy_tradable, scheduler="threads")
        tradable_pairs = [p for p in tradable_results if p is not None]
        
        logger.info(f"Прошли фильтр tradability: {len(tradable_pairs):,} из {len(top_pairs):,} пар ({len(tradable_pairs)/len(top_pairs)*100:.1f}%)")

    # Stage 3: cointegration filter и расширенные метрики
    logger.info("🔍 ЭТАП 3: Тесты коинтеграции и расширенные метрики")
    with time_block("cointegration and metrics testing"):
        # Загружаем все данные для пар, прошедших tradability фильтр
        all_pairs_data = {}
        max_pairs_in_memory = 1000  # Ограничение для контроля памяти
        
        if len(tradable_pairs) > max_pairs_in_memory:
            logger.warning(f"Слишком много пар ({len(tradable_pairs)}), ограничиваем до {max_pairs_in_memory} для контроля памяти")
            tradable_pairs = tradable_pairs[:max_pairs_in_memory]
            
        for s1, s2 in tradable_pairs:
            pair_data = handler.load_pair_data(s1, s2, start_date, end_date).dropna()
            if not pair_data.empty and len(pair_data.columns) >= 2:
                all_pairs_data[(s1, s2)] = pair_data
        
        logger.info(f"Загружены данные для {len(all_pairs_data)} пар")
        
        # Создаем DataFrame с данными всех пар для передачи в filter_pairs_by_coint_and_half_life
        if not all_pairs_data:
            logger.warning("Не удалось загрузить данные ни для одной пары")
            return []
            
        # Объединяем все данные в один DataFrame
        all_symbols = set()
        for s1, s2 in all_pairs_data.keys():
            all_symbols.add(s1)
            all_symbols.add(s2)
            
        # Создаем DataFrame с данными всех символов
        price_df = pd.DataFrame()
        for pair, data in all_pairs_data.items():
            s1, s2 = pair
            if s1 not in price_df.columns:
                price_df[s1] = data[s1]
            if s2 not in price_df.columns:
                price_df[s2] = data[s2]
                
        logger.info(f"Создан DataFrame с данными для {len(price_df.columns)} символов")
        
        # Импортируем функцию фильтрации
        from coint2.pipeline.filters import filter_pairs_by_coint_and_half_life
        
        # Применяем расширенную фильтрацию
        final_pairs = filter_pairs_by_coint_and_half_life(
            pairs=tradable_pairs,
            price_df=price_df,
            pvalue_threshold=p_value_threshold,
            min_half_life=cfg.pair_selection.min_half_life_days,
            max_half_life=cfg.pair_selection.max_half_life_days,
            min_mean_crossings=cfg.pair_selection.min_mean_crossings,
            max_hurst_exponent=getattr(cfg.pair_selection, 'max_hurst_exponent', 0.5),
            # Параметры commission_pct и slippage_pct удалены - больше не используются
        )
        success_rate = (len(final_pairs)/len(tradable_pairs)*100) if len(tradable_pairs) > 0 else 0
        logger.info(f"Прошли все фильтры: {len(final_pairs):,} из {len(tradable_pairs):,} пар ({success_rate:.1f}%)")
        logger.info(
            f"Фильтры: p-value < {p_value_threshold}, half-life = {cfg.pair_selection.min_half_life_days}-{cfg.pair_selection.max_half_life_days}, "
            f"mean_crossings >= {cfg.pair_selection.min_mean_crossings}"
        )

    
    # Summary
    if not final_pairs:
        logger.warning("⚠️  НЕ НАЙДЕНО ни одной коинтегрированной пары. Рекомендации:")
        logger.warning("  • Ослабить pvalue_threshold (например, до 0.10)")
        logger.warning("  • Увеличить max_half_life_days")
        logger.warning("  • Уменьшить min_mean_crossings")
        logger.warning("  • Увеличить ssd_top_n")
    else:
        # Сортируем найденные пары по p-value коинтеграции и берем топ-N
        pvalue_top_n = cfg.pair_selection.pvalue_top_n
        final_pairs = sorted(final_pairs, key=lambda x: x[5].get('pvalue', 1.0))[:pvalue_top_n]
        logger.info(
            f"✅ ИТОГО найдено {len(final_pairs)} качественных пар (Топ-{pvalue_top_n} по p-value):"
        )
        for i, (s1, s2, beta, mean, std, metrics) in enumerate(final_pairs[:5], 1):
            logger.info(f"  {i}. {s1}-{s2}: beta={beta:.4f}, mean={mean:.4f}, std={std:.4f}, "  
                      f"half_life={metrics['half_life']:.1f}, corr={metrics['correlation']:.2f}, "  
                      f"mean_crossings={metrics['mean_crossings']}")
        if len(final_pairs) > 5:
            logger.info(f"  ... и еще {len(final_pairs) - 5} пар")
            
    return final_pairs
