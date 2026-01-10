"""
Улучшенные методы нормализации для снижения потерь символов.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
import logging

logger = logging.getLogger("normalization")


def _fill_gaps_session_aware(df: pd.DataFrame, method: str = "ffill", limit: int = 5) -> pd.DataFrame:
    """
    КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Заполняет пропуски только внутри торговых сессий.
    Предотвращает утечку данных через ночные gaps.

    Args:
        df: DataFrame с данными
        method: Метод заполнения ("ffill" или "linear")
        limit: Лимит заполнения

    Returns:
        DataFrame с заполненными пропусками только внутри сессий
    """
    if df.empty:
        return df

    try:
        # Определяем торговые сессии по дням
        session_dates = df.index.normalize()

        def fill_within_session(group):
            """Заполняет пропуски только внутри одной торговой сессии."""
            if len(group) <= 1:
                return group

            # КРИТИЧНО: НЕ используем bfill для предотвращения заполнения назад через gaps
            if method == "ffill":
                # Только forward fill, без backward fill
                return group.ffill(limit=limit)
            elif method == "linear":
                # Интерполяция + forward fill, но БЕЗ backward fill
                return (group.interpolate(method='linear', limit=limit)
                       .ffill(limit=limit))
            else:
                return group

        # Группируем по дням и применяем заполнение только внутри каждого дня
        filled_df = (df.groupby(session_dates, group_keys=False)
                    .apply(fill_within_session))

        return filled_df

    except Exception as e:
        logger.warning(f"Ошибка в сессионном заполнении: {e}, используем fallback без bfill")
        # Fallback БЕЗ backward fill для предотвращения lookahead
        if method == "ffill":
            return df.ffill(limit=limit)
        elif method == "linear":
            return df.interpolate(method='linear', limit=limit).ffill(limit=limit)
        else:
            return df

def preprocess_and_normalize_data(
    price_df: pd.DataFrame,
    min_history_ratio: float = 0.8,
    fill_method: str = "ffill",
    norm_method: str = "zscore",
    handle_constant: bool = True,
    epsilon: float = 1e-8
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Улучшенная предобработка и нормализация данных цен.
    
    Args:
        price_df: DataFrame с ценами, индекс - timestamp, колонки - символы
        min_history_ratio: Минимальная доля непропущенных значений
        fill_method: Метод заполнения пропусков ('ffill', 'linear', None)
        norm_method: Метод нормализации ('minmax', 'log_returns')
        handle_constant: Обрабатывать ли символы с постоянной ценой
        epsilon: Малое число для предотвращения деления на ноль
        
    Returns:
        Нормализованный DataFrame и статистика обработки
    """
    stats = {
        "initial_symbols": len(price_df.columns),
        "low_history_ratio": 0,
        "constant_price": 0,
        "nan_after_norm": 0,
        "final_symbols": 0
    }
    
    # Проверка на пустые данные
    if price_df.empty or price_df.shape[0] == 0 or price_df.shape[1] == 0:
        logger.warning("Получен пустой DataFrame для нормализации")
        stats["final_symbols"] = 0
        return pd.DataFrame(), stats
    
    # Этап 1: Фильтрация по истории данных
    history_ratios = price_df.notna().mean()
    valid_history = history_ratios >= min_history_ratio
    filtered_df = price_df.loc[:, valid_history]
    stats["low_history_ratio"] = (~valid_history).sum()
    
    logger.info(f"Символы с недостаточной историей: {stats['low_history_ratio']} из {stats['initial_symbols']}")
    
    if filtered_df.shape[1] < 2:
        logger.warning(f"Осталось менее 2 символов после фильтрации по истории данных")
        stats["final_symbols"] = filtered_df.shape[1]
        return filtered_df, stats
    
    # Этап 2: Проверка на постоянные цены
    if handle_constant:
        constant_mask = filtered_df.max() == filtered_df.min()
        constant_symbols = filtered_df.columns[constant_mask].tolist()
        stats["constant_price"] = len(constant_symbols)
        
        if constant_symbols:
            logger.info(f"Символы с постоянной ценой: {len(constant_symbols)}")
            logger.debug(f"Список: {', '.join(constant_symbols)}")
            
            # Вместо удаления, заменим на нормализованные константы
            for col in constant_symbols:
                # Для каждого метода нормализации используем подходящее константное значение
                if norm_method == "minmax":
                    # Для min-max: заменяем на 0.5 (середина диапазона [0,1])
                    filtered_df[col] = 0.5
                elif norm_method == "log_returns":
                    # Для лог-доходности: заменяем на 0 (лог-доходность константы = 0)
                    filtered_df[col] = 0
                elif norm_method == "zscore":
                    # Для z-score: заменяем на 0 (среднее значение)
                    filtered_df[col] = 0
    
    if filtered_df.shape[1] < 2:
        logger.warning(f"Осталось менее 2 символов после удаления символов с постоянной ценой")
        stats["final_symbols"] = filtered_df.shape[1]
        return filtered_df, stats
    
    # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Заполнение пропусков только внутри торговых сессий
    if fill_method == "ffill":
        limit = 5  # Максимальное число подряд идущих пропусков для заполнения
        filled_df = _fill_gaps_session_aware(filtered_df, method="ffill", limit=limit)
    elif fill_method == "linear":
        limit = 5
        filled_df = _fill_gaps_session_aware(filtered_df, method="linear", limit=limit)
    else:
        filled_df = filtered_df
    
    # Этап 4: Нормализация
    if norm_method == "minmax":
        # Улучшенная min-max нормализация с защитой от деления на ноль
        data_min = filled_df.min()
        data_max = filled_df.max()
        data_range = data_max - data_min
        
        # Защита от деления на ноль
        data_range = data_range.replace(0, epsilon)
        
        normalized_df = (filled_df - data_min) / data_range
        

        
    elif norm_method == "log_returns":
        # Логарифмические доходности
        normalized_df = np.log(filled_df / filled_df.shift(1))
        normalized_df = normalized_df.iloc[1:]  # Удаляем первую строку с NaN

    elif norm_method == "zscore":
        # Z-score (standardization)
        data_mean = filled_df.mean()
        data_std = filled_df.std()

        # Защита от деления на ноль для колонок с нулевой вариацией
        data_std = data_std.replace(0, epsilon)

        normalized_df = (filled_df - data_mean) / data_std

    else:
        raise ValueError(f"Неизвестный метод нормализации: {norm_method}")
    
    # Проверка на NaN после нормализации
    nan_mask = normalized_df.isna().any()
    nan_columns = normalized_df.columns[nan_mask].tolist()
    stats["nan_after_norm"] = len(nan_columns)
    
    if nan_columns:
        logger.info(f"Символы с NaN после нормализации: {len(nan_columns)}")
        if len(nan_columns) <= 10:
            logger.debug(f"Список: {', '.join(nan_columns)}")
        normalized_df = normalized_df.dropna(axis=1)
    
    stats["final_symbols"] = normalized_df.shape[1]
    logger.info(f"Итоговое количество символов после нормализации: {stats['final_symbols']} из {stats['initial_symbols']}")
    
    return normalized_df, stats


def apply_normalization_with_params(
    price_df: pd.DataFrame,
    normalization_params: Dict[str, Dict[str, float]],
    norm_method: str = "minmax",
    fill_method: str = "ffill",
    epsilon: float = 1e-8
) -> pd.DataFrame:
    """
    Применяет нормализацию с предвычисленными параметрами (для предотвращения look-ahead bias).

    Args:
        price_df: DataFrame с ценами для нормализации
        normalization_params: Словарь с параметрами нормализации для каждого символа
                             Для minmax: {'symbol': {'min': value, 'max': value}}
        norm_method: Метод нормализации ('minmax', 'log_returns')
        fill_method: Метод заполнения пропусков
        epsilon: Малое число для предотвращения деления на ноль

    Returns:
        Нормализованный DataFrame
    """
    logger = logging.getLogger(__name__)

    if price_df.empty:
        return price_df

    # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Заполнение пропусков только внутри торговых сессий
    if fill_method == "ffill":
        limit = 5
        filled_df = _fill_gaps_session_aware(price_df, method="ffill", limit=limit)
    elif fill_method == "linear":
        limit = 5
        filled_df = _fill_gaps_session_aware(price_df, method="linear", limit=limit)
    else:
        filled_df = price_df

    # Применяем нормализацию с предвычисленными параметрами
    if norm_method == "minmax":
        normalized_df = filled_df.copy()

        for symbol in filled_df.columns:
            if symbol in normalization_params:
                params = normalization_params[symbol]
                data_min = params['min']
                data_max = params['max']
                data_range = data_max - data_min

                # Защита от деления на ноль
                if abs(data_range) < epsilon:
                    data_range = epsilon

                normalized_df[symbol] = (filled_df[symbol] - data_min) / data_range
            else:
                logger.warning(f"Нет параметров нормализации для символа {symbol}, пропускаем")
                normalized_df = normalized_df.drop(columns=[symbol])

    elif norm_method == "log_returns":
        # Для log_returns параметры не нужны
        normalized_df = np.log(filled_df / filled_df.shift(1))
        normalized_df = normalized_df.iloc[1:]  # Удаляем первую строку с NaN

    elif norm_method == "zscore":
        # Применяем z-score нормализацию с предвычисленными параметрами
        normalized_df = filled_df.copy()

        for symbol in filled_df.columns:
            if symbol in normalization_params:
                params = normalization_params[symbol]
                data_mean = params['mean']
                data_std = params['std']

                # Защита от деления на ноль
                if data_std == 0:
                    data_std = epsilon

                normalized_df[symbol] = (filled_df[symbol] - data_mean) / data_std
            else:
                logger.warning(f"Нет параметров нормализации для символа {symbol}, пропускаем")
                normalized_df = normalized_df.drop(columns=[symbol])

    else:
        raise ValueError(f"Неизвестный метод нормализации: {norm_method}")

    # Удаляем колонки с NaN
    normalized_df = normalized_df.dropna(axis=1)

    return normalized_df


def compute_normalization_params(
    price_df: pd.DataFrame,
    norm_method: str = "minmax"
) -> Dict[str, Dict[str, float]]:
    """
    Вычисляет параметры нормализации на тренировочных данных.

    Args:
        price_df: DataFrame с тренировочными данными
        norm_method: Метод нормализации

    Returns:
        Словарь с параметрами нормализации для каждого символа
    """
    params = {}

    if norm_method == "minmax":
        for symbol in price_df.columns:
            series = price_df[symbol].dropna()
            if len(series) > 0:
                params[symbol] = {
                    'min': float(series.min()),
                    'max': float(series.max())
                }
    elif norm_method == "log_returns":
        # Для log_returns параметры не нужны
        pass
    elif norm_method == "zscore":
        # Для z-score нужны среднее и стандартное отклонение
        for symbol in price_df.columns:
            series = price_df[symbol].dropna()
            if len(series) > 0:
                params[symbol] = {
                    'mean': float(series.mean()),
                    'std': float(series.std())
                }

    return params


def adaptive_normalization(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Адаптивная нормализация: автоматически выбирает лучший метод нормализации 
    на основе характеристик данных.
    
    Args:
        price_df: DataFrame с ценами
        
    Returns:
        Нормализованный DataFrame с максимальным сохранением символов
    """
    methods = ["minmax", "log_returns"]
    best_method = None
    max_symbols = 0
    
    logger.info(f"Запуск адаптивной нормализации для {len(price_df.columns)} символов")
    
    results = {}
    for method in methods:
        norm_df, stats = preprocess_and_normalize_data(
            price_df, 
            norm_method=method,
            fill_method="linear"
        )
        results[method] = (norm_df, stats)
        
        if stats["final_symbols"] > max_symbols:
            max_symbols = stats["final_symbols"]
            best_method = method
    
    logger.info(f"Выбран метод '{best_method}' с сохранением {max_symbols} символов")
    
    # Сравнение методов
    comparison = {method: stats["final_symbols"] for method, (_, stats) in results.items()}
    logger.info(f"Сравнение методов нормализации: {comparison}")
    
    return results[best_method][0]