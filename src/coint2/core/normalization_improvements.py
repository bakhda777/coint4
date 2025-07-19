"""
Улучшенные методы нормализации для снижения потерь символов.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
import logging

logger = logging.getLogger("normalization")

def preprocess_and_normalize_data(
    price_df: pd.DataFrame,
    min_history_ratio: float = 0.8,
    fill_method: str = "ffill",
    norm_method: str = "minmax",
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
    
    if filtered_df.shape[1] < 2:
        logger.warning(f"Осталось менее 2 символов после удаления символов с постоянной ценой")
        stats["final_symbols"] = filtered_df.shape[1]
        return filtered_df, stats
    
    # Этап 3: Заполнение пропусков
    if fill_method == "ffill":
        limit = 5  # Максимальное число подряд идущих пропусков для заполнения
        filled_df = filtered_df.ffill(limit=limit).bfill(limit=limit)
    elif fill_method == "linear":
        limit = 5
        filled_df = (
            filtered_df.interpolate(method='linear', limit=limit)
            .ffill(limit=limit)
            .bfill(limit=limit)
        )
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