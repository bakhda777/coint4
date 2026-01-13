"""
Улучшенные методы нормализации для снижения потерь символов.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Any
import logging
import warnings

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
    norm_method: str = "minmax",
    handle_constant: bool = True,
    epsilon: float = 1e-8,
    rolling_window: Optional[int] = None,
    return_stats: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Улучшенная предобработка и нормализация данных цен.
    
    Args:
        price_df: DataFrame с ценами, индекс - timestamp, колонки - символы
        min_history_ratio: Минимальная доля непропущенных значений
        fill_method: Метод заполнения пропусков ('ffill', 'linear', None)
        norm_method: Метод нормализации ('minmax', 'log_returns', 'rolling_zscore', 'percent')
        handle_constant: Обрабатывать ли символы с постоянной ценой
        epsilon: Малое число для предотвращения деления на ноль
        rolling_window: Размер окна для rolling статистик (обязателен для rolling_zscore)
        return_stats: Возвращать ли статистики нормализации для production
        
    Returns:
        Tuple из:
        - Нормализованный DataFrame
        - Dict со статистикой обработки (включая rolling stats если return_stats=True)
    """
    stats = {
        "initial_symbols": len(price_df.columns),
        "low_history_ratio": 0,
        "constant_price": 0,
        "nan_after_norm": 0,
        "final_symbols": 0
    }

    try:
        min_history_ratio = float(min_history_ratio)
    except (TypeError, ValueError):
        logger.warning("Некорректный min_history_ratio, используем 0.8")
        min_history_ratio = 0.8
    min_history_ratio = max(0.0, min(1.0, min_history_ratio))
    
    # Проверка на пустые данные
    if price_df.empty or price_df.shape[0] == 0 or price_df.shape[1] == 0:
        logger.warning("Получен пустой DataFrame для нормализации")
        stats["final_symbols"] = 0
        return pd.DataFrame(), stats
    
    # Этап 0: Удаляем нечисловые колонки
    numeric_cols = price_df.select_dtypes(include=[np.number]).columns
    price_df = price_df[numeric_cols]
    stats["initial_symbols"] = len(price_df.columns)
    
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
    
    # Заполнение пропусков только внутри торговых сессий
    if fill_method == "ffill":
        limit = 5  # Максимальное число подряд идущих пропусков для заполнения
        filled_df = _fill_gaps_session_aware(filtered_df, method="ffill", limit=limit)
    elif fill_method == "linear":
        limit = 5
        filled_df = _fill_gaps_session_aware(filtered_df, method="linear", limit=limit)
    else:
        filled_df = filtered_df
    
    # Этап 4: Нормализация
    normalization_stats = {}
    
    # ВАЖНО: Для production используем только rolling_zscore или percent методы
    production_compatible_methods = ['rolling_zscore', 'percent', 'log_returns']
    
    if norm_method == "rolling_zscore":
        # Rolling Z-score нормализация (production-ready)
        if rolling_window is None:
            raise ValueError("rolling_window обязателен для метода rolling_zscore")
        
        # Вычисляем rolling статистики
        rolling_mean = filled_df.rolling(window=rolling_window, min_periods=1).mean()
        rolling_std = filled_df.rolling(window=rolling_window, min_periods=1).std()
        
        # Защита от деления на ноль
        rolling_std = rolling_std.replace(0, epsilon)
        
        # Нормализация
        normalized_df = (filled_df - rolling_mean) / rolling_std
        
        # Сохраняем последние статистики для production
        if return_stats:
            normalization_stats['method'] = 'rolling_zscore'
            normalization_stats['window'] = rolling_window
            normalization_stats['rolling_mean'] = rolling_mean.iloc[-1].to_dict()
            normalization_stats['rolling_std'] = rolling_std.iloc[-1].to_dict()
    
    elif norm_method == "percent":
        # Процентная нормализация относительно первого значения (production-ready)
        first_values = filled_df.iloc[0]
        
        # Защита от деления на ноль
        first_values = first_values.replace(0, epsilon)
        
        normalized_df = (filled_df / first_values) * 100
        
        # Сохраняем первые значения для production
        if return_stats:
            normalization_stats['method'] = 'percent'
            normalization_stats['first_values'] = first_values.to_dict()
    
    elif norm_method == "minmax":
        # Min-max нормализация (НЕ для production!)
        message = (
            "ВНИМАНИЕ: minmax нормализация НЕ совместима с production! "
            "Используйте rolling_zscore или percent."
        )
        logger.warning(message)
        warnings.warn(message)
        
        data_min = filled_df.min()
        data_max = filled_df.max()
        data_range = data_max - data_min
        
        # Защита от деления на ноль
        data_range = data_range.replace(0, epsilon)
        
        normalized_df = (filled_df - data_min) / data_range
        
        if return_stats:
            normalization_stats['method'] = 'minmax'
            normalization_stats['min'] = data_min.to_dict()
            normalization_stats['max'] = data_max.to_dict()
        
    elif norm_method == "log_returns":
        # Логарифмические доходности (production-ready)
        normalized_df = np.log(filled_df / filled_df.shift(1))
        normalized_df = normalized_df.iloc[1:]  # Удаляем первую строку с NaN
        
        if return_stats:
            normalization_stats['method'] = 'log_returns'

    elif norm_method == "zscore":
        # Global Z-score (НЕ для production!)
        logger.warning("ВНИМАНИЕ: global zscore НЕ совместим с production! Используйте rolling_zscore.")
        
        data_mean = filled_df.mean()
        data_std = filled_df.std()

        # Защита от деления на ноль для колонок с нулевой вариацией
        data_std = data_std.replace(0, epsilon)

        normalized_df = (filled_df - data_mean) / data_std
        
        if return_stats:
            normalization_stats['method'] = 'zscore'
            normalization_stats['mean'] = data_mean.to_dict()
            normalization_stats['std'] = data_std.to_dict()

    else:
        raise ValueError(f"Неизвестный метод нормализации: {norm_method}")
    
    # Проверка на NaN после нормализации
    # Не удаляем колонки с NaN в первых строках от rolling расчетов
    # Вместо этого удаляем только первые строки с NaN (обычно первые rolling_window строк)
    nan_mask = normalized_df.isna().any()
    nan_columns = normalized_df.columns[nan_mask].tolist()
    stats["nan_after_norm"] = len(nan_columns)
    
    if nan_columns:
        logger.info(f"Символы с NaN после нормализации: {len(nan_columns)}")
        if len(nan_columns) <= 10:
            logger.debug(f"Список: {', '.join(nan_columns)}")

        # Это сохранит все символы, удалив только первые несколько строк без полных данных
        
        # Более надежный подход: просто удаляем все строки с любыми NaN
        # Это обычно только первые несколько строк от rolling расчетов
        initial_rows = len(normalized_df)
        normalized_df = normalized_df.dropna()  # Удаляем все строки с любыми NaN
        dropped_rows = initial_rows - len(normalized_df)
        
        if dropped_rows > 0:
            logger.info(f"Удалено {dropped_rows} начальных строк с NaN от rolling расчетов")
        
        # Если после этого остались колонки полностью с NaN (не должно быть), удаляем их
        if normalized_df.empty:
            logger.warning("После удаления NaN строк DataFrame пуст!")
        else:
            columns_all_nan = normalized_df.isna().all()
            if columns_all_nan.any():
                normalized_df = normalized_df.loc[:, ~columns_all_nan]
                logger.warning(f"Удалено {columns_all_nan.sum()} колонок с постоянными NaN")
    
    stats["final_symbols"] = normalized_df.shape[1]
    logger.info(f"Итоговое количество символов после нормализации: {stats['final_symbols']} из {stats['initial_symbols']}")
    
    # Добавляем статистики нормализации если запрошено
    if return_stats and normalization_stats:
        stats['normalization_stats'] = normalization_stats
    
    return normalized_df, stats

def apply_production_normalization(
    price_df: pd.DataFrame,
    normalization_stats: Dict[str, Any],
    epsilon: float = 1e-8
) -> pd.DataFrame:
    """
    Применяет нормализацию к новым данным используя сохраненные статистики.
    Для использования в production и тестировании.
    
    Args:
        price_df: DataFrame с ценами для нормализации
        normalization_stats: Словарь со статистиками из preprocess_and_normalize_data
        epsilon: Малое число для предотвращения деления на ноль
        
    Returns:
        Нормализованный DataFrame
    """
    if not normalization_stats:
        raise ValueError("normalization_stats не может быть пустым")
    
    method = normalization_stats.get('method')
    if not method:
        raise ValueError("Метод нормализации не указан в normalization_stats")
    
    if method == 'rolling_zscore':
        # Применяем rolling z-score с сохраненными статистиками
        rolling_mean = pd.Series(normalization_stats['rolling_mean'])
        rolling_std = pd.Series(normalization_stats['rolling_std'])
        
        # Защита от деления на ноль
        rolling_std = rolling_std.replace(0, epsilon)
        
        # Нормализация каждого символа
        normalized_df = pd.DataFrame(index=price_df.index)
        for symbol in price_df.columns:
            if symbol in rolling_mean.index and symbol in rolling_std.index:
                normalized_df[symbol] = (price_df[symbol] - rolling_mean[symbol]) / rolling_std[symbol]
            else:
                logger.warning(f"Символ {symbol} отсутствует в статистиках нормализации")
                
    elif method == 'percent':
        # Процентная нормализация с сохраненными первыми значениями
        first_values = pd.Series(normalization_stats['first_values'])
        
        # Защита от деления на ноль
        first_values = first_values.replace(0, epsilon)
        
        # Нормализация каждого символа
        normalized_df = pd.DataFrame(index=price_df.index)
        for symbol in price_df.columns:
            if symbol in first_values.index:
                normalized_df[symbol] = (price_df[symbol] / first_values[symbol]) * 100
            else:
                logger.warning(f"Символ {symbol} отсутствует в статистиках нормализации")
                
    elif method == 'log_returns':
        # Лог-доходности вычисляются динамически
        normalized_df = np.log(price_df / price_df.shift(1))
        normalized_df = normalized_df.iloc[1:]
        
    else:
        raise ValueError(f"Метод {method} не поддерживается для production нормализации")
    
    return normalized_df

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

    # Заполнение пропусков только внутри торговых сессий
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
    
    elif norm_method == "percent":
        # Процентная нормализация относительно первого значения
        normalized_df = filled_df.copy()
        
        for symbol in filled_df.columns:
            if symbol in normalization_params:
                params = normalization_params[symbol]
                first_value = params.get('first_value', 1.0)
                
                # Защита от деления на ноль
                if abs(first_value) < epsilon:
                    first_value = epsilon
                
                normalized_df[symbol] = (filled_df[symbol] / first_value) * 100
            else:
                # Используем первое значение из данных
                first_value = filled_df[symbol].iloc[0] if len(filled_df) > 0 else 1.0
                if abs(first_value) < epsilon:
                    first_value = epsilon
                normalized_df[symbol] = (filled_df[symbol] / first_value) * 100
    
    elif norm_method == "rolling_zscore":
        # Rolling z-score нормализация
        normalized_df = filled_df.copy()
        
        for symbol in filled_df.columns:
            # Получаем окно для конкретного символа или используем глобальное
            if symbol in normalization_params:
                window = normalization_params[symbol].get('window', 20)
            else:
                # Fallback на глобальное окно если есть
                window = normalization_params.get('window', 20) if isinstance(normalization_params.get('window'), int) else 20
            
            rolling_mean = filled_df[symbol].rolling(window=window, min_periods=1).mean()
            rolling_std = filled_df[symbol].rolling(window=window, min_periods=1).std()
            
            # Защита от деления на ноль
            rolling_std = rolling_std.fillna(epsilon)
            rolling_std[rolling_std < epsilon] = epsilon
            
            # Защита от None значений
            if rolling_mean is None or rolling_std is None:
                logger.warning(f"None значения для символа {symbol}, пропускаем")
                normalized_df = normalized_df.drop(columns=[symbol])
                continue
                
            normalized_df[symbol] = (filled_df[symbol] - rolling_mean) / rolling_std

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
    elif norm_method == "percent":
        # Для percent нужно первое значение
        for symbol in price_df.columns:
            series = price_df[symbol].dropna()
            if len(series) > 0:
                params[symbol] = {
                    'first_value': float(series.iloc[0])
                }
    elif norm_method == "rolling_zscore":
        # Для rolling_zscore нужно окно для каждого символа
        for symbol in price_df.columns:
            params[symbol] = {'window': 20}  # По умолчанию окно 20

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
