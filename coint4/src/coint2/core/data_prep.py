"""
Единый модуль подготовки данных для walk-forward анализа.
Гарантирует отсутствие lookahead bias во всех путях выполнения.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging
from pathlib import Path

from .data_loader import load_master_dataset, resolve_data_filters
from .stateful_normalizer import preprocess_data_no_lookahead

logger = logging.getLogger(__name__)


def prepare_walk_forward_slices(
    training_start: pd.Timestamp,
    training_end: pd.Timestamp,
    testing_start: pd.Timestamp,
    testing_end: pd.Timestamp,
    config: Dict[str, Any],
    data_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Единая функция подготовки данных для walk-forward шага.
    
    Гарантии:
    - Нормализация выполняется ПОСЛЕ разделения на train/test
    - Фильтрация символов только по train данным
    - Нет lookahead bias ни в одном пути выполнения
    
    Args:
        training_start: Начало тренировочного периода
        training_end: Конец тренировочного периода
        testing_start: Начало тестового периода
        testing_end: Конец тестового периода
        config: Конфигурация с параметрами обработки
        data_dir: Директория с данными (если None, берется из config)
        
    Returns:
        Tuple из:
        - training_data: Нормализованные тренировочные данные
        - testing_data: Нормализованные тестовые данные
        - full_data: Объединенные данные (train + test)
        - stats: Статистика обработки
    """
    stats = {
        "training_period": f"{training_start.date()} -> {training_end.date()}",
        "testing_period": f"{testing_start.date()} -> {testing_end.date()}",
        "raw_shape": None,
        "training_shape": None,
        "testing_shape": None,
        "normalization_method": None,
        "symbols_removed": 0
    }
    
    # Валидация временных границ
    if testing_start <= training_end:
        raise ValueError(
            f"Lookahead bias detected! Testing starts before training ends: "
            f"training_end={training_end}, testing_start={testing_start}"
        )
    
    gap = testing_start - training_end
    logger.info(f"Gap между train и test: {gap}")
    
    try:
        # 1. Загружаем сырые данные за весь период
        if data_dir is None:
            data_dir = config.get('data_dir', 'data_downloaded')
        
        clean_window, excluded_symbols = resolve_data_filters(config)
        raw_data = load_master_dataset(
            data_path=data_dir,
            start_date=training_start,
            end_date=testing_end,
            clean_window=clean_window,
            exclude_symbols=excluded_symbols,
        )
        
        stats["raw_shape"] = raw_data.shape
        logger.info(f"Загружено сырых данных: {raw_data.shape}")
        
        # 2. Преобразуем в широкий формат если необходимо
        if 'timestamp' in raw_data.columns and 'symbol' in raw_data.columns and 'close' in raw_data.columns:
            logger.info("Преобразование из длинного формата в широкий...")
            raw_data = raw_data.pivot(
                index='timestamp',
                columns='symbol',
                values='close'
            )
        
        # 3. Убеждаемся что индекс - datetime
        if not isinstance(raw_data.index, pd.DatetimeIndex):
            raw_data.index = pd.to_datetime(raw_data.index, errors="coerce")
            if getattr(raw_data.index, "tz", None) is not None:
                raw_data.index = raw_data.index.tz_localize(None)
            raw_data = raw_data.sort_index()
        
        # 4. КРИТИЧНО: Разделяем СЫРЫЕ данные до любой обработки
        training_raw = raw_data.loc[training_start:training_end]
        testing_raw = raw_data.loc[testing_start:testing_end]
        
        # 5. Получаем параметры обработки из конфига
        data_processing = config.get('data_processing', {})
        norm_method = data_processing.get('normalization_method', 'rolling_zscore')
        fill_method = data_processing.get('fill_method', 'ffill')
        min_history_ratio = data_processing.get('min_history_ratio', 0.8)
        
        # Rolling window из backtest секции
        backtest_config = config.get('backtest', {})
        rolling_window = backtest_config.get('rolling_window', 480)
        
        stats["normalization_method"] = norm_method
        
        # 6. Нормализация БЕЗ lookahead bias
        training_data, testing_data, norm_stats = preprocess_data_no_lookahead(
            train_data=training_raw,
            test_data=testing_raw,
            min_history_ratio=min_history_ratio,
            fill_method=fill_method,
            norm_method=norm_method,
            rolling_window=rolling_window
        )
        
        stats["training_shape"] = training_data.shape
        stats["testing_shape"] = testing_data.shape
        stats["symbols_removed"] = len(norm_stats.get('removed_symbols', []))
        
        # 7. Проверка на пустые данные
        if training_data.empty or testing_data.empty:
            logger.warning("Данные пусты после первой попытки нормализации")
            
            # Пробуем с ослабленными параметрами
            relaxed_min_history = min_history_ratio * 0.5
            relaxed_rolling_window = min(rolling_window, len(training_raw) // 2) if len(training_raw) > 0 else 20
            
            logger.info(f"Повторная попытка с ослабленными параметрами: "
                       f"min_history={relaxed_min_history:.2f}, "
                       f"rolling_window={relaxed_rolling_window}")
            
            training_data, testing_data, norm_stats = preprocess_data_no_lookahead(
                train_data=training_raw,
                test_data=testing_raw,
                min_history_ratio=relaxed_min_history,
                fill_method=fill_method,
                norm_method=norm_method,
                rolling_window=relaxed_rolling_window
            )
            
            stats["training_shape"] = training_data.shape
            stats["testing_shape"] = testing_data.shape
            stats["relaxed_params_used"] = True
        
        # 8. Объединяем для полного датасета
        if not training_data.empty and not testing_data.empty:
            full_data = pd.concat([training_data, testing_data])
        else:
            full_data = pd.DataFrame()
            logger.error("Не удалось подготовить данные даже с ослабленными параметрами")
        
        logger.info(f"✅ Данные подготовлены: train {training_data.shape}, test {testing_data.shape}")
        
        return training_data, testing_data, full_data, stats
        
    except Exception as e:
        logger.error(f"Ошибка подготовки данных: {e}")
        # Возвращаем пустые DataFrame в случае ошибки
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), stats


def validate_no_lookahead(
    training_data: pd.DataFrame,
    testing_data: pd.DataFrame,
    training_end: pd.Timestamp,
    testing_start: pd.Timestamp
) -> bool:
    """
    Валидирует что нет lookahead bias в подготовленных данных.
    
    Args:
        training_data: Тренировочные данные
        testing_data: Тестовые данные
        training_end: Конец тренировочного периода
        testing_start: Начало тестового периода
        
    Returns:
        True если lookahead bias не обнаружен
    """
    # Проверка временных границ
    if not training_data.empty:
        actual_train_end = training_data.index.max()
        if actual_train_end > training_end:
            logger.error(f"Training данные выходят за границу: {actual_train_end} > {training_end}")
            return False
    
    if not testing_data.empty:
        actual_test_start = testing_data.index.min()
        if actual_test_start < testing_start:
            logger.error(f"Testing данные начинаются раньше: {actual_test_start} < {testing_start}")
            return False
    
    # Проверка пересечения индексов
    if not training_data.empty and not testing_data.empty:
        train_indices = set(training_data.index)
        test_indices = set(testing_data.index)
        overlap = train_indices.intersection(test_indices)
        
        if overlap:
            logger.error(f"Обнаружено пересечение train и test: {len(overlap)} точек")
            return False
    
    return True
