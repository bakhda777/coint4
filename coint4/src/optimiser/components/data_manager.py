"""
Менеджер данных для оптимизации.
Отвечает за загрузку и подготовку данных для walk-forward анализа.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
from pathlib import Path
import logging
from dataclasses import dataclass

from coint2.core.data_prep import prepare_walk_forward_slices, validate_no_lookahead
from ..lookahead_validator import LookaheadValidator

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardData:
    """Структура данных для walk-forward шага."""
    full_data: pd.DataFrame
    training_data: pd.DataFrame
    testing_data: pd.DataFrame
    training_start: pd.Timestamp
    training_end: pd.Timestamp
    testing_start: pd.Timestamp
    testing_end: pd.Timestamp
    step_index: int


class OptimizationDataManager:
    """
    Менеджер данных для оптимизации.
    Инкапсулирует всю логику загрузки и подготовки данных.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Конфигурация с параметрами данных и walk-forward
        """
        self.config = config
        self.data_dir = config.get('data_dir', 'data_downloaded')
        self.walk_forward_config = config.get('walk_forward', {})
        
        # Инициализируем валидатор lookahead bias
        self.lookahead_validator = LookaheadValidator(strict_mode=True)
        
        # Кэш загруженных данных
        self._data_cache = {}
        
    def load_walk_forward_data(
        self, 
        training_start: pd.Timestamp,
        training_end: pd.Timestamp,
        testing_start: pd.Timestamp,
        testing_end: pd.Timestamp,
        step_index: int = 0
    ) -> WalkForwardData:
        """
        Загружает данные для конкретного walk-forward шага.
        
        Args:
            training_start: Начало тренировочного периода
            training_end: Конец тренировочного периода
            testing_start: Начало тестового периода
            testing_end: Конец тестового периода
            step_index: Индекс walk-forward шага
            
        Returns:
            WalkForwardData с загруженными и разделенными данными
            
        Raises:
            ValueError: При обнаружении lookahead bias
        """
        logger.info(f"📈 Загрузка данных для walk-forward шага {step_index}")
        logger.info(f"   Тренировка: {training_start.date()} -> {training_end.date()}")
        logger.info(f"   Тестирование: {testing_start.date()} -> {testing_end.date()}")
        
        # Создаем ключ кэша
        cache_key = f"{training_start}_{training_end}_{testing_start}_{testing_end}"
        
        # Проверяем кэш
        if cache_key in self._data_cache:
            logger.info("✅ Данные найдены в кэше")
            return self._data_cache[cache_key]
        
        try:
            # Используем единую функцию подготовки данных
            training_slice, testing_slice, step_df, stats = prepare_walk_forward_slices(
                training_start=training_start,
                training_end=training_end,
                testing_start=testing_start,
                testing_end=testing_end,
                config=self.config,
                data_dir=self.data_dir
            )
            
            # Валидация на lookahead bias (только если данные не пустые)
            if not training_slice.empty and not testing_slice.empty:
                self._validate_data_split(training_slice, testing_slice)
            
            # Создаем структуру данных
            walk_forward_data = WalkForwardData(
                full_data=step_df,
                training_data=training_slice,
                testing_data=testing_slice,
                training_start=training_start,
                training_end=training_end,
                testing_start=testing_start,
                testing_end=testing_end,
                step_index=step_index
            )
            
            # Сохраняем в кэш
            self._data_cache[cache_key] = walk_forward_data
            
            logger.info(f"✅ Данные загружены БЕЗ lookahead bias:")
            logger.info(f"   Тренировочный срез: {training_slice.shape}")
            logger.info(f"   Тестовый срез: {testing_slice.shape}")
            logger.info(f"   Метод нормализации: {stats.get('normalization_method')}")
            logger.info(f"   Символов удалено при фильтрации: {stats.get('symbols_removed', 0)}")
            
            return walk_forward_data
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки данных: {e}")
            raise
    
    def _validate_data_split(
        self, 
        training_data: pd.DataFrame,
        testing_data: pd.DataFrame
    ) -> None:
        """
        Валидирует разделение данных на train/test.
        
        Args:
            training_data: Тренировочные данные
            testing_data: Тестовые данные
            
        Raises:
            ValueError: При обнаружении lookahead bias или других проблем
        """
        if training_data.empty or testing_data.empty:
            raise ValueError("Один из датасетов пуст")
        
        train_end = training_data.index.max()
        test_start = testing_data.index.min()
        
        # Проверка на перекрытие
        if train_end >= test_start:
            raise ValueError(
                f"КРИТИЧНО: Перекрытие данных! "
                f"Train заканчивается {train_end}, Test начинается {test_start}"
            )
        
        # Проверка минимального gap
        gap = test_start - train_end
        gap_minutes = self.walk_forward_config.get('gap_minutes', 15)
        min_gap = pd.Timedelta(minutes=gap_minutes)
        
        if gap < min_gap:
            raise ValueError(
                f"Недостаточный gap между train и test: {gap} < {min_gap}"
            )
        
        # Дополнительная валидация через lookahead_validator
        gap_days = gap_minutes / (24 * 60)
        is_valid, message = self.lookahead_validator.validate_data_split(
            training_data, testing_data, gap_days
        )
        
        if not is_valid:
            raise ValueError(f"Lookahead validator: {message}")
    
    def get_walk_forward_periods(self) -> list:
        """
        Генерирует периоды для walk-forward анализа.
        
        Returns:
            Список кортежей (training_start, training_end, testing_start, testing_end)
        """
        start_date = pd.to_datetime(self.walk_forward_config['start_date'])
        end_date = pd.to_datetime(self.walk_forward_config['end_date'])
        training_days = self.walk_forward_config['training_period_days']
        testing_days = self.walk_forward_config['testing_period_days']
        step_days = self.walk_forward_config['step_size_days']
        gap_minutes = self.walk_forward_config.get('gap_minutes', 15)
        
        periods = []
        current_start = start_date
        
        while current_start + pd.Timedelta(days=training_days + testing_days) <= end_date:
            training_start = current_start
            training_end = training_start + pd.Timedelta(days=training_days)
            testing_start = training_end + pd.Timedelta(minutes=gap_minutes)
            testing_end = testing_start + pd.Timedelta(days=testing_days)
            
            periods.append((training_start, training_end, testing_start, testing_end))
            
            current_start += pd.Timedelta(days=step_days)
        
        return periods
    
    def clear_cache(self) -> None:
        """Очищает кэш данных."""
        self._data_cache.clear()
        logger.info("🗑️ Кэш данных очищен")
    
    def get_cache_size(self) -> int:
        """Возвращает размер кэша."""
        return len(self._data_cache)
