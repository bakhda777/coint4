"""
Модуль для валидации и предотвращения lookahead bias в процессе оптимизации.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class LookaheadValidator:
    """Класс для проверки и предотвращения lookahead bias."""
    
    def __init__(self, strict_mode: bool = True):
        """
        Args:
            strict_mode: Строгий режим проверки (рекомендуется True)
        """
        self.strict_mode = strict_mode
        self.violations = []
        
    def validate_data_split(
        self, 
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        gap_days: int = 1
    ) -> Tuple[bool, str]:
        """
        Проверяет корректность разделения данных на train/test.
        
        Args:
            train_data: Тренировочные данные
            test_data: Тестовые данные
            gap_days: Минимальный gap между периодами в днях
            
        Returns:
            (is_valid, message): Результат валидации и сообщение
        """
        if train_data is None or test_data is None:
            return True, "Один из датасетов None"
        if len(train_data) == 0 or len(test_data) == 0:
            return True, "Один из датасетов пуст"
            
        train_end = train_data.index.max()
        test_start = test_data.index.min()
        
        # Проверка на перекрытие
        if train_end >= test_start:
            violation = f"КРИТИЧНО: Перекрытие данных! Train заканчивается {train_end}, Test начинается {test_start}"
            self.violations.append(violation)
            return False, violation
            
        # Проверка минимального gap
        actual_gap = test_start - train_end
        min_gap = pd.Timedelta(days=gap_days)
        
        if actual_gap < min_gap:
            violation = f"ПРЕДУПРЕЖДЕНИЕ: Недостаточный gap {actual_gap} < {min_gap}"
            self.violations.append(violation)
            if self.strict_mode:
                return False, violation
            logger.warning(violation)
            
        return True, f"Данные корректно разделены с gap {actual_gap}"
        
    def validate_normalization(
        self,
        params: Dict[str, Any],
        train_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> Tuple[bool, str]:
        """
        Проверяет корректность нормализации без lookahead bias.
        
        Args:
            params: Параметры нормализации
            train_data: Тренировочные данные
            test_data: Тестовые данные
            
        Returns:
            (is_valid, message): Результат валидации
        """
        # Проверяем, что параметры нормализации вычислены только на train
        if 'normalization_params' in params:
            norm_params = params['normalization_params']
            
            # Проверка временных меток
            if 'computed_from' in norm_params and 'computed_to' in norm_params:
                computed_from = pd.to_datetime(norm_params['computed_from'])
                computed_to = pd.to_datetime(norm_params['computed_to'])
                
                # Параметры должны быть вычислены только на train периоде
                if computed_to > train_data.index.max():
                    violation = f"КРИТИЧНО: Нормализация использует данные после train периода"
                    self.violations.append(violation)
                    return False, violation
                    
        return True, "Нормализация корректна"
        
    def validate_rolling_windows(
        self,
        window_size: int,
        train_start: pd.Timestamp,
        test_start: pd.Timestamp,
        data_frequency: str = '15min'
    ) -> Tuple[bool, str]:
        """
        Проверяет корректность rolling window для избежания lookahead.
        
        Args:
            window_size: Размер окна в периодах
            train_start: Начало тренировочного периода
            test_start: Начало тестового периода
            data_frequency: Частота данных
            
        Returns:
            (is_valid, message): Результат валидации
        """
        # Конвертируем window_size в временной период
        freq_minutes = {
            '15min': 15,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }.get(data_frequency, 15)
        
        window_duration = pd.Timedelta(minutes=window_size * freq_minutes)
        
        # Проверяем, что окно не выходит за пределы train при инициализации
        min_required_history = train_start - window_duration
        
        if min_required_history < train_start - pd.Timedelta(days=365):
            violation = f"ПРЕДУПРЕЖДЕНИЕ: Слишком большое окно {window_size} периодов требует данных за {window_duration}"
            if self.strict_mode:
                self.violations.append(violation)
                return False, violation
                
        return True, f"Rolling window {window_size} корректен"
        
    def validate_cache_isolation(
        self,
        cache_key: str,
        train_period: Tuple[pd.Timestamp, pd.Timestamp],
        test_period: Tuple[pd.Timestamp, pd.Timestamp]
    ) -> Tuple[bool, str]:
        """
        Проверяет изоляцию кэша между периодами.
        
        Args:
            cache_key: Ключ кэша
            train_period: (start, end) тренировочного периода
            test_period: (start, end) тестового периода
            
        Returns:
            (is_valid, message): Результат валидации
        """
        # Проверяем, что ключ кэша содержит информацию о периоде
        if 'all_periods' in cache_key or 'global' in cache_key:
            violation = f"ПРЕДУПРЕЖДЕНИЕ: Кэш-ключ '{cache_key}' может содержать данные из разных периодов"
            self.violations.append(violation)
            if self.strict_mode:
                return False, violation
                
        # Проверяем временные метки в ключе
        train_start_str = train_period[0].strftime('%Y%m%d')
        train_end_str = train_period[1].strftime('%Y%m%d')
        
        if train_start_str not in cache_key or train_end_str not in cache_key:
            violation = f"ПРЕДУПРЕЖДЕНИЕ: Кэш-ключ должен содержать границы периода"
            if self.strict_mode:
                self.violations.append(violation)
                return False, violation
                
        return True, "Кэш корректно изолирован"
        
    def validate_pair_selection(
        self,
        selected_pairs: list,
        selection_data: pd.DataFrame,
        test_start: pd.Timestamp
    ) -> Tuple[bool, str]:
        """
        Проверяет, что отбор пар выполнен только на исторических данных.
        
        Args:
            selected_pairs: Отобранные пары
            selection_data: Данные для отбора
            test_start: Начало тестового периода
            
        Returns:
            (is_valid, message): Результат валидации
        """
        if selection_data.empty:
            return True, "Нет данных для проверки"
            
        # Проверяем, что данные для отбора не содержат будущих значений
        if selection_data.index.max() >= test_start:
            violation = f"КРИТИЧНО: Отбор пар использует данные после {test_start}"
            self.violations.append(violation)
            return False, violation
            
        return True, f"Отбор пар корректен, использованы данные до {selection_data.index.max()}"
        
    def get_violations_report(self) -> str:
        """
        Возвращает отчет о всех нарушениях.
        
        Returns:
            Строка с отчетом
        """
        if not self.violations:
            return "✅ Нарушений lookahead bias не обнаружено"
            
        report = "❌ Обнаружены нарушения lookahead bias:\n"
        for i, violation in enumerate(self.violations, 1):
            report += f"{i}. {violation}\n"
            
        return report
        
    def reset(self):
        """Очищает список нарушений."""
        self.violations = []


def create_temporal_validator(config: Dict[str, Any]) -> LookaheadValidator:
    """
    Создает валидатор с настройками из конфигурации.
    
    Args:
        config: Конфигурация с параметрами валидации
        
    Returns:
        Настроенный валидатор
    """
    # Проверяем тип конфигурации
    if hasattr(config, 'get'):
        # Словарь
        strict_mode = config.get('strict_lookahead_validation', True)
    else:
        # Pydantic модель - проверяем наличие атрибута
        strict_mode = getattr(config, 'strict_lookahead_validation', True)
    validator = LookaheadValidator(strict_mode=strict_mode)
    
    logger.info(f"Создан LookaheadValidator в {'строгом' if strict_mode else 'мягком'} режиме")
    
    return validator