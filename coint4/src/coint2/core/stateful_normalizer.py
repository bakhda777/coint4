"""
Stateful normalizer that prevents lookahead bias by separating fit and transform.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class StatefulNormalizer:
    """
    Нормализатор с состоянием для предотвращения lookahead bias.
    
    Fit вычисляет статистики только на тренировочных данных,
    Transform применяет сохраненные статистики без пересчета.
    """
    
    def __init__(self, method: str = "rolling_zscore", rolling_window: int = 480):
        """
        Args:
            method: Метод нормализации ('rolling_zscore', 'minmax', 'zscore')
            rolling_window: Окно для rolling статистик
        """
        self.method = method
        self.rolling_window = rolling_window
        self.is_fitted = False
        
        # Сохраняемые статистики
        self.stats: Dict[str, Any] = {}
        
    def fit(self, train_data: pd.DataFrame) -> 'StatefulNormalizer':
        """
        Вычисляет статистики нормализации ТОЛЬКО на тренировочных данных.
        
        Args:
            train_data: Тренировочные данные
            
        Returns:
            self для chain-вызовов
        """
        if train_data.empty:
            raise ValueError("Cannot fit on empty training data")
            
        logger.info(f"Fitting {self.method} normalizer on {train_data.shape} training data")
        
        if self.method == "rolling_zscore":
            # Для rolling_zscore сохраняем последние rolling статистики из train
            if self.rolling_window > len(train_data):
                logger.warning(f"Rolling window {self.rolling_window} > train size {len(train_data)}")
                self.rolling_window = max(20, len(train_data) // 2)
            
            # Сохраняем хвост тренировочных данных для корректного rolling на test
            tail_len = max(1, min(self.rolling_window - 1, len(train_data)))
            self.stats['tail'] = train_data.iloc[-tail_len:].copy()
            
        elif self.method == "minmax":
            # Для minmax сохраняем min и max из train
            self.stats['min'] = train_data.min()
            self.stats['max'] = train_data.max()
            self.stats['range'] = self.stats['max'] - self.stats['min']
            self.stats['range'] = self.stats['range'].replace(0, 1.0)  # Избегаем деления на 0
            
        elif self.method == "zscore":
            # Для zscore сохраняем mean и std из train
            self.stats['mean'] = train_data.mean()
            self.stats['std'] = train_data.std().replace(0, 1e-8)  # Избегаем деления на 0
            
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
            
        self.is_fitted = True
        logger.info(f"✅ Normalizer fitted with {self.method} method")
        return self
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет сохраненные статистики к данным БЕЗ пересчета.
        
        Args:
            data: Данные для трансформации (train или test)
            
        Returns:
            Нормализованные данные
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")
            
        if data.empty:
            return data
            
        logger.debug(f"Transforming {data.shape} data with {self.method}")
        
        if self.method == "rolling_zscore":
            # Rolling без lookahead: используем хвост train + текущий test
            tail = self.stats.get('tail')
            if tail is None or tail.empty:
                tail = data.iloc[:0]
            combined = pd.concat([tail, data])
            rolling_mean = combined.rolling(window=self.rolling_window, min_periods=1).mean()
            rolling_std = combined.rolling(window=self.rolling_window, min_periods=1).std()
            rolling_std = rolling_std.replace(0, 1e-8)
            normalized = (combined - rolling_mean) / rolling_std
            normalized = normalized.iloc[-len(data):]
            
        elif self.method == "minmax":
            # Применяем сохраненные min/max
            normalized = (data - self.stats['min']) / self.stats['range']
            # Клиппинг для test данных, которые могут выходить за границы train
            normalized = normalized.clip(0, 1)
            
        elif self.method == "zscore":
            # Применяем сохраненные mean/std
            normalized = (data - self.stats['mean']) / self.stats['std']
            
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
            
        return normalized
        
    def fit_transform(self, train_data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit и transform на одних данных (для тренировочной выборки).
        
        Args:
            train_data: Тренировочные данные
            
        Returns:
            Нормализованные тренировочные данные
        """
        return self.fit(train_data).transform(train_data)
        
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает сохраненные статистики."""
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted yet")
        return self.stats.copy()


def preprocess_data_no_lookahead(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    min_history_ratio: float = 0.8,
    fill_method: str = "ffill",
    norm_method: str = "rolling_zscore",
    rolling_window: int = 480,
    fill_limit: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Предобработка и нормализация данных БЕЗ lookahead bias.
    
    Args:
        train_data: Тренировочные данные
        test_data: Тестовые данные
        min_history_ratio: Минимальная доля непропущенных значений
        fill_method: Метод заполнения пропусков
        norm_method: Метод нормализации
        rolling_window: Окно для rolling статистик
        fill_limit: Максимальное количество последовательных пропусков для заполнения
        
    Returns:
        Tuple из:
        - Нормализованные тренировочные данные
        - Нормализованные тестовые данные
        - Статистики обработки
    """
    stats = {
        "train_shape": train_data.shape,
        "test_shape": test_data.shape,
        "removed_symbols": [],
        "normalization_stats": {}
    }
    
    # 0. Удаляем нечисловые колонки (timestamp, symbol и т.д.)
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns
    train_data = train_data[numeric_cols]
    test_data = test_data[numeric_cols]
    
    # 1. Фильтрация символов по истории ТОЛЬКО на train
    history_ratios = train_data.notna().mean()
    valid_symbols = history_ratios[history_ratios >= min_history_ratio].index
    
    # Удаляем символы с недостаточной историей
    removed = set(train_data.columns) - set(valid_symbols)
    if removed:
        logger.info(f"Removing {len(removed)} symbols with insufficient history in train")
        stats["removed_symbols"] = list(removed)
    
    # Если все символы отфильтрованы, возвращаем пустые DataFrame
    if len(valid_symbols) == 0:
        logger.warning("All symbols filtered out due to insufficient history")
        return train_data.iloc[:0], test_data.iloc[:0], stats
    
    train_filtered = train_data[valid_symbols]
    test_filtered = test_data[valid_symbols]
    
    # 2. Заполнение пропусков (ограниченное, чтобы не переносить данные через границы)
    if fill_method == "ffill":
        train_filled = train_filtered.fillna(method='ffill', limit=fill_limit)
        test_filled = test_filtered.fillna(method='ffill', limit=fill_limit)
    elif fill_method == "linear":
        train_filled = train_filtered.interpolate(method='linear', limit=fill_limit)
        # В тесте избегаем lookahead: только ffill
        test_filled = test_filtered.fillna(method='ffill', limit=fill_limit)
    else:
        train_filled = train_filtered
        test_filled = test_filtered
    
    # 3. Нормализация БЕЗ lookahead
    normalizer = StatefulNormalizer(method=norm_method, rolling_window=rolling_window)
    
    # Fit только на train
    train_normalized = normalizer.fit_transform(train_filled)
    
    # Transform test с сохраненными статистиками
    test_normalized = normalizer.transform(test_filled)
    
    # Сохраняем статистики
    stats["normalization_stats"] = normalizer.get_stats()
    stats["normalization_stats"]["method"] = norm_method
    stats["normalization_stats"]["rolling_window"] = rolling_window
    
    logger.info(f"✅ Preprocessing complete: train {train_normalized.shape}, test {test_normalized.shape}")
    
    return train_normalized, test_normalized, stats
