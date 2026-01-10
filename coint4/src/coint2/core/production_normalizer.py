"""
Production-ready нормализатор для реальной торговли.
Поддерживает только методы, совместимые с real-time обновлениями.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, Union
from collections import deque
import json
import pickle
from pathlib import Path
import logging

logger = logging.getLogger("production_normalizer")


class ProductionNormalizer:
    """
    Нормализатор для production торговли с обновляемыми rolling windows.
    Поддерживает только методы, применимые в реальном времени.
    """
    
    SUPPORTED_METHODS = ['rolling_zscore', 'percent', 'log_returns']
    
    def __init__(self, 
                 method: str = 'rolling_zscore',
                 window: int = 25,
                 epsilon: float = 1e-8):
        """
        Инициализация production нормализатора.
        
        Args:
            method: Метод нормализации ('rolling_zscore', 'percent', 'log_returns')
            window: Размер rolling window для rolling_zscore
            epsilon: Малое число для предотвращения деления на ноль
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Метод {method} не поддерживается в production. "
                           f"Используйте один из: {self.SUPPORTED_METHODS}")
        
        self.method = method
        self.window = window
        self.epsilon = epsilon
        
        # Буферы для каждого символа
        self.buffers: Dict[str, deque] = {}
        
        # Первые значения для percent нормализации
        self.first_values: Dict[str, float] = {}
        
        # Последние значения для log_returns
        self.last_values: Dict[str, float] = {}
        
        # Статистики для мониторинга
        self.update_count = 0
        self.symbols_tracked = set()
        
        logger.info(f"Инициализирован ProductionNormalizer: method={method}, window={window}")
    
    def update(self, symbol: str, price: float) -> None:
        """
        Обновляет буфер новой ценой.
        
        Args:
            symbol: Символ актива
            price: Новая цена
        """
        if symbol not in self.buffers:
            self.buffers[symbol] = deque(maxlen=self.window)
            self.first_values[symbol] = price
            self.symbols_tracked.add(symbol)
            logger.debug(f"Новый символ {symbol} добавлен с первой ценой {price}")
        
        self.buffers[symbol].append(price)
        self.last_values[symbol] = price
        self.update_count += 1
    
    def normalize(self, symbol: str, price: Optional[float] = None) -> Optional[float]:
        """
        Нормализует цену используя текущее состояние.
        
        Args:
            symbol: Символ актива
            price: Цена для нормализации (если None, использует последнюю)
            
        Returns:
            Нормализованное значение или None если недостаточно данных
        """
        if symbol not in self.buffers:
            logger.warning(f"Символ {symbol} не найден в буферах")
            return None
        
        if price is None:
            if symbol not in self.last_values:
                return None
            price = self.last_values[symbol]
        
        if self.method == 'rolling_zscore':
            return self._normalize_rolling_zscore(symbol, price)
        elif self.method == 'percent':
            return self._normalize_percent(symbol, price)
        elif self.method == 'log_returns':
            return self._normalize_log_returns(symbol, price)
        else:
            raise ValueError(f"Неизвестный метод: {self.method}")
    
    def _normalize_rolling_zscore(self, symbol: str, price: float) -> Optional[float]:
        """Rolling z-score нормализация."""
        buffer = self.buffers[symbol]
        
        if len(buffer) < 2:  # Нужно минимум 2 значения для std
            return 0.0
        
        mean = np.mean(buffer)
        std = np.std(buffer)
        
        if std < self.epsilon:
            return 0.0
        
        return (price - mean) / std
    
    def _normalize_percent(self, symbol: str, price: float) -> float:
        """Процентная нормализация относительно первого значения."""
        first_value = self.first_values.get(symbol, price)
        
        if abs(first_value) < self.epsilon:
            return 100.0
        
        return (price / first_value) * 100
    
    def _normalize_log_returns(self, symbol: str, price: float) -> Optional[float]:
        """Логарифмическая доходность."""
        if symbol not in self.last_values:
            return 0.0
        
        last_price = self.last_values[symbol]
        
        if last_price < self.epsilon:
            return 0.0
        
        return np.log(price / last_price)
    
    def normalize_batch(self, prices: Dict[str, float]) -> Dict[str, Optional[float]]:
        """
        Нормализует batch цен для нескольких символов.
        
        Args:
            prices: Словарь {symbol: price}
            
        Returns:
            Словарь {symbol: normalized_value}
        """
        result = {}
        for symbol, price in prices.items():
            # Обновляем буфер
            self.update(symbol, price)
            # Нормализуем
            result[symbol] = self.normalize(symbol, price)
        
        return result
    
    def get_statistics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Возвращает текущие статистики для символа или всех символов.
        
        Args:
            symbol: Символ для статистик (если None, возвращает общие)
            
        Returns:
            Словарь со статистиками
        """
        if symbol:
            if symbol not in self.buffers:
                return {}
            
            buffer = self.buffers[symbol]
            stats = {
                'symbol': symbol,
                'method': self.method,
                'buffer_size': len(buffer),
                'window': self.window,
                'first_value': self.first_values.get(symbol),
                'last_value': self.last_values.get(symbol)
            }
            
            if self.method == 'rolling_zscore' and len(buffer) >= 2:
                stats['rolling_mean'] = float(np.mean(buffer))
                stats['rolling_std'] = float(np.std(buffer))
            
            return stats
        else:
            # Общие статистики
            return {
                'method': self.method,
                'window': self.window,
                'symbols_tracked': len(self.symbols_tracked),
                'total_updates': self.update_count,
                'symbols': list(self.symbols_tracked)
            }
    
    def reset(self, symbol: Optional[str] = None) -> None:
        """
        Сбрасывает состояние для символа или всех символов.
        
        Args:
            symbol: Символ для сброса (если None, сбрасывает все)
        """
        if symbol:
            if symbol in self.buffers:
                del self.buffers[symbol]
            if symbol in self.first_values:
                del self.first_values[symbol]
            if symbol in self.last_values:
                del self.last_values[symbol]
            self.symbols_tracked.discard(symbol)
            logger.info(f"Сброшено состояние для символа {symbol}")
        else:
            self.buffers.clear()
            self.first_values.clear()
            self.last_values.clear()
            self.symbols_tracked.clear()
            self.update_count = 0
            logger.info("Сброшено все состояние нормализатора")
    
    def save_state(self, filepath: Union[str, Path]) -> None:
        """
        Сохраняет состояние нормализатора в файл.
        
        Args:
            filepath: Путь к файлу для сохранения
        """
        state = {
            'method': self.method,
            'window': self.window,
            'epsilon': self.epsilon,
            'buffers': {k: list(v) for k, v in self.buffers.items()},
            'first_values': self.first_values,
            'last_values': self.last_values,
            'update_count': self.update_count,
            'symbols_tracked': list(self.symbols_tracked)
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.suffix == '.json':
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
        
        logger.info(f"Состояние сохранено в {filepath}")
    
    def load_state(self, filepath: Union[str, Path]) -> None:
        """
        Загружает состояние нормализатора из файла.
        
        Args:
            filepath: Путь к файлу состояния
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Файл состояния не найден: {filepath}")
        
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                state = json.load(f)
        else:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
        
        self.method = state['method']
        self.window = state['window']
        self.epsilon = state.get('epsilon', 1e-8)
        self.buffers = {k: deque(v, maxlen=self.window) 
                       for k, v in state['buffers'].items()}
        self.first_values = state['first_values']
        self.last_values = state['last_values']
        self.update_count = state.get('update_count', 0)
        self.symbols_tracked = set(state.get('symbols_tracked', []))
        
        logger.info(f"Состояние загружено из {filepath}")
    
    @classmethod
    def from_training_stats(cls, 
                           normalization_stats: Dict[str, Any],
                           window: Optional[int] = None) -> 'ProductionNormalizer':
        """
        Создает нормализатор из статистик тренировочного периода.
        
        Args:
            normalization_stats: Статистики из preprocess_and_normalize_data
            window: Размер окна (если None, берется из статистик)
            
        Returns:
            Инициализированный ProductionNormalizer
        """
        method = normalization_stats.get('method', 'rolling_zscore')
        
        if window is None:
            window = normalization_stats.get('window', 25)
        
        normalizer = cls(method=method, window=window)
        
        # Инициализируем с сохраненными статистиками
        if method == 'rolling_zscore':
            rolling_mean = normalization_stats.get('rolling_mean', {})
            rolling_std = normalization_stats.get('rolling_std', {})
            
            # Предзаполняем буферы средними значениями
            for symbol in rolling_mean:
                mean_val = rolling_mean[symbol]
                # Заполняем буфер средним значением
                normalizer.buffers[symbol] = deque([mean_val] * window, maxlen=window)
                normalizer.symbols_tracked.add(symbol)
                
        elif method == 'percent':
            first_values = normalization_stats.get('first_values', {})
            normalizer.first_values = first_values.copy()
            normalizer.symbols_tracked = set(first_values.keys())
        
        logger.info(f"Создан ProductionNormalizer из статистик тренировки: "
                   f"{len(normalizer.symbols_tracked)} символов")
        
        return normalizer


def create_production_normalizer(config: Dict[str, Any]) -> ProductionNormalizer:
    """
    Фабричная функция для создания production нормализатора из конфигурации.
    
    Args:
        config: Словарь конфигурации
        
    Returns:
        Настроенный ProductionNormalizer
    """
    # Извлекаем параметры из конфигурации
    method = config.get('normalization_method', 'rolling_zscore')
    
    # Проверяем совместимость с production
    if method not in ProductionNormalizer.SUPPORTED_METHODS:
        logger.warning(f"Метод {method} не поддерживается в production, используем rolling_zscore")
        method = 'rolling_zscore'
    
    window = config.get('rolling_window', 25)
    epsilon = config.get('epsilon', 1e-8)
    
    return ProductionNormalizer(method=method, window=window, epsilon=epsilon)