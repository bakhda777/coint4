#!/usr/bin/env python
"""
Скрипт для тестирования улучшенных методов нормализации данных.
Сравнивает текущий метод с новыми подходами по сохранению символов.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Добавляем корневой каталог проекта в sys.path для импорта модулей
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.coint2.core.normalization_improvements import (
    preprocess_and_normalize_data,
    adaptive_normalization
)

# Настройка логирования
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("normalization_test")

def current_normalization(df):
    """Текущая реализация нормализации в системе"""
    # Удаление NaN значений
    df = df.dropna(axis=1)
    
    # Min-max нормализация
    normalized_df = (df - df.min()) / (df.max() - df.min())
    
    # Удаление колонок с NaN (которые могли появиться при делении на ноль)
    normalized_df = normalized_df.dropna(axis=1)
    
    return normalized_df

def load_sample_data(data_path=None):
    """Загрузка тестовых данных или создание синтетических, если файл не найден"""
    if data_path and os.path.exists(data_path):
        logger.info(f"Загрузка данных из {data_path}")
        return pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Создаем синтетические данные для тестирования
    logger.info("Создание синтетических данных для тестирования")
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2022-01-01', freq='D')
    
    symbols = []
    data = {}
    
    # Нормальные символы
    for i in range(50):
        symbol = f"NORMAL_{i}"
        symbols.append(symbol)
        # Случайный тренд с шумом
        data[symbol] = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
    
    # Символы с выбросами
    for i in range(10):
        symbol = f"OUTLIER_{i}"
        symbols.append(symbol)
        base = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
        # Добавляем выбросы
        outlier_idx = np.random.choice(len(dates), 5, replace=False)
        base[outlier_idx] = base[outlier_idx] * np.random.uniform(5, 10, 5)
        data[symbol] = base
    
    # Символы с постоянной ценой
    for i in range(10):
        symbol = f"CONSTANT_{i}"
        symbols.append(symbol)
        data[symbol] = np.ones(len(dates)) * (100 + i)
    
    # Символы с пропусками (NaN)
    for i in range(10):
        symbol = f"MISSING_{i}"
        symbols.append(symbol)
        values = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
        # Вставляем случайные NaN
        missing_idx = np.random.choice(len(dates), int(len(dates) * 0.3), replace=False)
        values[missing_idx] = np.nan
        data[symbol] = values
    
    df = pd.DataFrame(data, index=dates)
    return df

def compare_normalization_methods(price_df):
    """Сравнение различных методов нормализации"""
    # Запуск текущего метода
    logger.info("Применение текущей нормализации...")
    current_result = current_normalization(price_df.copy())
    current_symbols = current_result.shape[1]
    
    # Запуск улучшенной min-max нормализации
    logger.info("Применение улучшенной min-max нормализации...")
    minmax_result, minmax_stats = preprocess_and_normalize_data(
        price_df.copy(), 
        norm_method="minmax",
        handle_constant=True
    )
    minmax_symbols = minmax_result.shape[1]
    
    # Запуск Z-score нормализации
    logger.info("Применение Z-score нормализации...")
    zscore_result, zscore_stats = preprocess_and_normalize_data(
        price_df.copy(), 
        norm_method="zscore",
        handle_constant=True
    )
    zscore_symbols = zscore_result.shape[1]
    
    # Запуск log-returns нормализации
    logger.info("Применение log-returns нормализации...")
    logreturns_result, logreturns_stats = preprocess_and_normalize_data(
        price_df.copy(), 
        norm_method="log_returns",
        handle_constant=True
    )
    logreturns_symbols = logreturns_result.shape[1]
    
    # Запуск адаптивной нормализации
    logger.info("Применение адаптивной нормализации...")
    adaptive_result = adaptive_normalization(price_df.copy())
    adaptive_symbols = adaptive_result.shape[1]
    
    # Вывод результатов
    total_symbols = price_df.shape[1]
    logger.info(f"\nРезультаты сравнения методов нормализации:")
    logger.info(f"Всего символов в исходных данных: {total_symbols}")
    logger.info(f"Текущий метод: {current_symbols} символов ({current_symbols/total_symbols*100:.1f}%)")
    logger.info(f"Улучшенный min-max: {minmax_symbols} символов ({minmax_symbols/total_symbols*100:.1f}%)")
    logger.info(f"Z-score: {zscore_symbols} символов ({zscore_symbols/total_symbols*100:.1f}%)")
    logger.info(f"Log-returns: {logreturns_symbols} символов ({logreturns_symbols/total_symbols*100:.1f}%)")
    logger.info(f"Адаптивный метод: {adaptive_symbols} символов ({adaptive_symbols/total_symbols*100:.1f}%)")
    
    # Визуализация результатов
    methods = ['Текущий', 'Min-Max+', 'Z-score', 'Log-returns', 'Адаптивный']
    symbols_count = [current_symbols, minmax_symbols, zscore_symbols, logreturns_symbols, adaptive_symbols]
    retention = [count/total_symbols*100 for count in symbols_count]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(methods, retention, color=['gray', 'blue', 'green', 'orange', 'red'])
    plt.title('Сохранение символов разными методами нормализации')
    plt.ylabel('Процент сохраненных символов')
    plt.axhline(y=current_symbols/total_symbols*100, color='r', linestyle='--', alpha=0.3)
    
    # Добавляем значения над столбцами
    for bar, count, percent in zip(bars, symbols_count, retention):
        plt.text(bar.get_x() + bar.get_width()/2, percent + 1, 
                f'{count} ({percent:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Сохраняем график
    plt.savefig('normalization_comparison.png')
    logger.info("График сохранен в normalization_comparison.png")
    
    return {
        'current': current_symbols,
        'minmax_improved': minmax_symbols,
        'zscore': zscore_symbols,
        'log_returns': logreturns_symbols,
        'adaptive': adaptive_symbols,
        'total': total_symbols
    }

def main():
    """Основная функция скрипта"""
    logger.info("Запуск теста улучшенных методов нормализации")
    
    # Проверяем наличие файла с данными
    data_path = None
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    
    # Загружаем данные
    price_df = load_sample_data(data_path)
    logger.info(f"Загружено {price_df.shape[1]} символов за {price_df.shape[0]} дней")
    
    # Запускаем сравнение методов
    results = compare_normalization_methods(price_df)
    
    best_method = max(results.items(), key=lambda x: x[1] if x[0] != 'total' else 0)[0]
    logger.info(f"\nЛучший метод: {best_method}")
    
    improvement = results[best_method] - results['current']
    improvement_percent = improvement / results['total'] * 100
    logger.info(f"Улучшение: +{improvement} символов (+{improvement_percent:.1f}%)")

if __name__ == "__main__":
    main()