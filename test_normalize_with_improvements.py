#!/usr/bin/env python
"""
Скрипт для демонстрации улучшений в нормализации.
Загружает реальные данные и сравнивает стандартную и улучшенную нормализацию.
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("normalize_test")

# Добавляем корневой каталог проекта в sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Импорт необходимых модулей
from src.coint2.core.normalization_improvements import preprocess_and_normalize_data

def current_normalization(df):
    """Текущая реализация нормализации в системе"""
    # Удаление NaN значений
    df = df.dropna(axis=1)
    
    # Выявление символов с постоянной ценой (которые вызовут деление на ноль)
    constant_mask = df.max() == df.min()
    constant_symbols = df.columns[constant_mask].tolist()
    
    # Сохраняем количество удаляемых символов для отчета
    constant_count = len(constant_symbols)
    
    # Удаляем символы с постоянной ценой перед нормализацией
    if constant_symbols:
        df = df.drop(columns=constant_symbols)
    
    # Min-max нормализация
    normalized_df = (df - df.min()) / (df.max() - df.min())
    
    # Удаление колонок с NaN (которые могли появиться при делении на ноль)
    normalized_df = normalized_df.dropna(axis=1)
    
    return normalized_df, constant_count

def create_synthetic_data():
    """Создает синтетические данные для тестирования"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2020-02-01', freq='D')
    
    data = {}
    
    # Добавляем 100 обычных символов
    for i in range(100):
        symbol = f"NORMAL_{i}"
        data[symbol] = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
    
    # Добавляем 20 символов с константной ценой
    for i in range(20):
        symbol = f"CONSTANT_{i}"
        data[symbol] = np.ones(len(dates)) * (100 + i)
    
    # Добавляем 16 символов с NaN значениями
    for i in range(16):
        symbol = f"MISSING_{i}"
        values = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
        missing_rate = 0.2 + (i / 16) * 0.6  # От 20% до 80% пропусков
        missing_idx = np.random.choice(len(dates), int(len(dates) * missing_rate), replace=False)
        values[missing_idx] = np.nan
        data[symbol] = values
    
    return pd.DataFrame(data, index=dates)

def main():
    """Основная функция скрипта"""
    logger.info("Начало теста улучшенной нормализации с реалистичными данными")
    
    # Создаем синтетические данные, имитирующие реальную ситуацию
    df = create_synthetic_data()
    total_symbols = df.shape[1]
    logger.info(f"Создано {total_symbols} символов для тестирования")
    logger.info(f"  - Нормальные: 100 символов")
    logger.info(f"  - Константные: 20 символов")
    logger.info(f"  - С пропусками: 16 символов (от 20% до 80% пропусков)")
    
    # Проверяем текущий метод нормализации
    current_result, constant_removed = current_normalization(df.copy())
    current_symbols = current_result.shape[1]
    logger.info(f"Текущий метод нормализации: сохранено {current_symbols} символов ({current_symbols/total_symbols*100:.1f}%)")
    logger.info(f"Потеряно {total_symbols - current_symbols} символов ({(total_symbols - current_symbols)/total_symbols*100:.1f}%)")
    logger.info(f"  в том числе из-за постоянной цены: {constant_removed} символов")
    
    # Проверяем улучшенный метод нормализации с сохранением символов с постоянной ценой
    improved_result, stats = preprocess_and_normalize_data(
        df.copy(),
        norm_method="zscore",  # Используем Z-score нормализацию
        fill_method="linear",  # Линейная интерполяция для пропусков
        min_history_ratio=0.3,  # Снижаем требование к истории
        handle_constant=True    # Обрабатываем символы с постоянной ценой
    )
    improved_symbols = improved_result.shape[1]
    
    # Выводим статистику
    logger.info("\nСтатистика улучшенной нормализации:")
    logger.info(f"  - Исходные символы: {stats['initial_symbols']}")
    logger.info(f"  - Недостаточная история: {stats['low_history_ratio']}")
    logger.info(f"  - Символы с постоянной ценой (сохранены): {stats['constant_price']}")
    logger.info(f"  - NaN после нормализации: {stats['nan_after_norm']}")
    logger.info(f"  - Итоговые символы: {stats['final_symbols']}")
    
    # Сколько символов с постоянной ценой сохранено
    constant_saved = stats['constant_price']
    logger.info(f"\nУлучшенный метод нормализации: сохранено {improved_symbols} символов ({improved_symbols/total_symbols*100:.1f}%)")
    logger.info(f"Потеряно {total_symbols - improved_symbols} символов ({(total_symbols - improved_symbols)/total_symbols*100:.1f}%)")
    logger.info(f"Сохранено символов с постоянной ценой: {constant_saved}")
    
    # Сравнение результатов
    improvement = improved_symbols - current_symbols
    improvement_percent = (improvement / total_symbols) * 100
    
    logger.info(f"\nУлучшение: +{improvement} символов (+{improvement_percent:.1f}%)")
    logger.info(f"Относительное улучшение: +{improvement/current_symbols*100:.1f}% к базовому методу")
    
    if improvement > 0:
        logger.info("\n✅ РЕЗУЛЬТАТ: Новый метод нормализации значительно сократил потерю символов!")
    else:
        logger.info("\n⚠️ РЕЗУЛЬТАТ: Новый метод не показал улучшений на тестовых данных.")

if __name__ == "__main__":
    main()