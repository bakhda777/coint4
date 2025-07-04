#!/usr/bin/env python3
"""
Скрипт для исправления конфликта партиций.
Удаляет столбец 'symbol' из parquet файлов, поскольку он дублируется в структуре папок.
"""

import pandas as pd
from pathlib import Path
from typing import List
import logging
from tqdm import tqdm

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_all_parquet_files(data_dir: Path) -> List[Path]:
    """Находит все parquet файлы в структуре данных."""
    return list(data_dir.glob("**/data.parquet"))


def fix_parquet_file(file_path: Path) -> bool:
    """
    Исправляет один parquet файл - удаляет столбец symbol.
    
    Returns:
        True если файл был изменен, False если изменения не требуются
    """
    try:
        # Загружаем данные
        df = pd.read_parquet(file_path)
        
        # Проверяем есть ли столбец symbol
        if 'symbol' not in df.columns:
            return False  # Файл уже в порядке
        
        # Удаляем столбец symbol
        df_fixed = df.drop(columns=['symbol'])
        
        # Сохраняем исправленный файл
        df_fixed.to_parquet(file_path, index=False)
        
        logger.debug(f"Исправлен файл: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при обработке {file_path}: {e}")
        return False


def fix_data_clean_partitions(data_dir: Path = Path("data_clean")):
    """
    Исправляет все parquet файлы в data_clean, удаляя конфликтующий столбец symbol.
    """
    if not data_dir.exists():
        logger.error(f"Папка {data_dir} не существует")
        return
    
    logger.info(f"Поиск parquet файлов в {data_dir}")
    parquet_files = find_all_parquet_files(data_dir)
    logger.info(f"Найдено {len(parquet_files)} файлов для проверки")
    
    if not parquet_files:
        logger.warning("Parquet файлы не найдены")
        return
    
    # Статистика
    files_fixed = 0
    files_skipped = 0
    errors = 0
    
    # Обрабатываем файлы
    for file_path in tqdm(parquet_files, desc="Исправление файлов"):
        try:
            was_fixed = fix_parquet_file(file_path)
            if was_fixed:
                files_fixed += 1
            else:
                files_skipped += 1
        except Exception as e:
            logger.error(f"Ошибка при обработке {file_path}: {e}")
            errors += 1
    
    # Выводим статистику
    logger.info("=" * 50)
    logger.info("СТАТИСТИКА ИСПРАВЛЕНИЯ:")
    logger.info(f"Файлов проверено: {len(parquet_files)}")
    logger.info(f"Файлов исправлено: {files_fixed}")
    logger.info(f"Файлов пропущено (уже в порядке): {files_skipped}")
    logger.info(f"Ошибок: {errors}")
    logger.info("=" * 50)
    
    if files_fixed > 0:
        logger.info("✅ Конфликт партиций исправлен!")
        logger.info("🎉 Теперь Dask/PyArrow будет корректно работать с данными")
    else:
        logger.info("ℹ️ Все файлы уже в правильном формате")


def test_fixed_data():
    """Тестирует исправленные данные."""
    try:
        from src.coint2.core.data_loader import DataHandler
        from coint2.utils.config import load_config
        import pandas as pd
        
        logger.info("Тестирование исправленных данных...")
        
        config = load_config(Path("configs/main.yaml"))
        handler = DataHandler(config)
        
        # Попробуем загрузить данные
        start_date = pd.Timestamp('2024-01-01')  
        end_date = pd.Timestamp('2024-01-02')
        
        df = handler.preload_all_data(start_date, end_date)
        if not df.empty:
            logger.info(f"✅ Тест пройден! Загружено данных: {df.shape}")
            logger.info(f"Столбцы: {list(df.columns)[:10]}...")
        else:
            logger.warning("❌ Данные не загружены")
            
    except Exception as e:
        logger.error(f"❌ Ошибка при тестировании: {e}")


def main():
    """Основная функция."""
    logger.info("Начинаем исправление конфликта партиций в data_clean")
    
    # Исправляем файлы
    fix_data_clean_partitions()
    
    # Тестируем результат
    test_fixed_data()
    
    logger.info("Исправление завершено!")


if __name__ == "__main__":
    main() 