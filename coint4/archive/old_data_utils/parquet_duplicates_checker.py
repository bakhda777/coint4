#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Модуль для проверки и удаления дубликатов в parquet-файлах.
Используется для финальной проверки данных после массовой загрузки.
"""

import os
import logging
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Set, Tuple, Dict
import time

# Настройка логгера
logger = logging.getLogger()

class ParquetDuplicatesChecker:
    """Класс для проверки и удаления дубликатов в parquet-файлах."""
    
    def __init__(self, data_dir: str = "data_downloaded"):
        """
        Инициализация проверщика дубликатов.
        
        Args:
            data_dir: Директория с данными для проверки
        """
        self.data_dir = Path(data_dir)
        
    def scan_parquet_files(self) -> List[Path]:
        """
        Сканирует директорию и возвращает список всех parquet-файлов.
        
        Returns:
            List[Path]: Список путей к parquet-файлам
        """
        logger.info(f"🔍 Сканирование parquet-файлов в {self.data_dir}")
        parquet_files = list(self.data_dir.glob("**/data_part_*.parquet"))
        logger.info(f"✅ Найдено {len(parquet_files)} parquet-файлов")
        return parquet_files
    
    def check_file_for_duplicates(self, file_path: Path) -> Tuple[bool, int, int]:
        """
        Проверяет один файл на наличие дубликатов.
        
        Args:
            file_path: Путь к parquet-файлу
            
        Returns:
            Tuple[bool, int, int]: (есть ли дубликаты, количество строк до очистки, количество дубликатов)
        """
        try:
            # Читаем файл
            df = pd.read_parquet(file_path)
            
            # Получаем размер до удаления дубликатов
            initial_size = len(df)
            
            # Проверяем наличие нужных колонок
            if 'timestamp' not in df.columns or 'symbol' not in df.columns:
                logger.warning(f"⚠️ {file_path}: отсутствуют необходимые колонки")
                return False, initial_size, 0
            
            # Проверяем на дубликаты
            duplicates = df.duplicated(subset=['timestamp', 'symbol'])
            duplicate_count = duplicates.sum()
            
            # Если дубликатов нет, возвращаем результат
            if duplicate_count == 0:
                return False, initial_size, 0
            
            return True, initial_size, duplicate_count
            
        except Exception as e:
            logger.error(f"❌ Ошибка при проверке файла {file_path}: {e}")
            return False, 0, 0
    
    def remove_duplicates_from_file(self, file_path: Path) -> bool:
        """
        Удаляет дубликаты из файла и сохраняет результат.
        
        Args:
            file_path: Путь к parquet-файлу
            
        Returns:
            bool: True если операция успешна
        """
        try:
            # Создаем имя для файла бэкапа
            backup_path = file_path.with_name(f"{file_path.stem}_backup.parquet")
            
            # Создаем бэкап текущего файла
            if not os.path.exists(backup_path):
                os.rename(file_path, backup_path)
            
            # Читаем из бэкапа
            df = pd.read_parquet(backup_path)
            initial_size = len(df)
            
            # Удаляем дубликаты
            df = df.drop_duplicates(subset=['timestamp', 'symbol'], keep='last').reset_index(drop=True)
            final_size = len(df)
            
            # Сохраняем обратно
            df.to_parquet(file_path, index=False)
            
            logger.info(f"✅ {file_path}: удалено {initial_size - final_size} дубликатов, осталось {final_size} строк")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка при удалении дубликатов из {file_path}: {e}")
            if os.path.exists(backup_path) and not os.path.exists(file_path):
                # Восстанавливаем из бэкапа если файл поврежден
                os.rename(backup_path, file_path)
            return False
    
    def check_all_files(self, fix_duplicates: bool = True) -> Dict[str, int]:
        """
        Проверяет все файлы на наличие дубликатов и опционально удаляет их.
        
        Args:
            fix_duplicates: Удалять ли найденные дубликаты
            
        Returns:
            Dict[str, int]: Статистика проверки {всего файлов, файлов с дубликатами, всего строк, удалено дубликатов}
        """
        start_time = time.time()
        logger.info("🔍 Начало проверки файлов на дубликаты")
        
        parquet_files = self.scan_parquet_files()
        files_with_duplicates = 0
        total_rows = 0
        total_duplicates = 0
        
        for i, file_path in enumerate(parquet_files):
            # Выводим прогресс
            if i % 10 == 0:
                logger.info(f"📊 Прогресс: {i}/{len(parquet_files)} ({i/len(parquet_files)*100:.1f}%)")
            
            # Проверяем файл
            has_duplicates, row_count, duplicate_count = self.check_file_for_duplicates(file_path)
            total_rows += row_count
            
            # Если есть дубликаты и нужно их удалить
            if has_duplicates and fix_duplicates:
                files_with_duplicates += 1
                total_duplicates += duplicate_count
                success = self.remove_duplicates_from_file(file_path)
                if not success:
                    logger.warning(f"⚠️ Не удалось удалить дубликаты из {file_path}")
            elif has_duplicates:
                files_with_duplicates += 1
                total_duplicates += duplicate_count
                logger.info(f"⚠️ {file_path}: найдено {duplicate_count} дубликатов")
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Проверка завершена за {elapsed_time:.1f} секунд")
        logger.info(f"📊 Статистика: проверено {len(parquet_files)} файлов, найдено {files_with_duplicates} файлов с дубликатами")
        logger.info(f"📊 Всего строк: {total_rows}, удалено дубликатов: {total_duplicates}")
        
        return {
            "total_files": len(parquet_files),
            "files_with_duplicates": files_with_duplicates,
            "total_rows": total_rows,
            "duplicates_removed": total_duplicates
        }
        
def check_and_fix_duplicates(data_dir: str = "data_downloaded", fix: bool = True) -> Dict[str, int]:
    """
    Проверяет и удаляет дубликаты во всех parquet-файлах в указанной директории.
    
    Args:
        data_dir: Директория с данными
        fix: Удалять ли найденные дубликаты
        
    Returns:
        Dict[str, int]: Статистика проверки
    """
    checker = ParquetDuplicatesChecker(data_dir)
    return checker.check_all_files(fix_duplicates=fix)

if __name__ == "__main__":
    # Настройка логирования при запуске скрипта напрямую
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Запускаем проверку
    check_and_fix_duplicates()
