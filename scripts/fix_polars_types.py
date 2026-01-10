#!/usr/bin/env python3
"""
Скрипт для поиска и исправления проблем с типами данных в Polars.
Находит столбцы с некорректным типом (Utf8 вместо числа/даты) и конвертирует их.
"""

import polars as pl
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union, Tuple
import yaml
import sys

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config_dict(config_path: str = "configs/main_2024.yaml") -> dict:
    """Загружает конфигурацию из YAML файла как словарь для обратной совместимости."""
    try:
        from src.coint2.utils.config import load_config
        config_obj = load_config(config_path)
        # Преобразуем AppConfig в словарь
        config = config_obj.model_dump()
        logger.info(f"Конфигурация загружена из {config_path}")
        return config
    except Exception as e:
        logger.error(f"Ошибка при загрузке конфигурации: {e}")
        return {}


def find_parquet_files(data_dir: Path) -> List[Path]:
    """Находит все parquet файлы в структуре данных."""
    return list(data_dir.glob("**/*.parquet"))


def sanitize_numeric(s: pl.Expr) -> pl.Expr:
    """
    Очищает строковые данные для конвертации в числовой формат.
    Удаляет пробелы, заменяет запятые на точки.
    """
    return (
        s.str.replace_all(r"[ ,]", "")
         .cast(pl.Float64, strict=False)
    )


def check_column_types(df: pl.DataFrame) -> Dict[str, str]:
    """
    Проверяет типы столбцов и возвращает словарь с проблемными столбцами.
    
    Returns:
        Dict[str, str]: Словарь {имя_столбца: тип_данных} для строковых столбцов,
                        которые должны быть числовыми или датами.
    """
    problematic_columns = {}
    
    # Типичные числовые столбцы в финансовых данных
    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'price', 'amount']
    
    # Типичные столбцы с датами
    date_columns = ['date', 'time', 'datetime']
    
    for col in df.columns:
        col_type = str(df.schema[col])
        
        # Проверяем, является ли столбец строковым, но по имени должен быть числовым
        if col_type == 'Utf8':
            col_lower = col.lower()
            
            if any(num_col in col_lower for num_col in numeric_columns):
                problematic_columns[col] = 'numeric'
            elif any(date_col in col_lower for date_col in date_columns):
                problematic_columns[col] = 'date'
    
    return problematic_columns


def fix_column_types(df: pl.DataFrame, column_types: Dict[str, str]) -> pl.DataFrame:
    """
    Исправляет типы проблемных столбцов.
    
    Args:
        df: DataFrame с проблемными столбцами
        column_types: Словарь {имя_столбца: тип_данных} для исправления
        
    Returns:
        pl.DataFrame: DataFrame с исправленными типами
    """
    if not column_types:
        return df
    
    expressions = []
    
    for col, col_type in column_types.items():
        if col_type == 'numeric':
            expressions.append(
                sanitize_numeric(pl.col(col)).alias(col)
            )
        elif col_type == 'date':
            # Пытаемся определить формат даты и конвертировать
            expressions.append(
                pl.col(col).str.strptime(pl.Datetime, strict=False).alias(col)
            )
    
    # Применяем исправления к DataFrame
    if expressions:
        return df.with_columns(expressions)
    return df


def process_parquet_file(file_path: Path, save_fixed: bool = True) -> Tuple[bool, Optional[Dict[str, str]]]:
    """
    Обрабатывает один parquet файл - проверяет и исправляет типы данных.
    
    Args:
        file_path: Путь к parquet файлу
        save_fixed: Сохранять ли исправленный файл
        
    Returns:
        Tuple[bool, Optional[Dict[str, str]]]: 
            - True если файл был исправлен, False если исправления не требуются
            - Словарь с проблемными столбцами или None, если проблем нет
    """
    try:
        # Загружаем данные
        df = pl.scan_parquet(file_path).collect()
        
        # Проверяем типы столбцов
        problematic_columns = check_column_types(df)
        
        if not problematic_columns:
            return False, None  # Файл уже в порядке
        
        # Исправляем типы столбцов
        df_fixed = fix_column_types(df, problematic_columns)
        
        # Сохраняем исправленный файл
        if save_fixed:
            # Создаем новое имя файла с суффиксом _fixed
            fixed_path = file_path.with_name(f"{file_path.stem}_fixed{file_path.suffix}")
            df_fixed.write_parquet(fixed_path)
            logger.info(f"Исправленный файл сохранен: {fixed_path}")
        
        # Выводим информацию о исправленных столбцах
        for col, col_type in problematic_columns.items():
            old_type = str(df.schema[col])
            new_type = str(df_fixed.schema[col])
            logger.info(f"Столбец '{col}' исправлен: {old_type} -> {new_type}")
        
        return True, problematic_columns
        
    except Exception as e:
        logger.error(f"Ошибка при обработке {file_path}: {e}")
        return False, None


def fix_types_in_directory(data_dir: Path = Path("data_clean"), limit: int = None) -> Dict[str, int]:
    """
    Исправляет типы данных во всех parquet файлах в указанной директории.
    
    Args:
        data_dir: Директория с данными
        limit: Ограничение на количество обрабатываемых файлов (для тестирования)
        
    Returns:
        Dict[str, int]: Словарь с проблемными столбцами и количеством файлов
    """
    if not data_dir.exists():
        logger.error(f"Папка {data_dir} не существует")
        return {}
    
    logger.info(f"Поиск parquet файлов в {data_dir}")
    parquet_files = find_parquet_files(data_dir)
    
    if limit:
        parquet_files = parquet_files[:limit]
        
    logger.info(f"Найдено {len(parquet_files)} файлов для проверки")
    
    if not parquet_files:
        logger.warning("Parquet файлы не найдены")
        return {}
    
    # Статистика
    files_fixed = 0
    files_skipped = 0
    errors = 0
    all_problematic_columns = {}
    
    # Обрабатываем файлы
    for file_path in parquet_files:
        try:
            was_fixed, problematic_columns = process_parquet_file(file_path)
            
            if was_fixed:
                files_fixed += 1
                # Собираем информацию о проблемных столбцах
                if problematic_columns:
                    for col, col_type in problematic_columns.items():
                        if col in all_problematic_columns:
                            all_problematic_columns[col] += 1
                        else:
                            all_problematic_columns[col] = 1
            else:
                files_skipped += 1
                
        except Exception as e:
            logger.error(f"Ошибка при обработке {file_path}: {e}")
            errors += 1
    
    # Выводим статистику
    logger.info("=" * 50)
    logger.info("СТАТИСТИКА ИСПРАВЛЕНИЯ ТИПОВ:")
    logger.info(f"Файлов проверено: {len(parquet_files)}")
    logger.info(f"Файлов исправлено: {files_fixed}")
    logger.info(f"Файлов пропущено (уже в порядке): {files_skipped}")
    logger.info(f"Ошибок: {errors}")
    
    if all_problematic_columns:
        logger.info("\nПРОБЛЕМНЫЕ СТОЛБЦЫ:")
        for col, count in sorted(all_problematic_columns.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  - {col}: {count} файлов")
    
    logger.info("=" * 50)
    
    # Генерируем конфигурацию dtypes для YAML
    if all_problematic_columns:
        logger.info("\nДобавьте следующую конфигурацию в configs/main_2024.yaml:")
        logger.info("dtypes:")
        for col, _ in all_problematic_columns.items():
            if all_problematic_columns.get(col, 0) > 0:
                logger.info(f"  {col}: float64  # или int64 для целых чисел")
    
    if files_fixed > 0:
        logger.info("✅ Типы данных исправлены!")
    else:
        logger.info("✅ Проблем с типами данных не обнаружено!")
    
    return all_problematic_columns


def analyze_single_file(file_path: str):
    """
    Анализирует один parquet файл и выводит информацию о его структуре и типах данных.
    """
    try:
        df = pl.scan_parquet(file_path).collect()
        
        logger.info(f"Файл: {file_path}")
        logger.info(f"Размер: {df.shape[0]} строк, {df.shape[1]} столбцов")
        
        logger.info("\nСХЕМА:")
        for col, dtype in df.schema.items():
            logger.info(f"  - {col}: {dtype}")
        
        logger.info("\nПЕРВЫЕ 5 СТРОК:")
        logger.info(df.head(5))
        
        # Проверяем типы столбцов
        problematic_columns = check_column_types(df)
        
        if problematic_columns:
            logger.info("\nПРОБЛЕМНЫЕ СТОЛБЦЫ:")
            for col, col_type in problematic_columns.items():
                logger.info(f"  - {col}: должен быть {col_type}, но является {df.schema[col]}")
                
                # Показываем примеры значений
                logger.info(f"    Примеры значений: {df.select(col).head(3)}")
                
            # Исправляем типы и показываем результат
            df_fixed = fix_column_types(df, problematic_columns)
            
            logger.info("\nИСПРАВЛЕННЫЕ ТИПЫ:")
            for col in problematic_columns:
                logger.info(f"  - {col}: {df.schema[col]} -> {df_fixed.schema[col]}")
        else:
            logger.info("\n✅ Проблем с типами данных не обнаружено!")
            
    except Exception as e:
        logger.error(f"Ошибка при анализе файла {file_path}: {e}")


def update_config_with_dtypes(config_path: str, problematic_columns: Dict[str, int]):
    """
    Обновляет конфигурационный файл, добавляя секцию dtypes.
    """
    if not problematic_columns:
        logger.info("Нет проблемных столбцов для добавления в конфигурацию")
        return
    
    try:
        # Загружаем текущую конфигурацию
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Добавляем или обновляем секцию dtypes
        if 'dtypes' not in config:
            config['dtypes'] = {}
            
        for col, _ in problematic_columns.items():
            config['dtypes'][col] = 'float64'  # По умолчанию используем float64
        
        # Сохраняем обновленную конфигурацию
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
            
        logger.info(f"Конфигурация обновлена: {config_path}")
        
    except Exception as e:
        logger.error(f"Ошибка при обновлении конфигурации: {e}")


def main():
    """Основная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Исправление типов данных в Polars")
    parser.add_argument('--file', '-f', help='Путь к конкретному parquet файлу для анализа')
    parser.add_argument('--dir', '-d', default='data_clean', help='Директория с данными для обработки')
    parser.add_argument('--limit', '-l', type=int, help='Ограничение на количество обрабатываемых файлов')
    parser.add_argument('--update-config', '-u', action='store_true', help='Обновить конфигурацию с найденными типами')
    parser.add_argument('--config', '-c', default='configs/main_2024.yaml', help='Путь к конфигурационному файлу')
    
    args = parser.parse_args()
    
    if args.file:
        # Анализируем один файл
        analyze_single_file(args.file)
    else:
        # Обрабатываем все файлы в директории
        all_problematic_columns = fix_types_in_directory(Path(args.dir), args.limit)
        
        # Обновляем конфигурацию, если указан флаг
        if args.update_config:
            # all_problematic_columns был собран ранее в цикле обработки файлов
            # Вызываем функцию для обновления YAML-файла
            update_config_with_dtypes(args.config, all_problematic_columns)


if __name__ == "__main__":
    main()
