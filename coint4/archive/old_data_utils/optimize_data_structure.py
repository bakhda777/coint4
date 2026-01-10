#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для оптимизации структуры данных parquet.

Преобразует текущую структуру:
symbol=SYMBOL/year=YYYY/month=MM/day=DD/data.parquet

В новую оптимизированную структуру:
year=YYYY/month=MM/data_part_XX.parquet

Где каждый файл data_part_XX.parquet содержит данные всех символов за месяц
или его часть (если данных слишком много).
"""

import os
import glob
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_argparse() -> argparse.Namespace:
    """Настройка аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Оптимизация структуры данных parquet для повышения производительности запросов."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="/Users/admin/Desktop/coint4/data_clean",
        help="Директория с исходными данными в формате symbol=SYMBOL/year=YYYY/month=MM/day=DD/data.parquet",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="/Users/admin/Desktop/coint4/data_optimized",
        help="Директория для сохранения оптимизированной структуры year=YYYY/month=MM/data_part_XX.parquet",
    )
    parser.add_argument(
        "--max-file-size-mb",
        type=int,
        default=200,
        help="Максимальный размер одного parquet файла в МБ (по умолчанию 200)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Обработать только указанный год (опционально)",
    )
    parser.add_argument(
        "--month",
        type=int,
        default=None,
        help="Обработать только указанный месяц (опционально, требует указания года)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Режим проверки без записи файлов",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Подробный вывод информации",
    )
    return parser.parse_args()


def find_source_files(
    source_dir: Path, year: Optional[int] = None, month: Optional[int] = None
) -> Dict[Tuple[int, int], List[Path]]:
    """
    Находит все исходные файлы parquet, сгруппированные по году и месяцу.

    Parameters
    ----------
    source_dir : Path
        Директория с исходными данными
    year : Optional[int]
        Фильтр по году (опционально)
    month : Optional[int]
        Фильтр по месяцу (опционально)

    Returns
    -------
    Dict[Tuple[int, int], List[Path]]
        Словарь с ключами (год, месяц) и значениями - списками путей к файлам
    """
    result = {}
    
    # Создаем паттерн для поиска файлов
    if year is not None and month is not None:
        pattern = f"{source_dir}/symbol=*/year={year}/month={month:02d}/**/data.parquet"
    elif year is not None:
        pattern = f"{source_dir}/symbol=*/year={year}/**/data.parquet"
    else:
        pattern = f"{source_dir}/symbol=*/year=*/month=*/**/data.parquet"
    
    logger.info(f"Поиск файлов по шаблону: {pattern}")
    
    # Находим все файлы
    all_files = glob.glob(pattern, recursive=True)
    logger.info(f"Найдено файлов: {len(all_files)}")
    
    # Группируем файлы по году и месяцу
    for file_path in all_files:
        path = Path(file_path)
        # Извлекаем год и месяц из пути
        parts = path.parts
        
        # Находим индексы частей пути с годом и месяцем
        year_idx = next((i for i, part in enumerate(parts) if part.startswith("year=")), None)
        month_idx = next((i for i, part in enumerate(parts) if part.startswith("month=")), None)
        
        if year_idx is None or month_idx is None:
            logger.warning(f"Не удалось извлечь год и месяц из пути: {file_path}")
            continue
        
        # Извлекаем числовые значения года и месяца
        year_val = int(parts[year_idx].split("=")[1])
        month_val = int(parts[month_idx].split("=")[1])
        
        # Добавляем файл в соответствующую группу
        key = (year_val, month_val)
        if key not in result:
            result[key] = []
        result[key].append(path)
    
    return result


def process_month_data(
    files: List[Path], 
    target_dir: Path, 
    year: int, 
    month: int, 
    max_file_size_bytes: int,
    dry_run: bool = False,
    verbose: bool = False
) -> None:
    """
    Обрабатывает файлы за один месяц и сохраняет их в новую структуру.

    Parameters
    ----------
    files : List[Path]
        Список путей к файлам за один месяц
    target_dir : Path
        Директория для сохранения результатов
    year : int
        Год данных
    month : int
        Месяц данных
    max_file_size_bytes : int
        Максимальный размер одного файла в байтах
    dry_run : bool
        Если True, не сохраняет файлы
    verbose : bool
        Если True, выводит подробную информацию
    """
    logger.info(f"Обработка данных за {year}-{month:02d}, файлов: {len(files)}")
    
    # Создаем директорию для сохранения результатов
    output_dir = target_dir / f"year={year}" / f"month={month:02d}"
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Читаем все файлы и объединяем их в один DataFrame
    all_symbols = set()
    all_dfs = []
    total_rows = 0
    
    for file_path in tqdm(files, desc=f"Чтение файлов за {year}-{month:02d}"):
        try:
            # Извлекаем символ из пути
            symbol_part = next((part for part in file_path.parts if part.startswith("symbol=")), None)
            if symbol_part is None:
                logger.warning(f"Не удалось извлечь символ из пути: {file_path}")
                continue
            
            symbol = symbol_part.split("=")[1]
            all_symbols.add(symbol)
            
            # Читаем файл
            df = pl.read_parquet(file_path)
            
            # Добавляем столбец symbol, если его нет
            if "symbol" not in df.columns:
                df = df.with_columns(pl.lit(symbol).alias("symbol"))
            
            all_dfs.append(df)
            total_rows += len(df)
            
            if verbose:
                logger.info(f"Прочитан файл {file_path}, строк: {len(df)}")
                
        except Exception as e:
            logger.error(f"Ошибка при чтении файла {file_path}: {e}")
    
    if not all_dfs:
        logger.warning(f"Нет данных для обработки за {year}-{month:02d}")
        return
    
    # Объединяем все DataFrame
    logger.info(f"Объединение {len(all_dfs)} DataFrame за {year}-{month:02d}")
    combined_df = pl.concat(all_dfs)
    
    # Сортируем по timestamp и symbol для оптимальной компрессии
    combined_df = combined_df.sort(["timestamp", "symbol"])
    
    logger.info(f"Объединенный DataFrame: {len(combined_df)} строк, {len(all_symbols)} символов")
    
    # Оцениваем размер данных (грубо)
    estimated_size_bytes = len(combined_df) * len(combined_df.columns) * 8  # примерно 8 байт на значение
    
    # Определяем, нужно ли разбивать на части
    num_parts = max(1, estimated_size_bytes // max_file_size_bytes + 1)
    
    if num_parts > 1:
        logger.info(f"Разбиение на {num_parts} частей из-за большого размера")
        # Разбиваем DataFrame на части примерно равного размера
        rows_per_part = len(combined_df) // num_parts + 1
        
        for part_idx in range(num_parts):
            start_idx = part_idx * rows_per_part
            end_idx = min((part_idx + 1) * rows_per_part, len(combined_df))
            
            if start_idx >= len(combined_df):
                break
                
            part_df = combined_df.slice(start_idx, end_idx - start_idx)
            part_file = output_dir / f"data_part_{part_idx+1:02d}.parquet"
            
            logger.info(f"Сохранение части {part_idx+1}/{num_parts}: {len(part_df)} строк в {part_file}")
            
            if not dry_run:
                part_df.write_parquet(part_file)
    else:
        # Сохраняем весь DataFrame в один файл
        output_file = output_dir / "data_part_01.parquet"
        logger.info(f"Сохранение всех данных в один файл: {output_file}")
        
        if not dry_run:
            combined_df.write_parquet(output_file)
    
    logger.info(f"Завершена обработка данных за {year}-{month:02d}")


def main():
    """Основная функция скрипта."""
    args = setup_argparse()
    
    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    max_file_size_bytes = args.max_file_size_mb * 1024 * 1024  # Конвертируем МБ в байты
    
    logger.info(f"Начало оптимизации структуры данных")
    logger.info(f"Исходная директория: {source_dir}")
    logger.info(f"Целевая директория: {target_dir}")
    logger.info(f"Максимальный размер файла: {args.max_file_size_mb} МБ")
    
    if args.dry_run:
        logger.info("Режим проверки (dry run) - файлы не будут сохранены")
    
    # Находим все исходные файлы, сгруппированные по году и месяцу
    month_files = find_source_files(source_dir, args.year, args.month)
    
    if not month_files:
        logger.error("Не найдено файлов для обработки")
        return
    
    logger.info(f"Найдено {len(month_files)} групп файлов (год-месяц)")
    
    # Создаем целевую директорию, если она не существует
    if not args.dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
    
    # Обрабатываем каждый месяц
    for (year, month), files in sorted(month_files.items()):
        process_month_data(
            files=files,
            target_dir=target_dir,
            year=year,
            month=month,
            max_file_size_bytes=max_file_size_bytes,
            dry_run=args.dry_run,
            verbose=args.verbose
        )
    
    logger.info("Оптимизация структуры данных завершена")


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    elapsed = datetime.now() - start_time
    logger.info(f"Общее время выполнения: {elapsed}")
