#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
from pandas import DataFrame

try:
    from tqdm import tqdm
except ImportError:
    # Создаём заглушку для tqdm, если библиотека не установлена
    def tqdm(iterable, **kwargs):
        return iterable


def collect_parquet_files(root: Path) -> List[Path]:
    """Рекурсивно собирает все parquet файлы в директории.

    Parameters
    ----------
    root : Path
        Корневая директория для сканирования.

    Returns
    -------
    List[Path]
        Список абсолютных путей к parquet файлам.
    """
    return [p for p in root.rglob("*.parquet") if p.is_file()]


def extract_symbol_from_path(file_path: Path, data_dir: Path) -> str:
    """Извлекает символ валютной пары из пути файла в директории data.

    Parameters
    ----------
    file_path : Path
        Путь к файлу данных.
    data_dir : Path
        Корневая директория данных.

    Returns
    -------
    str
        Символ валютной пары или None, если не удалось извлечь.
    """
    rel_path = file_path.relative_to(data_dir)
    parts = rel_path.parts
    if parts:
        return parts[0]  # В директории data первая часть пути - символ (BTCUSDT и т.д.)
    return None


def collect_timestamps_by_symbol(data_dir: Path) -> Dict[str, Set[int]]:
    """Собирает все временные метки по символам из директории data.

    Parameters
    ----------
    data_dir : Path
        Путь к директории с данными.

    Returns
    -------
    Dict[str, Set[int]]
        Словарь {символ: множество временных меток}.
    """
    all_files = collect_parquet_files(data_dir)
    timestamps_by_symbol: Dict[str, Set[int]] = defaultdict(set)
    
    print(f"Сканирование временных меток в {data_dir}...")
    for file_path in tqdm(all_files):
        symbol = extract_symbol_from_path(file_path, data_dir)
        if not symbol:
            continue
            
        try:
            df = pd.read_parquet(file_path)
            if 'timestamp' in df.columns and 'symbol' in df.columns:
                # Файл содержит данные с временными метками
                for _, row in df.iterrows():
                    timestamps_by_symbol[row['symbol']].add(row['timestamp'])
            elif 'timestamp' in df.columns:
                # Файл без колонки symbol, используем символ из пути
                for ts in df['timestamp']:
                    timestamps_by_symbol[symbol].add(ts)
        except Exception as e:
            print(f"Ошибка при чтении {file_path}: {e}")
            
    return timestamps_by_symbol


def collect_timestamps_from_optimized(optimized_dir: Path) -> Dict[str, Set[int]]:
    """Собирает все временные метки из директории data_optimized.

    Parameters
    ----------
    optimized_dir : Path
        Путь к директории с оптимизированными данными.

    Returns
    -------
    Dict[str, Set[int]]
        Словарь {символ: множество временных меток}.
    """
    all_files = collect_parquet_files(optimized_dir)
    timestamps_by_symbol: Dict[str, Set[int]] = defaultdict(set)
    
    print(f"Сканирование временных меток в {optimized_dir}...")
    for file_path in tqdm(all_files):
        try:
            df = pd.read_parquet(file_path)
            if 'timestamp' in df.columns and 'symbol' in df.columns:
                for _, row in df.iterrows():
                    timestamps_by_symbol[row['symbol']].add(row['timestamp'])
        except Exception as e:
            print(f"Ошибка при чтении {file_path}: {e}")
            
    return timestamps_by_symbol


def sync_missing_data(
    data_dir: Path, 
    optimized_dir: Path, 
    data_timestamps: Dict[str, Set[int]], 
    optimized_timestamps: Dict[str, Set[int]]
) -> Tuple[int, int]:
    """Синхронизирует недостающие временные метки из data в data_optimized.

    Parameters
    ----------
    data_dir : Path
        Путь к исходной директории данных.
    optimized_dir : Path
        Путь к директории с оптимизированными данными.
    data_timestamps : Dict[str, Set[int]]
        Словарь {символ: множество временных меток} из data.
    optimized_timestamps : Dict[str, Set[int]]
        Словарь {символ: множество временных меток} из data_optimized.

    Returns
    -------
    Tuple[int, int]
        (количество добавленных строк, количество обработанных символов)
    """
    total_added_rows = 0
    processed_symbols = 0
    
    # Создаём директорию для временных файлов
    temp_dir = Path(os.path.join(optimized_dir.parent, 'temp_sync'))
    temp_dir.mkdir(exist_ok=True)
    
    print("Поиск и синхронизация недостающих временных меток...")
    for symbol in tqdm(data_timestamps):
        processed_symbols += 1
        
        # Находим отсутствующие временные метки
        if symbol not in optimized_timestamps:
            missing_timestamps = data_timestamps[symbol]
        else:
            missing_timestamps = data_timestamps[symbol] - optimized_timestamps[symbol]
        
        if not missing_timestamps:
            continue
            
        # Собираем все данные для отсутствующих временных меток
        missing_data_frames = []
        all_files = [f for f in collect_parquet_files(data_dir) 
                    if extract_symbol_from_path(f, data_dir) == symbol]
        
        for file_path in all_files:
            try:
                df = pd.read_parquet(file_path)
                if 'timestamp' not in df.columns:
                    continue
                    
                # Фильтруем только отсутствующие временные метки
                filtered_df = df[df['timestamp'].isin(missing_timestamps)]
                if not filtered_df.empty:
                    # Добавляем символ, если его нет
                    if 'symbol' not in filtered_df.columns:
                        filtered_df['symbol'] = symbol
                    missing_data_frames.append(filtered_df)
            except Exception as e:
                print(f"Ошибка при чтении {file_path}: {e}")
        
        if not missing_data_frames:
            continue
            
        # Объединяем все найденные данные
        combined_missing_data = pd.concat(missing_data_frames, ignore_index=True)
        combined_missing_data = combined_missing_data.drop_duplicates()
        
        # Определяем директории по годам и месяцам в data_optimized
        years_months = set()
        for ts in missing_timestamps:
            # Преобразуем timestamp в дату для определения года и месяца
            date = pd.to_datetime(ts, unit='ms')
            years_months.add((date.year, date.month))
        
        # Для каждой пары год-месяц добавляем недостающие данные в соответствующий файл
        for year, month in years_months:
            month_str = f"{month:02d}"
            target_dir = optimized_dir / f"year={year}" / f"month={month_str}"
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Фильтруем данные для текущего года и месяца
            date_mask = pd.to_datetime(combined_missing_data['timestamp'], unit='ms')
            year_month_df = combined_missing_data[
                (date_mask.dt.year == year) & (date_mask.dt.month == month)
            ]
            
            if year_month_df.empty:
                continue
                
            # Находим существующие файлы для этого месяца
            existing_files = list(target_dir.glob("*.parquet"))
            if existing_files:
                # Если файлы уже есть, добавляем данные в первый файл
                target_file = existing_files[0]
                existing_df = pd.read_parquet(target_file)
                
                # Объединяем без дубликатов
                combined_df = pd.concat([existing_df, year_month_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates()
                
                # Если добавились новые строки, сохраняем обновленный файл
                if len(combined_df) > len(existing_df):
                    added_rows = len(combined_df) - len(existing_df)
                    total_added_rows += added_rows
                    
                    # Сохраняем во временный файл и затем перемещаем
                    temp_file = temp_dir / f"temp_{year}_{month_str}_{symbol}.parquet"
                    combined_df.to_parquet(temp_file, index=False)
                    os.replace(temp_file, target_file)
            else:
                # Если файлов нет, создаем новый
                target_file = target_dir / f"data_part_{month_str}.parquet"
                year_month_df.to_parquet(target_file, index=False)
                total_added_rows += len(year_month_df)
    
    # Удаляем временную директорию, если она существует и пуста
    if temp_dir.exists():
        try:
            temp_dir.rmdir()
        except:
            pass
            
    return total_added_rows, processed_symbols


def analyze_missing_coverage(
    data_timestamps: Dict[str, Set[int]], 
    optimized_timestamps: Dict[str, Set[int]]
) -> None:
    """Анализирует и показывает статистику по покрытию временных меток.

    Parameters
    ----------
    data_timestamps : Dict[str, Set[int]]
        Словарь {символ: множество временных меток} из data.
    optimized_timestamps : Dict[str, Set[int]]
        Словарь {символ: множество временных меток} из data_optimized.
    """
    all_symbols = set(data_timestamps.keys())
    missing_symbols = all_symbols - set(optimized_timestamps.keys())
    
    print(f"\nВсего символов в data: {len(all_symbols)}")
    print(f"Символы в data_optimized: {len(optimized_timestamps)}")
    print(f"Отсутствующие символы в data_optimized: {len(missing_symbols)}")
    
    if missing_symbols:
        print("\nСписок отсутствующих символов:")
        for symbol in sorted(missing_symbols):
            print(f"  - {symbol}")
    
    total_data_timestamps = sum(len(ts) for ts in data_timestamps.values())
    total_optimized_timestamps = sum(len(ts) for ts in optimized_timestamps.values())
    
    print(f"\nВсего временных меток в data: {total_data_timestamps}")
    print(f"Всего временных меток в data_optimized: {total_optimized_timestamps}")
    
    # Анализ по каждому символу, который есть в обоих наборах
    common_symbols = all_symbols.intersection(set(optimized_timestamps.keys()))
    symbols_with_missing = []
    
    for symbol in common_symbols:
        data_ts = data_timestamps[symbol]
        optimized_ts = optimized_timestamps[symbol]
        
        missing_ts = data_ts - optimized_ts
        if missing_ts:
            symbols_with_missing.append((symbol, len(missing_ts)))
    
    if symbols_with_missing:
        print("\nСимволы с недостающими временными метками:")
        for symbol, count in sorted(symbols_with_missing, key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - {symbol}: недостает {count} меток")
            
        if len(symbols_with_missing) > 10:
            print(f"  ... и еще {len(symbols_with_missing) - 10} символов")


def parse_args() -> argparse.Namespace:
    """Разбор аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description=(
            "Сравнивает временные метки между data и data_optimized, "
            "добавляя отсутствующие данные без повторов."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Путь к исходной директории data.",
    )
    parser.add_argument(
        "--optimized-dir",
        type=Path,
        default=Path("data_optimized"),
        help="Путь к директории data_optimized.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Только анализировать различия без синхронизации.",
    )
    return parser.parse_args()


def main() -> None:
    """Основная функция скрипта."""
    args = parse_args()
    
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Директория {args.data_dir} не найдена.")
    if not args.optimized_dir.exists():
        raise FileNotFoundError(f"Директория {args.optimized_dir} не найдена.")

    # Собираем временные метки из обеих директорий
    data_timestamps = collect_timestamps_by_symbol(args.data_dir)
    optimized_timestamps = collect_timestamps_from_optimized(args.optimized_dir)
    
    # Анализируем и показываем статистику по покрытию временных меток
    analyze_missing_coverage(data_timestamps, optimized_timestamps)
    
    if not args.analyze_only:
        # Синхронизируем недостающие данные
        added_rows, processed_symbols = sync_missing_data(
            args.data_dir, 
            args.optimized_dir, 
            data_timestamps, 
            optimized_timestamps
        )
        print(f"\nСинхронизация завершена. Обработано символов: {processed_symbols}")
        print(f"Добавлено строк данных: {added_rows}")
    else:
        print("\nСинхронизация не выполнялась (режим только анализа).")


if __name__ == "__main__":
    main()
