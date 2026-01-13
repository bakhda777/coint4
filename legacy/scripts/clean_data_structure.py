#!/usr/bin/env python3
"""
Скрипт для создания очищенной структуры данных.
Объединяет множественные parquet файлы за день в один и удаляет дубликаты.
Создает структуру с symbol= партициями для совместимости с основным кодом.
"""

import pandas as pd
from pathlib import Path
from typing import List, Tuple
import logging
from tqdm import tqdm
import shutil

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_trading_pairs(data_dir: Path) -> List[str]:
    """Получает список торговых пар из директории данных."""
    pairs = []
    for item in data_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            pairs.append(item.name)
    return sorted(pairs)


def get_day_directories(pair_dir: Path) -> List[Tuple[str, str, str, Path]]:
    """Получает список всех директорий дней для торговой пары."""
    day_dirs = []
    
    for year_dir in pair_dir.iterdir():
        if not year_dir.is_dir() or not year_dir.name.startswith('year='):
            continue
        
        year = year_dir.name.split('=')[1]
        
        for month_dir in year_dir.iterdir():
            if not month_dir.is_dir() or not month_dir.name.startswith('month='):
                continue
            
            month = month_dir.name.split('=')[1]
            
            for day_dir in month_dir.iterdir():
                if not day_dir.is_dir() or not day_dir.name.startswith('day='):
                    continue
                
                day = day_dir.name.split('=')[1]
                day_dirs.append((year, month, day, day_dir))
    
    return day_dirs


def clean_day_data(day_dir: Path) -> pd.DataFrame:
    """Объединяет все parquet файлы за день и удаляет дубликаты."""
    parquet_files = list(day_dir.glob('*.parquet'))
    
    if not parquet_files:
        logger.warning(f"Нет parquet файлов в {day_dir}")
        return pd.DataFrame()
    
    # Загружаем все файлы за день
    dataframes = []
    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            dataframes.append(df)
        except Exception as e:
            logger.error(f"Ошибка при чтении {file}: {e}")
            continue
    
    if not dataframes:
        return pd.DataFrame()
    
    # Объединяем все данные
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Удаляем дубликаты по timestamp и сортируем
    cleaned_df = combined_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    
    # Сбрасываем индекс
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    logger.debug(f"День {day_dir.name}: {len(combined_df)} -> {len(cleaned_df)} записей")
    
    return cleaned_df


def create_clean_structure(source_dir: Path, target_dir: Path, pairs_limit: int = None):
    """Создает очищенную структуру данных с symbol= партициями."""
    
    # Создаем целевую директорию
    target_dir.mkdir(exist_ok=True)
    
    # Копируем служебные файлы
    for file_name in ['.symbols_cache.json', '.gitkeep', 'ignore.txt']:
        source_file = source_dir / file_name
        if source_file.exists():
            shutil.copy2(source_file, target_dir / file_name)
            logger.info(f"Скопирован файл: {file_name}")
    
    # Получаем список торговых пар
    trading_pairs = get_trading_pairs(source_dir)
    
    if pairs_limit:
        trading_pairs = trading_pairs[:pairs_limit]
        logger.info(f"Обрабатываем только первые {pairs_limit} пар для тестирования")
    
    logger.info(f"Найдено {len(trading_pairs)} торговых пар")
    
    # Статистика
    total_days_processed = 0
    total_files_created = 0
    errors = 0
    
    # Обрабатываем каждую торговую пару
    for pair in tqdm(trading_pairs, desc="Обработка торговых пар"):
        source_pair_dir = source_dir / pair
        
        # ✅ ГЛАВНОЕ ИЗМЕНЕНИЕ: создаем директорию с symbol= префиксом
        target_pair_dir = target_dir / f"symbol={pair}"
        target_pair_dir.mkdir(exist_ok=True)
        
        # Получаем все дни для этой пары
        day_directories = get_day_directories(source_pair_dir)
        
        if not day_directories:
            logger.warning(f"Нет данных для пары {pair}")
            continue
        
        # Обрабатываем каждый день
        for year, month, day, source_day_dir in tqdm(day_directories, 
                                                     desc=f"Дни для {pair}", 
                                                     leave=False):
            try:
                # Очищаем данные за день
                cleaned_df = clean_day_data(source_day_dir)
                
                if cleaned_df.empty:
                    logger.warning(f"Пустые данные для {pair} {year}-{month}-{day}")
                    errors += 1
                    continue
                
                # Убеждаемся, что столбец symbol корректный
                if 'symbol' not in cleaned_df.columns:
                    cleaned_df['symbol'] = pair
                elif not cleaned_df['symbol'].eq(pair).all():
                    cleaned_df['symbol'] = pair
                
                # Создаем структуру директорий в целевой папке
                target_year_dir = target_pair_dir / f"year={year}"
                # ✅ Добавляем ведущие нули для совместимости
                target_month_dir = target_year_dir / f"month={month.zfill(2)}"
                target_day_dir = target_month_dir / f"day={day.zfill(2)}"
                
                target_day_dir.mkdir(parents=True, exist_ok=True)
                
                # Сохраняем очищенные данные в один файл
                output_file = target_day_dir / "data.parquet"
                cleaned_df.to_parquet(output_file, index=False)
                
                total_days_processed += 1
                total_files_created += 1
                
            except Exception as e:
                logger.error(f"Ошибка при обработке {pair} {year}-{month}-{day}: {e}")
                errors += 1
                continue
    
    # Выводим статистику
    logger.info("=" * 50)
    logger.info("СТАТИСТИКА ОБРАБОТКИ:")
    logger.info(f"Торговых пар обработано: {len(trading_pairs)}")
    logger.info(f"Дней обработано: {total_days_processed}")
    logger.info(f"Файлов создано: {total_files_created}")
    logger.info(f"Ошибок: {errors}")
    logger.info("=" * 50)


def update_config_for_clean_data():
    """Обновляет конфигурацию для использования data_clean."""
    config_file = Path("configs/main.yaml")
    
    if not config_file.exists():
        logger.warning("Файл конфигурации configs/main.yaml не найден")
        return
    
    # Читаем конфигурацию
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Заменяем data_dir на data_clean
    if 'data_dir: "data_clean"' in content:
        logger.info("Конфигурация уже настроена на data_clean")
    else:
        # Заменяем любое значение data_dir на data_clean
        import re
        new_content = re.sub(
            r'^data_dir:\s*"[^"]*"',
            'data_dir: "data_clean"',
            content,
            flags=re.MULTILINE
        )
        
        # Сохраняем обновленную конфигурацию
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        logger.info("✅ Конфигурация обновлена: data_dir -> data_clean")


def main():
    """Основная функция."""
    source_dir = Path("data")
    target_dir = Path("data_clean")
    
    if not source_dir.exists():
        logger.error(f"Исходная директория {source_dir} не существует")
        return
    
    logger.info(f"Начинаем очистку данных из {source_dir} в {target_dir}")
    logger.info("✅ Создаем структуру с symbol= партициями для совместимости")
    
    # Для тестирования можно ограничить количество пар
    # create_clean_structure(source_dir, target_dir, pairs_limit=5)
    
    # Полная обработка
    create_clean_structure(source_dir, target_dir)
    
    # Обновляем конфигурацию
    update_config_for_clean_data()
    
    logger.info("Обработка завершена!")
    logger.info("🎉 Теперь основной код проекта может использовать data_clean напрямую!")


if __name__ == "__main__":
    main() 