#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Импортируем модуль группировки интервалов
try:
    from chunk_intervals import group_timestamps_into_chunks
except ImportError:
    # Если файл находится в другой директории
    try:
        from temporary_useful_files.chunk_intervals import group_timestamps_into_chunks
    except ImportError:
        def group_timestamps_into_chunks(timestamps, max_gap=15*60*1000, max_chunk_size=7*24*60*60*1000):
            """Резервная реализация, если модуль не найден"""
            if not timestamps:
                return []
            
            # Сортируем timestamps
            timestamps.sort()
            
            # Простая группировка по дням
            chunks = []
            date_groups = {}
            
            for ts in timestamps:
                # Группируем по дням
                dt = datetime.fromtimestamp(ts/1000)
                day_key = dt.strftime('%Y-%m-%d')
                if day_key not in date_groups:
                    date_groups[day_key] = []
                date_groups[day_key].append(ts)
            
            # Преобразуем группы дней в интервалы
            for day, day_timestamps in date_groups.items():
                if day_timestamps:
                    min_ts = min(day_timestamps)
                    max_ts = max(day_timestamps)
                    
                    start_dt = datetime.fromtimestamp(min_ts/1000) - timedelta(minutes=15)
                    end_dt = datetime.fromtimestamp(max_ts/1000) + timedelta(minutes=15)
                    
                    chunks.append((start_dt, end_dt))
            
            return chunks

# Стандартные библиотеки
import json, logging, os, configparser, time, random
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import ast
import multiprocessing
from functools import partial, lru_cache
from itertools import cycle
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Union, Set
from pathlib import Path
import shutil
import psutil
import gc
from logging.handlers import RotatingFileHandler
from threading import Lock, RLock
import signal
import sys
import argparse

# Импортируем модули для безопасной работы с данными
try:
    from file_lock_manager import FileLock, file_lock_manager
    from parquet_duplicates_checker import check_and_fix_duplicates
    modules_imported = True
except ImportError:
    modules_imported = False
    # Реализация базовой версии FileLock для случая, когда модуль недоступен
    class FileLock:
        def __init__(self, file_path, timeout=None):
            self.lock = Lock()
            self.file_path = file_path
            
        def __enter__(self):
            self.lock.acquire()
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.lock.release()

# Make line_profiler optional
try:
    from line_profiler import LineProfiler
except ImportError:
    line_profiler_missing = True
    class LineProfiler:
        def __init__(self, *args, **kwargs):
            pass
        def add_function(self, func):
            return self
        def enable_by_count(self):
            pass
        def disable_by_count(self):
            pass
        def print_stats(self, stream=None):
            print("LineProfiler not available")

# Make pybit optional
try:
    from pybit.unified_trading import HTTP, WebSocket
    from pybit.exceptions import FailedRequestError, InvalidRequestError
except ImportError:
    pybit_missing = True
    class HTTP:
        def __init__(self, *args, **kwargs):
            pass
    class WebSocket:
        def __init__(self, *args, **kwargs):
            pass
    class FailedRequestError(Exception):
        pass
    class InvalidRequestError(Exception):
        pass

# Библиотеки для работы с данными
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

# ================== КОНСТАНТЫ И НАСТРОЙКИ ==================

# Основные пути
DATA_DIR = Path("data_downloaded")  # Изменено на data_downloaded
MARKETS_FILE = "Markets.txt"

# Константы для работы с API
API_RATE_LIMIT = 120  # запросов в секунду
MIN_REQUEST_INTERVAL = 1.0 / API_RATE_LIMIT  # минимальный интервал между запросами
REQUEST_WINDOW = 1.0  # окно для подсчета RPS

# Настройки для обработки ошибок
MAX_RETRIES = 3
BASE_DELAY = 1.0
MAX_DELAY = 5.0
CHUNK_SIZE = 200  # Размер чанка для API запросов

# ================== ЛОГИРОВАНИЕ ==================

def setup_logging():
    """Настройка логирования."""
    logger = logging.getLogger()
    
    # Ротация логов
    file_handler = RotatingFileHandler(
        'data_loader.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(JsonFormatter())
    
    # Консольный вывод
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    
    return logger

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'time': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'function': record.funcName,
            'line': record.lineno
        }
        return json.dumps(log_data)

# Инициализация логгера
logger = setup_logging()

if 'line_profiler_missing' in globals():
    logger.warning("line_profiler not installed. Profiling functionality will be limited.")

if 'pybit_missing' in globals():
    logger.warning("pybit not installed. API functionality will be limited.")

# ================== КОНФИГУРАЦИЯ API ==================

# Значения по умолчанию
MAINNET_API_KEYS = ['dummy_key1', 'dummy_key2']
MAINNET_API_SECRETS = ['dummy_secret1', 'dummy_secret2']

if os.path.exists('config.ini'):
    try:
        config = configparser.ConfigParser()
        config.read('config.ini')
        
        if 'API' in config and 'MAINNET_API_KEY1' in config['API'] and 'MAINNET_API_KEY2' in config['API']:
            MAINNET_API_KEYS = [
                config['API']['MAINNET_API_KEY1'],
                config['API']['MAINNET_API_KEY2']
            ]
        if 'API' in config and 'MAINNET_API_SECRET1' in config['API'] and 'MAINNET_API_SECRET2' in config['API']:
            MAINNET_API_SECRETS = [
                config['API']['MAINNET_API_SECRET1'],
                config['API']['MAINNET_API_SECRET2']
            ]
    except Exception as e:
        logger.warning(f"Ошибка при чтении конфигурации: {str(e)}. Используются значения по умолчанию.")
else:
    logger.warning("Файл конфигурации config.ini не найден. Используются значения по умолчанию.")

def validate_api_keys() -> bool:
    """Проверка корректности API ключей"""
    if not MAINNET_API_KEYS or not MAINNET_API_SECRETS:
        logger.error("API ключи не настроены")
        return False
        
    if len(MAINNET_API_KEYS) != len(MAINNET_API_SECRETS):
        logger.error("Количество API ключей не совпадает с количеством секретов")
        return False
        
    return True

# ================== ФУНКЦИИ ЗАГРУЗКИ СИМВОЛОВ ==================

def fetch_bybit_markets(category: str = "spot", save_to_file: bool = True) -> List[str]:
    """
    Получает список всех доступных торговых пар с Bybit API.
    
    Args:
        category: Категория рынка ("spot", "linear", "inverse")
        save_to_file: Сохранять ли результат в файл Markets.txt
        
    Returns:
        List[str]: Список символов/пар
    """
    logger.info(f"🔍 Запрос списка пар Bybit для категории: {category}")
    
    try:
        # Создаем HTTP клиент с использованием API ключей
        session = HTTP(
            testnet=False,
            api_key=MAINNET_API_KEYS[0] if MAINNET_API_KEYS else None,
            api_secret=MAINNET_API_SECRETS[0] if MAINNET_API_SECRETS else None
        )
        
        # Получаем список торговых инструментов
        response = session.get_instruments_info(
            category=category
        )
        
        if response['retCode'] != 0:
            logger.error(f"❌ API ошибка: {response['retMsg']}")
            return []
        
        # Извлекаем список символов
        instruments = response['result']['list']
        symbols = [item['symbol'] for item in instruments if 'symbol' in item]
        
        # Фильтруем и сортируем (опционально - можно настроить фильтры)
        symbols = sorted(symbols)
        
        logger.info(f"✅ Получено {len(symbols)} пар с Bybit API")
        
        # Сохраняем в файл, если требуется
        if save_to_file:
            with open(MARKETS_FILE, 'w', encoding='utf-8') as f:
                f.write(', '.join(symbols))
            logger.info(f"✅ Список пар сохранен в {MARKETS_FILE}")
            
        # Дополнительно сохраняем расширенную информацию в JSON
        markets_info = []
        for item in instruments:
            if 'symbol' in item:
                markets_info.append({
                    'symbol': item.get('symbol', ''),
                    'status': item.get('status', ''),
                    'baseCoin': item.get('baseCoin', ''),
                    'quoteCoin': item.get('quoteCoin', ''),
                    'innovation': item.get('innovation', '0'),
                    'marginTrading': item.get('marginTrading', ''),
                })
        
        markets_json_file = 'Markets_extended.json'
        with open(markets_json_file, 'w', encoding='utf-8') as f:
            json.dump(markets_info, f, indent=2)
        logger.info(f"✅ Расширенная информация сохранена в {markets_json_file}")
        
        return symbols
        
    except Exception as e:
        logger.error(f"❌ Ошибка при получении списка пар: {e}")
        return []

def load_symbols_from_markets(markets_file: str) -> List[str]:
    """
    Загружает список символов из файла Markets.txt.
    
    Args:
        markets_file: Путь к файлу с символами
        
    Returns:
        List[str]: Список символов
    """
    try:
        with open(markets_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Разбиваем по запятым и очищаем
        symbols = [s.strip() for s in content.split(',') if s.strip()]
        
        # Удаляем дубликаты, сохраняя порядок
        unique_symbols = []
        seen = set()
        for symbol in symbols:
            if symbol not in seen:
                unique_symbols.append(symbol)
                seen.add(symbol)
                
        logger.info(f"Загружено {len(unique_symbols)} уникальных символов из {markets_file}")
        return unique_symbols
        
    except Exception as e:
        logger.error(f"Ошибка при чтении файла {markets_file}: {str(e)}")
        return []
# ================== ЗАГРУЗКА И ОБРАБОТКА ДАННЫХ ==================

class RequestTracker:
    def __init__(self):
        self.requests = []
        self.total_requests = 0
        self.start_time = time.time()
    
    def add_request(self):
        current_time = time.time()
        self.requests.append(current_time)
        self.total_requests += 1
        
        # Удаляем старые запросы
        while self.requests and current_time - self.requests[0] > REQUEST_WINDOW:
            self.requests.pop(0)
    
    def get_current_rps(self) -> float:
        return len(self.requests) / REQUEST_WINDOW
    
    def get_average_rps(self) -> float:
        elapsed = time.time() - self.start_time
        return self.total_requests / elapsed if elapsed > 0 else 0

# Глобальный трекер запросов
request_tracker = RequestTracker()
session_last_request = {}

def get_delay_time(session: HTTP) -> float:
    """Рассчитывает необходимую задержку для соблюдения лимитов."""
    current_time = time.time()
    last_request_time = session_last_request.get(id(session), 0)
    time_since_last_request = current_time - last_request_time
    if time_since_last_request < MIN_REQUEST_INTERVAL:
        delay = MIN_REQUEST_INTERVAL - time_since_last_request
        return delay
    return 0

def fetch_symbol_data(
    session: HTTP,
    symbol: str,
    start_time: datetime,
    end_time: datetime
) -> Optional[pd.DataFrame]:
    """
    Загружает данные для символа за указанный период.
    
    Args:
        session: HTTP сессия
        symbol: Символ для загрузки
        start_time: Начальное время
        end_time: Конечное время
        
    Returns:
        Optional[pd.DataFrame]: DataFrame с данными или None
    """
    logger.info(f"🔄 {symbol}: загрузка данных {start_time.strftime('%Y-%m-%d')} - {end_time.strftime('%Y-%m-%d')}")
    
    try:
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
    
        if start_ts >= end_ts:
            logger.error(f"❌ {symbol}: некорректный диапазон дат")
            return None
    
        # Задержка для соблюдения лимитов
        delay = get_delay_time(session)
        if delay > 0:
            time.sleep(delay)
    
        all_data = []
        current_start = start_ts
            
        # Разбиваем на чанки
        while current_start < end_ts:
            current_end = min(current_start + (CHUNK_SIZE * 15 * 60 * 1000), end_ts)
                
            try:
                response = session.get_kline(
                    category="spot",
                    symbol=symbol,
                    interval="15",  # 15 минут
                    start=current_start,
                    end=current_end,
                    limit=CHUNK_SIZE
                )
            
                if response['retCode'] != 0:
                    logger.error(f"❌ {symbol}: API ошибка - {response['retMsg']}")
                    return None
                    
                data = response['result']['list']
                if data:
                    all_data.extend(data)
                    
                # Обновляем статистику запросов
                request_tracker.add_request()
                current_rps = request_tracker.get_current_rps()
                
                # Динамическая задержка
                if current_rps > API_RATE_LIMIT:
                    excess_rps = current_rps - API_RATE_LIMIT
                    delay_time = excess_rps / (API_RATE_LIMIT * 10)
                    if delay_time > 0:
                        time.sleep(delay_time)
                
                session_last_request[id(session)] = time.time()
                current_start = current_end
            
            except Exception as e:
                logger.error(f"❌ {symbol}: ошибка запроса чанка: {e}")
                return None
        
        if not all_data:
            logger.warning(f"⚠️ {symbol}: нет данных за период")
            return None
        
        # Создаем DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])
        
        # Преобразуем типы
        df['ts_ms'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Добавляем символ для формата data_optimized
        df['symbol'] = symbol
        
        # Сортируем по времени
        df = df.sort_values('timestamp')
        
        logger.info(f"✅ {symbol}: загружено {len(df)} интервалов")
        return df
        
    except Exception as e:
        logger.error(f"❌ {symbol}: ошибка загрузки данных: {e}")
        return None

# Словарь для кэширования загруженных данных по месяцам
monthly_data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}

# Папка для временных файлов
TEMP_DIR = Path("data_temp")

def get_month_year_key(dt: datetime) -> str:
    """Возвращает ключ для группировки данных по месяцам в формате 'YYYY-MM'."""
    return f"{dt.year}-{dt.month:02d}"


def save_to_temp_file(symbol: str, df: pd.DataFrame, max_retries: int = 3) -> List[str]:
    """
    Сохраняет данные для одного символа во временные файлы (по одному на месяц).
    
    Args:
        symbol: Символ данных
        df: DataFrame с данными
        max_retries: Максимальное количество попыток при ошибке
        
    Returns:
        List[str]: Список путей к созданным временным файлам
    """
    if df.empty:
        logger.warning(f"⚠️ {symbol}: Нет данных для сохранения")
        return []
        
    # Создаем временную директорию если не существует
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Группируем данные по месяцам и сохраняем
    temp_files = []
    
    try:
        # Добавляем колонку с ключом year-month для группировки
        df['year_month'] = df['timestamp'].apply(lambda ts: 
                                               datetime.fromtimestamp(ts/1000).strftime("%Y-%m"))
        
        # Группируем по месяцам
        for month_key, month_df in df.groupby('year_month'):
            year, month = month_key.split('-')
            
            # Проверяем корректность временных меток
            if month_df['timestamp'].duplicated().any():
                logger.warning(f"⚠️ {symbol}: обнаружены дубликаты timestamp за {month_key}, удаляем")
                month_df = month_df.drop_duplicates(subset=['timestamp'])
                
            # Удаляем временную колонку группировки
            if 'year_month' in month_df.columns:
                month_df = month_df.drop('year_month', axis=1)
                
            # Генерируем имя временного файла
            temp_file = TEMP_DIR / f"{symbol}_{year}_{month}.parquet"
            
            # Сохраняем с повторными попытками при ошибке
            for attempt in range(max_retries):
                try:
                    # Используем PyArrow напрямую для более надежного сохранения
                    table = pa.Table.from_pandas(month_df)
                    pq.write_table(table, temp_file)
                    temp_files.append(str(temp_file))
                    logger.debug(f"✅ {symbol}: Временный файл сохранен: {temp_file}")
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 0.5 * (attempt + 1)
                        logger.warning(f"⚠️ {symbol}: Ошибка при сохранении во временный файл (попытка {attempt+1}/{max_retries}): {e}")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"❌ {symbol}: Не удалось сохранить временный файл после {max_retries} попыток: {e}")
                        
    except Exception as e:
        logger.error(f"❌ {symbol}: Ошибка при подготовке временных файлов: {e}")
        
    return temp_files


def merge_temp_files_to_monthly(year: str, month: str, max_retries: int = 3) -> bool:
    """
    Объединяет временные файлы для одного месяца в единый файл данных.
    Если файл уже существует, добавляет новые данные к существующим.
    
    Args:
        year: Год в формате YYYY
        month: Месяц в формате MM
        max_retries: Максимальное количество попыток при ошибке
        
    Returns:
        bool: True если успешно объединено
    """
    try:
        # Получаем список временных файлов для данного месяца
        pattern = f"*_{year}_{month}.parquet"
        temp_files = list(TEMP_DIR.glob(pattern))
        
        if not temp_files:
            logger.warning(f"⚠️ Не найдено временных файлов за {year}-{month}")
            return False
            
        # Создаем структуру директорий для итоговых файлов
        month_dir = DATA_DIR / f"year={year}" / f"month={month}"
        month_dir.mkdir(parents=True, exist_ok=True)
        
        # Итоговый файл данных
        output_file = month_dir / f"data_part_{month}.parquet"
        
        # Читаем и объединяем данные из всех временных файлов
        all_dfs = []
        
        # Проверяем существует ли уже файл с данными за этот месяц
        if output_file.exists():
            try:
                logger.info(f"📥 Файл {output_file} уже существует, добавляем к существующим данным")
                existing_df = pd.read_parquet(output_file)
                if not existing_df.empty:
                    all_dfs.append(existing_df)
                    logger.info(f"✅ Прочитаны существующие данные из {output_file}, {len(existing_df)} строк")
            except Exception as e:
                logger.error(f"❌ Ошибка при чтении существующего файла {output_file}: {e}")
        
        # Добавляем данные из временных файлов
        for temp_file in temp_files:
            try:
                # Читаем временный файл
                df = pd.read_parquet(temp_file)
                if not df.empty:
                    all_dfs.append(df)
                    logger.debug(f"✅ Прочитан файл {temp_file}, {len(df)} строк")
            except Exception as e:
                logger.error(f"❌ Ошибка при чтении временного файла {temp_file}: {e}")
        
        if not all_dfs:
            logger.error(f"❌ Нет данных для объединения за {year}-{month}")
            return False
            
        # Объединяем все данные
        merged_df = pd.concat(all_dfs, ignore_index=True)
        
        # Удаляем дубликаты по timestamp и symbol
        if merged_df.duplicated(subset=['timestamp', 'symbol']).any():
            dups_count = merged_df.duplicated(subset=['timestamp', 'symbol']).sum()
            logger.warning(f"⚠️ Обнаружены дубликаты в объединенных данных за {year}-{month}, удаляем {dups_count} записей")
            merged_df = merged_df.drop_duplicates(subset=['timestamp', 'symbol'])
            logger.info(f"✅ Удалено {dups_count} дубликатов")
        
        # Сохраняем объединенный файл с блокировкой и повторными попытками
        for attempt in range(max_retries):
            try:
                # Сохраняем с использованием блокировки
                with FileLock(str(output_file) + ".lock"):
                    # Используем PyArrow напрямую для более надежного сохранения
                    table = pa.Table.from_pandas(merged_df)
                    pq.write_table(table, output_file)
                logger.info(f"✅ Успешно сохранено {len(merged_df)} строк данных в {output_file}")
                
                # Удаляем временные файлы
                for temp_file in temp_files:
                    try:
                        temp_file.unlink()
                    except Exception as e:
                        logger.warning(f"⚠️ Не удалось удалить временный файл {temp_file}: {e}")
                        
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 0.5 * (attempt + 1)
                    logger.warning(f"⚠️ Ошибка при сохранении объединенного файла (попытка {attempt+1}/{max_retries}): {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ Не удалось сохранить объединенный файл после {max_retries} попыток: {e}")
                    return False
    except Exception as e:
        logger.error(f"❌ Ошибка при объединении временных файлов за {year}-{month}: {e}")
        return False
    if df.empty:
        logger.warning(f"⚠️ {symbol}: Нет данных для сохранения")
        return []
        
    # Создаем временную директорию если не существует
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Группируем данные по месяцам и сохраняем
    temp_files = []
    
    try:
        # Добавляем колонку с ключом year-month для группировки
        df['year_month'] = df['timestamp'].apply(lambda ts: 
                                               datetime.fromtimestamp(ts/1000).strftime("%Y-%m"))
        
        # Группируем по месяцам
        for month_key, month_df in df.groupby('year_month'):
            year, month = month_key.split('-')
            
            # Проверяем корректность временных меток
            if month_df['timestamp'].duplicated().any():
                logger.warning(f"⚠️ {symbol}: обнаружены дубликаты timestamp за {month_key}, удаляем")
                month_df = month_df.drop_duplicates(subset=['timestamp'])
                
            # Удаляем временную колонку группировки
            if 'year_month' in month_df.columns:
                month_df = month_df.drop('year_month', axis=1)
                
            # Генерируем имя временного файла
            temp_file = TEMP_DIR / f"{symbol}_{year}_{month}.parquet"
            
            # Сохраняем с повторными попытками при ошибке
            for attempt in range(max_retries):
                try:
                    # Используем PyArrow напрямую для более надежного сохранения
                    table = pa.Table.from_pandas(month_df)
                    pq.write_table(table, temp_file)
                    temp_files.append(str(temp_file))
                    logger.debug(f"✅ {symbol}: Временный файл сохранен: {temp_file}")
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 0.5 * (attempt + 1)
                        logger.warning(f"⚠️ {symbol}: Ошибка при сохранении во временный файл (попытка {attempt+1}/{max_retries}): {e}")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"❌ {symbol}: Не удалось сохранить временный файл после {max_retries} попыток: {e}")
                        
    except Exception as e:
        logger.error(f"❌ {symbol}: Ошибка при подготовке временных файлов: {e}")
        
    return temp_files
        
    # Создаем временную директорию если не существует
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Группируем данные по месяцам и сохраняем
    temp_files = []
    
    try:
        # Добавляем колонку с ключом year-month для группировки
        df['year_month'] = df['timestamp'].apply(lambda ts: 
                                               datetime.fromtimestamp(ts/1000).strftime("%Y-%m"))
        
        # Группируем по месяцам
        for month_key, month_df in df.groupby('year_month'):
            year, month = month_key.split('-')
            
            # Проверяем корректность временных меток
            if month_df['timestamp'].duplicated().any():
                logger.warning(f"⚠️ {symbol}: обнаружены дубликаты timestamp за {month_key}, удаляем")
                month_df = month_df.drop_duplicates(subset=['timestamp'])
                
            # Удаляем временную колонку группировки
            if 'year_month' in month_df.columns:
                month_df = month_df.drop('year_month', axis=1)
                
            # Генерируем имя временного файла
            temp_file = TEMP_DIR / f"{symbol}_{year}_{month}.parquet"
            
            # Сохраняем с повторными попытками при ошибке
            for attempt in range(max_retries):
                try:
                    # Используем PyArrow напрямую для более надежного сохранения
                    table = pa.Table.from_pandas(month_df)
                    pq.write_table(table, temp_file)
                    temp_files.append(str(temp_file))
                    logger.debug(f"✅ {symbol}: Временный файл сохранен: {temp_file}")
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 0.5 * (attempt + 1)
                        logger.warning(f"⚠️ {symbol}: Ошибка при сохранении во временный файл (попытка {attempt+1}/{max_retries}): {e}")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"❌ {symbol}: Не удалось сохранить временный файл после {max_retries} попыток: {e}")
                        
    except Exception as e:
        logger.error(f"❌ {symbol}: Ошибка при подготовке временных файлов: {e}")
        
    return temp_files

def save_data_by_month_optimized(symbol: str, df: pd.DataFrame, max_retries: int = 3) -> bool:
    """
    Сохраняет данные в оптимизированную структуру по месяцам с использованием блокировки файлов.
    Формат: /data_downloaded/year=YYYY/month=MM/data_part_MM.parquet
    
    Args:
        symbol: Символ данных
        df: DataFrame с данными
        max_retries: Максимальное количество попыток при ошибке
        
    Returns:
        bool: True если успешно сохранено
    """
    if df.empty:
        logger.warning(f"⚠️ {symbol}: нечего сохранять (пустой DataFrame)")
        return False
    
    try:
        # Добавляем datetime для группировки
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Группируем данные по году и месяцу
        groups = df.groupby([df['datetime'].dt.year, df['datetime'].dt.month])
        
        for (year, month), group_df in groups:
            # Удаляем datetime колонку перед сохранением
            group_df = group_df.drop(columns=['datetime'])
            
            # Создаем директорию для года и месяца
            month_str = f"{month:02d}"
            target_dir = DATA_DIR / f"year={year}" / f"month={month_str}"
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Путь к файлу данных
            target_file = target_dir / f"data_part_{month_str}.parquet"
            
            # Проверяем существующие данные с повторными попытками
            for attempt in range(max_retries):
                try:
                    if target_file.exists():
                        # Используем блокировку для безопасного чтения
                        with FileLock(str(target_file), timeout=30):
                            try:
                                existing_df = pd.read_parquet(target_file)
                                
                                # Проверяем наличие дубликатов
                                combined_df = pd.concat([existing_df, group_df], ignore_index=True)
                                combined_df = combined_df.drop_duplicates(
                                    subset=['timestamp', 'symbol'], 
                                    keep='last'
                                ).reset_index(drop=True)
                                
                                # Создаем временный файл
                                temp_file = target_file.with_name(f"{target_file.stem}_temp.parquet")
                                
                                # Сохраняем во временный файл
                                combined_df.to_parquet(temp_file, index=False)
                                
                                # Атомарно заменяем оригинальный файл на временный
                                shutil.move(str(temp_file), str(target_file))
                                
                                logger.info(f"✅ {symbol}: обновлены данные за {year}-{month_str}, всего {len(combined_df)} строк")
                            except Exception as e:
                                logger.warning(f"⚠️ {symbol}: попытка {attempt+1}/{max_retries} при обновлении данных за {year}-{month_str} не удалась: {e}")
                                if attempt == max_retries - 1:
                                    logger.error(f"❌ {symbol}: ошибка при обновлении данных за {year}-{month_str}: {e}")
                                    return False
                                time.sleep(1)  # Пауза перед повторной попыткой
                                continue
                    else:
                        # Для нового файла также используем блокировку директории
                        parent_lock_file = str(target_dir)
                        with FileLock(parent_lock_file, timeout=30):
                            # Двойная проверка, мог ли файл быть создан другим процессом
                            if not target_file.exists():
                                # Сохраняем через временный файл для атомарности
                                temp_file = target_file.with_name(f"{target_file.stem}_temp.parquet")
                                group_df.to_parquet(temp_file, index=False)
                                shutil.move(str(temp_file), str(target_file))
                                logger.info(f"✅ {symbol}: сохранены новые данные за {year}-{month_str}, всего {len(group_df)} строк")
                            else:
                                # Если файл уже создан, повторяем попытку с обновлением
                                continue
                    
                    # Если дошли сюда, значит всё успешно
                    break
                    
                except Exception as e:
                    logger.warning(f"⚠️ {symbol}: попытка {attempt+1}/{max_retries} при сохранении данных за {year}-{month_str} не удалась: {e}")
                    if attempt == max_retries - 1:
                        logger.error(f"❌ {symbol}: ошибка при сохранении данных за {year}-{month_str}: {e}")
                        return False
                    time.sleep(1)  # Пауза перед повторной попыткой
        
        return True
        
    except Exception as e:
        logger.error(f"❌ {symbol}: ошибка при сохранении данных: {e}")
        return False

def process_symbol_optimized(
    symbol: str, 
    sessions: List[HTTP], 
    start_date: datetime, 
    end_date: datetime,
    missing_timestamps: List[int] = None
) -> List[str]:
    """
    Обрабатывает один символ: загружает данные и сохраняет во временные файлы.
    
    Args:
        symbol: Символ для обработки
        sessions: Список HTTP сессий
        start_date: Начальная дата
        end_date: Конечная дата
        missing_timestamps: Список отсутствующих timestamp для инкрементального режима
        
    Returns:
        List[str]: Список путей к созданным временным файлам
    """
    session_index = random.randint(0, len(sessions)-1)
    session = sessions[session_index]
    
    try:
        # Если у нас режим инкрементальной загрузки с конкретными метками
        if missing_timestamps is not None and len(missing_timestamps) > 0:
            # Сортируем метки и группируем их в периоды для оптимизации запросов
            missing_timestamps.sort()
            
            # Если пропусков слишком мало, не имеет смысла разбивать на периоды
            if len(missing_timestamps) < 100:
                logger.info(f"🔹 {symbol}: Точечная загрузка {len(missing_timestamps)} пропущенных интервалов")
                
                # Создаем пустой DataFrame для накопления данных
                all_data = pd.DataFrame()
                
                # Конвертируем timestamps в даты для дебага
                first_date = datetime.fromtimestamp(missing_timestamps[0]/1000)
                last_date = datetime.fromtimestamp(missing_timestamps[-1]/1000)
                logger.info(f"📅 {symbol}: Интервал загрузки с {first_date.strftime('%Y-%m-%d')} по {last_date.strftime('%Y-%m-%d')}")
                
                # Загружаем данные для каждого timestamp
                chunks = [missing_timestamps[i:i+20] for i in range(0, len(missing_timestamps), 20)]
                for i, chunk in enumerate(chunks):
                    chunk_start = datetime.fromtimestamp(chunk[0]/1000)
                    chunk_end = datetime.fromtimestamp(chunk[-1]/1000)
                    
                    # Добавляем небольшой запас к интервалу
                    chunk_start = chunk_start - timedelta(minutes=15)
                    chunk_end = chunk_end + timedelta(minutes=15)
                    
                    logger.info(f"🔄 {symbol}: загрузка данных {chunk_start.strftime('%Y-%m-%d %H:%M')} - {chunk_end.strftime('%Y-%m-%d %H:%M')} (чанк {i+1}/{len(chunks)})")
                    
                    # Загружаем данные за короткий период
                    chunk_df = fetch_symbol_data(session, symbol, chunk_start, chunk_end)
                    
                    if chunk_df is not None and not chunk_df.empty:
                        # Фильтруем только по нужным timestamp
                        chunk_ts_set = set(chunk)
                        chunk_df = chunk_df[chunk_df['timestamp'].isin(chunk_ts_set)]
                        
                        # Добавляем к общему DataFrame
                        all_data = pd.concat([all_data, chunk_df])
                
                # Если получили данные, сохраняем
                if not all_data.empty:
                    df = all_data
                else:
                    logger.warning(f"⚠️ {symbol}: Не удалось загрузить ни одну из пропущенных меток")
                    return []
            else:
                # При большом числе пропусков разбиваем их на оптимальные интервалы (чанки)
                logger.info(f"🔍 {symbol}: Группировка {len(missing_timestamps)} пропущенных интервалов в оптимальные чанки")
                
                # Группируем пропуски в смежные интервалы
                chunks = group_timestamps_into_chunks(missing_timestamps)
                logger.info(f"📊 {symbol}: Создано {len(chunks)} оптимальных интервалов для загрузки")
                
                # Создаем пустой DataFrame для накопления данных
                all_data = pd.DataFrame()
                
                # Загружаем данные для каждого интервала
                for i, (chunk_start, chunk_end) in enumerate(chunks):
                    logger.info(f"🔄 {symbol}: загрузка данных {chunk_start.strftime('%Y-%m-%d %H:%M')} - {chunk_end.strftime('%Y-%m-%d %H:%M')} (чанк {i+1}/{len(chunks)})")
                    
                    # Загружаем данные за интервал
                    chunk_df = fetch_symbol_data(session, symbol, chunk_start, chunk_end)
                    
                    if chunk_df is not None and not chunk_df.empty:
                        # Фильтруем только по нужным timestamp
                        missing_ts_set = set(missing_timestamps)
                        filtered_df = chunk_df[chunk_df['timestamp'].isin(missing_ts_set)]
                        
                        if not filtered_df.empty:
                            # Добавляем к общему DataFrame
                            all_data = pd.concat([all_data, filtered_df])
                            logger.info(f"✅ {symbol}: Загружено {len(filtered_df)} интервалов из чанка {i+1}")
                        else:
                            logger.info(f"ℹ️ {symbol}: В чанке {i+1} нет требуемых данных после фильтрации")
                
                # Если получили данные, используем их
                if not all_data.empty:
                    df = all_data
                    logger.info(f"🔄 {symbol}: Всего загружено {len(df)} интервалов из {len(missing_timestamps)} пропущенных")
                else:
                    logger.warning(f"⚠️ {symbol}: Не удалось загрузить ни один из пропущенных интервалов")
                    return []
        else:
            # Стандартная загрузка за весь период
            logger.info(f"🔍 {symbol}: Загрузка данных с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")
            logger.info(f"🔄 {symbol}: загрузка данных {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
            
            df = fetch_symbol_data(session, symbol, start_date, end_date)
            
            if df is None or df.empty:
                logger.warning(f"⚠️ {symbol}: нет данных за период")
                return []  # Считаем успешной обработкой (just no data)
        
        # Сохраняем во временные файлы по месяцам
        temp_files = save_to_temp_file(symbol, df)
        
        if temp_files:
            logger.info(f"✅ {symbol}: загружено {len(df)} интервалов")
            return temp_files
        else:
            logger.error(f"❌ {symbol}: Ошибка при сохранении временных файлов")
            return []
            
    except Exception as e:
        logger.error(f"❌ {symbol}: ошибка при обработке: {e}")
        return []

def main_optimized(start_date_str: str = "2022-01-01", end_date_str: str = "2025-07-01", symbols_limit: int = None, api_key: str = None, api_secret: str = None, incremental: bool = False):
    """Основная функция программы для сохранения в оптимизированном формате с двухфазной схемой.
    
    Args:
        start_date_str: Строка с начальной датой в формате YYYY-MM-DD
        end_date_str: Строка с конечной датой в формате YYYY-MM-DD
        symbols_limit: Ограничение количества валютных пар для загрузки
        api_key: API ключ Bybit
        api_secret: API секрет Bybit
    """
    try:
        logger.info("🚀 Запуск загрузчика данных в оптимизированную структуру")
        
        # Проверяем API ключи
        if not validate_api_keys():
            logger.error("❌ Ошибка валидации API ключей")
            return
            
        # Запоминаем время начала выполнения
        start_time = time.time()
        
        # Создаем директорию для данных если не существует
        DATA_DIR.mkdir(parents=True, exist_ok=True)
            
        # Получаем список пар с API
        logger.info("📋 Получение списка пар напрямую с Bybit API")
        symbols = fetch_bybit_markets(category="spot", save_to_file=False)
            
        if not symbols:
            logger.error("❌ Список символов пуст, невозможно продолжить")
            return
            
        # Применяем ограничение количества валют если указано
        if symbols_limit is not None and symbols_limit > 0:
            logger.info(f"🔍 Применяем ограничение: загрузка только {symbols_limit} валютных пар для тестирования")
            symbols = symbols[:symbols_limit]
            logger.info(f"✅ Список валютных пар ограничен до {len(symbols)} пар")
            
        # Режим инкрементальной загрузки - проверяем существующие данные
        if incremental:
            logger.info("🔄 Режим инкрементальной загрузки: анализируем существующие данные")
            
            # Создаем список всех ожидаемых временных меток (15-минутные интервалы)
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
            
            # Анализируем недостающие данные за указанный период используя быстрый PyArrow метод
            logger.info(f"🔍 Анализируем недостающие данные с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")
            missing_data_map = analyze_missing_data_fast(symbols, start_date, end_date)
            
            # Фильтруем список валют, оставляем только те, где есть пропущенные данные
            symbols_with_missing_data = [symbol for symbol in symbols if missing_data_map.get(symbol, [])]
            
            if not symbols_with_missing_data:
                logger.info("✅ Все данные уже загружены, нет недостающих интервалов")
                return
            
            logger.info(f"🔄 Найдено {len(symbols_with_missing_data)} валютных пар с недостающими данными")
            symbols = symbols_with_missing_data
            
        # Получаем API ключи из глобальных переменных
        global MAINNET_API_KEYS, MAINNET_API_SECRETS
        
        # Создаем пул HTTP сессий для распределения нагрузки
        sessions = []
        # Используем все доступные API ключи
        for key, secret in zip(MAINNET_API_KEYS, MAINNET_API_SECRETS):
            try:
                session = HTTP(
                    testnet=False,
                    api_key=key,
                    api_secret=secret
                )
                sessions.append(session)
                logger.info(f"✅ Создана HTTP-сессия {len(sessions)}")
            except Exception as e:
                logger.error(f"❌ Ошибка создания сессии: {e}")
        
        # Проверяем, что сессии созданы
        if not sessions:
            logger.error("❌ Не удалось создать ни одной HTTP сессии")
            return

        # Создаем директории для данных и временных файлов
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        TEMP_DIR.mkdir(parents=True, exist_ok=True)

        # Определяем период для загрузки - фиксированный диапазон (01.01.2022 - 01.07.2025)
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        except ValueError as e:
            logger.error(f"❌ Ошибка в формате дат: {e}")
            logger.info("⚠️ Использую стандартный диапазон: 01.01.2022 - 01.07.2025")
            start_date = datetime.strptime("2022-01-01", "%Y-%m-%d")
            end_date = datetime.strptime("2025-07-01", "%Y-%m-%d")
        
        logger.info(f"📅 Период загрузки: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"🔢 Количество символов для загрузки: {len(symbols)}")
        
        # Фаза 1: Сохраняем данные во временные файлы параллельно
        logger.info("📊 ФАЗА 1: Сохранение данных во временные файлы")
        temp_files_by_month = {}  # Словарь месяц -> список временных файлов
        
        # Инициализация счетчиков для инкрементального объединения
        temp_files_by_month = {}  # Словарь месяц -> список временных файлов
        processed_count = 0       # Счетчик успешно обработанных валют
        batch_size = 10           # Размер батча для инкрементального объединения
        success_count = 0
        error_count = 0
        processed_files = set()   # Множество для отслеживания уже обработанных файлов
        
        # Создаем пул мультипроцессинга
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            # Запускаем потоки для обработки каждого символа
            futures = {}
            for symbol in symbols:
                # В режиме дозагрузки передаём список недостающих timestamp
                if incremental and symbol in missing_data_map:
                    future = executor.submit(
                        process_symbol_optimized,
                        symbol=symbol,
                        sessions=sessions,
                        start_date=start_date,
                        end_date=end_date,
                        missing_timestamps=missing_data_map[symbol]
                    )
                else:
                    future = executor.submit(
                        process_symbol_optimized,
                        symbol=symbol,
                        sessions=sessions,
                        start_date=start_date,
                        end_date=end_date
                    )
                futures[future] = symbol
            
            # Обработка результатов с инкрементальным объединением
            for future, symbol in futures.items():
                try:
                    # Получаем список временных файлов от задачи
                    temp_files = future.result()
                    
                    if temp_files:
                        success_count += 1
                        
                        # Группируем файлы по месяцам
                        for temp_file in temp_files:
                            path = Path(temp_file)
                            filename = path.name
                            parts = filename.split('_')
                            if len(parts) >= 3:
                                year, month = parts[-2], parts[-1].split('.')[0]
                                month_key = f"{year}-{month}"
                                
                                if month_key not in temp_files_by_month:
                                    temp_files_by_month[month_key] = []
                                    
                                temp_files_by_month[month_key].append(temp_file)
                        
                        processed_count += 1
                        
                        # Каждые batch_size успешно обработанных валют объединяем временные файлы
                        if processed_count % batch_size == 0:
                            logger.info(f"📊 Инкрементальное объединение после {processed_count} обработанных валют")
                            
                            # Инкрементальное объединение
                            success_merge = 0
                            error_merge = 0
                            processed_month_keys = []
                            
                            for month_key, files in temp_files_by_month.items():
                                # Фильтруем только новые файлы, которые еще не были обработаны
                                new_files = [f for f in files if f not in processed_files]
                                if not new_files:
                                    continue
                                    
                                year, month = month_key.split('-')
                                logger.info(f"📅 Инкрементальная обработка данных за {year}-{month}: {len(new_files)} временных файлов")
                                
                                if merge_temp_files_to_monthly(year, month):
                                    success_merge += 1
                                    # Добавляем обработанные файлы в множество
                                    processed_files.update(new_files)
                                    processed_month_keys.append(month_key)
                                else:
                                    error_merge += 1
                            
                            logger.info(f"✅ Инкрементальное объединение завершено: {success_merge} месяцев успешно обработано")
                    else:
                        error_count += 1
                except Exception as e:
                    logger.error(f"❌ Ошибка при обработке {symbol}: {e}")
                    error_count += 1
            
            logger.info(f"✅ Фаза 1 завершена: {success_count} успешно, {error_count} с ошибками")
            
            # Финальная фаза: объединяем оставшиеся временные файлы
            remaining_files = False
            for month_key, files in temp_files_by_month.items():
                # Проверяем, остались ли необработанные файлы
                new_files = [f for f in files if f not in processed_files]
                if new_files:
                    remaining_files = True
                    break
                    
            if remaining_files:
                logger.info("📊 Финальное объединение оставшихся временных файлов")
                
                success_merge = 0
                error_merge = 0
                
                for month_key, files in temp_files_by_month.items():
                    # Обрабатываем только те файлы, которые еще не были обработаны
                    if any(f not in processed_files for f in files):
                        year, month = month_key.split('-')
                        logger.info(f"📅 Финальная обработка данных за {year}-{month}")
                        
                        if merge_temp_files_to_monthly(year, month):
                            success_merge += 1
                        else:
                            error_merge += 1
                
                logger.info(f"✅ Финальное объединение завершено: {success_merge} месяцев успешно обработано, {error_merge} с ошибками")
            else:
                logger.warning("⚠️ Нет данных для объединения в итоговые файлы")

            end_time = time.time()
            elapsed = end_time - start_time
            logger.info(f"✅ Загрузка завершена за {elapsed:.2f} секунд")
            logger.info(f"📊 Статистика: обработано успешно {success_count}, с ошибками {error_count}")
            
            # Финальная проверка на дубликаты в конце загрузки
            if 'modules_imported' in globals() and modules_imported:
                logger.info("🔍 Запуск финальной проверки на дубликаты...")
                try:
                    stats = check_and_fix_duplicates(str(DATA_DIR), fix=True)
                    logger.info(f"✅ Финальная проверка завершена")
                    logger.info(f"📊 Результаты проверки: проверено {stats['total_files']} файлов, найдено {stats['files_with_duplicates']} файлов с дубликатами, удалено {stats['duplicates_removed']} дубликатов из {stats['total_rows']} строк")
                except Exception as e:
                    logger.error(f"❌ Ошибка при проверке дубликатов: {e}")
            else:
                logger.warning("⚠️ Модуль проверки дубликатов недоступен, финальная проверка не выполнена")

        logger.info("✨ Загрузка данных завершена")
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")

def signal_handler(signum, frame):
    logger.info("🛑 Получен сигнал прерывания. Завершение работы...")
    sys.exit(0)


def analyze_missing_data_fast(symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, List[int]]:
    """Анализирует недостающие данные в указанном диапазоне дат, используя оптимизированный метод с PyArrow.
    
    Args:
        symbols: Список валютных пар
        start_date: Начальная дата периода
        end_date: Конечная дата периода
        
    Returns:
        Словарь, где ключ - символ, значение - список недостающих timestamp
    """
    import pyarrow.dataset as ds
    import pyarrow.compute as pc
    
    logger.info(f"🔍 Анализируем недостающие данные с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")
    
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    
    # Проверяем существование директории данных
    if not DATA_DIR.exists():
        # Если директории нет, все метки недостающие для всех символов
        logger.warning("⚠️ Директория данных не существует, будем загружать все данные")
        expected_timestamps = list(range(start_ts, end_ts + 1, 15 * 60 * 1000))
        return {symbol: expected_timestamps for symbol in symbols}
    
    try:
        # 1️⃣ Ленивая «склейка» всех файлов
        logger.info("🔍 Создание dataset из parquet-файлов (только метаданные)...")
        dataset = ds.dataset(str(DATA_DIR), format="parquet")  # читает ТОЛЬКО метаданные
        
        # 2️⃣ Фильтруем сразу на уровне скана
        logger.info("🔍 Создание сканера с фильтрацией по времени...")
        scanner = dataset.scanner(
            columns=["timestamp", "symbol"],
            filter=(
                (pc.field("timestamp") >= start_ts) &
                (pc.field("timestamp") <= end_ts)
            )
        )
        
        # 3️⃣ Одним вызовом получаем Arrow-таблицу
        logger.info("📊 Сканирование parquet-файлов на низком уровне через PyArrow...")
        tbl = scanner.to_table()  # читает ровно два столбца, минуя pandas
        logger.info(f"✅ Прочитано {tbl.num_rows} записей из всех parquet-файлов")
        
        # Если нет данных вообще
        if tbl.num_rows == 0:
            logger.warning("⚠️ В существующих файлах нет данных за указанный период")
            expected_timestamps = list(range(start_ts, end_ts + 1, 15 * 60 * 1000))
            return {symbol: expected_timestamps for symbol in symbols}
        
        have = {}
        try:
            # 4️⃣ Пытаемся использовать C-агрегацию hash_set
            logger.info("🔄 Группировка данных с использованием hash_set агрегации...")
            groups = (
                tbl.group_by("symbol")
                   .aggregate([("timestamp", "hash_set")])
                   .to_pydict()
            )
            have = {sym: set(ts) for sym, ts in zip(groups["symbol"], groups["timestamp_hash_set"])}
        except Exception as e:
            # Если hash_set не поддерживается, используем pandas
            logger.warning(f"⚠️ Агрегация hash_set не поддерживается ({str(e)}), используем pandas")
            logger.info("🔄 Группировка данных через pandas...")
            
            df = tbl.to_pandas()
            for symbol, group in df.groupby('symbol'):
                have[symbol] = set(group['timestamp'].unique())
        
        # 5️⃣ Для каждой валюты определяем реальный диапазон торговли
        logger.info("📅 Определение реальных периодов торговли для каждой валюты...")
        
        # Для каждой валюты находим первый и последний timestamp
        real_ranges = {}
        
        # Берём только валюты, которые есть в данных
        for sym in have.keys():
            if sym in symbols and have[sym]:
                # Определяем диапазон существующих данных
                sym_min = min(have[sym])
                sym_max = max(have[sym])
                
                # Фиксируем диапазон глобальными рамками
                actual_start = max(sym_min, start_ts)
                actual_end = min(sym_max, end_ts)
                
                real_ranges[sym] = (actual_start, actual_end)
                logger.info(f"📆 {sym}: период торговли с {datetime.fromtimestamp(sym_min/1000).strftime('%Y-%m-%d')} по {datetime.fromtimestamp(sym_max/1000).strftime('%Y-%m-%d')}")
        
        # 6️⃣ Список ожиданий делаем один раз с фильтрацией
        logger.info("🔄 Создание списка недостающих временных меток с учетом реальных периодов торговли...")
        
        expected = {}
        for sym in symbols:
            # Если есть реальный диапазон - ищем пропуски внутри него
            if sym in real_ranges:
                actual_start, actual_end = real_ranges[sym]
                # Проверяем интервалы только внутри реального диапазона
                missing = [t for t in range(actual_start, actual_end + 1, 15 * 60 * 1000)
                          if t not in have.get(sym, set())]
                
                if missing:
                    expected[sym] = missing
            # Если вообще нет данных по этой валюте - скачиваем за весь период
            elif sym not in have or not have[sym]:
                expected[sym] = list(range(start_ts, end_ts + 1, 15 * 60 * 1000))
        
        # Возвращаем только символы, у которых есть недостающие метки
        result = {k: v for k, v in expected.items() if v}
        
        # Логируем результаты для информации
        for sym, missing in result.items():
            all_expected = list(range(start_ts, end_ts + 1, 15 * 60 * 1000))
            logger.info(f"⚠️ {sym}: отсутствует {len(missing)} из {len(all_expected)} интервалов ({len(missing)/len(all_expected)*100:.1f}%)")
            
        return result
        
    except Exception as e:
        logger.error(f"❌ Ошибка при анализе данных через PyArrow: {str(e)}")
        logger.warning("⚠️ Переключаемся на стандартный метод анализа (медленнее)")
        return analyze_missing_data(symbols, start_date, end_date)

def analyze_missing_data(symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, List[int]]:
    """Анализирует недостающие данные в указанном диапазоне дат.
    
    Args:
        symbols: Список валютных пар
        start_date: Начальная дата периода
        end_date: Конечная дата периода
        
    Returns:
        Словарь, где ключ - символ, значение - список недостающих timestamp
    """
    logger.info(f"🔍 Анализируем недостающие данные с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")
    
    # Создаем список всех временных меток в 15-минутных интервалах
    expected_timestamps = []
    current_date = start_date
    while current_date <= end_date:
        # Для каждого дня создаем 15-минутные интервалы (96 интервалов в день)
        for hour in range(24):
            for minute in range(0, 60, 15):
                dt = current_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                ts = int(dt.timestamp() * 1000)  # В миллисекундах как в Бибит
                expected_timestamps.append(ts)
        current_date += timedelta(days=1)
    
    logger.info(f"📆 Создано {len(expected_timestamps)} ожидаемых временных меток")
    
    # Получаем список файлов в data_downloaded
    result = {}
    
    # Проверяем существующие паркет-файлы
    parquet_files = []
    for year_dir in DATA_DIR.glob("year=*"):
        for month_dir in year_dir.glob("month=*"):
            for parquet_file in month_dir.glob("*.parquet"):
                parquet_files.append(parquet_file)
    
    if not parquet_files:
        # Если файлов нет, все метки недостающие для всех символов
        logger.warning("⚠️ Нет существующих файлов данных, будем загружать все данные")
        for symbol in symbols:
            result[symbol] = expected_timestamps
        return result
    
    logger.info(f"🗃️ Найдено {len(parquet_files)} parquet-файлов")
    
    # Создаем сет существующих комбинаций (timestamp, symbol)
    existing_data = set()
    
    # Читаем и объединяем данные из всех файлов
    for i, parquet_file in enumerate(parquet_files):
        try:
            if i % 10 == 0:
                logger.info(f"📈 Прогресс: {i}/{len(parquet_files)} ({i/len(parquet_files)*100:.1f}%)")
                
            df = pd.read_parquet(parquet_file)
            if df.empty:
                continue
                
            # Добавляем все комбинации timestamp-symbol в сет
            for _, row in df.iterrows():
                existing_data.add((row['timestamp'], row['symbol']))
        except Exception as e:
            logger.error(f"❌ Ошибка при чтении файла {parquet_file}: {e}")
    
    logger.info(f"📊 Из parquet-файлов получено {len(existing_data)} уникальных записей (timestamp, symbol)")
    
    # Для каждого символа определяем недостающие временные метки
    for symbol in symbols:
        # Находим недостающие временные метки для этого символа
        missing_timestamps = [
            ts for ts in expected_timestamps 
            if (ts, symbol) not in existing_data
        ]
        
        if missing_timestamps:
            result[symbol] = missing_timestamps
            logger.info(f"⚠️ {symbol}: отсутствует {len(missing_timestamps)} из {len(expected_timestamps)} интервалов ({len(missing_timestamps)/len(expected_timestamps)*100:.1f}%)")
    
    return result
    return result

logger.info("✨ Загрузка данных завершена")
        
def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description="Скачивание и обработка данных Bybit API")
    parser.add_argument(
        "--start-date",
        type=str,
        default="2022-01-01",
        help="Начальная дата в формате YYYY-MM-DD (по умолчанию: 2022-01-01)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-07-01",
        help="Конечная дата в формате YYYY-MM-DD (по умолчанию: 2025-07-01)"
    )
    parser.add_argument(
        "--markets-only",
        action="store_true",
        help="Только получить список рынков без загрузки данных"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Режим дозагрузки: загружать только недостающие данные (15-минутные интервалы)"
    )
    parser.add_argument(
        "--symbols-limit",
        type=int,
        default=None,
        help="Ограничение количества валютных пар для загрузки (для тестирования)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API ключ Bybit (опционально, но рекомендуется для избежания ограничений)"
    )
    parser.add_argument(
        "--api-secret",
        type=str,
        default=None,
        help="API секрет Bybit (опционально, но рекомендуется для избежания ограничений)"
    )
    
    return parser.parse_args()

# ================== ТОЧКА ВХОДА ==================
if __name__ == "__main__":
    # Парсинг аргументов командной строки
    args = parse_args()
    
    if args.markets_only:
        # Режим только получения списка пар
        logger.info("📋 Запуск в режиме получения только списка пар")
        symbols = fetch_bybit_markets(category="spot", save_to_file=True)
        logger.info(f"✅ Получено и сохранено {len(symbols)} пар")
    else:
        # Стандартный режим загрузки данных
        main_optimized(
            start_date_str=args.start_date,
            end_date_str=args.end_date,
            symbols_limit=args.symbols_limit,
            api_key=args.api_key,
            api_secret=args.api_secret,
            incremental=args.incremental
        )
