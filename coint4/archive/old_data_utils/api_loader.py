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
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
import cProfile
import io
import pstats
import shutil
import psutil
import gc
from logging.handlers import RotatingFileHandler
from threading import Lock
import signal
import sys

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
DATA_DIR = Path("data")
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

# ================== ПРОВЕРКА СУЩЕСТВУЮЩИХ ДАННЫХ ==================

def check_existing_data(symbol: str, start_date: datetime, end_date: datetime) -> List[Tuple[datetime, datetime]]:
    """
    Проверяет какие данные уже существуют для символа и возвращает список недостающих периодов.
    
    Args:
        symbol: Символ для проверки
        start_date: Начальная дата периода
        end_date: Конечная дата периода
        
    Returns:
        List[Tuple[datetime, datetime]]: Список недостающих периодов
    """
    try:
        symbol_path = DATA_DIR / symbol
        if not symbol_path.exists():
            # Если папки нет, нужно загрузить весь период
            return [(start_date, end_date)]
            
        # Читаем существующие данные
        try:
            dataset = ds.dataset(symbol_path, format="parquet", partitioning="hive")
            table = dataset.to_table()
            df = table.to_pandas()
            
            if df.empty:
                return [(start_date, end_date)]
                
            # Проверяем покрытие данных
            df['timestamp'] = pd.to_datetime(df['ts_ms'], unit='ms')
            existing_dates = set(df['timestamp'].dt.date)
            
            # Генерируем список всех дат в периоде
            current_date = start_date.date()
            end_date_only = end_date.date()
            needed_dates = []
            
            while current_date <= end_date_only:
                if current_date not in existing_dates:
                    needed_dates.append(current_date)
                current_date += timedelta(days=1)
            
            if not needed_dates:
                logger.info(f"✅ {symbol}: все данные уже существуют")
                return []
                
            # Группируем недостающие даты в непрерывные периоды
            missing_periods = []
            if needed_dates:
                period_start = needed_dates[0]
                period_end = needed_dates[0]
                
                for date in needed_dates[1:]:
                    if date == period_end + timedelta(days=1):
                        period_end = date
        else:
                        missing_periods.append((
                            datetime.combine(period_start, datetime.min.time()),
                            datetime.combine(period_end, datetime.max.time())
                        ))
                        period_start = date
                        period_end = date
                        
                missing_periods.append((
                    datetime.combine(period_start, datetime.min.time()),
                    datetime.combine(period_end, datetime.max.time())
                ))
                
            logger.info(f"📋 {symbol}: нужно загрузить {len(needed_dates)} дней в {len(missing_periods)} периодах")
            return missing_periods
            
        except Exception as e:
            logger.warning(f"⚠️ {symbol}: ошибка чтения существующих данных: {e}")
            return [(start_date, end_date)]
        
    except Exception as e:
        logger.error(f"❌ {symbol}: ошибка проверки данных: {e}")
        return [(start_date, end_date)]

# ================== ЗАГРУЗКА ДАННЫХ С API ==================

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
        df['timestamp'] = pd.to_datetime(df['ts_ms'], unit='ms')
        
                    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Добавляем партиционирующие колонки
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        
        # Удаляем дубликаты и сортируем
        df = df.drop_duplicates(subset=['ts_ms']).sort_values('ts_ms')
        
        logger.info(f"✅ {symbol}: загружено {len(df)} записей")
                        return df
                        
                except Exception as e:
        logger.error(f"❌ {symbol}: критическая ошибка загрузки: {e}")
                    return None
        
# ================== СОХРАНЕНИЕ В ПАРТИЦИОНИРОВАННУЮ СТРУКТУРУ ==================

def save_partitioned_data(symbol: str, df: pd.DataFrame) -> bool:
    """
    Сохраняет данные в партиционированную структуру.
    
    Args:
        symbol: Символ
        df: DataFrame с данными
        
    Returns:
        bool: True если успешно сохранено
    """
    try:
        if df.empty:
            logger.warning(f"⚠️ {symbol}: пустой DataFrame, пропускаем")
                return False
        
        # Создаем папку для символа
        symbol_path = DATA_DIR / symbol
        symbol_path.mkdir(parents=True, exist_ok=True)
        
        # Подготавливаем данные для партиционирования
        # Создаем схему для PyArrow
        schema = pa.schema([
            ('ts_ms', pa.int64()),
            ('timestamp', pa.timestamp('ms')),
            ('open', pa.float64()),
            ('high', pa.float64()),
            ('low', pa.float64()),
            ('close', pa.float64()),
            ('volume', pa.float64()),
            ('turnover', pa.float64()),
            ('year', pa.int32()),
            ('month', pa.int32()),
            ('day', pa.int32())
        ])
        
        # Конвертируем в PyArrow Table
        table = pa.Table.from_pandas(df, schema=schema)
        
        # Сохраняем с партиционированием
        ds.write_dataset(
            table,
            base_dir=str(symbol_path),
            format='parquet',
            partitioning=['year', 'month', 'day'],
            partitioning_flavor='hive',
            existing_data_behavior='overwrite_or_ignore',  # Не перезаписываем существующие файлы
            compression='snappy',
            max_rows_per_file=10000
        )
        
        logger.info(f"💾 {symbol}: сохранено в партиционированную структуру")
        return True
        
                except Exception as e:
        logger.error(f"❌ {symbol}: ошибка сохранения: {e}")
        return False

# ================== ОСНОВНАЯ ЛОГИКА ==================

def process_symbol(symbol: str, sessions: List[HTTP], start_date: datetime, end_date: datetime) -> bool:
    """
    Обрабатывает один символ: проверяет существующие данные и загружает недостающие.
    
    Args:
        symbol: Символ для обработки
        sessions: Список HTTP сессий
        start_date: Начальная дата
        end_date: Конечная дата
        
    Returns:
        bool: True если успешно обработан
    """
    try:
        # Проверяем какие данные нужно загрузить
        missing_periods = check_existing_data(symbol, start_date, end_date)
        
        if not missing_periods:
            logger.info(f"✅ {symbol}: все данные уже существуют")
            return True
        
        # Выбираем сессию для этого символа
        session = random.choice(sessions)
        
        success = True
        total_records = 0
        
        # Загружаем недостающие периоды
        for period_start, period_end in missing_periods:
            df = fetch_symbol_data(session, symbol, period_start, period_end)
            
            if df is not None and not df.empty:
                if save_partitioned_data(symbol, df):
                    total_records += len(df)
                else:
                    success = False
            else:
                logger.warning(f"⚠️ {symbol}: не удалось загрузить период {period_start} - {period_end}")
                success = False
        
        if success:
            logger.info(f"🎉 {symbol}: успешно обработан, загружено {total_records} записей")
        else:
            logger.error(f"❌ {symbol}: обработка завершена с ошибками")
            
        return success
        
    except Exception as e:
        logger.error(f"❌ {symbol}: критическая ошибка обработки: {e}")
        return False

def main():
    """Основная функция программы."""
    try:
        logger.info("🚀 Запуск загрузчика данных в партиционированную структуру")
        
        # Проверяем API ключи
        if not validate_api_keys():
            raise ValueError("Некорректная конфигурация API ключей")
        
        # Загружаем список символов
        symbols = load_symbols_from_markets(MARKETS_FILE)
        if not symbols:
            raise ValueError(f"Не удалось загрузить символы из {MARKETS_FILE}")
        
        logger.info(f"📋 Будет обработано {len(symbols)} символов")
        
        # Создаем HTTP сессии
        sessions = []
        for key, secret in zip(MAINNET_API_KEYS, MAINNET_API_SECRETS):
            try:
                session = HTTP(api_key=key, api_secret=secret)
                sessions.append(session)
            except Exception as e:
                logger.error(f"❌ Ошибка создания сессии: {e}")
        
        if not sessions:
            raise ValueError("Не удалось создать ни одной сессии API")
        
        logger.info(f"🔗 Создано {len(sessions)} API сессий")
        
        # Определяем период загрузки
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # Загружаем за год
        
        logger.info(f"📅 Период загрузки: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
        
        # Создаем директорию данных
        DATA_DIR.mkdir(exist_ok=True)
        
        # Обрабатываем символы параллельно
        successful = 0
        failed = 0
        
        max_workers = min(len(sessions), 10)  # Ограничиваем количество потоков
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Создаем задачи
            future_to_symbol = {
                executor.submit(process_symbol, symbol, sessions, start_date, end_date): symbol
                for symbol in symbols
            }
            
            # Собираем результаты
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    if future.result():
                        successful += 1
        else:
                        failed += 1
    except Exception as e:
                    logger.error(f"❌ {symbol}: исключение в потоке: {e}")
                    failed += 1
                
                # Логируем прогресс каждые 10 символов
                total_processed = successful + failed
                if total_processed % 10 == 0:
                    logger.info(f"📊 Прогресс: {total_processed}/{len(symbols)} ({successful} успешно, {failed} ошибок)")
        
        # Финальная статистика
        logger.info("🎯 Загрузка завершена!")
        logger.info(f"✅ Успешно обработано: {successful}")
        logger.info(f"❌ Ошибок: {failed}")
        logger.info(f"📈 Успешность: {(successful / len(symbols) * 100):.1f}%")
        
        # Закрываем сессии
    for session in sessions:
        try:
                if hasattr(session, "close"):
                session.close()
        except Exception as e:
                logger.error(f"❌ Ошибка закрытия сессии: {e}")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}")
        sys.exit(1)

# ================== ОБРАБОТЧИКИ СИГНАЛОВ ==================

def signal_handler(signum, frame):
    logger.info("⏹️ Получен сигнал завершения, выполняем корректное завершение...")
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ================== ТОЧКА ВХОДА ==================

if __name__ == '__main__':
    main()
