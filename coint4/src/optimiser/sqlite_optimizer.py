"""
Оптимизация SQLite для Optuna с настройкой PRAGMA.
Решает проблемы конкурентного доступа и производительности.
"""

import sqlite3
from typing import Optional, Dict, Any
import logging
from sqlalchemy import event
from sqlalchemy.pool import NullPool
import optuna
from optuna.storages import RDBStorage

logger = logging.getLogger(__name__)


def setup_sqlite_connection(dbapi_conn, connection_record):
    """
    Настраивает SQLite соединение с оптимальными PRAGMA параметрами.
    Вызывается автоматически SQLAlchemy при создании соединения.
    
    Args:
        dbapi_conn: SQLite connection объект
        connection_record: SQLAlchemy connection record
    """
    cursor = dbapi_conn.cursor()
    
    # КРИТИЧНО: WAL mode для параллельного доступа
    cursor.execute("PRAGMA journal_mode=WAL")
    
    # Оптимизация производительности
    cursor.execute("PRAGMA synchronous=NORMAL")  # Быстрее чем FULL, но безопасно
    cursor.execute("PRAGMA cache_size=-64000")   # 64MB кэш
    cursor.execute("PRAGMA temp_store=MEMORY")   # Временные данные в памяти
    
    # Таймауты и блокировки
    cursor.execute("PRAGMA busy_timeout=60000")    # 60 секунд таймаут
    cursor.execute("PRAGMA wal_autocheckpoint=1000")  # Авточекпоинт каждые 1000 страниц
    cursor.execute("PRAGMA mmap_size=268435456")   # 256MB mmap
    
    # Анализ и оптимизация
    cursor.execute("PRAGMA automatic_index=ON")    # Автоматические индексы
    cursor.execute("PRAGMA optimize")              # Оптимизация запросов
    
    cursor.close()
    
    logger.debug("✅ SQLite PRAGMA настройки применены")


def create_optuna_storage(
    db_path: str,
    n_jobs: int = 1,
    enable_heartbeat: bool = True
) -> RDBStorage:
    """
    Создает оптимизированное хранилище Optuna для SQLite.
    
    Args:
        db_path: Путь к SQLite базе данных
        n_jobs: Количество параллельных процессов
        enable_heartbeat: Включить heartbeat для обнаружения зависших trials
        
    Returns:
        Настроенное RDBStorage для Optuna
    """
    import sqlite3
    from pathlib import Path
    
    # Создаем директорию если её нет
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Настраиваем WAL режим напрямую
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # КРИТИЧНО: WAL mode для параллельного доступа
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=-64000")  # 64MB кэш
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.execute("PRAGMA busy_timeout=60000")  # 60 секунд
        cursor.execute("PRAGMA wal_autocheckpoint=1000")
        cursor.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
        
        conn.commit()
        conn.close()
        logger.info("✅ SQLite настроен с WAL режимом")
    except Exception as e:
        logger.warning(f"⚠️ Не удалось настроить WAL: {e}")
    
    # Формируем URL для SQLAlchemy
    db_url = f"sqlite:///{db_path}"

    engine_kwargs = {
        "poolclass": NullPool,
        "connect_args": {
            "timeout": 60,
            "check_same_thread": False,
        },
    }

    heartbeat_interval = 60 if enable_heartbeat else None
    grace_period = 120 if enable_heartbeat else None

    storage = RDBStorage(
        url=db_url,
        engine_kwargs=engine_kwargs,
        heartbeat_interval=heartbeat_interval,
        grace_period=grace_period,
    )

    try:
        if getattr(storage, "engine", None) is not None:
            event.listen(storage.engine, "connect", setup_sqlite_connection)
    except Exception as e:
        logger.warning(f"⚠️ Не удалось подключить PRAGMA listener: {e}")
    
    if n_jobs == 1:
        logger.info("📊 Используем однопоточный режим для SQLite")
    else:
        logger.info(f"📊 Используем многопоточный режим для {n_jobs} процессов")
    
    logger.info(f"✅ Optuna storage создан: {db_path}")
    
    return storage


def optimize_existing_database(db_path: str) -> None:
    """
    Оптимизирует существующую SQLite базу данных.
    
    Args:
        db_path: Путь к базе данных
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Анализируем таблицы для оптимизации запросов
        cursor.execute("ANALYZE")
        
        # Перестраиваем индексы
        cursor.execute("REINDEX")
        
        # Освобождаем неиспользуемое пространство
        cursor.execute("VACUUM")
        
        # Оптимизируем запросы
        cursor.execute("PRAGMA optimize")
        
        conn.commit()
        logger.info(f"✅ База данных оптимизирована: {db_path}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка оптимизации базы данных: {e}")
        conn.rollback()
    finally:
        conn.close()


def create_optimized_study(
    study_name: str,
    db_path: str,
    direction: str = "maximize",
    n_jobs: int = 1,
    sampler: Optional[optuna.samplers.BaseSampler] = None,
    pruner: Optional[optuna.pruners.BasePruner] = None
) -> optuna.Study:
    """
    Создает оптимизированное Optuna study с настроенным SQLite storage.
    
    Args:
        study_name: Имя study
        db_path: Путь к SQLite базе
        direction: Направление оптимизации ("maximize" или "minimize")
        n_jobs: Количество параллельных процессов
        sampler: Optuna sampler
        pruner: Optuna pruner
        
    Returns:
        Настроенное Optuna study
    """
    # Создаем оптимизированное хранилище
    storage = create_optuna_storage(db_path, n_jobs=n_jobs)
    
    # Создаем или загружаем study
    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=False
        )
        logger.info(f"✅ Создано новое study: {study_name}")
    except optuna.exceptions.DuplicatedStudyError:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            pruner=pruner
        )
        logger.info(f"📦 Загружено существующее study: {study_name}")
    
    return study


def get_sqlite_stats(db_path: str) -> Dict[str, Any]:
    """
    Получает статистику SQLite базы данных.
    
    Args:
        db_path: Путь к базе данных
        
    Returns:
        Словарь со статистикой
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    stats = {}
    
    try:
        # Размер базы данных
        cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
        stats['size_bytes'] = cursor.fetchone()[0]
        
        # Количество таблиц
        cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table'")
        stats['table_count'] = cursor.fetchone()[0]
        
        # Количество индексов
        cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='index'")
        stats['index_count'] = cursor.fetchone()[0]
        
        # WAL mode статус
        cursor.execute("PRAGMA journal_mode")
        stats['journal_mode'] = cursor.fetchone()[0]
        
        # Размер кэша
        cursor.execute("PRAGMA cache_size")
        stats['cache_size'] = cursor.fetchone()[0]
        
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")
    finally:
        conn.close()
    
    return stats
