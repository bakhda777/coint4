"""
Файловый кэш для замены multiprocessing.Manager.
Использует локальные файлы для межпроцессной синхронизации.
"""

import json
import pickle
import hashlib
import fcntl
import time
from pathlib import Path
from typing import Any, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class FileCache:
    """
    Потокобезопасный файловый кэш для межпроцессного обмена данными.
    Заменяет multiprocessing.Manager.dict() более надежным решением.
    """
    
    def __init__(self, cache_dir: str = ".cache/optuna"):
        """
        Args:
            cache_dir: Директория для хранения кэша
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock_dir = self.cache_dir / "locks"
        self._lock_dir.mkdir(exist_ok=True)
        
    def _get_cache_path(self, key: str) -> Path:
        """Генерирует путь к файлу кэша для ключа."""
        # Используем hash для безопасных имен файлов
        key_hash = hashlib.md5(str(key).encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"
    
    def _get_lock_path(self, key: str) -> Path:
        """Генерирует путь к файлу блокировки."""
        key_hash = hashlib.md5(str(key).encode()).hexdigest()
        return self._lock_dir / f"{key_hash}.lock"
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Получает значение из кэша.
        
        Args:
            key: Ключ
            default: Значение по умолчанию
            
        Returns:
            Закэшированное значение или default
        """
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return default
        
        lock_path = self._get_lock_path(key)
        
        # Используем файловую блокировку для безопасного чтения
        with open(lock_path, 'a') as lock_file:
            # Ждем пока файл разблокируется
            max_attempts = 10
            for attempt in range(max_attempts):
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                    break
                except IOError:
                    if attempt == max_attempts - 1:
                        logger.warning(f"Не удалось получить блокировку для {key}")
                        return default
                    time.sleep(0.1)
            
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Ошибка чтения кэша для {key}: {e}")
                return default
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    
    def set(self, key: str, value: Any) -> None:
        """
        Сохраняет значение в кэш.
        
        Args:
            key: Ключ
            value: Значение для сохранения
        """
        cache_path = self._get_cache_path(key)
        lock_path = self._get_lock_path(key)
        
        # Используем эксклюзивную блокировку для записи
        with open(lock_path, 'a') as lock_file:
            # Ждем пока файл разблокируется
            max_attempts = 10
            for attempt in range(max_attempts):
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except IOError:
                    if attempt == max_attempts - 1:
                        logger.warning(f"Не удалось получить блокировку для записи {key}")
                        return
                    time.sleep(0.1)
            
            try:
                # Атомарная запись через временный файл
                temp_path = cache_path.with_suffix('.tmp')
                with open(temp_path, 'wb') as f:
                    pickle.dump(value, f)
                temp_path.replace(cache_path)
            except Exception as e:
                logger.error(f"Ошибка записи кэша для {key}: {e}")
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    
    def __contains__(self, key: str) -> bool:
        """Проверяет наличие ключа в кэше."""
        return self._get_cache_path(key).exists()
    
    def __getitem__(self, key: str) -> Any:
        """Получает значение по ключу."""
        value = self.get(key)
        if value is None:
            raise KeyError(f"Key {key} not found in cache")
        return value
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Устанавливает значение по ключу."""
        self.set(key, value)
    
    def clear(self) -> None:
        """Очищает весь кэш."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Не удалось удалить {cache_file}: {e}")
        
        for lock_file in self._lock_dir.glob("*.lock"):
            try:
                lock_file.unlink()
            except Exception as e:
                logger.warning(f"Не удалось удалить {lock_file}: {e}")
    
    def size(self) -> int:
        """Возвращает количество элементов в кэше."""
        return len(list(self.cache_dir.glob("*.pkl")))
    
    def cleanup_old(self, max_age_hours: float = 24) -> int:
        """
        Удаляет старые записи кэша.
        
        Args:
            max_age_hours: Максимальный возраст файлов в часах
            
        Returns:
            Количество удаленных файлов
        """
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        removed = 0
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                age = current_time - cache_file.stat().st_mtime
                if age > max_age_seconds:
                    cache_file.unlink()
                    removed += 1
            except Exception as e:
                logger.warning(f"Ошибка при очистке {cache_file}: {e}")
        
        if removed > 0:
            logger.info(f"🗑️ Удалено {removed} старых файлов кэша")
        
        return removed


class DummyLock:
    """Заглушка для блокировки в однопоточном режиме."""
    
    def acquire(self, blocking=True):
        return True
    
    def release(self):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass