"""
Кроссплатформенный файловый кэш с использованием filelock.
Работает на Windows, Linux и macOS.
"""

import json
import pickle
import hashlib
from pathlib import Path
from typing import Any, Optional
from filelock import FileLock, Timeout
import logging

logger = logging.getLogger(__name__)


class CrossPlatformFileCache:
    """
    Файловый кэш с блокировками для межпроцессной синхронизации.
    Использует filelock для кроссплатформенной работы.
    """
    
    def __init__(self, cache_dir: str = ".cache", timeout: float = 10.0):
        """
        Args:
            cache_dir: Директория для кэша
            timeout: Таймаут на получение блокировки в секундах
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._locks = {}  # Кэш блокировок для повторного использования
        
    def _get_cache_path(self, key: str) -> Path:
        """Получает путь к файлу кэша для ключа."""
        # Используем хэш для безопасных имен файлов
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"
    
    def _get_lock_path(self, key: str) -> Path:
        """Получает путь к файлу блокировки для ключа."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.lock"
    
    def _get_lock(self, key: str) -> FileLock:
        """Получает или создает блокировку для ключа."""
        if key not in self._locks:
            lock_path = self._get_lock_path(key)
            self._locks[key] = FileLock(str(lock_path), timeout=self.timeout)
        return self._locks[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Получает значение из кэша.
        
        Args:
            key: Ключ
            default: Значение по умолчанию если ключ не найден
            
        Returns:
            Значение из кэша или default
        """
        cache_path = self._get_cache_path(key)
        lock = self._get_lock(key)
        
        try:
            with lock.acquire(timeout=self.timeout):
                if cache_path.exists():
                    try:
                        with open(cache_path, 'rb') as f:
                            return pickle.load(f)
                    except Exception as e:
                        logger.warning(f"Ошибка чтения кэша для {key}: {e}")
                        return default
                return default
        except Timeout:
            logger.warning(f"Таймаут при получении блокировки для {key}")
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """
        Сохраняет значение в кэш.
        
        Args:
            key: Ключ
            value: Значение для сохранения
            
        Returns:
            True если успешно сохранено
        """
        cache_path = self._get_cache_path(key)
        lock = self._get_lock(key)
        
        try:
            with lock.acquire(timeout=self.timeout):
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(value, f)
                    return True
                except Exception as e:
                    logger.error(f"Ошибка записи кэша для {key}: {e}")
                    return False
        except Timeout:
            logger.warning(f"Таймаут при получении блокировки для {key}")
            return False
    
    def __getitem__(self, key: str) -> Any:
        """Поддержка dict-like интерфейса."""
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Поддержка dict-like интерфейса."""
        if not self.set(key, value):
            raise RuntimeError(f"Failed to set cache for {key}")
    
    def __contains__(self, key: str) -> bool:
        """Проверяет наличие ключа в кэше."""
        cache_path = self._get_cache_path(key)
        return cache_path.exists()
    
    def clear(self) -> None:
        """Очищает весь кэш."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Не удалось удалить {cache_file}: {e}")
        
        for lock_file in self.cache_dir.glob("*.lock"):
            try:
                lock_file.unlink()
            except Exception as e:
                logger.warning(f"Не удалось удалить {lock_file}: {e}")
    
    def delete(self, key: str) -> bool:
        """
        Удаляет ключ из кэша.
        
        Args:
            key: Ключ для удаления
            
        Returns:
            True если успешно удален
        """
        cache_path = self._get_cache_path(key)
        lock = self._get_lock(key)
        
        try:
            with lock.acquire(timeout=self.timeout):
                if cache_path.exists():
                    cache_path.unlink()
                    return True
                return False
        except Timeout:
            logger.warning(f"Таймаут при получении блокировки для удаления {key}")
            return False
    
    def keys(self) -> list:
        """Возвращает список всех ключей в кэше."""
        keys = []
        for cache_file in self.cache_dir.glob("*.pkl"):
            # Восстанавливаем оригинальный ключ из имени файла
            # Это не всегда возможно из-за хэширования, 
            # поэтому возвращаем хэши
            keys.append(cache_file.stem)
        return keys


class DummyLock:
    """Заглушка для блокировки в однопоточном режиме."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def acquire(self, *args, **kwargs):
        return self
    
    def release(self):
        pass


# Для обратной совместимости
FileCache = CrossPlatformFileCache