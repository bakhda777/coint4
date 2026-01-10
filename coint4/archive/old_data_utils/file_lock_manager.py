#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Модуль для управления блокировками файлов при параллельной записи.
Предотвращает повреждение parquet-файлов при одновременной записи из разных потоков.
"""

import os
import time
import logging
from threading import Lock, RLock
from pathlib import Path
from typing import Dict, Optional

# Настройка логгера
logger = logging.getLogger()

class FileLockManager:
    """Менеджер блокировок файлов для безопасного доступа из разных потоков."""
    
    def __init__(self):
        """Инициализация менеджера блокировок."""
        # Используем RLock вместо Lock для рекурсивных блокировок
        self.global_lock = RLock()
        # Словарь блокировок для файлов
        self.file_locks: Dict[str, Lock] = {}
        
    def get_lock(self, file_path: str) -> Lock:
        """
        Получает блокировку для указанного файла.
        Если блокировки для файла нет, создает новую.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Lock: Объект блокировки для файла
        """
        # Нормализуем путь для предотвращения дубликатов блокировок
        norm_path = str(Path(file_path).resolve())
        
        with self.global_lock:
            if norm_path not in self.file_locks:
                self.file_locks[norm_path] = Lock()
            return self.file_locks[norm_path]
    
    def acquire(self, file_path: str, timeout: float = None) -> bool:
        """
        Блокирует файл для эксклюзивного доступа.
        
        Args:
            file_path: Путь к файлу
            timeout: Таймаут в секундах (None - ждать бесконечно)
            
        Returns:
            bool: True если блокировка успешно получена, False если истек таймаут
        """
        lock = self.get_lock(file_path)
        start_time = time.time()
        
        if timeout is not None:
            # Пытаемся получить блокировку с таймаутом
            got_lock = lock.acquire(timeout=timeout)
            if not got_lock:
                logger.warning(f"⏱️ Таймаут блокировки для {file_path} после {timeout} секунд")
            return got_lock
        else:
            # Блокируем без таймаута (блокирующий вызов)
            lock.acquire()
            return True
    
    def release(self, file_path: str):
        """
        Освобождает блокировку файла.
        
        Args:
            file_path: Путь к файлу
        """
        norm_path = str(Path(file_path).resolve())
        
        with self.global_lock:
            if norm_path in self.file_locks:
                try:
                    self.file_locks[norm_path].release()
                except RuntimeError:
                    # Игнорируем ошибку, если блокировка не была захвачена
                    logger.warning(f"⚠️ Попытка освободить незахваченную блокировку для {file_path}")
    
    def cleanup_unused_locks(self):
        """Удаляет неиспользуемые блокировки для экономии памяти."""
        with self.global_lock:
            # Очистка не реализована, так как сложно определить, какие блокировки действительно не используются
            pass

# Глобальный экземпляр менеджера блокировок
file_lock_manager = FileLockManager()

class FileLock:
    """Контекстный менеджер для блокировки файла."""
    
    def __init__(self, file_path: str, timeout: Optional[float] = None):
        """
        Инициализация блокировки для файла.
        
        Args:
            file_path: Путь к блокируемому файлу
            timeout: Таймаут в секундах (None - ждать бесконечно)
        """
        self.file_path = file_path
        self.timeout = timeout
        self.locked = False
    
    def __enter__(self):
        """Вход в контекстный менеджер - захват блокировки."""
        self.locked = file_lock_manager.acquire(self.file_path, self.timeout)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Выход из контекстного менеджера - освобождение блокировки."""
        if self.locked:
            file_lock_manager.release(self.file_path)
            self.locked = False
