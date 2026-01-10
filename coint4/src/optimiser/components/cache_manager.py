"""
Менеджер кэша для оптимизации.
Отвечает за кэширование результатов бэктестов.
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import logging
import time
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Менеджер кэша для оптимизации.
    Кэширует результаты бэктестов для ускорения оптимизации.
    """
    
    def __init__(
        self,
        cache_dir: str = "cache/optimization",
        max_cache_size_mb: int = 1000,
        ttl_hours: int = 24
    ):
        """
        Args:
            cache_dir: Директория для кэша
            max_cache_size_mb: Максимальный размер кэша в МБ
            ttl_hours: Время жизни кэша в часах
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size_mb = max_cache_size_mb
        self.ttl_hours = ttl_hours
        
        # Мемори кэш
        self._memory_cache = {}
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        
        # Thread lock для потокобезопасности
        self._lock = threading.RLock()
        
        # Очищаем старые файлы при старте
        self._cleanup_old_cache()
    
    def get(
        self,
        key: str,
        params: Dict[str, Any],
        computation_func: callable,
        use_disk: bool = True
    ) -> Any:
        """
        Получает значение из кэша или вычисляет его.
        
        Args:
            key: Ключ кэша
            params: Параметры для генерации хэша
            computation_func: Функция для вычисления значения
            use_disk: Использовать дисковый кэш
            
        Returns:
            Значение из кэша или вычисленное
        """
        # Генерируем хэш ключ
        cache_key = self._generate_cache_key(key, params)
        
        with self._lock:
            # Проверяем мемори кэш
            if cache_key in self._memory_cache:
                self._cache_stats['hits'] += 1
                return self._memory_cache[cache_key]['value']
            
            # Проверяем дисковый кэш
            if use_disk:
                disk_value = self._load_from_disk(cache_key)
                if disk_value is not None:
                    self._cache_stats['hits'] += 1
                    # Сохраняем в мемори кэш
                    self._memory_cache[cache_key] = {
                        'value': disk_value,
                        'timestamp': time.time()
                    }
                    return disk_value
            
            # Cache miss - вычисляем
            self._cache_stats['misses'] += 1
            
        # Вычисляем значение (вне lock)
        value = computation_func()
        
        with self._lock:
            # Сохраняем в кэш
            self._memory_cache[cache_key] = {
                'value': value,
                'timestamp': time.time()
            }
            
            if use_disk:
                self._save_to_disk(cache_key, value)
            
            # Очищаем мемори кэш если нужно
            self._evict_if_needed()
        
        return value
    
    def invalidate(self, key: Optional[str] = None, pattern: Optional[str] = None):
        """
        Инвалидирует кэш.
        
        Args:
            key: Конкретный ключ для инвалидации
            pattern: Паттерн для инвалидации нескольких ключей
        """
        with self._lock:
            if key:
                # Удаляем конкретный ключ
                cache_key = self._generate_cache_key(key, {})
                if cache_key in self._memory_cache:
                    del self._memory_cache[cache_key]
                self._delete_from_disk(cache_key)
                
            elif pattern:
                # Удаляем по паттерну
                keys_to_delete = [
                    k for k in self._memory_cache.keys()
                    if pattern in k
                ]
                for k in keys_to_delete:
                    del self._memory_cache[k]
                    self._delete_from_disk(k)
                    
            else:
                # Очищаем весь кэш
                self._memory_cache.clear()
                self._clear_disk_cache()
            
            logger.info(f"🗑️ Кэш инвалидирован: key={key}, pattern={pattern}")
    
    def _generate_cache_key(self, key: str, params: Dict[str, Any]) -> str:
        """
        Генерирует уникальный ключ кэша.
        
        Args:
            key: Базовый ключ
            params: Параметры
            
        Returns:
            Хэш ключ
        """
        # Сортируем параметры для стабильности
        sorted_params = json.dumps(params, sort_keys=True)
        combined = f"{key}_{sorted_params}"
        
        # Генерируем хэш
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _load_from_disk(self, cache_key: str) -> Optional[Any]:
        """
        Загружает значение с диска.
        
        Args:
            cache_key: Ключ кэша
            
        Returns:
            Значение или None
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        # Проверяем TTL
        file_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if file_age_hours > self.ttl_hours:
            cache_file.unlink()
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Ошибка чтения кэша {cache_key}: {e}")
            cache_file.unlink()
            return None
    
    def _save_to_disk(self, cache_key: str, value: Any):
        """
        Сохраняет значение на диск.
        
        Args:
            cache_key: Ключ кэша
            value: Значение
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"Ошибка сохранения кэша {cache_key}: {e}")
    
    def _delete_from_disk(self, cache_key: str):
        """
        Удаляет файл кэша с диска.
        
        Args:
            cache_key: Ключ кэша
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            cache_file.unlink()
    
    def _clear_disk_cache(self):
        """Очищает весь дисковый кэш."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
    
    def _cleanup_old_cache(self):
        """Удаляет старые файлы кэша."""
        cutoff_time = time.time() - (self.ttl_hours * 3600)
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            if cache_file.stat().st_mtime < cutoff_time:
                cache_file.unlink()
                logger.debug(f"Удален старый кэш: {cache_file.name}")
    
    def _evict_if_needed(self):
        """Очищает мемори кэш при превышении лимита."""
        # Простое LRU eviction
        max_items = 1000  # Максимальное количество элементов
        
        if len(self._memory_cache) > max_items:
            # Сортируем по timestamp и удаляем старые
            sorted_keys = sorted(
                self._memory_cache.keys(),
                key=lambda k: self._memory_cache[k]['timestamp']
            )
            
            # Удаляем 20% старых
            num_to_evict = len(self._memory_cache) - int(max_items * 0.8)
            for key in sorted_keys[:num_to_evict]:
                del self._memory_cache[key]
                self._cache_stats['evictions'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику кэша.
        
        Returns:
            Словарь со статистикой
        """
        with self._lock:
            total_requests = self._cache_stats['hits'] + self._cache_stats['misses']
            hit_rate = self._cache_stats['hits'] / max(total_requests, 1)
            
            return {
                'hits': self._cache_stats['hits'],
                'misses': self._cache_stats['misses'],
                'evictions': self._cache_stats['evictions'],
                'hit_rate': hit_rate,
                'memory_items': len(self._memory_cache),
                'disk_files': len(list(self.cache_dir.glob("*.pkl")))
            }
    
    def print_stats(self):
        """Выводит статистику кэша."""
        stats = self.get_stats()
        logger.info("📊 Статистика кэша:")
        logger.info(f"   Hits: {stats['hits']}")
        logger.info(f"   Misses: {stats['misses']}")
        logger.info(f"   Hit rate: {stats['hit_rate']:.2%}")
        logger.info(f"   Memory items: {stats['memory_items']}")
        logger.info(f"   Disk files: {stats['disk_files']}")
        logger.info(f"   Evictions: {stats['evictions']}")


class WalkForwardCacheManager(CacheManager):
    """
    Специализированный кэш менеджер для walk-forward оптимизации.
    """
    
    def get_backtest_result(
        self,
        params: Dict[str, Any],
        training_start: str,
        training_end: str,
        testing_start: str,
        testing_end: str,
        computation_func: callable
    ) -> Dict[str, Any]:
        """
        Получает результат бэктеста из кэша или вычисляет.
        
        Args:
            params: Параметры бэктеста
            training_start: Начало тренировочного периода
            training_end: Конец тренировочного периода
            testing_start: Начало тестового периода
            testing_end: Конец тестового периода
            computation_func: Функция для вычисления
            
        Returns:
            Результат бэктеста
        """
        # Создаем ключ с периодами
        key = f"wf_{training_start}_{training_end}_{testing_start}_{testing_end}"
        
        return self.get(key, params, computation_func)
    
    def invalidate_period(
        self,
        training_start: Optional[str] = None,
        training_end: Optional[str] = None
    ):
        """
        Инвалидирует кэш для конкретного периода.
        
        Args:
            training_start: Начало тренировочного периода
            training_end: Конец тренировочного периода
        """
        if training_start and training_end:
            pattern = f"wf_{training_start}_{training_end}"
            self.invalidate(pattern=pattern)
        else:
            # Очищаем весь walk-forward кэш
            self.invalidate(pattern="wf_")