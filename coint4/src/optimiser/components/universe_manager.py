"""
Менеджер universe пар для фиксации набора торгуемых инструментов.
Предотвращает изменение universe между trials в Optuna.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from pathlib import Path
import logging
import json
import hashlib

logger = logging.getLogger(__name__)


class UniverseManager:
    """
    Менеджер фиксированного universe пар.
    
    Гарантирует, что все trials в оптимизации используют
    одинаковый набор пар для честного сравнения.
    """
    
    def __init__(self, cache_dir: str = "outputs/universe_cache"):
        """
        Args:
            cache_dir: Директория для кэширования universe
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Текущий зафиксированный universe
        self._fixed_universe: Optional[List[Tuple[str, str]]] = None
        self._universe_hash: Optional[str] = None
        
    def fix_universe(
        self, 
        pairs: List[Tuple[str, str]],
        study_name: str,
        force_update: bool = False
    ) -> List[Tuple[str, str]]:
        """
        Фиксирует universe пар для study.
        
        Args:
            pairs: Список пар для фиксации
            study_name: Имя Optuna study
            force_update: Принудительное обновление universe
            
        Returns:
            Зафиксированный список пар
        """
        # Генерируем hash для universe
        universe_str = json.dumps(sorted(pairs), sort_keys=True)
        universe_hash = hashlib.md5(universe_str.encode()).hexdigest()[:8]
        
        # Путь к файлу кэша
        cache_file = self.cache_dir / f"{study_name}_universe_{universe_hash}.json"
        
        # Проверяем существующий кэш
        if cache_file.exists() and not force_update:
            logger.info(f"📦 Загружаем зафиксированный universe из кэша: {cache_file.name}")
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                self._fixed_universe = [tuple(pair) for pair in cached_data['pairs']]
                self._universe_hash = cached_data['hash']
                logger.info(f"✅ Загружено {len(self._fixed_universe)} пар из кэша")
                return self._fixed_universe
        
        # Фиксируем новый universe
        logger.info(f"🔒 Фиксируем новый universe: {len(pairs)} пар")
        
        self._fixed_universe = pairs
        self._universe_hash = universe_hash
        
        # Сохраняем в кэш
        cache_data = {
            'study_name': study_name,
            'hash': universe_hash,
            'pairs': pairs,
            'n_pairs': len(pairs),
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        logger.info(f"💾 Universe сохранен в: {cache_file.name}")
        
        # Выводим статистику
        self._log_universe_stats(pairs)
        
        return self._fixed_universe
    
    def get_fixed_universe(self) -> Optional[List[Tuple[str, str]]]:
        """
        Возвращает текущий зафиксированный universe.
        
        Returns:
            Список пар или None если universe не зафиксирован
        """
        if self._fixed_universe is None:
            logger.warning("⚠️ Universe не зафиксирован!")
        return self._fixed_universe
    
    def validate_pairs(
        self, 
        pairs: List[Tuple[str, str]],
        raise_on_mismatch: bool = True
    ) -> bool:
        """
        Проверяет, что пары соответствуют зафиксированному universe.
        
        Args:
            pairs: Пары для проверки
            raise_on_mismatch: Бросать исключение при несовпадении
            
        Returns:
            True если пары валидны
            
        Raises:
            ValueError: При несовпадении если raise_on_mismatch=True
        """
        if self._fixed_universe is None:
            logger.warning("Universe не зафиксирован, пропускаем валидацию")
            return True
        
        # Преобразуем в множества для сравнения
        fixed_set = set(self._fixed_universe)
        pairs_set = set(pairs)
        
        if fixed_set != pairs_set:
            missing = fixed_set - pairs_set
            extra = pairs_set - fixed_set
            
            msg = f"Universe mismatch! Missing: {len(missing)}, Extra: {len(extra)}"
            
            if raise_on_mismatch:
                raise ValueError(msg)
            else:
                logger.warning(f"⚠️ {msg}")
                return False
        
        return True
    
    def load_universe_for_study(self, study_name: str) -> Optional[List[Tuple[str, str]]]:
        """
        Загружает ранее зафиксированный universe для study.
        
        Args:
            study_name: Имя study
            
        Returns:
            Список пар или None если не найден
        """
        # Ищем файлы кэша для этого study
        pattern = f"{study_name}_universe_*.json"
        cache_files = list(self.cache_dir.glob(pattern))
        
        if not cache_files:
            logger.info(f"📭 Кэш universe не найден для study: {study_name}")
            return None
        
        # Берем последний по времени
        latest_file = max(cache_files, key=lambda p: p.stat().st_mtime)
        
        logger.info(f"📦 Загружаем universe из: {latest_file.name}")
        
        with open(latest_file, 'r') as f:
            cached_data = json.load(f)
            self._fixed_universe = [tuple(pair) for pair in cached_data['pairs']]
            self._universe_hash = cached_data['hash']
            
        logger.info(f"✅ Загружено {len(self._fixed_universe)} пар")
        return self._fixed_universe
    
    def _log_universe_stats(self, pairs: List[Tuple[str, str]]) -> None:
        """Выводит статистику по universe."""
        # Собираем уникальные символы
        all_symbols: Set[str] = set()
        for s1, s2 in pairs:
            all_symbols.add(s1)
            all_symbols.add(s2)
        
        logger.info(f"📊 Статистика universe:")
        logger.info(f"   Всего пар: {len(pairs)}")
        logger.info(f"   Уникальных символов: {len(all_symbols)}")
        
        # Топ символов по частоте
        symbol_counts: Dict[str, int] = {}
        for s1, s2 in pairs:
            symbol_counts[s1] = symbol_counts.get(s1, 0) + 1
            symbol_counts[s2] = symbol_counts.get(s2, 0) + 1
        
        top_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"   Топ-5 символов:")
        for symbol, count in top_symbols:
            logger.info(f"     {symbol}: {count} пар")
    
    def clear_cache(self, study_name: Optional[str] = None) -> None:
        """
        Очищает кэш universe.
        
        Args:
            study_name: Если указано, очищает только для конкретного study
        """
        if study_name:
            pattern = f"{study_name}_universe_*.json"
            files = list(self.cache_dir.glob(pattern))
            for f in files:
                f.unlink()
            logger.info(f"🗑️ Удалено {len(files)} файлов кэша для {study_name}")
        else:
            for f in self.cache_dir.glob("*.json"):
                f.unlink()
            logger.info("🗑️ Весь кэш universe очищен")