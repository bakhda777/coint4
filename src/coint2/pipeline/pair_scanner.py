"""Заглушка для модуля pair_scanner.

Этот модуль пока не реализован, но нужен для запуска тестов.
"""

from typing import List, Tuple, Optional
import pandas as pd
from dask import delayed


def find_cointegrated_pairs(handler, start_date, end_date, config) -> List[Tuple]:
    """Заглушка для поиска коинтегрированных пар.
    
    Args:
        handler: DataHandler instance
        start_date: Start date for analysis
        end_date: End date for analysis
        config: Configuration object
        
    Returns:
        List of tuples representing cointegrated pairs
    """
    # Простая заглушка - возвращаем пустой список
    return []


@delayed
def _test_pair_for_tradability(
    handler_arg,
    s1: str,
    s2: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    min_hl: float,
    max_hl: float,
    min_cross: int,
) -> Optional[Tuple[str, str]]:
    """Заглушка для тестирования торгуемости пары.
    
    Args:
        handler_arg: DataHandler instance
        s1: First symbol
        s2: Second symbol
        start_date: Start date
        end_date: End date
        min_hl: Minimum half-life
        max_hl: Maximum half-life
        min_cross: Minimum crossings
        
    Returns:
        Tuple of symbols if tradable, None otherwise
    """
    # Простая заглушка - возвращаем пару как есть
    return (s1, s2)