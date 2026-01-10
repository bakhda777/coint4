#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Tuple
from datetime import datetime, timedelta
import logging

# Настройка логирования
logger = logging.getLogger(__name__)

def group_timestamps_into_chunks(
    timestamps: List[int], 
    max_gap: int = 15 * 60 * 1000,  # 15 минут в миллисекундах
    max_chunk_size: int = 7 * 24 * 60 * 60 * 1000,  # Максимум 7 дней в миллисекундах
) -> List[Tuple[datetime, datetime]]:
    """
    Группирует timestamp'ы в смежные интервалы для оптимизации запросов.
    
    Args:
        timestamps: Список timestamp'ов для группировки (в миллисекундах)
        max_gap: Максимальный разрыв между timestamp'ами для включения в один интервал (в миллисекундах)
        max_chunk_size: Максимальная продолжительность интервала (в миллисекундах)
        
    Returns:
        List[Tuple[datetime, datetime]]: Список интервалов в формате (начало, конец)
    """
    if not timestamps:
        return []
    
    # Сортируем timestamps
    timestamps.sort()
    
    chunks = []
    chunk_start = timestamps[0]
    last_ts = chunk_start
    
    for i in range(1, len(timestamps)):
        current_ts = timestamps[i]
        
        # Если разрыв между текущим и предыдущим timestamp'ом слишком велик
        # или если текущий интервал достиг максимальной продолжительности,
        # завершаем текущий интервал и начинаем новый
        if (current_ts - last_ts > max_gap) or (current_ts - chunk_start > max_chunk_size):
            # Добавляем запас для API запроса
            start_datetime = datetime.fromtimestamp(chunk_start / 1000) - timedelta(minutes=15)
            end_datetime = datetime.fromtimestamp(last_ts / 1000) + timedelta(minutes=15)
            chunks.append((start_datetime, end_datetime))
            
            chunk_start = current_ts
        
        last_ts = current_ts
    
    # Добавляем последний интервал
    start_datetime = datetime.fromtimestamp(chunk_start / 1000) - timedelta(minutes=15)
    end_datetime = datetime.fromtimestamp(last_ts / 1000) + timedelta(minutes=15)
    chunks.append((start_datetime, end_datetime))
    
    # Выводим информацию о созданных интервалах
    logger.info(f"Создано {len(chunks)} интервалов из {len(timestamps)} timestamp'ов")
    
    return chunks
