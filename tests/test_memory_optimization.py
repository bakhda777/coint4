#!/usr/bin/env python3
"""
Тестовый скрипт для проверки memory-mapped оптимизации walk-forward анализа.
"""

import sys
import os
from pathlib import Path
import time
import psutil
import pandas as pd

# Добавляем путь к исходному коду
sys.path.insert(0, str(Path(__file__).parent / "src"))

from coint2.utils.config import load_config
from coint2.pipeline.walk_forward_orchestrator import run_walk_forward
from coint2.utils.logging_utils import get_logger

def monitor_memory():
    """Мониторинг использования памяти."""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        'percent': process.memory_percent()
    }

def test_memory_optimization():
    """Тестирование memory-mapped оптимизации (быстрая версия)."""
    logger = get_logger("test_memory_opt")
    
    # Загружаем конфигурацию
    config_path = Path("configs/main_2024.yaml")
    if not config_path.exists():
        logger.error(f"Конфигурационный файл не найден: {config_path}")
        return
    
    cfg = load_config(config_path)
    
    # Проверяем наличие данных
    if not cfg.data_dir.exists():
        logger.error(f"Директория с данными не найдена: {cfg.data_dir}")
        return
    
    # Оптимизируем конфигурацию для быстрого тестирования
    original_end_date = cfg.walk_forward.end_date
    original_training_period = cfg.walk_forward.training_period_days
    original_testing_period = cfg.walk_forward.testing_period_days
    
    # Уменьшаем период тестирования до 30 дней
    from datetime import datetime, timedelta
    start_date = datetime.strptime(cfg.walk_forward.start_date, "%Y-%m-%d")
    cfg.walk_forward.end_date = (start_date + timedelta(days=30)).strftime("%Y-%m-%d")
    
    # Уменьшаем размеры окон для ускорения
    cfg.walk_forward.training_period_days = min(cfg.walk_forward.training_period_days, 14)
    cfg.walk_forward.testing_period_days = min(cfg.walk_forward.testing_period_days, 7)
    
    logger.info("🧪 Тестирование memory-mapped оптимизации (быстрая версия)")
    logger.info(f"📁 Данные: {cfg.data_dir}")
    logger.info(f"📅 Период: {cfg.walk_forward.start_date} → {cfg.walk_forward.end_date} (сокращен для тестирования)")
    logger.info(f"🪟 Тренировка: {cfg.walk_forward.training_period_days} дней, тестирование: {cfg.walk_forward.testing_period_days} дней")
    
    # Тест 1: Традиционный метод (только проверка инициализации)
    logger.info("\n🔄 Тест 1: Традиционный метод (быстрая проверка)")
    memory_before = monitor_memory()
    logger.info(f"💾 Память до запуска: {memory_before['rss_mb']:.1f} MB")
    
    start_time = time.time()
    try:
        # Запускаем только первую итерацию для проверки
        results_traditional = run_walk_forward(cfg, use_memory_map=False, max_iterations=1)
        traditional_time = time.time() - start_time
        memory_after = monitor_memory()
        
        logger.info(f"✅ Традиционный метод завершен за {traditional_time:.1f}с")
        logger.info(f"💾 Память после: {memory_after['rss_mb']:.1f} MB (+{memory_after['rss_mb'] - memory_before['rss_mb']:.1f} MB)")
        logger.info(f"📊 Результат: P&L = ${results_traditional.get('total_pnl', 0):+,.2f}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка в традиционном методе: {e}")
        results_traditional = None
        traditional_time = None
    
    # Небольшая пауза для очистки памяти
    time.sleep(1)
    
    # Тест 2: Memory-mapped метод (быстрая проверка)
    logger.info("\n🧠 Тест 2: Memory-mapped метод (быстрая проверка)")
    memory_before = monitor_memory()
    logger.info(f"💾 Память до запуска: {memory_before['rss_mb']:.1f} MB")
    
    start_time = time.time()
    try:
        # Запускаем только первую итерацию для проверки
        results_mmap = run_walk_forward(cfg, use_memory_map=True, max_iterations=1)
        mmap_time = time.time() - start_time
        memory_after = monitor_memory()
        
        logger.info(f"✅ Memory-mapped метод завершен за {mmap_time:.1f}с")
        logger.info(f"💾 Память после: {memory_after['rss_mb']:.1f} MB (+{memory_after['rss_mb'] - memory_before['rss_mb']:.1f} MB)")
        logger.info(f"📊 Результат: P&L = ${results_mmap.get('total_pnl', 0):+,.2f}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка в memory-mapped методе: {e}")
        results_mmap = None
        mmap_time = None
    
    # Восстанавливаем оригинальные настройки
    cfg.walk_forward.end_date = original_end_date
    cfg.walk_forward.training_period_days = original_training_period
    cfg.walk_forward.testing_period_days = original_testing_period
    
    # Сравнение результатов
    logger.info("\n📊 Сравнение результатов:")
    
    if traditional_time and mmap_time:
        speedup = traditional_time / mmap_time
        logger.info(f"⏱️ Ускорение: {speedup:.2f}x ({traditional_time:.1f}с → {mmap_time:.1f}с)")
        
        if speedup > 1.1:
            logger.info("🚀 Memory-mapped оптимизация показала улучшение производительности!")
        elif speedup > 0.9:
            logger.info("⚖️ Производительность примерно одинаковая")
        else:
            logger.info("ℹ️ Memory-mapped метод может быть медленнее на малых данных")
    
    if results_traditional and results_mmap:
        # Сравниваем ключевые метрики
        pnl_diff = abs(results_traditional.get('total_pnl', 0) - results_mmap.get('total_pnl', 0))
        if pnl_diff < 0.01:  # Разница менее 1 цента
            logger.info("✅ Результаты идентичны - оптимизация работает корректно")
        else:
            logger.warning(f"⚠️ Результаты отличаются на ${pnl_diff:.2f}")
            logger.info("ℹ️ Небольшие различия могут быть нормальными при сокращенном тестировании")
    
    logger.info("\n🎉 Быстрое тестирование завершено!")
    logger.info("ℹ️ Для полного тестирования запустите с полными параметрами конфигурации")

if __name__ == "__main__":
    test_memory_optimization()