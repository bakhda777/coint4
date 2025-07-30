#!/usr/bin/env python3
"""Пример скрипта для быстрого запуска оптимизации параметров."""

import sys
from pathlib import Path

# Добавляем src в путь для импорта
sys.path.append(str(Path(__file__).parent.parent / "src"))

from optimiser import BacktestOptimizer
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Основная функция для запуска оптимизации."""
    
    # Пути к конфигам
    base_config = "configs/main_2024.yaml"
    search_space = "configs/search_space.yaml"
    
    # Создаем оптимизатор
    optimizer = BacktestOptimizer(
        base_config_path=base_config,
        search_space_path=search_space,
        study_name="quick_optimization"
    )
    
    print("🚀 Запускаем быструю оптимизацию параметров...")
    print(f"📁 Базовый конфиг: {base_config}")
    print(f"🔍 Пространство поиска: {search_space}")
    print("⏱️  Количество trials: 20 (для быстрого тестирования)")
    print()
    
    # Запускаем оптимизацию с небольшим количеством trials для тестирования
    result = optimizer.optimize(n_trials=20)
    
    # Выводим результаты
    print("\n" + "="*60)
    print("🎯 РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
    print("="*60)
    print(f"📈 Лучший Sharpe Ratio: {result.best_value:.4f}")
    print(f"🔢 Количество trials: {result.n_trials}")
    print(f"⏰ Время оптимизации: {result.optimization_time:.1f} сек")
    print("\n🏆 Лучшие параметры:")
    
    for param, value in result.best_params.items():
        if isinstance(value, float):
            print(f"   {param}: {value:.4f}")
        else:
            print(f"   {param}: {value}")
    
    # Сохраняем лучший конфиг
    output_config = "configs/optimized_quick_test.yaml"
    optimizer.save_best_config(result, output_config)
    print(f"\n💾 Лучший конфиг сохранен: {output_config}")
    
    print("\n✅ Оптимизация завершена успешно!")
    print("\n💡 Для полной оптимизации запустите:")
    print("   python src/optimiser/run_optimization.py --n-trials 100")

if __name__ == "__main__":
    main()