#!/usr/bin/env python3
"""
Демонстрация ускорения оптимизации Optuna.

Показывает все 7 реализованных оптимизаций:
1. Оптимизированные настройки TPESampler и MedianPruner
2. Оптимизированное storage (PostgreSQL/SQLite WAL)
3. Настройка BLAS threading для параллельности
4. Быстрая предварительная фильтрация параметров
5. Кэширование данных между trials
6. Суженное пространство поиска
7. Агрессивный pruning

Использование:
    python scripts/demo_optimization_speed.py --mode ultra_fast --trials 50
    python scripts/demo_optimization_speed.py --mode fast --trials 100
    python scripts/demo_optimization_speed.py --mode full --trials 200
"""

import argparse
import time
import sys
import os
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimiser.run_optimization import run_optimization
from src.coint2.core.memory_optimization import setup_optimized_threading


def demo_optimization_modes():
    """Демонстрация различных режимов оптимизации."""
    
    print("🚀 ДЕМОНСТРАЦИЯ УСКОРЕНИЯ ОПТИМИЗАЦИИ OPTUNA")
    print("=" * 60)
    
    # Режимы оптимизации
    modes = {
        "ultra_fast": {
            "description": "Ультра-быстрый режим",
            "search_space": "configs/search_space_ultra_fast.yaml",
            "trials": 20,
            "features": [
                "✅ Фиксированные costs параметры",
                "✅ Суженные диапазоны параметров", 
                "✅ Агрессивный pruning (warmup=1, interval=1)",
                "✅ Быстрая предварительная фильтрация",
                "✅ Оптимизированное threading"
            ]
        },
        "fast": {
            "description": "Быстрый режим",
            "search_space": "configs/search_space_fast.yaml", 
            "trials": 50,
            "features": [
                "✅ Исключены filters параметры",
                "✅ Умеренно суженные диапазоны",
                "✅ Умеренный pruning (warmup=2, interval=3)",
                "✅ Кэширование данных между trials",
                "✅ Оптимизированное storage"
            ]
        },
        "full": {
            "description": "Полный режим",
            "search_space": "configs/search_space.yaml",
            "trials": 100,
            "features": [
                "✅ Полное пространство поиска",
                "✅ Стандартный pruning",
                "✅ Все оптимизации включены",
                "✅ PostgreSQL для параллельности"
            ]
        }
    }
    
    for mode_name, mode_config in modes.items():
        print(f"\n📊 {mode_config['description'].upper()}")
        print("-" * 40)
        print(f"🔍 Пространство поиска: {mode_config['search_space']}")
        print(f"🎯 Рекомендуемые trials: {mode_config['trials']}")
        print("🚀 Активные оптимизации:")
        for feature in mode_config['features']:
            print(f"   {feature}")
    
    print(f"\n💡 КЛЮЧЕВЫЕ ОПТИМИЗАЦИИ:")
    print("-" * 40)
    print("1️⃣  TPESampler: n_ei_candidates=24, constant_liar=True")
    print("2️⃣  MedianPruner: Агрессивные настройки для быстрого отсева")
    print("3️⃣  Threading: 1 BLAS thread на процесс для параллельности")
    print("4️⃣  Quick Filter: Предварительная проверка логики параметров")
    print("5️⃣  Data Cache: LRU кэш с ограничением 100 элементов")
    print("6️⃣  Storage: PostgreSQL connection pooling + SQLite WAL")
    print("7️⃣  Search Space: Фиксация менее важных параметров")


def run_speed_comparison(n_trials: int = 20):
    """Сравнение скорости разных режимов."""
    
    print(f"\n⏱️  СРАВНЕНИЕ СКОРОСТИ ({n_trials} trials)")
    print("=" * 60)
    
    modes = [
        ("ultra_fast", "configs/search_space_ultra_fast.yaml"),
        ("fast", "configs/search_space_fast.yaml")
    ]
    
    results = {}
    
    for mode_name, search_space in modes:
        print(f"\n🚀 Запуск режима: {mode_name}")
        print("-" * 30)
        
        start_time = time.time()
        
        # Настройка оптимизированного threading
        threading_result = setup_optimized_threading(n_jobs=4, verbose=True)
        print(f"🔧 Threading: {threading_result['optimization_mode']} режим")
        
        try:
            success = run_optimization(
                n_trials=n_trials,
                study_name=f"speed_test_{mode_name}",
                storage_path=f"speed_test_{mode_name}.db",
                base_config_path="configs/main_2024.yaml",
                search_space_path=search_space,
                n_jobs=4,
                seed=42
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            results[mode_name] = {
                "success": success,
                "duration": duration,
                "trials_per_minute": n_trials / (duration / 60) if duration > 0 else 0
            }
            
            print(f"✅ Завершено за {duration:.1f} сек")
            print(f"📊 Скорость: {results[mode_name]['trials_per_minute']:.1f} trials/мин")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            results[mode_name] = {"success": False, "error": str(e)}
    
    # Сводка результатов
    print(f"\n📈 СВОДКА РЕЗУЛЬТАТОВ")
    print("=" * 40)
    
    for mode_name, result in results.items():
        if result.get("success"):
            print(f"{mode_name:12}: {result['duration']:6.1f}s ({result['trials_per_minute']:5.1f} trials/мин)")
        else:
            print(f"{mode_name:12}: ОШИБКА - {result.get('error', 'Unknown')}")
    
    # Вычисляем ускорение
    if len([r for r in results.values() if r.get("success")]) >= 2:
        durations = [r["duration"] for r in results.values() if r.get("success")]
        if len(durations) >= 2:
            speedup = max(durations) / min(durations)
            print(f"\n🚀 Ускорение: {speedup:.1f}x")


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(description="Демонстрация ускорения оптимизации")
    parser.add_argument("--mode", choices=["demo", "speed", "ultra_fast", "fast", "full"],
                       default="demo", help="Режим демонстрации")
    parser.add_argument("--trials", type=int, default=20,
                       help="Количество trials для тестирования скорости")
    parser.add_argument("--n-jobs", type=int, default=4,
                       help="Количество параллельных процессов")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        demo_optimization_modes()
    elif args.mode == "speed":
        run_speed_comparison(args.trials)
    else:
        # Запуск конкретного режима
        mode_configs = {
            "ultra_fast": "configs/search_space_ultra_fast.yaml",
            "fast": "configs/search_space_fast.yaml", 
            "full": "configs/search_space.yaml"
        }
        
        search_space = mode_configs.get(args.mode, "configs/search_space_fast.yaml")
        
        print(f"🚀 Запуск оптимизации в режиме: {args.mode}")
        print(f"📊 Trials: {args.trials}")
        print(f"🔧 Параллельность: {args.n_jobs}")
        print(f"🔍 Пространство поиска: {search_space}")
        
        start_time = time.time()
        
        success = run_optimization(
            n_trials=args.trials,
            study_name=f"optimized_{args.mode}",
            storage_path=f"optimized_{args.mode}.db",
            base_config_path="configs/main_2024.yaml",
            search_space_path=search_space,
            n_jobs=args.n_jobs,
            seed=42
        )
        
        duration = time.time() - start_time
        
        if success:
            print(f"\n✅ Оптимизация завершена успешно за {duration:.1f} сек")
            print(f"📊 Скорость: {args.trials / (duration / 60):.1f} trials/мин")
        else:
            print(f"\n❌ Оптимизация завершилась с ошибкой")


if __name__ == "__main__":
    main()
