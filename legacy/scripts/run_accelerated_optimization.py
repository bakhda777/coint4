#!/usr/bin/env python3
"""
Скрипт для запуска ускоренной оптимизации с PostgreSQL и кэшированием.
"""

import sys
from pathlib import Path
import argparse
import subprocess
import time

# Добавляем корневую директорию проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.optimiser.run_optimization import run_optimization


def setup_postgresql():
    """Настройка PostgreSQL для оптимизации."""
    print("🚀 Настройка PostgreSQL...")
    
    try:
        # Запускаем скрипт настройки PostgreSQL
        setup_script = project_root / "scripts" / "setup_postgresql.sh"
        result = subprocess.run([str(setup_script)], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ PostgreSQL успешно настроен")
            return True
        else:
            print(f"❌ Ошибка настройки PostgreSQL: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка при настройке PostgreSQL: {e}")
        return False


def run_accelerated_optimization(
    n_trials: int = 200,
    study_name: str = "accelerated_pairs_strategy",
    use_postgresql: bool = True,
    n_jobs: int = -1,
    base_config_path: str = "configs/main_2024.yaml",
    search_space_path: str = "configs/search_space_fast.yaml",
    optimization_mode: str = "fast"
):
    """
    Запуск ускоренной оптимизации с кэшированием и PostgreSQL.

    Args:
        n_trials: Количество trials для оптимизации
        study_name: Имя исследования
        use_postgresql: Использовать PostgreSQL для параллелизации
        n_jobs: Количество параллельных процессов (-1 = все ядра)
        base_config_path: Путь к базовой конфигурации
        search_space_path: Путь к пространству поиска
        optimization_mode: Режим оптимизации (fast, ultra_fast, full)
    """

    # ОПТИМИЗАЦИЯ: Выбор пространства поиска в зависимости от режима
    if optimization_mode == "ultra_fast":
        search_space_path = "configs/search_space_ultra_fast.yaml"
        print(f"🚀 Ультра-быстрый режим: используем {search_space_path}")
    elif optimization_mode == "fast":
        search_space_path = "configs/search_space_fast.yaml"
        print(f"⚡ Быстрый режим: используем {search_space_path}")

    # ОПТИМИЗАЦИЯ: Ограничение количества процессов для стабильности
    import psutil
    max_cores = psutil.cpu_count()
    if n_jobs == -1:
        n_jobs = min(max_cores, 8)  # Не больше 8 процессов для стабильности
    elif n_jobs > max_cores:
        print(f"⚠️  Ограничиваем n_jobs с {n_jobs} до {max_cores} (количество ядер)")
        n_jobs = max_cores
    
    # Определяем storage_path
    if use_postgresql:
        # Настраиваем PostgreSQL
        if not setup_postgresql():
            print("❌ Не удалось настроить PostgreSQL, используем SQLite")
            storage_path = f"outputs/studies/{study_name}.db"
            n_jobs = 1  # SQLite не поддерживает параллельность
        else:
            storage_path = "postgresql://localhost:5432/optuna_studies"
    else:
        storage_path = f"outputs/studies/{study_name}.db"
        n_jobs = 1  # SQLite не поддерживает параллельность
    
    print(f"📊 Запуск ускоренной оптимизации:")
    print(f"   Trials: {n_trials}")
    print(f"   Study: {study_name}")
    print(f"   Storage: {storage_path}")
    print(f"   Параллельность: {n_jobs} процессов")
    print(f"   Конфигурация: {base_config_path}")
    print(f"   Пространство поиска: {search_space_path}")
    
    # Запускаем оптимизацию
    start_time = time.time()
    
    success = run_optimization(
        n_trials=n_trials,
        study_name=study_name,
        storage_path=storage_path,
        base_config_path=base_config_path,
        search_space_path=search_space_path,
        n_jobs=n_jobs,
        seed=42
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    if success:
        print(f"✅ Оптимизация завершена успешно за {duration:.1f} секунд")
        print(f"📈 Среднее время на trial: {duration/n_trials:.2f} секунд")
        
        # Выводим информацию о ускорении
        if use_postgresql and n_jobs > 1:
            estimated_sequential_time = duration * n_jobs
            speedup = estimated_sequential_time / duration
            print(f"🚀 Ускорение от параллелизации: ~{speedup:.1f}x")
            
    else:
        print(f"❌ Оптимизация завершилась с ошибкой")
        
    return success


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(description="Запуск ускоренной оптимизации")
    
    parser.add_argument("--n-trials", type=int, default=200,
                       help="Количество trials (по умолчанию: 200)")
    parser.add_argument("--study-name", default="accelerated_pairs_strategy",
                       help="Имя исследования")
    parser.add_argument("--use-postgresql", action="store_true", default=True,
                       help="Использовать PostgreSQL для параллелизации")
    parser.add_argument("--no-postgresql", dest="use_postgresql", action="store_false",
                       help="Не использовать PostgreSQL (SQLite)")
    parser.add_argument("--n-jobs", type=int, default=-1,
                       help="Количество параллельных процессов (-1 = все ядра)")
    parser.add_argument("--base-config", default="configs/main_2024.yaml",
                       help="Путь к базовой конфигурации")
    parser.add_argument("--search-space", default="configs/search_space_fast.yaml",
                       help="Путь к пространству поиска")
    parser.add_argument("--optimization-mode", default="fast",
                       choices=["fast", "ultra_fast", "full"],
                       help="Режим оптимизации (fast, ultra_fast, full)")

    args = parser.parse_args()
    
    print("🎯 Запуск ускоренной оптимизации с кэшированием")
    print("=" * 60)
    
    success = run_accelerated_optimization(
        n_trials=args.n_trials,
        study_name=args.study_name,
        use_postgresql=args.use_postgresql,
        n_jobs=args.n_jobs,
        base_config_path=args.base_config,
        search_space_path=args.search_space,
        optimization_mode=args.optimization_mode
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
