#!/usr/bin/env python3
"""
Унифицированный скрипт для запуска оптимизации гиперпараметров стратегии парного трейдинга.

Объединяет функциональность всех скриптов оптимизации:
- fast_optimize.py (быстрая оптимизация на предотобранных парах)
- bp_optimize.py (best practice оптимизация)
- simple_fast_optimize.py (упрощенная оптимизация)
- run_optimization.py (основная оптимизация)

Поддерживает:
- Различные режимы оптимизации (fast/full/simple/bp)
- Optuna оптимизацию с различными sampler'ами
- Параллельное выполнение
- Сохранение результатов в SQLite
- Валидацию параметров
- Обработку ошибок
"""

import argparse
import logging
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import optuna
import numpy as np
import pandas as pd
import random
import yaml

# Добавляем корневую директорию проекта в PYTHONPATH
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.optimiser.objective import WalkForwardObjective
from src.coint2.utils.config import load_config

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seeds(seed: int = 42):
    """Установка seeds для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)


def setup_logging(study_name: str) -> logging.Logger:
    """Настройка логирования для исследования."""
    log_file = f"logs/{study_name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    Path("logs").mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger = logging.getLogger(study_name)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    return logger


def create_objective(
    mode: str,
    base_config_path: str,
    search_space_path: Optional[str] = None
) -> WalkForwardObjective:
    """
    Создает унифицированную целевую функцию в зависимости от режима.

    Args:
        mode: Режим оптимизации ('fast', 'full', 'simple', 'bp')
        base_config_path: Путь к базовой конфигурации
        search_space_path: Путь к пространству поиска

    Returns:
        WalkForwardObjective: Настроенная целевая функция
    """
    if mode == "fast":
        return WalkForwardObjective(
            base_config_path=base_config_path,
            search_space_path=search_space_path,
            fast_mode=True,
            simple_mode=False
        )
    elif mode == "simple":
        return WalkForwardObjective(
            base_config_path=base_config_path,
            search_space_path=search_space_path,
            fast_mode=True,
            simple_mode=True
        )
    elif mode == "bp":
        return WalkForwardObjective(
            base_config_path=base_config_path,
            search_space_path=search_space_path,
            fast_mode=True,  # BP режим тоже использует предотобранные пары
            simple_mode=False
        )
    elif mode == "full":
        return WalkForwardObjective(
            base_config_path=base_config_path,
            search_space_path=search_space_path,
            fast_mode=False,  # Полный отбор пар
            simple_mode=False
        )
    else:
        raise ValueError(f"Неизвестный режим: {mode}")


def create_study(study_name: str, storage_path: str, mode: str, n_trials: int) -> optuna.Study:
    """Создает Optuna study с оптимальными параметрами для режима."""
    
    # Настройки sampler в зависимости от режима
    if mode == "fast":
        n_startup_trials = min(20, max(5, n_trials // 10))
        sampler = optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=n_startup_trials,
            multivariate=True
        )
        pruner = optuna.pruners.MedianPruner(
            n_warmup_steps=10,
            interval_steps=5
        )
    elif mode == "bp":
        sampler = optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=50,
            multivariate=True
        )
        pruner = optuna.pruners.MedianPruner(
            n_warmup_steps=30,
            interval_steps=5
        )
    else:  # simple, full
        sampler = optuna.samplers.TPESampler(seed=42)
        pruner = optuna.pruners.MedianPruner()
    
    return optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner
    )


def check_prerequisites(mode: str) -> bool:
    """Проверяет наличие необходимых файлов для режима."""
    required_files = {
        "fast": ["outputs/preselected_pairs.csv"],
        "simple": [
            "outputs/preselected_pairs.csv",
            "outputs/full_step_data.csv",
            "outputs/training_normalization_params.csv"
        ],
        "bp": [],
        "full": []
    }
    
    files_to_check = required_files.get(mode, [])
    
    for file_path in files_to_check:
        if not Path(file_path).exists():
            logger.error(f"❌ Файл {file_path} не найден!")
            if mode == "fast":
                logger.info("🔧 Пары будут отобраны динамически для каждого walk-forward шага")
            return False
    
    return True


def save_results(study: optuna.Study, study_name: str, mode: str):
    """Сохраняет результаты оптимизации."""
    if not study.best_trial:
        logger.warning("Нет успешных trials для сохранения")
        return
    
    # Создаем директории
    Path("outputs/studies").mkdir(parents=True, exist_ok=True)
    Path("outputs/best_params").mkdir(parents=True, exist_ok=True)
    
    # Сохраняем лучшие параметры
    best_params_file = f"outputs/best_params/{study_name}_{mode}.json"
    results_data = {
        'study_name': study_name,
        'mode': mode,
        'best_value': study.best_trial.value,
        'best_params': study.best_trial.params,
        'n_trials': len(study.trials),
        'timestamp': str(datetime.now())
    }
    
    with open(best_params_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Лучшие параметры сохранены: {best_params_file}")
    
    # Выводим результаты
    logger.info(f"🎯 Лучший результат: {study.best_trial.value:.6f}")
    logger.info("📊 Лучшие параметры:")
    for key, value in study.best_trial.params.items():
        logger.info(f"   {key}: {value}")


def run_optimization(
    mode: str = "fast",
    n_trials: int = 200,
    study_name: Optional[str] = None,
    storage_path: Optional[str] = None,
    base_config_path: str = "configs/main_2024.yaml",
    search_space_path: str = "configs/search_space.yaml",
    n_jobs: int = 1,
    seed: int = 42,
    timeout: Optional[int] = None
) -> bool:
    """
    Запуск унифицированной оптимизации.

    Args:
        mode: Режим оптимизации ('fast', 'full', 'simple', 'bp')
        n_trials: Количество trials
        study_name: Имя исследования (автогенерируется если None)
        storage_path: Путь к базе данных (автогенерируется если None)
        base_config_path: Путь к базовой конфигурации
        search_space_path: Путь к пространству поиска
        n_jobs: Количество параллельных процессов
        seed: Seed для воспроизводимости
        timeout: Таймаут в секундах

    Returns:
        True если оптимизация прошла успешно
    """

    # Установка seeds
    set_seeds(seed)

    # Автогенерация имени study
    if study_name is None:
        study_name = f"{mode}_optimization_{datetime.now():%Y%m%d_%H%M%S}"

    # Автогенерация storage path
    if storage_path is None:
        Path("outputs/studies").mkdir(parents=True, exist_ok=True)
        storage_path = f"sqlite:///outputs/studies/{study_name}.db"

    # Настройка логирования
    study_logger = setup_logging(study_name)

    # Проверка предварительных условий
    if not check_prerequisites(mode):
        return False

    study_logger.info(f"🚀 Запуск {mode.upper()} оптимизации")
    study_logger.info(f"📊 Study: {study_name}")
    study_logger.info(f"🔢 Trials: {n_trials}")
    study_logger.info(f"📁 Storage: {storage_path}")
    study_logger.info(f"⚙️  Base config: {base_config_path}")
    if mode in ["fast", "full", "bp"]:
        study_logger.info(f"🔍 Search space: {search_space_path}")

    try:
        # Создание целевой функции в зависимости от режима
        objective_func = create_objective(mode, base_config_path, search_space_path)

        # Создание study
        study = create_study(study_name, storage_path, mode, n_trials)

        # Запуск оптимизации
        study_logger.info("⏱️  Начало оптимизации...")
        start_time = time.time()

        study.optimize(
            objective_func,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout
        )

        optimization_time = time.time() - start_time
        study_logger.info(f"⏱️  Время оптимизации: {optimization_time:.1f} секунд")

        # Сохранение результатов
        save_results(study, study_name, mode)

        # Статистика
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        study_logger.info(f"✅ Завершено trials: {len(completed_trials)}/{len(study.trials)}")

        if len(completed_trials) >= 10:
            study_logger.info("🎉 Оптимизация завершена успешно!")
            return True
        else:
            study_logger.warning("⚠️  Мало успешных trials. Возможно, нужно ослабить фильтры.")
            return False

    except Exception as e:
        study_logger.error(f"❌ Ошибка оптимизации: {e}")
        import traceback
        study_logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def main():
    """Главная функция с CLI интерфейсом."""
    parser = argparse.ArgumentParser(
        description="Унифицированный скрипт оптимизации стратегии парного трейдинга",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Режимы оптимизации:
  fast    - Быстрая оптимизация на предотобранных парах (рекомендуется)
  simple  - Упрощенная оптимизация без сложностей
  bp      - Best Practice оптимизация с анти-чурн штрафами
  full    - Полная оптимизация со всеми функциями

Примеры использования:
  # Быстрая оптимизация (рекомендуется)
  python src/optimiser/unified_optimization.py --mode fast --trials 200

  # Простая оптимизация для тестирования
  python src/optimiser/unified_optimization.py --mode simple --trials 50 --timeout 300

  # Best Practice оптимизация
  python src/optimiser/unified_optimization.py --mode bp --trials 400 --n-jobs 2
        """
    )

    parser.add_argument(
        "--mode",
        choices=["fast", "simple", "bp", "full"],
        default="fast",
        help="Режим оптимизации (по умолчанию: fast)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=200,
        help="Количество trials (по умолчанию: 200)"
    )
    parser.add_argument(
        "--study-name",
        help="Имя исследования (автогенерируется если не указано)"
    )
    parser.add_argument(
        "--storage",
        help="Путь к базе данных SQLite (автогенерируется если не указано)"
    )
    parser.add_argument(
        "--base-config",
        default="configs/main_2024.yaml",
        help="Путь к базовой конфигурации (по умолчанию: configs/main_2024.yaml)"
    )
    parser.add_argument(
        "--search-space",
        default="configs/search_space.yaml",
        help="Путь к пространству поиска (по умолчанию: configs/search_space.yaml)"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Количество параллельных процессов (по умолчанию: 1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed для воспроизводимости (по умолчанию: 42)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Таймаут в секундах (без ограничения если не указано)"
    )

    args = parser.parse_args()

    # Запуск оптимизации
    success = run_optimization(
        mode=args.mode,
        n_trials=args.trials,
        study_name=args.study_name,
        storage_path=args.storage,
        base_config_path=args.base_config,
        search_space_path=args.search_space,
        n_jobs=args.n_jobs,
        seed=args.seed,
        timeout=args.timeout
    )

    if success:
        print("\n🎉 ОПТИМИЗАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
        print("\n💡 Следующие шаги:")
        print("1. Проанализируйте результаты в outputs/best_params/")
        print("2. Запустите валидацию с лучшими параметрами")
        print("3. Обновите веб-интерфейс анализа")
    else:
        print("\n❌ ОПТИМИЗАЦИЯ НЕ УДАЛАСЬ")
        print("Проверьте логи в директории logs/")
        sys.exit(1)


if __name__ == "__main__":
    main()
