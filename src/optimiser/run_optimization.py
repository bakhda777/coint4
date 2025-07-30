#!/usr/bin/env python3
"""
Скрипт для запуска оптимизации стратегии с использованием Optuna.
"""

import sys
from pathlib import Path

# Добавляем корневую директорию проекта в PYTHONPATH
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import optuna
import yaml
import logging
import argparse
import random
import numpy as np
import hashlib
from typing import Optional, Dict, Any

# ИСПРАВЛЕНО: Правильные импорты
from src.optimiser.fast_objective import FastWalkForwardObjective
from src.coint2.utils.config import load_config

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _compute_config_hash(base_config_path: str, search_space_path: str) -> str:
    """Вычисляет хэш конфигурации для проверки совместимости study."""
    hash_obj = hashlib.sha256()

    # Добавляем содержимое base config
    with open(base_config_path, 'r', encoding='utf-8') as f:
        hash_obj.update(f.read().encode('utf-8'))

    # Добавляем содержимое search space
    with open(search_space_path, 'r', encoding='utf-8') as f:
        hash_obj.update(f.read().encode('utf-8'))

    return hash_obj.hexdigest()[:16]  # Первые 16 символов для краткости


def run_optimization(n_trials: int = 200,
                    study_name: str = "pairs_strategy_v1",
                    storage_path: str = "outputs/studies/pairs_strategy_v1.db",
                    base_config_path: str = "configs/main_2024.yaml",
                    search_space_path: str = "configs/search_space.yaml",
                    n_jobs: int = -1,
                    seed: int = 42) -> bool:
    """Запуск оптимизации с валидацией параметров и обработкой ошибок.
    
    Args:
        n_trials: Количество trials для оптимизации
        study_name: Имя исследования
        storage_path: Путь к базе данных
        base_config_path: Путь к базовой конфигурации
        search_space_path: Путь к пространству поиска
        n_jobs: Количество параллельных процессов (-1 = все ядра)
        seed: Seed для воспроизводимости
        
    Returns:
        bool: True если оптимизация прошла успешно
    """
    # Валидация параметров
    if n_trials <= 0:
        logger.error(f"Некорректное количество trials: {n_trials}")
        return False
        
    if n_trials > 2000:
        logger.warning(f"Большое количество trials: {n_trials}. Рекомендуется <= 2000")
    
    # Проверка существования файлов конфигурации
    if not Path(base_config_path).exists():
        logger.error(f"Базовая конфигурация не найдена: {base_config_path}")
        return False
        
    if not Path(search_space_path).exists():
        logger.error(f"Пространство поиска не найдено: {search_space_path}")
        return False
    
    try:
        logger.info(f"🚀 Запуск оптимизации: {study_name}")
        logger.info(f"📊 Количество trials: {n_trials}")
        logger.info(f"💾 База данных: {storage_path}")

        # Устанавливаем глобальные сиды для воспроизводимости
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"🎲 Установлены глобальные сиды: {seed}")

        # Создаем директорию для хранения результатов
        outputs_dir = Path(storage_path).parent
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # ИСПРАВЛЕНО: Используем RDBStorage с таймаутами для SQLite
        if storage_path.endswith('.db') or 'sqlite' in storage_path:
            from optuna.storages import RDBStorage
            storage_url = f"sqlite:///{storage_path}"

            # Создаем RDBStorage с таймаутами для предотвращения блокировок
            storage = RDBStorage(
                url=storage_url,
                engine_kwargs={
                    "connect_args": {
                        "timeout": 600,  # 10 минут таймаут
                        "check_same_thread": False
                    },
                    "pool_pre_ping": True,
                    "pool_recycle": 300
                }
            )

            # ИСПРАВЛЕНО: Принудительное отключение параллельности для SQLite
            if n_jobs != 1:
                logger.warning("⚠️  SQLite НЕ поддерживает безопасную параллельность!")
                logger.warning(f"   Принудительно устанавливаем n_jobs=1 (было: {n_jobs})")
                logger.warning("   Для параллельной оптимизации используйте PostgreSQL/MySQL")
                n_jobs = 1
        else:
            storage = storage_path

        # Создаем objective-функцию
        logger.info("🎯 Создание БЫСТРОЙ objective-функции...")
        objective = FastWalkForwardObjective(
            base_config_path=base_config_path,
            search_space_path=search_space_path
        )

        # Создаем study с улучшенными настройками
        logger.info("📈 Создание study...")
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=seed,
                multivariate=True,
                group=True,  # Добавляем group для лучшей производительности
                n_startup_trials=max(10, n_trials // 10)  # Исправлено: min -> max
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=20,  # Увеличиваем для предотвращения преждевременного pruning
                n_warmup_steps=2,
                interval_steps=1
            )
        )

        # ИСПРАВЛЕНО: Проверяем совместимость конфигурации
        config_hash = _compute_config_hash(base_config_path, search_space_path)
        logger.info(f"🔐 Хэш конфигурации: {config_hash}")

        if len(study.trials) > 0:  # Если study уже существует
            existing_hash = study.user_attrs.get("config_hash")
            if existing_hash and existing_hash != config_hash:
                logger.error(f"❌ НЕСОВМЕСТИМАЯ КОНФИГУРАЦИЯ!")
                logger.error(f"   Существующий хэш: {existing_hash}")
                logger.error(f"   Новый хэш: {config_hash}")
                logger.error(f"   Измените study_name или используйте совместимую конфигурацию")
                raise ValueError(f"Study '{study_name}' создан с другой конфигурацией. "
                               f"Используйте другое имя study или совместимую конфигурацию.")
            elif not existing_hash:
                logger.warning("⚠️  Существующий study без хэша конфигурации - добавляем")

        # Сохраняем хэш конфигурации в study
        study.set_user_attr("config_hash", config_hash)
        study.set_user_attr("base_config_path", base_config_path)
        study.set_user_attr("search_space_path", search_space_path)

        # Запускаем оптимизацию
        logger.info(f"⚡ Запуск оптимизации с {n_jobs} процессами...")
        study.optimize(
            objective, 
            n_trials=n_trials, 
            n_jobs=n_jobs,
            show_progress_bar=True
        )

        # Проверяем результаты
        if len(study.trials) == 0:
            logger.error("Не было выполнено ни одного trial")
            return False
            
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed_trials) == 0:
            logger.error("Не было завершено ни одного trial")
            return False

        logger.info("\n" + "="*50)
        logger.info("🎉 ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
        logger.info("="*50)
        # Исправленное логирование с детальными метриками
        best_trial = study.best_trial
        logger.info(f"Лучший композитный скор: {best_trial.value:.6f} (trial #{best_trial.number})")

        # Логируем детальные метрики если они есть
        metrics = best_trial.user_attrs.get("metrics", {})
        if metrics:
            logger.info("Детальные метрики лучшего trial:")
            logger.info(f"  Sharpe ratio: {metrics.get('sharpe', 'N/A'):.4f}")
            logger.info(f"  Max drawdown: {metrics.get('max_drawdown', 'N/A'):.4f}")
            logger.info(f"  Win rate: {metrics.get('win_rate', 'N/A'):.4f}")
            logger.info(f"  Total trades: {metrics.get('total_trades', 'N/A')}")
            logger.info(f"  DD penalty: {metrics.get('dd_penalty', 'N/A'):.4f}")
            logger.info(f"  Win rate bonus: {metrics.get('win_rate_bonus', 'N/A'):.4f}")
            logger.info(f"  Win rate penalty: {metrics.get('win_rate_penalty', 'N/A'):.4f}")
        else:
            logger.info(f"Лучшее значение (композитный скор): {best_trial.value:.6f}")

        logger.info("Лучшие параметры:")
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")
        
        # Статистика
        failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
        logger.info(f"\nСтатистика:")
        logger.info(f"  Всего trials: {len(study.trials)}")
        logger.info(f"  Завершено: {len(completed_trials)}")
        logger.info(f"  Неудачных: {len(failed_trials)}")

        # Сохраняем лучшую конфигурацию
        if not _save_best_config(study.best_params, base_config_path):
            logger.warning("Не удалось сохранить лучшую конфигурацию")
            
        return True
        
    except Exception as e:
        logger.error(f"Критическая ошибка оптимизации: {e}")
        return False


def _save_best_config(best_params: Dict[str, Any], base_config_path: str) -> bool:
    """Сохраняет лучшую конфигурацию.
    
    Args:
        best_params: Лучшие параметры из оптимизации
        base_config_path: Путь к базовой конфигурации
        
    Returns:
        bool: True если сохранение прошло успешно
    """
    try:
        logger.info("💾 Сохранение лучшей конфигурации...")
        
        # Загружаем базовую конфигурацию
        best_cfg = load_config(base_config_path)
        
        # ИСПРАВЛЕНО: Правильное обновление параметров согласно search_space.yaml
        # Сигналы
        if "zscore_threshold" in best_params:
            best_cfg.backtest.zscore_threshold = best_params["zscore_threshold"]
        if "zscore_exit" in best_params:
            best_cfg.backtest.zscore_exit = best_params["zscore_exit"]
        
        # Управление риском
        if "stop_loss_multiplier" in best_params:
            best_cfg.backtest.stop_loss_multiplier = best_params["stop_loss_multiplier"]
        if "time_stop_multiplier" in best_params:
            best_cfg.backtest.time_stop_multiplier = best_params["time_stop_multiplier"]
        
        # Портфель
        if "max_active_positions" in best_params:
            best_cfg.portfolio.max_active_positions = best_params["max_active_positions"]
        if "risk_per_position_pct" in best_params:
            best_cfg.portfolio.risk_per_position_pct = best_params["risk_per_position_pct"]
        if "max_position_size_pct" in best_params:
            best_cfg.portfolio.max_position_size_pct = best_params["max_position_size_pct"]
        
        # Создаем директорию если не существует
        config_dir = Path("configs")
        config_dir.mkdir(exist_ok=True)
        
        # Сохраняем лучшую конфигурацию с преобразованием Path в строки
        best_config_path = "configs/best_config.yaml"
        config_dict = best_cfg.model_dump()

        # Преобразуем Path объекты в строки
        if 'data_dir' in config_dict and hasattr(config_dict['data_dir'], '__fspath__'):
            config_dict['data_dir'] = str(config_dict['data_dir'])
        if 'results_dir' in config_dict and hasattr(config_dict['results_dir'], '__fspath__'):
            config_dict['results_dir'] = str(config_dict['results_dir'])

        with open(best_config_path, "w", encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"✅ Лучшая конфигурация сохранена: {best_config_path}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при сохранении конфигурации: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск оптимизации Optuna")
    parser.add_argument("--n-trials", type=int, default=200,
                       help="Количество trials (по умолчанию: 200)")
    parser.add_argument("--study-name", default="pairs_strategy_v1",
                       help="Имя исследования")
    parser.add_argument("--storage-path", default="outputs/studies/pairs_strategy_v1.db",
                       help="Путь к базе данных")
    parser.add_argument("--base-config", default="configs/main_2024.yaml",
                       help="Путь к базовой конфигурации")
    parser.add_argument("--search-space", default="configs/search_space.yaml",
                       help="Путь к пространству поиска")
    parser.add_argument("--n-jobs", type=int, default=-1,
                       help="Количество параллельных процессов (-1 = все ядра)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Seed для воспроизводимости")
    args = parser.parse_args()

    # Валидация аргументов
    if args.n_trials <= 0:
        logger.error(f"Некорректное количество trials: {args.n_trials}")
        sys.exit(1)

    success = run_optimization(
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage_path=args.storage_path,
        base_config_path=args.base_config,
        search_space_path=args.search_space,
        n_jobs=args.n_jobs,
        seed=args.seed
    )
    
    if not success:
        logger.error("Оптимизация завершилась с ошибкой")
        sys.exit(1)
    
    logger.info("🎉 Оптимизация завершена успешно!")