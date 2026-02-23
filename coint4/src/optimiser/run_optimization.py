#!/usr/bin/env python3
"""
Скрипт для запуска оптимизации стратегии с использованием Optuna.
"""

import sys
from pathlib import Path

import optuna
import yaml
import logging
import argparse
import random
import numpy as np
import pandas as pd
import hashlib
import os
from typing import Optional, Dict, Any

# Импорты для прямого запуска
if __name__ == "__main__":
    from fast_objective import FastWalkForwardObjective
    from sqlite_optimizer import create_optimized_study
    from coint2.utils.config import load_config
else:
    from .fast_objective import FastWalkForwardObjective
    from .sqlite_optimizer import create_optimized_study
    from coint2.utils.config import load_config

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _fmt4(x):
    """Безопасное форматирование числовых значений."""
    return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)


def _convert_numpy_types(obj):
    """Рекурсивно преобразует numpy типы в python типы для безопасной сериализации."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    else:
        return obj


def _compute_config_hash(base_config_path: str, search_space_path: str, preselected_pairs_path: str = None) -> str:
    """Вычисляет хэш конфигурации для проверки совместимости study."""
    hash_obj = hashlib.sha256()

    # Добавляем содержимое base config
    with open(base_config_path, 'r', encoding='utf-8') as f:
        hash_obj.update(f.read().encode('utf-8'))

    # Добавляем содержимое search space
    with open(search_space_path, 'r', encoding='utf-8') as f:
        hash_obj.update(f.read().encode('utf-8'))

    # Добавляем хэш списка пар для предотвращения смешивания разных наборов
    if preselected_pairs_path and os.path.exists(preselected_pairs_path):
        with open(preselected_pairs_path, 'r', encoding='utf-8') as f:
            hash_obj.update(f.read().encode('utf-8'))

    return hash_obj.hexdigest()[:16]  # Первые 16 символов для краткости


def _worker_init(base_config_path: str, search_space_path: str):
    """
    Инициализация worker процесса для глобального кэша.
    Вызывается один раз при создании каждого worker процесса.
    """
    import os
    print(f"🔄 Инициализация worker процесса PID: {os.getpid()}")

    try:
        # Создаем objective в worker процессе для инициализации кэша
        if __name__ == "__main__":
            from fast_objective import FastWalkForwardObjective
        else:
            from .fast_objective import FastWalkForwardObjective
        objective = FastWalkForwardObjective(base_config_path, search_space_path)

        # Проверяем статус кэша
        from coint2.core.global_rolling_cache import get_global_rolling_manager
        manager = get_global_rolling_manager()
        if manager.initialized:
            print(f"✅ Глобальный кэш инициализирован в worker PID: {os.getpid()}")
        else:
            print(f"❌ Глобальный кэш НЕ инициализирован в worker PID: {os.getpid()}")

    except Exception as e:
        print(f"❌ Ошибка инициализации worker PID {os.getpid()}: {e}")


def run_optimization(n_trials: int = 50,
                    study_name: str = "pairs_strategy_v1",
                    storage_path: str = "outputs/studies/pairs_strategy_v1.db",
                    base_config_path: str = "configs/main_2024.yaml",
                    search_space_path: str = "configs/search_spaces/fast.yaml",
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
        # ОПТИМИЗАЦИЯ: Настройка оптимизированного threading
        from coint2.core.memory_optimization import setup_optimized_threading
        threading_result = setup_optimized_threading(n_jobs=n_jobs, verbose=True)
        logger.info(f"🔧 Threading настроен: {threading_result['optimization_mode']} режим")

        logger.info(f"🚀 Запуск оптимизации: {study_name}")
        logger.info(f"📊 Количество trials: {n_trials}")
        logger.info(f"💾 База данных: {storage_path}")

        # Устанавливаем глобальные сиды для полной воспроизводимости
        random.seed(seed)
        np.random.seed(seed)
        
        # Дополнительные seed'ы для других библиотек
        import os
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # Если есть torch
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
        
        # Если есть tensorflow
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass
            
        logger.info(f"🎲 Установлены все сиды для воспроизводимости: {seed}")

        # Создаем директорию для хранения результатов
        outputs_dir = Path(storage_path).parent
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # Создаем оптимизированное хранилище
        def get_optimized_storage(storage_path: str, n_jobs: int = 1):
            """
            Создает оптимизированное storage для параллельной работы.

            Args:
                storage_path: Путь к базе данных
                n_jobs: Количество параллельных процессов

            Returns:
                Оптимизированное storage
            """
            from optuna.storages import RDBStorage

            is_sqlalchemy_url = "://" in storage_path
            if is_sqlalchemy_url:
                # PostgreSQL/MySQL - оптимально для параллельности
                if storage_path.startswith("postgresql://") or storage_path.startswith("mysql://"):
                    # Для PostgreSQL/MySQL - используем URL напрямую
                    # Современные версии Optuna не требуют engine_kwargs
                    return RDBStorage(url=storage_path)
                else:
                    # Другие удаленные БД
                    return RDBStorage(url=storage_path)

            # SQLite - оптимизируем для безопасной параллельности
            
            # Всегда настраиваем WAL режим, независимо от n_jobs
            logger.info("🔧 Настройка SQLite с WAL режимом для оптимальной производительности")
            
            import sqlite3
            try:
                # Создаём файл БД, если его нет
                from pathlib import Path
                db_path = Path(storage_path)
                db_path.parent.mkdir(parents=True, exist_ok=True)
                
                conn = sqlite3.connect(storage_path)
                # Оптимальные настройки для WAL режима
                conn.execute("PRAGMA journal_mode=WAL;")       # WAL режим для параллельности
                conn.execute("PRAGMA synchronous=NORMAL;")     # Баланс между скоростью и безопасностью
                conn.execute("PRAGMA cache_size=-64000;")      # 64MB кэш в памяти
                conn.execute("PRAGMA temp_store=MEMORY;")      # Временные данные в памяти
                conn.execute("PRAGMA busy_timeout=60000;")     # 60 секунд таймаут для блокировок
                conn.execute("PRAGMA wal_autocheckpoint=1000;") # Частые checkpoint'ы для WAL
                conn.execute("PRAGMA mmap_size=268435456;")    # 256MB memory-mapped I/O
                conn.close()
                logger.info("✅ SQLite настроен с WAL режимом для оптимальной производительности")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось настроить WAL режим: {e}")

            # Для SQLite используем только URL без engine_kwargs
            # WAL режим уже настроен выше через прямое подключение
            sqlite_url = f"sqlite:///{storage_path}"
            return RDBStorage(url=sqlite_url)

        storage = get_optimized_storage(storage_path, n_jobs)

        # Разрешаем параллельность для SQLite с WAL режимом
        if storage_path.endswith('.db') or ('sqlite' in storage_path and '://' not in storage_path):
            if n_jobs > 1:
                logger.info(f"🚀 Используем SQLite с параллельностью: {n_jobs} процессов")
                logger.info("   WAL режим обеспечивает безопасную параллельную работу")
                # Ограничиваем количество процессов для SQLite для стабильности
                if n_jobs > 4:
                    logger.warning(f"⚠️ Ограничиваем n_jobs до 4 для SQLite (было: {n_jobs})")
                    n_jobs = 4

        # Создаем objective-функцию
        logger.info("🎯 Создание БЫСТРОЙ objective-функции...")
        objective = FastWalkForwardObjective(
            base_config_path=base_config_path,
            search_space_path=search_space_path
        )

        logger.info("📈 Создание study...")

        # Улучшенная логика startup trials - более консервативная для малых бюджетов
        n_startup_trials = max(5, min(15, n_trials // 10))

        # Нормализация n_jobs для безопасности
        if n_jobs is None or n_jobs < 1:
            n_jobs = os.cpu_count() or 1
        elif n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        logger.info(f"🔧 Используем {n_jobs} потоков для оптимизации")

        # ОПТИМИЗАЦИЯ: Улучшенные настройки TPESampler для ускорения
        sampler_kwargs = {
            "seed": seed,
            "multivariate": True,
            "group": True,  # Лучше для параллельных предложений
            "constant_liar": (n_jobs > 1),  # Включаем при параллельности
            "n_startup_trials": n_startup_trials,
        }

        # ОПТИМИЗАЦИЯ: Дополнительные параметры для ускорения параллельной работы
        if n_jobs > 1:
            sampler_kwargs.update({
                "n_ei_candidates": 24,  # Больше кандидатов для лучшего параллелизма
                "warn_independent_sampling": False,  # Отключаем предупреждения
            })

        sampler = optuna.samplers.TPESampler(**sampler_kwargs)

        # Промежуточные отчеты привязаны к walk-forward шагам, а не к парам
        # Это обеспечивает стабильность прюнинга независимо от состава пар
        start_date = pd.to_datetime(objective.base_config.walk_forward.start_date)
        end_date = pd.to_datetime(getattr(
            objective.base_config.walk_forward,
            'end_date',
            start_date + pd.Timedelta(days=objective.base_config.walk_forward.testing_period_days)
        ))
        step_size_days = getattr(
            objective.base_config.walk_forward,
            'step_size_days',
            None
        )

        if step_size_days is None or step_size_days <= 0:
            refit_frequency = getattr(objective.base_config.walk_forward, 'refit_frequency', None)
            refit_map = {
                "daily": 1,
                "weekly": 7,
                "monthly": 30,
            }
            key = str(refit_frequency).lower() if refit_frequency is not None else ""
            step_size_days = refit_map.get(key, objective.base_config.walk_forward.testing_period_days)

        try:
            step_size_days = int(step_size_days)
        except (TypeError, ValueError):
            step_size_days = int(objective.base_config.walk_forward.testing_period_days)

        if step_size_days < objective.base_config.walk_forward.testing_period_days:
            step_size_days = objective.base_config.walk_forward.testing_period_days

        max_steps = getattr(objective.base_config.walk_forward, 'max_steps', None)
        if max_steps is not None:
            try:
                max_steps = int(max_steps)
            except (TypeError, ValueError):
                max_steps = None

        total_walk_forward_steps = 0
        current_step_start = start_date
        while current_step_start < end_date:
            total_walk_forward_steps += 1
            if max_steps and total_walk_forward_steps >= max_steps:
                break
            current_step_start += pd.Timedelta(days=step_size_days)

        total_walk_forward_steps = max(1, total_walk_forward_steps)

        total_reports = max(1, total_walk_forward_steps)

        # ОПТИМИЗАЦИЯ: Агрессивный pruning для ускорения
        if total_reports < 2:
            logger.info(f"🚫 Отключаем pruner: слишком мало отчетов ({total_reports})")
            pruner = optuna.pruners.NopPruner()
        else:
            # ОПТИМИЗАЦИЯ: Более агрессивные настройки pruner для быстрого отсева
            n_warmup_steps = max(1, min(2, total_reports // 3))  # Меньше warmup для быстрого отсева
            interval_steps = min(3, max(1, total_reports // 4))  # Более частая проверка

            logger.info(f"✂️  Агрессивный pruner: {n_warmup_steps} warmup, проверка каждые {interval_steps} шагов")
            pruner = optuna.pruners.MedianPruner(
                n_warmup_steps=n_warmup_steps,
                interval_steps=interval_steps,
                n_min_trials=max(3, n_startup_trials // 2)  # Минимум trials для начала pruning
            )

        # ОПТИМИЗАЦИЯ SQLite: Используем оптимизированное создание study
        study = create_optimized_study(
            study_name=study_name,
            db_path=storage_path,  # Исправлено: используем storage_path вместо db_path
            direction="maximize",
            n_jobs=n_jobs,
            sampler=sampler,
            pruner=pruner
        )

        # Проверяем совместимость конфигурации включая список пар
        preselected_pairs_path = "outputs/preselected_pairs.csv"
        config_hash = _compute_config_hash(base_config_path, search_space_path, preselected_pairs_path)
        logger.info(f"🔐 Хэш конфигурации (включая пары): {config_hash}")

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

        # УЛУЧШЕНИЕ: Добавляем базовую точку для TPE с правильными значениями
        if len(study.trials) == 0:  # Только если study пустой
            logger.info("🎯 Добавляем базовую точку для TPE...")

            def _pick_base(value, low, high, step=None, as_int=False):
                if value is None:
                    base = (low + high) / 2
                else:
                    base = max(low, min(high, value))
                if step is not None and step > 0:
                    base = low + round((base - low) / step) * step
                if as_int:
                    base = int(round(base))
                return base

            with open(search_space_path, "r", encoding="utf-8") as f:
                search_space = yaml.safe_load(f) or {}

            base_cfg = objective.base_config
            base_params = {}

            signals = search_space.get("signals", {})
            if "zscore_threshold" in signals:
                cfg = signals["zscore_threshold"]
                base_params["zscore_threshold"] = _pick_base(
                    getattr(base_cfg.backtest, "zscore_threshold", None),
                    cfg["low"],
                    cfg["high"],
                )
            if "rolling_window" in signals:
                cfg = signals["rolling_window"]
                base_params["rolling_window"] = _pick_base(
                    getattr(base_cfg.backtest, "rolling_window", None),
                    cfg["low"],
                    cfg["high"],
                    step=cfg.get("step"),
                    as_int=True,
                )
            if "zscore_exit" in signals:
                cfg = signals["zscore_exit"]
                threshold = base_params.get("zscore_threshold")
                z_low = cfg["low"]
                z_high = cfg["high"]
                if threshold is not None:
                    z_high = min(z_high, threshold - 0.1)
                    z_low = max(z_low, -threshold + 0.1)
                base_params["zscore_exit"] = _pick_base(
                    getattr(base_cfg.backtest, "zscore_exit", None),
                    z_low,
                    z_high,
                )

            risk = search_space.get("risk_management", {})
            if "stop_loss_multiplier" in risk:
                cfg = risk["stop_loss_multiplier"]
                base_params["stop_loss_multiplier"] = _pick_base(
                    getattr(base_cfg.backtest, "stop_loss_multiplier", None),
                    cfg["low"],
                    cfg["high"],
                )
            if "time_stop_multiplier" in risk:
                cfg = risk["time_stop_multiplier"]
                base_params["time_stop_multiplier"] = _pick_base(
                    getattr(base_cfg.backtest, "time_stop_multiplier", None),
                    cfg["low"],
                    cfg["high"],
                )
            if "cooldown_hours" in risk:
                cfg = risk["cooldown_hours"]
                base_params["cooldown_hours"] = _pick_base(
                    getattr(base_cfg.backtest, "cooldown_hours", None),
                    cfg["low"],
                    cfg["high"],
                    step=cfg.get("step"),
                    as_int=True,
                )

            portfolio = search_space.get("portfolio", {})
            if "max_active_positions" in portfolio:
                cfg = portfolio["max_active_positions"]
                base_params["max_active_positions"] = _pick_base(
                    getattr(base_cfg.portfolio, "max_active_positions", None),
                    cfg["low"],
                    cfg["high"],
                    step=cfg.get("step"),
                    as_int=True,
                )
            if "risk_per_position_pct" in portfolio:
                cfg = portfolio["risk_per_position_pct"]
                base_params["risk_per_position_pct"] = _pick_base(
                    getattr(base_cfg.portfolio, "risk_per_position_pct", None),
                    cfg["low"],
                    cfg["high"],
                )
            if "max_position_size_pct" in portfolio:
                cfg = portfolio["max_position_size_pct"]
                base_params["max_position_size_pct"] = _pick_base(
                    getattr(base_cfg.portfolio, "max_position_size_pct", None),
                    cfg["low"],
                    cfg["high"],
                )

            normalization = search_space.get("normalization", {})
            if "normalization_method" in normalization:
                base_params["normalization_method"] = getattr(
                    base_cfg.data_processing, "normalization_method", "rolling_zscore"
                )
            if "min_history_ratio" in normalization:
                cfg = normalization["min_history_ratio"]
                base_params["min_history_ratio"] = _pick_base(
                    getattr(base_cfg.data_processing, "min_history_ratio", None),
                    cfg["low"],
                    cfg["high"],
                )

            study.enqueue_trial(base_params)

        # Запускаем оптимизацию
        logger.info(f"⚡ Запуск оптимизации с {n_jobs} процессами...")

        logger.info(f"🚀 Запуск оптимизации с {n_jobs} процессами")

        if n_jobs > 1:
            logger.info("🔄 Многопроцессорный режим: каждый процесс инициализирует свой кэш")

        # Запускаем оптимизацию с настоящей параллельностью
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,  # Используем настоящую параллельность
            show_progress_bar=True,
            gc_after_trial=True
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

        metrics = best_trial.user_attrs.get("metrics", {})
        if metrics:
            logger.info("Детальные метрики лучшего trial:")
            logger.info(f"  Sharpe ratio: {_fmt4(metrics.get('sharpe'))}")
            logger.info(f"  Max drawdown: {_fmt4(metrics.get('max_drawdown'))}")
            logger.info(f"  Win rate: {_fmt4(metrics.get('win_rate'))}")
            logger.info(f"  Total trades: {metrics.get('total_trades', 'N/A')}")
            logger.info(f"  DD penalty: {_fmt4(metrics.get('dd_penalty'))}")
            logger.info(f"  Win rate bonus: {_fmt4(metrics.get('win_rate_bonus'))}")
            logger.info(f"  Win rate penalty: {_fmt4(metrics.get('win_rate_penalty'))}")
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
        if not _save_best_config(study.best_params, base_config_path, study_name):
            logger.warning("Не удалось сохранить лучшую конфигурацию")
            
        return True
        
    except Exception as e:
        logger.error(f"Критическая ошибка оптимизации: {e}")
        return False


def _save_best_config(best_params: Dict[str, Any], base_config_path: str, study_name: str = "default") -> bool:
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
        
        logger.info(f"📝 Обновляем конфигурацию параметрами: {list(best_params.keys())}")

        # Группа 1: Сигналы и торговые параметры (backtest секция)
        if "zscore_threshold" in best_params:
            best_cfg.backtest.zscore_threshold = best_params["zscore_threshold"]
            logger.info(f"   ✅ zscore_threshold: {best_params['zscore_threshold']}")
        if "zscore_exit" in best_params:
            best_cfg.backtest.zscore_exit = best_params["zscore_exit"]
            logger.info(f"   ✅ zscore_exit: {best_params['zscore_exit']}")
        if "rolling_window" in best_params:
            best_cfg.backtest.rolling_window = best_params["rolling_window"]
            logger.info(f"   ✅ rolling_window: {best_params['rolling_window']}")

        # Группа 2: Управление риском (backtest секция)
        if "stop_loss_multiplier" in best_params:
            best_cfg.backtest.stop_loss_multiplier = best_params["stop_loss_multiplier"]
            logger.info(f"   ✅ stop_loss_multiplier: {best_params['stop_loss_multiplier']}")
        if "time_stop_multiplier" in best_params:
            best_cfg.backtest.time_stop_multiplier = best_params["time_stop_multiplier"]
            logger.info(f"   ✅ time_stop_multiplier: {best_params['time_stop_multiplier']}")
        if "cooldown_hours" in best_params:
            best_cfg.backtest.cooldown_hours = best_params["cooldown_hours"]
            logger.info(f"   ✅ cooldown_hours: {best_params['cooldown_hours']}")

        # Группа 3: Издержки (backtest секция)
        if "commission_pct" in best_params:
            best_cfg.backtest.commission_pct = best_params["commission_pct"]
            logger.info(f"   ✅ commission_pct: {best_params['commission_pct']}")
        if "slippage_pct" in best_params:
            best_cfg.backtest.slippage_pct = best_params["slippage_pct"]
            logger.info(f"   ✅ slippage_pct: {best_params['slippage_pct']}")

        # Группа 4: Портфель (portfolio секция)
        if "max_active_positions" in best_params:
            best_cfg.portfolio.max_active_positions = best_params["max_active_positions"]
            logger.info(f"   ✅ max_active_positions: {best_params['max_active_positions']}")
        if "risk_per_position_pct" in best_params:
            best_cfg.portfolio.risk_per_position_pct = best_params["risk_per_position_pct"]
            logger.info(f"   ✅ risk_per_position_pct: {best_params['risk_per_position_pct']}")
        if "max_position_size_pct" in best_params:
            best_cfg.portfolio.max_position_size_pct = best_params["max_position_size_pct"]
            logger.info(f"   ✅ max_position_size_pct: {best_params['max_position_size_pct']}")

        # Группа 5: Обработка данных (data_processing секция)
        if "normalization_method" in best_params:
            best_cfg.data_processing.normalization_method = best_params["normalization_method"]
            logger.info(f"   ✅ normalization_method: {best_params['normalization_method']}")
        if "min_history_ratio" in best_params:
            best_cfg.data_processing.min_history_ratio = best_params["min_history_ratio"]
            logger.info(f"   ✅ min_history_ratio: {best_params['min_history_ratio']}")

        # Группа 6: Фильтры пар (pair_selection секция) - если есть
        if "ssd_top_n" in best_params:
            best_cfg.pair_selection.ssd_top_n = best_params["ssd_top_n"]
            logger.info(f"   ✅ ssd_top_n: {best_params['ssd_top_n']}")
        if "coint_pvalue_threshold" in best_params:
            best_cfg.pair_selection.coint_pvalue_threshold = best_params["coint_pvalue_threshold"]
            logger.info(f"   ✅ coint_pvalue_threshold: {best_params['coint_pvalue_threshold']}")
        if "min_half_life_days" in best_params:
            best_cfg.pair_selection.min_half_life_days = best_params["min_half_life_days"]
            logger.info(f"   ✅ min_half_life_days: {best_params['min_half_life_days']}")
        if "max_half_life_days" in best_params:
            best_cfg.pair_selection.max_half_life_days = best_params["max_half_life_days"]
            logger.info(f"   ✅ max_half_life_days: {best_params['max_half_life_days']}")
        if "min_mean_crossings" in best_params:
            best_cfg.pair_selection.min_mean_crossings = best_params["min_mean_crossings"]
            logger.info(f"   ✅ min_mean_crossings: {best_params['min_mean_crossings']}")

        # Группа 7: Дополнительные параметры
        if "hysteresis" in best_params:
            # Hysteresis может быть в разных местах в зависимости от версии конфигурации
            if hasattr(best_cfg.backtest, 'hysteresis'):
                best_cfg.backtest.hysteresis = best_params["hysteresis"]
                logger.info(f"   ✅ hysteresis: {best_params['hysteresis']}")

        logger.info(f"📝 Обновлено {len([k for k in best_params.keys() if k in ['zscore_threshold', 'zscore_exit', 'rolling_window', 'stop_loss_multiplier', 'time_stop_multiplier', 'cooldown_hours', 'commission_pct', 'slippage_pct', 'max_active_positions', 'risk_per_position_pct', 'max_position_size_pct', 'normalization_method', 'min_history_ratio']])} параметров из {len(best_params)}")
        
        # Создаем директорию если не существует
        config_dir = Path("configs")
        config_dir.mkdir(exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        best_config_path = f"configs/best_config__{study_name}__{timestamp}.yaml"
        config_dict = best_cfg.model_dump()

        # Безопасная сериализация с преобразованием numpy типов
        config_dict = _convert_numpy_types(config_dict)

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


def cli_main():
    """Точка входа для CLI (используется в pyproject.toml)."""
    import argparse
    parser = argparse.ArgumentParser(description="Запуск оптимизации Optuna")
    parser.add_argument("--n-trials", type=int, default=50,
                       help="Количество trials (по умолчанию: 50)")
    parser.add_argument("--study-name", default="pairs_strategy_v1",
                       help="Имя исследования")
    parser.add_argument("--storage-path", default="outputs/studies/pairs_strategy_v1.db",
                       help="Путь к базе данных")
    parser.add_argument("--base-config", default="configs/main_2024.yaml",
                       help="Путь к базовой конфигурации")
    parser.add_argument("--search-space", default="configs/search_spaces/fast.yaml",
                       help="Путь к пространству поиска")
    parser.add_argument("--n-jobs", type=int, default=-1,
                       help="Количество параллельных процессов (-1 = все ядра)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Seed для воспроизводимости")
    parser.add_argument(
        "--skip-heavy-guardrails",
        action="store_true",
        help="Disable ALLOW_HEAVY_RUN/host/resource checks (not recommended).",
    )
    parser.add_argument(
        "--heavy-allow-env",
        default=os.environ.get("HEAVY_ALLOW_ENV", "ALLOW_HEAVY_RUN"),
        help="Env var required for heavy execution (must equal 1).",
    )
    parser.add_argument(
        "--heavy-host-allowlist",
        default=os.environ.get("HEAVY_HOSTNAME_ALLOWLIST", "85.198.90.128,coint"),
        help="Comma-separated hostname/IP allowlist for heavy execution.",
    )
    parser.add_argument(
        "--heavy-min-ram-gb",
        type=float,
        default=float(os.environ.get("HEAVY_MIN_RAM_GB", "28")),
        help="Minimum RAM requirement for heavy execution.",
    )
    parser.add_argument(
        "--heavy-min-cpu",
        type=int,
        default=int(os.environ.get("HEAVY_MIN_CPU", "8")),
        help="Minimum CPU core requirement for heavy execution.",
    )
    args = parser.parse_args()

    # Валидация аргументов
    if args.n_trials <= 0:
        logger.error(f"Некорректное количество trials: {args.n_trials}")
        sys.exit(1)
    
    # Проверка существования файлов конфигурации
    if not Path(args.base_config).exists():
        logger.error(f"Файл базовой конфигурации не найден: {args.base_config}")
        sys.exit(1)
    
    if not Path(args.search_space).exists():
        logger.error(f"Файл пространства поиска не найден: {args.search_space}")
        logger.info("Доступные файлы пространства поиска:")
        for f in Path("configs/search_spaces").glob("*.yaml"):
            logger.info(f"  - {f}")
        sys.exit(1)

    if not args.skip_heavy_guardrails:
        from coint2.ops.heavy_guardrails import (
            HeavyGuardrailConfig,
            ensure_heavy_run_allowed,
            parse_host_allowlist,
        )

        ensure_heavy_run_allowed(
            HeavyGuardrailConfig(
                entrypoint="src/optimiser/run_optimization.py:cli_main",
                allow_env=str(args.heavy_allow_env),
                host_allowlist=parse_host_allowlist(str(args.heavy_host_allowlist)),
                min_ram_gb=float(args.heavy_min_ram_gb),
                min_cpu=int(args.heavy_min_cpu),
            )
        )

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


if __name__ == "__main__":
    # При запуске как скрипта используем cli_main
    cli_main()
