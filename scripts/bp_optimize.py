#!/usr/bin/env python3
"""
Best Practice Optuna Optimization Script
Реализует правильную логику zscore с гистерезисом и анти-чурн штрафами
"""

import argparse
import yaml
import optuna
import numpy as np
import random
import json
import logging
from pathlib import Path
from datetime import datetime
import sys
import os

# Добавляем путь к модулям проекта
sys.path.append(str(Path(__file__).parent.parent / "src"))

from coint2.pipeline.walk_forward_orchestrator import FastWalkForwardObjective


def setup_logging(study_name: str) -> logging.Logger:
    """Настройка логирования"""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    log_file = logs_dir / f"{study_name}_bp_optimization.log"
    
    logger = logging.getLogger(f"bp_optuna_{study_name}")
    logger.setLevel(logging.INFO)
    
    # Удаляем существующие handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def load_search_space(space_file: str) -> dict:
    """Загрузка пространства поиска"""
    with open(space_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_seeds(seed: int = 42):
    """Установка seeds для воспроизводимости"""
    random.seed(seed)
    np.random.seed(seed)
    # Если используется torch/tensorflow, добавить их seeds


def create_objective_function(base_config: str, search_space: dict, logger: logging.Logger):
    """Создание objective функции с best practices"""
    
    # Загружаем базовый конфиг
    with open(base_config, 'r', encoding='utf-8') as f:
        base_params = yaml.safe_load(f)
    
    # Создаем objective
    objective = FastWalkForwardObjective(base_params)
    
    # Настройки валидации из search space
    validation = search_space.get('validation', {})
    min_zscore_gap = validation.get('min_zscore_gap', 0.7)
    max_zscore_exit = validation.get('max_zscore_exit', 0.6)
    max_trades_per_day = validation.get('max_trades_per_day', 5)
    anti_churn_penalty = validation.get('anti_churn_penalty', 0.02)
    
    def bp_objective(trial):
        """Best Practice Objective Function"""
        trial_start = datetime.now()
        
        logger.info(f"\n--- TRIAL {trial.number + 1} ---")
        logger.info(f"Время начала: {trial_start}")
        
        try:
            # 1. Сэмплирование zscore параметров с гистерезисом
            z_entry = trial.suggest_float("zscore_threshold", 
                                        search_space['signals']['zscore_threshold']['low'],
                                        search_space['signals']['zscore_threshold']['high'])
            
            gap = trial.suggest_float("hysteresis",
                                    search_space['signals']['hysteresis']['low'], 
                                    search_space['signals']['hysteresis']['high'])
            
            # 2. Вычисление zscore_exit
            zscore_exit = min(max_zscore_exit, max(0.0, z_entry - gap))
            
            # 3. Валидация параметров
            if zscore_exit > z_entry - min_zscore_gap:
                logger.warning(f"Невалидные zscore параметры: entry={z_entry:.3f}, exit={zscore_exit:.3f}, gap={gap:.3f}")
                raise optuna.TrialPruned("Недостаточный gap между zscore_threshold и zscore_exit")
            
            if z_entry < 1.6:
                raise optuna.TrialPruned("zscore_threshold слишком низкий")
                
            if zscore_exit > max_zscore_exit:
                raise optuna.TrialPruned("zscore_exit слишком высокий")
            
            # 4. Сэмплирование остальных параметров
            params = {
                'zscore_threshold': z_entry,
                'zscore_exit': zscore_exit,
            }
            
            # Rolling window
            params['rolling_window'] = trial.suggest_int(
                'rolling_window',
                search_space['signals']['rolling_window']['low'],
                search_space['signals']['rolling_window']['high'],
                step=search_space['signals']['rolling_window'].get('step', 1)
            )
            
            # Portfolio parameters
            params['max_active_positions'] = trial.suggest_int(
                'max_active_positions',
                search_space['portfolio']['max_active_positions']['low'],
                search_space['portfolio']['max_active_positions']['high'],
                step=search_space['portfolio']['max_active_positions'].get('step', 1)
            )
            
            params['risk_per_position_pct'] = trial.suggest_float(
                'risk_per_position_pct',
                search_space['portfolio']['risk_per_position_pct']['low'],
                search_space['portfolio']['risk_per_position_pct']['high']
            )
            
            params['max_position_size_pct'] = trial.suggest_float(
                'max_position_size_pct',
                search_space['portfolio']['max_position_size_pct']['low'],
                search_space['portfolio']['max_position_size_pct']['high']
            )
            
            # Risk management
            params['stop_loss_multiplier'] = trial.suggest_float(
                'stop_loss_multiplier',
                search_space['risk_management']['stop_loss_multiplier']['low'],
                search_space['risk_management']['stop_loss_multiplier']['high']
            )
            
            params['time_stop_multiplier'] = trial.suggest_float(
                'time_stop_multiplier',
                search_space['risk_management']['time_stop_multiplier']['low'],
                search_space['risk_management']['time_stop_multiplier']['high']
            )
            
            params['cooldown_hours'] = trial.suggest_int(
                'cooldown_hours',
                search_space['risk_management']['cooldown_hours']['low'],
                search_space['risk_management']['cooldown_hours']['high'],
                step=search_space['risk_management']['cooldown_hours'].get('step', 1)
            )
            
            # Normalization
            params['normalization_method'] = trial.suggest_categorical(
                'normalization_method',
                search_space['normalization']['normalization_method']['choices']
            )
            
            params['min_history_ratio'] = trial.suggest_float(
                'min_history_ratio',
                search_space['normalization']['min_history_ratio']['low'],
                search_space['normalization']['min_history_ratio']['high']
            )
            
            # Фиксированные параметры
            fixed = search_space.get('fixed_params', {})
            params.update(fixed)
            
            # Логирование параметров
            logger.info("ПАРАМЕТРЫ TRIAL:")
            logger.info(f"  zscore_threshold: {params['zscore_threshold']:.4f}")
            logger.info(f"  zscore_exit: {params['zscore_exit']:.4f}")
            logger.info(f"  hysteresis: {gap:.4f}")
            logger.info(f"  rolling_window: {params['rolling_window']}")
            logger.info(f"  max_active_positions: {params['max_active_positions']}")
            logger.info(f"  normalization_method: {params['normalization_method']}")
            
            # 5. Выполнение бэктеста
            backtest_start = datetime.now()

            # Получаем детальные результаты бэктеста
            backtest_result = objective._run_fast_backtest(params)
            backtest_duration = datetime.now() - backtest_start

            # Проверяем валидность результата
            if backtest_result is None or backtest_result.get('sharpe_ratio_abs') is None:
                logger.warning("Невалидный результат бэктеста")
                raise optuna.TrialPruned("Невалидный результат бэктеста")

            # Извлекаем метрики
            base_sharpe = backtest_result['sharpe_ratio_abs']
            total_trades = backtest_result.get('total_trades', 0)

            # Вычисляем trades_per_day (предполагаем 252 торговых дня в году)
            # Данные за период walk-forward, примерно 1 год
            trading_days = 252  # Можно сделать более точно из конфига
            trades_per_day = total_trades / trading_days if trading_days > 0 else 0

            # 6. Анти-чурн штраф
            penalty = anti_churn_penalty * max(0, trades_per_day - max_trades_per_day)
            final_value = base_sharpe - penalty

            # Логирование результата
            logger.info(f"РЕЗУЛЬТАТ TRIAL:")
            logger.info(f"  Базовая метрика (Sharpe): {base_sharpe:.6f}")
            logger.info(f"  Всего сделок: {total_trades}")
            logger.info(f"  Сделок в день: {trades_per_day:.2f}")
            logger.info(f"  Анти-чурн штраф: {penalty:.6f}")
            logger.info(f"  Финальное значение: {final_value:.6f}")
            logger.info(f"  Время выполнения: {backtest_duration}")

            return final_value
            
        except optuna.TrialPruned:
            logger.info("Trial прерван (pruned)")
            raise
        except Exception as e:
            logger.error(f"ОШИБКА TRIAL: {str(e)}")
            return -10.0
    
    return bp_objective


def main():
    parser = argparse.ArgumentParser(description='Best Practice Optuna Optimization')
    parser.add_argument('--base', required=True, help='Базовый конфиг (main_2024.yaml)')
    parser.add_argument('--space', required=True, help='Пространство поиска')
    parser.add_argument('--trials', type=int, default=400, help='Количество trials')
    parser.add_argument('--study', required=True, help='Название study')
    parser.add_argument('--storage', required=True, help='Storage URL (sqlite:///studies.db)')
    parser.add_argument('--n-jobs', type=int, default=1, help='Количество параллельных процессов')
    
    args = parser.parse_args()
    
    # Установка seeds
    set_seeds(42)
    
    # Настройка логирования
    logger = setup_logging(args.study)
    
    logger.info("="*80)
    logger.info(f"НАЧАЛО BEST PRACTICE ОПТИМИЗАЦИИ: {args.study}")
    logger.info(f"Базовый конфиг: {args.base}")
    logger.info(f"Пространство поиска: {args.space}")
    logger.info(f"Количество trials: {args.trials}")
    logger.info(f"Storage: {args.storage}")
    logger.info("="*80)
    
    # Загрузка пространства поиска
    search_space = load_search_space(args.space)
    
    # Создание objective функции
    objective_func = create_objective_function(args.base, search_space, logger)
    
    # Настройка Optuna
    study = optuna.create_study(
        study_name=args.study,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=50,
            multivariate=True
        ),
        pruner=optuna.pruners.MedianPruner(
            n_warmup_steps=30,
            interval_steps=1
        )
    )
    
    # Запуск оптимизации
    logger.info(f"Запуск оптимизации на {args.trials} trials...")
    study.optimize(objective_func, n_trials=args.trials, n_jobs=args.n_jobs)
    
    # Результаты
    logger.info("\n" + "="*80)
    logger.info("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
    logger.info("="*80)
    
    if study.best_trial:
        logger.info(f"Лучший результат: {study.best_trial.value:.6f}")
        logger.info(f"Лучшие параметры:")
        for key, value in study.best_trial.params.items():
            logger.info(f"  {key}: {value}")
        
        # Сохранение лучших параметров
        best_params_file = f"best_params_{args.study}.json"
        with open(best_params_file, 'w', encoding='utf-8') as f:
            json.dump({
                'study_name': args.study,
                'best_value': study.best_trial.value,
                'best_params': study.best_trial.params,
                'timestamp': str(datetime.now())
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Лучшие параметры сохранены: {best_params_file}")
    
    logger.info(f"Завершено trials: {len(study.trials)}")
    logger.info("Оптимизация завершена!")


if __name__ == "__main__":
    main()
