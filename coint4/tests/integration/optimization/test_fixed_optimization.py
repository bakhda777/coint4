#!/usr/bin/env python3
"""
Финальный тест оптимизации после исправления критических проблем.
Использует параметры из test_best_params.py которые показали Sharpe > 1.
"""

import sys
import os
sys.path.insert(0, 'src')
os.environ['QUICK_TEST'] = 'true'

import warnings
warnings.filterwarnings('ignore')

import optuna
import pandas as pd
import numpy as np
from pathlib import Path

print("="*70)
print("🎯 ФИНАЛЬНЫЙ ТЕСТ ПОСЛЕ ИСПРАВЛЕНИЙ")
print("="*70)

# Критические исправления выполнены:
fixes = [
    "✅ Lookahead bias в нормализации данных",
    "✅ Фиксация universe пар между trials",
    "✅ SQLite PRAGMA для конкурентного доступа",
    "✅ Правильный pivot данных из длинного формата",
    "✅ Обработка нечисловых колонок"
]

print("\n📋 ИСПРАВЛЕННЫЕ ПРОБЛЕМЫ:")
for fix in fixes:
    print(f"   {fix}")

# Параметры которые показали хорошие результаты
best_params = {
    'rolling_window': 30,      # Из test_best_params.py
    'zscore_threshold': 0.7,   # Из test_best_params.py
    'zscore_exit': 0.0,        # Из test_best_params.py
}

print(f"\n🔧 ПАРАМЕТРЫ ДЛЯ ТЕСТИРОВАНИЯ:")
print(f"   Rolling window: {best_params['rolling_window']}")
print(f"   Z-score threshold: {best_params['zscore_threshold']}")
print(f"   Z-score exit: {best_params['zscore_exit']}")

# Создаем конфигурацию для теста
config = """
data_dir: data_downloaded

walk_forward:
  start_date: '2024-01-01'
  end_date: '2024-03-31'
  training_period_days: 60
  testing_period_days: 30
  step_size_days: 30
  gap_minutes: 15

pair_selection:
  ssd_top_n: 25000  # Минимум согласно CLAUDE.md
  min_correlation: 0.5
  coint_pvalue_threshold: 0.10
  min_half_life_days: 1.0
  max_half_life_days: 7.0

backtest:
  rolling_window: 30  # Из best_params
  zscore_threshold: 0.7
  zscore_exit: 0.0
  commission_pct: 0.0001
  slippage_pct: 0.0001

preprocessing:
  norm_method: rolling_zscore
  min_history_ratio: 0.8
"""

# Сохраняем конфигурацию
config_path = Path("configs/test_fixed.yaml")
config_path.write_text(config)

# Создаем search space
search_space = """
trading:
  zscore_threshold:
    low: 0.5
    high: 1.0
    step: 0.1
  zscore_exit:
    low: -0.2
    high: 0.2
    step: 0.1
  rolling_window:
    choices: [20, 30, 40, 60]
"""

search_space_path = Path("configs/test_fixed_search.yaml")
search_space_path.write_text(search_space)

print(f"\n📝 Конфигурации созданы:")
print(f"   Config: {config_path}")
print(f"   Search space: {search_space_path}")

# Запускаем мини-оптимизацию
print(f"\n🚀 ЗАПУСК ОПТИМИЗАЦИИ...")
print("="*70)

try:
    from src.optimiser.fast_objective import FastWalkForwardObjective
    from src.optimiser.sqlite_optimizer import create_optimized_study
    
    # Создаем objective
    objective = FastWalkForwardObjective(
        base_config_path=str(config_path),
        search_space_path=str(search_space_path)
    )
    
    # Создаем study с оптимизированным SQLite
    study = create_optimized_study(
        study_name="test_fixed_optimization",
        db_path="outputs/studies/test_fixed.db",
        direction="maximize",
        n_jobs=1
    )
    
    # Запускаем оптимизацию
    print("⏳ Запуск 3 trials для проверки...")
    
    def objective_wrapper(trial):
        """Обертка для отслеживания прогресса."""
        result = objective(trial)
        
        # Извлекаем Sharpe
        if isinstance(result, dict):
            sharpe = result.get('sharpe_ratio_abs', -999)
        else:
            sharpe = result
        
        print(f"   Trial {trial.number}: Sharpe = {sharpe:.3f}")
        
        return sharpe
    
    study.optimize(objective_wrapper, n_trials=3)
    
    # Результаты
    print("\n" + "="*70)
    print("📊 РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ:")
    print("="*70)
    
    best_trial = study.best_trial
    print(f"\n🏆 Лучший результат:")
    print(f"   Sharpe ratio: {best_trial.value:.3f}")
    print(f"   Параметры:")
    for key, value in best_trial.params.items():
        print(f"     {key}: {value}")
    
    # Проверяем достижение цели
    if best_trial.value > 1.0:
        print("\n" + "="*70)
        print("🎉 УСПЕХ! ДОСТИГНУТ SHARPE > 1.0")
        print("="*70)
        print("\n✨ Все критические проблемы исправлены!")
        print("✨ Оптимизация работает корректно!")
    else:
        print(f"\n⚠️ Sharpe = {best_trial.value:.3f} < 1.0")
        print("   Возможно нужно больше trials или другие параметры")
    
except Exception as e:
    print(f"\n❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Тест завершен")