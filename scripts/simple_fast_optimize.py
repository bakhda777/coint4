#!/usr/bin/env python3
"""
ПРОСТАЯ быстрая оптимизация - минимальная версия, которая точно работает.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import optuna
import time

# Добавляем src в путь для импорта
sys.path.append(str(Path(__file__).parent.parent / "src"))

from coint2.utils.config import load_config
from coint2.engine.numba_engine import NumbaPairBacktester as PairBacktester

def simple_objective(trial):
    """Простая целевая функция без лишних сложностей."""
    
    # Загружаем конфигурацию
    cfg = load_config("configs/main_2024.yaml")
    
    # Генерируем параметры
    z_entry = trial.suggest_float("z_entry", 1.2, 1.8)
    z_exit = trial.suggest_float("z_exit", -0.2, 0.2)
    
    # Загружаем предотобранные пары
    pairs_df = pd.read_csv("outputs/preselected_pairs.csv")
    
    # Загружаем данные
    full_data = pd.read_csv("outputs/full_step_data.csv", index_col=0, parse_dates=True)
    
    # Определяем тестовый период (как в оригинальной системе)
    start_date = pd.to_datetime(cfg.walk_forward.start_date)
    testing_start = start_date
    testing_end = testing_start + pd.Timedelta(days=cfg.walk_forward.testing_period_days)
    
    # Загружаем параметры нормализации
    norm_params_df = pd.read_csv("outputs/training_normalization_params.csv", index_col=0)
    norm_params = norm_params_df.iloc[:, 0].to_dict()
    
    total_pnl = 0.0
    successful_pairs = 0
    
    # Тестируем только первые 10 пар для скорости
    for _, pair_row in pairs_df.head(10).iterrows():
        s1, s2 = pair_row['s1'], pair_row['s2']
        
        # Проверяем наличие данных
        if s1 not in full_data.columns or s2 not in full_data.columns:
            continue
        if s1 not in norm_params or s2 not in norm_params:
            continue
            
        # Извлекаем данные тестового периода
        pair_data = full_data.loc[testing_start:testing_end, [s1, s2]].dropna()
        if len(pair_data) < 50:
            continue
            
        # Применяем нормализацию
        norm_s1, norm_s2 = norm_params[s1], norm_params[s2]
        if norm_s1 == 0 or norm_s2 == 0:
            continue
            
        normalized_data = pair_data.copy()
        normalized_data[s1] = (pair_data[s1] / norm_s1) * 100
        normalized_data[s2] = (pair_data[s2] / norm_s2) * 100
        
        try:
            # Создаем бэктестер
            backtester = PairBacktester(
                pair_data=normalized_data,
                rolling_window=cfg.backtest.rolling_window,
                z_threshold=z_entry,
                z_exit=z_exit,
                stop_loss_multiplier=cfg.backtest.stop_loss_multiplier,
                commission_pct=0.0004,
                slippage_pct=0.0005,
                capital_at_risk=10000,  # Фиксированный капитал
                pair_name=f"{s1}-{s2}",
                annualizing_factor=365
            )
            
            # Запускаем бэктест
            backtester.run()
            results = backtester.get_results()
            
            if results and 'pnl' in results:
                pnl_sum = results['pnl'].sum()
                if not pd.isna(pnl_sum):
                    total_pnl += pnl_sum
                    successful_pairs += 1
                    
        except Exception as e:
            continue
    
    # Возвращаем результат
    if successful_pairs == 0:
        return -999.0
    
    # Простая метрика: средний PnL на пару
    avg_pnl = total_pnl / successful_pairs
    return float(avg_pnl)

def main():
    """Запуск простой оптимизации."""
    
    print("🚀 Запуск ПРОСТОЙ быстрой оптимизации...")
    
    # Проверяем наличие файлов
    required_files = [
        "outputs/preselected_pairs.csv",
        "outputs/full_step_data.csv", 
        "outputs/training_normalization_params.csv"
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ Файл {file_path} не найден!")
            return
    
    # Создаем study
    study = optuna.create_study(direction='maximize')
    
    print("⏱️  Запуск оптимизации...")
    start_time = time.time()
    
    # Запускаем оптимизацию
    study.optimize(simple_objective, n_trials=50, timeout=300)
    
    optimization_time = time.time() - start_time
    
    # Выводим результаты
    print("\n" + "="*60)
    print("🎯 РЕЗУЛЬТАТЫ ПРОСТОЙ ОПТИМИЗАЦИИ")
    print("="*60)
    
    if study.best_trial:
        print(f"📈 Лучший результат: {study.best_value:.4f}")
        print(f"🔢 Завершенных trials: {len(study.trials)}")
        print(f"⏰ Время оптимизации: {optimization_time:.1f} сек")
        print(f"⚡ Скорость: {len(study.trials)/optimization_time*60:.1f} trials/мин")
        
        print("\n🏆 Лучшие параметры:")
        for param, value in study.best_params.items():
            print(f"   {param}: {value:.4f}")
            
        if study.best_value > -999:
            print("\n✅ Оптимизация успешна!")
            print(f"💰 Средний PnL на пару: ${study.best_value:.2f}")
        else:
            print("\n⚠️  Все trials провалились")
    else:
        print("❌ Нет успешных trials")
    
    print("="*60)

if __name__ == "__main__":
    main()
