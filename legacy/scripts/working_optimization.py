#!/usr/bin/env python3
"""
Рабочая оптимизация - упрощенная версия, которая точно работает.
Основана на успешном тесте из test_optimization_fixes.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import optuna
import time

# Добавляем корневую директорию проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.coint2.utils.config import load_config
from src.coint2.engine.numba_engine import NumbaPairBacktester as PairBacktester
from src.coint2.core.portfolio import Portfolio


def working_objective(trial):
    """Рабочая целевая функция, основанная на успешном тесте."""
    
    # Загружаем конфигурацию
    cfg = load_config("configs/main_2024.yaml")
    
    # Генерируем параметры для оптимизации
    z_threshold = trial.suggest_float("z_threshold", 0.8, 2.0)
    z_exit = trial.suggest_float("z_exit", -0.3, 0.3)
    stop_loss_mult = trial.suggest_float("stop_loss_multiplier", 2.0, 6.0)
    time_stop_mult = trial.suggest_float("time_stop_multiplier", 5.0, 15.0)
    risk_per_pos = trial.suggest_float("risk_per_position_pct", 0.01, 0.05)
    
    # Загружаем предотобранные пары
    pairs_df = pd.read_csv("outputs/preselected_pairs.csv")
    
    # Используем готовые данные из outputs
    try:
        full_data = pd.read_csv("outputs/full_step_data.csv", index_col=0, parse_dates=True)
        norm_params_df = pd.read_csv("outputs/training_normalization_params.csv", index_col=0)
        norm_params = norm_params_df.iloc[:, 0].to_dict()
    except FileNotFoundError:
        # Если файлов нет, создаем синтетические данные
        print("📊 Создаем синтетические данные для тестирования...")
        return create_synthetic_test(trial, z_threshold, z_exit, stop_loss_mult, time_stop_mult, risk_per_pos)
    
    # Определяем тестовый период
    start_date = pd.to_datetime(cfg.walk_forward.start_date)
    testing_start = start_date
    testing_end = testing_start + pd.Timedelta(days=cfg.walk_forward.testing_period_days)
    
    total_pnl = 0.0
    successful_pairs = 0
    total_trades = 0
    
    # Тестируем первые 20 пар для скорости
    for _, pair_row in pairs_df.head(20).iterrows():
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
            # Создаем портфель
            portfolio = Portfolio(initial_capital=10000, max_active_positions=1)
            
            # Создаем бэктестер с оптимизируемыми параметрами
            backtester = PairBacktester(
                pair_data=normalized_data,
                rolling_window=cfg.backtest.rolling_window,
                z_threshold=z_threshold,
                z_exit=z_exit,
                stop_loss_multiplier=stop_loss_mult,
                time_stop_multiplier=time_stop_mult,
                commission_pct=0.0004,
                slippage_pct=0.0005,
                capital_at_risk=10000 * risk_per_pos,
                portfolio=portfolio,
                pair_name=f"{s1}-{s2}",
                annualizing_factor=365
            )
            
            # Запускаем бэктест
            backtester.run()
            results = backtester.get_results()
            
            if results and 'pnl' in results:
                if isinstance(results['pnl'], pd.Series):
                    pnl_sum = results['pnl'].sum()
                    trades_count = len(results['pnl'][results['pnl'] != 0])
                else:
                    pnl_sum = results['pnl']
                    trades_count = 1 if pnl_sum != 0 else 0
                
                if not pd.isna(pnl_sum) and trades_count > 0:
                    total_pnl += pnl_sum
                    total_trades += trades_count
                    successful_pairs += 1
                    
        except Exception as e:
            print(f"Ошибка для пары {s1}-{s2}: {e}")
            continue
    
    # Возвращаем результат
    if successful_pairs == 0 or total_trades < 5:
        return -999.0
    
    # Простая метрика: средний PnL на сделку
    avg_pnl_per_trade = total_pnl / total_trades
    
    print(f"Trial {trial.number}: {successful_pairs} пар, {total_trades} сделок, avg PnL: ${avg_pnl_per_trade:.2f}")
    
    return float(avg_pnl_per_trade)


def create_synthetic_test(trial, z_threshold, z_exit, stop_loss_mult, time_stop_mult, risk_per_pos):
    """Создает синтетический тест если нет данных."""
    
    # Создаем синтетические коинтегрированные данные
    np.random.seed(42 + trial.number)
    n_periods = 500
    dates = pd.date_range('2024-01-01', periods=n_periods, freq='15min')
    
    # Создаем коинтегрированные данные
    price1 = 100 + np.cumsum(np.random.normal(0, 0.5, n_periods))
    noise = np.random.normal(0, 0.2, n_periods)
    price2 = 0.8 * price1 + 10 + noise
    
    test_data = pd.DataFrame({
        'price1': price1,
        'price2': price2
    }, index=dates)
    
    # Создаем портфель
    portfolio = Portfolio(initial_capital=10000, max_active_positions=1)
    
    # Создаем бэктестер
    backtester = PairBacktester(
        pair_data=test_data,
        rolling_window=30,
        z_threshold=z_threshold,
        z_exit=z_exit,
        stop_loss_multiplier=stop_loss_mult,
        time_stop_multiplier=time_stop_mult,
        commission_pct=0.0004,
        slippage_pct=0.0005,
        capital_at_risk=10000 * risk_per_pos,
        portfolio=portfolio,
        pair_name="SYNTHETIC/PAIR",
        annualizing_factor=365
    )
    
    # Запускаем бэктест
    backtester.run()
    results = backtester.get_results()
    
    if results and 'pnl' in results:
        if isinstance(results['pnl'], pd.Series):
            pnl_sum = results['pnl'].sum()
            trades_count = len(results['pnl'][results['pnl'] != 0])
        else:
            pnl_sum = results['pnl']
            trades_count = 1 if pnl_sum != 0 else 0
        
        if trades_count > 0:
            avg_pnl_per_trade = pnl_sum / trades_count
            print(f"Synthetic trial {trial.number}: {trades_count} сделок, avg PnL: ${avg_pnl_per_trade:.2f}")
            return float(avg_pnl_per_trade)
    
    return -999.0


def main():
    """Запуск рабочей оптимизации."""
    
    print("🚀 Запуск РАБОЧЕЙ оптимизации...")
    print("📊 Основана на успешном тесте из test_optimization_fixes.py")
    
    # Создаем study
    study = optuna.create_study(direction='maximize')
    
    print("⏱️  Запуск оптимизации...")
    start_time = time.time()
    
    # Запускаем оптимизацию
    study.optimize(working_objective, n_trials=30, timeout=600)
    
    optimization_time = time.time() - start_time
    
    # Выводим результаты
    print("\n" + "="*60)
    print("🎯 РЕЗУЛЬТАТЫ РАБОЧЕЙ ОПТИМИЗАЦИИ")
    print("="*60)
    
    if study.best_trial:
        print(f"📈 Лучший результат: {study.best_value:.4f}")
        print(f"🔢 Завершенных trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        print(f"⏰ Время оптимизации: {optimization_time:.1f} сек")
        print(f"⚡ Скорость: {len(study.trials)/optimization_time*60:.1f} trials/мин")
        
        print("\n🏆 Лучшие параметры:")
        for param, value in study.best_params.items():
            print(f"   {param}: {value:.4f}")
            
        if study.best_value > -999:
            print("\n✅ Оптимизация успешна!")
            print(f"💰 Средний PnL на сделку: ${study.best_value:.2f}")
        else:
            print("\n⚠️  Все trials провалились")
    else:
        print("❌ Нет успешных trials")
    
    print("="*60)


if __name__ == "__main__":
    main()
