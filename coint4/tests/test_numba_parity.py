"""
Тест паритета между Reference и Numba движками.
Критически важный тест для обеспечения одинакового поведения.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from typing import Dict, Tuple
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_test_data() -> pd.DataFrame:
    """Подготовка тестовых данных для обоих движков."""
    np.random.seed(42)
    n_points = 2000
    
    # Генерируем коинтегрированную пару
    t = np.arange(n_points)
    
    # Общий тренд
    trend = 0.001 * t + 0.5 * np.sin(t / 100)
    
    # Asset 1 (y)
    noise1 = np.random.normal(0, 0.02, n_points)
    y = 100 + trend + np.cumsum(noise1)
    
    # Asset 2 (x) - коинтегрирован с y
    beta = 1.2
    noise2 = np.random.normal(0, 0.015, n_points)
    x = (100 + trend) / beta + np.cumsum(noise2)
    
    # Добавляем mean-reverting spread
    spread_noise = np.random.normal(0, 0.5, n_points)
    for i in range(1, n_points):
        spread_noise[i] = 0.8 * spread_noise[i-1] + 0.2 * np.random.normal(0, 0.5)
    
    y = y + spread_noise
    
    df = pd.DataFrame({
        'y': y,
        'x': x,
        'timestamp': pd.date_range('2024-01-01', periods=n_points, freq='15min')
    })
    
    # Убираем NaN и конвертируем в float64
    df = df[['y', 'x']].dropna().astype('float64')
    
    return df


def run_reference_engine(df: pd.DataFrame, params: Dict) -> Dict:
    """Запуск Reference движка."""
    from src.coint2.engine.reference_engine import ReferenceEngine
    
    engine = ReferenceEngine(
        rolling_window=params['rolling_window'],
        z_enter=params['z_enter'],
        z_exit=params['z_exit'],
        max_holding_period=params['max_holding_period'],
        commission_pct=params['commission_pct'],
        slippage_pct=params['slippage_pct'],
        verbose=False
    )
    
    # Reference engine ожидает колонки symbol1/symbol2
    results = engine.backtest(df, symbol1_col='y', symbol2_col='x')
    
    return {
        'positions': results['positions'],
        'trades': None,  # Reference engine не возвращает trades
        'pnl': results['pnl'],
        'z_scores': results.get('z_scores'),
        'num_trades': results['num_trades'],
        'total_pnl': results['total_pnl'],
        'sharpe': results.get('sharpe_ratio', 0)
    }


def run_numba_engine(df: pd.DataFrame, params: Dict) -> Dict:
    """Запуск Numba движка."""
    from src.coint2.core.numba_parity_v2 import compute_positions_v2
    
    y = df['y'].to_numpy()
    x = df['x'].to_numpy()
    
    positions, trades, pnl_series, z_scores, spreads = compute_positions_v2(
        y=y,
        x=x,
        rolling_window=params['rolling_window'],
        z_enter=params['z_enter'],
        z_exit=params['z_exit'],
        max_holding_period=params['max_holding_period'],
        commission=params['commission_pct'],
        slippage=params['slippage_pct']
    )
    
    num_trades = np.sum(np.abs(np.diff(positions)) > 0)
    total_pnl = np.sum(pnl_series)
    
    # Простой расчет Sharpe
    if len(pnl_series) > 1:
        returns = np.diff(pnl_series)
        returns = returns[~np.isnan(returns)]
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 96)  # 15min bars
        else:
            sharpe = 0
    else:
        sharpe = 0
    
    return {
        'positions': positions,
        'trades': trades,
        'pnl': pnl_series,
        'z_scores': z_scores,
        'num_trades': num_trades,
        'total_pnl': total_pnl,
        'sharpe': sharpe
    }


def compare_engines(ref_results: Dict, numba_results: Dict, tolerance: float = 0.1):
    """Сравнение результатов двух движков."""
    
    print("\n" + "="*60)
    print("СРАВНЕНИЕ ДВИЖКОВ")
    print("="*60)
    
    # 1. Сравнение числа сделок
    print(f"\n📊 Количество сделок:")
    print(f"  Reference: {ref_results['num_trades']}")
    print(f"  Numba:     {numba_results['num_trades']}")
    
    if ref_results['num_trades'] == 0 and numba_results['num_trades'] == 0:
        print("  ⚠️ Оба движка не сгенерировали сделок!")
        return False
    
    trade_diff = abs(ref_results['num_trades'] - numba_results['num_trades'])
    trade_match = trade_diff / max(ref_results['num_trades'], 1) < tolerance
    print(f"  Совпадение: {'✅' if trade_match else '❌'} (разница: {trade_diff})")
    
    # 2. Сравнение позиций
    ref_pos = ref_results['positions']
    numba_pos = numba_results['positions']
    
    # Находим изменения позиций
    ref_changes = np.where(np.diff(ref_pos) != 0)[0]
    numba_changes = np.where(np.diff(numba_pos) != 0)[0]
    
    print(f"\n📍 Смены позиций:")
    print(f"  Reference: {len(ref_changes)} смен")
    print(f"  Numba:     {len(numba_changes)} смен")
    
    # Проверяем совпадение индексов (с допуском ±1 бар)
    if len(ref_changes) > 0 and len(numba_changes) > 0:
        matches = 0
        for ref_idx in ref_changes[:10]:  # Проверяем первые 10
            if np.any(np.abs(numba_changes - ref_idx) <= 1):
                matches += 1
        match_pct = matches / min(10, len(ref_changes)) * 100
        print(f"  Совпадение индексов: {match_pct:.1f}%")
    
    # 3. Сравнение z-scores
    if ref_results['z_scores'] is not None and numba_results['z_scores'] is not None:
        ref_z = ref_results['z_scores']
        numba_z = numba_results['z_scores']
        
        # Убираем NaN для сравнения
        ref_z_clean = ref_z[~np.isnan(ref_z)]
        numba_z_clean = numba_z[~np.isnan(numba_z)]
        
        if len(ref_z_clean) > 0 and len(numba_z_clean) > 0:
            print(f"\n📈 Z-scores статистика:")
            print(f"  Reference: max|z| = {np.max(np.abs(ref_z_clean)):.2f}")
            print(f"  Numba:     max|z| = {np.max(np.abs(numba_z_clean)):.2f}")
            
            # Проверяем сколько раз |z| > z_enter
            z_enter = 2.0  # Из параметров
            ref_signals = np.sum(np.abs(ref_z_clean) > z_enter)
            numba_signals = np.sum(np.abs(numba_z_clean) > z_enter)
            print(f"  Сигналов |z| > {z_enter}:")
            print(f"    Reference: {ref_signals}")
            print(f"    Numba:     {numba_signals}")
    
    # 4. Сравнение PnL
    print(f"\n💰 PnL:")
    print(f"  Reference: {ref_results['total_pnl']:.2f}")
    print(f"  Numba:     {numba_results['total_pnl']:.2f}")
    print(f"  Sharpe Reference: {ref_results['sharpe']:.3f}")
    print(f"  Sharpe Numba:     {numba_results['sharpe']:.3f}")
    
    # 5. Детальная диагностика первых сделок
    if ref_results['num_trades'] > 0 or numba_results['num_trades'] > 0:
        print(f"\n🔍 Первые 20 баров с позициями:")
        print(f"{'Bar':<5} {'Ref Pos':<8} {'Numba Pos':<10} {'Ref Z':<8} {'Numba Z':<8}")
        print("-" * 50)
        
        for i in range(min(200, len(ref_pos))):
            if ref_pos[i] != 0 or numba_pos[i] != 0 or i < 100:
                ref_z_val = ref_results['z_scores'][i] if ref_results['z_scores'] is not None else np.nan
                numba_z_val = numba_results['z_scores'][i] if numba_results['z_scores'] is not None else np.nan
                
                if not np.isnan(ref_z_val) or not np.isnan(numba_z_val):
                    print(f"{i:<5} {ref_pos[i]:<8.0f} {numba_pos[i]:<10.0f} "
                          f"{ref_z_val:<8.2f} {numba_z_val:<8.2f}")
    
    # Итоговая оценка
    print("\n" + "="*60)
    success = trade_match and (ref_results['num_trades'] > 0)
    print(f"РЕЗУЛЬТАТ: {'✅ ПАРИТЕТ ДОСТИГНУТ' if success else '❌ ПАРИТЕТ НЕ ДОСТИГНУТ'}")
    print("="*60 + "\n")
    
    return success


def test_engine_parity():
    """Основной тест паритета движков."""
    
    # Подготовка данных
    df = prepare_test_data()
    print(f"\n📊 Подготовлены данные: {len(df)} баров")
    
    # Единые параметры для обоих движков
    params = {
        'rolling_window': 60,
        'z_enter': 2.0,
        'z_exit': 0.5,
        'max_holding_period': 100,
        'commission_pct': 0.0004,
        'slippage_pct': 0.0005
    }
    
    print(f"\n⚙️ Параметры:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    # Запуск движков
    print(f"\n🚀 Запуск Reference Engine...")
    ref_results = run_reference_engine(df, params)
    
    print(f"🚀 Запуск Numba Engine...")
    numba_results = run_numba_engine(df, params)
    
    # Сравнение
    parity_achieved = compare_engines(ref_results, numba_results)
    
    # Assert для pytest
    assert parity_achieved, "Движки не достигли паритета!"
    

if __name__ == "__main__":
    # Запуск теста напрямую
    test_engine_parity()
