#!/usr/bin/env python3
"""
Пример использования новой функциональности определения рыночных режимов
и защиты от структурных сдвигов в системе парного трейдинга.
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from pathlib import Path

from coint2.utils.config import load_config
from coint2.engine.backtest_engine import PairBacktester

def create_sample_data():
    """Создает тестовые данные для демонстрации."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='D')
    
    # Создаем коинтегрированные временные ряды
    price1 = 100 + np.cumsum(np.random.normal(0, 1, 1000))
    
    # Второй актив коинтегрирован с первым + шум
    price2 = 0.8 * price1 + 20 + np.cumsum(np.random.normal(0, 0.5, 1000))
    
    # Добавляем структурный сдвиг в середине периода
    price2[500:] += 15  # Структурный сдвиг
    
    data = pd.DataFrame({
        'Y': price1,  # Первый актив (зависимая переменная)
        'X': price2   # Второй актив (независимая переменная)
    }, index=dates)  # Устанавливаем даты как индекс
    
    return data

def main():
    """Демонстрация новой функциональности."""
    print("🚀 Демонстрация определения рыночных режимов и защиты от структурных сдвигов")
    print("=" * 80)
    
    # Загружаем конфигурацию
    config = load_config('configs/main_2024.yaml')
    print(f"✅ Конфигурация загружена")
    print(f"   - Определение рыночных режимов: {config.backtest.market_regime_detection}")
    print(f"   - Защита от структурных сдвигов: {config.backtest.structural_break_protection}")
    print(f"   - Окно для Hurst Exponent: {config.backtest.hurst_window}")
    print(f"   - Окно для корреляции: {config.backtest.correlation_window}")
    print()
    
    # Создаем тестовые данные
    data = create_sample_data()
    print(f"📊 Создано {len(data)} точек данных")
    print(f"   - Период: {data.index.min()} - {data.index.max()}")
    print()
    
    # Инициализируем бэктестер
    backtester = PairBacktester(
        pair_data=data,
        rolling_window=config.backtest.rolling_window,
        z_threshold=config.backtest.zscore_threshold,
        z_exit=config.backtest.zscore_exit or 0.0,
        commission_pct=config.backtest.commission_pct,
        slippage_pct=config.backtest.slippage_pct,
        # Новые параметры для определения рыночных режимов
        market_regime_detection=config.backtest.market_regime_detection,
        structural_break_protection=config.backtest.structural_break_protection,
        hurst_window=config.backtest.hurst_window,
        correlation_window=config.backtest.correlation_window,
        min_correlation_threshold=config.backtest.min_correlation_threshold,
        max_half_life_days=config.backtest.max_half_life_days
    )
    
    print("🔄 Запуск бэктеста с новой функциональностью...")
    backtester.run()
    results = backtester.results
    
    print(f"✅ Бэктест завершен! Обработано {len(results)} периодов")
    print()
    
    # Анализ результатов
    print("📈 Анализ рыночных режимов:")
    regime_counts = results['market_regime'].value_counts()
    for regime, count in regime_counts.items():
        percentage = (count / len(results)) * 100
        print(f"   - {regime}: {count} периодов ({percentage:.1f}%)")
    print()
    
    print("🚨 Анализ структурных сдвигов:")
    structural_breaks = results['structural_break_detected'].sum()
    print(f"   - Обнаружено структурных сдвигов: {structural_breaks}")
    if structural_breaks > 0:
        break_indices = results[results['structural_break_detected']].index
        print(f"   - Индексы сдвигов: {break_indices.tolist()[:5]}...")
    print()
    
    print("📊 Статистика индикаторов:")
    hurst_mean = results['hurst_exponent'].mean()
    variance_ratio_mean = results['variance_ratio'].mean()
    correlation_mean = results['rolling_correlation'].mean()
    
    print(f"   - Средний Hurst Exponent: {hurst_mean:.3f}")
    print(f"   - Средний Variance Ratio: {variance_ratio_mean:.3f}")
    print(f"   - Средняя корреляция: {correlation_mean:.3f}")
    print()
    
    print("💰 Торговые результаты:")
    print(f"   - Доступные колонки: {list(results.columns)}")
    
    total_pnl = results['pnl'].sum() if 'pnl' in results.columns else 0
    cumulative_pnl = results['cumulative_pnl'].iloc[-1] if 'cumulative_pnl' in results.columns else 0
    winning_periods = len(results[results['pnl'] > 0]) if 'pnl' in results.columns else 0
    
    print(f"   - Общий PnL: ${total_pnl:.2f}")
    print(f"   - Кумулятивный PnL: ${cumulative_pnl:.2f}")
    print(f"   - Прибыльных периодов: {winning_periods}/{len(results)} ({winning_periods/len(results)*100:.1f}%)")
    print()
    
    print("💡 Влияние новой функциональности:")
    # Подсчитываем, сколько раз торговля была ограничена из-за режимов
    trend_periods = len(results[results['market_regime'] == 'trending'])
    break_periods = len(results[results['structural_break_detected']])
    
    print(f"   - Периодов с трендовым режимом (ограниченная торговля): {trend_periods}")
    print(f"   - Периодов со структурными сдвигами: {break_periods}")
    print(f"   - Общее влияние на торговлю: {(trend_periods + break_periods)/len(results)*100:.1f}% периодов")
    
    print("\n🎯 Демонстрация завершена успешно!")

if __name__ == "__main__":
    main()