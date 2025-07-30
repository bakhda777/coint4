#!/usr/bin/env python3
"""
Отладочный скрипт для критических тестов.
Проверяет генерацию сделок и расчет PnL.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Добавляем корневую директорию проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.coint2.engine.base_engine import BasePairBacktester


def create_test_data():
    """Создает тестовые данные с сильными сигналами."""
    np.random.seed(42)
    n = 200
    
    # Создаем коинтегрированные данные с сильными сигналами
    x = np.cumsum(np.random.randn(n) * 0.02) + 100
    true_beta = 1.5
    
    # Добавляем периодические отклонения для генерации сигналов
    y = np.zeros(n)
    for i in range(n):
        base_y = true_beta * x[i] + 50
        # Каждые 30 баров создаем сильное отклонение
        if i % 30 == 15:
            y[i] = base_y + 5.0  # Сильное отклонение вверх
        elif i % 30 == 25:
            y[i] = base_y - 3.0  # Отклонение вниз
        else:
            y[i] = base_y + np.random.randn() * 0.2
    
    dates = pd.date_range('2024-01-01', periods=n, freq='15min')
    return pd.DataFrame({'price_a': y, 'price_b': x}, index=dates)


def debug_backtest():
    """Отладка бэктеста."""
    print("🔍 ОТЛАДКА КРИТИЧЕСКИХ ТЕСТОВ")
    print("=" * 50)
    
    # Создаем данные
    data = create_test_data()
    print(f"📊 Создано {len(data)} баров данных")
    print(f"   Y: {data['price_a'].min():.2f} - {data['price_a'].max():.2f}")
    print(f"   X: {data['price_b'].min():.2f} - {data['price_b'].max():.2f}")
    
    # Создаем движок
    engine = BasePairBacktester(
        pair_data=data,
        z_threshold=0.5,  # Низкий порог для генерации сделок
        z_exit=0.0,       # Простой выход
        rolling_window=20, # Меньшее окно для быстрых сигналов
        capital_at_risk=10000,
        commission_pct=0.0001,  # Низкие издержки
        slippage_pct=0.0001,
        bid_ask_spread_pct_s1=0.0001,
        bid_ask_spread_pct_s2=0.0001,
        stop_loss_multiplier=10.0,  # Высокий стоп-лосс
        time_stop_multiplier=20.0   # Высокий тайм-стоп
    )
    
    print(f"🔧 Параметры движка:")
    print(f"   z_threshold: {engine.zscore_entry_threshold}")
    print(f"   rolling_window: {engine.rolling_window}")
    print(f"   capital_at_risk: {engine.capital_at_risk}")
    
    # Запускаем бэктест
    print("\n⚡ Запуск бэктеста...")
    engine.run()
    
    # Анализируем результаты
    df = engine.results
    print(f"📈 Результаты бэктеста:")
    print(f"   Всего баров: {len(df)}")
    
    # Проверяем позиции
    positions = df[df['position'] != 0]
    print(f"   Баров с позициями: {len(positions)}")
    
    if len(positions) > 0:
        print(f"   Первая позиция: {positions.iloc[0]['position']:.4f}")
        print(f"   Последняя позиция: {positions.iloc[-1]['position']:.4f}")
        
        # Проверяем entry_beta
        print(f"   Entry beta в движке: {engine.entry_beta}")
        
        # Проверяем realized_pnl
        realized_pnl_data = df[df['realized_pnl'] != 0]
        print(f"   Баров с realized_pnl: {len(realized_pnl_data)}")
        
        if len(realized_pnl_data) > 0:
            print(f"   Realized PnL: {realized_pnl_data['realized_pnl'].sum():.4f}")
        
        # Проверяем trades
        trades_data = df[df['trades'] > 0]
        print(f"   Баров с trades: {len(trades_data)}")
        
        if len(trades_data) > 0:
            print(f"   Общее количество сделок: {trades_data['trades'].sum():.0f}")
        
        # Показываем первые несколько строк с позициями
        print("\n📋 Первые строки с позициями:")
        print(positions[['position', 'pnl', 'unrealized_pnl', 'realized_pnl', 'trades', 'costs']].head())
        
    else:
        print("❌ Позиции не генерируются!")
        
        # Проверяем z-scores
        if 'z_score' in df.columns:
            z_scores = df['z_score'].dropna()
            print(f"   Z-scores: min={z_scores.min():.2f}, max={z_scores.max():.2f}")
            print(f"   Превышения порога: {(abs(z_scores) > 0.5).sum()}")
        
        # Проверяем beta
        if 'beta' in df.columns:
            betas = df['beta'].dropna()
            print(f"   Beta: min={betas.min():.2f}, max={betas.max():.2f}")
    
    # Проверяем общие метрики
    total_pnl = df['pnl'].sum()
    print(f"\n💰 Общий PnL: {total_pnl:.4f}")
    
    return engine, df


if __name__ == "__main__":
    engine, df = debug_backtest()
