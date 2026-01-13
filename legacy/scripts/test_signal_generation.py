#!/usr/bin/env python3
"""
Тестирование генерации торговых сигналов с разными параметрами
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from coint2.utils.config import load_config
from coint2.engine.base_engine import BasePairBacktester
from coint2.core.data_loader import DataHandler

def test_signal_generation():
    print("🔍 ТЕСТИРОВАНИЕ ГЕНЕРАЦИИ СИГНАЛОВ")
    print("=" * 50)
    
    # Загружаем конфигурацию
    config = load_config('configs/main_2024.yaml')
    
    # Загружаем данные
    data_loader = DataHandler(config)
    
    # Читаем предварительно отобранные пары
    pairs_df = pd.read_csv('outputs/preselected_pairs.csv')
    if pairs_df.empty:
        print("❌ Нет предварительно отобранных пар")
        return
    
    # Берем первую пару для тестирования
    test_pair = pairs_df.iloc[0]
    symbol1 = test_pair['s1']
    symbol2 = test_pair['s2']
    pair_name = f"{symbol1}/{symbol2}"
    print(f"📊 Тестируем пару: {pair_name}")
    
    # Загружаем данные для пары
    
    # Загружаем данные пары
    pair_data = data_loader.load_pair_data(
        symbol1=symbol1,
        symbol2=symbol2,
        start_date=config.walk_forward.start_date,
        end_date=config.walk_forward.end_date
    )
    
    if pair_data is None or pair_data.empty:
        print(f"❌ Не удалось загрузить данные для {pair_name}")
        return
    
    print(f"📈 Данные загружены: {len(pair_data)} записей для пары {pair_name}")
    
    # Разделяем данные на отдельные серии
    data1 = pair_data[pair_data['symbol'] == symbol1].set_index('timestamp')['close']
    data2 = pair_data[pair_data['symbol'] == symbol2].set_index('timestamp')['close']
    
    # Тестируем с разными параметрами z_threshold
    test_params = [0.5, 1.0, 1.5, 2.0, 2.5]
    
    for z_thresh in test_params:
        print(f"\n🎯 Тестируем z_threshold = {z_thresh}")
        
        # Создаем бэктестер
        backtester = BasePairBacktester(
            symbol1=symbol1,
            symbol2=symbol2,
            z_threshold=z_thresh,
            z_exit=0.1,
            lookback_window=60,
            initial_capital=10000,
            cost_per_trade=0.0,
            max_position_size=0.5
        )
        
        # Запускаем бэктест
        try:
            results = backtester.run_backtest(
                data1=data1,
                data2=data2,
                start_date=config.walk_forward.start_date,
                end_date=config.walk_forward.end_date
            )
            
            num_trades = len(results['trades'])
            total_pnl = results['final_capital'] - results['initial_capital']
            
            print(f"   📊 Сделок: {num_trades}")
            print(f"   💰 P&L: ${total_pnl:.2f}")
            
            if num_trades > 0:
                print(f"   ✅ Сигналы генерируются!")
                # Показываем первые несколько сделок
                trades_df = pd.DataFrame(results['trades'])
                print(f"   📋 Первые 3 сделки:")
                for i, trade in trades_df.head(3).iterrows():
                    print(f"      {trade['entry_date']}: {trade['side']} -> P&L: ${trade['pnl']:.2f}")
            else:
                print(f"   ❌ Нет сделок")
                
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 РЕКОМЕНДАЦИИ:")
    print("1. Используйте z_threshold >= 1.0 для получения сделок")
    print("2. Текущий z_threshold = 0.5 слишком строгий")
    print("3. Рассмотрите увеличение z_threshold до 1.5-2.0")

if __name__ == "__main__":
    test_signal_generation()