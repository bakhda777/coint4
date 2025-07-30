#!/usr/bin/env python3
"""
Валидация оптимизированных параметров на полном walk-forward тесте.
"""

import sys
from pathlib import Path

# Добавляем src в путь для импорта
sys.path.append(str(Path(__file__).parent.parent / "src"))

from coint2.pipeline.walk_forward_orchestrator import run_walk_forward
from coint2.utils.config import load_config

def main():
    print('🚀 Запуск полной валидации с оптимизированными параметрами...')
    
    try:
        cfg = load_config('configs/optimized_manual.yaml')
        print(f'📊 Z-score порог: {cfg.backtest.zscore_threshold}')
        print(f'📉 Z-score выход: {cfg.backtest.zscore_exit}')
        
        result = run_walk_forward(cfg)
        
        if result:
            print('\n🎯 РЕЗУЛЬТАТЫ ПОЛНОЙ ВАЛИДАЦИИ:')
            print(f'📊 Общее количество сделок: {result.get("total_trades", 0)}')
            print(f'💰 Общий P&L: ${result.get("total_pnl", 0):.2f}')
            print(f'📈 Sharpe Ratio: {result.get("sharpe_ratio_abs", 0):.4f}')
            print(f'💸 Общие издержки: ${result.get("total_costs", 0):.2f}')
            
            # Сравнение с предыдущими результатами
            print('\n📊 СРАВНЕНИЕ С ПРЕДЫДУЩИМИ РЕЗУЛЬТАТАМИ:')
            print('Предыдущие результаты (до оптимизации):')
            print('  • Сделок: 89')
            print('  • P&L: $45.47')
            print('  • Sharpe: 0.7460')
            print('  • Издержки: $0 (ошибка)')
            
            current_trades = result.get("total_trades", 0)
            current_pnl = result.get("total_pnl", 0)
            current_sharpe = result.get("sharpe_ratio_abs", 0)
            
            print(f'\nТекущие результаты (после простой оптимизации):')
            print(f'  • Сделок: {current_trades}')
            print(f'  • P&L: ${current_pnl:.2f}')
            print(f'  • Sharpe: {current_sharpe:.4f}')

            print(f'\n📊 Текущие параметры:')
            print(f'  • z_entry: {cfg.backtest.zscore_threshold} (было 2.2)')
            print(f'  • z_exit: {cfg.backtest.zscore_exit} (было 0.0)')

            if current_trades > 0:
                print('\n✅ Оптимизация успешна - есть торговая активность!')
                improvement = (current_pnl - 45.47) / 45.47 * 100 if current_pnl > 0 else -100
                print(f'📈 Улучшение P&L: {improvement:+.1f}%')
            else:
                print('\n⚠️  Нет торговой активности - параметры слишком строгие')
                
        else:
            print('❌ Не удалось получить результаты')
            
    except Exception as e:
        print(f'❌ Ошибка: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
