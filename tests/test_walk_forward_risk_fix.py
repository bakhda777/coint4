#!/usr/bin/env python3
"""
Тест для проверки исправления ошибки с неподдерживаемым аргументом 'risk_per_position_pct'
в walk_forward_orchestrator.py
"""

import sys
sys.path.append('/Users/admin/Desktop/coint4/src')

from src.coint2.utils.config import AppConfig
from src.coint2.pipeline.walk_forward_orchestrator import run_walk_forward
from datetime import datetime

def test_walk_forward_risk_fix():
    """Тест исправления ошибки с risk_per_position_pct"""
    
    print("🧪 Тестирование исправления ошибки с risk_per_position_pct...")
    
    # Создаем минимальную конфигурацию
    from src.coint2.utils.config import (
        PairSelectionConfig, PortfolioConfig, BacktestConfig, 
        WalkForwardConfig, DataProcessingConfig
    )
    from pathlib import Path
    
    try:
        cfg = AppConfig(
            data_dir=Path('/Users/admin/Desktop/coint4'),
            results_dir=Path('/tmp/test_results'),
            portfolio=PortfolioConfig(
                initial_capital=10000.0,
                risk_per_position_pct=0.02,
                max_active_positions=5
            ),
            pair_selection=PairSelectionConfig(
                lookback_days=1,
                coint_pvalue_threshold=0.05,
                ssd_top_n=3,
                min_half_life_days=1,
                max_half_life_days=100,
                min_mean_crossings=1
            ),
            backtest=BacktestConfig(
                 timeframe='15min',
                 rolling_window=96,
                 zscore_threshold=1.5,
                 zscore_exit=0.8,
                 commission_pct=0.001,
                 slippage_pct=0.0005,
                 stop_loss_multiplier=2.0,
                 fill_limit_pct=0.1,
                 annualizing_factor=365
             ),
            walk_forward=WalkForwardConfig(
                start_date='2024-01-01',
                end_date='2024-01-02',
                training_period_days=1,
                testing_period_days=1,
                step_size_days=1
            )
        )
        print(f"✅ Конфигурация создана успешно")
        
        # Запускаем walk-forward анализ
        print("🚀 Запуск walk-forward анализа...")
        metrics = run_walk_forward(cfg)
        
        print(f"✅ Walk-forward анализ завершен успешно!")
        print(f"📊 Метрики: {metrics}")
        
        print("\n🎉 УСПЕХ: Ошибка с 'risk_per_position_pct' исправлена!")
        print("✅ PairBacktester теперь вызывается с корректными аргументами")
        
        return True
        
    except Exception as e:
        print(f"❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_walk_forward_risk_fix()
    sys.exit(0 if success else 1)