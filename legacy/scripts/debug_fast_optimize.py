#!/usr/bin/env python3
"""
Диагностическая версия быстрой оптимизации для отладки ошибок.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Добавляем src в путь для импорта
sys.path.append(str(Path(__file__).parent.parent / "src"))

from coint2.utils.config import load_config
from coint2.core.data_loader import load_master_dataset
from coint2.engine.numba_engine import NumbaPairBacktester as PairBacktester

def debug_single_pair():
    """Отладка одной пары для понимания проблемы."""
    
    print("🔍 Диагностика быстрой оптимизации...")
    
    # Загружаем конфигурацию
    cfg = load_config("configs/main_2024.yaml")
    
    # Загружаем предотобранные пары
    pairs_df = pd.read_csv("outputs/preselected_pairs.csv")
    print(f"✅ Загружено {len(pairs_df)} пар")
    
    # Загружаем данные
    start_date = pd.to_datetime(cfg.walk_forward.start_date) - pd.Timedelta(days=cfg.walk_forward.training_period_days)
    end_date = pd.to_datetime(cfg.walk_forward.end_date)
    
    print(f"📈 Загрузка данных: {start_date.date()} -> {end_date.date()}")
    
    raw_data = load_master_dataset(
        data_path=cfg.data_dir,
        start_date=start_date,
        end_date=end_date
    )
    
    # Преобразуем в формат для бэктестинга
    price_data = raw_data.pivot(
        index='timestamp', 
        columns='symbol', 
        values='close'
    ).ffill().dropna()
    
    print(f"✅ Данные загружены: {price_data.shape}")
    
    # Берем первую пару для тестирования
    test_pair = pairs_df.iloc[0]
    s1, s2 = test_pair['s1'], test_pair['s2']
    
    print(f"🧪 Тестируем пару: {s1}-{s2}")
    
    # Проверяем наличие данных
    if s1 not in price_data.columns:
        print(f"❌ {s1} не найден в данных")
        return
    if s2 not in price_data.columns:
        print(f"❌ {s2} не найден в данных")
        return
    
    # Подготавливаем данные пары
    pair_data = price_data[[s1, s2]].dropna()
    print(f"📊 Данные пары: {pair_data.shape}")
    
    if len(pair_data) < cfg.backtest.rolling_window + 50:
        print(f"❌ Недостаточно данных: {len(pair_data)} < {cfg.backtest.rolling_window + 50}")
        return
    
    # Переименовываем колонки
    pair_data_renamed = pair_data.copy()
    pair_data_renamed.columns = ['y', 'x']
    
    print(f"✅ Данные подготовлены: {pair_data_renamed.columns.tolist()}")
    print(f"📈 Первые 5 строк:")
    print(pair_data_renamed.head())
    
    try:
        # Создаем бэктестер ТОЧНО как в быстрой оптимизации
        print("🔧 Создание бэктестера...")

        # Проверяем все параметры
        print(f"📊 Параметры бэктестера:")
        print(f"   pair_data shape: {pair_data_renamed.shape}")
        print(f"   rolling_window: {cfg.backtest.rolling_window}")
        print(f"   z_threshold: {cfg.backtest.zscore_threshold}")
        print(f"   z_exit: {cfg.backtest.zscore_exit}")
        print(f"   stop_loss_multiplier: {cfg.backtest.stop_loss_multiplier}")
        print(f"   commission_pct: {getattr(cfg.backtest, 'commission_pct', 0.0004)}")
        print(f"   slippage_pct: {getattr(cfg.backtest, 'slippage_pct', 0.0005)}")
        print(f"   capital_at_risk: {cfg.portfolio.initial_capital * cfg.portfolio.risk_per_position_pct}")
        print(f"   pair_name: {s1}-{s2}")
        print(f"   annualizing_factor: {getattr(cfg.backtest, 'annualizing_factor', 365)}")

        # Тестируем с более мягкими параметрами
        test_z_threshold = 1.5  # Вместо 2.2
        test_z_exit = 0.0       # Вместо отрицательного

        print(f"🧪 ТЕСТИРУЕМ с мягкими параметрами:")
        print(f"   z_threshold: {test_z_threshold} (было {cfg.backtest.zscore_threshold})")
        print(f"   z_exit: {test_z_exit} (было {cfg.backtest.zscore_exit})")

        backtester = PairBacktester(
            pair_data=pair_data_renamed,
            rolling_window=cfg.backtest.rolling_window,
            z_threshold=test_z_threshold,  # Мягче
            z_exit=test_z_exit,            # Мягче
            stop_loss_multiplier=cfg.backtest.stop_loss_multiplier,
            commission_pct=getattr(cfg.backtest, 'commission_pct', 0.0004),
            slippage_pct=getattr(cfg.backtest, 'slippage_pct', 0.0005),
            capital_at_risk=cfg.portfolio.initial_capital * cfg.portfolio.risk_per_position_pct,
            pair_name=f"{s1}-{s2}",
            annualizing_factor=getattr(cfg.backtest, 'annualizing_factor', 365)
        )
        
        print("✅ Бэктестер создан")
        
        # Запускаем бэктест
        print("🚀 Запуск бэктеста...")
        backtester.run()
        print("✅ Бэктест завершен")
        
        # Получаем результаты
        print("📊 Получение результатов...")
        results = backtester.get_results()
        print("✅ Результаты получены")

        print(f"🔍 Тип результатов: {type(results)}")

        if isinstance(results, dict):
            print(f"📋 Ключи результатов: {list(results.keys())}")

            if 'pnl' in results:
                pnl_series = results['pnl']
                print(f"💰 PnL тип: {type(pnl_series)}")
                print(f"💰 PnL длина: {len(pnl_series) if hasattr(pnl_series, '__len__') else 'N/A'}")
                print(f"💰 PnL сумма: {pnl_series.sum() if hasattr(pnl_series, 'sum') else 'N/A'}")

                # Проверяем, есть ли ненулевые значения
                if hasattr(pnl_series, 'sum') and pnl_series.sum() != 0:
                    print(f"✅ PnL ненулевой! Первые 5 значений: {pnl_series.head().tolist()}")
                else:
                    print(f"⚠️  PnL равен нулю")

            if 'trades_log' in results:
                trades_log = results['trades_log']
                print(f"📝 Trades log тип: {type(trades_log)}")
                print(f"📝 Trades log длина: {len(trades_log) if hasattr(trades_log, '__len__') else 'N/A'}")

                if hasattr(trades_log, '__len__') and len(trades_log) > 0:
                    print(f"✅ Есть сделки! Первая сделка: {trades_log[0] if len(trades_log) > 0 else 'N/A'}")
                else:
                    print(f"⚠️  Нет сделок")
        else:
            print(f"❌ Результаты не являются словарем: {results}")

        # Тестируем доступ к элементам, как в быстрой оптимизации
        print("\n🧪 ТЕСТИРУЕМ ДОСТУП К РЕЗУЛЬТАТАМ:")
        try:
            if results and 'pnl' in results:
                pnl_series = results['pnl']
                print(f"✅ Доступ к PnL успешен")

                if not pnl_series.empty:
                    print(f"✅ PnL не пустой")

                    # Тестируем добавление в список (как в быстрой оптимизации)
                    all_pnls = []
                    all_pnls.append(pnl_series)
                    print(f"✅ Добавление в список успешно")

                    # Тестируем конкатенацию (как в быстрой оптимизации)
                    combined_pnl = pd.concat(all_pnls, axis=1).sum(axis=1).fillna(0)
                    print(f"✅ Конкатенация успешна: {combined_pnl.sum()}")
                else:
                    print(f"⚠️  PnL пустой")

                # Тестируем доступ к trades_log
                trades_log = results.get('trades_log', [])
                print(f"✅ Доступ к trades_log успешен: {len(trades_log)} сделок")

        except Exception as e:
            print(f"❌ ОШИБКА при доступе к результатам: {e}")
            import traceback
            traceback.print_exc()

        print("🎉 Диагностика завершена успешно!")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_single_pair()
