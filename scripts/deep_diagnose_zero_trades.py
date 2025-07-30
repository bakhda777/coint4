#!/usr/bin/env python3
"""
Глубокая диагностика проблемы отсутствия сделок.
"""

import sys
from pathlib import Path

# Добавляем корневую директорию проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.optimiser.fast_objective import FastWalkForwardObjective
from src.coint2.utils.config import load_config

def test_single_pair_manually():
    """Тестирует одну пару вручную для понимания проблемы."""
    
    print("🔍 РУЧНОЙ ТЕСТ ОДНОЙ ПАРЫ")
    print("=" * 60)
    
    try:
        # Загружаем предотобранные пары
        pairs_df = pd.read_csv("outputs/preselected_pairs.csv")
        print(f"✅ Загружено {len(pairs_df)} пар")
        
        # Берем первую пару
        first_pair = pairs_df.iloc[0]
        print(f"📊 Тестируем пару: {first_pair['s1']} / {first_pair['s2']}")
        print(f"   Beta: {first_pair.get('beta', 'N/A')}")
        print(f"   Half-life: {first_pair.get('half_life', 'N/A')}")
        
        # Создаем objective
        objective = FastWalkForwardObjective(
            base_config_path="configs/main_2024.yaml",
            search_space_path="configs/search_space_relaxed.yaml"
        )
        
        # Очень простые параметры
        simple_params = {
            'zscore_threshold': 1.0,
            'zscore_exit': 0.0,
            'stop_loss_multiplier': 5.0,
            'time_stop_multiplier': 10.0,
            'risk_per_position_pct': 0.02,
            'max_position_size_pct': 0.1,
            'max_active_positions': 1,  # Только одна позиция
            'commission_pct': 0.0001,
            'slippage_pct': 0.0001,
            'normalization_method': 'minmax',
            'min_history_ratio': 0.5,
            'trial_number': 999
        }
        
        print(f"📊 Простые параметры: {simple_params}")
        
        # Запускаем тест
        result = objective(simple_params)
        print(f"📊 Результат: {result}")
        
        return result
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None

def analyze_data_period():
    """Анализирует данные за тестовый период."""
    
    print("\n📅 АНАЛИЗ ДАННЫХ ЗА ПЕРИОД")
    print("=" * 60)
    
    try:
        from src.coint2.core.data_loader import load_master_dataset
        
        # Загружаем данные за тестовый период
        start_date = "2024-01-15"
        end_date = "2024-01-20"
        
        print(f"📊 Загрузка данных: {start_date} -> {end_date}")
        
        raw_data = load_master_dataset(
            data_path="data_downloaded",
            start_date=start_date,
            end_date=end_date
        )
        
        if raw_data.empty:
            print(f"❌ Данные за период пусты!")
            return False
        
        print(f"✅ Загружено {len(raw_data)} записей")
        
        # Анализируем данные
        symbols = raw_data['symbol'].unique()
        print(f"📊 Символов: {len(symbols)}")
        print(f"📊 Первые 10 символов: {symbols[:10]}")
        
        # Проверяем временной диапазон
        timestamps = pd.to_datetime(raw_data['timestamp'])
        print(f"📊 Временной диапазон: {timestamps.min()} -> {timestamps.max()}")
        
        # Проверяем полноту данных
        pivot_data = raw_data.pivot_table(index="timestamp", columns="symbol", values="close")
        print(f"📊 Размер pivot данных: {pivot_data.shape}")
        
        # Проверяем пропуски
        missing_pct = pivot_data.isnull().sum().sum() / (pivot_data.shape[0] * pivot_data.shape[1])
        print(f"📊 Процент пропусков: {missing_pct:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка анализа данных: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_signal_generation():
    """Тестирует генерацию сигналов на простых данных."""
    
    print("\n🎯 ТЕСТ ГЕНЕРАЦИИ СИГНАЛОВ")
    print("=" * 60)
    
    try:
        from src.coint2.engine.numba_engine import NumbaPairBacktester
        from src.coint2.core.portfolio import Portfolio
        
        # Создаем синтетические данные для тестирования
        dates = pd.date_range('2024-01-15', '2024-01-20', freq='15min')
        n_points = len(dates)
        
        # Создаем коинтегрированные данные
        np.random.seed(42)
        price1 = 100 + np.cumsum(np.random.randn(n_points) * 0.01)
        price2 = 50 + 0.5 * price1 + np.random.randn(n_points) * 0.1  # Коинтегрированы
        
        test_data = pd.DataFrame({
            'timestamp': dates,
            'price1': price1,
            'price2': price2
        }).set_index('timestamp')
        
        print(f"✅ Создали синтетические данные: {test_data.shape}")
        print(f"📊 Корреляция: {test_data['price1'].corr(test_data['price2']):.3f}")
        
        # Создаем портфель
        portfolio = Portfolio(initial_capital=10000, max_active_positions=1)
        
        # Создаем бэктестер с простыми параметрами
        backtester = NumbaPairBacktester(
            pair_data=test_data,
            rolling_window=30,
            z_threshold=1.0,  # Очень низкий порог
            z_exit=0.0,
            cooldown_periods=1,
            commission_pct=0.0001,
            slippage_pct=0.0001,
            stop_loss_multiplier=5.0,
            time_stop_multiplier=10.0,
            portfolio=portfolio,
            pair_name="TEST/PAIR"
        )
        
        # Запускаем бэктест
        results = backtester.run_backtest()
        
        print(f"📊 Результаты бэктеста:")
        if isinstance(results, dict):
            print(f"   Сделки: {results.get('total_trades', 'N/A')}")
            print(f"   PnL: {results.get('total_pnl', 'N/A')}")
            print(f"   Sharpe: {results.get('sharpe_ratio', 'N/A')}")
        else:
            print(f"   Тип результата: {type(results)}")
        
        return results
        
    except Exception as e:
        print(f"❌ Ошибка тестирования сигналов: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None

def check_config_issues():
    """Проверяет проблемы в конфигурации."""
    
    print("\n⚙️ ПРОВЕРКА КОНФИГУРАЦИИ")
    print("=" * 60)
    
    try:
        cfg = load_config("configs/main_2024.yaml")
        
        # Проверяем walk-forward настройки
        print(f"📊 Walk-forward настройки:")
        print(f"   Включен: {cfg.walk_forward.enabled}")
        print(f"   Период: {cfg.walk_forward.start_date} -> {cfg.walk_forward.end_date}")
        print(f"   Тренировка: {cfg.walk_forward.training_period_days} дней")
        print(f"   Тестирование: {cfg.walk_forward.testing_period_days} дней")
        
        # Проверяем пороги
        print(f"\n📊 Пороги сигналов:")
        print(f"   zscore_threshold: {cfg.backtest.zscore_threshold}")
        print(f"   zscore_exit: {cfg.backtest.zscore_exit}")
        
        # Проверяем портфель
        print(f"\n📊 Настройки портфеля:")
        print(f"   Капитал: {cfg.portfolio.initial_capital}")
        print(f"   Риск на позицию: {cfg.portfolio.risk_per_position_pct}")
        print(f"   Макс размер позиции: {cfg.portfolio.max_position_size_pct}")
        print(f"   Макс позиций: {cfg.portfolio.max_active_positions}")
        
        # Проверяем потенциальные проблемы
        issues = []
        
        if cfg.walk_forward.enabled == False:
            issues.append("Walk-forward отключен - может использоваться простой бэктест")
        
        if cfg.backtest.zscore_threshold > 2.0:
            issues.append(f"Высокий порог входа: {cfg.backtest.zscore_threshold}")
        
        if cfg.portfolio.max_active_positions > 20:
            issues.append(f"Много позиций: {cfg.portfolio.max_active_positions}")
        
        if cfg.portfolio.risk_per_position_pct < 0.005:
            issues.append(f"Очень низкий риск: {cfg.portfolio.risk_per_position_pct}")
        
        if issues:
            print(f"\n⚠️ Потенциальные проблемы:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print(f"\n✅ Явных проблем в конфигурации не найдено")
        
        return len(issues) == 0
        
    except Exception as e:
        print(f"❌ Ошибка проверки конфигурации: {e}")
        return False

def create_minimal_test():
    """Создает минимальный тест для проверки работоспособности."""
    
    print("\n🧪 МИНИМАЛЬНЫЙ ТЕСТ")
    print("=" * 60)
    
    try:
        # Создаем минимальную конфигурацию
        minimal_config = {
            'data_dir': 'data_downloaded',
            'results_dir': 'results',
            'portfolio': {
                'initial_capital': 10000.0,
                'risk_per_position_pct': 0.02,
                'max_active_positions': 1,
                'max_position_size_pct': 1.0
            },
            'backtest': {
                'timeframe': '15min',
                'rolling_window': 30,
                'zscore_threshold': 1.0,
                'zscore_exit': 0.0,
                'stop_loss_multiplier': 5.0,
                'time_stop_multiplier': 10.0,
                'commission_pct': 0.0001,
                'slippage_pct': 0.0001,
                'cooldown_hours': 1
            },
            'walk_forward': {
                'enabled': True,
                'start_date': '2024-01-15',
                'end_date': '2024-01-20',
                'training_period_days': 10,
                'testing_period_days': 3,
                'step_size_days': 1
            }
        }
        
        # Сохраняем минимальную конфигурацию
        import yaml
        with open("configs/minimal_test.yaml", 'w') as f:
            yaml.dump(minimal_config, f, default_flow_style=False)
        
        print(f"✅ Создана минимальная конфигурация: configs/minimal_test.yaml")
        
        # Тестируем с минимальными параметрами
        minimal_params = {
            'zscore_threshold': 1.0,
            'zscore_exit': 0.0,
            'stop_loss_multiplier': 5.0,
            'time_stop_multiplier': 10.0,
            'risk_per_position_pct': 0.02,
            'max_position_size_pct': 1.0,
            'max_active_positions': 1,
            'commission_pct': 0.0001,
            'slippage_pct': 0.0001,
            'normalization_method': 'minmax',
            'min_history_ratio': 0.5,
            'trial_number': 999
        }
        
        print(f"📊 Минимальные параметры: {minimal_params}")
        
        return minimal_params
        
    except Exception as e:
        print(f"❌ Ошибка создания минимального теста: {e}")
        return None

if __name__ == "__main__":
    print("🚀 ГЛУБОКАЯ ДИАГНОСТИКА ПРОБЛЕМЫ '0 СДЕЛОК'")
    print("=" * 60)
    
    # Запускаем все тесты
    tests = [
        ("Анализ данных за период", analyze_data_period),
        ("Проверка конфигурации", check_config_issues),
        ("Тест генерации сигналов", test_signal_generation),
        ("Создание минимального теста", create_minimal_test),
        ("Ручной тест одной пары", test_single_pair_manually),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name.upper()}")
        print("-" * 40)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Ошибка в тесте: {e}")
            results.append((test_name, False))
    
    # Итоговый отчет
    print(f"\n📊 ИТОГОВЫЙ ОТЧЕТ ДИАГНОСТИКИ")
    print("=" * 60)
    for test_name, result in results:
        if isinstance(result, bool):
            status = "✅ ПРОШЕЛ" if result else "❌ ПРОВАЛЕН"
        elif result is None:
            status = "❌ ОШИБКА"
        else:
            status = f"📊 РЕЗУЛЬТАТ: {result}"
        print(f"{test_name}: {status}")
    
    print(f"\n💡 СЛЕДУЮЩИЕ ШАГИ:")
    print(f"1. Проанализируйте результаты тестов выше")
    print(f"2. Если данные корректны, проблема в логике стратегии")
    print(f"3. Если данные некорректны, проверьте источник данных")
    print(f"4. Рассмотрите использование configs/minimal_test.yaml для тестирования")
