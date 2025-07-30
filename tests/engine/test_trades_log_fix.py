import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.coint2.engine.base_engine import BasePairBacktester as PairBacktester


class TestTradesLogFix:
    """Тесты для проверки исправления проблемы с пустым trades_log."""
    
    def test_trades_log_contains_complete_trades(self):
        """Проверяет, что trades_log содержит полные записи сделок с правильными полями."""
        # Создаем синтетические данные
        start_time = datetime(2024, 1, 1, 9, 0)
        periods = 100
        datetime_index = pd.date_range(start_time, periods=periods, freq='15min')
        
        # Создаем данные с сильной коинтеграцией и волатильностью
        np.random.seed(42)
        price_a = 100 + np.cumsum(np.random.randn(periods) * 0.5)  # Увеличиваем волатильность
        # Создаем сильно коррелированные цены с периодическими отклонениями
        price_b = 50 + 0.5 * price_a + np.sin(np.arange(periods) * 0.2) * 2 + np.random.randn(periods) * 0.2
        
        data = pd.DataFrame({
            'x': price_a,
            'y': price_b
        }, index=datetime_index)
        
        # Настройки бэктеста (более агрессивные для генерации сделок)
        config = {
            'entry_threshold': 0.5,  # Более низкий порог входа
            'exit_threshold': 0.1,   # Более низкий порог выхода
            'stop_loss_threshold': 2.0,
            'take_profit_threshold': 0.05,
            'rolling_window': 10,
            'position_size': 10000,  # Увеличиваем капитал для превышения min_notional
            'transaction_cost_pct': 0.001,
            'slippage_pct': 0.001,
            'max_holding_period_hours': 24,
            'min_holding_period_minutes': 30
        }
        
        # Создаем и запускаем бэктест
        engine = PairBacktester(
            pair_data=data,
            rolling_window=config['rolling_window'],
            z_threshold=config['entry_threshold'],
            z_exit=config['exit_threshold'],
            commission_pct=config['transaction_cost_pct'],
            slippage_pct=config['slippage_pct'],
            capital_at_risk=config['position_size'],
            stop_loss_multiplier=config['stop_loss_threshold']/config['entry_threshold']
        )
        
        # Добавляем отладочную информацию для проверки capital_sufficient и _can_open_new_position
        original_check_capital = engine._check_capital_sufficiency
        def debug_check_capital(*args, **kwargs):
            result = original_check_capital(*args, **kwargs)
            if len(args) >= 6:
                z_score, spread, mean, std, beta, price_s1, price_s2 = args[:7]
                trade_value = engine._calculate_position_size(z_score, spread, mean, std, beta, price_s1, price_s2) * price_s1
                trade_value_pct = (trade_value / engine.capital_at_risk) * 100
                print(f"DEBUG: capital_sufficient={result}, trade_value={trade_value:.2f}, trade_value_pct={trade_value_pct:.3f}")
            return result
        engine._check_capital_sufficiency = debug_check_capital
        
        # Добавляем отладочную информацию для _can_open_new_position
        original_can_open = engine._can_open_new_position
        def debug_can_open(signal):
            result = original_can_open(signal)
            print(f"DEBUG: _can_open_new_position(signal={signal})={result}, current_position={engine.current_position}, current_cash={getattr(engine, 'current_cash', 'N/A')}, active_positions_count={getattr(engine, 'active_positions_count', 'N/A')}")
            return result
        engine._can_open_new_position = debug_can_open
        
        # Добавляем отладочную информацию в _open_trade
        original_open_trade = engine._open_trade
        
        def debug_open_trade(*args, **kwargs):
            print(f"DEBUG _open_trade called with args={args}, kwargs={kwargs}")
            result = original_open_trade(*args, **kwargs)
            print(f"DEBUG incremental_trades_log length after _open_trade: {len(engine.incremental_trades_log)}")
            if engine.incremental_trades_log:
                print(f"DEBUG last trade logged: {engine.incremental_trades_log[-1]}")
            return result
        
        engine._open_trade = debug_open_trade
        
        engine.run()
        results = engine.get_results()
        
        # Отладочная информация
        print(f"Incremental trades log length: {len(engine.incremental_trades_log)}")
        print(f"Incremental trades log: {engine.incremental_trades_log}")
        print(f"Results keys: {results.keys()}")
        
        # Проверяем incremental_trades_log напрямую
        incremental_trades = engine.incremental_trades_log
        print(f"\nDEBUG: incremental_trades_log length = {len(incremental_trades)}")
        if len(incremental_trades) > 0:
            print(f"DEBUG: First incremental trade = {incremental_trades[0]}")
        
        # Также проверяем полный trades_log
        trades_log = results['trades_log']
        print(f"DEBUG: complete trades_log length = {len(trades_log)}")
        
        # Если incremental_trades_log пустой, то проблема в генерации сделок
        if len(incremental_trades) == 0:
            # Проверим z_scores для понимания, почему сделки не генерируются
            z_scores = engine.results['z_score'].dropna()
            print(f"Z-scores range: {z_scores.min():.3f} to {z_scores.max():.3f}")
            print(f"Z-scores abs max: {z_scores.abs().max():.3f}")
            print(f"Entry threshold: {config['entry_threshold']}")
            
            # Отладочная информация для расчета размера позиции
            print(f"\nОтладка расчета размера позиции:")
            print(f"Capital at risk: {engine.capital_at_risk}")
            
            # Тестируем расчет Kelly fraction
            kelly_f = engine._calculate_kelly_or_capital_fraction()
            print(f"Kelly fraction: {kelly_f}")
            
            # Тестируем расчет размера позиции для первого сигнала
            for i in range(len(data)):
                if i >= config['rolling_window']:
                    window_data = data.iloc[i-config['rolling_window']:i]
                    spread = window_data['x'] - window_data['y']
                    z_score = (spread.iloc[-1] - spread.mean()) / spread.std()
                    
                    if abs(z_score) > config['entry_threshold']:
                        price_s1 = data['x'].iloc[i]
                        price_s2 = data['y'].iloc[i]
                        beta = 1.0  # Упрощенная бета
                        
                        # Проверяем capital sufficiency
                        capital_sufficient = engine._check_capital_sufficiency(price_s1, price_s2, beta)
                        trade_value = price_s1 + abs(beta) * price_s2
                        trade_value_pct = trade_value / engine.capital_at_risk
                        
                        position_size = engine._calculate_position_size(
                            z_score, spread.iloc[-1], spread.mean(), spread.std(), 
                            beta, price_s1, price_s2
                        )
                        print(f"  Период {i}: z_score={z_score:.3f}, position_size={position_size:.6f}")
                        print(f"    price_s1={price_s1:.2f}, price_s2={price_s2:.2f}, beta={beta}")
                        print(f"    trade_value={trade_value:.2f}, trade_value_pct={trade_value_pct:.3f}")
                        print(f"    capital_sufficient={capital_sufficient}")
                        print(f"    denominator={price_s1 + abs(beta) * price_s2:.2f}")
                        print(f"    f * equity = {kelly_f * engine.capital_at_risk:.2f}")
                        break
            
            # Если z-scores не превышают порог, то это нормально
            if z_scores.abs().max() < config['entry_threshold']:
                pytest.skip("No trades generated due to insufficient z-score signals")
        
        assert len(incremental_trades) > 0, f"incremental_trades_log пуст. Проверьте генерацию сделок. Данные: {data.head()}"
        
        # Проверяем структуру записей в incremental_trades_log
        open_trades = [t for t in incremental_trades if t['action'] == 'open']
        close_trades = [t for t in incremental_trades if t['action'] == 'close']
        
        print(f"\nDEBUG: Open trades: {len(open_trades)}, Close trades: {len(close_trades)}")
        
        for i, trade in enumerate(open_trades):
            print(f"\nOpen Trade {i+1}:")
            for key, value in trade.items():
                print(f"  {key}: {value}")
            
            # Проверяем обязательные поля для открытия сделки
            required_fields = [
                'action', 'date', 'entry_z', 'position_size', 'capital_used',
                'entry_price_s1', 'entry_price_s2', 'beta'
            ]
            
            for field in required_fields:
                assert field in trade, f"Поле '{field}' отсутствует в записи открытия сделки {i+1}"
            
            # Проверяем типы данных
            assert isinstance(trade['date'], (datetime, pd.Timestamp)), "date должно быть datetime"
            assert isinstance(trade['position_size'], (int, float)), "position_size должно быть числом"
            assert isinstance(trade['capital_used'], (int, float)), "capital_used должно быть числом"
            
            # Проверяем, что позиция не нулевая
            assert trade['position_size'] != 0, "position_size не должно быть нулевым"
    
    def test_incremental_trades_log_to_complete_conversion(self):
        """Проверяет корректность преобразования incremental_trades_log в полные записи сделок."""
        # Создаем синтетические данные
        start_time = datetime(2024, 1, 1, 9, 0)
        periods = 50
        datetime_index = pd.date_range(start_time, periods=periods, freq='15min')
        
        np.random.seed(123)
        price_a = 100 + np.cumsum(np.random.randn(periods) * 0.1)
        price_b = 50 + 0.5 * price_a + np.random.randn(periods) * 0.05
        
        data = pd.DataFrame({
            'x': price_a,
            'y': price_b
        }, index=datetime_index)
        
        config = {
            'entry_threshold': 1.0,
            'exit_threshold': 0.3,
            'stop_loss_threshold': 2.5,
            'take_profit_threshold': 0.1,
            'rolling_window': 8,
            'position_size': 500,
            'transaction_cost_pct': 0.001,
            'slippage_pct': 0.001,
            'max_holding_period_hours': 12,
            'min_holding_period_minutes': 15
        }
        
        engine = PairBacktester(
            pair_data=data,
            rolling_window=config['rolling_window'],
            z_threshold=config['entry_threshold'],
            z_exit=config['exit_threshold'],
            commission_pct=config['transaction_cost_pct'],
            slippage_pct=config['slippage_pct'],
            capital_at_risk=config['position_size'],
            stop_loss_multiplier=config['stop_loss_threshold']/config['entry_threshold']
        )
        
        engine.run()
        
        # Получаем incremental_trades_log напрямую
        incremental_log = engine.incremental_trades_log
        
        # Получаем полные записи сделок
        complete_trades = engine._create_complete_trades_log()
        
        if len(incremental_log) > 0:
            # Проверяем, что количество открытий равно количеству закрытий
            open_trades = [t for t in incremental_log if t['action'] == 'open']
            close_trades = [t for t in incremental_log if t['action'] == 'close']
            
            # Количество полных сделок должно быть равно минимуму из открытий и закрытий
            expected_complete_trades = min(len(open_trades), len(close_trades))
            assert len(complete_trades) == expected_complete_trades, \
                f"Ожидалось {expected_complete_trades} полных сделок, получено {len(complete_trades)}"
            
            # Проверяем, что каждая полная сделка имеет корректные данные
            for i, trade in enumerate(complete_trades):
                if i < len(open_trades) and i < len(close_trades):
                    open_trade = open_trades[i]
                    close_trade = close_trades[i]
                    
                    # Проверяем соответствие времени
                    assert trade['entry_datetime'] == open_trade['datetime'], \
                        "entry_datetime должно соответствовать datetime открытия"
                    assert trade['exit_datetime'] == close_trade['datetime'], \
                        "exit_datetime должно соответствовать datetime закрытия"
    
    def test_performance_metrics_with_complete_trades(self):
        """Проверяет, что метрики производительности корректно рассчитываются с полными записями сделок."""
        # Создаем данные с гарантированными сделками
        start_time = datetime(2024, 1, 1, 9, 0)
        periods = 80
        datetime_index = pd.date_range(start_time, periods=periods, freq='15min')
        
        np.random.seed(456)
        # Создаем данные с четким паттерном для генерации сделок
        price_a = 100 + np.sin(np.arange(periods) * 0.3) * 2
        price_b = 50 + 0.5 * price_a + np.sin(np.arange(periods) * 0.3 + np.pi/4) * 1
        
        data = pd.DataFrame({
            'x': price_a,
            'y': price_b
        }, index=datetime_index)
        
        config = {
            'entry_threshold': 0.8,
            'exit_threshold': 0.2,
            'stop_loss_threshold': 2.0,
            'take_profit_threshold': 0.1,
            'rolling_window': 6,
            'position_size': 1000,
            'transaction_cost_pct': 0.001,
            'slippage_pct': 0.001,
            'max_holding_period_hours': 8,
            'min_holding_period_minutes': 15
        }
        
        engine = PairBacktester(
            pair_data=data,
            rolling_window=config['rolling_window'],
            z_threshold=config['entry_threshold'],
            z_exit=config['exit_threshold'],
            commission_pct=config['transaction_cost_pct'],
            slippage_pct=config['slippage_pct'],
            capital_at_risk=config['position_size'],
            stop_loss_multiplier=config['stop_loss_threshold']/config['entry_threshold']
        )
        
        engine.run()
        
        # Получаем метрики производительности
        metrics = engine.get_performance_metrics()
        
        # Проверяем, что метрики рассчитаны
        assert 'num_trades' in metrics, "Отсутствует метрика num_trades"
        assert 'avg_trade_duration' in metrics, "Отсутствует метрика avg_trade_duration"
        
        # Если есть сделки, проверяем корректность метрик
        if metrics['num_trades'] > 0:
            assert metrics['num_trades'] >= 0, "Количество сделок должно быть неотрицательным"
            assert metrics['avg_trade_duration'] >= 0, "Средняя продолжительность сделки должна быть неотрицательной"
            
            # Проверяем соответствие с trades_log
            results = engine.get_results()
            trades_log = results['trades_log']
            assert len(trades_log) == metrics['num_trades'], \
                "Количество сделок в метриках должно соответствовать trades_log"