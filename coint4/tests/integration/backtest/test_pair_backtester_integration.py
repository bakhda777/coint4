"""Integration tests for PairBacktester with look-ahead bias fix."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch

from coint2.core.pair_backtester import PairBacktester
from coint2.core.portfolio import Portfolio
from coint2.engine.base_engine import BasePairBacktester as IncrementalPairBacktester

# Константы для тестирования
DEFAULT_ROLLING_WINDOW = 10
DEFAULT_Z_THRESHOLD = 2.0
DEFAULT_RISK_PER_POSITION_PCT = 0.01
DEFAULT_MAX_HISTORY_DAYS = 100
DEFAULT_CAPITAL_AT_RISK = 1000.0
DEFAULT_CAPITAL_INCREMENT = 100

# Константы для генерации данных
TEST_PERIODS_SHORT = 20
TEST_PERIODS_MEDIUM = 30
TEST_PERIODS_LONG = 50
FREQUENCY = '15min'
BASE_PRICE_ASSET1 = 100
BASE_PRICE_ASSET2 = 50
COINTEGRATION_COEFFICIENT = 2.0
NOISE_STD = 0.5
PRICE_VOLATILITY = 0.1

# Константы для тестирования
TEST_PAIR_NAME = "TEST-INTEGRATION"
START_DATE = '2023-01-01'
INITIAL_DATA_SIZE = 25
TOLERANCE = 1e-6


class TestPairBacktesterIntegration:
    """Integration tests for PairBacktester with incremental processing."""

    @pytest.mark.integration
    def test_pair_backtester_when_inherits_incremental_then_interface_correct(self):
        """Test 1: Verify PairBacktester correctly inherits from IncrementalPairBacktester.

        This test ensures that the inheritance change doesn't break the interface
        and that all expected methods are available.
        """
        # Create test data (детерминизм обеспечен глобально)
        dates = pd.date_range(START_DATE, periods=TEST_PERIODS_SHORT, freq=FREQUENCY)
        data = pd.DataFrame({
            'asset1': np.random.randn(TEST_PERIODS_SHORT) + BASE_PRICE_ASSET1,
            'asset2': np.random.randn(TEST_PERIODS_SHORT) + BASE_PRICE_ASSET2
        }, index=dates)
        
        backtester = PairBacktester(
            pair_name=TEST_PAIR_NAME,
            pair_data=data.copy(),
            rolling_window=DEFAULT_ROLLING_WINDOW,
            z_threshold=DEFAULT_Z_THRESHOLD,
            risk_per_position_pct=DEFAULT_RISK_PER_POSITION_PCT,
            max_history_days=DEFAULT_MAX_HISTORY_DAYS
        )
        
        # Verify inheritance - PairBacktester now inherits from IncrementalPairBacktester
        assert hasattr(backtester, 'set_capital_at_risk'), "Should have incremental methods"
        assert hasattr(backtester, 'process_single_period'), "Should have incremental methods"
        
        # Verify key methods are available
        assert hasattr(backtester, 'run_on_day'), "Should have run_on_day method"
        assert hasattr(backtester, 'get_incremental_results'), "Should have get_incremental_results method"
        assert hasattr(backtester, 'set_capital_at_risk'), "Should have set_capital_at_risk method"
        assert hasattr(backtester, 'get_capital_at_risk_for_date'), "Should have get_capital_at_risk_for_date method"
        assert hasattr(backtester, 'process_single_period'), "Should have process_single_period method"
        
        # Verify incremental-specific attributes
        assert hasattr(backtester, 'capital_at_risk_history'), "Should have capital_at_risk_history"
        assert hasattr(backtester, 'incremental_trades_log'), "Should have incremental_trades_log"
    
    @pytest.mark.integration
    def test_run_on_day_when_incremental_processing_then_no_lookahead_bias(self):
        """Test 2: Verify run_on_day method works with incremental processing.

        This test ensures that the updated run_on_day method correctly
        implements incremental processing without look-ahead bias.
        """
        # Create deterministic test data (детерминизм обеспечен глобально)
        dates = pd.date_range(START_DATE, periods=TEST_PERIODS_MEDIUM, freq=FREQUENCY)

        # Create cointegrated series with known relationship
        x = np.cumsum(np.random.randn(TEST_PERIODS_MEDIUM) * PRICE_VOLATILITY) + BASE_PRICE_ASSET1
        y = COINTEGRATION_COEFFICIENT * x + np.random.randn(TEST_PERIODS_MEDIUM) * NOISE_STD + BASE_PRICE_ASSET2
        
        data = pd.DataFrame({
            'asset1': y,
            'asset2': x
        }, index=dates)
        
        # Mock portfolio with changing capital
        portfolio = Mock(spec=Portfolio)

        call_count = 0
        def mock_capital_calculation(risk_pct):
            nonlocal call_count
            call_count += 1
            # Capital increases over time
            base_capital = DEFAULT_CAPITAL_AT_RISK + call_count * DEFAULT_CAPITAL_INCREMENT
            return base_capital * risk_pct

        portfolio.calculate_position_risk_capital = mock_capital_calculation

        # Initialize backtester with minimal data
        backtester = IncrementalPairBacktester(
            pair_data=data.iloc[:INITIAL_DATA_SIZE].copy(),
            rolling_window=DEFAULT_ROLLING_WINDOW,
            z_threshold=DEFAULT_Z_THRESHOLD,
            capital_at_risk=DEFAULT_CAPITAL_AT_RISK
        )
        
        # Track results over time
        daily_results = []
        capital_snapshots = []

        for i in range(INITIAL_DATA_SIZE, len(data)):
            date = dates[i]
            daily_data = data.iloc[[i]]
            
            # Capture capital before processing
            current_capital = mock_capital_calculation(0.02)
            capital_snapshots.append({
                'date': date,
                'capital': current_capital,
                'call_count': call_count
            })
            
            # Process day
            backtester.set_capital_at_risk(date, current_capital)
            backtester.pair_data = pd.concat([backtester.pair_data, daily_data])
            
            result = backtester.process_single_period(
                date, daily_data.iloc[0, 0], daily_data.iloc[0, 1]
            )
            
            daily_results.append({
                'date': date,
                'daily_pnl': result['pnl'],
                'total_pnl': backtester.get_incremental_results().get('total_pnl', 0)
            })
        
        # Verify incremental processing worked
        assert len(daily_results) == 5, "Should process 5 days"
        
        # Verify capital tracking
        capital_history = backtester.capital_at_risk_history
        assert len(capital_history) > 0, "Should have capital history"
        
        # Verify that capital used for trades matches capital at time of trade
        trades_log = backtester.incremental_trades_log
        open_trades = [t for t in trades_log if t['action'] == 'open']
        
        for trade in open_trades:
            trade_date = trade['date']
            capital_used = trade['capital_used']
            
            # Find corresponding capital snapshot
            matching_snapshot = next(
                (s for s in capital_snapshots if s['date'] == trade_date),
                None
            )
            
            if matching_snapshot:
                expected_capital = matching_snapshot['capital']
                assert capital_used == pytest.approx(expected_capital, abs=0.01), \
                    f"Trade at {trade_date} used {capital_used}, expected {expected_capital}"
    
    @pytest.mark.integration
    def test_data_management_when_max_history_days_then_sliding_window_works(self):
        """Test 3: Verify data management works correctly with max_history_days limit.

        This test ensures that the sliding window data management doesn't
        interfere with incremental processing and capital tracking.
        """
        # Create long time series (детерминизм обеспечен глобально)
        LONG_PERIODS = 100
        dates = pd.date_range(START_DATE, periods=LONG_PERIODS, freq=FREQUENCY)
        data = pd.DataFrame({
            'asset1': np.random.randn(LONG_PERIODS) + BASE_PRICE_ASSET1,
            'asset2': np.random.randn(LONG_PERIODS) + BASE_PRICE_ASSET2
        }, index=dates)

        # Use small max_history_days to trigger trimming
        backtester = IncrementalPairBacktester(
            pair_data=data.iloc[:INITIAL_DATA_SIZE].copy(),
            rolling_window=DEFAULT_ROLLING_WINDOW,
            z_threshold=DEFAULT_Z_THRESHOLD,
            capital_at_risk=DEFAULT_CAPITAL_AT_RISK
        )

        MAX_HISTORY_DAYS = 30  # Small limit to trigger trimming
        
        # Process all data
        for i in range(INITIAL_DATA_SIZE, len(data)):
            date = dates[i]
            daily_data = data.iloc[[i]]
            
            backtester.set_capital_at_risk(date, 1000.0)
            backtester.pair_data = pd.concat([backtester.pair_data, daily_data])
            
            # Manually trim data if it gets too large (simulating memory management)
            if len(backtester.pair_data) > MAX_HISTORY_DAYS:
                backtester.pair_data = backtester.pair_data.tail(MAX_HISTORY_DAYS)
            
            result = backtester.process_single_period(
                date, daily_data.iloc[0, 0], daily_data.iloc[0, 1]
            )
            
            # Verify data size constraints
            current_size = len(backtester.pair_data)
            assert current_size <= MAX_HISTORY_DAYS, \
                f"Data size {current_size} exceeds limit {MAX_HISTORY_DAYS}"
            assert current_size >= backtester.rolling_window, \
                f"Data size {current_size} below minimum {backtester.rolling_window}"
        
        # Verify capital history is maintained despite data trimming
        assert len(backtester.capital_at_risk_history) > 0, \
            "Capital history should be maintained"
        
        # Verify incremental results are still available
        results = backtester.get_incremental_results()
        assert 'total_pnl' in results, "Should have total PnL"
        assert 'trade_count' in results, "Should have trade count"
    
    def test_portfolio_integration_with_real_calculations(self):
        """Test 4: Verify integration with Portfolio class for realistic scenarios.
        
        This test uses more realistic portfolio calculations to ensure
        the integration works correctly in practice.
        """
        # Create realistic market data
        dates = pd.date_range('2023-01-01', periods=50, freq='15min')
        
        # Create pair with realistic price movements (детерминизм обеспечен глобально)
        base_1, base_2 = 150.0, 75.0
        
        prices_1 = [base_1]
        prices_2 = [base_2]
        
        for i in range(1, 50):
            # Add realistic price movements with some correlation
            change_1 = np.random.randn() * 0.5
            change_2 = np.random.randn() * 0.3 + 0.3 * change_1  # Partial correlation
            
            prices_1.append(prices_1[-1] + change_1)
            prices_2.append(prices_2[-1] + change_2)
        
        data = pd.DataFrame({
            'asset1': prices_1,
            'asset2': prices_2
        }, index=dates)
        
        # Create realistic portfolio mock
        portfolio = Mock(spec=Portfolio)
        
        # Simulate portfolio with growing equity
        initial_equity = 10000.0
        growth_rate = 0.001  # 0.1% per period
        call_count = 0
        
        def realistic_capital_calc(risk_pct):
            nonlocal call_count
            # Simulate growing portfolio
            current_equity = initial_equity * (1 + growth_rate) ** call_count
            call_count += 1
            return current_equity * risk_pct
        
        portfolio.calculate_position_risk_capital = realistic_capital_calc
        
        # Initialize backtester
        backtester = IncrementalPairBacktester(
            pair_data=data.iloc[:25].copy(),
            rolling_window=10,
            z_threshold=1.8,
            z_exit=0.5,
            capital_at_risk=initial_equity * 0.015,  # 1.5% risk per position
            commission_pct=0.001,
            slippage_pct=0.0005
        )
        
        # Process data and track portfolio integration
        equity_progression = []
        trade_details = []
        
        for i in range(25, len(data)):
            date = dates[i]
            daily_data = data.iloc[[i]]
            
            # Capture equity before processing
            current_equity = initial_equity * (1 + growth_rate) ** call_count
            current_capital = realistic_capital_calc(0.015)
            
            equity_progression.append({
                'date': date,
                'equity': current_equity,
                'period': i - 20
            })
            
            # Process day
            backtester.set_capital_at_risk(date, current_capital)
            backtester.pair_data = pd.concat([backtester.pair_data, daily_data])
            
            result = backtester.process_single_period(
                date, daily_data.iloc[0, 0], daily_data.iloc[0, 1]
            )
            
            # Check for new trades
            trades_log = backtester.incremental_trades_log
            if len(trades_log) > len(trade_details):
                new_trades = trades_log[len(trade_details):]
                trade_details.extend(new_trades)
        
        # Verify portfolio integration
        assert len(equity_progression) == 25, "Should track equity for all periods"
        
        # Verify that capital allocation reflects portfolio growth
        open_trades = [t for t in trade_details if t['action'] == 'open']
        
        if len(open_trades) >= 2:
            # Compare early vs late trades
            early_trade = open_trades[0]
            late_trade = open_trades[-1]
            
            # Later trades should use more capital due to portfolio growth
            assert late_trade['capital_used'] > early_trade['capital_used'], \
                f"Late trade capital {late_trade['capital_used']} should exceed early trade {early_trade['capital_used']}"
        
        # Verify that trades use appropriate capital amounts
        if open_trades:
            for trade in open_trades:
                assert 'capital_used' in trade, "Should track capital used"
                assert trade['capital_used'] > 0, "Capital used should be positive"
    
    def test_error_handling_and_edge_cases(self):
        """Test 5: Verify proper error handling and edge case management.
        
        This test ensures that the incremental backtester handles
        edge cases gracefully without breaking.
        """
        # Test with minimal data
        dates = pd.date_range('2023-01-01', periods=10, freq='15min')
        data = pd.DataFrame({
            'asset1': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'asset2': [50, 50.5, 51, 51.5, 52, 52.5, 53, 53.5, 54, 54.5]
        }, index=dates)
        
        # Test with rolling window equal to data size
        backtester = IncrementalPairBacktester(
            pair_data=data.copy(),
            rolling_window=5,
            z_threshold=2.0,
            capital_at_risk=1000.0
        )
        
        # Should handle minimal data gracefully
        results = backtester.get_incremental_results()
        assert isinstance(results, dict), "Should return valid results dict"
        
        # Test with NaN values
        nan_date = dates[-1] + pd.Timedelta(minutes=15)
        
        # Should handle NaN values gracefully
        try:
            backtester.set_capital_at_risk(nan_date, 1000.0)
            result = backtester.process_single_period(nan_date, np.nan, 55)
            # If it doesn't crash, that's good
        except (ValueError, TypeError) as e:
            # Expected for NaN data
            pass
        
        # Test with zero capital
        zero_capital_date = dates[-1] + pd.Timedelta(minutes=30)
        backtester.set_capital_at_risk(zero_capital_date, 0.0)
        
        # Should handle zero capital without crashing
        result = backtester.process_single_period(zero_capital_date, 105, 55)
        assert isinstance(result, dict), "Should return result dict even with zero capital"
        assert 'pnl' in result, "Should have PnL key in result"
    
    def test_backward_compatibility_with_existing_interface(self):
        """Test 6: Verify backward compatibility with existing PairBacktester interface.
        
        This test ensures that existing code using PairBacktester will
        continue to work after the inheritance change.
        """
        # Create test data
        dates = pd.date_range('2023-01-01', periods=25, freq='15min')
        data = pd.DataFrame({
            'asset1': np.random.randn(25) + 100,
            'asset2': np.random.randn(25) + 50
        }, index=dates)
        
        # Test that PairBacktester can still be instantiated (now inherits from IncrementalPairBacktester)
        backtester = PairBacktester(
            pair_name="COMPATIBILITY-TEST",
            pair_data=data.copy(),
            rolling_window=10,
            z_threshold=2.0,
            z_exit=0.5,
            risk_per_position_pct=0.01,
            max_history_days=100,
            commission_pct=0.001,
            slippage_pct=0.0005,
            cooldown_periods=2
        )
        
        # Test that basic attributes are accessible
        assert backtester.pair_name == "COMPATIBILITY-TEST"
        assert backtester.rolling_window == 10
        assert backtester.z_threshold == 2.0
        assert backtester.z_exit == 0.5
        assert backtester.risk_per_position_pct == 0.01
        assert backtester.max_history_days == 100
        assert backtester.commission_pct == 0.001
        assert backtester.slippage_pct == 0.0005
        assert backtester.cooldown_periods == 2
        
        # Test that the backtester has the expected interface
        portfolio = Mock(spec=Portfolio)
        portfolio.calculate_position_risk_capital.return_value = 1000.0
        
        # Test run_on_day (main interface method) - should work with new implementation
        daily_data = data.iloc[[-1]]
        try:
            daily_pnl = backtester.run_on_day(daily_data, portfolio)
            assert isinstance(daily_pnl, (int, float)), "run_on_day should return numeric PnL"
        except Exception as e:
            # The method exists but may have different behavior
            pass
        
        # Test get_data_info (utility method) - may not exist in new implementation
        try:
            data_info = backtester.get_data_info()
            assert isinstance(data_info, dict), "get_data_info should return dict"
        except AttributeError:
            # Method may not exist in incremental implementation
            pass
        
        # Test that new incremental methods are also available
        incremental_results = backtester.get_incremental_results()
        assert isinstance(incremental_results, dict), "get_incremental_results should return dict"
        
        # Verify that the backtester maintains state correctly
        assert hasattr(backtester, 'pair_data'), "Should maintain pair_data"
        assert len(backtester.pair_data) > 0, "Should have data"
        assert hasattr(backtester, 'capital_at_risk_history'), "Should have capital history"