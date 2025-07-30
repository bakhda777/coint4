import logging
import pandas as pd

from coint2.engine.numba_backtest_engine_full import FullNumbaPairBacktester as EnginePairBacktester
from coint2.engine.numba_backtest_engine_full import FullNumbaPairBacktester as IncrementalPairBacktester

from .portfolio import Portfolio

logger = logging.getLogger(__name__)


class PairBacktester(IncrementalPairBacktester):
    """Extension of IncrementalPairBacktester with portfolio awareness and memory management.
    
    This version fixes the look-ahead bias in capital allocation by using incremental
    processing and maintaining a time series of capital_at_risk values.
    """

    def __init__(
        self,
        pair_name: str,
        *args,
        portfolio: Portfolio = None,
        risk_per_position_pct: float = 0.01,
        max_history_days: int = 252,
        incremental_mode: bool = False,
        **kwargs
    ) -> None:
        self.pair_name = pair_name
        self.risk_per_position_pct = risk_per_position_pct
        self.max_history_days = max_history_days
        self.incremental_mode = incremental_mode
        
        # Pass portfolio and pair_name to the parent class
        super().__init__(*args, portfolio=portfolio, pair_name=pair_name, **kwargs)
        
        # Only clear data if in incremental mode
        if self.incremental_mode and hasattr(self, 'pair_data') and not self.pair_data.empty:
            # Keep only the column structure, clear the data
            self.pair_data = self.pair_data.iloc[:0].copy()
        
        # Track previous PnL for incremental calculation
        self._previous_total_pnl = 0.0
        
        # Validate that we have enough history for rolling calculations
        min_required_days = self.rolling_window + 10  # Add buffer for safety
        if self.max_history_days < min_required_days:
            logger.warning(
                f"max_history_days ({self.max_history_days}) is less than recommended "
                f"minimum ({min_required_days}) for rolling_window ({self.rolling_window}). "
                "This may affect calculation accuracy."
            )

    def run_on_day(self, daily_data: pd.DataFrame, portfolio: Portfolio) -> float:
        """Process one day of data and return daily PnL using incremental approach.

        This implementation fixes the look-ahead bias by:
        1. Setting capital_at_risk for the current date before processing
        2. Using incremental processing to avoid full recalculation
        3. Maintaining proper time series of capital allocation
        
        Memory management: Maintains a sliding window of historical data
        limited by max_history_days to prevent unlimited memory growth.
        """

        if daily_data.empty:
            return 0.0

        # Get current date from daily_data
        current_date = daily_data.index[0]
        
        # Calculate capital at risk for this specific date
        current_capital_at_risk = portfolio.calculate_position_risk_capital(
            risk_per_position_pct=self.risk_per_position_pct,
            max_position_size_pct=getattr(portfolio, 'max_position_size_pct', 1.0)
        )
        
        # Set capital at risk for this date in the time series
        self.set_capital_at_risk(current_date, current_capital_at_risk)

        # Append new data
        self.pair_data = pd.concat([self.pair_data, daily_data])
        
        # Implement sliding window memory management
        current_size = len(self.pair_data)
        if current_size > self.max_history_days:
            # Calculate how many rows to remove
            rows_to_remove = current_size - self.max_history_days
            
            # Keep only the most recent max_history_days rows
            self.pair_data = self.pair_data.iloc[rows_to_remove:].copy()
            
            logger.debug(
                f"Trimmed pair_data from {current_size} to {len(self.pair_data)} rows "
                f"for pair {self.pair_name}"
            )
        
        # Ensure we have minimum required data for calculations
        min_required = self.rolling_window + 1
        if len(self.pair_data) < min_required:
            logger.warning(
                f"Insufficient data for pair {self.pair_name}: "
                f"{len(self.pair_data)} rows, need at least {min_required}"
            )
            return 0.0
        
        # Get current prices
        price_s1 = daily_data.iloc[0, 0]
        price_s2 = daily_data.iloc[0, 1]
        
        # Process single period incrementally
        result = self.process_single_period(current_date, price_s1, price_s2)
        
        return float(result.get('pnl', 0.0))
    
    def get_data_info(self) -> dict:
        """Get information about current data size and memory usage.
        
        Returns
        -------
        dict
            Dictionary containing data size metrics and memory information.
        """
        data_size = len(self.pair_data)
        memory_usage_mb = self.pair_data.memory_usage(deep=True).sum() / (1024 * 1024)
        
        return {
            "pair_name": self.pair_name,
            "current_data_size": data_size,
            "max_history_days": self.max_history_days,
            "memory_usage_mb": round(memory_usage_mb, 2),
            "rolling_window": self.rolling_window,
            "data_utilization_pct": round((data_size / self.max_history_days) * 100, 1) if self.max_history_days > 0 else 0
        }
