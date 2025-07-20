import pandas as pd
import numpy as np
from typing import Optional


class Portfolio:
    """Enhanced portfolio to track equity, cash, reserved margin and open positions."""

    def __init__(self, initial_capital: float, max_active_positions: int, leverage_limit: float = 1.0) -> None:
        self.initial_capital = initial_capital
        self.max_active_positions = max_active_positions
        self.leverage_limit = leverage_limit
        
        # Portfolio state tracking
        self.equity_curve = pd.Series(dtype=float)
        self.cash = initial_capital  # Available cash
        self.reserved_margin = 0.0  # Margin reserved for open positions
        self.active_positions: dict[str, dict] = {}
        
        # Track first test window start for equity initialization
        self.first_test_start: Optional[pd.Timestamp] = None
        self.equity_initialized = False

    def initialize_equity_curve(self, first_test_date: pd.Timestamp) -> None:
        """Initialize equity curve with first test window start date (removes artificial 1970-01-01)."""
        if not self.equity_initialized:
            self.first_test_start = first_test_date
            self.equity_curve = pd.Series(dtype=float)
            self.equity_curve[first_test_date] = self.initial_capital
            self.equity_initialized = True

    def can_open_position(self) -> bool:
        """Return True if a new position can be opened."""
        return len(self.active_positions) < self.max_active_positions

    def get_current_equity(self) -> float:
        """Return current equity (cash + unrealized PnL)."""
        if self.equity_curve.empty:
            return self.initial_capital
        return self.equity_curve.iloc[-1]

    def check_margin_requirements(self, notional: float) -> bool:
        """Check if position can be opened given margin requirements.
        
        Args:
            notional: Total notional value of the position
            
        Returns:
            bool: True if position can be opened within leverage limits
        """
        current_equity = self.get_current_equity()
        total_margin_after = self.reserved_margin + notional
        return total_margin_after <= current_equity * self.leverage_limit

    def calculate_position_risk_capital(self, risk_per_position_pct: float) -> float:
        """Amount of capital to risk for a new position."""
        return self.get_current_equity() * risk_per_position_pct

    def record_daily_pnl(self, date: pd.Timestamp, daily_pnl: float) -> None:
        """Update equity curve with daily PnL."""
        # Initialize equity curve if not done yet
        if not self.equity_initialized:
            self.initialize_equity_curve(date)
        
        last_equity = self.get_current_equity()
        new_equity = last_equity + daily_pnl
        self.equity_curve[date] = new_equity
        
        # Update cash (assuming PnL affects cash directly)
        self.cash = new_equity - self.reserved_margin

    def open_position(self, pair_name: str, position_details: dict) -> None:
        """Register an opened position and reserve margin."""
        notional = position_details.get('notional', 0.0)
        
        # Check margin requirements before opening
        if not self.check_margin_requirements(notional):
            raise ValueError(f"Insufficient margin for position {pair_name}: notional={notional}, available_margin={self.get_current_equity() * self.leverage_limit - self.reserved_margin}")
        
        self.active_positions[pair_name] = position_details
        self.reserved_margin += notional
        self.cash -= notional  # Reserve cash for margin

    def close_position(self, pair_name: str) -> None:
        """Remove a closed position from tracking and release margin."""
        if pair_name in self.active_positions:
            position_details = self.active_positions[pair_name]
            notional = position_details.get('notional', 0.0)
            
            # Release reserved margin
            self.reserved_margin -= notional
            self.cash += notional
            
            del self.active_positions[pair_name]

    def get_available_margin(self) -> float:
        """Get available margin for new positions."""
        current_equity = self.get_current_equity()
        max_margin = current_equity * self.leverage_limit
        return max_margin - self.reserved_margin
