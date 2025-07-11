import pandas as pd


class Portfolio:
    """Simple portfolio to track equity and open positions."""

    def __init__(self, initial_capital: float, max_active_positions: int) -> None:
        self.initial_capital = initial_capital
        self.max_active_positions = max_active_positions
        self.equity_curve = pd.Series(dtype=float)
        self.equity_curve[pd.Timestamp("1970-01-01")] = initial_capital
        self.active_positions: dict[str, dict] = {}

    def can_open_position(self) -> bool:
        """Return True if a new position can be opened."""
        return len(self.active_positions) < self.max_active_positions

    def get_current_equity(self) -> float:
        """Return last recorded equity value."""
        return self.equity_curve.iloc[-1]

    def calculate_position_risk_capital(self, risk_per_position_pct: float) -> float:
        """Amount of capital to risk for a new position."""
        return self.get_current_equity() * risk_per_position_pct

    def record_daily_pnl(self, date: pd.Timestamp, daily_pnl: float) -> None:
        """Update equity curve with daily PnL."""
        last_equity = self.get_current_equity()
        self.equity_curve[date] = last_equity + daily_pnl

    def open_position(self, pair_name: str, position_details: dict) -> None:
        """Register an opened position."""
        self.active_positions[pair_name] = position_details

    def close_position(self, pair_name: str) -> None:
        """Remove a closed position from tracking."""
        if pair_name in self.active_positions:
            del self.active_positions[pair_name]
