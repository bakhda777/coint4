import pandas as pd

from coint2.engine.backtest_engine import PairBacktester as EnginePairBacktester

from .portfolio import Portfolio


class PairBacktester(EnginePairBacktester):
    """Extension of EnginePairBacktester with portfolio awareness."""

    def __init__(self, pair_name: str, *args, risk_per_position_pct: float = 0.01, **kwargs) -> None:
        self.pair_name = pair_name
        self.risk_per_position_pct = risk_per_position_pct
        super().__init__(*args, **kwargs)

    def run_on_day(self, daily_data: pd.DataFrame, portfolio: Portfolio) -> float:
        """Process one day of data and return daily PnL.

        This simplified implementation recalculates the backtest on the
        accumulated data and takes the latest PnL value as the result for the
        provided ``daily_data`` slice. It also adjusts the ``capital_at_risk``
        based on the current portfolio equity.
        """

        if daily_data.empty:
            return 0.0

        self.capital_at_risk = portfolio.calculate_position_risk_capital(
            self.risk_per_position_pct
        )

        # Append new data and rerun the vectorised engine
        self.pair_data = pd.concat([self.pair_data, daily_data])
        super().run()
        results = super().get_results()
        pnl_series = results.get("pnl", pd.Series(dtype=float))
        if pnl_series.empty:
            return 0.0

        daily_pnl = pnl_series.iloc[-1]
        return float(daily_pnl)
