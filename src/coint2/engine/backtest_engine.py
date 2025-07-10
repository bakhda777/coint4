import numpy as np
import pandas as pd

from ..core import performance


class PairBacktester:
    """Vectorized backtester for a single pair."""

    def __init__(
        self,
        pair_data: pd.DataFrame,
        beta: float,
        spread_mean: float,
        spread_std: float,
        z_threshold: float,
        z_exit: float = 0.0,
        commission_pct: float = 0.0,
        slippage_pct: float = 0.0,
        annualizing_factor: int = 365,
        capital_at_risk: float = 1.0,
        stop_loss_multiplier: float = 2.0,
        take_profit_multiplier: float = None,
        cooldown_periods: int = 0,
        wait_for_candle_close: bool = False,
        max_margin_usage: float = 1.0,
    ) -> None:
        """Initialize backtester with pre-computed parameters.

        Parameters
        ----------
        pair_data : pd.DataFrame
            DataFrame with two columns containing price series for the pair.
        beta : float
            Regression coefficient between ``y`` and ``x`` estimated on the
            training period.
        spread_mean : float
            Mean of the spread from the training period.
        spread_std : float
            Standard deviation of the spread from the training period.
        z_threshold : float
            Z-score absolute threshold for entry signals.
        z_exit : float
            Z-score absolute threshold for exit signals (default 0.0).
        cooldown_periods : int
            Number of periods to wait after closing position before re-entering.
        """
        self.pair_data = pair_data.copy()
        self.beta = beta
        self.mean = spread_mean
        self.std = spread_std
        self.z_threshold = z_threshold
        self.z_exit = z_exit
        self.cooldown_periods = cooldown_periods
        self.results: pd.DataFrame | None = None
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.annualizing_factor = annualizing_factor
        self.capital_at_risk = capital_at_risk
        self.stop_loss_multiplier = stop_loss_multiplier
        self.take_profit_multiplier = take_profit_multiplier if take_profit_multiplier is not None else stop_loss_multiplier
        self.wait_for_candle_close = wait_for_candle_close
        self.max_margin_usage = max_margin_usage

    def run(self) -> None:
        """Run backtest and store results in ``self.results``."""
        if self.pair_data.empty or len(self.pair_data.columns) < 2:
            self.results = pd.DataFrame(
                columns=["spread", "z_score", "position", "pnl", "cumulative_pnl"]
            )
            return

        # Переименовываем столбцы для удобства
        df = self.pair_data.rename(
            columns={
                self.pair_data.columns[0]: "y",
                self.pair_data.columns[1]: "x",
            }
        ).copy()

        # Рассчитываем спред и z-score
        df["spread"] = df["y"] - self.beta * df["x"]

        if self.std == 0:
            df["z_score"] = 0.0
        else:
            df["z_score"] = (df["spread"] - self.mean) / self.std

        # Инициализируем столбцы для результатов
        df["position"] = 0.0
        df["trades"] = 0.0
        df["pnl"] = 0.0
        df["costs"] = 0.0
        
        # Добавляем столбцы для расширенного логирования
        df["entry_price_s1"] = np.nan
        df["entry_price_s2"] = np.nan
        df["exit_price_s1"] = np.nan
        df["exit_price_s2"] = np.nan
        df["entry_z"] = np.nan
        df["exit_z"] = np.nan
        df["exit_reason"] = ""
        df["trade_duration"] = 0
        df["entry_date"] = pd.NaT

        total_cost_pct = self.commission_pct + self.slippage_pct

        position = 0.0
        entry_z = 0.0
        stop_loss_z = 0.0
        cooldown_remaining = 0  # Tracks remaining cooldown periods

        # Вынесем get_loc вычисления из цикла для оптимизации
        position_col_idx = df.columns.get_loc("position")
        trades_col_idx = df.columns.get_loc("trades")
        costs_col_idx = df.columns.get_loc("costs")
        pnl_col_idx = df.columns.get_loc("pnl")

        for i in range(1, len(df)):
            spread_prev = df["spread"].iat[i - 1]
            spread_curr = df["spread"].iat[i]
            z_curr = df["z_score"].iat[i]

            pnl = position * (spread_curr - spread_prev)

            new_position = position

            # Уменьшаем cooldown счетчик
            if cooldown_remaining > 0:
                cooldown_remaining -= 1

            # Закрытие позиций в конце теста для чистоты метрик
            if i == len(df) - 1 and position != 0:
                new_position = 0.0  # Форс-закрытие в последнем периоде
                cooldown_remaining = self.cooldown_periods
                # Логируем выход
                df.loc[df.index[i], 'exit_reason'] = 'end_of_test'
                df.loc[df.index[i], 'exit_price_s1'] = df.loc[df.index[i], 'y']
                df.loc[df.index[i], 'exit_price_s2'] = df.loc[df.index[i], 'x']
                df.loc[df.index[i], 'exit_z'] = z_curr
                # Рассчитываем длительность сделки
                entry_date = df.loc[(df['entry_date'].notna()) & (df.index <= df.index[i])]['entry_date'].iloc[-1]
                if pd.notna(entry_date):
                    df.loc[df.index[i], 'trade_duration'] = (df.index[i] - entry_date).total_seconds() / 3600  # В часах
            # Стоп-лосс выходы
            elif position > 0 and z_curr <= stop_loss_z:
                new_position = 0.0
                cooldown_remaining = self.cooldown_periods
                # Логируем стоп-лосс
                df.loc[df.index[i], 'exit_reason'] = 'stop_loss'
                df.loc[df.index[i], 'exit_price_s1'] = df.loc[df.index[i], 'y']
                df.loc[df.index[i], 'exit_price_s2'] = df.loc[df.index[i], 'x']
                df.loc[df.index[i], 'exit_z'] = z_curr
                # Рассчитываем длительность сделки
                entry_date = df.loc[(df['entry_date'].notna()) & (df.index <= df.index[i])]['entry_date'].iloc[-1]
                if pd.notna(entry_date):
                    df.loc[df.index[i], 'trade_duration'] = (df.index[i] - entry_date).total_seconds() / 3600  # В часах
            elif position < 0 and z_curr >= stop_loss_z:
                new_position = 0.0
                cooldown_remaining = self.cooldown_periods
                # Логируем стоп-лосс
                df.loc[df.index[i], 'exit_reason'] = 'stop_loss'
                df.loc[df.index[i], 'exit_price_s1'] = df.loc[df.index[i], 'y']
                df.loc[df.index[i], 'exit_price_s2'] = df.loc[df.index[i], 'x']
                df.loc[df.index[i], 'exit_z'] = z_curr
                # Рассчитываем длительность сделки
                entry_date = df.loc[(df['entry_date'].notna()) & (df.index <= df.index[i])]['entry_date'].iloc[-1]
                if pd.notna(entry_date):
                    df.loc[df.index[i], 'trade_duration'] = (df.index[i] - entry_date).total_seconds() / 3600  # В часах
            # Take-profit выходы (новая логика)
            elif position > 0 and z_curr >= float(np.sign(entry_z) * self.take_profit_multiplier):
                new_position = 0.0
                cooldown_remaining = self.cooldown_periods
                # Логируем take-profit
                df.loc[df.index[i], 'exit_reason'] = 'take_profit'
                df.loc[df.index[i], 'exit_price_s1'] = df.loc[df.index[i], 'y']
                df.loc[df.index[i], 'exit_price_s2'] = df.loc[df.index[i], 'x']
                df.loc[df.index[i], 'exit_z'] = z_curr
                # Рассчитываем длительность сделки
                entry_date = df.loc[(df['entry_date'].notna()) & (df.index <= df.index[i])]['entry_date'].iloc[-1]
                if pd.notna(entry_date):
                    df.loc[df.index[i], 'trade_duration'] = (df.index[i] - entry_date).total_seconds() / 3600  # В часах
            elif position < 0 and z_curr <= float(np.sign(entry_z) * self.take_profit_multiplier):
                new_position = 0.0
                cooldown_remaining = self.cooldown_periods
                # Логируем take-profit
                df.loc[df.index[i], 'exit_reason'] = 'take_profit'
                df.loc[df.index[i], 'exit_price_s1'] = df.loc[df.index[i], 'y']
                df.loc[df.index[i], 'exit_price_s2'] = df.loc[df.index[i], 'x']
                df.loc[df.index[i], 'exit_z'] = z_curr
                # Рассчитываем длительность сделки
                entry_date = df.loc[(df['entry_date'].notna()) & (df.index <= df.index[i])]['entry_date'].iloc[-1]
                if pd.notna(entry_date):
                    df.loc[df.index[i], 'trade_duration'] = (df.index[i] - entry_date).total_seconds() / 3600  # В часах
            # Z-score exit: закрываем позицию если z-score вернулся к заданному уровню
            elif position != 0 and abs(z_curr) <= self.z_exit:
                new_position = 0.0
                cooldown_remaining = self.cooldown_periods
                # Логируем выход по z-score
                df.loc[df.index[i], 'exit_reason'] = 'z_exit'
                df.loc[df.index[i], 'exit_price_s1'] = df.loc[df.index[i], 'y']
                df.loc[df.index[i], 'exit_price_s2'] = df.loc[df.index[i], 'x']
                df.loc[df.index[i], 'exit_z'] = z_curr
                # Рассчитываем длительность сделки
                entry_date = df.loc[(df['entry_date'].notna()) & (df.index <= df.index[i])]['entry_date'].iloc[-1]
                if pd.notna(entry_date):
                    df.loc[df.index[i], 'trade_duration'] = (df.index[i] - entry_date).total_seconds() / 3600  # В часах

            # Проверяем сигналы входа только если не в последнем периоде и не в cooldown
            if i < len(df) - 1 and cooldown_remaining == 0:
                # Сигнал на покупку y и продажу x
                if z_curr <= -self.z_threshold and position == 0:
                    # Рассчитываем стоп-лосс и тейк-профит
                    entry_z = z_curr
                    stop_loss_z = float(np.sign(entry_z) * self.stop_loss_multiplier)
                    take_profit_z = float(np.sign(entry_z) * self.take_profit_multiplier)
                    
                    # Рассчитываем размер позиции на основе риска
                    stop_loss_price = self.mean + stop_loss_z * self.std
                    risk_per_unit = abs(spread_curr - stop_loss_price)
                    trade_value = df["y"].iat[i] + abs(self.beta) * df["x"].iat[i]
                    size_risk = self.capital_at_risk / risk_per_unit if risk_per_unit != 0 else 0.0
                    size_value = self.capital_at_risk / trade_value if trade_value != 0 else 0.0
                    # Ограничиваем размер позиции с учетом max_margin_usage
                    margin_limit = self.capital_at_risk * self.max_margin_usage / trade_value if trade_value != 0 else 0.0
                    size = min(size_risk, size_value, margin_limit)
                    
                    new_position = 1.0 * size
                    
                    # Логируем параметры входа
                    df.loc[df.index[i], 'entry_price_s1'] = df.loc[df.index[i], 'y']
                    df.loc[df.index[i], 'entry_price_s2'] = df.loc[df.index[i], 'x']
                    df.loc[df.index[i], 'entry_z'] = z_curr
                    df.loc[df.index[i], 'entry_date'] = df.index[i]
                    
                # Сигнал на продажу y и покупку x
                elif z_curr >= self.z_threshold and position == 0:
                    # Рассчитываем стоп-лосс и тейк-профит
                    entry_z = z_curr
                    stop_loss_z = float(np.sign(entry_z) * self.stop_loss_multiplier)
                    take_profit_z = float(np.sign(entry_z) * self.take_profit_multiplier)
                    
                    # Рассчитываем размер позиции на основе риска
                    stop_loss_price = self.mean + stop_loss_z * self.std
                    risk_per_unit = abs(spread_curr - stop_loss_price)
                    trade_value = df["y"].iat[i] + abs(self.beta) * df["x"].iat[i]
                    size_risk = self.capital_at_risk / risk_per_unit if risk_per_unit != 0 else 0.0
                    size_value = self.capital_at_risk / trade_value if trade_value != 0 else 0.0
                    # Ограничиваем размер позиции с учетом max_margin_usage
                    margin_limit = self.capital_at_risk * self.max_margin_usage / trade_value if trade_value != 0 else 0.0
                    size = min(size_risk, size_value, margin_limit)
                    
                    new_position = -1.0 * size
                    
                    # Логируем параметры входа
                    df.loc[df.index[i], 'entry_price_s1'] = df.loc[df.index[i], 'y']
                    df.loc[df.index[i], 'entry_price_s2'] = df.loc[df.index[i], 'x']
                    df.loc[df.index[i], 'entry_z'] = z_curr
                    df.loc[df.index[i], 'entry_date'] = df.index[i]

            trades = abs(new_position - position)
            trade_value = df["y"].iat[i] + abs(self.beta) * df["x"].iat[i]
            # Ограничиваем общее плечо/маржу с помощью max_margin_usage
            # Рассчитываем текущую экспозицию (плечо) как процент от капитала
            current_exposure = abs(new_position) * trade_value
            max_allowed_exposure = self.capital_at_risk / self.max_margin_usage
            
            # Если текущая экспозиция превышает максимально допустимую, корректируем размер позиции
            if current_exposure > max_allowed_exposure and max_allowed_exposure > 0:
                new_position = new_position * (max_allowed_exposure / current_exposure)
            costs = trades * trade_value * total_cost_pct

            df.iat[i, position_col_idx] = new_position
            df.iat[i, trades_col_idx] = trades
            df.iat[i, costs_col_idx] = costs
            df.iat[i, pnl_col_idx] = pnl - costs

            position = new_position

        df["cumulative_pnl"] = df["pnl"].cumsum()

        self.results = df

    def get_results(self) -> pd.DataFrame:
        if self.results is None:
            raise ValueError("Backtest not yet run")
        return self.results[["spread", "z_score", "position", "pnl", "costs", "cumulative_pnl"]]

    def get_performance_metrics(self) -> dict:
        if self.results is None or self.results.empty:
            raise ValueError("Backtest has not been run or produced no results")

        pnl = self.results["pnl"].dropna()
        cum_pnl = self.results["cumulative_pnl"].dropna()

        # Если после dropna ничего не осталось, возвращаем нулевые метрики
        if pnl.empty:
            return {
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_pnl": 0.0,
            }

        sharpe = performance.sharpe_ratio(pnl, self.annualizing_factor)
        
        return {
            "sharpe_ratio": 0.0 if np.isnan(sharpe) else sharpe,
            "max_drawdown": performance.max_drawdown(cum_pnl),
            "total_pnl": cum_pnl.iloc[-1] if not cum_pnl.empty else 0.0,
        }
