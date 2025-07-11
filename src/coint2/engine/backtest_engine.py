import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..core import performance


class PairBacktester:
    """Vectorized backtester for a single pair."""

    def __init__(
        self,
        pair_data: pd.DataFrame,
        rolling_window: int,
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
        max_margin_usage: float = float("inf"),
        half_life: float | None = None,
        time_stop_multiplier: float | None = None,
    ) -> None:
        """Initialize backtester.

        Parameters
        ----------
        pair_data : pd.DataFrame
            DataFrame with two columns containing price series for the pair.
        rolling_window : int
            Window size for rolling parameter estimation.
        z_threshold : float
            Z-score absolute threshold for entry signals.
        z_exit : float
            Z-score absolute threshold for exit signals (default 0.0).
        cooldown_periods : int
            Number of periods to wait after closing position before re-entering.
        """
        self.pair_data = pair_data.copy()
        self.rolling_window = rolling_window
        self.z_threshold = z_threshold
        self.z_exit = z_exit
        self.cooldown_periods = cooldown_periods
        self.results: pd.DataFrame | None = None
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.annualizing_factor = annualizing_factor
        self.capital_at_risk = capital_at_risk
        self.stop_loss_multiplier = stop_loss_multiplier
        self.take_profit_multiplier = take_profit_multiplier
        self.wait_for_candle_close = wait_for_candle_close
        self.max_margin_usage = max_margin_usage
        self.half_life = half_life
        self.time_stop_multiplier = time_stop_multiplier
        self.s1 = pair_data.columns[0]
        self.s2 = pair_data.columns[1]
        self.trades_log: list[dict] = []
        # Track trade entry time for time based exits
        self.entry_time = None

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

        # Prepare columns for rolling parameters
        df["beta"] = np.nan
        df["mean"] = np.nan
        df["std"] = np.nan
        df["spread"] = np.nan
        df["z_score"] = np.nan

        for i in range(self.rolling_window, len(df)):
            window = df.iloc[i - self.rolling_window : i]
            y_win = window["y"]
            x_win = window["x"]
            x_const = sm.add_constant(x_win)
            model = sm.OLS(y_win, x_const).fit()
            beta = model.params.iloc[1]
            spread_win = y_win - beta * x_win
            mean = spread_win.mean()
            std = spread_win.std()
            if std < 1e-6:
                continue
            current_spread = df["y"].iat[i] - beta * df["x"].iat[i]
            z = (current_spread - mean) / std
            df.loc[df.index[i], ["beta", "mean", "std", "spread", "z_score"]] = [
                beta,
                mean,
                std,
                current_spread,
                z,
            ]

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
        df["trade_duration"] = 0.0
        df["entry_date"] = np.nan

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

        # Variables for detailed trade logging
        entry_datetime = None
        entry_spread = 0.0
        entry_position_size = 0.0
        entry_index = 0
        current_trade_pnl = 0.0

        for i in range(1, len(df)):
            if (
                pd.isna(df["spread"].iat[i])
                or pd.isna(df["spread"].iat[i - 1])
                or pd.isna(df["z_score"].iat[i])
            ):
                df.iat[i, position_col_idx] = position
                df.iat[i, trades_col_idx] = 0.0
                df.iat[i, costs_col_idx] = 0.0
                df.iat[i, pnl_col_idx] = 0.0
                continue

            beta = df["beta"].iat[i]
            mean = df["mean"].iat[i]
            std = df["std"].iat[i]
            spread_prev = df["spread"].iat[i - 1]
            spread_curr = df["spread"].iat[i]
            z_curr = df["z_score"].iat[i]

            pnl = position * (spread_curr - spread_prev)

            new_position = position

            # Time-based stop-loss
            if (
                position != 0
                and self.half_life is not None
                and self.time_stop_multiplier is not None
                and self.entry_time is not None
            ):
                trade_duration_days = (
                    df.index[i] - self.entry_time
                ).total_seconds() / (60 * 60 * 24)
                time_stop_limit = self.half_life * self.time_stop_multiplier
                if trade_duration_days >= time_stop_limit:
                    new_position = 0.0
                    cooldown_remaining = self.cooldown_periods
                    df.loc[df.index[i], 'exit_reason'] = 'time_stop'
                    df.loc[df.index[i], 'exit_price_s1'] = df.loc[df.index[i], 'y']
                    df.loc[df.index[i], 'exit_price_s2'] = df.loc[df.index[i], 'x']
                    df.loc[df.index[i], 'exit_z'] = z_curr
                    entry_date = df.loc[(df['entry_date'].notna()) & (df.index <= df.index[i])]['entry_date'].iloc[-1]
                    if pd.notna(entry_date):
                        if isinstance(df.index, pd.DatetimeIndex):
                            df.loc[df.index[i], 'trade_duration'] = (
                                df.index[i] - pd.to_datetime(entry_date)
                            ).total_seconds() / 3600
                        else:
                            df.loc[df.index[i], 'trade_duration'] = float(df.index[i] - entry_date)

            # Уменьшаем cooldown счетчик
            if cooldown_remaining > 0:
                cooldown_remaining -= 1

            # Закрытие позиций в конце теста для чистоты метрик
            if i == len(df) - 1 and position != 0:
                new_position = 0.0  # Форс-закрытие в последнем периоде
                cooldown_remaining = self.cooldown_periods
                # Логируем выход
                df.loc[df.index[i], "exit_reason"] = "end_of_test"
                df.loc[df.index[i], "exit_price_s1"] = df.loc[df.index[i], "y"]
                df.loc[df.index[i], "exit_price_s2"] = df.loc[df.index[i], "x"]
                df.loc[df.index[i], "exit_z"] = z_curr
                # Рассчитываем длительность сделки
                entry_date = df.loc[
                    (df["entry_date"].notna()) & (df.index <= df.index[i])
                ]["entry_date"].iloc[-1]
                if pd.notna(entry_date):
                    if isinstance(df.index, pd.DatetimeIndex):
                        df.loc[df.index[i], "trade_duration"] = (
                            df.index[i] - pd.to_datetime(entry_date)
                        ).total_seconds() / 3600
                    else:
                        df.loc[df.index[i], "trade_duration"] = float(
                            df.index[i] - entry_date
                        )
            # Стоп-лосс выходы
            elif position > 0 and z_curr <= stop_loss_z:
                new_position = 0.0
                cooldown_remaining = self.cooldown_periods
                # Логируем стоп-лосс
                df.loc[df.index[i], "exit_reason"] = "stop_loss"
                df.loc[df.index[i], "exit_price_s1"] = df.loc[df.index[i], "y"]
                df.loc[df.index[i], "exit_price_s2"] = df.loc[df.index[i], "x"]
                df.loc[df.index[i], "exit_z"] = z_curr
                # Рассчитываем длительность сделки
                entry_date = df.loc[
                    (df["entry_date"].notna()) & (df.index <= df.index[i])
                ]["entry_date"].iloc[-1]
                if pd.notna(entry_date):
                    if isinstance(df.index, pd.DatetimeIndex):
                        df.loc[df.index[i], "trade_duration"] = (
                            df.index[i] - pd.to_datetime(entry_date)
                        ).total_seconds() / 3600
                    else:
                        df.loc[df.index[i], "trade_duration"] = float(
                            df.index[i] - entry_date
                        )
            elif position < 0 and z_curr >= stop_loss_z:
                new_position = 0.0
                cooldown_remaining = self.cooldown_periods
                # Логируем стоп-лосс
                df.loc[df.index[i], "exit_reason"] = "stop_loss"
                df.loc[df.index[i], "exit_price_s1"] = df.loc[df.index[i], "y"]
                df.loc[df.index[i], "exit_price_s2"] = df.loc[df.index[i], "x"]
                df.loc[df.index[i], "exit_z"] = z_curr
                # Рассчитываем длительность сделки
                entry_date = df.loc[
                    (df["entry_date"].notna()) & (df.index <= df.index[i])
                ]["entry_date"].iloc[-1]
                if pd.notna(entry_date):
                    if isinstance(df.index, pd.DatetimeIndex):
                        df.loc[df.index[i], "trade_duration"] = (
                            df.index[i] - pd.to_datetime(entry_date)
                        ).total_seconds() / 3600
                    else:
                        df.loc[df.index[i], "trade_duration"] = float(
                            df.index[i] - entry_date
                        )
            # Take-profit выходы (новая логика)
            elif (
                self.take_profit_multiplier is not None
                and position > 0
                and z_curr >= float(np.sign(entry_z) * self.take_profit_multiplier)
            ):
                new_position = 0.0
                cooldown_remaining = self.cooldown_periods
                # Логируем take-profit
                df.loc[df.index[i], "exit_reason"] = "take_profit"
                df.loc[df.index[i], "exit_price_s1"] = df.loc[df.index[i], "y"]
                df.loc[df.index[i], "exit_price_s2"] = df.loc[df.index[i], "x"]
                df.loc[df.index[i], "exit_z"] = z_curr
                # Рассчитываем длительность сделки
                entry_date = df.loc[
                    (df["entry_date"].notna()) & (df.index <= df.index[i])
                ]["entry_date"].iloc[-1]
                if pd.notna(entry_date):
                    if isinstance(df.index, pd.DatetimeIndex):
                        df.loc[df.index[i], "trade_duration"] = (
                            df.index[i] - pd.to_datetime(entry_date)
                        ).total_seconds() / 3600
                    else:
                        df.loc[df.index[i], "trade_duration"] = float(
                            df.index[i] - entry_date
                        )
            elif (
                self.take_profit_multiplier is not None
                and position < 0
                and z_curr <= float(np.sign(entry_z) * self.take_profit_multiplier)
            ):
                new_position = 0.0
                cooldown_remaining = self.cooldown_periods
                # Логируем take-profit
                df.loc[df.index[i], "exit_reason"] = "take_profit"
                df.loc[df.index[i], "exit_price_s1"] = df.loc[df.index[i], "y"]
                df.loc[df.index[i], "exit_price_s2"] = df.loc[df.index[i], "x"]
                df.loc[df.index[i], "exit_z"] = z_curr
                # Рассчитываем длительность сделки
                entry_date = df.loc[
                    (df["entry_date"].notna()) & (df.index <= df.index[i])
                ]["entry_date"].iloc[-1]
                if pd.notna(entry_date):
                    if isinstance(df.index, pd.DatetimeIndex):
                        df.loc[df.index[i], "trade_duration"] = (
                            df.index[i] - pd.to_datetime(entry_date)
                        ).total_seconds() / 3600
                    else:
                        df.loc[df.index[i], "trade_duration"] = float(
                            df.index[i] - entry_date
                        )
            # Z-score exit: закрываем позицию если z-score вернулся к заданному уровню
            elif position != 0 and abs(z_curr) <= self.z_exit:
                new_position = 0.0
                cooldown_remaining = self.cooldown_periods
                # Логируем выход по z-score
                df.loc[df.index[i], "exit_reason"] = "z_exit"
                df.loc[df.index[i], "exit_price_s1"] = df.loc[df.index[i], "y"]
                df.loc[df.index[i], "exit_price_s2"] = df.loc[df.index[i], "x"]
                df.loc[df.index[i], "exit_z"] = z_curr
                # Рассчитываем длительность сделки
                entry_date = df.loc[
                    (df["entry_date"].notna()) & (df.index <= df.index[i])
                ]["entry_date"].iloc[-1]
                if pd.notna(entry_date):
                    if isinstance(df.index, pd.DatetimeIndex):
                        df.loc[df.index[i], "trade_duration"] = (
                            df.index[i] - pd.to_datetime(entry_date)
                        ).total_seconds() / 3600
                    else:
                        df.loc[df.index[i], "trade_duration"] = float(
                            df.index[i] - entry_date
                        )

            # Проверяем сигналы входа только если не в последнем периоде и не в cooldown
            if i < len(df) - 1 and cooldown_remaining == 0:
                signal = 0
                if z_curr >= self.z_threshold:
                    signal = -1
                elif z_curr <= -self.z_threshold:
                    signal = 1

                z_prev = df["z_score"].iat[i - 1]
                long_confirmation = (signal == 1) and (z_curr > z_prev)
                short_confirmation = (signal == -1) and (z_curr < z_prev)

                if new_position == 0 and (long_confirmation or short_confirmation):
                    entry_z = z_curr
                    stop_loss_z = float(np.sign(entry_z) * self.stop_loss_multiplier)
                    if self.take_profit_multiplier is not None:
                        take_profit_z = float(
                            np.sign(entry_z) * self.take_profit_multiplier
                        )

                    stop_loss_price = mean + stop_loss_z * std
                    risk_per_unit = abs(spread_curr - stop_loss_price)
                    trade_value = df["y"].iat[i] + abs(beta) * df["x"].iat[i]
                    size_risk = (
                        self.capital_at_risk / risk_per_unit
                        if risk_per_unit != 0
                        else 0.0
                    )
                    size_value = (
                        self.capital_at_risk / trade_value if trade_value != 0 else 0.0
                    )
                    margin_limit = (
                        self.capital_at_risk * self.max_margin_usage / trade_value
                        if trade_value != 0
                        else 0.0
                    )
                    size = min(size_risk, size_value, margin_limit)

                    new_position = signal * size

                    df.loc[df.index[i], "entry_price_s1"] = df.loc[df.index[i], "y"]
                    df.loc[df.index[i], "entry_price_s2"] = df.loc[df.index[i], "x"]
                    df.loc[df.index[i], "entry_z"] = z_curr
                    if isinstance(df.index, pd.DatetimeIndex):
                        df.loc[df.index[i], "entry_date"] = df.index[i]
                    else:
                        df.loc[df.index[i], "entry_date"] = float(i)
                elif (
                    new_position != 0
                    and (long_confirmation or short_confirmation)
                    and np.sign(new_position) != signal
                ):
                    entry_z = z_curr
                    stop_loss_z = float(np.sign(entry_z) * self.stop_loss_multiplier)
                    if self.take_profit_multiplier is not None:
                        take_profit_z = float(
                            np.sign(entry_z) * self.take_profit_multiplier
                        )

                    stop_loss_price = mean + stop_loss_z * std
                    risk_per_unit = abs(spread_curr - stop_loss_price)
                    trade_value = df["y"].iat[i] + abs(beta) * df["x"].iat[i]
                    size_risk = (
                        self.capital_at_risk / risk_per_unit
                        if risk_per_unit != 0
                        else 0.0
                    )
                    size_value = (
                        self.capital_at_risk / trade_value if trade_value != 0 else 0.0
                    )
                    margin_limit = (
                        self.capital_at_risk * self.max_margin_usage / trade_value
                        if trade_value != 0
                        else 0.0
                    )
                    size = min(size_risk, size_value, margin_limit)

                    new_position = signal * size

                    df.loc[df.index[i], "entry_price_s1"] = df.loc[df.index[i], "y"]
                    df.loc[df.index[i], "entry_price_s2"] = df.loc[df.index[i], "x"]
                    df.loc[df.index[i], "entry_z"] = z_curr
                    if isinstance(df.index, pd.DatetimeIndex):
                        df.loc[df.index[i], "entry_date"] = df.index[i]
                    else:
                        df.loc[df.index[i], "entry_date"] = float(i)

            trades = abs(new_position - position)
            trade_value = df["y"].iat[i] + abs(beta) * df["x"].iat[i]
            # Ограничиваем общее плечо/маржу с помощью max_margin_usage
            # Рассчитываем текущую экспозицию (плечо) как процент от капитала
            current_exposure = abs(new_position) * trade_value

            if np.isfinite(self.max_margin_usage):
                max_allowed_exposure = self.capital_at_risk / self.max_margin_usage

                if current_exposure > max_allowed_exposure and max_allowed_exposure > 0:
                    new_position = new_position * (
                        max_allowed_exposure / current_exposure
                    )
            # 1. Рассчитываем PnL и точные расходы (из версии main)
            # Стоимость изменения позиции, учитывая комиссии и проскальзывание
            price_s1 = df["y"].iat[i]
            price_s2 = df["x"].iat[i]
            position_s1_change = new_position - position
            position_s2_change = -new_position * beta - (-position * beta)

            notional_change_s1 = abs(position_s1_change * price_s1)
            notional_change_s2 = abs(position_s2_change * price_s2)

            commission = (notional_change_s1 + notional_change_s2) * self.commission_pct
            slippage = (notional_change_s1 + notional_change_s2) * self.slippage_pct
            # Combine costs and subtract from PnL for this step
            total_costs = commission + slippage

            # PnL after accounting for trading costs
            step_pnl = pnl - total_costs

            # 2. Обновляем основной DataFrame
            df.iat[i, position_col_idx] = new_position
            df.iat[i, trades_col_idx] = trades
            df.iat[i, costs_col_idx] = total_costs
            df.iat[i, pnl_col_idx] = step_pnl

            # 3. Накапливаем PnL для детального лога (из версии codex)
            # Update trade PnL accumulator if a trade is open
            if position != 0 or (position == 0 and new_position != 0):
                current_trade_pnl += step_pnl

            # Handle entry logging
            if position == 0 and new_position != 0:
                entry_datetime = df.index[i]
                entry_spread = spread_curr
                entry_position_size = new_position
                entry_index = i
                self.entry_time = df.index[i]

            # Handle exit logging
            if position != 0 and new_position == 0 and entry_datetime is not None:
                exit_datetime = df.index[i]
                if isinstance(df.index, pd.DatetimeIndex):
                    duration_hours = (
                        exit_datetime - entry_datetime
                    ).total_seconds() / 3600
                else:
                    duration_hours = float(i - entry_index)

                trade_info = {
                    "pair": f"{self.s1}-{self.s2}",
                    "entry_datetime": entry_datetime,
                    "exit_datetime": exit_datetime,
                    "position_type": "long" if entry_position_size > 0 else "short",
                    "entry_price_spread": entry_spread,
                    "exit_price_spread": spread_curr,
                    "pnl": current_trade_pnl,
                    "exit_reason": df.loc[df.index[i], "exit_reason"],
                    "trade_duration_hours": duration_hours,
                }
                self.trades_log.append(trade_info)

                # Сбрасываем счетчики для следующей сделки
                current_trade_pnl = 0.0
                entry_datetime = None
                entry_spread = 0.0
                entry_position_size = 0.0
                self.entry_time = None

            position = new_position

        df["cumulative_pnl"] = df["pnl"].cumsum()

        self.results = df

    def get_results(self) -> dict:
        if self.results is None:
            raise ValueError("Backtest not yet run")

        return {
            "spread": self.results["spread"],
            "z_score": self.results["z_score"],
            "position": self.results["position"],
            "trades": self.results["trades"],
            "costs": self.results["costs"],
            "pnl": self.results["pnl"],
            "cumulative_pnl": self.results["cumulative_pnl"],
            "trades_log": self.trades_log,
        }

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
