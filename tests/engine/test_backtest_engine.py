# ВНИМАНИЕ: Строки с sys.path.insert удалены! Они больше не нужны благодаря conftest.py.

import numpy as np
import pandas as pd
import statsmodels.api as sm

from coint2.core import performance

# Импортируем код проекта напрямую
from coint2.engine.backtest_engine import PairBacktester


def calc_params(df: pd.DataFrame) -> tuple[float, float, float]:
    """Calculate beta, mean and std of spread for the given DataFrame."""
    y_col, x_col = df.columns[0], df.columns[1]
    beta = df[y_col].cov(df[x_col]) / df[x_col].var()
    spread = df[y_col] - beta * df[x_col]
    return beta, spread.mean(), spread.std()


def manual_backtest(
    df: pd.DataFrame,
    rolling_window: int,
    z_threshold: float,
    z_exit: float,
    commission_pct: float,
    slippage_pct: float,
    capital_at_risk: float,
    stop_loss_multiplier: float,
    cooldown_periods: int,
    half_life: float | None = None,
    time_stop_multiplier: float | None = None,
) -> pd.DataFrame:
    """Эталонная реализация логики бэктеста для проверки."""
    df = df.copy()
    y_col, x_col = df.columns[0], df.columns[1]

    df["beta"] = np.nan
    df["mean"] = np.nan
    df["std"] = np.nan
    df["spread"] = np.nan
    df["z_score"] = np.nan

    for i in range(rolling_window, len(df)):
        window = df.iloc[i - rolling_window : i]
        y_win = window[y_col]
        x_win = window[x_col]
        x_const = sm.add_constant(x_win)
        model = sm.OLS(y_win, x_const).fit()
        beta = model.params.iloc[1]
        spread_win = y_win - beta * x_win
        mean = spread_win.mean()
        std = spread_win.std()
        if std < 1e-6:
            continue
        curr_spread = df[y_col].iat[i] - beta * df[x_col].iat[i]
        z = (curr_spread - mean) / std
        df.loc[df.index[i], ["beta", "mean", "std", "spread", "z_score"]] = [
            beta,
            mean,
            std,
            curr_spread,
            z,
        ]

    df["position"] = 0.0
    df["trades"] = 0.0
    df["costs"] = 0.0
    df["pnl"] = 0.0

    position = 0.0
    entry_z = 0.0
    stop_loss_z = 0.0
    cooldown_remaining = 0
    entry_index = 0

    # Вынесем get_loc вычисления из цикла для оптимизации
    position_col_idx = df.columns.get_loc("position")
    trades_col_idx = df.columns.get_loc("trades")
    costs_col_idx = df.columns.get_loc("costs")
    pnl_col_idx = df.columns.get_loc("pnl")

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

        if (
            position != 0
            and half_life is not None
            and time_stop_multiplier is not None
        ):
            trade_duration = i - entry_index
            time_stop_limit = half_life * time_stop_multiplier
            if trade_duration > time_stop_limit:
                new_position = 0.0
                cooldown_remaining = cooldown_periods

        # Уменьшаем cooldown счетчик
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        # Закрытие позиций в конце теста для чистоты метрик
        if i == len(df) - 1 and position != 0:
            new_position = 0.0  # Форс-закрытие в последнем периоде
            cooldown_remaining = cooldown_periods
        elif position > 0 and z_curr <= stop_loss_z:
            new_position = 0.0
            cooldown_remaining = cooldown_periods
        elif position < 0 and z_curr >= stop_loss_z:
            new_position = 0.0
            cooldown_remaining = cooldown_periods
        # Z-score exit: закрываем позицию если z-score вернулся к нулю
        elif position != 0 and abs(z_curr) <= z_exit:
            new_position = 0.0
            cooldown_remaining = cooldown_periods

        # Проверяем сигналы входа только если не в последнем периоде и не в cooldown
        if i < len(df) - 1 and cooldown_remaining == 0:
            signal = 0
            if z_curr > z_threshold:
                signal = -1
            elif z_curr < -z_threshold:
                signal = 1

            z_prev = df["z_score"].iat[i - 1]
            long_confirmation = (signal == 1) and (z_curr > z_prev)
            short_confirmation = (signal == -1) and (z_curr < z_prev)

            if new_position == 0 and (long_confirmation or short_confirmation):
                entry_z = z_curr
                stop_loss_z = float(np.sign(entry_z) * stop_loss_multiplier)
                stop_loss_price = mean + stop_loss_z * std
                risk_per_unit = abs(spread_curr - stop_loss_price)
                trade_value = df[y_col].iat[i] + abs(beta) * df[x_col].iat[i]
                size_risk = (
                    capital_at_risk / risk_per_unit if risk_per_unit != 0 else 0.0
                )
                size_value = capital_at_risk / trade_value if trade_value != 0 else 0.0
                size = min(size_risk, size_value)
                new_position = signal * size

            elif (
                new_position != 0
                and (long_confirmation or short_confirmation)
                and np.sign(new_position) != signal
            ):

                entry_z = z_curr
                stop_loss_z = float(np.sign(entry_z) * stop_loss_multiplier)
                stop_loss_price = mean + stop_loss_z * std
                risk_per_unit = abs(spread_curr - stop_loss_price)
                trade_value = df[y_col].iat[i] + abs(beta) * df[x_col].iat[i]
                size_risk = (
                    capital_at_risk / risk_per_unit if risk_per_unit != 0 else 0.0
                )
                size_value = capital_at_risk / trade_value if trade_value != 0 else 0.0
                size = min(size_risk, size_value)
                new_position = signal * size
                entry_index = i

        trades = abs(new_position - position)
        trade_value = df[y_col].iat[i] + abs(beta) * df[x_col].iat[i]

        price_s1 = df[y_col].iat[i]
        price_s2 = df[x_col].iat[i]
        position_s1_change = new_position - position
        position_s2_change = -new_position * beta - (-position * beta)

        notional_change_s1 = abs(position_s1_change * price_s1)
        notional_change_s2 = abs(position_s2_change * price_s2)

        commission = (notional_change_s1 + notional_change_s2) * commission_pct
        slippage = (notional_change_s1 + notional_change_s2) * slippage_pct
        costs = commission + slippage

        df.iat[i, position_col_idx] = new_position
        df.iat[i, trades_col_idx] = trades
        df.iat[i, costs_col_idx] = costs
        df.iat[i, pnl_col_idx] = pnl - costs

        position = new_position

    df["cumulative_pnl"] = df["pnl"].cumsum()
    return df


def test_backtester_outputs():
    """Проверяет, что каждый столбец и метрика бэктестера совпадают с эталоном."""
    np.random.seed(0)
    # Используем произвольные имена колонок для проверки надежности
    data = pd.DataFrame(
        {
            "ASSET_Y": np.linspace(1, 20, 20) + np.random.normal(0, 0.5, size=20),
            "ASSET_X": np.linspace(1, 20, 20),
        }
    )

    z_threshold = 1.0
    commission = 0.001
    slippage = 0.0005
    annualizing_factor = 365

    rolling_window = 3

    bt = PairBacktester(
        data,
        rolling_window=rolling_window,
        z_threshold=z_threshold,
        z_exit=0.0,
        commission_pct=commission,
        slippage_pct=slippage,
        annualizing_factor=annualizing_factor,
        capital_at_risk=100.0,
        stop_loss_multiplier=2.0,
        cooldown_periods=0,
    )
    bt.run()
    result = bt.get_results()

    result_df = pd.DataFrame(
        {
            "spread": result["spread"],
            "z_score": result["z_score"],
            "position": result["position"],
            "pnl": result["pnl"],
            "cumulative_pnl": result["cumulative_pnl"],
        }
    )

    # Сравниваем с эталоном
    expected = manual_backtest(
        data,
        rolling_window,
        1.0,
        0.0,
        commission_pct=commission,
        slippage_pct=slippage,
        capital_at_risk=100.0,
        stop_loss_multiplier=2.0,
        cooldown_periods=0,
    )
    expected_for_comparison = expected[
        ["spread", "z_score", "position", "pnl", "cumulative_pnl"]
    ]

    pd.testing.assert_frame_equal(result_df, expected_for_comparison)
    assert isinstance(result["trades_log"], list)

    # Проверяем метрики
    metrics = bt.get_performance_metrics()

    expected_pnl = expected["pnl"].dropna()
    expected_cum_pnl = expected["cumulative_pnl"].dropna()
    expected_metrics = {
        "sharpe_ratio": performance.sharpe_ratio(expected_pnl, annualizing_factor),
        "max_drawdown": performance.max_drawdown(expected_cum_pnl),
        "total_pnl": expected_cum_pnl.iloc[-1] if not expected_cum_pnl.empty else 0.0,
    }

    # Надежное сравнение словарей с float-числами
    assert metrics.keys() == expected_metrics.keys()
    assert np.isclose(metrics["sharpe_ratio"], expected_metrics["sharpe_ratio"])
    assert np.isclose(metrics["max_drawdown"], expected_metrics["max_drawdown"])
    assert np.isclose(metrics["total_pnl"], expected_metrics["total_pnl"])


def test_zero_std_handling() -> None:
    """Проверяет корректность работы при нулевом стандартном отклонении спреда."""
    data = pd.DataFrame({"Y": 2 * np.arange(1, 11), "X": np.arange(1, 11)})

    rolling_window = 3

    beta, mean, std = calc_params(data)
    assert std == 0

    bt = PairBacktester(
        data,
        rolling_window=rolling_window,
        z_threshold=1.0,
        z_exit=0.0,
        commission_pct=0.001,
        slippage_pct=0.0005,
        capital_at_risk=100.0,
        stop_loss_multiplier=2.0,
        cooldown_periods=0,
    )
    bt.run()
    result = bt.get_results()

    result_df = pd.DataFrame(
        {
            "spread": result["spread"],
            "z_score": result["z_score"],
            "position": result["position"],
            "pnl": result["pnl"],
            "cumulative_pnl": result["cumulative_pnl"],
        }
    )

    expected = manual_backtest(
        data,
        rolling_window,
        1.0,
        0.0,
        commission_pct=0.001,
        slippage_pct=0.0005,
        capital_at_risk=100.0,
        stop_loss_multiplier=2.0,
        cooldown_periods=0,
    )
    expected_for_comparison = expected[
        ["spread", "z_score", "position", "pnl", "cumulative_pnl"]
    ]

    pd.testing.assert_frame_equal(result_df, expected_for_comparison)
    assert isinstance(result["trades_log"], list)

    metrics = bt.get_performance_metrics()
    assert metrics == {"sharpe_ratio": 0.0, "max_drawdown": 0.0, "total_pnl": 0.0}


def test_step_pnl_includes_costs() -> None:
    """Ensure step PnL subtracts trading costs for each period."""
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "Y": np.linspace(1, 10, 10) + np.random.normal(0, 0.1, size=10),
            "X": np.linspace(1, 10, 10),
        }
    )

    bt = PairBacktester(
        data,
        rolling_window=3,
        z_threshold=0.5,
        z_exit=0.0,
        commission_pct=0.001,
        slippage_pct=0.0005,
        capital_at_risk=50.0,
        stop_loss_multiplier=2.0,
        cooldown_periods=0,
    )
    bt.run()
    result = bt.get_results()
    df = pd.DataFrame(
        {
            "spread": result["spread"],
            "position": result["position"],
            "pnl": result["pnl"],
            "costs": result["costs"],
        }
    )

    df = df.dropna()

    for i in range(1, len(df)):
        position_prev = df["position"].iloc[i - 1]
        spread_diff = df["spread"].iloc[i] - df["spread"].iloc[i - 1]
        expected_step_pnl = position_prev * spread_diff - df["costs"].iloc[i]
        assert np.isclose(expected_step_pnl, df["pnl"].iloc[i])

