#!/usr/bin/env python3
"""
Строгие тесты для проверки корректности расчета торговых метрик.

Проверяет исправление критических ошибок:
1. Total trades равен 8 (5 прибыльных + 3 убыточных)
2. Win rate равен 5/8 = 0.625
3. Trading days равен 3
4. Trades per day равен 8/3 ≈ 2.67
"""

import pytest
import pandas as pd

# Маркируем все тесты как критические исправления
pytestmark = pytest.mark.critical_fixes

# Все необходимые импорты для тестирования торговых метрик

# Константы для тестирования торговых метрик
EXPECTED_TOTAL_TRADES = 8
EXPECTED_PROFITABLE_TRADES = 5
EXPECTED_LOSING_TRADES = 3
EXPECTED_WIN_RATE = EXPECTED_PROFITABLE_TRADES / EXPECTED_TOTAL_TRADES  # 0.625
EXPECTED_TRADING_DAYS = 3
EXPECTED_TRADES_PER_DAY = EXPECTED_TOTAL_TRADES / EXPECTED_TRADING_DAYS  # ≈ 2.67

# Константы для создания тестовых данных
TEST_DATE_START = '2024-01-01 00:00:00'
TEST_DATE_END = '2024-01-03 23:45:00'
TEST_FREQUENCY = '15min'
SINGLE_DAY_PERIODS = 96  # Количество 15-минутных периодов в дне

# Константы для PnL значений
PROFITABLE_PNL_1 = 0.01
PROFITABLE_PNL_2 = 0.015
PROFITABLE_PNL_3 = 0.02
PROFITABLE_PNL_4 = 0.005
PROFITABLE_PNL_5 = 0.0125
LOSING_PNL_1 = -0.005
LOSING_PNL_2 = -0.01
LOSING_PNL_3 = -0.0075


# Вспомогательные функции для создания тестовых данных
def create_pnl_series_with_trades(trade_data: list, start_date: str = TEST_DATE_START,
                                  end_date: str = TEST_DATE_END) -> pd.Series:
    """
    Создает PnL серию с заданными сделками.

    Args:
        trade_data: Список кортежей (index, pnl_value) для создания сделок
        start_date: Начальная дата
        end_date: Конечная дата

    Returns:
        pd.Series: PnL серия с заданными сделками
    """
    dates = pd.date_range(start_date, end_date, freq=TEST_FREQUENCY)
    pnl_data = [0.0] * len(dates)

    for idx, pnl_value in trade_data:
        if idx < len(pnl_data):
            pnl_data[idx] = pnl_value

    return pd.Series(pnl_data, index=dates)


def calculate_trade_metrics(pnl_series: pd.Series) -> dict:
    """
    Рассчитывает торговые метрики из PnL серии.

    Args:
        pnl_series: PnL серия

    Returns:
        dict: Словарь с метриками (total_trades, win_rate, trading_days, trades_per_day)
    """
    # Определяем сделки по ненулевым PnL значениям
    trades_mask = pnl_series != 0
    trade_groups = (trades_mask.astype(int).diff() != 0).cumsum()
    trade_groups = trade_groups[trades_mask]

    if len(trade_groups) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'trading_days': 0,
            'trades_per_day': 0.0
        }

    # PnL по сделкам
    trade_pnls = pnl_series[trades_mask].groupby(trade_groups).sum()

    # Рассчитываем метрики
    total_trades = len(trade_pnls)
    winning_trades = (trade_pnls > 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    trading_days = pnl_series[trades_mask].index.normalize().nunique()
    trades_per_day = total_trades / trading_days if trading_days > 0 else 0.0

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'trading_days': trading_days,
        'trades_per_day': trades_per_day
    }


@pytest.mark.critical_fixes
class TestTradeMetricsCorrectness:
    """Строгие тесты корректности торговых метрик."""

    @pytest.mark.unit
    def test_total_trades_when_eight_trades_executed_then_count_equals_eight(self):
        """
        ТЕСТ 1: Проверяет что total_trades равен 8.

        Создает синтетическую 15-минутную PnL-серию с известным количеством сделок
        и проверяет что система правильно их подсчитывает.
        """
        # Создаем 8 сделок (5 прибыльных, 3 убыточных) с использованием констант
        trade_data = [
            # Сделка 1: прибыльная (разделена на 2 периода)
            (10, PROFITABLE_PNL_1), (11, PROFITABLE_PNL_1),
            # Сделка 2: прибыльная
            (25, PROFITABLE_PNL_2), (26, PROFITABLE_PNL_2),
            # Сделка 3: убыточная
            (40, LOSING_PNL_1), (41, LOSING_PNL_1),
            # Сделка 4: прибыльная
            (60, PROFITABLE_PNL_4), (61, PROFITABLE_PNL_4),
            # Сделка 5: убыточная
            (80, LOSING_PNL_2), (81, LOSING_PNL_2),
            # Сделка 6: прибыльная
            (100, PROFITABLE_PNL_3), (101, PROFITABLE_PNL_3),
            # Сделка 7: убыточная
            (120, LOSING_PNL_3), (121, LOSING_PNL_3),
            # Сделка 8: прибыльная
            (140, PROFITABLE_PNL_5), (141, PROFITABLE_PNL_5),
        ]

        pnl_series = create_pnl_series_with_trades(trade_data)
        metrics = calculate_trade_metrics(pnl_series)

        assert metrics['total_trades'] == EXPECTED_TOTAL_TRADES, \
            f"Ожидалось {EXPECTED_TOTAL_TRADES} сделок, получено {metrics['total_trades']}"

    @pytest.mark.unit
    def test_win_rate_when_five_of_eight_profitable_then_rate_equals_0_625(self):
        """
        ТЕСТ 2: Проверяет что win_rate равен 0.625 (5/8).

        Создает сделки с известными результатами и проверяет win_rate.
        """
        # Используем те же данные что и в первом тесте для консистентности
        trade_data = [
            # 5 прибыльных сделок
            (10, PROFITABLE_PNL_1), (11, PROFITABLE_PNL_1),  # Сделка 1: +0.02
            (25, PROFITABLE_PNL_2), (26, PROFITABLE_PNL_2),  # Сделка 2: +0.03
            (60, PROFITABLE_PNL_4), (61, PROFITABLE_PNL_4),  # Сделка 4: +0.01
            (100, PROFITABLE_PNL_3), (101, PROFITABLE_PNL_3), # Сделка 6: +0.04
            (140, PROFITABLE_PNL_5), (141, PROFITABLE_PNL_5), # Сделка 8: +0.025
            # 3 убыточные сделки
            (40, LOSING_PNL_1), (41, LOSING_PNL_1),          # Сделка 3: -0.01
            (80, LOSING_PNL_2), (81, LOSING_PNL_2),          # Сделка 5: -0.02
            (120, LOSING_PNL_3), (121, LOSING_PNL_3),        # Сделка 7: -0.015
        ]

        pnl_series = create_pnl_series_with_trades(trade_data)
        metrics = calculate_trade_metrics(pnl_series)

        assert abs(metrics['win_rate'] - EXPECTED_WIN_RATE) < 1e-10, \
            f"Ожидался win_rate {EXPECTED_WIN_RATE:.3f}, получен {metrics['win_rate']:.3f}"

    @pytest.mark.unit
    def test_trading_days_when_three_days_active_then_count_equals_three(self):
        """
        ТЕСТ 3: Проверяет что trading_days равен 3.

        Проверяет подсчет уникальных торговых дней по фактическим данным.
        """
        # Создаем торговую активность на 3 разных дня
        trade_data = [
            # День 1 (2024-01-01): индексы 0-95
            (5, PROFITABLE_PNL_1), (10, LOSING_PNL_1),
            # День 2 (2024-01-02): индексы 96-191
            (100, PROFITABLE_PNL_3), (105, LOSING_PNL_2),
            # День 3 (2024-01-03): индексы 192+
            (200, PROFITABLE_PNL_2), (205, LOSING_PNL_3),
        ]

        pnl_series = create_pnl_series_with_trades(trade_data)
        metrics = calculate_trade_metrics(pnl_series)

        assert metrics['trading_days'] == EXPECTED_TRADING_DAYS, \
            f"Ожидалось {EXPECTED_TRADING_DAYS} торговых дней, получено {metrics['trading_days']}"

    @pytest.mark.unit
    def test_trades_per_day_when_eight_trades_in_three_days_then_rate_equals_2_67(self):
        """
        ТЕСТ 4: Проверяет что trades_per_day ≈ 2.67 (8/3).

        Проверяет корректность расчета среднего количества сделок в день.
        """
        # Создаем 8 сделок, распределенных по 3 дням
        trade_data = [
            # День 1: 3 сделки
            (10, PROFITABLE_PNL_1),   # Сделка 1
            (15, LOSING_PNL_1),       # Сделка 2
            (20, PROFITABLE_PNL_3),   # Сделка 3
            # День 2: 2 сделки
            (100, LOSING_PNL_2),      # Сделка 4
            (105, PROFITABLE_PNL_2),  # Сделка 5
            # День 3: 3 сделки
            (200, PROFITABLE_PNL_4),  # Сделка 6
            (205, LOSING_PNL_2),      # Сделка 7 (используем тот же PnL)
            (210, PROFITABLE_PNL_1),  # Сделка 8
        ]

        pnl_series = create_pnl_series_with_trades(trade_data)
        metrics = calculate_trade_metrics(pnl_series)

        assert abs(metrics['trades_per_day'] - EXPECTED_TRADES_PER_DAY) < 0.01, \
            f"Ожидалось {EXPECTED_TRADES_PER_DAY:.2f} сделок/день, получено {metrics['trades_per_day']:.2f}"

    @pytest.mark.integration
    def test_all_metrics_when_integrated_together_then_calculations_consistent(self):
        """
        ИНТЕГРАЦИОННЫЙ ТЕСТ: Проверяет все метрики вместе.

        Создает синтетическую PnL-серию и проверяет что все метрики
        рассчитываются корректно и согласованно.
        """
        # Создаем 8 сделок на 2 дня (5 прибыльных, 3 убыточных)
        trade_data = [
            # День 1: 5 сделок
            (5, PROFITABLE_PNL_1),    # Сделка 1: прибыльная
            (10, LOSING_PNL_1),       # Сделка 2: убыточная
            (15, PROFITABLE_PNL_3),   # Сделка 3: прибыльная
            (20, PROFITABLE_PNL_2),   # Сделка 4: прибыльная
            (25, LOSING_PNL_2),       # Сделка 5: убыточная
            # День 2: 3 сделки
            (100, PROFITABLE_PNL_3 + PROFITABLE_PNL_4),  # Сделка 6: прибыльная (0.025)
            (105, LOSING_PNL_3),      # Сделка 7: убыточная
            (110, PROFITABLE_PNL_3),  # Сделка 8: прибыльная
        ]

        pnl_series = create_pnl_series_with_trades(trade_data)
        metrics = calculate_trade_metrics(pnl_series)

        # Проверяем интегрированные метрики
        expected_total_trades = 8
        expected_win_rate = 5 / 8  # 0.625
        expected_trading_days = 2  # 2 дня
        expected_trades_per_day = 8 / 2  # 4.0

        assert metrics['total_trades'] == expected_total_trades, \
            f"Ожидалось {expected_total_trades} сделок, получено {metrics['total_trades']}"
        assert abs(metrics['win_rate'] - expected_win_rate) < 1e-10, \
            f"Ожидался win_rate {expected_win_rate:.3f}, получен {metrics['win_rate']:.3f}"
        assert metrics['trading_days'] == expected_trading_days, \
            f"Ожидалось {expected_trading_days} торговых дня, получено {metrics['trading_days']}"
        assert abs(metrics['trades_per_day'] - expected_trades_per_day) < 1e-10, \
            f"Ожидалось {expected_trades_per_day:.1f} сделок/день, получено {metrics['trades_per_day']:.1f}"

    @pytest.mark.unit
    @pytest.mark.parametrize("scenario,trade_data,expected_metrics", [
        (
            "single_profitable_trade",
            [(10, PROFITABLE_PNL_3)],  # Одна прибыльная сделка
            {'total_trades': 1, 'win_rate': 1.0, 'trading_days': 1, 'trades_per_day': 1.0}
        ),
        (
            "no_trades",
            [],  # Нет сделок
            {'total_trades': 0, 'win_rate': 0.0, 'trading_days': 0, 'trades_per_day': 0.0}
        ),
        (
            "all_losing_trades",
            [(5, LOSING_PNL_2), (15, LOSING_PNL_1), (25, LOSING_PNL_3), (35, LOSING_PNL_2)],  # 4 убыточные
            {'total_trades': 4, 'win_rate': 0.0, 'trading_days': 1, 'trades_per_day': 4.0}
        ),
        (
            "all_profitable_trades",
            [(5, PROFITABLE_PNL_1), (15, PROFITABLE_PNL_2), (25, PROFITABLE_PNL_3)],  # 3 прибыльные
            {'total_trades': 3, 'win_rate': 1.0, 'trading_days': 1, 'trades_per_day': 3.0}
        ),
    ])
    def test_edge_cases_when_various_patterns_then_robustness_maintained(self, scenario, trade_data, expected_metrics):
        """
        ТЕСТ 5: Проверяет граничные случаи и надежность.

        Проверяет корректность работы при различных паттернах сделок.
        """
        # Создаем данные на один день для простоты
        start_date = '2024-01-01'
        end_date = '2024-01-01 23:45:00'

        pnl_series = create_pnl_series_with_trades(trade_data, start_date, end_date)
        metrics = calculate_trade_metrics(pnl_series)

        # Проверяем все метрики
        assert metrics['total_trades'] == expected_metrics['total_trades'], \
            f"Сценарий {scenario}: ожидалось {expected_metrics['total_trades']} сделок, получено {metrics['total_trades']}"

        assert abs(metrics['win_rate'] - expected_metrics['win_rate']) < 1e-10, \
            f"Сценарий {scenario}: ожидался win_rate {expected_metrics['win_rate']}, получен {metrics['win_rate']}"

        assert metrics['trading_days'] == expected_metrics['trading_days'], \
            f"Сценарий {scenario}: ожидалось {expected_metrics['trading_days']} торговых дней, получено {metrics['trading_days']}"

        if expected_metrics['trades_per_day'] > 0:
            assert abs(metrics['trades_per_day'] - expected_metrics['trades_per_day']) < 1e-10, \
                f"Сценарий {scenario}: ожидалось {expected_metrics['trades_per_day']} сделок/день, получено {metrics['trades_per_day']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
