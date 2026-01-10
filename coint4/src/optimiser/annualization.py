"""
Единый модуль для расчета annualization factor для криптовалют.
Обеспечивает консистентность между всеми компонентами системы.
"""

import numpy as np
from typing import Union


# Константы для криптовалютного рынка
CRYPTO_TRADING_DAYS_PER_YEAR = 365  # Крипто торгуется 365 дней в году
BARS_PER_DAY_15MIN = 96  # 24 часа * 4 (15-минутных бара в часе)
BARS_PER_YEAR_15MIN = CRYPTO_TRADING_DAYS_PER_YEAR * BARS_PER_DAY_15MIN  # 35040


def get_annualization_factor(frequency: str = "15min", return_type: str = "sharpe") -> float:
    """
    Возвращает коэффициент для приведения к годовой доходности.
    
    Args:
        frequency: Частота данных ("15min", "1h", "4h", "1d")
        return_type: Тип расчета ("sharpe" для коэффициента Шарпа, "returns" для доходности)
        
    Returns:
        Коэффициент для annualization
        
    Примеры:
        - Для 15-мин данных и Sharpe: sqrt(96 * 365) = ~187
        - Для дневных данных и Sharpe: sqrt(365) = ~19.1
        - Для 15-мин данных и returns: 96 * 365 = 35040
        - Для дневных данных и returns: 365
    """
    # Количество баров в году для каждой частоты
    bars_per_year = {
        "15min": BARS_PER_YEAR_15MIN,  # 35040
        "30min": CRYPTO_TRADING_DAYS_PER_YEAR * 48,  # 17520
        "1h": CRYPTO_TRADING_DAYS_PER_YEAR * 24,  # 8760
        "2h": CRYPTO_TRADING_DAYS_PER_YEAR * 12,  # 4380
        "4h": CRYPTO_TRADING_DAYS_PER_YEAR * 6,  # 2190
        "6h": CRYPTO_TRADING_DAYS_PER_YEAR * 4,  # 1460
        "12h": CRYPTO_TRADING_DAYS_PER_YEAR * 2,  # 730
        "1d": CRYPTO_TRADING_DAYS_PER_YEAR,  # 365
    }
    
    if frequency not in bars_per_year:
        raise ValueError(f"Неподдерживаемая частота: {frequency}. "
                        f"Доступные: {list(bars_per_year.keys())}")
    
    annual_bars = bars_per_year[frequency]
    
    if return_type == "sharpe":
        # Для коэффициента Шарпа используем квадратный корень
        return np.sqrt(annual_bars)
    elif return_type == "returns":
        # Для доходности используем простое умножение
        return annual_bars
    else:
        raise ValueError(f"Неподдерживаемый тип: {return_type}. "
                        f"Доступные: 'sharpe', 'returns'")


def calculate_sharpe_ratio(
    returns: Union[list, np.ndarray],
    frequency: str = "15min",
    risk_free_rate: float = 0.0
) -> float:
    """
    Рассчитывает годовой коэффициент Шарпа для криптовалютных данных.
    
    Args:
        returns: Массив доходностей
        frequency: Частота данных ("15min", "1h", "4h", "1d")
        risk_free_rate: Безрисковая ставка (годовая)
        
    Returns:
        Годовой коэффициент Шарпа
    """
    returns = np.array(returns)
    
    if len(returns) == 0:
        return 0.0
    
    # Средняя доходность
    mean_return = np.mean(returns)
    
    # Стандартное отклонение
    std_return = np.std(returns, ddof=1)
    
    if std_return == 0:
        return 0.0
    
    # Коэффициент annualization
    ann_factor = get_annualization_factor(frequency, "sharpe")
    
    # Приводим безрисковую ставку к частоте данных
    bars_per_year = get_annualization_factor(frequency, "returns")
    risk_free_per_period = risk_free_rate / bars_per_year
    
    # Коэффициент Шарпа
    sharpe = (mean_return - risk_free_per_period) / std_return * ann_factor
    
    return sharpe


def validate_sharpe_ratio(
    sharpe: float,
    frequency: str = "15min",
    num_periods: int = None
) -> tuple[bool, str]:
    """
    Валидирует коэффициент Шарпа на реалистичность.
    
    Args:
        sharpe: Коэффициент Шарпа для проверки
        frequency: Частота данных
        num_periods: Количество периодов в выборке (для проверки статистической значимости)
        
    Returns:
        (is_valid, message) - валиден ли Sharpe и сообщение
    """
    # Границы реалистичности для годового Sharpe в крипто
    MIN_REALISTIC_SHARPE = -5.0
    MAX_REALISTIC_SHARPE = 10.0
    
    # Для высокочастотных данных границы могут быть выше
    if frequency in ["15min", "30min", "1h"]:
        MAX_REALISTIC_SHARPE = 15.0
    
    if sharpe < MIN_REALISTIC_SHARPE:
        return False, f"Sharpe {sharpe:.2f} слишком низкий (< {MIN_REALISTIC_SHARPE})"
    
    if sharpe > MAX_REALISTIC_SHARPE:
        return False, f"Sharpe {sharpe:.2f} слишком высокий (> {MAX_REALISTIC_SHARPE})"
    
    # Проверка статистической значимости если известно количество периодов
    if num_periods is not None and num_periods > 0:
        # Стандартная ошибка Sharpe ratio
        se_sharpe = np.sqrt((1 + 0.5 * sharpe**2) / num_periods)
        
        # t-статистика для проверки значимости
        t_stat = abs(sharpe) / se_sharpe
        
        # Для 95% уровня значимости t-критическое ~ 1.96
        if t_stat < 1.96:
            return True, f"Sharpe {sharpe:.2f} статистически незначим (t={t_stat:.2f} < 1.96)"
    
    return True, f"Sharpe {sharpe:.2f} в пределах нормы"


def get_default_frequency_from_config(config: dict) -> str:
    """
    Извлекает частоту данных из конфигурации.
    
    Args:
        config: Словарь конфигурации
        
    Returns:
        Строка с частотой ("15min", "1h", etc.)
    """
    # Проверяем разные возможные места в конфиге
    if "data" in config and "timeframe" in config["data"]:
        return config["data"]["timeframe"]
    
    if "backtest" in config and "bar_duration_minutes" in config["backtest"]:
        minutes = config["backtest"]["bar_duration_minutes"]
        if minutes == 15:
            return "15min"
        elif minutes == 30:
            return "30min"
        elif minutes == 60:
            return "1h"
        elif minutes == 240:
            return "4h"
        elif minutes == 1440:
            return "1d"
    
    # По умолчанию предполагаем 15-минутные бары
    return "15min"