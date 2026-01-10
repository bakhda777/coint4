"""
Референс-ядро для pairs trading backtester.
Минимальная, предельно прозрачная реализация в ~200 строк.
Используется для проверки гипотез и изоляции багов.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import warnings


class ReferenceEngine:
    """
    Минимальный backtester для pairs trading.
    Без Numba, без оптимизаций, максимальная прозрачность.
    """
    
    def __init__(
        self,
        rolling_window: int = 60,
        z_enter: float = 2.0,
        z_exit: float = 0.5,
        max_holding_period: int = 100,
        commission_pct: float = 0.0004,
        slippage_pct: float = 0.0005,
        verbose: bool = True
    ):
        """
        Args:
            rolling_window: Окно для расчета rolling статистик
            z_enter: Порог входа по z-score (абсолютное значение)
            z_exit: Порог выхода по z-score (абсолютное значение)
            max_holding_period: Максимальный период удержания позиции
            commission_pct: Комиссия в процентах от объема
            slippage_pct: Проскальзывание в процентах
            verbose: Выводить ли диагностику
        """
        # Валидация параметров
        assert z_exit < z_enter - 0.2, f"z_exit ({z_exit}) должен быть меньше z_enter ({z_enter}) минимум на 0.2"
        assert rolling_window >= 10, f"rolling_window ({rolling_window}) должен быть >= 10"
        assert max_holding_period > 0, f"max_holding_period ({max_holding_period}) должен быть > 0"
        
        self.rolling_window = rolling_window
        self.z_enter = z_enter
        self.z_exit = z_exit
        self.max_holding_period = max_holding_period
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.verbose = verbose
        
    def backtest(
        self, 
        data: pd.DataFrame,
        symbol1_col: str = 'symbol1',
        symbol2_col: str = 'symbol2'
    ) -> Dict:
        """
        Запуск бэктеста на данных.
        
        Args:
            data: DataFrame с ценами двух инструментов
            symbol1_col: Название колонки первого инструмента
            symbol2_col: Название колонки второго инструмента
            
        Returns:
            Словарь с результатами
        """
        # Подготовка данных
        df = data[[symbol1_col, symbol2_col]].copy()
        df = df.dropna()
        
        # Валидация данных
        assert len(df) > self.rolling_window * 2, \
            f"Недостаточно данных: {len(df)} < {self.rolling_window * 2}"
        
        y = df[symbol1_col].values.astype(float)
        x = df[symbol2_col].values.astype(float)
        n = len(df)
        
        # Расчет rolling OLS: y = alpha + beta * x
        y_series = pd.Series(y, index=df.index)
        x_series = pd.Series(x, index=df.index)
        
        # Beta = Cov(x,y) / Var(x)
        beta = (x_series.rolling(self.rolling_window).cov(y_series) / 
                x_series.rolling(self.rolling_window).var()).values
        
        # Alpha = Mean(y) - Beta * Mean(x)
        alpha = (y_series.rolling(self.rolling_window).mean() - 
                 beta * x_series.rolling(self.rolling_window).mean()).values
        
        # Spread = y - (alpha + beta * x)
        spread = np.full(n, np.nan)
        for i in range(n):
            if np.isfinite(alpha[i]) and np.isfinite(beta[i]):
                spread[i] = y[i] - (alpha[i] + beta[i] * x[i])
        
        # Z-score нормализация
        spread_series = pd.Series(spread, index=df.index)
        mu = spread_series.rolling(self.rolling_window).mean().values
        sigma = spread_series.rolling(self.rolling_window).std().values
        
        z_scores = np.full(n, np.nan)
        for i in range(n):
            if np.isfinite(mu[i]) and np.isfinite(sigma[i]) and sigma[i] > 1e-8:
                z_scores[i] = (spread[i] - mu[i]) / sigma[i]
        
        # Торговая логика
        positions = np.zeros(n, dtype=int)  # -1, 0, +1
        holding_period = 0
        entries = []
        exits = []
        
        for i in range(self.rolling_window, n):
            # Пропускаем если z-score невалидный
            if not np.isfinite(z_scores[i]):
                positions[i] = positions[i-1] if i > 0 else 0
                continue
            
            current_z = z_scores[i]
            prev_pos = positions[i-1] if i > 0 else 0
            
            # ВЫХОД (приоритет над входом)
            if prev_pos != 0:
                should_exit = False
                exit_reason = ""
                
                # Выход по z-score
                if abs(current_z) <= self.z_exit:
                    should_exit = True
                    exit_reason = f"z-score ({current_z:.2f}) <= exit ({self.z_exit})"
                
                # Выход по времени
                elif holding_period >= self.max_holding_period:
                    should_exit = True
                    exit_reason = f"max holding period ({self.max_holding_period})"
                
                if should_exit:
                    positions[i] = 0
                    holding_period = 0
                    exits.append((i, exit_reason))
                else:
                    positions[i] = prev_pos
                    holding_period += 1
            
            # ВХОД (только если не в позиции)
            elif prev_pos == 0:
                if current_z > self.z_enter:
                    positions[i] = -1  # Short spread (short y, long x)
                    holding_period = 1
                    entries.append((i, f"z-score ({current_z:.2f}) > enter ({self.z_enter})"))
                    
                elif current_z < -self.z_enter:
                    positions[i] = 1  # Long spread (long y, short x)
                    holding_period = 1
                    entries.append((i, f"z-score ({current_z:.2f}) < -enter ({-self.z_enter})"))
                    
                else:
                    positions[i] = 0
            
            # УДЕРЖАНИЕ
            else:
                positions[i] = prev_pos
                if prev_pos != 0:
                    holding_period += 1
        
        # Расчет PnL
        spread_returns = np.diff(spread, prepend=spread[0])
        pnl = positions * spread_returns
        
        # Учет издержек
        position_changes = np.diff(positions, prepend=0)
        trade_costs = np.abs(position_changes) * (self.commission_pct + self.slippage_pct) * 100  # Примерная стоимость
        net_pnl = pnl - trade_costs
        
        # Расчет метрик
        total_pnl = np.nansum(net_pnl)
        num_trades = np.sum(np.abs(position_changes) > 0)
        
        # Sharpe ratio (упрощенный)
        # Используем все returns, включая нулевые
        returns = net_pnl[self.rolling_window:]  # Пропускаем warmup период
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 96)  # Annualized для 15min bars
        else:
            sharpe = 0
        
        # Диагностика
        if self.verbose:
            print("=" * 60)
            print("РЕФЕРЕНС-ЯДРО: ДИАГНОСТИКА")
            print("=" * 60)
            print(f"Данные: {len(df)} баров")
            print(f"Rolling window: {self.rolling_window}")
            print(f"Z-score thresholds: enter={self.z_enter}, exit={self.z_exit}")
            print()
            print(f"Z-score статистика:")
            print(f"  Max |z-score|: {np.nanmax(np.abs(z_scores)):.2f}")
            print(f"  Точек где |z| > {self.z_enter}: {np.sum(np.abs(z_scores) > self.z_enter)}")
            print(f"  Процент времени |z| > {self.z_enter}: {np.sum(np.abs(z_scores) > self.z_enter) / len(z_scores) * 100:.1f}%")
            print()
            print(f"Торговая активность:")
            print(f"  Количество входов: {len(entries)}")
            print(f"  Количество выходов: {len(exits)}")
            print(f"  Всего смен позиций: {num_trades}")
            print(f"  Процент времени в позиции: {np.sum(positions != 0) / len(positions) * 100:.1f}%")
            print()
            print(f"Результаты:")
            print(f"  Total PnL: ${total_pnl:.2f}")
            print(f"  Sharpe ratio: {sharpe:.3f}")
            print("=" * 60)
        
        return {
            'data': df,
            'spread': spread,
            'z_scores': z_scores,
            'positions': positions,
            'pnl': net_pnl,
            'total_pnl': total_pnl,
            'num_trades': num_trades,
            'sharpe_ratio': sharpe,
            'entries': entries,
            'exits': exits,
            'beta': beta,
            'alpha': alpha,
            'mu': mu,
            'sigma': sigma
        }


def make_synthetic_pair(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Создает синтетическую коинтегрированную пару для тестирования.
    
    Args:
        n: Количество точек
        seed: Seed для воспроизводимости
        
    Returns:
        DataFrame с колонками ['symbol1', 'symbol2']
    """
    np.random.seed(seed)
    
    # Генерируем случайное блуждание для x
    x = np.cumsum(np.random.randn(n) * 0.01) + 100
    
    # y коинтегрирован с x: y = beta * x + стационарный шум
    beta = 1.2
    noise = np.random.randn(n) * 0.5
    # Добавляем mean-reverting компонент
    ar_component = np.zeros(n)
    for i in range(1, n):
        ar_component[i] = 0.7 * ar_component[i-1] + np.random.randn() * 0.3
    
    y = beta * x + ar_component + noise
    
    # Создаем DataFrame
    dates = pd.date_range('2023-01-01', periods=n, freq='15min')
    df = pd.DataFrame({
        'symbol1': y,
        'symbol2': x
    }, index=dates)
    
    return df


if __name__ == "__main__":
    # Тест на синтетических данных
    print("ТЕСТ НА СИНТЕТИЧЕСКИХ ДАННЫХ")
    print("-" * 60)
    
    # Создаем синтетическую пару
    synthetic_data = make_synthetic_pair(n=2000)
    
    # Создаем движок
    engine = ReferenceEngine(
        rolling_window=60,
        z_enter=2.0,
        z_exit=0.5,
        max_holding_period=100,
        commission_pct=0.0004,
        slippage_pct=0.0005,
        verbose=True
    )
    
    # Запускаем бэктест
    results = engine.backtest(synthetic_data)
    
    # Проверка что есть сделки
    assert results['num_trades'] > 0, "❌ КРИТИЧЕСКАЯ ОШИБКА: Нет сделок на синтетических данных!"
    print("\n✅ ТЕСТ ПРОЙДЕН: Сделки генерируются на синтетических данных")