from .base_engine import BasePairBacktester
from ..core.numba_kernels import calculate_positions_and_pnl_full, rolling_ols
import numpy as np
import pandas as pd


class NumbaPairBacktester(BasePairBacktester):
    """Numba-оптимизированная версия BasePairBacktester.
    
    Наследует всю логику от BasePairBacktester, но переопределяет метод run()
    для использования быстрых Numba-функций.
    """
    
    def run(self) -> None:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Запуск Numba-оптимизированного бэктеста для {self.pair_name}")

        if self.pair_data.empty or len(self.pair_data.columns) < 2 or len(self.pair_data) < self.rolling_window + 2:
            logger.warning("Недостаточно данных для Numba бэктеста, результаты будут пустыми.")
            self.results = self._create_empty_results_df()
            return

        y = self.pair_data.iloc[:, 0].values.astype(np.float32)
        x = self.pair_data.iloc[:, 1].values.astype(np.float32)

        beta, mu, sigma = rolling_ols(y, x, self.rolling_window)
        
        positions, pnl, cumulative_pnl, costs = calculate_positions_and_pnl_full(
            y, x,
            rolling_window=self.rolling_window,
            entry_threshold=self.zscore_entry_threshold,
            exit_threshold=self.z_exit,
            commission=self.commission_pct,
            slippage=self.slippage_pct,
            max_holding_period=99999,
            enable_regime_detection=self.market_regime_detection,
            enable_structural_breaks=self.structural_break_protection,
            min_volatility=self.min_volatility,
            adaptive_threshold_factor=1.0 if self.adaptive_thresholds else 0.0
        )

        spread = y - beta * x
        z_scores = np.full_like(spread, np.nan)
        valid_sigma = sigma > 1e-12
        z_scores[valid_sigma] = (spread[valid_sigma] - mu[valid_sigma]) / sigma[valid_sigma]

        self.results = pd.DataFrame({
            "spread": spread,
            "z_score": z_scores,
            "position": positions,
            "pnl": pnl,
            "cumulative_pnl": cumulative_pnl,
            "beta": beta,
            "mean": mu,
            "std": sigma,
        }, index=self.pair_data.index)
        
        for col in ['trades', 'costs', 'commission_costs', 'slippage_costs', 'bid_ask_costs', 'impact_costs']:
            self.results[col] = 0.0

        self.results["costs"] = costs
        total_cost_pct = self.commission_pct + self.slippage_pct
        if total_cost_pct > 0:
            self.results["commission_costs"] = costs * (self.commission_pct / total_cost_pct)
            self.results["slippage_costs"] = costs * (self.slippage_pct / total_cost_pct)
        
        trades_mask = self.results['position'].diff() != 0
        self.results.loc[trades_mask, 'trades'] = 1.0

        # ДИАГНОСТИКА: Подсчитываем сделки
        trades_count = trades_mask.sum()

        if hasattr(self, 'pair_name') and 'HVHUSDT' in self.pair_name:  # Логируем только первую пару
            logger.info(f"🔍 ДИАГНОСТИКА {self.pair_name}:")
            logger.info(f"   entry_threshold: {self.zscore_entry_threshold}")
            logger.info(f"   exit_threshold: {self.z_exit}")
            logger.info(f"   Макс |z_score|: {np.nanmax(np.abs(z_scores)):.4f}")
            logger.info(f"   Позиций != 0: {np.sum(positions != 0)}")
            logger.info(f"   Изменений позиций: {trades_count}")
            logger.info(f"   PnL сумма: {pnl.sum():.4f}")

            # Показываем первые несколько z_scores
            valid_z = z_scores[~np.isnan(z_scores)]
            if len(valid_z) > 0:
                logger.info(f"   Первые 10 z_scores: {valid_z[:10]}")
                logger.info(f"   Превышают порог: {np.sum(np.abs(valid_z) > self.zscore_entry_threshold)}")

        # Обновляем портфель (метод update_pnl не существует, убираем)
        # if hasattr(self, 'portfolio') and self.portfolio is not None:
        #     self.portfolio.update_pnl(pnl.sum())
    
    def _create_empty_results_df(self) -> pd.DataFrame:
        """Создает пустой DataFrame с результатами."""
        return pd.DataFrame(
            columns=["spread", "z_score", "position", "pnl", "cumulative_pnl", 
                    "beta", "mean", "std", "trades", "costs", "commission_costs", 
                    "slippage_costs", "bid_ask_costs", "impact_costs"]
        )
