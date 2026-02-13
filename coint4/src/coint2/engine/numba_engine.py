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

        bar_minutes = 15
        if isinstance(self.pair_data.index, pd.DatetimeIndex) and len(self.pair_data.index) > 1:
            deltas = self.pair_data.index.to_series().diff().dropna()
            if not deltas.empty:
                median_seconds = deltas.dt.total_seconds().median()
                if median_seconds and median_seconds > 0:
                    bar_minutes = int(round(median_seconds / 60))

        min_hold_periods = 0
        if getattr(self, "min_position_hold_minutes", 0) > 0 and bar_minutes > 0:
            min_hold_periods = int(np.ceil(self.min_position_hold_minutes / bar_minutes))

        cooldown_periods = int(getattr(self, "cooldown_periods", 0) or 0)
        if getattr(self, "anti_churn_cooldown_minutes", 0) > 0 and bar_minutes > 0:
            anti_churn_periods = int(np.ceil(self.anti_churn_cooldown_minutes / bar_minutes))
            if anti_churn_periods > cooldown_periods:
                cooldown_periods = anti_churn_periods

        max_holding_period = 99999
        if getattr(self, "time_stop_multiplier", None) is not None and getattr(self, "half_life", None) is not None:
            try:
                time_stop_days = float(self.half_life) * float(self.time_stop_multiplier)
                if time_stop_days > 0 and bar_minutes > 0:
                    max_holding_period = int(np.ceil(time_stop_days * 1440 / bar_minutes))
                    if max_holding_period < 1:
                        max_holding_period = 1
            except (TypeError, ValueError):
                pass

        min_notional = 0.0
        max_notional = 0.0
        if getattr(self, "portfolio", None) is not None and getattr(self.portfolio, "config", None) is not None:
            min_notional = float(getattr(self.portfolio.config, "min_notional_per_trade", 0.0) or 0.0)
            max_notional = float(getattr(self.portfolio.config, "max_notional_per_trade", 0.0) or 0.0)

        capital_at_risk = float(getattr(self, "capital_at_risk", 0.0) or 0.0)
        
        positions, pnl, cumulative_pnl, costs = calculate_positions_and_pnl_full(
            y, x,
            beta, mu, sigma,
            rolling_window=self.rolling_window,
            entry_threshold=self.zscore_entry_threshold,
            exit_threshold=self.z_exit,
            commission=self.commission_pct,
            slippage=self.slippage_pct,
            max_holding_period=max_holding_period,
            enable_regime_detection=self.market_regime_detection,
            enable_structural_breaks=self.structural_break_protection,
            structural_break_min_correlation=float(getattr(self, "structural_break_min_correlation", 0.3) or 0.3),
            structural_break_entry_multiplier=float(getattr(self, "structural_break_entry_multiplier", 1.5) or 1.5),
            structural_break_exit_multiplier=float(getattr(self, "structural_break_exit_multiplier", 1.2) or 1.2),
            min_volatility=self.min_volatility,
            adaptive_threshold_factor=1.0 if self.adaptive_thresholds else 0.0,
            max_var_multiplier=self.max_var_multiplier,
            cooldown_periods=cooldown_periods,
            min_hold_periods=min_hold_periods,
            stop_loss_zscore=float(getattr(self, "pair_stop_loss_zscore", 0.0) or 0.0),
            min_spread_move_sigma=float(getattr(self, "min_spread_move_sigma", 0.0) or 0.0),
            capital_at_risk=capital_at_risk,
            min_notional_per_trade=min_notional,
            max_notional_per_trade=max_notional,
        )

        spread = y - beta * x
        z_scores = np.full_like(spread, np.nan)
        valid_sigma = sigma > 1e-6
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
