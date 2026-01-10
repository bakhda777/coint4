from .base_engine import BasePairBacktester
from ..core.numba_kernels import calculate_positions_and_pnl_full, rolling_ols
import numpy as np
import pandas as pd


class NumbaPairBacktester(BasePairBacktester):
    """Numba-оптимизированная версия BasePairBacktester.
    
    Наследует всю логику от BasePairBacktester, но переопределяет метод run()
    для использования быстрых Numba-функций.
    """
    
    def run(self) -> dict:
        try:
            # print(f"DEBUG: NumbaPairBacktester.run called for {self.pair_name}")
            import logging
            logger = logging.getLogger(__name__)
            # logger.info(f"Запуск Numba-оптимизированного бэктеста для {self.pair_name}")

            if self.pair_data.empty or len(self.pair_data.columns) < 2 or len(self.pair_data) < self.rolling_window + 2:
                logger.warning(f"DATA_WARNING pair={self.pair_name} issue=Insufficient_data len={len(self.pair_data)}")
                self.results = self._create_empty_results_df()
                return {
                    'pnl_series': pd.Series(dtype=float),
                    'positions': [],
                    'trades': [],
                    'costs': pd.Series(dtype=float),
                    'results_df': self.results
                }

            # Check for NaNs/Infs
            if self.pair_data.isnull().values.any():
                 logger.debug(f"DATA_WARNING pair={self.pair_name} issue=NaN_in_price")
            
            y = self.pair_data.iloc[:, 0].values.astype(np.float32)
            x = self.pair_data.iloc[:, 1].values.astype(np.float32)

            beta, mu, sigma = rolling_ols(y, x, self.rolling_window)
            
            # Determine max_zscore_entry from stop_loss_multiplier or config
            max_zscore_entry = 100.0
            stop_loss_threshold = 100.0
            
            # Override if config has explicit parameter
            # Also prepare other params
            min_holding_period = 0
            cooldown_period = 0
            
            # Cost parameters
            commission_val = 0.0004 # Default 0.04%
            slippage_val = 0.0005   # Default 0.05%
            
            enable_regime = True
            enable_breaks = True
            
            if hasattr(self, 'config'):
                if hasattr(self.config, 'max_zscore_entry'):
                     max_zscore_entry = float(self.config.max_zscore_entry)
                # UNIFIED: Use pair_stop_loss_zscore as source for Z-Stop
                if hasattr(self.config, 'pair_stop_loss_zscore'):
                     stop_loss_threshold = float(self.config.pair_stop_loss_zscore)
                elif hasattr(self.config, 'zscore_stop_loss'): # Fallback
                     stop_loss_threshold = float(self.config.zscore_stop_loss)
                
                # Time params (minutes -> bars)
                if hasattr(self.config, 'min_position_hold_minutes'):
                    min_holding_period = int(self.config.min_position_hold_minutes // 15)
                if hasattr(self.config, 'anti_churn_cooldown_minutes'):
                    cooldown_period = int(self.config.anti_churn_cooldown_minutes // 15)
                    
                # UNIFIED COST PARAMETERS
                # Commission: commission_rate_per_leg
                if hasattr(self.config, 'commission_rate_per_leg'):
                    commission_val = float(self.config.commission_rate_per_leg)
                
                # Slippage: slippage_pct (Unified)
                if hasattr(self.config, 'slippage_pct'):
                    slippage_val = float(self.config.slippage_pct)
            
            # Log cost params once
            import logging
            engine_logger = logging.getLogger("engine.costs")
            engine_logger.debug(f"DEBUG ENGINE COSTS: commission={commission_val:.6f}, slippage={slippage_val:.6f}, z_stop={stop_loss_threshold}")
            
            # Feature flags
            adaptive_threshold_factor = 1.0
            
            if hasattr(self, 'config'):
                if hasattr(self.config, 'market_regime_detection'):
                    enable_regime = bool(self.config.market_regime_detection)
                if hasattr(self.config, 'structural_break_protection'):
                    enable_breaks = bool(self.config.structural_break_protection)
                if hasattr(self.config, 'adaptive_thresholds'):
                    # If disabled, set factor to 0.0
                    adaptive_threshold_factor = 1.0 if self.config.adaptive_thresholds else 0.0

            # Also check direct attributes
            if hasattr(self, 'max_zscore_entry'): 
                 max_zscore_entry = float(self.max_zscore_entry)
            
            # NEW: Risk Management - Pair Max Loss (Hard Dollar Limit)
            max_loss_per_unit = 1000000.0 # Default infinite
            pair_max_loss_usd = 0.0
            
            if hasattr(self, 'config'):
                 # Check pair_stop_loss_usd first (primary source from YAML)
                 if hasattr(self.config, 'pair_stop_loss_usd') and self.config.pair_stop_loss_usd is not None:
                     pair_max_loss_usd = float(self.config.pair_stop_loss_usd)
                 # Legacy fallback
                 elif hasattr(self.config, 'risk_management'):
                     rm = self.config.risk_management
                     if hasattr(rm, 'pair_max_loss_usd'):
                         pair_max_loss_usd = float(rm.pair_max_loss_usd)
                     elif isinstance(rm, dict) and 'pair_max_loss_usd' in rm:
                         pair_max_loss_usd = float(rm['pair_max_loss_usd'])

            # Calculate dynamic min_volatility to avoid adaptive threshold explosion
            # If sigma is ~3.6 and min_vol is 0.0001, ratio is 36000 -> huge threshold multiplier.
            # We use 10th percentile of sigma as baseline min_volatility
            valid_sigma = sigma[~np.isnan(sigma)]
            if len(valid_sigma) > 0:
                calc_min_vol = np.percentile(valid_sigma, 10)
                # Ensure it's not too small (if log returns)
                # FIX: Increased minimum volatility threshold to avoid extreme Z-scores
                min_volatility = max(float(calc_min_vol), 1e-5) 
            else:
                min_volatility = 0.0001

            # === PREPARE SCALING ===
            # Calculate avg_price early for cost scaling
            avg_price = np.nanmean(y)
            scaling_factor = 1.0
            if avg_price > 0 and self.capital_at_risk > 0:
                scaling_factor = self.capital_at_risk / avg_price
            
            if pair_max_loss_usd > 0 and scaling_factor > 0:
                max_loss_per_unit = pair_max_loss_usd / scaling_factor
            
            # ADJUST COMMISSION TO PRICE UNITS IF PERCENTAGE
            # Kernel expects cost to be subtracted from PnL (Price Units).
            # If commission is %, we must multiply by Price.
            # We assume all config rates are percentages (e.g. 0.0004 = 0.04%)
            if avg_price > 0:
                commission_val *= avg_price
                slippage_val *= avg_price

            # NEW: PnL Stop Loss Calculation (Price Units)
            pnl_stop_loss_threshold = 1e9 # Default infinite
            self.current_R_price_units = 1.0 # Default
            
            # Track which stop condition is active for logging/reasoning
            self.stop_loss_reason_mode = "None" 
            
            if hasattr(self.config, 'stop_loss_type') and self.config.stop_loss_type == 'mixed':
                 # Get risk_pct from portfolio config or default
                 risk_pct = 0.005
                 if hasattr(self.config, 'portfolio') and hasattr(self.config.portfolio, 'risk_per_position_pct'):
                      risk_pct = float(self.config.portfolio.risk_per_position_pct)
                 
                 # Get R multiple
                 r_multiple = 1.5
                 if hasattr(self.config, 'pnl_stop_loss_r_multiple'):
                      r_multiple = float(self.config.pnl_stop_loss_r_multiple)
                 
                 # Calculate Threshold in Price Units
                 # StopLossPriceUnits = AvgPrice * RiskPct * Multiple
                 if avg_price > 0:
                     self.current_R_price_units = avg_price * risk_pct
                     r_based_threshold = self.current_R_price_units * r_multiple
                     
                     # Check for explicit USD Stop
                     usd_based_threshold = 1e9
                     if hasattr(self.config, 'pair_stop_loss_usd') and self.config.pair_stop_loss_usd is not None:
                          # Convert USD to Price Units: USD / ScalingFactor
                          # ScalingFactor = Capital / Price
                          # So PriceUnits = USD / (Capital/Price) = USD * Price / Capital
                          if scaling_factor > 0:
                              usd_val = float(self.config.pair_stop_loss_usd)
                              if usd_val > 0:
                                  usd_based_threshold = usd_val / scaling_factor
                     
                     # Take the stricter (smaller) threshold
                     if usd_based_threshold < r_based_threshold:
                         pnl_stop_loss_threshold = usd_based_threshold
                         self.stop_loss_reason_mode = "USD"
                     else:
                         pnl_stop_loss_threshold = r_based_threshold
                         self.stop_loss_reason_mode = "R"


            # Calculate Trade Limits params
            max_round_trips = 100000
            max_entries_per_day = 100000
            
            if hasattr(self.config, 'trade_limits'):
                 tl = self.config.trade_limits
                 if isinstance(tl, dict):
                     max_round_trips = int(tl.get('max_round_trips_per_pair_step', 100000))
                     max_entries_per_day = int(tl.get('max_new_entries_per_pair_day', 100000))
                 else:
                     # Assume object
                     max_round_trips = int(getattr(tl, 'max_round_trips_per_pair_step', 100000))
                     max_entries_per_day = int(getattr(tl, 'max_new_entries_per_pair_day', 100000))
            
            # Also check direct attributes on config (flat structure)
            if hasattr(self.config, 'max_round_trips_per_pair_step'):
                 max_round_trips = int(self.config.max_round_trips_per_pair_step)
            if hasattr(self.config, 'max_new_entries_per_pair_day'):
                 max_entries_per_day = int(self.config.max_new_entries_per_pair_day)
            
            # Prepare day indices
            day_indices = None
            if hasattr(self.pair_data.index, 'dayofyear') and hasattr(self.pair_data.index, 'year'):
                 # Create a unique day identifier: year * 1000 + dayofyear
                 years = self.pair_data.index.year.values.astype(np.int32)
                 days = self.pair_data.index.dayofyear.values.astype(np.int32)
                 day_indices = years * 1000 + days
            else:
                 # Fallback: try to infer from index if it is datetime
                 try:
                     idx = pd.to_datetime(self.pair_data.index)
                     years = idx.year.values.astype(np.int32)
                     days = idx.dayofyear.values.astype(np.int32)
                     day_indices = years * 1000 + days
                 except:
                     logger.warning(f"Could not determine day indices for pair {self.pair_name}")
                     day_indices = np.zeros(len(self.pair_data), dtype=np.int32) # All same day

            # NEW: Kill Switch Params
            max_negative_pair_step_r = 3.0 # Default
            
            # Read pair_step_r_limit from config (priority to flat config which is backtest section usually)
            if hasattr(self.config, 'pair_step_r_limit'):
                 max_negative_pair_step_r = abs(float(self.config.pair_step_r_limit))
            
            # Support for risk_limits section as requested
            if hasattr(self.config, 'risk_limits') and isinstance(self.config.risk_limits, dict):
                 if 'pair_step_r_multiple' in self.config.risk_limits:
                      max_negative_pair_step_r = abs(float(self.config.risk_limits['pair_step_r_multiple']))
            elif hasattr(self.config, 'risk_limits') and hasattr(self.config.risk_limits, 'pair_step_r_multiple'):
                  max_negative_pair_step_r = abs(float(self.config.risk_limits.pair_step_r_multiple))
            
            if hasattr(self.config, 'risk_management'):
                 # config.risk_management might be an object or dict
                 rm = self.config.risk_management
                 # Check for custom kill switch param if added later, or use hardcoded 3.0 for now as requested
                 # We can use a parameter if it exists
                 pass
            
            positions, pnl, cumulative_pnl, cost_series = calculate_positions_and_pnl_full(
                y, x,
                rolling_window=int(self.rolling_window),
                entry_threshold=float(self.zscore_entry_threshold),
                exit_threshold=float(self.z_exit),
                commission=commission_val,
                slippage=slippage_val,
                max_holding_period=99999,
                enable_regime_detection=enable_regime,
                enable_structural_breaks=enable_breaks,
                min_volatility=float(min_volatility),
                adaptive_threshold_factor=float(adaptive_threshold_factor),
                max_zscore_entry=float(max_zscore_entry),
                stop_loss_threshold=float(stop_loss_threshold),
                min_holding_period=min_holding_period,
                cooldown_period=cooldown_period,
                max_loss_per_unit=float(max_loss_per_unit),
                pnl_stop_loss_threshold=float(pnl_stop_loss_threshold),
                day_indices=day_indices,
                max_round_trips=max_round_trips,
                max_entries_per_day=max_entries_per_day,
                current_R_price_units=float(self.current_R_price_units),
                max_negative_pair_step_r=float(max_negative_pair_step_r)
            )

            # Log if limits were hit
            if len(cumulative_pnl) > 0:
                final_cum_pnl = cumulative_pnl[-1]
                final_cum_pnl_r = final_cum_pnl / float(self.current_R_price_units) if self.current_R_price_units > 0 else 0.0
                
                if final_cum_pnl_r <= -max_negative_pair_step_r:
                    logger.info(f"[PAIR STEP LIMIT HIT] pair={self.pair_name}, step_pnl_r={final_cum_pnl_r:.2f}, limit={max_negative_pair_step_r:.2f}")

                if pair_max_loss_usd > 0:
                     # Scaling factor = Capital / Price. PnL (PriceUnits) * ScalingFactor = USD
                     final_cum_pnl_usd = final_cum_pnl * scaling_factor
                     if final_cum_pnl_usd <= -pair_max_loss_usd:
                          logger.info(f"[PAIR USD LIMIT HIT] pair={self.pair_name}, step_pnl_usd={final_cum_pnl_usd:.2f}, limit={pair_max_loss_usd:.2f}")

            spread = y - beta * x
            z_scores = np.full_like(spread, np.nan)
            
            # Safe Z-score calculation matching numba_kernels logic
            # Use min_volatility from above to match kernel logic
            # min_volatility = 0.0001 # REMOVED hardcoded value
            
            # Vectorized safe calculation
            # We need to handle cases where sigma is small or NaN
            safe_sigma = np.maximum(sigma, min_volatility)
            
            # Calculate Z-scores where data is valid
            valid_mask = ~np.isnan(spread) & ~np.isnan(mu) & ~np.isnan(sigma)
            z_scores[valid_mask] = (spread[valid_mask] - mu[valid_mask]) / safe_sigma[valid_mask]
            
            # Clamp extreme values to prevent JSON errors and logging noise
            # FIX: Stricter clamping for reporting
            z_scores = np.clip(z_scores, -20.0, 20.0)
            
            # Handle any remaining NaNs or Infs just in case
            z_scores = np.nan_to_num(z_scores, nan=0.0, posinf=100.0, neginf=-100.0)

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
            
            trades_mask = self.results['position'].diff() != 0
            self.results.loc[trades_mask, 'trades'] = 1.0
            
            # FIX: Define trades_count
            trades_count = trades_mask.sum()

            # ANOMALY CHECK: Signals detected but no trades
            if trades_count == 0 and len(positions) > 0:
                 # Check if we had high Z-scores that were ignored
                 high_z_count = np.sum(np.abs(z_scores) > self.zscore_entry_threshold)
                 if high_z_count > 10: # Arbitrary threshold
                      logger.debug(f"ANOMALY_NO_TRADES pair={self.pair_name} high_z_count={high_z_count} trades=0")

            # === SCALING PNL TO CAPITAL ===
            # avg_price and scaling_factor calculated above
            self.current_R_value = self.current_R_price_units # Default to Price Units
            
            if avg_price > 0 and self.capital_at_risk > 0:
                # Apply scaling
                self.results['pnl'] *= scaling_factor
                self.results['cumulative_pnl'] *= scaling_factor
                self.results['position'] *= scaling_factor
                pnl *= scaling_factor
                cumulative_pnl *= scaling_factor
                
                # Update R to USD
                self.current_R_value *= scaling_factor
                
                # NEW: Scale costs
                # cost_series from kernel is now in Price Units (because we scaled inputs).
                # scaling_factor is (Capital / Price).
                # cost_scaled = (Cost_Price_Units) * (Capital / Price).
                # Example: Cost_Price_Units = 0.1 (0.1% of 100).
                # Capital = 10000. Price = 100. Scaling = 100.
                # Scaled = 0.1 * 100 = 10. Correct (0.1% of 10000).
                cost_series = cost_series * scaling_factor
                self.results['costs'] = cost_series
            else:
                # If no scaling, still save cost_series
                self.results['costs'] = cost_series
            
            # SIGNAL DEBUG LOGGING
            # if trades_count > 0: # Log only if there are trades
            #      valid_z = z_scores[~np.isnan(z_scores)]
            #      if len(valid_z) > 0:
            #           logger.info(
            #                f"[SIGNAL_DEBUG] {self.pair_name} "
            #                f"spread={spread[-1]:.6f} mean={mu[-1]:.6f} std={sigma[-1]:.6f} "
            #                f"z_last={valid_z[-1]:.3f} max_z={np.nanmax(np.abs(valid_z)):.3f} "
            #                f"pos={positions[-1]}"
            #           )

            # if hasattr(self, 'pair_name') and 'HVHUSDT' in self.pair_name:  # Логируем только первую пару
            #     logger.info(f"🔍 ДИАГНОСТИКА {self.pair_name}:")
            #     logger.info(f"   entry_threshold: {self.zscore_entry_threshold}")
            #     logger.info(f"   exit_threshold: {self.z_exit}")
            #     logger.info(f"   Макс |z_score|: {np.nanmax(np.abs(z_scores)):.4f}")
            #     logger.info(f"   Позиций != 0: {np.sum(positions != 0)}")
            #     logger.info(f"   Изменений позиций: {trades_count}")
            #     logger.info(f"   PnL сумма: {pnl.sum():.4f}")
            #
            #     # Показываем первые несколько z_scores
            #     valid_z = z_scores[~np.isnan(z_scores)]
            #     if len(valid_z) > 0:
            #         logger.info(f"   Первые 10 z_scores: {valid_z[:10]}")
            #         logger.info(f"   Превышают порог: {np.sum(np.abs(valid_z) > self.zscore_entry_threshold)}")

            # Обновляем портфель (метод update_pnl не существует, убираем)
            # if hasattr(self, 'portfolio') and self.portfolio is not None:
            #     self.portfolio.update_pnl(pnl.sum())
            
            # Возвращаем результаты в формате ожидаемом в walk_forward
            detailed_trades = self._extract_detailed_trades()
            print("DEBUG: RETURNING DICT")
            return {
                'pnl_series': self.results['pnl'] if not self.results.empty else pd.Series(dtype=float),
                'positions': self.results['position'].tolist() if not self.results.empty else [],
                'trades': detailed_trades,
                'costs': self.results['costs'] if not self.results.empty else pd.Series(dtype=float),
                'results_df': self.results
            }
        except Exception as e:
            print(f"CRITICAL ERROR IN NumbaPairBacktester: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_detailed_trades(self) -> list:
        """Extracts detailed trade information from results DataFrame."""
        if self.results.empty:
            return []
            
        trades = []
        df = self.results
        pos_diff = df['position'].diff().fillna(0)
        change_indices = df.index[pos_diff != 0]
        
        current_trade = None
        
        for ts in change_indices:
            row = df.loc[ts]
            new_pos = row['position']
            diff = pos_diff[ts]
            prev_pos = new_pos - diff
            
            if prev_pos != 0: # Closing or Flipping
                if current_trade:
                    current_trade['exit_time'] = ts
                    current_trade['exit_z'] = row['z_score']
                    duration = ts - current_trade['entry_time']
                    current_trade['hold'] = str(duration)
                    
                    mask = (df.index >= current_trade['entry_time']) & (df.index <= ts)
                    trade_pnl = df.loc[mask, 'pnl'].sum()
                    current_trade['pnl'] = float(trade_pnl) # This is Net PnL now
                    
                    # NEW: Calculate Gross and Costs
                    trade_costs = df.loc[mask, 'costs'].sum()
                    current_trade['costs'] = float(trade_costs)
                    current_trade['gross_pnl'] = float(trade_pnl + trade_costs)
                    current_trade['net_pnl'] = float(trade_pnl)
                    
                    # NEW: Calculate R-based metrics
                    r_val = getattr(self, 'current_R_value', 1.0)
                    if r_val <= 0: r_val = 1.0
                    
                    current_trade['final_pnl_r'] = float(current_trade['net_pnl'] / r_val)
                    
                    # Calculate Max Drawdown for Trade
                    trade_pnl_series = df.loc[mask, 'pnl']
                    trade_cum_pnl = trade_pnl_series.cumsum()
                    trade_min_pnl = float(trade_cum_pnl.min())
                    current_trade['max_drawdown_r'] = float(trade_min_pnl / r_val)
                    
                    exit_z_abs = abs(row['z_score'])
                    exit_z = row['z_score']
                    
                    # Determine side from previous position (before this row closed it)
                    # We know we are closing, so new_pos is 0 (or different). 
                    # prev_pos (calculated above) tells us what we held.
                    side = 'LONG' if prev_pos > 0 else 'SHORT'
                    
                    # Heuristics for exit reason
                    current_trade['exit_reason'] = "Signal" # Default
                    current_trade['exit_by_pnl'] = False
                    current_trade['exit_by_z'] = False
                    current_trade['exit_by_time'] = False
                    
                    is_stop_loss = False
                    
                    # 1. Check PnL Stop
                    pnl_stop_multiple = 1.5
                    if hasattr(self.config, 'pnl_stop_loss_r_multiple'):
                         pnl_stop_multiple = float(self.config.pnl_stop_loss_r_multiple)
                    
                    # Check if PnL hit the stop (Logic from kernel: current_trade_pnl < -pnl_stop_loss_threshold)
                    # We need to replicate the "which stop hit" logic based on net_pnl
                    
                    # Re-calculate thresholds here for labeling
                    r_based_stop = r_val * pnl_stop_multiple
                    
                    # Check USD stop
                    usd_based_stop = 1e9
                    # scaling_factor = capital / avg_price
                    # usd_val / scaling_factor = price_units
                    # We need net_pnl in price units to compare
                    # But net_pnl is in USD (if scaled) or PriceUnits (if not).
                    # Wait, self.results['pnl'] is SCALED at end of run() method.
                    # So net_pnl here is in USD (assuming scaling enabled).
                    
                    # If scaled:
                    # r_val = Capital * RiskPct (USD)
                    # net_pnl = USD
                    # r_based_stop = USD
                    
                    if hasattr(self.config, 'pair_stop_loss_usd') and self.config.pair_stop_loss_usd is not None:
                         usd_val = float(self.config.pair_stop_loss_usd)
                         if usd_val > 0:
                              usd_based_stop = usd_val

                    # Strict check (no tolerance)
                    # Note: Kernel now caps the PnL to threshold, so net_pnl should be roughly -threshold.
                    # But we add 0.0001 epsilon just in case floating point logic varies.
                    if current_trade['net_pnl'] <= -min(r_based_stop, usd_based_stop) + 0.0001: # Epsilon
                         is_stop_loss = True
                         current_trade['exit_by_pnl'] = True
                         
                         # Labeling
                         if usd_based_stop < r_based_stop and current_trade['net_pnl'] <= -usd_based_stop + 0.0001:
                              current_trade['exit_reason'] = "PnLStopHardUSD"
                              stop_type_str = "USD"
                         else:
                              current_trade['exit_reason'] = "PnLStopHardR"
                              stop_type_str = "R"

                         import logging
                         logger = logging.getLogger(__name__)
                         logger.info(
                              f"[PNL_STOP_HARD] pair={self.pair_name} "
                              f"pnl_usd={current_trade['net_pnl']:.2f} "
                              f"pnl_r={current_trade['final_pnl_r']:.2f} "
                              f"r_stop={r_based_stop:.2f} "
                              f"usd_stop={usd_based_stop:.2f} "
                              f"chosen={stop_type_str} "
                              f"reason={current_trade['exit_reason']}"
                         )
                         
                    # NEW: Step Risk Log
                    if hasattr(self.config, 'risk_limits') and hasattr(self.config.risk_limits, 'pair_step_r_multiple'):
                         limit_r = abs(float(self.config.risk_limits.pair_step_r_multiple))
                         # Check if we breached it
                         # cumulative_pnl is in USD. We need R.
                         # Actually, we have cumulative_pnl column in results.
                         # But step risk is based on cumulative_pnl_R inside kernel.
                         # We can check current_trade['final_pnl_r'] + previous trades.
                         # Easier: check if trade exit time aligns with a forced close?
                         # Or just calculate cumulative R here.
                         pass # Difficult to detect exact step breach event post-factum without kernel log.
                         # We rely on walk_forward orchestrator log.


                    
                    # 2. Check Z-score Stop
                    stop_loss_thresh = 100.0
                    # Use unified logic for stop threshold source
                    if hasattr(self.config, 'pair_stop_loss_zscore'):
                         stop_loss_thresh = float(self.config.pair_stop_loss_zscore)
                    elif hasattr(self.config, 'zscore_stop_loss'):
                         stop_loss_thresh = float(self.config.zscore_stop_loss)
                    elif hasattr(self, 'stop_loss_multiplier') and self.stop_loss_multiplier > 0:
                         stop_loss_thresh = float(self.stop_loss_multiplier)

                    if side == 'LONG' and exit_z < -stop_loss_thresh:
                        is_stop_loss = True
                        current_trade['exit_reason'] = "ZStop"
                        current_trade['exit_by_z'] = True
                    elif side == 'SHORT' and exit_z > stop_loss_thresh:
                        is_stop_loss = True
                        current_trade['exit_reason'] = "ZStop"
                        current_trade['exit_by_z'] = True
                            
                    if is_stop_loss:
                        # NEW: Explicit logging for risk trigger
                        pass # Logged above if PnL stop, or ZStop is self-explanatory
                             
                    elif exit_z_abs <= self.z_exit + 0.1:
                        current_trade['exit_reason'] = "Target"
                    elif exit_z_abs > self.zscore_entry_threshold: 
                        # If we exit at high Z but it's not a stop loss, it's likely an overshoot profit or signal flip
                        current_trade['exit_reason'] = "Signal" 

                    # NEW: Add explicit log about hold time and constraints
                    hold_duration = duration # Timedelta
                    hold_minutes = hold_duration.total_seconds() / 60.0
                    
                    min_hold_minutes = 0
                    if hasattr(self.config, 'min_position_hold_minutes'):
                         min_hold_minutes = int(self.config.min_position_hold_minutes)
                    
                    cooldown_minutes = 0
                    if hasattr(self.config, 'anti_churn_cooldown_minutes'):
                         cooldown_minutes = int(self.config.anti_churn_cooldown_minutes)
                         
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(
                        f"[TRADE_EXIT] pair={self.pair_name}, "
                        f"reason={current_trade['exit_reason']}, "
                        f"hold_minutes={hold_minutes:.1f}, "
                        f"min_hold={min_hold_minutes}, "
                        f"cooldown={cooldown_minutes}, "
                        f"pnl_usd={current_trade['net_pnl']:.2f}"
                    )
                    
                    trades.append(current_trade)
                    current_trade = None
            
            if new_pos != 0: # Opening
                entry_price = 1.0
                if hasattr(self, 'pair_data') and ts in self.pair_data.index:
                     # Use Y as price proxy
                     entry_price = float(self.pair_data.loc[ts].iloc[0])

                current_trade = {
                    'entry_time': ts,
                    'entry_z': row['z_score'],
                    'side': 'LONG' if new_pos > 0 else 'SHORT',
                    'size': float(new_pos),
                    'entry_price': entry_price,
                    'notional': float(abs(new_pos) * entry_price)
                }
                
        if current_trade:
            current_trade['exit_time'] = df.index[-1]
            current_trade['exit_z'] = df.iloc[-1]['z_score']
            current_trade['hold'] = str(df.index[-1] - current_trade['entry_time'])
            current_trade['exit_reason'] = "ForceClose"
            
            mask = df.index >= current_trade['entry_time']
            trade_pnl = df.loc[mask, 'pnl'].sum()
            trade_costs = df.loc[mask, 'costs'].sum()
            
            current_trade['pnl'] = float(trade_pnl)
            current_trade['costs'] = float(trade_costs)
            current_trade['gross_pnl'] = float(trade_pnl + trade_costs)
            current_trade['net_pnl'] = float(trade_pnl)
            
            trades.append(current_trade)
            
        return trades

    def _create_empty_results_df(self) -> pd.DataFrame:
        """Создает пустой DataFrame с результатами."""
        return pd.DataFrame(
            columns=["spread", "z_score", "position", "pnl", "cumulative_pnl", 
                    "beta", "mean", "std", "trades", "costs", "commission_costs", 
                    "slippage_costs", "bid_ask_costs", "impact_costs"]
        )