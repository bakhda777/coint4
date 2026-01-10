#!/usr/bin/env python3
"""
Real-time Optuna optimization runner for Streamlit UI.
Provides live updates and visualization of optimization progress.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import yaml
import time
import subprocess
import threading
import queue
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import optuna
    from optuna import visualization as optuna_vis
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    st.error("⚠️ Optuna не установлен. Установите: `pip install optuna`")

from scripts.optimization.web_optimizer import WebOptimizer


class StreamlitOptimizationRunner:
    """Runs optimization with real-time Streamlit UI updates."""
    
    def __init__(self):
        self.optimizer = None
        self.study = None
        self.progress_queue = queue.Queue()
        self.stop_flag = threading.Event()
        self.optimization_thread = None
        
    def create_search_space_from_ui(self, param_ranges: Dict[str, Tuple[float, float]]) -> Dict:
        """Convert UI parameter ranges to search space format."""
        search_space = {
            'signals': {},
            'risk': {},
            'filters': {},
            'costs': {}
        }
        
        for param_name, (min_val, max_val) in param_ranges.items():
            # Categorize parameters
            if 'zscore' in param_name or 'rolling_window' in param_name:
                category = 'signals'
            elif 'stop' in param_name or 'position_size' in param_name:
                category = 'risk'
            elif 'coint' in param_name or 'hurst' in param_name or 'half_life' in param_name:
                category = 'filters'
            elif 'commission' in param_name or 'slippage' in param_name:
                category = 'costs'
            else:
                continue
            
            param_type = 'int' if 'window' in param_name else 'float'
            
            search_space[category][param_name] = {
                'type': param_type,
                'low': min_val,
                'high': max_val
            }
        
        return search_space
    
    def run_optimization(
        self,
        n_trials: int,
        param_ranges: Dict[str, Tuple[float, float]],
        target_metric: str,
        sampler: str,
        pruner: bool,
        config_path: str = "configs/main_2024.yaml",
        progress_callback = None
    ) -> Optional[optuna.Study]:
        """Run optimization in background thread with progress updates."""
        
        def optimize():
            try:
                # Create WebOptimizer instance
                self.optimizer = WebOptimizer(
                    base_config_path=config_path,
                    search_space_path="configs/search_spaces/web_ui.yaml"
                )
                
                # Create study
                study_name = f"streamlit_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                storage = f"sqlite:///outputs/studies/{study_name}.db"
                
                # Select sampler
                if sampler == "TPE":
                    sampler_obj = optuna.samplers.TPESampler(seed=42)
                elif sampler == "Random":
                    sampler_obj = optuna.samplers.RandomSampler(seed=42)
                else:  # Grid
                    sampler_obj = optuna.samplers.GridSampler(param_ranges)
                
                # Select pruner
                if pruner:
                    pruner_obj = optuna.pruners.MedianPruner(
                        n_startup_trials=5,
                        n_warmup_steps=10
                    )
                else:
                    pruner_obj = optuna.pruners.NopPruner()
                
                self.study = optuna.create_study(
                    study_name=study_name,
                    storage=storage,
                    direction="maximize",
                    sampler=sampler_obj,
                    pruner=pruner_obj,
                    load_if_exists=False
                )
                
                # Create objective function
                objective = self.optimizer.create_objective(param_ranges)
                
                # Run optimization trial by trial for progress updates
                for i in range(n_trials):
                    if self.stop_flag.is_set():
                        break
                    
                    # Run single trial
                    self.study.optimize(objective, n_trials=1)
                    
                    # Get latest trial
                    trial = self.study.trials[-1]
                    
                    # Send progress update
                    self.progress_queue.put({
                        'trial': trial.number,
                        'value': trial.value if trial.value is not None else -999,
                        'params': trial.params,
                        'state': str(trial.state),
                        'status': 'completed'
                    })
                    
                    # Callback for UI updates
                    if progress_callback:
                        progress_callback(trial.number, trial.value, trial.params)
                
                # Send completion signal
                self.progress_queue.put({
                    'status': 'finished',
                    'best_value': self.study.best_value,
                    'best_params': self.study.best_params,
                    'n_trials': len(self.study.trials)
                })
                
            except Exception as e:
                self.progress_queue.put({
                    'status': 'error',
                    'error': str(e)
                })
        
        # Start optimization in background thread
        self.optimization_thread = threading.Thread(target=optimize)
        self.optimization_thread.start()
        
        return self.study
    
    def stop_optimization(self):
        """Stop the running optimization."""
        self.stop_flag.set()
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)
    
    def get_progress(self) -> Optional[Dict[str, Any]]:
        """Get latest progress update."""
        try:
            return self.progress_queue.get_nowait()
        except queue.Empty:
            return None
    
    def wait_for_completion(self, timeout: int = 3600) -> bool:
        """Wait for optimization to complete."""
        if self.optimization_thread:
            self.optimization_thread.join(timeout=timeout)
            return not self.optimization_thread.is_alive()
        return True


def render_optimization_ui():
    """Main UI for optimization in Streamlit."""
    
    st.title("🔬 Оптимизация параметров стратегии")
    st.caption("Используйте Optuna для поиска оптимальных параметров")
    
    if not OPTUNA_AVAILABLE:
        st.error("Optuna не установлен. Установите: pip install optuna")
        return
    
    # Initialize session state
    if 'opt_runner' not in st.session_state:
        st.session_state.opt_runner = StreamlitOptimizationRunner()
    if 'opt_running' not in st.session_state:
        st.session_state.opt_running = False
    if 'opt_results' not in st.session_state:
        st.session_state.opt_results = None
    
    runner = st.session_state.opt_runner
    
    # Settings columns
    col_settings, col_params = st.columns([1, 2])
    
    with col_settings:
        st.subheader("⚙️ Настройки оптимизации")
        
        # Preset selection
        preset = st.selectbox(
            "🎯 Режим",
            ["🏃 Fast (10 trials)", "⚖️ Balanced (50 trials)", "🔬 Deep (100 trials)", "🎯 Custom"]
        )
        
        if "Fast" in preset:
            n_trials = 10
        elif "Balanced" in preset:
            n_trials = 50
        elif "Deep" in preset:
            n_trials = 100
        else:
            n_trials = st.number_input("Trials", min_value=5, max_value=500, value=50)
        
        sampler = st.selectbox("🎲 Sampler", ["TPE", "Random"])
        pruner = st.checkbox("✂️ Pruner", value=True)
        
        target_metric = st.selectbox(
            "🎯 Метрика",
            ["Sharpe Ratio", "Total PnL", "Win Rate", "Calmar Ratio"]
        )
        
        st.markdown("---")
        
        # Config selection
        config_path = st.selectbox(
            "📋 Конфигурация",
            ["configs/main_2024.yaml", "configs/main.yaml"],
            help="Базовая конфигурация для оптимизации"
        )
    
    with col_params:
        st.subheader("🎛️ Диапазоны параметров")
        
        param_tabs = st.tabs(["📈 Сигналы", "🛡️ Риски", "🔍 Фильтры", "💰 Издержки"])
        
        param_ranges = {}
        
        with param_tabs[0]:  # Signals
            st.markdown("**Параметры входа/выхода**")
            
            col1, col2 = st.columns(2)
            with col1:
                zscore_range = st.slider(
                    "Z-score threshold",
                    min_value=0.5, max_value=3.0,
                    value=(1.0, 2.0), step=0.1
                )
                param_ranges['zscore_threshold'] = zscore_range
                
                rolling_range = st.slider(
                    "Rolling window",
                    min_value=10, max_value=100,
                    value=(20, 50), step=5
                )
                param_ranges['rolling_window'] = rolling_range
            
            with col2:
                exit_range = st.slider(
                    "Z-score exit",
                    min_value=-0.5, max_value=0.5,
                    value=(-0.2, 0.2), step=0.05
                )
                param_ranges['zscore_exit'] = exit_range
        
        with param_tabs[1]:  # Risk
            st.markdown("**Управление рисками**")
            
            col1, col2 = st.columns(2)
            with col1:
                stop_loss_range = st.slider(
                    "Stop loss multiplier",
                    min_value=2.0, max_value=5.0,
                    value=(2.5, 3.5), step=0.5
                )
                param_ranges['stop_loss_multiplier'] = stop_loss_range
            
            with col2:
                position_size_range = st.slider(
                    "Max position size %",
                    min_value=0.01, max_value=0.10,
                    value=(0.02, 0.05), step=0.005
                )
                param_ranges['max_position_size_pct'] = position_size_range
        
        with param_tabs[2]:  # Filters
            st.markdown("**Фильтры пар**")
            
            col1, col2 = st.columns(2)
            with col1:
                pvalue_range = st.slider(
                    "Cointegration p-value",
                    min_value=0.01, max_value=0.10,
                    value=(0.01, 0.05), step=0.01
                )
                param_ranges['coint_pvalue_threshold'] = pvalue_range
            
            with col2:
                hurst_range = st.slider(
                    "Max Hurst exponent",
                    min_value=0.4, max_value=0.6,
                    value=(0.45, 0.55), step=0.05
                )
                param_ranges['max_hurst_exponent'] = hurst_range
        
        with param_tabs[3]:  # Costs
            st.markdown("**Торговые издержки**")
            
            col1, col2 = st.columns(2)
            with col1:
                commission_range = st.slider(
                    "Commission %",
                    min_value=0.0002, max_value=0.0010,
                    value=(0.0003, 0.0005), step=0.0001,
                    format="%.4f"
                )
                param_ranges['commission_pct'] = commission_range
            
            with col2:
                slippage_range = st.slider(
                    "Slippage %",
                    min_value=0.0002, max_value=0.0010,
                    value=(0.0004, 0.0006), step=0.0001,
                    format="%.4f"
                )
                param_ranges['slippage_pct'] = slippage_range
    
    # Run optimization button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if not st.session_state.opt_running:
            if st.button("🚀 ЗАПУСТИТЬ ОПТИМИЗАЦИЮ", type="primary", use_container_width=True):
                st.session_state.opt_running = True
                st.rerun()
        else:
            if st.button("🛑 ОСТАНОВИТЬ", type="secondary", use_container_width=True):
                runner.stop_optimization()
                st.session_state.opt_running = False
                st.rerun()
    
    # Show optimization progress
    if st.session_state.opt_running:
        st.markdown("---")
        st.subheader("📊 Прогресс оптимизации")
        
        # Progress indicators
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                trials_metric = st.empty()
            with col2:
                best_metric = st.empty()
            with col3:
                time_metric = st.empty()
            with col4:
                improvement_metric = st.empty()
            
            # Log expander
            with st.expander("📝 Логи оптимизации", expanded=False):
                log_container = st.empty()
                logs = []
            
            # Start optimization
            def progress_callback(trial_num, value, params):
                # This will be called from the optimization thread
                pass
            
            runner.run_optimization(
                n_trials=n_trials,
                param_ranges=param_ranges,
                target_metric=target_metric.lower().replace(" ", "_"),
                sampler=sampler,
                pruner=pruner,
                config_path=config_path,
                progress_callback=progress_callback
            )
            
            # Track progress
            start_time = time.time()
            best_value = -float('inf')
            baseline_value = None
            completed_trials = 0
            
            while completed_trials < n_trials:
                # Get progress update
                update = runner.get_progress()
                
                if update:
                    if update['status'] == 'completed':
                        completed_trials += 1
                        trial_value = update['value']
                        
                        # Update best
                        if trial_value > best_value:
                            best_value = trial_value
                        
                        if baseline_value is None:
                            baseline_value = trial_value
                        
                        # Update UI
                        progress = completed_trials / n_trials
                        progress_bar.progress(progress)
                        status_text.text(f"Trial {completed_trials}/{n_trials}")
                        
                        trials_metric.metric("🔄 Trials", f"{completed_trials}/{n_trials}")
                        best_metric.metric("🏆 Best", f"{best_value:.3f}")
                        
                        elapsed = time.time() - start_time
                        time_metric.metric("⏱️ Время", f"{elapsed:.1f}s")
                        
                        if baseline_value and baseline_value != 0:
                            improvement = (best_value / baseline_value - 1) * 100
                            improvement_metric.metric("📈 Улучшение", f"{improvement:+.1f}%")
                        
                        # Add to log
                        log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Trial {completed_trials}: {trial_value:.3f}"
                        logs.append(log_entry)
                        log_container.text("\n".join(logs[-20:]))
                    
                    elif update['status'] == 'finished':
                        st.session_state.opt_running = False
                        st.session_state.opt_results = update
                        break
                    
                    elif update['status'] == 'error':
                        st.error(f"Ошибка: {update['error']}")
                        st.session_state.opt_running = False
                        break
                
                time.sleep(0.1)
            
            # Show completion
            if not st.session_state.opt_running:
                st.success(f"✅ Оптимизация завершена! Лучший результат: {best_value:.3f}")
    
    # Show results
    if st.session_state.opt_results and runner.study:
        st.markdown("---")
        st.subheader("📊 Результаты оптимизации")
        
        # Visualization tabs
        viz_tabs = st.tabs(["📈 История", "🎯 Важность", "📊 Parallel", "🔥 Contour"])
        
        with viz_tabs[0]:  # History
            fig = optuna_vis.plot_optimization_history(runner.study)
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tabs[1]:  # Importance
            try:
                fig = optuna_vis.plot_param_importances(runner.study)
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("Недостаточно данных для расчета важности")
        
        with viz_tabs[2]:  # Parallel coordinates
            fig = optuna_vis.plot_parallel_coordinate(runner.study)
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tabs[3]:  # Contour
            params = list(runner.study.best_params.keys())
            if len(params) >= 2:
                fig = optuna_vis.plot_contour(runner.study, params=params[:2])
                st.plotly_chart(fig, use_container_width=True)
        
        # Best parameters
        st.markdown("---")
        st.subheader("🏆 Лучшие параметры")
        
        best_params = st.session_state.opt_results['best_params']
        
        # Display in columns
        param_items = list(best_params.items())
        n_params = len(param_items)
        n_cols = 3
        cols = st.columns(n_cols)
        
        for i, (key, value) in enumerate(param_items):
            col_idx = i % n_cols
            with cols[col_idx]:
                if isinstance(value, float):
                    st.metric(key, f"{value:.4f}")
                else:
                    st.metric(key, value)
        
        # Export buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export as YAML
            yaml_str = yaml.dump(best_params, default_flow_style=False)
            st.download_button(
                "📥 Скачать YAML",
                yaml_str,
                file_name=f"best_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml",
                mime="text/yaml"
            )
        
        with col2:
            # Export as JSON
            json_str = json.dumps(best_params, indent=2)
            st.download_button(
                "📥 Скачать JSON",
                json_str,
                file_name=f"best_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col3:
            # Apply to Quick Start
            if st.button("✅ Применить в Quick Start"):
                st.session_state['quick_params'] = best_params
                st.success("Параметры применены!")


if __name__ == "__main__":
    # Test the UI
    render_optimization_ui()