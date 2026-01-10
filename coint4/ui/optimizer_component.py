"""
Real Optuna optimization component for Streamlit UI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import time
import subprocess
import threading
import queue
from typing import Dict, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px

# Import optimization modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import optuna
    from optuna import visualization as optuna_vis
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class OptimizationRunner:
    """Manages real Optuna optimization runs."""
    
    def __init__(self, output_dir: str = "outputs/optimization"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_study = None
        self.optimization_thread = None
        self.progress_queue = queue.Queue()
        self.stop_flag = threading.Event()
        
    def run_optimization_subprocess(
        self,
        n_trials: int,
        param_ranges: Dict[str, Tuple[float, float]],
        config_path: str,
        search_space_path: str,
        train_start: str,
        train_end: str,
        test_days: int
    ) -> Dict[str, Any]:
        """Run optimization in subprocess for better isolation.
        
        Returns:
            Dictionary with results
        """
        # Create temporary config with parameter ranges
        temp_search_space = self.output_dir / "temp_search_space.yaml"
        
        # Convert param_ranges to search space format
        search_space = self._create_search_space(param_ranges)
        with open(temp_search_space, 'w') as f:
            yaml.dump(search_space, f)
        
        # Build command
        cmd = [
            sys.executable,
            "scripts/core/optimize.py",
            "--mode", "fast",
            "--n-trials", str(n_trials),
            "--base-config", config_path,
            "--search-space", str(temp_search_space),
            "--study-name", f"web_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "--verbose"
        ]
        
        # Run optimization
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Collect output
        output_lines = []
        for line in process.stdout:
            output_lines.append(line.rstrip())
            # Parse progress
            if "Trial" in line and "finished" in line:
                self._parse_trial_result(line)
        
        process.wait()
        
        # Load results from database
        results = self._load_optimization_results()
        
        return results
    
    def run_optimization_thread(
        self,
        n_trials: int,
        param_ranges: Dict[str, Tuple[float, float]],
        target_metric: str = "sharpe_ratio",
        config_path: str = "configs/main_2024.yaml"
    ):
        """Run optimization in separate thread with progress updates."""
        
        def optimize():
            try:
                # Create study
                study_name = f"web_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                storage = f"sqlite:///outputs/studies/{study_name}.db"
                
                self.current_study = optuna.create_study(
                    study_name=study_name,
                    storage=storage,
                    direction="maximize",
                    load_if_exists=False
                )
                
                # Import real objective
                from scripts.optimization.web_optimizer import WebOptimizer
                
                # Initialize web optimizer
                web_opt = WebOptimizer(
                    base_config_path=config_path,
                    search_space_path="configs/search_spaces/web_ui.yaml"
                )
                
                # Create objective with real backtesting
                def objective(trial):
                    # Sample parameters
                    params = {}
                    for param_name, (min_val, max_val) in param_ranges.items():
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                        else:
                            params[param_name] = trial.suggest_float(param_name, min_val, max_val)
                    
                    # Run real backtest via FastWalkForwardObjective
                    try:
                        # Call the real objective
                        value = web_opt.objective(trial)
                        
                        # Send progress update
                        self.progress_queue.put({
                            'trial': trial.number,
                            'value': value,
                            'params': params,
                            'status': 'completed'
                        })
                        
                        return value
                        
                    except Exception as e:
                        # Log error and return bad value
                        self.progress_queue.put({
                            'trial': trial.number,
                            'value': -999.0,
                            'params': params,
                            'status': 'failed',
                            'error': str(e)
                        })
                        return -999.0
                
                # Run optimization
                for i in range(n_trials):
                    if self.stop_flag.is_set():
                        break
                    
                    self.current_study.optimize(objective, n_trials=1)
                
                # Send completion
                self.progress_queue.put({
                    'status': 'finished',
                    'best_value': self.current_study.best_value,
                    'best_params': self.current_study.best_params
                })
                
            except Exception as e:
                self.progress_queue.put({
                    'status': 'error',
                    'error': str(e)
                })
        
        # Start thread
        self.optimization_thread = threading.Thread(target=optimize)
        self.optimization_thread.start()
    
    def stop_optimization(self):
        """Stop running optimization."""
        self.stop_flag.set()
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)
    
    def get_progress(self) -> Optional[Dict[str, Any]]:
        """Get progress update from queue."""
        try:
            return self.progress_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _create_search_space(self, param_ranges: Dict[str, Tuple[float, float]]) -> Dict:
        """Convert parameter ranges to search space format."""
        search_space = {}
        
        for param_name, (min_val, max_val) in param_ranges.items():
            # Determine category
            if 'zscore' in param_name or 'threshold' in param_name:
                category = 'signals'
            elif 'stop' in param_name or 'risk' in param_name or 'position' in param_name:
                category = 'risk'
            elif 'coint' in param_name or 'hurst' in param_name or 'half_life' in param_name:
                category = 'filters'
            elif 'commission' in param_name or 'slippage' in param_name:
                category = 'costs'
            else:
                category = 'other'
            
            if category not in search_space:
                search_space[category] = {}
            
            search_space[category][param_name] = {
                'type': 'int' if isinstance(min_val, int) else 'float',
                'low': min_val,
                'high': max_val
            }
        
        return search_space
    
    def _parse_trial_result(self, line: str):
        """Parse trial result from output line."""
        # Example: "Trial 5 finished with value: 0.523"
        try:
            if "Trial" in line and "value" in line:
                parts = line.split()
                trial_num = int(parts[1])
                value = float(parts[-1])
                
                self.progress_queue.put({
                    'trial': trial_num,
                    'value': value,
                    'status': 'completed'
                })
        except:
            pass
    
    def _load_optimization_results(self) -> Dict[str, Any]:
        """Load results from the latest optimization."""
        # Find latest study database
        study_files = list(Path("outputs/studies").glob("web_opt_*.db"))
        if not study_files:
            return {}
        
        latest_db = max(study_files, key=lambda x: x.stat().st_mtime)
        
        # Load study
        storage = f"sqlite:///{latest_db}"
        study = optuna.load_study(
            study_name=latest_db.stem,
            storage=storage
        )
        
        return {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'study': study
        }


def render_optimization_results(study: optuna.Study):
    """Render Optuna visualization plots."""
    
    if not OPTUNA_AVAILABLE:
        st.warning("Optuna visualization not available")
        return
    
    st.subheader("📊 Визуализация результатов")
    
    # Create tabs for different plots
    plot_tabs = st.tabs([
        "📈 История",
        "🎯 Важность",
        "📊 Parallel Coordinates",
        "🔥 Contour"
    ])
    
    with plot_tabs[0]:  # History
        fig = optuna_vis.plot_optimization_history(study)
        st.plotly_chart(fig, use_container_width=True)
    
    with plot_tabs[1]:  # Importance
        try:
            fig = optuna_vis.plot_param_importances(study)
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Недостаточно данных для расчета важности параметров")
    
    with plot_tabs[2]:  # Parallel Coordinates
        fig = optuna_vis.plot_parallel_coordinate(study)
        st.plotly_chart(fig, use_container_width=True)
    
    with plot_tabs[3]:  # Contour
        if len(study.best_params) >= 2:
            param_names = list(study.best_params.keys())[:2]
            fig = optuna_vis.plot_contour(study, params=param_names)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Нужно минимум 2 параметра для contour plot")


def run_optimization_with_ui(
    n_trials: int,
    param_ranges: Dict[str, Tuple[float, float]],
    config_path: str = "configs/main_2024.yaml"
):
    """Run optimization with Streamlit UI updates."""
    
    runner = OptimizationRunner()
    
    # Create progress container
    progress_container = st.container()
    
    with progress_container:
        st.info("🔄 Оптимизация запущена...")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            trials_metric = st.empty()
        with col2:
            best_metric = st.empty()
        with col3:
            time_metric = st.empty()
        with col4:
            improvement_metric = st.empty()
        
        # Log container
        with st.expander("📝 Логи оптимизации", expanded=False):
            log_container = st.empty()
            logs = []
        
        # Start optimization
        runner.run_optimization_thread(
            n_trials=n_trials,
            param_ranges=param_ranges,
            config_path=config_path
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
                    best_metric.metric("🏆 Best Sharpe", f"{best_value:.3f}")
                    
                    elapsed = time.time() - start_time
                    time_metric.metric("⏱️ Время", f"{elapsed:.1f}s")
                    
                    if baseline_value:
                        improvement = (best_value / baseline_value - 1) * 100
                        improvement_metric.metric("📈 Улучшение", f"{improvement:+.1f}%")
                    
                    # Add to log
                    log_entry = f"Trial {completed_trials}: {trial_value:.3f}"
                    logs.append(log_entry)
                    log_container.text("\n".join(logs[-20:]))  # Show last 20 entries
                
                elif update['status'] == 'finished':
                    break
                
                elif update['status'] == 'error':
                    st.error(f"Ошибка: {update['error']}")
                    break
            
            time.sleep(0.1)  # Small delay to not overwhelm UI
        
        # Show final results
        st.success(f"✅ Оптимизация завершена! Лучший Sharpe: {best_value:.3f}")
        
        # Load and visualize results
        if runner.current_study:
            render_optimization_results(runner.current_study)
            
            # Best parameters
            st.subheader("🏆 Лучшие параметры")
            best_params = runner.current_study.best_params
            
            col1, col2 = st.columns(2)
            params_items = list(best_params.items())
            
            with col1:
                for key, value in params_items[:len(params_items)//2]:
                    st.metric(key, f"{value:.4f}" if isinstance(value, float) else value)
            
            with col2:
                for key, value in params_items[len(params_items)//2:]:
                    st.metric(key, f"{value:.4f}" if isinstance(value, float) else value)
            
            return runner.current_study
        
        return None


def validate_on_out_of_sample(
    best_params: Dict[str, Any],
    test_start: str,
    test_end: str,
    config_path: str = "configs/main_2024.yaml"
) -> Dict[str, float]:
    """Validate best parameters on out-of-sample data.
    
    Returns:
        Dictionary with validation metrics
    """
    st.info("🧪 Валидация на out-of-sample данных...")
    
    # Create temporary config with best params
    config = yaml.safe_load(open(config_path))
    
    # Update config with best params
    for param, value in best_params.items():
        if 'zscore' in param or 'rolling' in param:
            config['backtest'][param] = value
        elif 'commission' in param or 'slippage' in param:
            config['backtest'][param] = value
        elif 'stop' in param:
            config['backtest'][param] = value
        # Add more mappings as needed
    
    # Save temporary config
    temp_config = Path("outputs/temp_validation_config.yaml")
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    # Run backtest
    cmd = [
        sys.executable,
        "scripts/trading/run_fixed.py",
        "--period-start", test_start,
        "--period-end", test_end,
        "--config", str(temp_config),
        "--pairs-file", "artifacts/universe/pairs_universe.yaml",
        "--out-dir", "outputs/validation"
    ]
    
    # Execute
    with st.spinner("Запуск валидации..."):
        result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse results
    if result.returncode == 0:
        # Load results
        results_file = Path("outputs/validation/backtest_results.yaml")
        if results_file.exists():
            results = yaml.safe_load(open(results_file))
            
            metrics = {
                'sharpe_ratio': results.get('sharpe_ratio', 0),
                'total_pnl': results.get('total_pnl', 0),
                'win_rate': results.get('win_rate', 0),
                'max_drawdown': results.get('max_drawdown', 0)
            }
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")
            with col2:
                st.metric("Total PnL", f"${metrics['total_pnl']:.2f}")
            with col3:
                st.metric("Win Rate", f"{metrics['win_rate']:.1%}")
            with col4:
                st.metric("Max Drawdown", f"{metrics['max_drawdown']:.1%}")
            
            return metrics
        else:
            st.warning("Результаты валидации не найдены")
    else:
        st.error(f"Ошибка валидации: {result.stderr}")
    
    return {}