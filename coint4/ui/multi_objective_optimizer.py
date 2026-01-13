#!/usr/bin/env python3
"""
Multi-objective optimization component for Streamlit UI.
Optimizes multiple metrics simultaneously using Pareto frontier.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import optuna
    from optuna.multi_objective import trial as multi_trial
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


@dataclass
class OptimizationObjective:
    """Represents an optimization objective."""
    name: str
    direction: str  # 'maximize' or 'minimize'
    weight: float = 1.0
    threshold: Optional[float] = None  # Minimum acceptable value


class MultiObjectiveOptimizer:
    """Multi-objective optimization using Optuna."""
    
    def __init__(self, base_config_path: str = "configs/main_2024.yaml"):
        self.base_config_path = base_config_path
        self.study = None
        self.objectives = []
        
    def add_objective(
        self,
        name: str,
        direction: str = "maximize",
        weight: float = 1.0,
        threshold: Optional[float] = None
    ):
        """Add an optimization objective.
        
        Args:
            name: Metric name (sharpe_ratio, total_pnl, win_rate, etc.)
            direction: 'maximize' or 'minimize'
            weight: Relative importance weight
            threshold: Minimum acceptable value
        """
        self.objectives.append(
            OptimizationObjective(name, direction, weight, threshold)
        )
    
    def create_multi_objective_function(
        self,
        param_ranges: Dict[str, Tuple[float, float]]
    ):
        """Create multi-objective function for Optuna.
        
        Returns:
            Function that returns multiple objective values
        """
        from scripts.optimization.web_optimizer import WebOptimizer
        
        # Initialize optimizer
        optimizer = WebOptimizer(
            base_config_path=self.base_config_path,
            search_space_path="configs/search_spaces/web_ui.yaml"
        )
        
        def multi_objective(trial):
            """Multi-objective function."""
            
            # Sample parameters
            params = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                else:
                    params[param_name] = trial.suggest_float(param_name, min_val, max_val)
            
            # Run backtest and get multiple metrics
            try:
                # This would call the actual backtest
                # For now, simulate multiple metrics
                results = self._run_backtest_for_metrics(params)
                
                # Extract objective values
                objective_values = []
                for obj in self.objectives:
                    value = results.get(obj.name, 0)
                    
                    # Apply direction
                    if obj.direction == "minimize":
                        value = -value
                    
                    # Apply weight
                    value *= obj.weight
                    
                    # Check threshold
                    if obj.threshold is not None:
                        if obj.direction == "maximize" and value < obj.threshold:
                            return [-float('inf')] * len(self.objectives)
                        elif obj.direction == "minimize" and value > obj.threshold:
                            return [float('inf')] * len(self.objectives)
                    
                    objective_values.append(value)
                
                return objective_values
                
            except Exception as e:
                # Return bad values for all objectives
                return [-float('inf')] * len(self.objectives)
        
        return multi_objective
    
    def _run_backtest_for_metrics(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Run backtest and return multiple metrics.
        
        This should integrate with the actual backtesting engine.
        For now, returns simulated metrics.
        """
        # Simulate metrics based on parameters
        # In production, this would run actual backtest
        
        base_sharpe = 0.5 + params.get('zscore_threshold', 1.5) * 0.2
        base_pnl = 10000 * (1 + params.get('rolling_window', 30) / 100)
        base_win_rate = 0.45 + params.get('zscore_exit', 0) * 0.1
        base_drawdown = 0.15 - params.get('stop_loss_multiplier', 3) * 0.02
        
        # Add some randomness for simulation
        import random
        noise = random.gauss(0, 0.1)
        
        return {
            'sharpe_ratio': base_sharpe + noise,
            'total_pnl': base_pnl * (1 + noise),
            'win_rate': min(max(base_win_rate + noise * 0.1, 0), 1),
            'max_drawdown': max(base_drawdown + noise * 0.05, 0.05),
            'calmar_ratio': (base_sharpe + noise) / max(base_drawdown, 0.01),
            'sortino_ratio': (base_sharpe + noise) * 1.2,
            'profit_factor': 1.5 + noise * 0.3,
            'recovery_factor': 2.0 + noise * 0.5
        }
    
    def optimize(
        self,
        n_trials: int,
        param_ranges: Dict[str, Tuple[float, float]],
        n_jobs: int = 1
    ) -> optuna.Study:
        """Run multi-objective optimization.
        
        Args:
            n_trials: Number of trials
            param_ranges: Parameter ranges
            n_jobs: Number of parallel jobs
        
        Returns:
            Optuna study with Pareto front solutions
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available")
        
        # Create study with multiple objectives
        directions = [obj.direction for obj in self.objectives]
        
        study_name = f"multi_obj_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        studies_dir = Path("outputs/studies")
        studies_dir.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{studies_dir / f'{study_name}.db'}"
        
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            directions=directions,
            load_if_exists=False
        )
        
        # Create objective function
        objective = self.create_multi_objective_function(param_ranges)
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
        
        return self.study
    
    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """Get Pareto optimal solutions.
        
        Returns:
            List of Pareto optimal solutions with their parameters and values
        """
        if not self.study:
            return []
        
        pareto_front = []
        
        # Get best trials (Pareto optimal)
        best_trials = self.study.best_trials
        
        for trial in best_trials:
            solution = {
                'trial_number': trial.number,
                'parameters': trial.params,
                'values': {}
            }
            
            # Map values back to objective names
            for i, obj in enumerate(self.objectives):
                value = trial.values[i]
                # Reverse transformation for minimized objectives
                if obj.direction == "minimize":
                    value = -value
                # Remove weight
                value /= obj.weight
                solution['values'][obj.name] = value
            
            pareto_front.append(solution)
        
        return pareto_front
    
    def visualize_pareto_front(self) -> go.Figure:
        """Create Pareto front visualization.
        
        Returns:
            Plotly figure with Pareto front
        """
        if not self.study or len(self.objectives) < 2:
            return go.Figure()
        
        pareto_solutions = self.get_pareto_front()
        
        if len(self.objectives) == 2:
            # 2D Pareto front
            obj1_name = self.objectives[0].name
            obj2_name = self.objectives[1].name
            
            obj1_values = [s['values'][obj1_name] for s in pareto_solutions]
            obj2_values = [s['values'][obj2_name] for s in pareto_solutions]
            
            fig = go.Figure()
            
            # All trials
            all_obj1 = []
            all_obj2 = []
            for trial in self.study.trials:
                if trial.values and len(trial.values) >= 2:
                    val1 = trial.values[0] / self.objectives[0].weight
                    val2 = trial.values[1] / self.objectives[1].weight
                    if self.objectives[0].direction == "minimize":
                        val1 = -val1
                    if self.objectives[1].direction == "minimize":
                        val2 = -val2
                    all_obj1.append(val1)
                    all_obj2.append(val2)
            
            # Plot all trials
            fig.add_trace(go.Scatter(
                x=all_obj1,
                y=all_obj2,
                mode='markers',
                name='All Trials',
                marker=dict(size=6, color='lightgray'),
                opacity=0.5
            ))
            
            # Plot Pareto front
            fig.add_trace(go.Scatter(
                x=obj1_values,
                y=obj2_values,
                mode='markers+lines',
                name='Pareto Front',
                marker=dict(size=10, color='red'),
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title="Pareto Front",
                xaxis_title=obj1_name.replace('_', ' ').title(),
                yaxis_title=obj2_name.replace('_', ' ').title(),
                height=500
            )
            
        elif len(self.objectives) == 3:
            # 3D Pareto front
            obj1_name = self.objectives[0].name
            obj2_name = self.objectives[1].name
            obj3_name = self.objectives[2].name
            
            obj1_values = [s['values'][obj1_name] for s in pareto_solutions]
            obj2_values = [s['values'][obj2_name] for s in pareto_solutions]
            obj3_values = [s['values'][obj3_name] for s in pareto_solutions]
            
            fig = go.Figure(data=[
                go.Scatter3d(
                    x=obj1_values,
                    y=obj2_values,
                    z=obj3_values,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=obj3_values,
                        colorscale='Viridis',
                        showscale=True
                    ),
                    text=[f"Trial {s['trial_number']}" for s in pareto_solutions],
                    hovertemplate='<b>%{text}</b><br>' +
                                  f'{obj1_name}: %{{x:.3f}}<br>' +
                                  f'{obj2_name}: %{{y:.3f}}<br>' +
                                  f'{obj3_name}: %{{z:.3f}}'
                )
            ])
            
            fig.update_layout(
                title="3D Pareto Front",
                scene=dict(
                    xaxis_title=obj1_name.replace('_', ' ').title(),
                    yaxis_title=obj2_name.replace('_', ' ').title(),
                    zaxis_title=obj3_name.replace('_', ' ').title()
                ),
                height=600
            )
        else:
            # Parallel coordinates for > 3 objectives
            import plotly.express as px
            
            data = []
            for solution in pareto_solutions:
                row = {'trial': solution['trial_number']}
                row.update(solution['values'])
                data.append(row)
            
            df = pd.DataFrame(data)
            
            dimensions = [
                dict(label=obj.name.replace('_', ' ').title(),
                     values=df[obj.name])
                for obj in self.objectives
            ]
            
            fig = go.Figure(data=
                go.Parcoords(
                    line=dict(color=df['trial'], colorscale='Viridis'),
                    dimensions=dimensions
                )
            )
            
            fig.update_layout(
                title="Pareto Front (Parallel Coordinates)",
                height=500
            )
        
        return fig
    
    def select_solution(
        self,
        preference_weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """Select best solution from Pareto front based on preferences.
        
        Args:
            preference_weights: Weights for each objective
        
        Returns:
            Selected solution with parameters and metrics
        """
        pareto_solutions = self.get_pareto_front()
        
        if not pareto_solutions:
            return {}
        
        if preference_weights is None:
            # Equal weights for all objectives
            preference_weights = {
                obj.name: 1.0 / len(self.objectives)
                for obj in self.objectives
            }
        
        # Calculate weighted score for each solution
        best_score = -float('inf')
        best_solution = None
        
        for solution in pareto_solutions:
            score = 0
            for obj_name, weight in preference_weights.items():
                if obj_name in solution['values']:
                    # Normalize value (simple min-max scaling)
                    value = solution['values'][obj_name]
                    score += weight * value
            
            if score > best_score:
                best_score = score
                best_solution = solution
        
        return best_solution


def render_multi_objective_ui():
    """Render multi-objective optimization UI in Streamlit."""
    
    st.title("🎯 Multi-Objective Оптимизация")
    st.caption("Оптимизируйте несколько метрик одновременно с помощью Pareto frontier")
    
    if not OPTUNA_AVAILABLE:
        st.error("⚠️ Optuna не установлен. Установите: pip install optuna")
        return
    
    # Initialize optimizer
    if 'multi_optimizer' not in st.session_state:
        st.session_state.multi_optimizer = MultiObjectiveOptimizer()
    
    optimizer = st.session_state.multi_optimizer
    
    # Settings columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Выбор метрик")
        
        # Metric selection
        available_metrics = [
            "Sharpe Ratio",
            "Total PnL",
            "Win Rate",
            "Calmar Ratio",
            "Sortino Ratio",
            "Max Drawdown",
            "Profit Factor",
            "Recovery Factor"
        ]
        
        selected_metrics = st.multiselect(
            "Выберите метрики для оптимизации",
            available_metrics,
            default=["Sharpe Ratio", "Max Drawdown"]
        )
        
        if len(selected_metrics) < 2:
            st.warning("⚠️ Выберите минимум 2 метрики")
        
        # Configure each metric
        st.markdown("---")
        st.markdown("**Настройки метрик:**")
        
        metric_configs = {}
        for metric in selected_metrics:
            with st.expander(f"⚙️ {metric}"):
                col_a, col_b = st.columns(2)
                
                direction = col_a.selectbox(
                    "Направление",
                    ["Maximize", "Minimize"],
                    key=f"dir_{metric}",
                    index=0 if "Drawdown" not in metric else 1
                )
                
                weight = col_b.slider(
                    "Вес",
                    min_value=0.1,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    key=f"weight_{metric}"
                )
                
                use_threshold = st.checkbox(
                    "Использовать порог",
                    key=f"thresh_check_{metric}"
                )
                
                threshold = None
                if use_threshold:
                    if "Sharpe" in metric:
                        default_val = 0.5
                    elif "Win Rate" in metric:
                        default_val = 0.4
                    elif "Drawdown" in metric:
                        default_val = 0.2
                    else:
                        default_val = 0.0
                    
                    threshold = st.number_input(
                        "Минимальное значение",
                        value=default_val,
                        key=f"thresh_{metric}"
                    )
                
                metric_configs[metric] = {
                    'direction': direction.lower(),
                    'weight': weight,
                    'threshold': threshold
                }
    
    with col2:
        st.subheader("🎛️ Параметры оптимизации")
        
        # Number of trials
        n_trials = st.slider(
            "Количество trials",
            min_value=50,
            max_value=500,
            value=100,
            step=50
        )
        
        # Parallel jobs
        n_jobs = st.slider(
            "Параллельные воркеры",
            min_value=1,
            max_value=8,
            value=4
        )
        
        st.markdown("---")
        st.markdown("**Диапазоны параметров:**")
        
        # Parameter ranges (simplified)
        param_ranges = {}
        
        with st.expander("📈 Сигналы", expanded=True):
            zscore_range = st.slider(
                "Z-score threshold",
                min_value=0.5, max_value=3.0,
                value=(1.0, 2.0), step=0.1,
                key="mo_zscore"
            )
            param_ranges['zscore_threshold'] = zscore_range
            
            window_range = st.slider(
                "Rolling window",
                min_value=10, max_value=100,
                value=(20, 50), step=5,
                key="mo_window"
            )
            param_ranges['rolling_window'] = window_range
        
        with st.expander("🛡️ Риски"):
            stop_loss_range = st.slider(
                "Stop loss multiplier",
                min_value=2.0, max_value=5.0,
                value=(2.5, 3.5), step=0.5,
                key="mo_stop"
            )
            param_ranges['stop_loss_multiplier'] = stop_loss_range
    
    # Run optimization
    st.markdown("---")
    
    if st.button("🚀 ЗАПУСТИТЬ MULTI-OBJECTIVE ОПТИМИЗАЦИЮ", type="primary", use_container_width=True):
        if len(selected_metrics) < 2:
            st.error("❌ Выберите минимум 2 метрики")
            return
        
        with st.spinner("🔄 Выполняется multi-objective оптимизация..."):
            # Clear previous objectives
            optimizer.objectives = []
            
            # Add objectives
            for metric, config in metric_configs.items():
                metric_key = metric.lower().replace(' ', '_')
                optimizer.add_objective(
                    name=metric_key,
                    direction=config['direction'],
                    weight=config['weight'],
                    threshold=config['threshold']
                )
            
            # Run optimization
            study = optimizer.optimize(
                n_trials=n_trials,
                param_ranges=param_ranges,
                n_jobs=n_jobs
            )
            
            st.success(f"✅ Оптимизация завершена! Найдено {len(study.best_trials)} Pareto-оптимальных решений")
            
            # Store in session state
            st.session_state['multi_study'] = study
            st.session_state['pareto_solutions'] = optimizer.get_pareto_front()
    
    # Show results
    if 'pareto_solutions' in st.session_state and st.session_state['pareto_solutions']:
        st.markdown("---")
        st.subheader("📊 Pareto Front")
        
        # Visualization
        fig = optimizer.visualize_pareto_front()
        st.plotly_chart(fig, use_container_width=True)
        
        # Solution selection
        st.markdown("---")
        st.subheader("🎯 Выбор решения")
        
        selection_method = st.radio(
            "Метод выбора",
            ["Равные веса", "Пользовательские веса", "Интерактивный выбор"],
            horizontal=True
        )
        
        if selection_method == "Равные веса":
            selected = optimizer.select_solution()
        
        elif selection_method == "Пользовательские веса":
            st.markdown("**Укажите предпочтения (веса):**")
            
            preference_weights = {}
            cols = st.columns(len(selected_metrics))
            
            for i, metric in enumerate(selected_metrics):
                with cols[i]:
                    weight = st.slider(
                        metric,
                        min_value=0.0,
                        max_value=1.0,
                        value=1.0/len(selected_metrics),
                        key=f"pref_{metric}"
                    )
                    preference_weights[metric.lower().replace(' ', '_')] = weight
            
            # Normalize weights
            total_weight = sum(preference_weights.values())
            if total_weight > 0:
                preference_weights = {
                    k: v/total_weight 
                    for k, v in preference_weights.items()
                }
            
            selected = optimizer.select_solution(preference_weights)
        
        else:  # Interactive selection
            # Show Pareto solutions table
            solutions_data = []
            for sol in st.session_state['pareto_solutions']:
                row = {'Trial': sol['trial_number']}
                row.update(sol['values'])
                solutions_data.append(row)
            
            solutions_df = pd.DataFrame(solutions_data)
            
            st.markdown("**Pareto-оптимальные решения:**")
            selected_row = st.selectbox(
                "Выберите решение",
                range(len(solutions_df)),
                format_func=lambda x: f"Trial {solutions_df.iloc[x]['Trial']}"
            )
            
            selected = st.session_state['pareto_solutions'][selected_row]
        
        # Display selected solution
        if selected:
            st.markdown("---")
            st.subheader("✅ Выбранное решение")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            metric_items = list(selected['values'].items())
            
            for i, (metric, value) in enumerate(metric_items):
                col_idx = i % 3
                if col_idx == 0:
                    col = col1
                elif col_idx == 1:
                    col = col2
                else:
                    col = col3
                
                with col:
                    st.metric(
                        metric.replace('_', ' ').title(),
                        f"{value:.3f}"
                    )
            
            # Parameters
            st.markdown("**Параметры:**")
            params_df = pd.DataFrame([
                {"Parameter": k, "Value": v}
                for k, v in selected['parameters'].items()
            ])
            st.dataframe(params_df, use_container_width=True)
            
            # Export
            col1, col2 = st.columns(2)
            
            with col1:
                yaml_str = yaml.dump(selected, default_flow_style=False)
                st.download_button(
                    "📥 Скачать решение (YAML)",
                    yaml_str,
                    file_name=f"pareto_solution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml",
                    mime="text/yaml"
                )
            
            with col2:
                json_str = json.dumps(selected, indent=2)
                st.download_button(
                    "📥 Скачать решение (JSON)",
                    json_str,
                    file_name=f"pareto_solution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )


if __name__ == "__main__":
    render_multi_objective_ui()
