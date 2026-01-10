#!/usr/bin/env python3
"""
Auto-ML component for automatic hyperparameter tuning.
Uses advanced techniques like Bayesian optimization, meta-learning, and ensemble methods.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Callable
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass, field
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


@dataclass
class HyperparameterSpace:
    """Defines the hyperparameter search space."""
    name: str
    type: str  # 'float', 'int', 'categorical'
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False
    prior_mean: Optional[float] = None
    prior_std: Optional[float] = None


@dataclass
class MetaFeatures:
    """Meta-features of the dataset/problem for meta-learning."""
    n_pairs: int
    data_frequency: str  # '15T', '1H', etc.
    market_volatility: float
    correlation_mean: float
    correlation_std: float
    trend_strength: float
    seasonality_strength: float
    market_regime: str  # 'bull', 'bear', 'sideways'


class AutoMLOptimizer:
    """Automatic machine learning optimizer for hyperparameter tuning."""
    
    def __init__(self, base_config_path: str = "configs/main_2024.yaml"):
        self.base_config_path = base_config_path
        self.meta_learner = None
        self.surrogate_model = None
        self.hyperparameter_spaces = self._define_search_spaces()
        self.meta_knowledge_base = []
        self.best_configurations = []
        
    def _define_search_spaces(self) -> Dict[str, HyperparameterSpace]:
        """Define intelligent search spaces based on domain knowledge."""
        return {
            # Signal parameters
            'zscore_threshold': HyperparameterSpace(
                name='zscore_threshold',
                type='float',
                low=0.5,
                high=3.0,
                prior_mean=1.5,
                prior_std=0.5
            ),
            'zscore_exit': HyperparameterSpace(
                name='zscore_exit',
                type='float',
                low=-0.5,
                high=0.5,
                prior_mean=0.0,
                prior_std=0.2
            ),
            'rolling_window': HyperparameterSpace(
                name='rolling_window',
                type='int',
                low=10,
                high=100,
                prior_mean=30,
                prior_std=15
            ),
            
            # Risk parameters
            'stop_loss_multiplier': HyperparameterSpace(
                name='stop_loss_multiplier',
                type='float',
                low=2.0,
                high=5.0,
                prior_mean=3.0,
                prior_std=0.5
            ),
            'max_position_size_pct': HyperparameterSpace(
                name='max_position_size_pct',
                type='float',
                low=0.01,
                high=0.10,
                prior_mean=0.03,
                prior_std=0.02,
                log_scale=True
            ),
            
            # Filter parameters
            'coint_pvalue_threshold': HyperparameterSpace(
                name='coint_pvalue_threshold',
                type='float',
                low=0.01,
                high=0.10,
                prior_mean=0.05,
                prior_std=0.02
            ),
            'max_hurst_exponent': HyperparameterSpace(
                name='max_hurst_exponent',
                type='float',
                low=0.4,
                high=0.6,
                prior_mean=0.5,
                prior_std=0.05
            ),
            
            # Adaptive parameters
            'enable_regime_switching': HyperparameterSpace(
                name='enable_regime_switching',
                type='categorical',
                choices=[True, False]
            ),
            'volatility_scaling': HyperparameterSpace(
                name='volatility_scaling',
                type='categorical',
                choices=['none', 'linear', 'sqrt', 'log']
            )
        }
    
    def extract_meta_features(self, data_path: str) -> MetaFeatures:
        """Extract meta-features from the dataset.
        
        Returns:
            MetaFeatures object describing the dataset
        """
        # In production, this would analyze actual data
        # For demo, return synthetic features
        return MetaFeatures(
            n_pairs=100,
            data_frequency='15T',
            market_volatility=np.random.uniform(0.1, 0.3),
            correlation_mean=np.random.uniform(0.3, 0.7),
            correlation_std=np.random.uniform(0.05, 0.15),
            trend_strength=np.random.uniform(0, 1),
            seasonality_strength=np.random.uniform(0, 1),
            market_regime=np.random.choice(['bull', 'bear', 'sideways'])
        )
    
    def suggest_initial_configs(
        self,
        meta_features: MetaFeatures,
        n_suggestions: int = 5
    ) -> List[Dict[str, Any]]:
        """Suggest initial configurations based on meta-learning.
        
        Returns:
            List of suggested parameter configurations
        """
        suggestions = []
        
        # Rule-based suggestions based on meta-features
        if meta_features.market_volatility > 0.2:
            # High volatility - conservative parameters
            suggestions.append({
                'zscore_threshold': 2.0,
                'zscore_exit': 0.0,
                'rolling_window': 50,
                'stop_loss_multiplier': 3.5,
                'max_position_size_pct': 0.02
            })
        else:
            # Low volatility - aggressive parameters
            suggestions.append({
                'zscore_threshold': 1.0,
                'zscore_exit': -0.2,
                'rolling_window': 20,
                'stop_loss_multiplier': 2.5,
                'max_position_size_pct': 0.05
            })
        
        # Trend-following configuration
        if meta_features.trend_strength > 0.6:
            suggestions.append({
                'zscore_threshold': 1.5,
                'zscore_exit': 0.2,
                'rolling_window': 40,
                'stop_loss_multiplier': 4.0,
                'max_position_size_pct': 0.03
            })
        
        # Mean-reversion configuration
        if meta_features.correlation_mean > 0.5:
            suggestions.append({
                'zscore_threshold': 1.8,
                'zscore_exit': -0.1,
                'rolling_window': 30,
                'stop_loss_multiplier': 3.0,
                'max_position_size_pct': 0.04
            })
        
        # Fill remaining with random variations
        while len(suggestions) < n_suggestions:
            config = {}
            for name, space in self.hyperparameter_spaces.items():
                if space.type == 'float':
                    if space.prior_mean:
                        # Sample from prior distribution
                        value = np.random.normal(space.prior_mean, space.prior_std)
                        value = np.clip(value, space.low, space.high)
                    else:
                        value = np.random.uniform(space.low, space.high)
                    config[name] = value
                elif space.type == 'int':
                    if space.prior_mean:
                        value = int(np.random.normal(space.prior_mean, space.prior_std))
                        value = np.clip(value, space.low, space.high)
                    else:
                        value = np.random.randint(space.low, space.high + 1)
                    config[name] = value
                elif space.type == 'categorical':
                    config[name] = np.random.choice(space.choices)
            
            suggestions.append(config)
        
        return suggestions[:n_suggestions]
    
    def build_surrogate_model(self, X: np.ndarray, y: np.ndarray):
        """Build a surrogate model for Bayesian optimization.
        
        Args:
            X: Parameter configurations
            y: Corresponding performance metrics
        """
        # Gaussian Process as surrogate model
        kernel = Matern(length_scale=1.0, nu=2.5)
        self.surrogate_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        
        # Fit the model
        self.surrogate_model.fit(X, y)
    
    def acquisition_function(
        self,
        X: np.ndarray,
        mode: str = 'ei'
    ) -> np.ndarray:
        """Calculate acquisition function values.
        
        Args:
            X: Parameter configurations to evaluate
            mode: Acquisition function type ('ei', 'ucb', 'pi')
        
        Returns:
            Acquisition function values
        """
        if self.surrogate_model is None:
            return np.zeros(X.shape[0])
        
        # Predict mean and std
        mu, sigma = self.surrogate_model.predict(X, return_std=True)
        
        # Current best
        y_best = np.max(self.surrogate_model.y_train_)
        
        if mode == 'ei':  # Expected Improvement
            with np.errstate(divide='warn'):
                Z = (mu - y_best) / sigma
                ei = (mu - y_best) * stats.norm.cdf(Z) + sigma * stats.norm.pdf(Z)
                ei[sigma == 0.0] = 0.0
            return ei
        
        elif mode == 'ucb':  # Upper Confidence Bound
            kappa = 2.0
            return mu + kappa * sigma
        
        elif mode == 'pi':  # Probability of Improvement
            Z = (mu - y_best) / sigma
            return stats.norm.cdf(Z)
        
        return mu
    
    def suggest_next_config(
        self,
        previous_configs: List[Dict[str, Any]],
        previous_results: List[float]
    ) -> Dict[str, Any]:
        """Suggest next configuration using Bayesian optimization.
        
        Returns:
            Suggested parameter configuration
        """
        if len(previous_configs) < 3:
            # Not enough data for surrogate model
            meta_features = self.extract_meta_features("")
            suggestions = self.suggest_initial_configs(meta_features, 1)
            return suggestions[0]
        
        # Convert configs to array
        X = self._configs_to_array(previous_configs)
        y = np.array(previous_results)
        
        # Build surrogate model
        self.build_surrogate_model(X, y)
        
        # Generate candidates
        n_candidates = 1000
        candidates = []
        
        for _ in range(n_candidates):
            config = {}
            for name, space in self.hyperparameter_spaces.items():
                if space.type == 'float':
                    value = np.random.uniform(space.low, space.high)
                    config[name] = value
                elif space.type == 'int':
                    value = np.random.randint(space.low, space.high + 1)
                    config[name] = value
                elif space.type == 'categorical':
                    config[name] = np.random.choice(space.choices)
            candidates.append(config)
        
        # Evaluate acquisition function
        X_candidates = self._configs_to_array(candidates)
        acquisition_values = self.acquisition_function(X_candidates, mode='ei')
        
        # Select best candidate
        best_idx = np.argmax(acquisition_values)
        return candidates[best_idx]
    
    def _configs_to_array(self, configs: List[Dict[str, Any]]) -> np.ndarray:
        """Convert configuration dictionaries to numpy array."""
        X = []
        for config in configs:
            row = []
            for name, space in self.hyperparameter_spaces.items():
                if name in config:
                    value = config[name]
                    if space.type == 'categorical':
                        # One-hot encode
                        value = space.choices.index(value) if value in space.choices else 0
                    row.append(value)
                else:
                    row.append(0)
            X.append(row)
        return np.array(X)
    
    def ensemble_predict(
        self,
        configs: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ensemble prediction using multiple models.
        
        Returns:
            Tuple of (mean predictions, uncertainty estimates)
        """
        predictions = []
        
        # Gaussian Process
        if self.surrogate_model:
            X = self._configs_to_array(configs)
            gp_pred, gp_std = self.surrogate_model.predict(X, return_std=True)
            predictions.append(gp_pred)
        
        # Random Forest (for comparison)
        # In production, would train RF model
        rf_pred = np.random.normal(0.5, 0.2, len(configs))
        predictions.append(rf_pred)
        
        if predictions:
            ensemble_mean = np.mean(predictions, axis=0)
            ensemble_std = np.std(predictions, axis=0)
            return ensemble_mean, ensemble_std
        
        return np.zeros(len(configs)), np.ones(len(configs))


class AutoTuner:
    """Automatic hyperparameter tuner with scheduling."""
    
    def __init__(self, optimizer: AutoMLOptimizer):
        self.optimizer = optimizer
        self.tuning_schedule = []
        self.tuning_history = []
        
    def create_tuning_schedule(
        self,
        start_date: datetime,
        end_date: datetime,
        frequency: str = 'weekly'
    ) -> List[Dict[str, Any]]:
        """Create automated tuning schedule.
        
        Returns:
            List of scheduled tuning tasks
        """
        schedule = []
        current_date = start_date
        
        if frequency == 'daily':
            delta = timedelta(days=1)
        elif frequency == 'weekly':
            delta = timedelta(weeks=1)
        elif frequency == 'monthly':
            delta = timedelta(days=30)
        else:
            delta = timedelta(weeks=1)
        
        while current_date <= end_date:
            schedule.append({
                'date': current_date,
                'status': 'pending',
                'config': None,
                'result': None
            })
            current_date += delta
        
        self.tuning_schedule = schedule
        return schedule
    
    def run_scheduled_tuning(self, date: datetime) -> Dict[str, Any]:
        """Run tuning for a scheduled date.
        
        Returns:
            Tuning results
        """
        # Get previous results
        previous_configs = [h['config'] for h in self.tuning_history if h['config']]
        previous_results = [h['result'] for h in self.tuning_history if h['result']]
        
        # Suggest new configuration
        new_config = self.optimizer.suggest_next_config(
            previous_configs,
            previous_results
        )
        
        # Simulate evaluation (in production, would run actual backtest)
        result = np.random.normal(0.8, 0.2)
        
        # Store in history
        self.tuning_history.append({
            'date': date,
            'config': new_config,
            'result': result
        })
        
        return {
            'date': date,
            'config': new_config,
            'result': result,
            'improvement': result - np.mean(previous_results) if previous_results else 0
        }


def render_automl_ui():
    """Render Auto-ML optimization UI in Streamlit."""
    
    st.title("🤖 Auto-ML Hyperparameter Optimization")
    st.caption("Intelligent automatic hyperparameter tuning using Bayesian optimization and meta-learning")
    
    # Initialize AutoML
    if 'automl_optimizer' not in st.session_state:
        st.session_state.automl_optimizer = AutoMLOptimizer()
        st.session_state.auto_tuner = AutoTuner(st.session_state.automl_optimizer)
    
    optimizer = st.session_state.automl_optimizer
    tuner = st.session_state.auto_tuner
    
    # Tabs
    tabs = st.tabs([
        "🎯 Auto-Tune",
        "📊 Meta-Learning",
        "🔮 Bayesian Optimization",
        "📅 Scheduled Tuning",
        "📈 Performance Analysis"
    ])
    
    with tabs[0]:  # Auto-Tune
        st.subheader("🎯 Automatic Parameter Tuning")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Configuration")
            
            # Data selection
            data_path = st.text_input(
                "📁 Data Path",
                value="data_downloaded/",
                help="Path to historical data"
            )
            
            # Extract meta-features
            if st.button("🔍 Analyze Data"):
                meta_features = optimizer.extract_meta_features(data_path)
                st.session_state.meta_features = meta_features
                st.success("✅ Meta-features extracted")
            
            # Show meta-features
            if 'meta_features' in st.session_state:
                meta = st.session_state.meta_features
                
                st.markdown("**Dataset Characteristics:**")
                st.write(f"- Pairs: {meta.n_pairs}")
                st.write(f"- Frequency: {meta.data_frequency}")
                st.write(f"- Volatility: {meta.market_volatility:.3f}")
                st.write(f"- Correlation: {meta.correlation_mean:.3f} ± {meta.correlation_std:.3f}")
                st.write(f"- Market Regime: {meta.market_regime}")
        
        with col2:
            st.markdown("### Optimization Settings")
            
            # Optimization mode
            opt_mode = st.selectbox(
                "Optimization Mode",
                ["Quick (10 trials)", "Standard (50 trials)", "Deep (100 trials)", "Custom"]
            )
            
            if opt_mode == "Custom":
                n_trials = st.number_input("Number of trials", 5, 500, 50)
            else:
                n_trials = 10 if "Quick" in opt_mode else 50 if "Standard" in opt_mode else 100
            
            # Acquisition function
            acq_function = st.selectbox(
                "Acquisition Function",
                ["Expected Improvement", "Upper Confidence Bound", "Probability of Improvement"]
            )
            
            # Advanced options
            with st.expander("Advanced Options"):
                use_ensemble = st.checkbox("Use ensemble models", value=True)
                use_meta_learning = st.checkbox("Use meta-learning", value=True)
                adaptive_sampling = st.checkbox("Adaptive sampling", value=True)
        
        # Run optimization
        st.markdown("---")
        
        if st.button("🚀 START AUTO-ML OPTIMIZATION", type="primary", use_container_width=True):
            if 'meta_features' not in st.session_state:
                st.error("Please analyze data first")
            else:
                with st.spinner("🤖 Running intelligent optimization..."):
                    # Get initial suggestions
                    meta_features = st.session_state.meta_features
                    initial_configs = optimizer.suggest_initial_configs(meta_features, 5)
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Results container
                    results = []
                    configs = []
                    
                    # Run optimization trials
                    for i in range(n_trials):
                        progress = (i + 1) / n_trials
                        progress_bar.progress(progress)
                        status_text.text(f"Trial {i+1}/{n_trials}")
                        
                        if i < len(initial_configs):
                            # Use initial suggestions
                            config = initial_configs[i]
                        else:
                            # Use Bayesian optimization
                            config = optimizer.suggest_next_config(configs, results)
                        
                        # Simulate evaluation (in production, run actual backtest)
                        result = np.random.normal(0.8, 0.2) + len(results) * 0.01  # Simulate improvement
                        
                        configs.append(config)
                        results.append(result)
                    
                    # Store results
                    st.session_state.automl_results = {
                        'configs': configs,
                        'results': results,
                        'best_idx': np.argmax(results),
                        'best_config': configs[np.argmax(results)],
                        'best_result': np.max(results)
                    }
                    
                    st.success(f"✅ Optimization complete! Best result: {np.max(results):.3f}")
        
        # Show results
        if 'automl_results' in st.session_state:
            results = st.session_state.automl_results
            
            st.markdown("---")
            st.subheader("📊 Optimization Results")
            
            # Best configuration
            st.markdown("**Best Configuration:**")
            best_config_df = pd.DataFrame([results['best_config']])
            st.dataframe(best_config_df.T, use_container_width=True)
            
            # Convergence plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=list(range(1, len(results['results']) + 1)),
                y=results['results'],
                mode='markers+lines',
                name='Trial Results',
                marker=dict(size=8)
            ))
            
            # Mark best
            fig.add_trace(go.Scatter(
                x=[results['best_idx'] + 1],
                y=[results['best_result']],
                mode='markers',
                name='Best Result',
                marker=dict(size=15, color='red', symbol='star')
            ))
            
            fig.update_layout(
                title="Optimization Convergence",
                xaxis_title="Trial",
                yaxis_title="Performance",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:  # Meta-Learning
        st.subheader("📊 Meta-Learning Insights")
        
        # Knowledge base
        st.markdown("### Knowledge Base")
        
        # Simulate meta-knowledge
        knowledge_base = [
            {
                'dataset': 'High Volatility Crypto',
                'best_params': {'zscore_threshold': 2.0, 'rolling_window': 50},
                'performance': 0.85,
                'characteristics': 'High volatility, trending market'
            },
            {
                'dataset': 'Stable Pairs',
                'best_params': {'zscore_threshold': 1.2, 'rolling_window': 20},
                'performance': 0.92,
                'characteristics': 'Low volatility, mean-reverting'
            },
            {
                'dataset': 'Mixed Market',
                'best_params': {'zscore_threshold': 1.5, 'rolling_window': 30},
                'performance': 0.78,
                'characteristics': 'Mixed volatility, regime changes'
            }
        ]
        
        kb_df = pd.DataFrame(knowledge_base)
        st.dataframe(kb_df, use_container_width=True)
        
        # Transfer learning
        st.markdown("---")
        st.markdown("### Transfer Learning")
        
        source_dataset = st.selectbox(
            "Select source dataset for transfer",
            [kb['dataset'] for kb in knowledge_base]
        )
        
        if st.button("🔄 Apply Transfer Learning"):
            selected_kb = next(kb for kb in knowledge_base if kb['dataset'] == source_dataset)
            st.success(f"✅ Transferred parameters from {source_dataset}")
            st.json(selected_kb['best_params'])
    
    with tabs[2]:  # Bayesian Optimization
        st.subheader("🔮 Bayesian Optimization Details")
        
        if 'automl_results' in st.session_state:
            results = st.session_state.automl_results
            
            # Surrogate model visualization
            st.markdown("### Surrogate Model Predictions")
            
            # Build surrogate model
            X = optimizer._configs_to_array(results['configs'])
            y = np.array(results['results'])
            
            if len(X) > 3:
                optimizer.build_surrogate_model(X, y)
                
                # Generate test points
                n_test = 100
                test_configs = []
                for _ in range(n_test):
                    config = {}
                    for name, space in optimizer.hyperparameter_spaces.items():
                        if space.type == 'float':
                            config[name] = np.random.uniform(space.low, space.high)
                        elif space.type == 'int':
                            config[name] = np.random.randint(space.low, space.high + 1)
                    test_configs.append(config)
                
                X_test = optimizer._configs_to_array(test_configs)
                
                # Predict
                mu, sigma = optimizer.surrogate_model.predict(X_test, return_std=True)
                
                # Acquisition function values
                ei = optimizer.acquisition_function(X_test, mode='ei')
                
                # Visualization
                param_name = 'zscore_threshold'
                param_values = [c[param_name] for c in test_configs]
                
                fig = go.Figure()
                
                # Mean predictions
                fig.add_trace(go.Scatter(
                    x=param_values,
                    y=mu,
                    mode='markers',
                    name='Predicted Mean',
                    marker=dict(size=5, color=mu, colorscale='Viridis')
                ))
                
                # Uncertainty bands
                fig.add_trace(go.Scatter(
                    x=param_values,
                    y=mu + 2*sigma,
                    mode='lines',
                    line=dict(width=0.5, color='gray'),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=param_values,
                    y=mu - 2*sigma,
                    mode='lines',
                    line=dict(width=0.5, color='gray'),
                    fill='tonexty',
                    name='95% CI'
                ))
                
                # Actual observations
                actual_params = [c[param_name] for c in results['configs']]
                fig.add_trace(go.Scatter(
                    x=actual_params,
                    y=results['results'],
                    mode='markers',
                    name='Observations',
                    marker=dict(size=10, color='red', symbol='x')
                ))
                
                fig.update_layout(
                    title=f"Surrogate Model: {param_name}",
                    xaxis_title=param_name,
                    yaxis_title="Predicted Performance",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Acquisition function plot
                st.markdown("### Acquisition Function")
                
                fig2 = go.Figure()
                
                fig2.add_trace(go.Scatter(
                    x=param_values,
                    y=ei,
                    mode='markers',
                    name='Expected Improvement',
                    marker=dict(size=5, color=ei, colorscale='Hot')
                ))
                
                fig2.update_layout(
                    title="Expected Improvement",
                    xaxis_title=param_name,
                    yaxis_title="EI Value",
                    height=300
                )
                
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Run optimization first to see Bayesian optimization details")
    
    with tabs[3]:  # Scheduled Tuning
        st.subheader("📅 Automated Scheduled Tuning")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Schedule Configuration")
            
            start_date = st.date_input("Start Date", value=datetime.now().date())
            end_date = st.date_input("End Date", value=(datetime.now() + timedelta(days=30)).date())
            
            frequency = st.selectbox(
                "Tuning Frequency",
                ["Daily", "Weekly", "Monthly"]
            )
            
            if st.button("📅 Create Schedule"):
                schedule = tuner.create_tuning_schedule(
                    datetime.combine(start_date, datetime.min.time()),
                    datetime.combine(end_date, datetime.min.time()),
                    frequency.lower()
                )
                st.session_state.tuning_schedule = schedule
                st.success(f"✅ Created {len(schedule)} tuning tasks")
        
        with col2:
            st.markdown("### Schedule Status")
            
            if 'tuning_schedule' in st.session_state:
                schedule = st.session_state.tuning_schedule
                
                # Status summary
                pending = len([s for s in schedule if s['status'] == 'pending'])
                completed = len([s for s in schedule if s['status'] == 'completed'])
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("📅 Pending", pending)
                with col_b:
                    st.metric("✅ Completed", completed)
                
                # Next scheduled
                next_task = next((s for s in schedule if s['status'] == 'pending'), None)
                if next_task:
                    st.info(f"Next tuning: {next_task['date'].strftime('%Y-%m-%d')}")
        
        # Schedule table
        if 'tuning_schedule' in st.session_state:
            st.markdown("---")
            st.markdown("### Tuning Schedule")
            
            schedule_df = pd.DataFrame(st.session_state.tuning_schedule)
            schedule_df['date'] = pd.to_datetime(schedule_df['date']).dt.strftime('%Y-%m-%d')
            
            st.dataframe(schedule_df[['date', 'status']], use_container_width=True)
            
            # Run scheduled tuning
            if st.button("▶️ Run Next Scheduled Tuning"):
                next_task = next((s for s in st.session_state.tuning_schedule 
                                 if s['status'] == 'pending'), None)
                
                if next_task:
                    result = tuner.run_scheduled_tuning(next_task['date'])
                    next_task['status'] = 'completed'
                    next_task['config'] = result['config']
                    next_task['result'] = result['result']
                    
                    st.success(f"✅ Tuning completed! Result: {result['result']:.3f}")
                    
                    if result['improvement'] > 0:
                        st.balloons()
                        st.success(f"🎉 Improvement: {result['improvement']:.3f}")
                else:
                    st.info("No pending tuning tasks")
    
    with tabs[4]:  # Performance Analysis
        st.subheader("📈 Auto-ML Performance Analysis")
        
        if tuner.tuning_history:
            # Convert to DataFrame
            history_df = pd.DataFrame(tuner.tuning_history)
            
            # Performance over time
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=history_df['date'],
                y=history_df['result'],
                mode='lines+markers',
                name='Performance',
                line=dict(width=2)
            ))
            
            # Add trend line
            z = np.polyfit(range(len(history_df)), history_df['result'], 1)
            p = np.poly1d(z)
            
            fig.add_trace(go.Scatter(
                x=history_df['date'],
                y=p(range(len(history_df))),
                mode='lines',
                name='Trend',
                line=dict(dash='dash', color='red')
            ))
            
            fig.update_layout(
                title="Performance Evolution",
                xaxis_title="Date",
                yaxis_title="Performance Metric",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Parameter evolution
            st.markdown("---")
            st.markdown("### Parameter Evolution")
            
            # Extract parameter values over time
            param_evolution = []
            for entry in tuner.tuning_history:
                if entry['config']:
                    row = {'date': entry['date']}
                    row.update(entry['config'])
                    param_evolution.append(row)
            
            if param_evolution:
                param_df = pd.DataFrame(param_evolution)
                
                # Select parameter to visualize
                numeric_cols = [col for col in param_df.columns 
                               if col != 'date' and param_df[col].dtype in ['float64', 'int64']]
                
                selected_param = st.selectbox("Select parameter", numeric_cols)
                
                if selected_param:
                    fig2 = go.Figure()
                    
                    fig2.add_trace(go.Scatter(
                        x=param_df['date'],
                        y=param_df[selected_param],
                        mode='lines+markers',
                        name=selected_param
                    ))
                    
                    fig2.update_layout(
                        title=f"Evolution of {selected_param}",
                        xaxis_title="Date",
                        yaxis_title="Value",
                        height=300
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No tuning history available yet")


if __name__ == "__main__":
    render_automl_ui()