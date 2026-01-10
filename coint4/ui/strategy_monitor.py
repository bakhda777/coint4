#!/usr/bin/env python3
"""
Strategy performance monitoring and degradation detection system.
Tracks strategy performance in real-time and alerts on degradation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Dict, Any, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from dataclasses import dataclass, field
import warnings
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: datetime
    sharpe_ratio: float
    total_pnl: float
    win_rate: float
    max_drawdown: float
    n_trades: int
    avg_trade_pnl: float
    volatility: float
    calmar_ratio: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'sharpe_ratio': self.sharpe_ratio,
            'total_pnl': self.total_pnl,
            'win_rate': self.win_rate,
            'max_drawdown': self.max_drawdown,
            'n_trades': self.n_trades,
            'avg_trade_pnl': self.avg_trade_pnl,
            'volatility': self.volatility,
            'calmar_ratio': self.calmar_ratio
        }


@dataclass
class DegradationAlert:
    """Alert for strategy degradation."""
    timestamp: datetime
    severity: str  # 'warning', 'critical', 'emergency'
    metric: str
    current_value: float
    baseline_value: float
    threshold: float
    message: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity,
            'metric': self.metric,
            'current_value': self.current_value,
            'baseline_value': self.baseline_value,
            'threshold': self.threshold,
            'message': self.message
        }


class StrategyMonitor:
    """Monitors strategy performance and detects degradation."""
    
    def __init__(
        self,
        baseline_window: int = 30,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        """Initialize monitor.
        
        Args:
            baseline_window: Days for baseline calculation
            alert_thresholds: Thresholds for alerts
        """
        self.baseline_window = baseline_window
        self.performance_history: List[PerformanceMetrics] = []
        self.alerts: List[DegradationAlert] = []
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'sharpe_ratio_drop': 0.3,  # 30% drop
            'win_rate_drop': 0.1,  # 10% absolute drop
            'drawdown_increase': 0.5,  # 50% increase
            'pnl_drop': 0.4,  # 40% drop
            'volatility_spike': 2.0,  # 2x increase
            'trades_drop': 0.5  # 50% drop in trade count
        }
        
        # Statistical thresholds
        self.z_score_threshold = 2.0  # For anomaly detection
        self.cusum_threshold = 5.0  # For change detection
        
    def add_performance_data(self, metrics: PerformanceMetrics):
        """Add new performance data point."""
        self.performance_history.append(metrics)
        
        # Check for degradation
        if len(self.performance_history) > self.baseline_window:
            alerts = self.check_degradation(metrics)
            self.alerts.extend(alerts)
    
    def calculate_baseline(self) -> Dict[str, float]:
        """Calculate baseline metrics from historical data.
        
        Returns:
            Dictionary of baseline metrics
        """
        if len(self.performance_history) < self.baseline_window:
            return {}
        
        # Get baseline period
        baseline_data = self.performance_history[-self.baseline_window:]
        
        baseline = {
            'sharpe_ratio': np.mean([m.sharpe_ratio for m in baseline_data]),
            'win_rate': np.mean([m.win_rate for m in baseline_data]),
            'max_drawdown': np.mean([m.max_drawdown for m in baseline_data]),
            'total_pnl': np.mean([m.total_pnl for m in baseline_data]),
            'volatility': np.mean([m.volatility for m in baseline_data]),
            'n_trades': np.mean([m.n_trades for m in baseline_data])
        }
        
        # Add standard deviations for statistical testing
        baseline['sharpe_std'] = np.std([m.sharpe_ratio for m in baseline_data])
        baseline['pnl_std'] = np.std([m.total_pnl for m in baseline_data])
        
        return baseline
    
    def check_degradation(self, current: PerformanceMetrics) -> List[DegradationAlert]:
        """Check for performance degradation.
        
        Returns:
            List of alerts if degradation detected
        """
        alerts = []
        baseline = self.calculate_baseline()
        
        if not baseline:
            return alerts
        
        # Check Sharpe ratio degradation
        if baseline['sharpe_ratio'] > 0:
            sharpe_drop = (baseline['sharpe_ratio'] - current.sharpe_ratio) / baseline['sharpe_ratio']
            if sharpe_drop > self.alert_thresholds['sharpe_ratio_drop']:
                alerts.append(DegradationAlert(
                    timestamp=datetime.now(),
                    severity='critical' if sharpe_drop > 0.5 else 'warning',
                    metric='sharpe_ratio',
                    current_value=current.sharpe_ratio,
                    baseline_value=baseline['sharpe_ratio'],
                    threshold=self.alert_thresholds['sharpe_ratio_drop'],
                    message=f"Sharpe ratio dropped {sharpe_drop:.1%} from baseline"
                ))
        
        # Check win rate degradation
        win_rate_drop = baseline['win_rate'] - current.win_rate
        if win_rate_drop > self.alert_thresholds['win_rate_drop']:
            alerts.append(DegradationAlert(
                timestamp=datetime.now(),
                severity='warning',
                metric='win_rate',
                current_value=current.win_rate,
                baseline_value=baseline['win_rate'],
                threshold=self.alert_thresholds['win_rate_drop'],
                message=f"Win rate dropped {win_rate_drop:.1%} from baseline"
            ))
        
        # Check drawdown increase
        if baseline['max_drawdown'] > 0:
            dd_increase = (current.max_drawdown - baseline['max_drawdown']) / baseline['max_drawdown']
            if dd_increase > self.alert_thresholds['drawdown_increase']:
                alerts.append(DegradationAlert(
                    timestamp=datetime.now(),
                    severity='critical',
                    metric='max_drawdown',
                    current_value=current.max_drawdown,
                    baseline_value=baseline['max_drawdown'],
                    threshold=self.alert_thresholds['drawdown_increase'],
                    message=f"Drawdown increased {dd_increase:.1%} from baseline"
                ))
        
        # Check volatility spike
        if baseline['volatility'] > 0:
            vol_ratio = current.volatility / baseline['volatility']
            if vol_ratio > self.alert_thresholds['volatility_spike']:
                alerts.append(DegradationAlert(
                    timestamp=datetime.now(),
                    severity='warning',
                    metric='volatility',
                    current_value=current.volatility,
                    baseline_value=baseline['volatility'],
                    threshold=self.alert_thresholds['volatility_spike'],
                    message=f"Volatility spiked {vol_ratio:.1f}x from baseline"
                ))
        
        # Statistical anomaly detection (Z-score)
        if baseline.get('sharpe_std', 0) > 0:
            z_score = (current.sharpe_ratio - baseline['sharpe_ratio']) / baseline['sharpe_std']
            if abs(z_score) > self.z_score_threshold:
                alerts.append(DegradationAlert(
                    timestamp=datetime.now(),
                    severity='warning',
                    metric='statistical_anomaly',
                    current_value=z_score,
                    baseline_value=0,
                    threshold=self.z_score_threshold,
                    message=f"Statistical anomaly detected (Z-score: {z_score:.2f})"
                ))
        
        return alerts
    
    def calculate_cusum(self, metric: str = 'sharpe_ratio') -> Tuple[List[float], List[float]]:
        """Calculate CUSUM for change detection.
        
        Returns:
            Tuple of (cusum_positive, cusum_negative) series
        """
        if len(self.performance_history) < 2:
            return [], []
        
        values = [getattr(m, metric) for m in self.performance_history]
        mean = np.mean(values[:self.baseline_window]) if len(values) > self.baseline_window else np.mean(values)
        
        cusum_pos = []
        cusum_neg = []
        s_pos = 0
        s_neg = 0
        
        for value in values:
            s_pos = max(0, s_pos + value - mean - 0.5 * np.std(values))
            s_neg = max(0, s_neg - value + mean - 0.5 * np.std(values))
            cusum_pos.append(s_pos)
            cusum_neg.append(s_neg)
        
        return cusum_pos, cusum_neg
    
    def get_health_score(self) -> float:
        """Calculate overall strategy health score (0-100).
        
        Returns:
            Health score from 0 (dead) to 100 (perfect)
        """
        if len(self.performance_history) < 2:
            return 100.0
        
        baseline = self.calculate_baseline()
        if not baseline:
            return 100.0
        
        current = self.performance_history[-1]
        
        # Calculate component scores
        scores = []
        
        # Sharpe ratio score
        if baseline['sharpe_ratio'] > 0:
            sharpe_score = min(100, max(0, (current.sharpe_ratio / baseline['sharpe_ratio']) * 100))
            scores.append(sharpe_score * 0.3)  # 30% weight
        
        # Win rate score
        win_score = min(100, max(0, (current.win_rate / max(0.01, baseline['win_rate'])) * 100))
        scores.append(win_score * 0.2)  # 20% weight
        
        # Drawdown score (inverse)
        if current.max_drawdown > 0:
            dd_score = min(100, max(0, (baseline['max_drawdown'] / current.max_drawdown) * 100))
            scores.append(dd_score * 0.2)  # 20% weight
        
        # PnL score
        if baseline['total_pnl'] > 0:
            pnl_score = min(100, max(0, (current.total_pnl / baseline['total_pnl']) * 100))
            scores.append(pnl_score * 0.2)  # 20% weight
        
        # Trade activity score
        if baseline['n_trades'] > 0:
            trade_score = min(100, max(0, (current.n_trades / baseline['n_trades']) * 100))
            scores.append(trade_score * 0.1)  # 10% weight
        
        return sum(scores) if scores else 100.0


class RollingPerformanceAnalyzer:
    """Analyzes rolling performance windows."""
    
    def __init__(self, window_sizes: List[int] = [7, 30, 90]):
        self.window_sizes = window_sizes
        
    def calculate_rolling_metrics(
        self,
        performance_history: List[PerformanceMetrics]
    ) -> Dict[int, pd.DataFrame]:
        """Calculate rolling metrics for different windows.
        
        Returns:
            Dictionary of window_size -> DataFrame with rolling metrics
        """
        if not performance_history:
            return {}
        
        # Convert to DataFrame
        df = pd.DataFrame([m.to_dict() for m in performance_history])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        rolling_data = {}
        
        for window in self.window_sizes:
            rolling = pd.DataFrame(index=df.index)
            
            # Calculate rolling statistics
            rolling['sharpe_mean'] = df['sharpe_ratio'].rolling(window).mean()
            rolling['sharpe_std'] = df['sharpe_ratio'].rolling(window).std()
            rolling['pnl_sum'] = df['total_pnl'].rolling(window).sum()
            rolling['win_rate_mean'] = df['win_rate'].rolling(window).mean()
            rolling['drawdown_max'] = df['max_drawdown'].rolling(window).max()
            rolling['trades_sum'] = df['n_trades'].rolling(window).sum()
            
            # Calculate rolling Sharpe
            returns = df['total_pnl'].pct_change()
            rolling['rolling_sharpe'] = (
                returns.rolling(window).mean() / returns.rolling(window).std()
            ) * np.sqrt(252)  # Annualized
            
            rolling_data[window] = rolling
        
        return rolling_data


def render_monitoring_dashboard():
    """Render strategy monitoring dashboard in Streamlit."""
    
    st.title("📊 Strategy Performance Monitor")
    st.caption("Real-time monitoring and degradation detection")
    
    # Initialize monitor
    if 'strategy_monitor' not in st.session_state:
        st.session_state.strategy_monitor = StrategyMonitor()
    
    monitor = st.session_state.strategy_monitor
    
    # Simulate or load performance data
    if st.button("🔄 Load Latest Performance Data"):
        # In production, this would load actual performance data
        # For demo, simulate data
        for i in range(100):
            timestamp = datetime.now() - timedelta(days=100-i)
            
            # Simulate degradation after day 70
            degradation_factor = 1.0 if i < 70 else 0.7
            
            metrics = PerformanceMetrics(
                timestamp=timestamp,
                sharpe_ratio=np.random.normal(1.2, 0.3) * degradation_factor,
                total_pnl=np.random.normal(1000, 200) * degradation_factor,
                win_rate=np.random.uniform(0.48, 0.55) * degradation_factor,
                max_drawdown=np.random.uniform(0.05, 0.15) / degradation_factor,
                n_trades=np.random.randint(10, 30),
                avg_trade_pnl=np.random.normal(50, 20) * degradation_factor,
                volatility=np.random.uniform(0.15, 0.25) / degradation_factor,
                calmar_ratio=np.random.normal(2.0, 0.5) * degradation_factor
            )
            
            monitor.add_performance_data(metrics)
        
        st.success("✅ Performance data loaded")
    
    if monitor.performance_history:
        # Health score
        health_score = monitor.get_health_score()
        
        # Header metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            health_color = "🟢" if health_score > 80 else "🟡" if health_score > 60 else "🔴"
            st.metric(
                f"{health_color} Health Score",
                f"{health_score:.0f}%",
                delta=f"{health_score - 100:.0f}%" if health_score < 100 else None
            )
        
        with col2:
            recent_alerts = [a for a in monitor.alerts if 
                           (datetime.now() - a.timestamp).days < 7]
            st.metric("🚨 Recent Alerts", len(recent_alerts))
        
        with col3:
            latest = monitor.performance_history[-1]
            st.metric("📈 Current Sharpe", f"{latest.sharpe_ratio:.2f}")
        
        with col4:
            st.metric("💰 Current PnL", f"${latest.total_pnl:.0f}")
        
        # Tabs for different views
        tabs = st.tabs([
            "📈 Performance Trends",
            "🚨 Alerts & Warnings",
            "📊 Rolling Analysis",
            "🔬 Statistical Tests",
            "⚙️ Settings"
        ])
        
        with tabs[0]:  # Performance Trends
            st.subheader("Performance Metrics Over Time")
            
            # Convert to DataFrame for plotting
            df = pd.DataFrame([m.to_dict() for m in monitor.performance_history])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Metric selector
            metrics_to_plot = st.multiselect(
                "Select metrics to display",
                ['sharpe_ratio', 'total_pnl', 'win_rate', 'max_drawdown', 'volatility'],
                default=['sharpe_ratio', 'total_pnl']
            )
            
            # Create subplots
            for metric in metrics_to_plot:
                fig = go.Figure()
                
                # Add main line
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df[metric],
                    mode='lines',
                    name=metric.replace('_', ' ').title(),
                    line=dict(width=2)
                ))
                
                # Add baseline if available
                baseline = monitor.calculate_baseline()
                if baseline and metric in baseline:
                    fig.add_hline(
                        y=baseline[metric],
                        line_dash="dash",
                        line_color="gray",
                        annotation_text="Baseline"
                    )
                
                # Mark alerts
                metric_alerts = [a for a in monitor.alerts if a.metric == metric]
                if metric_alerts:
                    alert_times = [a.timestamp for a in metric_alerts]
                    alert_values = [a.current_value for a in metric_alerts]
                    
                    fig.add_trace(go.Scatter(
                        x=alert_times,
                        y=alert_values,
                        mode='markers',
                        name='Alerts',
                        marker=dict(size=10, color='red', symbol='x')
                    ))
                
                fig.update_layout(
                    title=metric.replace('_', ' ').title(),
                    xaxis_title="Date",
                    yaxis_title="Value",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:  # Alerts
            st.subheader("🚨 Performance Alerts")
            
            if monitor.alerts:
                # Alert summary
                alert_counts = {
                    'critical': len([a for a in monitor.alerts if a.severity == 'critical']),
                    'warning': len([a for a in monitor.alerts if a.severity == 'warning']),
                    'emergency': len([a for a in monitor.alerts if a.severity == 'emergency'])
                }
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🔴 Critical", alert_counts['critical'])
                with col2:
                    st.metric("🟡 Warnings", alert_counts['warning'])
                with col3:
                    st.metric("🚨 Emergency", alert_counts['emergency'])
                
                # Alert timeline
                st.markdown("---")
                st.markdown("**Alert Timeline**")
                
                # Convert alerts to DataFrame
                alerts_df = pd.DataFrame([a.to_dict() for a in monitor.alerts[-50:]])
                
                if not alerts_df.empty:
                    alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
                    
                    # Color code by severity
                    severity_colors = {
                        'warning': '🟡',
                        'critical': '🔴',
                        'emergency': '🚨'
                    }
                    
                    alerts_df['icon'] = alerts_df['severity'].map(severity_colors)
                    alerts_df['display'] = alerts_df['icon'] + ' ' + alerts_df['message']
                    
                    # Display recent alerts
                    for _, alert in alerts_df.tail(10).iterrows():
                        st.write(f"{alert['timestamp'].strftime('%Y-%m-%d %H:%M')} - {alert['display']}")
                
                # Alert distribution chart
                st.markdown("---")
                st.markdown("**Alert Distribution by Metric**")
                
                if not alerts_df.empty:
                    alert_counts_by_metric = alerts_df['metric'].value_counts()
                    
                    fig = px.bar(
                        x=alert_counts_by_metric.index,
                        y=alert_counts_by_metric.values,
                        labels={'x': 'Metric', 'y': 'Count'},
                        title="Alerts by Metric"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No alerts generated yet")
        
        with tabs[2]:  # Rolling Analysis
            st.subheader("📊 Rolling Window Analysis")
            
            analyzer = RollingPerformanceAnalyzer()
            rolling_data = analyzer.calculate_rolling_metrics(monitor.performance_history)
            
            # Window selector
            window_size = st.selectbox(
                "Select rolling window",
                list(rolling_data.keys()),
                format_func=lambda x: f"{x} days"
            )
            
            if window_size in rolling_data:
                rolling_df = rolling_data[window_size]
                
                # Plot rolling metrics
                metrics_to_show = ['sharpe_mean', 'pnl_sum', 'win_rate_mean', 'rolling_sharpe']
                
                for metric in metrics_to_show:
                    if metric in rolling_df.columns:
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=rolling_df.index,
                            y=rolling_df[metric],
                            mode='lines',
                            name=metric.replace('_', ' ').title()
                        ))
                        
                        # Add confidence bands for mean metrics
                        if 'mean' in metric and metric.replace('mean', 'std') in rolling_df.columns:
                            std_col = metric.replace('mean', 'std')
                            upper = rolling_df[metric] + 2 * rolling_df[std_col]
                            lower = rolling_df[metric] - 2 * rolling_df[std_col]
                            
                            fig.add_trace(go.Scatter(
                                x=rolling_df.index,
                                y=upper,
                                fill=None,
                                mode='lines',
                                line=dict(width=0.5, color='rgba(0,100,255,0.2)'),
                                showlegend=False
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=rolling_df.index,
                                y=lower,
                                fill='tonexty',
                                mode='lines',
                                line=dict(width=0.5, color='rgba(0,100,255,0.2)'),
                                name='95% CI'
                            ))
                        
                        fig.update_layout(
                            title=f"Rolling {window_size}-day {metric.replace('_', ' ').title()}",
                            xaxis_title="Date",
                            yaxis_title="Value",
                            height=300
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        
        with tabs[3]:  # Statistical Tests
            st.subheader("🔬 Statistical Change Detection")
            
            # CUSUM analysis
            st.markdown("**CUSUM Change Detection**")
            
            cusum_metric = st.selectbox(
                "Select metric for CUSUM",
                ['sharpe_ratio', 'total_pnl', 'win_rate']
            )
            
            cusum_pos, cusum_neg = monitor.calculate_cusum(cusum_metric)
            
            if cusum_pos and cusum_neg:
                fig = go.Figure()
                
                timestamps = [m.timestamp for m in monitor.performance_history]
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=cusum_pos,
                    mode='lines',
                    name='CUSUM+',
                    line=dict(color='green')
                ))
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=cusum_neg,
                    mode='lines',
                    name='CUSUM-',
                    line=dict(color='red')
                ))
                
                # Add threshold line
                fig.add_hline(
                    y=monitor.cusum_threshold,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text="Threshold"
                )
                
                fig.update_layout(
                    title=f"CUSUM Chart for {cusum_metric.replace('_', ' ').title()}",
                    xaxis_title="Date",
                    yaxis_title="CUSUM Value",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Check for change points
                change_detected = any(c > monitor.cusum_threshold for c in cusum_pos + cusum_neg)
                if change_detected:
                    st.warning("⚠️ Significant change detected in strategy behavior")
                else:
                    st.success("✅ No significant changes detected")
            
            # Stability analysis
            st.markdown("---")
            st.markdown("**Stability Analysis**")
            
            if len(monitor.performance_history) > 30:
                # Calculate stability metrics
                recent_30 = monitor.performance_history[-30:]
                recent_7 = monitor.performance_history[-7:]
                
                stability_metrics = {
                    'Sharpe Stability': np.std([m.sharpe_ratio for m in recent_30]),
                    'PnL Stability': np.std([m.total_pnl for m in recent_30]),
                    'Recent Volatility': np.mean([m.volatility for m in recent_7]),
                    'Trade Consistency': np.std([m.n_trades for m in recent_30])
                }
                
                col1, col2 = st.columns(2)
                
                for i, (metric, value) in enumerate(stability_metrics.items()):
                    col = col1 if i % 2 == 0 else col2
                    with col:
                        st.metric(metric, f"{value:.3f}")
        
        with tabs[4]:  # Settings
            st.subheader("⚙️ Monitoring Settings")
            
            # Alert thresholds
            st.markdown("**Alert Thresholds**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                sharpe_threshold = st.slider(
                    "Sharpe Ratio Drop (%)",
                    min_value=10, max_value=50,
                    value=int(monitor.alert_thresholds['sharpe_ratio_drop'] * 100),
                    step=5
                )
                monitor.alert_thresholds['sharpe_ratio_drop'] = sharpe_threshold / 100
                
                win_rate_threshold = st.slider(
                    "Win Rate Drop (absolute %)",
                    min_value=5, max_value=20,
                    value=int(monitor.alert_thresholds['win_rate_drop'] * 100),
                    step=1
                )
                monitor.alert_thresholds['win_rate_drop'] = win_rate_threshold / 100
                
                drawdown_threshold = st.slider(
                    "Drawdown Increase (%)",
                    min_value=20, max_value=100,
                    value=int(monitor.alert_thresholds['drawdown_increase'] * 100),
                    step=10
                )
                monitor.alert_thresholds['drawdown_increase'] = drawdown_threshold / 100
            
            with col2:
                volatility_threshold = st.slider(
                    "Volatility Spike (x)",
                    min_value=1.5, max_value=3.0,
                    value=monitor.alert_thresholds['volatility_spike'],
                    step=0.1
                )
                monitor.alert_thresholds['volatility_spike'] = volatility_threshold
                
                z_score_threshold = st.slider(
                    "Z-Score Threshold",
                    min_value=1.5, max_value=3.0,
                    value=monitor.z_score_threshold,
                    step=0.1
                )
                monitor.z_score_threshold = z_score_threshold
                
                cusum_threshold = st.slider(
                    "CUSUM Threshold",
                    min_value=3.0, max_value=10.0,
                    value=monitor.cusum_threshold,
                    step=0.5
                )
                monitor.cusum_threshold = cusum_threshold
            
            # Baseline window
            st.markdown("---")
            baseline_window = st.number_input(
                "Baseline Window (days)",
                min_value=7, max_value=90,
                value=monitor.baseline_window,
                step=1
            )
            monitor.baseline_window = baseline_window
            
            # Export settings
            st.markdown("---")
            if st.button("💾 Export Monitoring Data"):
                export_data = {
                    'performance_history': [m.to_dict() for m in monitor.performance_history],
                    'alerts': [a.to_dict() for a in monitor.alerts],
                    'settings': {
                        'baseline_window': monitor.baseline_window,
                        'alert_thresholds': monitor.alert_thresholds,
                        'z_score_threshold': monitor.z_score_threshold,
                        'cusum_threshold': monitor.cusum_threshold
                    },
                    'health_score': health_score,
                    'timestamp': datetime.now().isoformat()
                }
                
                json_str = json.dumps(export_data, indent=2)
                st.download_button(
                    "📥 Download Monitoring Data",
                    json_str,
                    file_name=f"monitoring_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    else:
        st.info("No performance data available. Click 'Load Latest Performance Data' to begin.")


if __name__ == "__main__":
    render_monitoring_dashboard()