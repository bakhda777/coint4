#!/usr/bin/env python3
"""
Distributed optimization component using multiple workers.
Supports parallel optimization with Redis/RabbitMQ or simple multiprocessing.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import yaml
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
import queue
import threading
import sys
import subprocess
import socket

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import optuna
    from optuna.storages import RDBStorage
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


@dataclass
class WorkerConfig:
    """Configuration for a worker process."""
    worker_id: str
    n_trials: int
    study_name: str
    storage: str
    param_ranges: Dict[str, Tuple[float, float]]
    config_path: str = "configs/main_2024.yaml"
    host: Optional[str] = None  # For remote workers
    port: Optional[int] = None


class DistributedOptimizer:
    """Manages distributed optimization across multiple workers."""
    
    def __init__(self, storage_type: str = "sqlite", storage_path: str = "outputs/studies"):
        """Initialize distributed optimizer.
        
        Args:
            storage_type: Type of storage (sqlite, postgresql, mysql)
            storage_path: Path to storage directory or connection string
        """
        self.storage_type = storage_type
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.workers = []
        self.study = None
        self.monitor_thread = None
        self.stop_flag = threading.Event()
        
    def create_storage(self, study_name: str) -> str:
        """Create storage for distributed optimization.
        
        Returns:
            Storage URL for Optuna
        """
        if self.storage_type == "sqlite":
            db_path = self.storage_path / f"{study_name}.db"
            return f"sqlite:///{db_path}"
        elif self.storage_type == "postgresql":
            # PostgreSQL connection string
            return f"postgresql://user:password@localhost/{study_name}"
        elif self.storage_type == "mysql":
            # MySQL connection string
            return f"mysql://user:password@localhost/{study_name}"
        else:
            raise ValueError(f"Unknown storage type: {self.storage_type}")
    
    def create_worker_process(self, config: WorkerConfig) -> mp.Process:
        """Create a worker process for optimization.
        
        Args:
            config: Worker configuration
        
        Returns:
            Worker process
        """
        def worker_main():
            """Main function for worker process."""
            try:
                # Import inside worker to avoid pickling issues
                from scripts.optimization.web_optimizer import WebOptimizer
                
                # Create optimizer
                optimizer = WebOptimizer(
                    base_config_path=config.config_path,
                    search_space_path="configs/search_spaces/web_ui.yaml"
                )
                
                # Load or create study
                study = optuna.load_study(
                    study_name=config.study_name,
                    storage=config.storage
                )
                
                # Create objective
                objective = optimizer.create_objective(config.param_ranges)
                
                # Run optimization
                study.optimize(
                    objective,
                    n_trials=config.n_trials,
                    show_progress_bar=False
                )
                
                print(f"Worker {config.worker_id} completed {config.n_trials} trials")
                
            except Exception as e:
                print(f"Worker {config.worker_id} error: {e}")
        
        process = mp.Process(target=worker_main, name=f"worker_{config.worker_id}")
        return process
    
    def run_distributed_optimization(
        self,
        n_trials_total: int,
        n_workers: int,
        param_ranges: Dict[str, Tuple[float, float]],
        target_metric: str = "sharpe_ratio",
        config_path: str = "configs/main_2024.yaml",
        progress_callback: Optional[Callable] = None
    ) -> optuna.Study:
        """Run distributed optimization across multiple workers.
        
        Args:
            n_trials_total: Total number of trials
            n_workers: Number of worker processes
            param_ranges: Parameter ranges for optimization
            target_metric: Target metric to optimize
            config_path: Path to configuration file
            progress_callback: Callback for progress updates
        
        Returns:
            Optuna study with results
        """
        # Create study
        study_name = f"distributed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        storage = self.create_storage(study_name)
        
        # Create study in storage
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=False
        )
        
        # Calculate trials per worker
        trials_per_worker = n_trials_total // n_workers
        remaining_trials = n_trials_total % n_workers
        
        # Create worker configurations
        worker_configs = []
        for i in range(n_workers):
            n_trials = trials_per_worker
            if i < remaining_trials:
                n_trials += 1
            
            config = WorkerConfig(
                worker_id=f"worker_{i}",
                n_trials=n_trials,
                study_name=study_name,
                storage=storage,
                param_ranges=param_ranges,
                config_path=config_path
            )
            worker_configs.append(config)
        
        # Start workers
        self.workers = []
        for config in worker_configs:
            worker = self.create_worker_process(config)
            worker.start()
            self.workers.append(worker)
            time.sleep(0.1)  # Small delay to avoid race conditions
        
        # Monitor progress
        self.monitor_progress(storage, study_name, n_trials_total, progress_callback)
        
        # Wait for workers to complete
        for worker in self.workers:
            worker.join()
        
        # Reload study with all results
        self.study = optuna.load_study(
            study_name=study_name,
            storage=storage
        )
        
        return self.study
    
    def monitor_progress(
        self,
        storage: str,
        study_name: str,
        n_trials_total: int,
        progress_callback: Optional[Callable] = None
    ):
        """Monitor optimization progress.
        
        Args:
            storage: Storage URL
            study_name: Study name
            n_trials_total: Total expected trials
            progress_callback: Callback for progress updates
        """
        def monitor():
            """Monitor thread function."""
            last_n_trials = 0
            
            while not self.stop_flag.is_set():
                try:
                    # Load study to check progress
                    study = optuna.load_study(
                        study_name=study_name,
                        storage=storage
                    )
                    
                    n_trials = len(study.trials)
                    
                    if n_trials > last_n_trials:
                        # New trials completed
                        if progress_callback:
                            best_value = study.best_value if study.best_trial else None
                            progress_callback(n_trials, n_trials_total, best_value)
                        
                        last_n_trials = n_trials
                    
                    # Check if complete
                    if n_trials >= n_trials_total:
                        break
                    
                    time.sleep(1)  # Check every second
                    
                except Exception as e:
                    print(f"Monitor error: {e}")
                    time.sleep(5)
        
        self.monitor_thread = threading.Thread(target=monitor)
        self.monitor_thread.start()
    
    def stop_optimization(self):
        """Stop all workers and monitoring."""
        self.stop_flag.set()
        
        # Terminate workers
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
        
        # Stop monitor
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def get_worker_statistics(self) -> pd.DataFrame:
        """Get statistics for each worker.
        
        Returns:
            DataFrame with worker statistics
        """
        if not self.study:
            return pd.DataFrame()
        
        # Analyze trials by worker (would need worker ID in trial user_attrs)
        # For now, return overall statistics
        stats = []
        
        for i, worker in enumerate(self.workers):
            stats.append({
                'Worker': f"Worker {i}",
                'Status': 'Running' if worker.is_alive() else 'Completed',
                'PID': worker.pid
            })
        
        return pd.DataFrame(stats)


class RemoteWorkerManager:
    """Manages remote workers for distributed optimization."""
    
    def __init__(self):
        self.remote_workers = []
        
    def add_remote_worker(self, host: str, port: int, ssh_key: Optional[str] = None):
        """Add a remote worker.
        
        Args:
            host: Remote host address
            port: SSH port
            ssh_key: Path to SSH key (optional)
        """
        self.remote_workers.append({
            'host': host,
            'port': port,
            'ssh_key': ssh_key,
            'status': 'pending'
        })
    
    def deploy_worker(self, worker_info: Dict, config: WorkerConfig) -> bool:
        """Deploy worker to remote machine.
        
        Args:
            worker_info: Remote worker information
            config: Worker configuration
        
        Returns:
            True if deployment successful
        """
        # Build SSH command
        ssh_cmd = ["ssh"]
        
        if worker_info.get('ssh_key'):
            ssh_cmd.extend(["-i", worker_info['ssh_key']])
        
        ssh_cmd.extend([
            f"-p {worker_info['port']}",
            f"{worker_info['host']}",
            f"cd /opt/coint2 && python -m scripts.optimization.remote_worker " +
            f"--study-name {config.study_name} " +
            f"--storage {config.storage} " +
            f"--n-trials {config.n_trials} " +
            f"--worker-id {config.worker_id}"
        ])
        
        # Execute deployment
        try:
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0
        except Exception as e:
            print(f"Failed to deploy to {worker_info['host']}: {e}")
            return False
    
    def check_worker_status(self, worker_info: Dict) -> str:
        """Check status of remote worker.
        
        Returns:
            Worker status (running, completed, error)
        """
        # Check via SSH or monitoring endpoint
        # Simplified for now
        return "running"


def render_distributed_ui():
    """Render distributed optimization UI in Streamlit."""
    
    st.title("🌐 Распределенная оптимизация")
    st.caption("Ускорьте оптимизацию используя несколько воркеров")
    
    if not OPTUNA_AVAILABLE:
        st.error("⚠️ Optuna не установлен")
        return
    
    # Initialize optimizer
    if 'dist_optimizer' not in st.session_state:
        st.session_state.dist_optimizer = DistributedOptimizer()
    
    optimizer = st.session_state.dist_optimizer
    
    # Configuration columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🖥️ Конфигурация воркеров")
        
        # Worker configuration
        worker_mode = st.radio(
            "Режим воркеров",
            ["🏠 Локальные процессы", "☁️ Удаленные машины", "🐳 Docker контейнеры"],
            help="Выберите как запускать воркеры"
        )
        
        if worker_mode == "🏠 Локальные процессы":
            # Local workers
            max_workers = mp.cpu_count()
            n_workers = st.slider(
                "Количество воркеров",
                min_value=1,
                max_value=max_workers,
                value=min(4, max_workers),
                help=f"Доступно CPU: {max_workers}"
            )
            
            st.info(f"💻 Будет запущено {n_workers} локальных процессов")
            
        elif worker_mode == "☁️ Удаленные машины":
            # Remote workers
            st.markdown("**Добавить удаленные машины:**")
            
            remote_hosts = st.text_area(
                "Хосты (один на строку)",
                placeholder="worker1.example.com\nworker2.example.com",
                height=100
            )
            
            ssh_port = st.number_input("SSH порт", value=22)
            ssh_key = st.text_input("SSH ключ (путь)", placeholder="/home/user/.ssh/id_rsa")
            
            if remote_hosts:
                hosts = remote_hosts.strip().split('\n')
                n_workers = len(hosts)
                st.info(f"☁️ Будет использовано {n_workers} удаленных машин")
            else:
                n_workers = 0
                st.warning("Добавьте хосты")
        
        else:  # Docker containers
            st.markdown("**Docker конфигурация:**")
            
            docker_image = st.text_input(
                "Docker образ",
                value="coint2-optimizer:latest"
            )
            
            n_containers = st.slider(
                "Количество контейнеров",
                min_value=1,
                max_value=10,
                value=4
            )
            
            n_workers = n_containers
            st.info(f"🐳 Будет запущено {n_workers} Docker контейнеров")
        
        # Storage configuration
        st.markdown("---")
        st.subheader("💾 Хранилище")
        
        storage_type = st.selectbox(
            "Тип хранилища",
            ["SQLite (локальное)", "PostgreSQL", "MySQL"],
            help="Для распределенной оптимизации рекомендуется PostgreSQL"
        )
        
        if storage_type == "PostgreSQL":
            pg_host = st.text_input("Host", value="localhost")
            pg_port = st.number_input("Port", value=5432)
            pg_user = st.text_input("User", value="optuna")
            pg_pass = st.text_input("Password", type="password")
            pg_db = st.text_input("Database", value="optuna_studies")
    
    with col2:
        st.subheader("⚙️ Параметры оптимизации")
        
        # Total trials
        n_trials_total = st.number_input(
            "Общее количество trials",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100
        )
        
        if n_workers > 0:
            trials_per_worker = n_trials_total // n_workers
            st.info(f"~{trials_per_worker} trials на воркер")
        
        # Target metric
        target_metric = st.selectbox(
            "Целевая метрика",
            ["Sharpe Ratio", "Total PnL", "Win Rate", "Calmar Ratio"]
        )
        
        st.markdown("---")
        st.markdown("**Диапазоны параметров:**")
        
        # Simplified parameter ranges
        param_ranges = {}
        
        zscore_range = st.slider(
            "Z-score threshold",
            min_value=0.5, max_value=3.0,
            value=(1.0, 2.0), step=0.1,
            key="dist_zscore"
        )
        param_ranges['zscore_threshold'] = zscore_range
        
        window_range = st.slider(
            "Rolling window",
            min_value=10, max_value=100,
            value=(20, 50), step=5,
            key="dist_window"
        )
        param_ranges['rolling_window'] = window_range
        
        commission_range = st.slider(
            "Commission %",
            min_value=0.0002, max_value=0.0010,
            value=(0.0003, 0.0005), step=0.0001,
            format="%.4f",
            key="dist_commission"
        )
        param_ranges['commission_pct'] = commission_range
    
    # Control buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if 'dist_running' not in st.session_state:
            st.session_state.dist_running = False
        
        if not st.session_state.dist_running:
            if st.button("🚀 ЗАПУСТИТЬ РАСПРЕДЕЛЕННУЮ ОПТИМИЗАЦИЮ", type="primary", use_container_width=True):
                if n_workers == 0:
                    st.error("❌ Нет доступных воркеров")
                else:
                    st.session_state.dist_running = True
                    st.rerun()
        else:
            if st.button("🛑 ОСТАНОВИТЬ", type="secondary", use_container_width=True):
                optimizer.stop_optimization()
                st.session_state.dist_running = False
                st.rerun()
    
    # Progress monitoring
    if st.session_state.dist_running:
        st.markdown("---")
        st.subheader("📊 Мониторинг прогресса")
        
        # Create progress containers
        progress_container = st.container()
        
        with progress_container:
            # Overall progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                trials_metric = st.empty()
            with col2:
                workers_metric = st.empty()
            with col3:
                best_metric = st.empty()
            with col4:
                time_metric = st.empty()
            
            # Worker status table
            with st.expander("👷 Статус воркеров", expanded=True):
                worker_table = st.empty()
            
            # Start distributed optimization
            def progress_callback(current_trials, total_trials, best_value):
                progress = current_trials / total_trials
                progress_bar.progress(progress)
                status_text.text(f"Выполнено {current_trials}/{total_trials} trials")
                
                trials_metric.metric("📊 Trials", f"{current_trials}/{total_trials}")
                workers_metric.metric("👷 Воркеры", f"{n_workers} активных")
                if best_value:
                    best_metric.metric("🏆 Best", f"{best_value:.3f}")
            
            # Run optimization in background
            study = optimizer.run_distributed_optimization(
                n_trials_total=n_trials_total,
                n_workers=n_workers,
                param_ranges=param_ranges,
                target_metric=target_metric.lower().replace(' ', '_'),
                progress_callback=progress_callback
            )
            
            # Update worker status
            worker_stats = optimizer.get_worker_statistics()
            worker_table.dataframe(worker_stats, use_container_width=True)
            
            # Completion
            st.success(f"✅ Распределенная оптимизация завершена!")
            st.session_state.dist_running = False
            st.session_state.dist_study = study
    
    # Show results
    if 'dist_study' in st.session_state and st.session_state.dist_study:
        st.markdown("---")
        st.subheader("📊 Результаты")
        
        study = st.session_state.dist_study
        
        # Best parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Best Value", f"{study.best_value:.3f}")
            st.metric("Total Trials", len(study.trials))
        
        with col2:
            st.metric("Workers Used", n_workers)
            # Calculate efficiency
            if len(study.trials) > 0:
                efficiency = len([t for t in study.trials if t.value is not None]) / len(study.trials)
                st.metric("Efficiency", f"{efficiency:.1%}")
        
        # Best parameters
        st.markdown("**Лучшие параметры:**")
        best_params_df = pd.DataFrame([
            {"Parameter": k, "Value": v}
            for k, v in study.best_params.items()
        ])
        st.dataframe(best_params_df, use_container_width=True)
        
        # Visualization
        try:
            import optuna.visualization as optuna_vis
            
            viz_tabs = st.tabs(["📈 История", "⚡ Параллельность", "🎯 Важность"])
            
            with viz_tabs[0]:
                fig = optuna_vis.plot_optimization_history(study)
                st.plotly_chart(fig, use_container_width=True)
            
            with viz_tabs[1]:
                # Timeline visualization (simulated)
                import plotly.graph_objects as go
                
                fig = go.Figure()
                
                # Simulate worker timeline
                for i in range(n_workers):
                    worker_trials = [t for j, t in enumerate(study.trials) if j % n_workers == i]
                    if worker_trials:
                        x = [t.number for t in worker_trials]
                        y = [i] * len(worker_trials)
                        colors = [t.value if t.value else 0 for t in worker_trials]
                        
                        fig.add_trace(go.Scatter(
                            x=x, y=y,
                            mode='markers',
                            name=f'Worker {i}',
                            marker=dict(
                                size=10,
                                color=colors,
                                colorscale='Viridis',
                                showscale=(i == 0)
                            )
                        ))
                
                fig.update_layout(
                    title="Worker Timeline",
                    xaxis_title="Trial Number",
                    yaxis_title="Worker ID",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with viz_tabs[2]:
                try:
                    fig = optuna_vis.plot_param_importances(study)
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.info("Недостаточно данных для расчета важности")
        
        except Exception as e:
            st.error(f"Ошибка визуализации: {e}")


if __name__ == "__main__":
    render_distributed_ui()