#!/usr/bin/env python3
"""
Live Snapshot Extractor - Утилита для сбора моментального снимка системы.

Собирает:
- Последние 100 строк логов
- Последние N сделок  
- Сводку метрик
- Статус системы

Сохраняет в artifacts/live/SNAPSHOT_{timestamp}.md
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coint2.utils.config import load_config


class LiveSnapshotExtractor:
    """Извлекает снапшот системы в реальном времени."""
    
    def __init__(self, logs_dir: str = "artifacts/live/logs", 
                 metrics_dir: str = "artifacts/live/metrics"):
        """Инициализация экстрактора.
        
        Args:
            logs_dir: Директория с логами
            metrics_dir: Директория с метриками
        """
        self.logs_dir = Path(logs_dir)
        self.metrics_dir = Path(metrics_dir)
        self.artifacts_dir = Path("artifacts/live")
        self.timestamp = datetime.now()
        
        # Ensure directories exist
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_latest_log_lines(self, n_lines: int = 100) -> List[str]:
        """Получить последние N строк из логов."""
        log_lines = []
        
        # Поиск файлов логов (по приоритету: main, trades, alerts, metrics)
        log_files = []
        for pattern in ["main.jsonl", "trades.jsonl", "alerts.jsonl", "metrics.jsonl"]:
            files = list(self.logs_dir.glob(pattern)) + list(self.logs_dir.glob(f"*{pattern}*"))
            if files:
                # Берем самый новый файл
                latest_file = max(files, key=lambda x: x.stat().st_mtime)
                log_files.append(latest_file)
        
        # Если нет структурированных логов, ищем любые .log/.jsonl файлы
        if not log_files:
            for pattern in ["*.log", "*.jsonl"]:
                files = list(self.logs_dir.glob(pattern))
                log_files.extend(files)
        
        # Читаем последние строки из каждого файла
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # Берем последние строки из файла
                    recent_lines = lines[-min(n_lines//len(log_files) if log_files else n_lines, len(lines)):]
                    
                    for line in recent_lines:
                        line = line.strip()
                        if line:
                            # Добавляем метку файла для JSON логов
                            if log_file.suffix == '.jsonl':
                                try:
                                    parsed = json.loads(line)
                                    source = log_file.stem
                                    log_lines.append(f"[{source}] {line}")
                                except:
                                    log_lines.append(f"[{log_file.stem}] {line}")
                            else:
                                log_lines.append(line)
                                
            except Exception as e:
                log_lines.append(f"ERROR reading {log_file}: {str(e)}")
        
        # Сортируем по времени если возможно
        return log_lines[-n_lines:] if log_lines else ["No recent log entries found"]
    
    def _get_recent_trades(self, n_trades: int = 10) -> List[Dict[str, Any]]:
        """Получить последние N сделок."""
        trades = []
        
        # Ищем файлы сделок
        trades_files = list(self.artifacts_dir.glob("trades_*.json"))
        
        # Читаем из индекса если есть
        trades_index = self.artifacts_dir / "TRADES_INDEX.csv"
        if trades_index.exists():
            try:
                df = pd.read_csv(trades_index)
                if not df.empty:
                    # Берем последние N записей
                    recent_trades = df.tail(n_trades)
                    return recent_trades.to_dict('records')
            except Exception as e:
                trades.append({"error": f"Failed to read trades index: {str(e)}"})
        
        # Если индекса нет, читаем из JSON файлов
        for trades_file in sorted(trades_files, key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                with open(trades_file, 'r') as f:
                    file_trades = json.load(f)
                    trades.extend(file_trades)
                    if len(trades) >= n_trades:
                        break
            except Exception as e:
                trades.append({"error": f"Failed to read {trades_file}: {str(e)}"})
        
        return trades[-n_trades:] if trades else [{"message": "No recent trades found"}]
    
    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Получить сводку метрик."""
        metrics = {}
        
        # Ищем файлы метрик
        metrics_files = list(self.metrics_dir.glob("*.jsonl"))
        
        if not metrics_files:
            return {"error": "No metrics files found"}
        
        # Читаем последние метрики
        latest_metrics = {}
        for metrics_file in metrics_files:
            try:
                with open(metrics_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        # Берем последнюю строку (самые свежие метрики)
                        last_line = lines[-1].strip()
                        if last_line:
                            parsed = json.loads(last_line)
                            latest_metrics.update(parsed)
            except Exception as e:
                metrics[f"error_{metrics_file.stem}"] = str(e)
        
        # Формируем сводку
        if latest_metrics:
            metrics = {
                "timestamp": latest_metrics.get("timestamp", "Unknown"),
                "total_pnl": latest_metrics.get("total_pnl", 0),
                "trade_count": latest_metrics.get("trade_count", 0),
                "win_rate": latest_metrics.get("win_rate", 0),
                "max_drawdown": latest_metrics.get("max_drawdown", 0),
                "current_drawdown": latest_metrics.get("current_drawdown", 0),
                "sharpe_ratio": latest_metrics.get("sharpe_ratio", 0),
                "exposure": latest_metrics.get("exposure", 0),
                "active_positions": latest_metrics.get("active_positions", 0)
            }
        else:
            metrics = {"message": "No recent metrics available"}
        
        return metrics
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Получить статус системы."""
        status = {}
        
        # Проверка файлов конфигурации
        config_files = {
            "prod_config": "configs/prod.yaml",
            "risk_config": "configs/risk.yaml",
            "main_config": "configs/main_2024.yaml"
        }
        
        configs_status = {}
        for name, path in config_files.items():
            file_path = Path(path)
            configs_status[name] = {
                "exists": file_path.exists(),
                "size": file_path.stat().st_size if file_path.exists() else 0,
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat() if file_path.exists() else None
            }
        
        # Проверка процессов (простая проверка файлов)
        process_status = {}
        
        # Проверяем наличие PID файла или lock файла
        for pid_pattern in ["*.pid", "*.lock"]:
            pid_files = list(self.artifacts_dir.glob(pid_pattern))
            if pid_files:
                process_status["running"] = True
                process_status["pid_files"] = [str(f) for f in pid_files]
                break
        else:
            process_status["running"] = False
        
        # Проверка свежести данных
        data_status = {}
        log_files = list(self.logs_dir.glob("*.jsonl")) + list(self.logs_dir.glob("*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            last_modified = datetime.fromtimestamp(latest_log.stat().st_mtime)
            minutes_ago = (self.timestamp - last_modified).total_seconds() / 60
            
            data_status = {
                "latest_log_file": str(latest_log),
                "last_modified": last_modified.isoformat(),
                "minutes_ago": round(minutes_ago, 1),
                "is_fresh": minutes_ago < 30  # Fresh if updated within 30 minutes
            }
        else:
            data_status = {"message": "No log files found"}
        
        return {
            "snapshot_time": self.timestamp.isoformat(),
            "configs": configs_status,
            "processes": process_status,
            "data_freshness": data_status,
            "directories": {
                "logs_exists": self.logs_dir.exists(),
                "metrics_exists": self.metrics_dir.exists(),
                "artifacts_exists": self.artifacts_dir.exists()
            }
        }
    
    def extract_snapshot(self, n_logs: int = 100, n_trades: int = 10) -> str:
        """Извлечь полный снапшот системы.
        
        Args:
            n_logs: Количество последних строк логов
            n_trades: Количество последних сделок
            
        Returns:
            Путь к созданному файлу снапшота
        """
        print(f"📸 Extracting live snapshot...")
        
        # Собираем данные
        log_lines = self._get_latest_log_lines(n_logs)
        recent_trades = self._get_recent_trades(n_trades)
        metrics_summary = self._get_metrics_summary()
        system_status = self._get_system_status()
        
        # Формируем отчет
        snapshot_content = self._format_snapshot_report(
            log_lines, recent_trades, metrics_summary, system_status
        )
        
        # Сохраняем файл
        snapshot_filename = f"SNAPSHOT_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.md"
        snapshot_path = self.artifacts_dir / snapshot_filename
        
        with open(snapshot_path, 'w', encoding='utf-8') as f:
            f.write(snapshot_content)
        
        print(f"✅ Snapshot saved: {snapshot_path}")
        return str(snapshot_path)
    
    def _format_snapshot_report(self, log_lines: List[str], trades: List[Dict], 
                              metrics: Dict[str, Any], status: Dict[str, Any]) -> str:
        """Форматировать отчет снапшота."""
        
        timestamp_str = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        # Форматируем метрики
        if "error" not in metrics and "message" not in metrics:
            metrics_table = f"""
| Metric | Value |
|--------|-------|
| Total PnL | ${metrics.get('total_pnl', 0):.2f} |
| Trade Count | {metrics.get('trade_count', 0)} |
| Win Rate | {metrics.get('win_rate', 0)*100:.1f}% |
| Max Drawdown | {metrics.get('max_drawdown', 0)*100:.1f}% |
| Current Drawdown | {metrics.get('current_drawdown', 0)*100:.1f}% |
| Sharpe Ratio | {metrics.get('sharpe_ratio', 0):.2f} |
| Exposure | {metrics.get('exposure', 0)*100:.1f}% |
| Active Positions | {metrics.get('active_positions', 0)} |
"""
        else:
            metrics_table = f"**Status**: {metrics.get('error', metrics.get('message', 'Unknown'))}"
        
        # Форматируем сделки
        if trades and "error" not in trades[0] and "message" not in trades[0]:
            trades_table = "| Time | Action | Pair | Price | PnL |\n|------|--------|------|-------|-----|\n"
            for trade in trades[:10]:  # Показываем максимум 10
                timestamp = trade.get('timestamp', 'Unknown')[:16]  # YYYY-MM-DD HH:MM
                action = trade.get('action', 'N/A')
                pair = trade.get('pair', 'N/A')
                price = trade.get('price', 0)
                pnl = trade.get('pnl', 0)
                
                trades_table += f"| {timestamp} | {action} | {pair} | ${price:.2f} | ${pnl:.2f} |\n"
        else:
            trades_table = f"**Status**: {trades[0].get('error', trades[0].get('message', 'No data')) if trades else 'No trades available'}"
        
        # Форматируем логи
        recent_logs = "\n".join([f"    {line}" for line in log_lines[-20:]])  # Показываем последние 20
        
        # Статус системы
        is_fresh = status.get('data_freshness', {}).get('is_fresh', False)
        freshness_status = "🟢 Fresh" if is_fresh else "🟡 Stale"
        
        configs_ok = all(c['exists'] for c in status.get('configs', {}).values())
        config_status = "🟢 OK" if configs_ok else "🟡 Missing files"
        
        report = f"""# Live System Snapshot
*Generated: {timestamp_str}*

## System Status Overview

| Component | Status |
|-----------|--------|
| Data Freshness | {freshness_status} |
| Configurations | {config_status} |
| Log Collection | 🟢 {len(log_lines)} lines collected |
| Trade History | 🟢 {len(trades)} trades found |
| Metrics | 🟢 Available |

## Performance Metrics

{metrics_table}

## Recent Trades (Last {len(trades)})

{trades_table}

## System Health

### Data Freshness
- **Last Activity**: {status.get('data_freshness', {}).get('last_modified', 'Unknown')}
- **Minutes Ago**: {status.get('data_freshness', {}).get('minutes_ago', 'N/A')}
- **Status**: {"Fresh (< 30 min)" if is_fresh else "Stale (> 30 min)"}

### Configuration Files
"""
        
        # Добавляем статус конфигов
        for name, config_info in status.get('configs', {}).items():
            exists = "✅" if config_info['exists'] else "❌"
            size = f"({config_info['size']} bytes)" if config_info['exists'] else ""
            report += f"- **{name}**: {exists} {size}\n"
        
        report += f"""
### Directories
- **Logs**: {"✅" if status['directories']['logs_exists'] else "❌"} {self.logs_dir}
- **Metrics**: {"✅" if status['directories']['metrics_exists'] else "❌"} {self.metrics_dir}
- **Artifacts**: {"✅" if status['directories']['artifacts_exists'] else "❌"} {self.artifacts_dir}

## Recent Log Entries (Last {min(len(log_lines), 20)})

```
{recent_logs}
```

## Collection Details

- **Log Lines Collected**: {len(log_lines)}
- **Trades Collected**: {len(trades)}
- **Snapshot Generated**: {timestamp_str}
- **Collection Source**: {self.logs_dir}, {self.metrics_dir}

---
*Snapshot extracted by coint2 Live Snapshot Extractor v0.1.1*
"""
        
        return report


def main():
    """Main snapshot extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract live system snapshot")
    parser.add_argument("--logs", type=int, default=100, 
                       help="Number of recent log lines to include")
    parser.add_argument("--trades", type=int, default=10,
                       help="Number of recent trades to include") 
    parser.add_argument("--logs-dir", default="artifacts/live/logs",
                       help="Directory containing log files")
    parser.add_argument("--metrics-dir", default="artifacts/live/metrics", 
                       help="Directory containing metrics files")
    
    args = parser.parse_args()
    
    # Extract snapshot
    extractor = LiveSnapshotExtractor(
        logs_dir=args.logs_dir,
        metrics_dir=args.metrics_dir
    )
    
    try:
        snapshot_path = extractor.extract_snapshot(
            n_logs=args.logs,
            n_trades=args.trades
        )
        
        print(f"\n📄 Snapshot Report: {snapshot_path}")
        print(f"🔍 Use: cat {snapshot_path}")
        return 0
        
    except Exception as e:
        print(f"❌ Snapshot extraction failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())