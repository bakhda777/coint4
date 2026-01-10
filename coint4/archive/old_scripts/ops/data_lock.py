#!/usr/bin/env python3
"""
Data Lock - заморозка датасета для обеспечения воспроизводимости.
Сканирует DATA_ROOT и создаёт манифест с sha256, временными границами и метаданными.
"""

import sys
import argparse
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd


class DataLockManager:
    """Менеджер заморозки датасета."""
    
    def __init__(self, data_root: str = "data_downloaded", verbose: bool = False):
        """Initialize data lock manager."""
        self.data_root = Path(data_root)
        self.verbose = verbose
        self.lock_data = {
            "generated_at": datetime.now().isoformat(),
            "data_root": str(self.data_root),
            "files": [],
            "summary": {
                "total_files": 0,
                "total_size_bytes": 0,
                "total_rows": 0,
                "time_range": {"min": None, "max": None},
                "symbols": set(),
                "timeframes": set()
            }
        }
    
    def _calculate_sha256(self, file_path: Path) -> str:
        """Вычислить SHA256 файла."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _analyze_parquet_file(self, file_path: Path) -> Dict[str, Any]:
        """Анализ parquet файла для извлечения метаданных."""
        try:
            df = pd.read_parquet(file_path)
            
            file_info = {
                "path": str(file_path),
                "size_bytes": file_path.stat().st_size,
                "sha256": self._calculate_sha256(file_path),
                "rows": len(df),
                "columns": list(df.columns),
                "time_range": {"min": None, "max": None},
                "symbols": [],
                "timeframe": None
            }
            
            # Анализ временных меток
            if 'timestamp' in df.columns:
                if not df['timestamp'].empty:
                    # Проверяем, timestamp в миллисекундах или уже datetime
                    if df['timestamp'].dtype == 'int64':
                        # Конвертируем из миллисекунд в datetime
                        min_time = pd.to_datetime(df['timestamp'].min(), unit='ms')
                        max_time = pd.to_datetime(df['timestamp'].max(), unit='ms')
                    else:
                        min_time = pd.to_datetime(df['timestamp'].min())
                        max_time = pd.to_datetime(df['timestamp'].max())
                    
                    file_info["time_range"]["min"] = str(min_time)
                    file_info["time_range"]["max"] = str(max_time)
                    
                    # Обновление глобального диапазона
                    if (self.lock_data["summary"]["time_range"]["min"] is None or 
                        min_time < pd.to_datetime(self.lock_data["summary"]["time_range"]["min"])):
                        self.lock_data["summary"]["time_range"]["min"] = str(min_time)
                        
                    if (self.lock_data["summary"]["time_range"]["max"] is None or 
                        max_time > pd.to_datetime(self.lock_data["summary"]["time_range"]["max"])):
                        self.lock_data["summary"]["time_range"]["max"] = str(max_time)
            
            # Анализ символов
            if 'symbol' in df.columns:
                unique_symbols = df['symbol'].unique().tolist()
                file_info["symbols"] = unique_symbols
                self.lock_data["summary"]["symbols"].update(unique_symbols)
            
            # Определение таймфрейма по частоте данных
            if 'timestamp' in df.columns and len(df) > 1:
                if df['timestamp'].dtype == 'int64':
                    # Timestamps в миллисекундах
                    time_diff = (df['timestamp'].iloc[1] - df['timestamp'].iloc[0]) / 1000 / 60  # в минутах
                    minutes = int(time_diff)
                else:
                    time_diff = pd.to_datetime(df['timestamp'].iloc[1]) - pd.to_datetime(df['timestamp'].iloc[0])
                    minutes = int(time_diff.total_seconds() / 60)
                
                if minutes == 1:
                    file_info["timeframe"] = "1m"
                elif minutes == 5:
                    file_info["timeframe"] = "5m"
                elif minutes == 15:
                    file_info["timeframe"] = "15m"
                elif minutes == 60:
                    file_info["timeframe"] = "1h"
                elif minutes == 240:
                    file_info["timeframe"] = "4h"
                elif minutes == 1440:
                    file_info["timeframe"] = "1d"
                else:
                    file_info["timeframe"] = f"{minutes}m"
                
                if file_info["timeframe"]:
                    self.lock_data["summary"]["timeframes"].add(file_info["timeframe"])
            
            return file_info
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Ошибка анализа {file_path}: {e}")
            
            # Базовая информация без анализа содержимого
            return {
                "path": str(file_path),
                "size_bytes": file_path.stat().st_size,
                "sha256": self._calculate_sha256(file_path),
                "rows": 0,
                "columns": [],
                "time_range": {"min": None, "max": None},
                "symbols": [],
                "timeframe": None,
                "error": str(e)
            }
    
    def scan_data_directory(self) -> None:
        """Сканировать директорию с данными."""
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_root}")
        
        if self.verbose:
            print(f"📂 Сканирование {self.data_root}...")
        
        # Поиск всех parquet файлов
        parquet_files = list(self.data_root.rglob("*.parquet"))
        
        if not parquet_files:
            print(f"⚠️ Не найдено parquet файлов в {self.data_root}")
            return
        
        if self.verbose:
            print(f"📊 Найдено {len(parquet_files)} parquet файлов")
        
        for i, parquet_file in enumerate(parquet_files):
            if self.verbose:
                print(f"   {i+1:3d}/{len(parquet_files)}: {parquet_file.name}")
            
            file_info = self._analyze_parquet_file(parquet_file)
            self.lock_data["files"].append(file_info)
            
            # Обновление сводки
            self.lock_data["summary"]["total_files"] += 1
            self.lock_data["summary"]["total_size_bytes"] += file_info["size_bytes"]
            self.lock_data["summary"]["total_rows"] += file_info["rows"]
    
    def generate_lock_files(self, output_dir: str = "artifacts/data") -> None:
        """Генерация файлов заморозки."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Конвертация sets в lists для JSON сериализации
        self.lock_data["summary"]["symbols"] = sorted(list(self.lock_data["summary"]["symbols"]))
        self.lock_data["summary"]["timeframes"] = sorted(list(self.lock_data["summary"]["timeframes"]))
        
        # JSON манифест
        json_file = output_path / "DATA_LOCK.json"
        with open(json_file, 'w') as f:
            json.dump(self.lock_data, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"💾 Создан {json_file}")
        
        # Человекочитаемый отчёт
        self._generate_markdown_report(output_path / "DATA_LOCK.md")
    
    def _generate_markdown_report(self, output_file: Path) -> None:
        """Генерация markdown отчёта."""
        summary = self.lock_data["summary"]
        
        report = f"""# Data Lock Report
Generated: {self.lock_data['generated_at']}

## Summary
- **Data Root:** `{self.lock_data['data_root']}`
- **Total Files:** {summary['total_files']}
- **Total Size:** {summary['total_size_bytes'] / 1024 / 1024:.1f} MB
- **Total Rows:** {summary['total_rows']:,}
- **Time Range:** {summary['time_range']['min']} to {summary['time_range']['max']}

## Symbols ({len(summary['symbols'])})
{', '.join(summary['symbols'][:50])}{'...' if len(summary['symbols']) > 50 else ''}

## Timeframes
{', '.join(summary['timeframes'])}

## Files Detail
| File | Size (KB) | Rows | SHA256 (first 8) | Timeframe |
|------|-----------|------|------------------|-----------|
"""
        
        # Добавление информации о файлах
        for file_info in self.lock_data["files"][:20]:  # Показать первые 20
            size_kb = file_info["size_bytes"] / 1024
            sha256_short = file_info["sha256"][:8]
            timeframe = file_info.get("timeframe", "N/A")
            
            report += f"| `{Path(file_info['path']).name}` | {size_kb:.1f} | {file_info['rows']:,} | `{sha256_short}` | {timeframe} |\n"
        
        if len(self.lock_data["files"]) > 20:
            report += f"\n... и ещё {len(self.lock_data['files']) - 20} файлов\n"
        
        report += f"""
## Integrity Verification
To verify data integrity:
```bash
python scripts/data_lock.py --verify artifacts/data/DATA_LOCK.json
```

## Reproduction Command
```bash
python scripts/reproduce.py --data-lock artifacts/data/DATA_LOCK.json
```
"""
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        if self.verbose:
            print(f"📄 Создан {output_file}")
    
    def verify_lock(self, lock_file: str) -> bool:
        """Проверить целостность данных против lock файла."""
        if self.verbose:
            print(f"🔍 Проверка целостности против {lock_file}")
        
        try:
            with open(lock_file, 'r') as f:
                expected_lock = json.load(f)
            
            mismatches = []
            
            for expected_file in expected_lock["files"]:
                file_path = Path(expected_file["path"])
                
                if not file_path.exists():
                    mismatches.append(f"Отсутствует: {file_path}")
                    continue
                
                # Проверка SHA256
                actual_sha256 = self._calculate_sha256(file_path)
                if actual_sha256 != expected_file["sha256"]:
                    mismatches.append(f"SHA256 mismatch: {file_path}")
                
                # Проверка размера
                actual_size = file_path.stat().st_size
                if actual_size != expected_file["size_bytes"]:
                    mismatches.append(f"Size mismatch: {file_path}")
            
            if mismatches:
                print("❌ Обнаружены расхождения:")
                for mismatch in mismatches:
                    print(f"   - {mismatch}")
                return False
            else:
                print("✅ Все файлы соответствуют lock манифесту")
                return True
                
        except Exception as e:
            print(f"❌ Ошибка при проверке: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Data Lock - заморозка датасета для воспроизводимости')
    
    parser.add_argument('--data-root', '--root', default='data_downloaded',
                       help='Корневая директория с данными')
    parser.add_argument('--output-dir', default='artifacts/data',
                       help='Директория для сохранения lock файлов')
    parser.add_argument('--scan', action='store_true',
                       help='Режим сканирования и создания lock')
    parser.add_argument('--verify', metavar='LOCK_FILE',
                       help='Проверить целостность против существующего lock файла')
    parser.add_argument('--verbose', action='store_true',
                       help='Подробный вывод')
    
    args = parser.parse_args()
    
    manager = DataLockManager(
        data_root=args.data_root,
        verbose=args.verbose
    )
    
    if args.verify:
        success = manager.verify_lock(args.verify)
        sys.exit(0 if success else 1)
    elif args.scan or (not args.verify):
        # Создание нового lock
        manager.scan_data_directory()
        manager.generate_lock_files(args.output_dir)
        
        if args.verbose:
            print(f"\n✅ Data lock завершён:")
            print(f"   JSON: {args.output_dir}/DATA_LOCK.json")
            print(f"   MD:   {args.output_dir}/DATA_LOCK.md")


if __name__ == '__main__':
    main()