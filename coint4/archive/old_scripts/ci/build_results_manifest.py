#!/usr/bin/env python3
"""
Results Manifest Builder - создание полного манифеста результатов для воспроизводимости.
Собирает все метаданные о том, как был получен конкретный результат.
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib


class ResultsManifestBuilder:
    """Строитель манифеста результатов."""
    
    def __init__(self, verbose: bool = False):
        """Initialize results manifest builder."""
        self.verbose = verbose
        self.manifest = {
            "generated_at": datetime.now().isoformat(),
            "version": "1.0",
            "git": {},
            "environment": {},
            "data": {},
            "configuration": {},
            "execution": {},
            "artifacts": {},
            "reproduction": {}
        }
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Вычислить SHA256 файла."""
        if not file_path.exists():
            return "N/A"
        
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception:
            return "ERROR"
    
    def collect_git_info(self) -> None:
        """Сбор git метаданных."""
        if self.verbose:
            print("📝 Сбор git информации...")
        
        try:
            # Git commit
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10)
            commit_hash = result.stdout.strip() if result.returncode == 0 else 'unknown'
            
            # Short hash
            result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, timeout=10)
            short_hash = result.stdout.strip() if result.returncode == 0 else 'unknown'
            
            # Branch
            result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], capture_output=True, text=True, timeout=10)
            branch = result.stdout.strip() if result.returncode == 0 else 'unknown'
            
            # Uncommitted changes
            result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, timeout=10)
            has_changes = len(result.stdout.strip()) > 0 if result.returncode == 0 else False
            
            self.manifest["git"] = {
                "commit": commit_hash,
                "short_commit": short_hash,
                "branch": branch,
                "has_uncommitted_changes": has_changes,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Ошибка сбора git info: {e}")
            
            self.manifest["git"] = {
                "commit": "unknown",
                "short_commit": "unknown",
                "branch": "unknown",
                "has_uncommitted_changes": False,
                "error": str(e)
            }
    
    def collect_environment_refs(self) -> None:
        """Сбор ссылок на environment lock."""
        if self.verbose:
            print("🔧 Сбор environment references...")
        
        env_lock_json = Path("artifacts/env/ENV_LOCK.json")
        env_lock_txt = Path("artifacts/env/ENV_LOCK.txt")
        
        self.manifest["environment"] = {
            "env_lock_json": {
                "path": str(env_lock_json) if env_lock_json.exists() else None,
                "hash": self._get_file_hash(env_lock_json),
                "exists": env_lock_json.exists()
            },
            "env_lock_txt": {
                "path": str(env_lock_txt) if env_lock_txt.exists() else None,
                "hash": self._get_file_hash(env_lock_txt),
                "exists": env_lock_txt.exists()
            },
            "python_version": sys.version,
            "python_executable": sys.executable
        }
    
    def collect_data_refs(self) -> None:
        """Сбор ссылок на data lock."""
        if self.verbose:
            print("📊 Сбор data references...")
        
        data_lock_json = Path("artifacts/data/DATA_LOCK.json")
        data_lock_md = Path("artifacts/data/DATA_LOCK.md")
        
        self.manifest["data"] = {
            "data_lock_json": {
                "path": str(data_lock_json) if data_lock_json.exists() else None,
                "hash": self._get_file_hash(data_lock_json),
                "exists": data_lock_json.exists()
            },
            "data_lock_md": {
                "path": str(data_lock_md) if data_lock_md.exists() else None,
                "hash": self._get_file_hash(data_lock_md),
                "exists": data_lock_md.exists()
            },
            "data_root": "data_downloaded"
        }
    
    def collect_configuration_info(self) -> None:
        """Сбор конфигурационных файлов."""
        if self.verbose:
            print("⚙️ Сбор конфигураций...")
        
        config_files = [
            "configs/main_2024.yaml",
            "configs/portfolio_optimizer.yaml", 
            "configs/ci_gates.yaml",
            "configs/drift_monitor.yaml",
            "pyproject.toml"
        ]
        
        configs = {}
        for config_path in config_files:
            path = Path(config_path)
            key = path.stem  # Имя файла без расширения
            
            configs[key] = {
                "path": config_path,
                "hash": self._get_file_hash(path),
                "exists": path.exists(),
                "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else None
            }
        
        self.manifest["configuration"] = configs
    
    def collect_execution_context(self) -> None:
        """Сбор контекста выполнения."""
        if self.verbose:
            print("🏃 Сбор execution context...")
        
        # Поиск seed/deterministic context
        seed = None
        
        # Проверить различные места где может быть seed
        seed_sources = [
            "configs/main_2024.yaml",
            "artifacts/wfa/WFA_REPORT.md", 
            "artifacts/uncertainty/CONFIDENCE_REPORT.md"
        ]
        
        for source in seed_sources:
            path = Path(source)
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        content = f.read()
                        if 'seed' in content.lower():
                            # Попытаться извлечь seed
                            import re
                            match = re.search(r'seed[:\s]*(\d+)', content, re.IGNORECASE)
                            if match:
                                seed = int(match.group(1))
                                break
                except Exception:
                    pass
        
        self.manifest["execution"] = {
            "seed": seed,
            "working_directory": str(Path.cwd()),
            "timestamp": datetime.now().isoformat(),
            "user": "system"  # Можно расширить
        }
    
    def collect_artifacts_info(self) -> None:
        """Сбор информации об артефактах."""
        if self.verbose:
            print("📁 Сбор информации об артефактах...")
        
        # Ключевые артефакты для воспроизведения
        key_artifacts = [
            "artifacts/wfa/WFA_REPORT.md",
            "artifacts/portfolio/weights.csv",
            "artifacts/portfolio/PORTFOLIO_REPORT.md",
            "artifacts/uncertainty/CONFIDENCE_REPORT.md",
            "artifacts/monitoring/DRIFT_DASHBOARD.md",
            "bench/pairs_portfolio.yaml"
        ]
        
        artifacts = {}
        
        for artifact_path in key_artifacts:
            path = Path(artifact_path)
            key = artifact_path.replace('artifacts/', '').replace('/', '_').replace('.', '_')
            
            artifacts[key] = {
                "path": artifact_path,
                "hash": self._get_file_hash(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else None
            }
        
        # Дополнительно - поиск Optuna studies
        optuna_studies = []
        studies_dir = Path("outputs/studies")
        if studies_dir.exists():
            for study_file in studies_dir.glob("*.db"):
                optuna_studies.append({
                    "path": str(study_file),
                    "name": study_file.stem,
                    "hash": self._get_file_hash(study_file),
                    "size_mb": study_file.stat().st_size / 1024 / 1024
                })
        
        artifacts["optuna_studies"] = optuna_studies
        
        self.manifest["artifacts"] = artifacts
    
    def build_reproduction_commands(self) -> None:
        """Построить команды для воспроизведения."""
        if self.verbose:
            print("🔄 Построение команд воспроизведения...")
        
        commands = []
        
        # 1. Проверка окружения
        if self.manifest["environment"]["env_lock_json"]["exists"]:
            commands.append({
                "step": "verify_environment",
                "command": "python scripts/env_lock.py --verify artifacts/env/ENV_LOCK.json",
                "description": "Проверить совместимость окружения"
            })
        
        # 2. Проверка данных
        if self.manifest["data"]["data_lock_json"]["exists"]:
            commands.append({
                "step": "verify_data",
                "command": "python scripts/data_lock.py --verify artifacts/data/DATA_LOCK.json",
                "description": "Проверить целостность данных"
            })
        
        # 3. Основные команды воспроизведения
        if Path("bench/pairs_portfolio.yaml").exists():
            commands.extend([
                {
                    "step": "run_uncertainty",
                    "command": "python scripts/run_uncertainty.py --output-dir artifacts/uncertainty",
                    "description": "Воспроизвести анализ неопределённости"
                },
                {
                    "step": "run_drift_monitoring", 
                    "command": "python scripts/monitor_drift.py --config configs/drift_monitor.yaml",
                    "description": "Воспроизвести мониторинг дрейфа"
                },
                {
                    "step": "run_paper_week",
                    "command": "python scripts/run_paper_week.py --pairs-file bench/pairs_portfolio.yaml",
                    "description": "Воспроизвести paper week симуляцию"
                }
            ])
        
        # 4. CI gates проверки
        commands.append({
            "step": "run_ci_gates",
            "command": "python scripts/ci_gates.py --config configs/ci_gates.yaml --verbose",
            "description": "Запустить CI gates для проверки качества"
        })
        
        self.manifest["reproduction"] = {
            "commands": commands,
            "total_steps": len(commands),
            "estimated_duration_minutes": len(commands) * 2  # Примерная оценка
        }
    
    def build_manifest(self) -> None:
        """Собрать полный манифест."""
        self.collect_git_info()
        self.collect_environment_refs()
        self.collect_data_refs()
        self.collect_configuration_info()
        self.collect_execution_context()
        self.collect_artifacts_info()
        self.build_reproduction_commands()
    
    def save_manifest(self, output_dir: str = "artifacts/repro") -> None:
        """Сохранить манифест."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # JSON манифест
        json_file = output_path / "RESULTS_MANIFEST.json"
        with open(json_file, 'w') as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"💾 Создан {json_file}")
        
        # Markdown отчёт
        self._generate_markdown_report(output_path / "RESULTS_MANIFEST.md")
    
    def _generate_markdown_report(self, output_file: Path) -> None:
        """Генерация markdown отчёта."""
        git = self.manifest["git"]
        env = self.manifest["environment"]
        data = self.manifest["data"]
        config = self.manifest["configuration"]
        exec_ctx = self.manifest["execution"]
        artifacts = self.manifest["artifacts"]
        repro = self.manifest["reproduction"]
        
        report = f"""# Results Manifest
Generated: {self.manifest['generated_at']}

## 🎯 Reproduction Context
This manifest describes exactly how these results were produced and how to reproduce them.

## 📝 Git State
- **Commit:** `{git['commit']}`
- **Branch:** `{git['branch']}`
- **Has Changes:** {'⚠️ YES' if git['has_uncommitted_changes'] else '✅ NO'}

## 🔧 Environment
- **Python:** `{sys.version.split()[0]}`
- **Environment Lock:** {'✅ Available' if env['env_lock_json']['exists'] else '❌ Missing'}
- **Environment Hash:** `{env['env_lock_json']['hash'][:16]}...`

## 📊 Data
- **Data Lock:** {'✅ Available' if data['data_lock_json']['exists'] else '❌ Missing'}  
- **Data Hash:** `{data['data_lock_json']['hash'][:16]}...`
- **Data Root:** `{data['data_root']}`

## ⚙️ Configuration Files
| Config | Status | Hash (8 chars) |
|--------|--------|----------------|
"""
        
        for name, info in config.items():
            status = '✅ Found' if info['exists'] else '❌ Missing'
            hash_short = info['hash'][:8] if info['hash'] != 'N/A' else 'N/A'
            report += f"| `{name}` | {status} | `{hash_short}` |\n"
        
        report += f"""
## 🏃 Execution Context
- **Working Dir:** `{exec_ctx['working_directory']}`
- **Seed:** {exec_ctx['seed'] if exec_ctx['seed'] else 'Not detected'}
- **Timestamp:** {exec_ctx['timestamp']}

## 📁 Key Artifacts
| Artifact | Status | Size | Hash (8 chars) |
|----------|--------|------|----------------|
"""
        
        for name, info in artifacts.items():
            if name != "optuna_studies":
                status = '✅ Found' if info['exists'] else '❌ Missing'
                size = f"{info['size_bytes'] / 1024:.1f} KB" if info['size_bytes'] > 0 else "0 KB"
                hash_short = info['hash'][:8] if info['hash'] != 'N/A' else 'N/A'
                report += f"| `{Path(info['path']).name}` | {status} | {size} | `{hash_short}` |\n"
        
        if artifacts.get("optuna_studies"):
            report += f"\n### Optuna Studies ({len(artifacts['optuna_studies'])})\n"
            for study in artifacts["optuna_studies"][:5]:  # Первые 5
                report += f"- `{study['name']}` ({study['size_mb']:.1f} MB)\n"
        
        report += f"""
## 🔄 Reproduction Commands ({repro['total_steps']} steps)
Estimated time: ~{repro['estimated_duration_minutes']} minutes

"""
        
        for i, cmd in enumerate(repro["commands"], 1):
            report += f"### {i}. {cmd['description']}\n"
            report += f"```bash\n{cmd['command']}\n```\n\n"
        
        report += f"""## 🚀 One-Click Reproduction
```bash
python scripts/reproduce.py --manifest artifacts/repro/RESULTS_MANIFEST.json
```

## 📋 Verification Checklist
- [ ] Environment verified: `python scripts/env_lock.py --verify artifacts/env/ENV_LOCK.json`
- [ ] Data verified: `python scripts/data_lock.py --verify artifacts/data/DATA_LOCK.json`
- [ ] Git commit matches: `{git['commit']}`
- [ ] All configs present and unchanged
- [ ] All reproduction steps executed successfully
- [ ] Final CI gates pass: `python scripts/ci_gates.py --config configs/ci_gates.yaml`

---
*Generated by Results Manifest Builder v{self.manifest['version']}*
"""
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        if self.verbose:
            print(f"📊 Создан {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Results Manifest Builder - создание манифеста воспроизводимости')
    
    parser.add_argument('--output-dir', default='artifacts/repro',
                       help='Директория для сохранения манифеста')
    parser.add_argument('--verbose', action='store_true',
                       help='Подробный вывод')
    
    args = parser.parse_args()
    
    builder = ResultsManifestBuilder(verbose=args.verbose)
    
    if args.verbose:
        print("🔨 Создание Results Manifest...")
    
    builder.build_manifest()
    builder.save_manifest(args.output_dir)
    
    if args.verbose:
        print(f"\n✅ Results Manifest завершён:")
        print(f"   JSON: {args.output_dir}/RESULTS_MANIFEST.json")
        print(f"   MD:   {args.output_dir}/RESULTS_MANIFEST.md")


if __name__ == '__main__':
    main()