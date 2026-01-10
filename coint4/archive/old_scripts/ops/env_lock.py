#!/usr/bin/env python3
"""
Environment Lock - фиксация окружения для воспроизводимости.
Собирает информацию о Python, OS, зависимостях и создаёт манифест.
"""

import sys
import platform
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


class EnvironmentLockManager:
    """Менеджер заморозки окружения."""
    
    def __init__(self, verbose: bool = False):
        """Initialize environment lock manager."""
        self.verbose = verbose
        self.env_data = {
            "generated_at": datetime.now().isoformat(),
            "python": {},
            "system": {},
            "git": {},
            "packages": {},
            "project": {}
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
    
    def collect_python_info(self) -> None:
        """Сбор информации о Python."""
        if self.verbose:
            print("🐍 Сбор информации о Python...")
        
        self.env_data["python"] = {
            "version": sys.version,
            "version_info": {
                "major": sys.version_info.major,
                "minor": sys.version_info.minor,
                "micro": sys.version_info.micro,
                "releaselevel": sys.version_info.releaselevel,
                "serial": sys.version_info.serial
            },
            "executable": sys.executable,
            "platform": sys.platform,
            "implementation": sys.implementation.name if hasattr(sys, 'implementation') else 'unknown'
        }
    
    def collect_system_info(self) -> None:
        """Сбор информации о системе."""
        if self.verbose:
            print("💻 Сбор информации о системе...")
        
        self.env_data["system"] = {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "hostname": platform.node()
        }
    
    def collect_git_info(self) -> None:
        """Сбор информации о git."""
        if self.verbose:
            print("📝 Сбор информации о git...")
        
        try:
            # Git commit
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'], 
                capture_output=True, text=True, timeout=10
            )
            commit_hash = result.stdout.strip() if result.returncode == 0 else 'unknown'
            
            # Git branch
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                capture_output=True, text=True, timeout=10
            )
            branch = result.stdout.strip() if result.returncode == 0 else 'unknown'
            
            # Git status (check for uncommitted changes)
            result = subprocess.run(
                ['git', 'status', '--porcelain'], 
                capture_output=True, text=True, timeout=10
            )
            has_changes = len(result.stdout.strip()) > 0 if result.returncode == 0 else False
            
            # Git remote
            result = subprocess.run(
                ['git', 'remote', 'get-url', 'origin'], 
                capture_output=True, text=True, timeout=10
            )
            remote = result.stdout.strip() if result.returncode == 0 else 'unknown'
            
            self.env_data["git"] = {
                "commit": commit_hash,
                "branch": branch,
                "has_uncommitted_changes": has_changes,
                "remote": remote
            }
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Ошибка сбора git info: {e}")
            
            self.env_data["git"] = {
                "commit": "unknown",
                "branch": "unknown", 
                "has_uncommitted_changes": False,
                "remote": "unknown",
                "error": str(e)
            }
    
    def collect_package_info(self) -> None:
        """Сбор информации о пакетах."""
        if self.verbose:
            print("📦 Сбор информации о пакетах...")
        
        # pip freeze
        packages_list = []
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'freeze'], 
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip() and '==' in line:
                        try:
                            name, version = line.strip().split('==')
                            packages_list.append({'name': name, 'version': version})
                        except ValueError:
                            packages_list.append({'name': line.strip(), 'version': 'unknown'})
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Ошибка pip freeze: {e}")
        
        self.env_data["packages"]["pip_freeze"] = packages_list
        
        # Poetry lock hash (если есть)
        poetry_lock = Path("poetry.lock")
        if poetry_lock.exists():
            self.env_data["packages"]["poetry_lock_hash"] = self._get_file_hash(poetry_lock)
            
            # Попробуем получить инфу из poetry.lock
            try:
                with open(poetry_lock, 'r') as f:
                    content = f.read()
                    # Найти секцию [[package]]
                    poetry_packages = []
                    lines = content.split('\n')
                    current_package = None
                    
                    for line in lines[:200]:  # Ограничить парсинг первыми 200 строками
                        line = line.strip()
                        if line.startswith('[[package]]'):
                            if current_package:
                                poetry_packages.append(current_package)
                            current_package = {}
                        elif line.startswith('name = ') and current_package is not None:
                            current_package['name'] = line.split(' = ')[1].strip('"')
                        elif line.startswith('version = ') and current_package is not None:
                            current_package['version'] = line.split(' = ')[1].strip('"')
                    
                    if current_package:
                        poetry_packages.append(current_package)
                    
                    self.env_data["packages"]["poetry_packages"] = poetry_packages[:20]  # Первые 20
                    
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Ошибка парсинга poetry.lock: {e}")
        else:
            self.env_data["packages"]["poetry_lock_hash"] = "N/A"
    
    def collect_project_info(self) -> None:
        """Сбор информации о проекте."""
        if self.verbose:
            print("🔧 Сбор информации о проекте...")
        
        # pyproject.toml hash
        pyproject = Path("pyproject.toml")
        if pyproject.exists():
            self.env_data["project"]["pyproject_hash"] = self._get_file_hash(pyproject)
            
            # Попробуем получить версию проекта
            try:
                with open(pyproject, 'r') as f:
                    content = f.read()
                    for line in content.split('\n'):
                        if line.strip().startswith('version = '):
                            version = line.split('=')[1].strip().strip('"').strip("'")
                            self.env_data["project"]["version"] = version
                            break
            except Exception:
                pass
        else:
            self.env_data["project"]["pyproject_hash"] = "N/A"
        
        # requirements.txt hash (если есть)
        requirements = Path("requirements.txt")
        self.env_data["project"]["requirements_hash"] = self._get_file_hash(requirements)
        
        # Рабочая директория
        self.env_data["project"]["working_directory"] = str(Path.cwd())
        
        # Python path
        self.env_data["project"]["python_path"] = sys.path.copy()[:10]  # Первые 10
    
    def collect_all_info(self) -> None:
        """Собрать всю информацию об окружении."""
        self.collect_python_info()
        self.collect_system_info()
        self.collect_git_info()
        self.collect_package_info()
        self.collect_project_info()
    
    def generate_lock_files(self, output_dir: str = "artifacts/env") -> None:
        """Генерация файлов заморозки окружения."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # JSON манифест
        json_file = output_path / "ENV_LOCK.json"
        with open(json_file, 'w') as f:
            json.dump(self.env_data, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"💾 Создан {json_file}")
        
        # Текстовый список пакетов для быстрого сравнения
        packages_file = output_path / "ENV_LOCK.txt"
        with open(packages_file, 'w') as f:
            f.write(f"# Environment Lock - Generated {self.env_data['generated_at']}\n\n")
            f.write(f"Python: {self.env_data['python']['version']}\n")
            f.write(f"System: {self.env_data['system']['platform']}\n")
            f.write(f"Git: {self.env_data['git']['commit']} ({self.env_data['git']['branch']})\n")
            f.write(f"Poetry Lock Hash: {self.env_data['packages']['poetry_lock_hash']}\n")
            f.write(f"PyProject Hash: {self.env_data['project']['pyproject_hash']}\n\n")
            
            f.write("# Pip Packages:\n")
            for pkg in self.env_data["packages"]["pip_freeze"]:
                f.write(f"{pkg['name']}=={pkg['version']}\n")
        
        if self.verbose:
            print(f"📄 Создан {packages_file}")
        
        # Человекочитаемый отчёт
        self._generate_markdown_report(output_path / "ENV_REPORT.md")
    
    def _generate_markdown_report(self, output_file: Path) -> None:
        """Генерация markdown отчёта."""
        python = self.env_data["python"]
        system = self.env_data["system"]
        git = self.env_data["git"]
        packages = self.env_data["packages"]
        project = self.env_data["project"]
        
        report = f"""# Environment Lock Report
Generated: {self.env_data['generated_at']}

## System Environment
- **OS:** {system['platform']}
- **Architecture:** {system['machine']} ({system['architecture'][0]})
- **Hostname:** {system['hostname']}

## Python Environment
- **Version:** {python['version'].split()[0]}
- **Executable:** `{python['executable']}`
- **Implementation:** {python['implementation']}
- **Platform:** {python['platform']}

## Git Repository
- **Commit:** `{git['commit']}`
- **Branch:** `{git['branch']}`
- **Has Changes:** {'⚠️ YES' if git['has_uncommitted_changes'] else '✅ NO'}
- **Remote:** `{git.get('remote', 'unknown')}`

## Project Configuration
- **Working Dir:** `{project['working_directory']}`
- **PyProject Hash:** `{project['pyproject_hash'][:16]}...` ({'✅ Found' if project['pyproject_hash'] != 'N/A' else '❌ Missing'})
- **Poetry Lock Hash:** `{packages['poetry_lock_hash'][:16]}...` ({'✅ Found' if packages['poetry_lock_hash'] != 'N/A' else '❌ Missing'})
- **Requirements Hash:** `{project['requirements_hash'][:16]}...` ({'✅ Found' if project['requirements_hash'] != 'N/A' else '❌ Missing'})

## Installed Packages ({len(packages['pip_freeze'])})
| Package | Version |
|---------|---------|
"""
        
        # Показать первые 20 пакетов
        for pkg in packages["pip_freeze"][:20]:
            report += f"| `{pkg['name']}` | {pkg['version']} |\n"
        
        if len(packages["pip_freeze"]) > 20:
            report += f"\n... и ещё {len(packages['pip_freeze']) - 20} пакетов\n"
        
        if packages.get("poetry_packages"):
            report += f"\n## Poetry Packages ({len(packages['poetry_packages'])})\n"
            report += "| Package | Version |\n"
            report += "|---------|----------|\n"
            
            for pkg in packages["poetry_packages"]:
                report += f"| `{pkg.get('name', 'unknown')}` | {pkg.get('version', 'unknown')} |\n"
        
        report += f"""

## Verification
To verify environment compatibility:
```bash
python scripts/env_lock.py --verify artifacts/env/ENV_LOCK.json
```

## Reproduction Setup
To setup similar environment:
```bash
# Check Python version
python --version  # Should be {python['version'].split()[0]}

# Install from lock file
pip install -r artifacts/env/ENV_LOCK.txt

# Verify git state
git checkout {git['commit']}
```

## Environment Hash
Environment signature: `{self._calculate_env_hash()}`
"""
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        if self.verbose:
            print(f"📊 Создан {output_file}")
    
    def _calculate_env_hash(self) -> str:
        """Вычислить хеш окружения для быстрого сравнения."""
        # Ключевые компоненты для хеша
        key_components = [
            self.env_data["python"]["version"],
            self.env_data["system"]["platform"],
            self.env_data["git"]["commit"],
            self.env_data["packages"]["poetry_lock_hash"],
            self.env_data["project"]["pyproject_hash"]
        ]
        
        combined = '|'.join(str(c) for c in key_components)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def verify_environment(self, lock_file: str) -> bool:
        """Проверить совместимость текущего окружения с lock файлом."""
        if self.verbose:
            print(f"🔍 Проверка окружения против {lock_file}")
        
        try:
            with open(lock_file, 'r') as f:
                expected_env = json.load(f)
            
            # Собрать текущее окружение для сравнения
            self.collect_all_info()
            
            warnings = []
            errors = []
            
            # Проверка Python версии
            expected_py = expected_env["python"]["version_info"]
            current_py = self.env_data["python"]["version_info"]
            
            if (expected_py["major"] != current_py["major"] or 
                expected_py["minor"] != current_py["minor"]):
                errors.append(f"Python version mismatch: expected {expected_py['major']}.{expected_py['minor']}, got {current_py['major']}.{current_py['minor']}")
            
            # Проверка git commit
            if expected_env["git"]["commit"] != self.env_data["git"]["commit"]:
                warnings.append(f"Git commit differs: expected {expected_env['git']['commit']}, got {self.env_data['git']['commit']}")
            
            # Проверка poetry.lock hash
            expected_poetry_hash = expected_env["packages"]["poetry_lock_hash"]
            current_poetry_hash = self.env_data["packages"]["poetry_lock_hash"]
            
            if expected_poetry_hash != "N/A" and expected_poetry_hash != current_poetry_hash:
                errors.append("Poetry.lock hash mismatch - dependencies may differ")
            
            # Показать результаты
            if errors:
                print("❌ Критические несоответствия:")
                for error in errors:
                    print(f"   - {error}")
            
            if warnings:
                print("⚠️ Предупреждения:")
                for warning in warnings:
                    print(f"   - {warning}")
            
            if not errors and not warnings:
                print("✅ Окружение полностью совместимо")
                return True
            elif not errors:
                print("⚠️ Окружение совместимо с предупреждениями")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Ошибка при проверке: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Environment Lock - фиксация окружения для воспроизводимости')
    
    parser.add_argument('--output-dir', default='artifacts/env',
                       help='Директория для сохранения lock файлов')
    parser.add_argument('--verify', metavar='LOCK_FILE',
                       help='Проверить совместимость с существующим lock файлом')
    parser.add_argument('--verbose', action='store_true',
                       help='Подробный вывод')
    
    args = parser.parse_args()
    
    manager = EnvironmentLockManager(verbose=args.verbose)
    
    if args.verify:
        success = manager.verify_environment(args.verify)
        sys.exit(0 if success else 1)
    else:
        # Создание нового lock
        manager.collect_all_info()
        manager.generate_lock_files(args.output_dir)
        
        if args.verbose:
            print(f"\n✅ Environment lock завершён:")
            print(f"   JSON: {args.output_dir}/ENV_LOCK.json")
            print(f"   TXT:  {args.output_dir}/ENV_LOCK.txt")
            print(f"   MD:   {args.output_dir}/ENV_REPORT.md")


if __name__ == '__main__':
    main()