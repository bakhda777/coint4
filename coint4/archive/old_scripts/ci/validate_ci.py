#!/usr/bin/env python3
"""
Валидация CI система - проверяет что все компоненты smoke тестов настроены правильно.
"""

import sys
import os
from pathlib import Path
import yaml

def validate_file_structure():
    """Проверяет структуру файлов для CI."""
    print("🔍 Проверка структуры файлов...")
    
    required_files = [
        "scripts/ci_smoke.py",
        ".github/workflows/ci.yml",
        "configs/main_2024.yaml",
        "pytest.ini",
        "pyproject.toml"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Отсутствующие файлы: {missing_files}")
        return False
    
    print("✅ Все требуемые файлы найдены")
    return True

def validate_ci_script():
    """Проверяет CI smoke script."""
    print("🧪 Проверка CI smoke script...")
    
    script_path = Path("scripts/ci_smoke.py")
    
    if not script_path.exists():
        print("❌ ci_smoke.py не найден")
        return False
    
    # Проверяем что скрипт исполняемый
    if not os.access(script_path, os.X_OK):
        print("⚠️  ci_smoke.py не исполняемый, исправляем...")
        script_path.chmod(0o755)
    
    # Проверяем базовый синтаксис
    try:
        with open(script_path) as f:
            code = f.read()
            compile(code, str(script_path), 'exec')
        print("✅ Синтаксис ci_smoke.py корректен")
    except SyntaxError as e:
        print(f"❌ Синтаксическая ошибка в ci_smoke.py: {e}")
        return False
    
    return True

def validate_github_workflow():
    """Проверяет GitHub workflow."""
    print("⚙️  Проверка GitHub workflow...")
    
    workflow_path = Path(".github/workflows/ci.yml")
    
    if not workflow_path.exists():
        print("❌ .github/workflows/ci.yml не найден")
        return False
    
    try:
        with open(workflow_path) as f:
            workflow = yaml.safe_load(f)
        
        # Проверяем базовые секции ('on' может быть интерпретирован как True)
        if 'name' not in workflow:
            print("❌ Отсутствует секция 'name' в workflow")
            return False
        if True not in workflow and 'on' not in workflow:  # YAML parser может интерпретировать 'on' как True
            print("❌ Отсутствует секция 'on' в workflow")
            return False
        if 'jobs' not in workflow:
            print("❌ Отсутствует секция 'jobs' в workflow")
            return False
        
        # Проверяем что есть smoke-tests job
        if 'smoke-tests' not in workflow['jobs']:
            print("❌ Отсутствует job 'smoke-tests' в workflow")
            return False
        
        print("✅ GitHub workflow корректен")
        return True
        
    except yaml.YAMLError as e:
        print(f"❌ Ошибка YAML в workflow: {e}")
        return False

def validate_pytest_config():
    """Проверяет pytest конфигурацию."""
    print("🔬 Проверка pytest конфигурации...")
    
    pytest_ini = Path("pytest.ini")
    
    if not pytest_ini.exists():
        print("❌ pytest.ini не найден")
        return False
    
    # Проверяем что есть smoke маркер
    with open(pytest_ini) as f:
        content = f.read()
        if 'smoke:' not in content:
            print("❌ Smoke маркер не найден в pytest.ini")
            return False
    
    print("✅ pytest конфигурация корректна")
    return True

def validate_project_config():
    """Проверяет конфигурацию проекта."""
    print("📋 Проверка конфигурации проекта...")
    
    # Проверяем main_2024.yaml
    config_path = Path("configs/main_2024.yaml")
    if not config_path.exists():
        print("❌ configs/main_2024.yaml не найден")
        return False
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Проверяем основные секции
        required_sections = ['portfolio', 'pair_selection']
        for section in required_sections:
            if section not in config:
                print(f"❌ Отсутствует секция '{section}' в main_2024.yaml")
                return False
        
        print("✅ Конфигурация проекта корректна")
        return True
        
    except yaml.YAMLError as e:
        print(f"❌ Ошибка YAML в конфигурации: {e}")
        return False

def validate_python_imports():
    """Проверяет базовые Python импорты."""
    print("🐍 Проверка Python импортов...")
    
    critical_imports = [
        'numpy',
        'pandas', 
        'optuna',
        'pytest',
        'numba',
        'yaml'
    ]
    
    failed_imports = []
    for module in critical_imports:
        try:
            __import__(module)
        except ImportError:
            failed_imports.append(module)
    
    if failed_imports:
        print(f"❌ Не удалось импортировать: {failed_imports}")
        return False
    
    print("✅ Все критичные модули импортируются")
    return True

def main():
    """Основная функция валидации."""
    print("=" * 60)
    print("ВАЛИДАЦИЯ CI СИСТЕМЫ")
    print("=" * 60)
    
    # Переходим в корневую директорию
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    print(f"📂 Рабочая директория: {project_root}")
    
    # Запускаем все проверки
    checks = [
        ("Структура файлов", validate_file_structure),
        ("CI smoke script", validate_ci_script),
        ("GitHub workflow", validate_github_workflow), 
        ("pytest конфигурация", validate_pytest_config),
        ("Конфигурация проекта", validate_project_config),
        ("Python импорты", validate_python_imports)
    ]
    
    passed = 0
    failed = 0
    
    for name, check_func in checks:
        print(f"\n📋 {name}:")
        try:
            if check_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"💥 Неожиданная ошибка: {e}")
            failed += 1
    
    # Итоговый отчет
    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 60)
    
    total = passed + failed
    success_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"Общее количество проверок: {total}")
    print(f"Пройденные проверки: {passed}")
    print(f"Провалившиеся проверки: {failed}")
    print(f"Процент успеха: {success_rate:.1f}%")
    
    if failed == 0:
        print("\n🎉 ВСЕ ПРОВЕРКИ ПРОШЛИ УСПЕШНО!")
        print("CI система готова к работе.")
        return 0
    else:
        print(f"\n💥 НАЙДЕНЫ ПРОБЛЕМЫ!")
        print("Исправьте ошибки перед использованием CI.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)