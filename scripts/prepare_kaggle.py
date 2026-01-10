#!/usr/bin/env python3
"""
Скрипт для подготовки проекта к запуску на Kaggle.
1. Генерирует requirements.txt из pyproject.toml
2. Архивирует код (src, configs, scripts) в code_archive.zip
3. Выводит инструкцию
"""

import tomllib
import shutil
import zipfile
import os
from pathlib import Path

def generate_requirements(pyproject_path: Path, output_path: Path):
    print(f"Generatig requirements.txt from {pyproject_path}...")
    try:
        with open(pyproject_path, 'rb') as f:
            data = tomllib.load(f)
        dependencies = data.get('tool', {}).get('poetry', {}).get('dependencies', {})
        
        with open(output_path, 'w') as f:
            for pkg, version in dependencies.items():
                if pkg == "python":
                    continue
                
                # Обработка сложных зависимостей типа dask
                if isinstance(version, dict):
                    extras = version.get('extras', [])
                    ver = version.get('version', '*')
                    pkg_str = f"{pkg}[{','.join(extras)}]" if extras else pkg
                else:
                    ver = version
                    pkg_str = pkg

                if ver == "*":
                    f.write(f"{pkg_str}\n")
                elif ver.startswith("^"):
                    f.write(f"{pkg_str}>={ver[1:]}\n")
                elif ver.startswith(">") or ver.startswith("<") or ver.startswith("="):
                     f.write(f"{pkg_str}{ver}\n")
                else:
                    f.write(f"{pkg_str}=={ver}\n")
        print(f"Saved to {output_path}")
    except Exception as e:
        print(f"Error generating requirements.txt: {e}")

def create_code_archive(root_dir: Path, output_path: Path):
    print(f"Creating code archive {output_path}...")
    
    # Список директорий для включения
    dirs_to_include = ['src', 'configs', 'scripts']
    
    with zipfile.ZipFile(output_path, 'w') as zipf:
        # Добавляем requirements.txt
        if (root_dir / 'requirements.txt').exists():
            zipf.write(root_dir / 'requirements.txt', 'requirements.txt')
            
        for dir_name in dirs_to_include:
            dir_path = root_dir / dir_name
            if not dir_path.exists():
                print(f"Warning: Directory {dir_name} not found, skipping.")
                continue
                
            for file_path in dir_path.rglob('*'):
                if file_path.is_file() and not file_path.name.startswith('.') and '__pycache__' not in str(file_path):
                    rel_path = file_path.relative_to(root_dir)
                    zipf.write(file_path, rel_path)
    
    print(f"Archive created: {output_path}")

def main():
    root_dir = Path(__file__).parent.parent
    pyproject_path = root_dir / "pyproject.toml"
    requirements_path = root_dir / "requirements.txt"
    archive_path = root_dir / "code_archive.zip"
    
    # 1. Generate requirements.txt
    if not requirements_path.exists():
        generate_requirements(pyproject_path, requirements_path)
    else:
        print("requirements.txt already exists, updating...")
        generate_requirements(pyproject_path, requirements_path)

    # 2. Create archive
    create_code_archive(root_dir, archive_path)
    
    print("\n" + "="*50)
    print("ГОТОВО! Для запуска на Kaggle:")
    print("1. Создайте новый Dataset на Kaggle и загрузите туда:")
    print(f"   - {archive_path.name}")
    print(f"   - Вашу папку data/ (архивируйте её в data.zip если нужно)")
    print("2. В Kaggle Notebook подключите этот Dataset.")
    print("3. Используйте следующий код для настройки окружения:")
    print("="*50)
    print("""
import os
import shutil
import zipfile
import subprocess

# 1. Распаковка кода
print("Unpacking code...")
with zipfile.ZipFile('/kaggle/input/YOUR_DATASET_NAME/code_archive.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/code')

# 2. Установка зависимостей
print("Installing dependencies...")
subprocess.check_call(['pip', 'install', '-r', '/kaggle/working/code/requirements.txt'])

# 3. Настройка путей
# Предполагаем, что данные смонтированы в /kaggle/input/YOUR_DATASET_NAME/data
# Или распакуйте их, если они в архиве
DATA_DIR = '/kaggle/input/YOUR_DATASET_NAME/data' 
# Если data была в архиве:
# with zipfile.ZipFile('/kaggle/input/YOUR_DATASET_NAME/data.zip', 'r') as zip_ref:
#    zip_ref.extractall('/kaggle/working/data')
# DATA_DIR = '/kaggle/working/data'

os.environ['DATA_ROOT'] = DATA_DIR
os.environ['PYTHONPATH'] = '/kaggle/working/code/src'

# 4. Запуск
print("Running optimization...")
# Пример запуска
# !python /kaggle/working/code/scripts/fast_optimize.py --output /kaggle/working/results.db
    """)
    print("="*50)

if __name__ == "__main__":
    main()
