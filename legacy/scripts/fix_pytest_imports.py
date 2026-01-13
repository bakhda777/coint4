#!/usr/bin/env python3
"""
Скрипт для добавления импорта pytest в файлы, где используются маркеры pytest.
"""

import re
from pathlib import Path

def fix_pytest_import(file_path: Path):
    """Добавляет импорт pytest, если его нет, но есть маркеры pytest."""
    content = file_path.read_text()
    
    # Проверяем, есть ли маркеры pytest
    if '@pytest.mark.' not in content:
        return False
    
    # Проверяем, есть ли уже импорт pytest
    if re.search(r'^import pytest', content, re.MULTILINE):
        return False
    
    # Ищем первый импорт и добавляем pytest после него
    lines = content.split('\n')
    import_added = False
    
    for i, line in enumerate(lines):
        if line.strip().startswith('import ') and not import_added:
            lines.insert(i + 1, 'import pytest')
            import_added = True
            break
        elif line.strip().startswith('from ') and not import_added:
            # Если есть только from импорты, добавляем pytest перед первым
            lines.insert(i, 'import pytest')
            import_added = True
            break
    
    if import_added:
        new_content = '\n'.join(lines)
        file_path.write_text(new_content)
        print(f"✅ Добавлен импорт pytest в {file_path}")
        return True
    
    return False

def main():
    """Основная функция."""
    project_root = Path(__file__).parent.parent
    test_files = list(project_root.glob('tests/**/*.py'))
    
    print("🔧 Исправление импортов pytest...")
    
    fixed_count = 0
    for file_path in test_files:
        if fix_pytest_import(file_path):
            fixed_count += 1
    
    print(f"✅ Исправлено файлов: {fixed_count}")

if __name__ == "__main__":
    main()
