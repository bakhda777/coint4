#!/bin/bash
# Быстрые unit тесты - только изолированные тесты с моками

set -e

echo "⚡ БЫСТРЫЕ UNIT ТЕСТЫ"
echo "==================="
echo "Цель: Только изолированные unit тесты с моками"
echo "Время выполнения: <30 секунд"
echo ""

# Переходим в корневую директорию проекта
cd "$(dirname "$0")/.."

# Сначала smoke тесты для уверенности
echo "1️⃣ Smoke тесты..."
time pytest -m smoke --maxfail=1 -q --tb=short
echo "✅ Smoke тесты пройдены"
echo ""

# Только unit тесты с моками (исключаем все интеграционные)
echo "2️⃣ Unit тесты с моками (параллельно)..."
time pytest -n auto \
    -m "unit or (fast and not integration and not slow and not serial)" \
    --ignore=tests/test_*integration*.py \
    --ignore=tests/test_*comprehensive*.py \
    --ignore=tests/test_*full*.py \
    --ignore=tests/test_*performance*.py \
    --ignore=tests/test_*optimization*.py \
    --ignore=tests/test_*walk_forward*.py \
    --ignore=tests/test_*global_cache*.py \
    --ignore=tests/test_*thread*.py \
    --ignore=tests/test_*parallel*.py \
    --maxfail=5 -q --tb=short --durations=5

echo "✅ Unit тесты пройдены"
echo ""

echo "🎯 БЫСТРЫЕ UNIT ТЕСТЫ ЗАВЕРШЕНЫ!"
echo "Время выполнения должно быть <30 секунд"
