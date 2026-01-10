#!/bin/bash
# Full test suite including slow and serial tests

set -e
echo "🧪 Running full test suite..."
cd "$(dirname "$0")/.."

echo "1/2: Running parallel tests..."
pytest -n auto -m "not serial" --tb=short --durations=10

echo "2/2: Running serial tests..."
pytest -m serial --tb=short

echo "✅ All tests passed!"
