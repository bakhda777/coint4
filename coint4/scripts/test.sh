#!/bin/bash
# Quick smoke tests

echo "🔥 Running smoke tests..."
pytest -m "smoke and not slow" --tb=short --maxfail=5 -q

if [ $? -eq 0 ]; then
    echo "✅ Smoke tests passed"
else
    echo "❌ Smoke tests failed"
    exit 1
fi