#!/bin/bash
# Скрипт для запуска Optuna Dashboard

echo "Запуск Optuna Dashboard..."
echo "Откройте браузер и перейдите по адресу: http://localhost:8080"
echo "Для остановки нажмите Ctrl+C"

optuna-dashboard sqlite:///outputs/studies/fast_optimization.db
