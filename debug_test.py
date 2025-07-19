import numpy as np
import pandas as pd
from src.coint2.engine.backtest_engine import PairBacktester
import statsmodels.api as sm

# Создаем тестовые данные
np.random.seed(42)
n_periods = 100

# Генерируем более разнообразные коинтегрированные данные
prices_1 = np.cumsum(np.random.normal(0, 2, n_periods)) + 100
prices_2 = 0.8 * prices_1 + np.cumsum(np.random.normal(0, 1, n_periods)) + 50

# Добавляем некоторую волатильность для создания торговых сигналов
for i in range(10, n_periods, 20):
    prices_2[i:i+5] += np.random.normal(0, 3, min(5, n_periods-i))

test_data = pd.DataFrame({
    'asset1': prices_1,
    'asset2': prices_2
})

print(f"Данные созданы: {len(test_data)} строк")
print(f"Asset1 range: {test_data['asset1'].min():.2f} - {test_data['asset1'].max():.2f}")
print(f"Asset2 range: {test_data['asset2'].min():.2f} - {test_data['asset2'].max():.2f}")
print(f"Asset1 std: {test_data['asset1'].std():.2f}")
print(f"Asset2 std: {test_data['asset2'].std():.2f}")

# Проверяем корреляцию
correlation = test_data['asset1'].corr(test_data['asset2'])
print(f"Корреляция: {correlation:.4f}")

# Тестируем OLS на первом окне вручную
print("\n=== Тестирование OLS вручную ===")
y_win = test_data['asset1'].iloc[0:10]
x_win = test_data['asset2'].iloc[0:10]
print(f"Размер окна: y={len(y_win)}, x={len(x_win)}")
print(f"Y данные: {y_win.values[:5]}...")
print(f"X данные: {x_win.values[:5]}...")

try:
    x_const = sm.add_constant(x_win)
    model = sm.OLS(y_win, x_const).fit()
    beta_manual = model.params.iloc[1]
    spread_manual = y_win - beta_manual * x_win
    mean_manual = spread_manual.mean()
    std_manual = spread_manual.std()
    print(f"Ручной расчет OLS: beta={beta_manual:.4f}, mean={mean_manual:.4f}, std={std_manual:.4f}")
except Exception as e:
    print(f"Ошибка в ручном расчете OLS: {e}")

# Создаем бэктестер
engine = PairBacktester(
    test_data,
    rolling_window=10,
    z_threshold=1.5,  # Снижаем порог для увеличения вероятности сигналов
    capital_at_risk=10000
)

print(f"\nБэктестер создан с rolling_window={engine.rolling_window}")

# Тестируем метод _calculate_ols_with_cache напрямую
print("\n=== Тестирование _calculate_ols_with_cache ===")
try:
    beta_cached, mean_cached, std_cached = engine._calculate_ols_with_cache(y_win, x_win)
    print(f"Кэшированный расчет: beta={beta_cached:.4f}, mean={mean_cached:.4f}, std={std_cached:.4f}")
except Exception as e:
    print(f"Ошибка в кэшированном расчете: {e}")

# Запускаем бэктест
print("\n=== Запуск бэктеста ===")
engine.run()
print("Бэктест завершен")

# Проверяем результаты
results = engine.results
print(f"\nРезультаты: {len(results)} строк")
print(f"Столбцы: {list(results.columns)}")

if 'beta' in results.columns:
    beta_values = results['beta'].dropna()
    print(f"Beta значения: {len(beta_values)} non-NaN из {len(results)}")
    if len(beta_values) > 0:
        print(f"Beta range: {beta_values.min():.4f} - {beta_values.max():.4f}")
        print(f"Beta mean: {beta_values.mean():.4f}")
        print(f"Первые 10 beta: {beta_values.head(10).values}")
    else:
        print("Все beta значения NaN!")
        # Проверим, где должны быть значения
        print(f"Индексы с NaN beta: {results[results['beta'].isna()].index.tolist()[:10]}")
else:
    print("Столбец 'beta' не найден")

if 'z_score' in results.columns:
    z_values = results['z_score'].dropna()
    print(f"Z-score значения: {len(z_values)} non-NaN из {len(results)}")
    if len(z_values) > 0:
        print(f"Z-score range: {z_values.min():.4f} - {z_values.max():.4f}")
        print(f"Z-score > 1.5: {len(z_values[abs(z_values) > 1.5])}")

if 'position' in results.columns:
    position_values = results['position']
    non_zero_positions = position_values[position_values != 0]
    print(f"Позиции: {len(non_zero_positions)} non-zero из {len(position_values)}")
    if len(non_zero_positions) > 0:
        print(f"Position range: {non_zero_positions.min():.4f} - {non_zero_positions.max():.4f}")
else:
    print("Столбец 'position' не найден")