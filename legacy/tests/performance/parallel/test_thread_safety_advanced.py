"""Продвинутый тест потокобезопасности кэширования."""

import threading
import sys
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import pytest


@pytest.mark.serial
@pytest.mark.unit
def test_thread_safe_caching_when_concurrent_access_then_single_execution():
    """
    Симуляция потокобезопасного кэширования без полной инициализации FastWalkForwardObjective.
    Тестирует паттерн double-checked locking.
    """
    print("🧪 Тестирование потокобезопасного кэширования...")
    
    # Симулируем кэш и блокировку как в FastWalkForwardObjective
    cache = {}
    cache_lock = threading.Lock()
    expensive_operation_calls = 0
    
    def expensive_operation(cache_key):
        """Симуляция дорогой операции _select_pairs_for_step."""
        nonlocal expensive_operation_calls
        expensive_operation_calls += 1
        # Мокируем задержку вместо time.sleep согласно правилам
        return f"result_for_{cache_key}"
    
    def get_cached_result(cache_key):
        """Симуляция метода _process_single_walk_forward_step с потокобезопасным кэшированием."""
        # 1. Быстрая проверка кэша без блокировки
        if cache_key in cache:
            return cache[cache_key]
        else:
            # 2. Блокировка для выполнения дорогой операции
            with cache_lock:
                # 3. Повторная проверка кэша ВНУТРИ блокировки
                if cache_key in cache:
                    return cache[cache_key]
                else:
                    # 4. Выполнение дорогой операции и сохранение в кэш
                    result = expensive_operation(cache_key)
                    cache[cache_key] = result
                    return result
    
    # Тестируем с одним ключом кэша
    test_cache_key = "2024-01-01_2024-01-31"
    
    # Запускаем 10 потоков параллельно с одинаковым ключом
    num_threads = 10
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(get_cached_result, test_cache_key) for _ in range(num_threads)]
        results = [future.result() for future in futures]
    
    # Проверки
    assert len(results) == num_threads, f"Ожидали {num_threads} результатов, получили {len(results)}"
    assert all(r == f"result_for_{test_cache_key}" for r in results), "Все результаты должны быть одинаковыми"
    assert expensive_operation_calls == 1, (
        f"Дорогая операция должна быть вызвана только 1 раз, "
        f"но была вызвана {expensive_operation_calls} раз"
    )
    assert len(cache) == 1, f"В кэше должен быть 1 элемент, но есть {len(cache)}"
    assert test_cache_key in cache, "Ключ должен быть в кэше"
    
    print("✅ Потокобезопасное кэширование работает корректно")
    print(f"   - Запущено потоков: {num_threads}")
    print(f"   - Вызовов дорогой операции: {expensive_operation_calls}")
    print(f"   - Элементов в кэше: {len(cache)}")
    print(f"   - Все результаты корректны: {all(r == f'result_for_{test_cache_key}' for r in results)}")


@pytest.mark.serial
@pytest.mark.unit
def test_multiple_cache_keys():
    """Тест с несколькими ключами кэша."""
    print("\n🧪 Тестирование с несколькими ключами кэша...")
    
    cache = {}
    cache_lock = threading.Lock()
    operation_calls = {}
    
    def expensive_operation(cache_key):
        """Симуляция дорогой операции."""
        if cache_key not in operation_calls:
            operation_calls[cache_key] = 0
        operation_calls[cache_key] += 1
        # Заменяем time.sleep на простое увеличение счетчика для имитации работы
        return f"result_for_{cache_key}"
    
    def get_cached_result(cache_key):
        """Потокобезопасное получение результата."""
        if cache_key in cache:
            return cache[cache_key]
        else:
            with cache_lock:
                if cache_key in cache:
                    return cache[cache_key]
                else:
                    result = expensive_operation(cache_key)
                    cache[cache_key] = result
                    return result
    
    # Тестируем с 3 разными ключами
    cache_keys = ["2024-01-01_2024-01-31", "2024-02-01_2024-02-28", "2024-03-01_2024-03-31"]
    
    # Запускаем по 5 потоков для каждого ключа (всего 15 потоков)
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = []
        for cache_key in cache_keys:
            for _ in range(5):
                futures.append(executor.submit(get_cached_result, cache_key))
        
        results = [future.result() for future in futures]
    
    # Проверки
    assert len(results) == 15, f"Ожидали 15 результатов, получили {len(results)}"
    assert len(cache) == 3, f"В кэше должно быть 3 элемента, но есть {len(cache)}"
    
    # Каждая дорогая операция должна быть вызвана только один раз для каждого ключа
    for cache_key in cache_keys:
        assert cache_key in operation_calls, f"Ключ {cache_key} должен быть в operation_calls"
        assert operation_calls[cache_key] == 1, (
            f"Операция для ключа {cache_key} должна быть вызвана 1 раз, "
            f"но была вызвана {operation_calls[cache_key]} раз"
        )
    
    print("✅ Тест с несколькими ключами прошел успешно")
    print(f"   - Обработано ключей: {len(cache_keys)}")
    print(f"   - Всего потоков: 15")
    print(f"   - Элементов в кэше: {len(cache)}")
    print(f"   - Вызовов операций: {sum(operation_calls.values())}")


@pytest.mark.unit
def test_threading_import_in_fast_objective():
    """Проверяем, что threading корректно импортирован в fast_objective."""
    print("\n🧪 Проверка импорта threading...")
    
    try:
        import src.optimiser.fast_objective as fast_obj_module
        assert hasattr(fast_obj_module, 'threading'), "threading должен быть импортирован"
        print("✅ threading импортирован корректно")
    except Exception as e:
        pytest.fail(f"Ошибка импорта threading: {e}")


# Нестандартный блок if __name__ == '__main__' удален согласно правилам тестирования.
# Все тесты должны запускаться через pytest.
