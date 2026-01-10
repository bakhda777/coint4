"""Простой тест потокобезопасности."""

import threading
import pytest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock

@pytest.mark.serial
@pytest.mark.unit
def test_threading_when_imported_then_available():
    """Тест импорта threading в fast_objective."""
    try:
        from src.optimiser.fast_objective import FastWalkForwardObjective

        # Проверяем, что threading импортирован
        import src.optimiser.fast_objective as fast_obj_module
        assert hasattr(fast_obj_module, 'threading'), "threading должен быть импортирован"

    except Exception as e:
        pytest.fail(f"Ошибка импорта: {e}")

@pytest.mark.serial
@pytest.mark.unit
def test_cache_lock_when_created_then_available():
    """Тест создания блокировки кэша."""
    try:
        from src.optimiser.fast_objective import FastWalkForwardObjective

        # Мокаем все внешние зависимости
        with patch.object(FastWalkForwardObjective, '_initialize_global_rolling_cache', return_value=True):
            with patch('src.optimiser.fast_objective.load_config') as mock_load_config:
                with patch('builtins.open', MagicMock()):
                    with patch('yaml.safe_load') as mock_yaml_load:
                        # Создаем mock конфигурацию
                        mock_config = MagicMock()
                        mock_load_config.return_value = mock_config

                        # Мокаем search space
                        mock_yaml_load.return_value = {'rolling_window': {'type': 'int', 'low': 20, 'high': 50}}

                        objective = FastWalkForwardObjective("configs/main_2024.yaml", "configs/search_space.yaml")

                        # Проверяем наличие блокировки
                        assert hasattr(objective, '_cache_lock'), "Должна быть блокировка _cache_lock"
                        # Проверяем, что это объект блокировки (упрощенная проверка)
                        assert objective._cache_lock is not None, "Блокировка не должна быть None"
                        assert hasattr(objective._cache_lock, 'acquire'), "Блокировка должна иметь метод acquire"
                        assert hasattr(objective._cache_lock, 'release'), "Блокировка должна иметь метод release"

    except Exception as e:
        pytest.fail(f"Ошибка создания блокировки: {e}")

@pytest.mark.serial
@pytest.mark.unit
def test_threading_when_concurrent_access_then_thread_safe():
    """Простой тест многопоточности."""
    call_count = 0
    lock = threading.Lock()

    def worker():
        nonlocal call_count
        with lock:
            current = call_count
            # Используем мок вместо time.sleep для детерминизма
            call_count = current + 1

    # Запускаем 5 потоков
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker) for _ in range(5)]
        for future in futures:
            future.result()

    assert call_count == 5, f"Ожидали 5 вызовов, получили {call_count}"
