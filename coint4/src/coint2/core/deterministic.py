#!/usr/bin/env python3
"""
Deterministic execution context для воспроизводимых результатов.
"""

import random
import numpy as np
from contextlib import contextmanager
from typing import Optional


@contextmanager
def DeterministicContext(seed: int):
    """Context manager для детерминистичного выполнения."""
    
    # Сохранить текущие состояния
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    
    try:
        # Установить seeds
        random.seed(seed)
        np.random.seed(seed)
        
        yield
        
    finally:
        # Восстановить состояния
        random.setstate(python_state)
        np.random.set_state(numpy_state)