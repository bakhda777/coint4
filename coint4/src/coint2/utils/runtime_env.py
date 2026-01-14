"""Runtime environment adjustments for constrained systems."""

from __future__ import annotations

import logging
import multiprocessing
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _dev_shm_writable() -> bool:
    path = Path("/dev/shm")
    if not path.exists() or not path.is_dir():
        return False
    try:
        with tempfile.NamedTemporaryFile(prefix="coint2_", dir=str(path)) as _:
            return True
    except OSError:
        return False


def configure_runtime_environment() -> None:
    if _dev_shm_writable():
        return
    os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    logger.warning(
        "Runtime: /dev/shm not writable; using spawn start method and thread-only parallelism."
    )
