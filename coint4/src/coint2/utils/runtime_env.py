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
    tmpdir = os.environ.get("TMPDIR")
    joblib_tmp = os.environ.get("JOBLIB_TEMP_FOLDER")
    if tmpdir or joblib_tmp:
        if not tmpdir:
            tmpdir = "/tmp"
            os.environ.setdefault("TMPDIR", tmpdir)
        if not joblib_tmp:
            joblib_tmp = str(Path(tmpdir) / "joblib")
            os.environ.setdefault("JOBLIB_TEMP_FOLDER", joblib_tmp)
        logger.info(
            "Runtime: /dev/shm not writable; using %s and keeping multiprocessing enabled.",
            joblib_tmp,
        )
        return
    os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    logger.warning(
        "Runtime: /dev/shm not writable; using spawn start method and thread-only parallelism."
    )
