import yaml
from pathlib import Path
from typing import Any


def load_config(path: Path | str) -> dict[str, Any]:
    """Load configuration from a YAML file using ``yaml.safe_load``."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
