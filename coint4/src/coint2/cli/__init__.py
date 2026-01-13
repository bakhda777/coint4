"""Command-line interface entry points for coint2."""

from . import check_coint_health
from . import build_universe

__all__ = [
    "check_coint_health",
    "build_universe",
]


def main() -> int:
    """Basic CLI dispatcher placeholder."""
    print("coint2.cli: available commands: check_coint_health, build_universe")
    return 0
