"""CLI entry point for live trading."""

import sys
from pathlib import Path

# Import the main function from scripts
scripts_path = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_path))

from run_live import main

if __name__ == "__main__":
    main()