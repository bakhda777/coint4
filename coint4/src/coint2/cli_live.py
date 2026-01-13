"""Minimal live CLI entry point placeholder."""

import argparse


def main() -> int:
    """Entry point for live CLI."""
    parser = argparse.ArgumentParser(description="Live trading CLI")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry mode")
    parser.parse_args()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
