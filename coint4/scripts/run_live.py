#!/usr/bin/env python3
"""Entrypoint for Bybit trading runner (demo/testnet/live).

WARNING: `BYBIT_ENV=live` places real orders on Bybit.
"""

from coint2.live.runner import main


if __name__ == "__main__":
    raise SystemExit(main())
