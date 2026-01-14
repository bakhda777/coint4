# Architecture

```mermaid
graph TD
    subgraph "1. Scanning"
        A[CLI: scan] --> B[DataHandler]
        B --> C[Pair Scanner]
        C --> D[bench/pairs_universe.yaml]
    end
    subgraph "2. Backtesting"
        E[CLI: backtest] --> F[Load pairs_universe.yaml]
        F --> G[PairBacktester]
        G --> H[outputs/fixed_run/*]
    end
    subgraph "3. Walk-forward"
        I[CLI: walk-forward] --> J[run_walk_forward]
        J --> K[DataHandler]
        K --> L[PairBacktester]
        L --> M[results_dir/*]
    end
    D --> F
```

## Backtest timing

- Rolling statistics for bar `i` are computed from `[i - rolling_window, i - 1]` only.
- Signal generated on bar `i` is executed on bar `i + 1`; entry `z_score`/`beta` use the signal bar.
- Numba full engine follows the same 1-bar lag and entry-beta conventions for costs and PnL.
- Cost model is aggregated: `commission_pct + slippage_pct` are applied; `enable_realistic_costs` parameters are reserved and not wired into the core PnL path yet.
