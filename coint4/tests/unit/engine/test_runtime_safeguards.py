from types import SimpleNamespace

import pandas as pd

from coint2.engine.base_engine import BasePairBacktester


def test_pair_stop_loss_usd_respects_runtime_config() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="15min")
    data = pd.DataFrame(
        {
            "y": [100.0, 100.0, 100.0, 100.0, 100.0, 90.0],
            "x": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        },
        index=index,
    )

    config = SimpleNamespace(
        pair_stop_loss_usd=5.0,
        pair_stop_loss_zscore=10.0,
    )
    bt = BasePairBacktester(
        pair_data=data,
        rolling_window=3,
        z_threshold=1.0,
        config=config,
    )
    bt.current_position = 1.0
    bt.entry_price_s1 = 100.0
    bt.entry_price_s2 = 100.0
    bt.entry_beta = 1.0

    assert bt._check_stop_loss_conditions(data, 5, z_curr=0.0) is True

    bt.pair_stop_loss_usd = 15.0
    assert bt._check_stop_loss_conditions(data, 5, z_curr=0.0) is False
