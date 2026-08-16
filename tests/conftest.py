from __future__ import annotations

import pandas as pd
import pytest

from sentinelml.data.simulator import SimulatorConfig, generate_transactions


@pytest.fixture
def tiny_tx() -> pd.DataFrame:
    cfg = SimulatorConfig(
        n_customers=40,
        n_merchants=12,
        n_transactions=400,
        n_days=30,
        seed=42,
    )
    return generate_transactions(cfg)
