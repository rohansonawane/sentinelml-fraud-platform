from __future__ import annotations

import pandas as pd

from sentinelml.features.consistency import assert_no_lookahead_velocity, offline_online_consistency
from sentinelml.features.engineering import FEATURE_NAMES, FeatureState, transform_frame


def test_first_customer_tx_has_zero_velocity(tiny_tx: pd.DataFrame) -> None:
    assert_no_lookahead_velocity(tiny_tx)


def test_features_ignore_future_transactions(tiny_tx: pd.DataFrame) -> None:
    ordered = tiny_tx.sort_values("event_time").reset_index(drop=True)
    feats, _ = transform_frame(ordered)
    # For every row, 24h count cannot exceed how many prior txs that customer has.
    prior = {}
    for idx, row in ordered.iterrows():
        cid = row["customer_id"]
        seen = prior.get(cid, 0)
        assert feats.loc[idx, "tx_count_24h"] <= seen
        prior[cid] = seen + 1


def test_merchant_fraud_rate_uses_delay(tiny_tx: pd.DataFrame) -> None:
    state = FeatureState(label_delay_days=7)
    ordered = tiny_tx.sort_values("event_time")
    first = ordered.iloc[0].to_dict()
    feats = state.extract(first)
    assert feats["merchant_fraud_rate_historical"] == 0.0
    state.update(first)
    # immediate next tx at same merchant still 0 because label not yet available
    second = dict(first)
    second["transaction_id"] = "T_next"
    second["event_id"] = "E_next"
    second["event_time"] = first["event_time"] + pd.Timedelta(minutes=1)
    feats2 = state.extract(second)
    assert feats2["merchant_fraud_rate_historical"] == 0.0


def test_offline_online_consistency(tiny_tx: pd.DataFrame) -> None:
    result = offline_online_consistency(tiny_tx.head(120))
    assert result["consistent"] is True
    assert result["mismatched_features"] == []
    assert result["max_abs_diff"] == 0.0


def test_feature_names_contract() -> None:
    assert "amount_zscore_customer" in FEATURE_NAMES
    assert "merchant_fraud_rate_historical" in FEATURE_NAMES
    assert len(FEATURE_NAMES) == 18
