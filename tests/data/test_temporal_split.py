from __future__ import annotations

from sentinelml.data.splits import temporal_split


def test_temporal_not_random(tiny_tx) -> None:
    split = temporal_split(tiny_tx)
    assert split.train["event_time"].max() <= split.val["event_time"].min()
    assert split.val["event_time"].max() <= split.test["event_time"].min()
    assert len(split.train) and len(split.val) and len(split.test)
    assert "train" in split.hashes
    # shuffled row order must not change membership if we sort first
    shuffled = tiny_tx.sample(frac=1.0, random_state=1)
    split2 = temporal_split(shuffled)
    assert set(split.train["transaction_id"]) == set(split2.train["transaction_id"])
