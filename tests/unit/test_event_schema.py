from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from sentinelml.data.schemas import SCHEMA_VERSION, TransactionEvent


def test_valid_event() -> None:
    ev = TransactionEvent(
        event_id="E1",
        transaction_id="T1",
        event_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        customer_id="C1",
        merchant_id="M1",
        amount=12.5,
    )
    assert ev.schema_version == SCHEMA_VERSION
    assert ev.idempotency_key() == "1.0:T1"


def test_rejects_non_positive_amount() -> None:
    with pytest.raises(ValidationError):
        TransactionEvent(
            event_id="E1",
            transaction_id="T1",
            event_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            customer_id="C1",
            merchant_id="M1",
            amount=0,
        )


def test_rejects_empty_ids() -> None:
    with pytest.raises(ValidationError):
        TransactionEvent(
            event_id="",
            transaction_id="T1",
            event_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            customer_id="C1",
            merchant_id="M1",
            amount=1,
        )
