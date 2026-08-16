import pandas as pd

from churn_prediction.repositories.local import (
    LocalEventRepository,
)


def test_local_repository_returns_customer_events() -> None:
    events = pd.DataFrame(
        [
            {
                "customer_id": "cust_00001",
                "event_type": "session",
            },
            {
                "customer_id": "cust_00002",
                "event_type": "session",
            },
            {
                "customer_id": "cust_00001",
                "event_type": "purchase",
            },
        ]
    )

    repository = LocalEventRepository(events)

    result = repository.get_customer_events("cust_00001")

    assert len(result) == 2
    assert set(result["customer_id"]) == {"cust_00001"}


def test_local_repository_returns_empty_for_unknown_customer() -> None:
    events = pd.DataFrame(
        [
            {
                "customer_id": "cust_00001",
                "event_type": "session",
            },
        ]
    )

    repository = LocalEventRepository(events)

    result = repository.get_customer_events("cust_missing")

    assert result.empty