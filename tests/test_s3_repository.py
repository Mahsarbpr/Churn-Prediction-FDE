import io

import pandas as pd

from churn_prediction.repositories.s3 import (
    S3EventRepository,
)


class FakeBody:
    def __init__(
        self,
        data: bytes,
    ) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class FakeS3Client:
    def get_object(
        self,
        Bucket: str,
        Key: str,
    ) -> dict:
        events = pd.DataFrame(
            [
                {
                    "event_id": "evt_1",
                    "customer_id": "cust_1",
                    "event_type": "session",
                    "timestamp": "2024-05-01T12:00:00Z",
                    "properties": {
                        "duration_sec": 100,
                    },
                },
                {
                    "event_id": "evt_2",
                    "customer_id": "cust_2",
                    "event_type": "session",
                    "timestamp": "2024-05-02T12:00:00Z",
                    "properties": {
                        "duration_sec": 80,
                    },
                },
            ]
        )

        buffer = io.BytesIO()
        events.to_json(
            buffer,
            orient="records",
        )

        return {
            "Body": FakeBody(
                buffer.getvalue()
            )
        }


def test_s3_repository_returns_customer_events(
    monkeypatch,
) -> None:
    repository = S3EventRepository(
        bucket="test-bucket",
        key="raw/events.json",
    )

    repository._client = FakeS3Client()

    result = repository.get_customer_events(
        "cust_1"
    )

    assert len(result) == 1
    assert (
        result.iloc[0]["customer_id"]
        == "cust_1"
    )