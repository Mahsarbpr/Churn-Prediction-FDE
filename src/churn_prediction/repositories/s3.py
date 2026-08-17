from __future__ import annotations

import io
import json

import boto3
import pandas as pd


class S3EventRepository:
    def __init__(
        self,
        bucket: str,
        key: str,
    ) -> None:
        self._bucket = bucket
        self._key = key
        self._client = boto3.client("s3")

    def get_customer_events(
        self,
        customer_id: str,
    ) -> pd.DataFrame:
        response = self._client.get_object(
            Bucket=self._bucket,
            Key=self._key,
        )

        body = response["Body"].read()

        events = pd.read_json(
            io.BytesIO(body)
        )

        events["timestamp"] = pd.to_datetime(
            events["timestamp"],
            utc=True,
        )

        return events[
            events["customer_id"] == customer_id
        ].copy()