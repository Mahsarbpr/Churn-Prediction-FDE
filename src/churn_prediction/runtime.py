from __future__ import annotations

import json
import os
from pathlib import Path

import boto3
import pandas as pd

from churn_prediction.features import load_events
from churn_prediction.inference.factory import load_churn_model
from churn_prediction.inference.prediction_service import (
    PredictionService,
)
from churn_prediction.repositories.local import (
    LocalEventRepository,
)
from churn_prediction.repositories.s3 import (
    S3EventRepository,
)


def _download_model_from_s3(
    bucket: str,
    key: str,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    s3 = boto3.client("s3")

    s3.download_file(
        bucket,
        key,
        str(destination),
    )


def build_prediction_service() -> PredictionService:
    runtime_mode = os.getenv(
        "CHURN_RUNTIME_MODE",
        "local",
    )

    if runtime_mode == "aws":
        bucket = os.environ["CHURN_S3_BUCKET"]

        repository = S3EventRepository(
            bucket=bucket,
            key="raw/events.json",
        )

        model_path = Path(
            "/tmp/churn_model.json"
        )

        _download_model_from_s3(
            bucket=bucket,
            key="models/xgb-v1/churn_model.json",
            destination=model_path,
        )

    else:
        events = load_events(
            "data/events.json"
        )

        repository = LocalEventRepository(
            events
        )

        model_path = Path(
            "artifacts/churn_model.json"
        )

    model = load_churn_model(
        model_type="xgboost",
        model_path=model_path,
        model_version="xgb-v1",
    )

    return PredictionService(
        model=model,
        event_repository=repository,
        clock=lambda: pd.Timestamp(
            "2024-06-01T12:00:00Z"
        ),
    )