from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from churn_prediction.features import load_events
from churn_prediction.inference.factory import load_churn_model
from churn_prediction.inference.prediction_service import (
    PredictionService,
)
from churn_prediction.repositories.local import (
    LocalEventRepository,
)


def build_prediction_service() -> PredictionService:
    model_path = Path(
        os.getenv(
            "CHURN_MODEL_PATH",
            "artifacts/churn_model.json",
        )
    )

    metadata_path = Path(
        os.getenv(
            "CHURN_MODEL_METADATA_PATH",
            "artifacts/churn_model_metadata.json",
        )
    )

    events_path = Path(
        os.getenv(
            "CHURN_EVENTS_PATH",
            "data/events.json",
        )
    )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    model = load_churn_model(
        model_type=metadata["model_type"],
        model_path=model_path,
        model_version=metadata["model_version"],
    )

    events = load_events(str(events_path))

    repository = LocalEventRepository(events)

    scoring_time = os.getenv("CHURN_SCORING_TIME")

    if scoring_time:
        clock = lambda: pd.Timestamp(scoring_time)
    else:
        clock = lambda: pd.Timestamp.now(tz="UTC")

    return PredictionService(
        model=model,
        event_repository=repository,
        clock=lambda: pd.Timestamp( #later we will change to clock
        "2024-06-01T12:00:00Z"
        ),
    )